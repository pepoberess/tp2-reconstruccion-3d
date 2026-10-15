"""
TP2 - Reconstrucción 3D. Corre el pipeline completo sobre un dataset:

    python pipeline/run_pipeline.py configs/stereo_budha_charuco.yaml [--show]

Salidas en <output>/:
    stereo_calibration.pkl, stereo_maps.pkl   calibración + mapas de rectificación
    figures/                                   imágenes para el informe
    reconstruction.ply                         nube final optimizada (entregable)
    summary.json                               parámetros, puntos por etapa, medición
    intermediate/                              pesados, NO se entregan
"""
import argparse
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import calib  # módulo de la cátedra (tutorial 6)
import calibration as stereo_calib
import fusion
import pose as pose_mod
import stereo
from dataset import list_stereo_pairs, read_checkerboard_txt, read_markers_yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mundo con Z hacia arriba: rotación de 180° sobre X respecto del marco del tablero
# de OpenCV (que tiene Z entrando a la mesa).
FLIP_Z_UP = np.diag([1.0, -1.0, -1.0, 1.0])


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


# ---------------------------------------------------------------------------
# figuras
# ---------------------------------------------------------------------------

def save_checkerboard_figure(calib_dir, checkerboard, used, left_pts, right_pts, path):
    """Detección de corners en un par de calibración (calib.draw_checkerboard, tutorial 6)."""
    i = len(used) // 2
    pairs = {idx: (lf, rf) for idx, lf, rf in list_stereo_pairs(calib_dir)}
    lf, rf = pairs[used[i]]
    style = {"corner_radius": 10, "corner_thickness": 5, "line_thickness": 4}
    left = calib.draw_checkerboard(cv2.imread(lf), checkerboard, left_pts[i], True, **style)
    right = calib.draw_checkerboard(cv2.imread(rf), checkerboard, right_pts[i], True, **style)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, img, title in zip(axes, [left, right], ["left", "right"]):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{title} - par {used[i]}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_per_view_errors_figure(calibration, path):
    """Error de reproyección por par (stereoCalibrateExtended, tutorial 6)."""
    errors = calibration["per_view_errors"]
    used = calibration["used_pairs"]
    x = np.arange(len(used))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - 0.2, errors[:, 0], 0.4, label="left")
    ax.bar(x + 0.2, errors[:, 1], 0.4, label="right")
    ax.axhline(calibration["rms_stereo"], color="k", linestyle="--",
               label=f"RMS global = {calibration['rms_stereo']:.3f} px")
    ax.set_xticks(x, [str(u) for u in used])
    ax.set_xlabel("par de calibración")
    ax.set_ylabel("error de reproyección [px]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_rectification_figure(left, right, left_r, right_r, path, n_lines=8):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    h = left.shape[0]
    for ax, img, title in zip(
        axes.ravel(),
        [left, right, left_r, right_r],
        ["left original", "right original", "left rectificada", "right rectificada"],
    ):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for y in np.linspace(h * 0.1, h * 0.9, n_lines):
            ax.axhline(y, color="lime", linewidth=0.7)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_disparity_figure(left_r, disparity, path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].imshow(cv2.cvtColor(left_r, cv2.COLOR_BGR2RGB))
    axes[0].set_title("left rectificada")
    d = np.where(disparity > 0, disparity, np.nan)
    im = axes[1].imshow(d, cmap="turbo")
    axes[1].set_title("disparidad SGBM [px]")
    fig.colorbar(im, ax=axes[1], fraction=0.03)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_poses_figure(pcd, poses_world, path, max_points=40000):
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    if len(pts) > max_points:
        sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, cols = pts[sel], cols[sel]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=0.5)
    centers = np.array([T[:3, 3] for T in poses_world])
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], "b.-", label="cámaras")
    for i, T in enumerate(poses_world):
        c, z = T[:3, 3], T[:3, 2]  # centro y eje óptico
        ax.quiver(*c, *(z * 60), color="r", linewidth=0.8)
        ax.text(*c, str(i), fontsize=7)
    # ejes del mundo
    for v, color in zip(np.eye(3) * 100, "rgb"):
        ax.quiver(0, 0, 0, *v, color=color, linewidth=2)
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_zlabel("Z [mm]")
    ax.set_box_aspect(np.ptp(np.vstack([pts, centers]), axis=0))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_cloud_views(pcd, path, title, max_points=60000):
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    if len(pts) > max_points:
        sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, cols = pts[sel], cols[sel]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, (i, j, name) in zip(axes, [(0, 2, "frente (X-Z)"), (1, 2, "lateral (Y-Z)"), (0, 1, "planta (X-Y)")]):
        ax.scatter(pts[:, i], pts[:, j], c=cols, s=0.3)
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.set_xlabel("XYZ"[i] + " [mm]"); ax.set_ylabel("XYZ"[j] + " [mm]")
    fig.suptitle(f"{title} ({len(pcd.points):,} puntos)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

def step_calibration(cfg, dataset_dir, out_dir, fig_dir):
    print("\n[4.1] Calibración y rectificación estéreo")
    calib_dir = os.path.join(dataset_dir, "calib")

    if cfg.get("calibration_pkl"):
        calibration = stereo_calib.load_pickle(resolve(cfg["calibration_pkl"]))
        print("  calibración leída de", cfg["calibration_pkl"])
    else:
        checkerboard, square = read_checkerboard_txt(os.path.join(calib_dir, "checkerboard.txt"))
        print(f"  checkerboard {checkerboard}, cuadrado {square} mm")
        world, left_pts, right_pts, size, used = stereo_calib.detect_calibration_corners(
            calib_dir, checkerboard, square
        )
        calibration = stereo_calib.stereo_calibrate(
            world, left_pts, right_pts, size, mono_init=cfg.get("mono_init", True)
        )
        calibration["used_pairs"] = used
        save_checkerboard_figure(calib_dir, checkerboard, used, left_pts, right_pts,
                                 os.path.join(fig_dir, "00_deteccion_checkerboard.png"))
        save_per_view_errors_figure(calibration, os.path.join(fig_dir, "00_error_por_vista.png"))
    stereo_calib.print_calibration(calibration)

    if cfg.get("maps_pkl"):
        maps = stereo_calib.load_pickle(resolve(cfg["maps_pkl"]))
    else:
        maps = stereo_calib.stereo_rectify(calibration, alpha=cfg.get("rectify_alpha", 0))

    stereo_calib.save_pickle(calibration, os.path.join(out_dir, "stereo_calibration.pkl"))
    stereo_calib.save_pickle(maps, os.path.join(out_dir, "stereo_maps.pkl"))

    # figura original vs rectificada (un par de calibración)
    pairs = list_stereo_pairs(calib_dir)
    if pairs:
        _, lf, rf = pairs[len(pairs) // 2]
        left, right = cv2.imread(lf), cv2.imread(rf)
        save_rectification_figure(
            left, right, *stereo_calib.rectify_pair(left, right, maps),
            os.path.join(fig_dir, "01_rectificacion_calib.png"),
        )
    return calibration, maps


def step_reconstruction(cfg, dataset_dir, maps, fig_dir, inter_dir):
    print("\n[4.2] Disparidad, nube local y pose de cada captura")
    captures_dir = os.path.join(dataset_dir, "captures")
    markers = read_markers_yaml(os.path.join(captures_dir, "markers.yaml"))
    board = pose_mod.make_charuco_board(markers) if markers["type"] == "charuco" else None
    print("  objeto de referencia:", markers)

    K_rect = maps["P1"][:, :3]  # intrínseca de la cámara izquierda rectificada
    Q = maps["Q"]
    axis_len = (markers.get("square_length") or markers.get("square_size")) * 3

    all_points, all_colors = [], []
    poses_world, per_view = [], []

    for n, (idx, lf, rf) in enumerate(list_stereo_pairs(captures_dir)):
        left, right = cv2.imread(lf), cv2.imread(rf)
        left_r, right_r = stereo_calib.rectify_pair(left, right, maps)

        # 1. stereo matching
        disparity = stereo.compute_disparity(
            cv2.cvtColor(left_r, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(right_r, cv2.COLOR_BGR2GRAY),
            cfg.get("sgbm"),
        )

        # 2. nube local (cámara izquierda rectificada)
        points, colors = stereo.disparity_to_points(
            disparity, Q, left_r, z_range=cfg.get("z_range", (50, 2000))
        )

        # 3. pose global con solvePnP sobre la imagen izquierda rectificada
        p = pose_mod.estimate_pose(left_r, markers, K_rect, board=board)
        if p is None:
            print(f"  captura {idx}: no se detectó el objeto de referencia, se descarta")
            continue

        # 4. local -> mundo
        T_world_cam = FLIP_Z_UP @ pose_mod.pose_to_matrix(p["rvec"], p["tvec"])
        world_points = pose_mod.transform_points(points, T_world_cam)

        all_points.append(world_points)
        all_colors.append(colors)
        poses_world.append(T_world_cam)
        per_view.append({
            "idx": idx, "points": len(points), "pose_method": p["method"],
            "pnp_points": p["num_points"], "pnp_reproj_error_px": p["reproj_error"],
            "camera_center_mm": T_world_cam[:3, 3].round(1).tolist(),
        })
        print(f"  captura {idx:>2}: {len(points):>8,} pts | pose {p['method']} "
              f"({p['num_points']} pts, err {p['reproj_error']:.2f}px) | "
              f"cámara en {T_world_cam[:3, 3].round(0)}")

        if n == 0:
            save_rectification_figure(left, right, left_r, right_r,
                                      os.path.join(fig_dir, "02_rectificacion_captura.png"))
            save_disparity_figure(left_r, disparity, os.path.join(fig_dir, "03_disparidad.png"))
            cv2.imwrite(os.path.join(fig_dir, "04_pose_pnp.jpg"),
                        pose_mod.draw_pose(left_r, p["rvec"], p["tvec"], K_rect, axis_len=axis_len))
            local = fusion.make_pcd(points, colors).voxel_down_sample(2.0)
            o3d.io.write_point_cloud(os.path.join(inter_dir, "local_cloud_0.ply"), local)

    pcd = fusion.make_pcd(np.vstack(all_points), np.vstack(all_colors))
    return pcd, poses_world, per_view, K_rect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--show", action="store_true", help="abrir visores de Open3D")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset_dir = resolve(cfg["dataset"])
    out_dir = resolve(cfg["output"])
    fig_dir = os.path.join(out_dir, "figures")
    inter_dir = os.path.join(out_dir, "intermediate")
    for d in (out_dir, fig_dir, inter_dir):
        os.makedirs(d, exist_ok=True)

    calibration, maps = step_calibration(cfg, dataset_dir, out_dir, fig_dir)
    pcd, poses_world, per_view, K_rect = step_reconstruction(cfg, dataset_dir, maps, fig_dir, inter_dir)

    print("\n[4.3] Fusión y optimización")
    scene_preview = pcd.voxel_down_sample(3.0)
    o3d.io.write_point_cloud(os.path.join(inter_dir, "scene_preview.ply"), scene_preview)
    save_poses_figure(scene_preview.crop(o3d.geometry.AxisAlignedBoundingBox(
        [-200, -568, -50], [463, 200, 400])), poses_world, os.path.join(fig_dir, "05_poses_camaras.png"))

    out = cfg["outliers"]
    final, obb, plane, stages = fusion.optimize(
        pcd, cfg["box_min"], cfg["box_max"], cfg["voxel_size"],
        out["nb_neighbors"], out["std_ratio"], plane_margin=cfg.get("plane_margin"),
    )
    if cfg.get("largest_cluster_eps"):
        final = fusion.keep_largest_cluster(final, cfg["largest_cluster_eps"])
        stages["largest_cluster"] = len(final.points)
        print(f"  {'cluster':>13}: {stages['largest_cluster']:>10,} puntos")

    save_cloud_views(final, os.path.join(fig_dir, "06_nube_final.png"), "nube final optimizada")

    ply_path = os.path.join(out_dir, "reconstruction.ply")
    o3d.io.write_point_cloud(ply_path, final, write_ascii=False)
    print(f"  guardado {ply_path} ({os.path.getsize(ply_path) / 1e6:.1f} MB)")

    print("\n[5] Medición métrica")
    m = fusion.measure(final, plane)
    print(f"  alto (sobre el plano de apoyo): {m['height_mm']:.1f} mm (máx {m['height_max_mm']:.1f})")
    print(f"  ancho X: {m['width_x_mm']:.1f} mm | profundidad Y: {m['depth_y_mm']:.1f} mm")
    print(f"  OBB mínima (lados): {np.round(m['obb_extent_mm'], 1)} mm")
    if cfg.get("real_height_mm"):
        rel = abs(m["height_mm"] - cfg["real_height_mm"]) / cfg["real_height_mm"]
        m["height_rel_error"] = rel
        print(f"  error relativo vs real ({cfg['real_height_mm']} mm): {rel:.1%}")

    summary = {
        "calibration": {
            "rms_stereo_px": calibration.get("rms_stereo"),
            "rms_left_px": calibration.get("rms_left"),
            "rms_right_px": calibration.get("rms_right"),
            "baseline_mm": float(np.linalg.norm(calibration["T"])),
            "left_K": calibration["left_K"].tolist(),
            "right_K": calibration["right_K"].tolist(),
            "left_dist": calibration["left_dist"].ravel().tolist(),
            "right_dist": calibration["right_dist"].ravel().tolist(),
            "R": calibration["R"].tolist(),
            "T": calibration["T"].ravel().tolist(),
            "used_pairs": calibration.get("used_pairs"),
        },
        "views": per_view,
        "points_per_stage": stages,
        "measurement": m,
        "config": cfg,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    if args.show:
        world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
        cams = fusion.camera_geometries(poses_world, K_rect, calibration["image_size"])
        o3d.visualization.draw_geometries([scene_preview, world_axes, obb] + cams,
                                          window_name="escena + cámaras + caja")
        o3d.visualization.draw_geometries([final, world_axes], window_name="nube final")


if __name__ == "__main__":
    main()
