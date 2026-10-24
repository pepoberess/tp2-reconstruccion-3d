"""
Pasos del pipeline que usa el notebook (calibración y reconstrucción de cada captura)
y las funciones que dibujan las figuras de output/<dataset>/figures/.
"""
import hashlib
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import calib
import calibration as stereo_calib
import fusion
import pose as pose_mod
import stereo
from dataset import list_stereo_pairs, read_checkerboard_txt, read_markers_yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FLIP_Z_UP = np.diag([1.0, -1.0, -1.0, 1.0])


def resolve(path):
    """Resuelve una ruta relativa desde la raíz del repositorio."""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def save_checkerboard_figure(calib_dir, checkerboard, used, left_pts, right_pts, path):
    """Detección de corners en un par de calibración."""
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


def save_per_view_errors_figure(calibration, path, threshold=None):
    """Error de reproyección por par."""
    errors = calibration["per_view_errors"]
    used = calibration["used_pairs"]
    x = np.arange(len(used))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(x - 0.2, errors[:, 0], 0.4, label="left")
    ax.bar(x + 0.2, errors[:, 1], 0.4, label="right")
    ax.axhline(calibration["rms_stereo"], color="k", linestyle="--",
               label=f"RMS global = {calibration['rms_stereo']:.3f} px")
    if threshold:
        ax.axhline(threshold, color="tab:red", linestyle=":",
                   label=f"umbral para descartar = {threshold} px")
    step = max(1, len(used) // 10)
    ax.set_xticks(x[::step], [str(u) for u in used[::step]])
    ax.set_xlim(-1, len(used))
    ax.set_xlabel("par de calibración")
    ax.set_ylabel("error de reproyección [px]")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_rectification_figure(left, right, left_r, right_r, path, n_lines=8):
    """Guarda una comparación del par original y rectificado."""
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


def save_disparity_figure(left_r, disparity, path, title="disparidad [px]"):
    """Guarda la imagen izquierda junto con su mapa de disparidad."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].imshow(cv2.cvtColor(left_r, cv2.COLOR_BGR2RGB))
    axes[0].set_title("left rectificada")
    d = np.where(disparity > 0, disparity, np.nan)
    im = axes[1].imshow(d, cmap="turbo")
    axes[1].set_title(title)
    fig.colorbar(im, ax=axes[1], fraction=0.03)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_disparity_comparison(left_r, sgbm, cre, path):
    """Mismo par con SGBM (clásico) y CREStereo (red neuronal), misma escala de color."""
    vmax = np.nanpercentile(np.where(cre > 0, cre, np.nan), 99)
    fig, axes = plt.subplots(1, 3, figsize=(20, 4.5))
    axes[0].imshow(cv2.cvtColor(left_r, cv2.COLOR_BGR2RGB))
    axes[0].set_title("left rectificada")
    for ax, d, name in [(axes[1], sgbm, "SGBM"), (axes[2], cre, "CREStereo")]:
        valid = (d > 0).mean()
        im = ax.imshow(np.where(d > 0, d, np.nan), cmap="turbo", vmin=0, vmax=vmax)
        ax.set_title(f"{name} ({valid:.0%} píxeles válidos)")
    fig.colorbar(im, ax=axes[1:], fraction=0.02, label="disparidad [px]")
    for ax in axes:
        ax.axis("off")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_pnp_errors_figure(per_view, path):
    """Error de reproyección de solvePnP en cada captura (calidad de cada pose)."""
    idx = [v["idx"] for v in per_view]
    err = np.array([v["pnp_reproj_error_px"] for v in per_view], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(range(len(idx)), err, color="tab:blue")
    ax.axhline(np.nanmean(err), color="k", linestyle="--",
               label=f"media = {np.nanmean(err):.2f} px")
    step = max(1, len(idx) // 10)
    ax.set_xticks(range(0, len(idx), step), [str(i) for i in idx[::step]])
    ax.set_xlim(-1, len(idx))
    ax.set_xlabel("captura")
    ax.set_ylabel("error de reproyección PnP [px]")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def frustum_lines(T_world_cam, K, image_size, depth):
    """Segmentos 3D de una cámara dibujada como pirámide (centro + plano imagen a `depth` mm)."""
    w, h = image_size
    pixels = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    corners = np.c_[(pixels - K[:2, 2]) / np.diag(K)[:2], np.ones(4)] * depth
    corners = corners @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
    center = T_world_cam[:3, 3]
    lines = [(center, c) for c in corners]
    lines += [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    return lines


def save_poses_figure(pcd, poses_world, path, K, image_size, max_points=40000):
    """Guarda la nube junto con las poses estimadas de las cámaras."""
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
    label_step = max(1, len(poses_world) // 12)
    for i, T in enumerate(poses_world):
        for a, b in frustum_lines(T, K, image_size, depth=15):
            ax.plot(*np.c_[a, b], color="r", linewidth=0.5)
        if i % label_step == 0:
            ax.text(*(T[:3, 3] + [0, 0, 12]), str(i), fontsize=8, ha="center")
    for v, color in zip(np.eye(3) * 100, "rgb"):
        ax.quiver(0, 0, 0, *v, color=color, linewidth=2)
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_zlabel("Z [mm]")
    ax.set_box_aspect(np.ptp(np.vstack([pts, centers]), axis=0))
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_cloud_views(pcd, path, title, max_points=60000):
    """Guarda vistas ortogonales de una nube de puntos."""
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


def save_stages_figure(clouds, path, max_points=40000):
    """Nube después de cada etapa de optimización, vista lateral 3D, con la cantidad de puntos."""
    names = {"bounding_box": "bounding box", "voxel": "voxel downsample",
             "plane_removal": "sin plano del tablero", "outliers": "sin outliers",
             "largest_cluster": "cluster principal"}
    keys = [k for k in names if k in clouds]
    fig = plt.figure(figsize=(4.5 * len(keys), 5))
    rng = np.random.default_rng(0)
    for k, key in enumerate(keys):
        pts = np.asarray(clouds[key].points)
        cols = np.asarray(clouds[key].colors)
        n = len(pts)
        if n > max_points:
            sel = rng.choice(n, max_points, replace=False)
            pts, cols = pts[sel], cols[sel]
        ax = fig.add_subplot(1, len(keys), k + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=0.3)
        ax.set_box_aspect(np.ptp(pts, axis=0))
        ax.view_init(elev=25, azim=-60)
        ax.set_title(f"{k + 1}. {names[key]}\n{n:,} puntos", fontsize=20)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_mesh_figure(mesh, path, max_triangles=60000):
    """Render de la malla con matplotlib (vista en perspectiva, color por vértice)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if len(mesh.triangles) > max_triangles:
        mesh = mesh.simplify_quadric_decimation(max_triangles)
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)[t].mean(axis=1)

    fig = plt.figure(figsize=(14, 6))
    for k, azim in enumerate([-60, 30, 120]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(v[t], facecolors=colors, linewidths=0))
        ax.set_xlim(v[:, 0].min(), v[:, 0].max())
        ax.set_ylim(v[:, 1].min(), v[:, 1].max())
        ax.set_zlim(v[:, 2].min(), v[:, 2].max())
        ax.set_box_aspect(np.ptp(v, axis=0))
        ax.view_init(elev=20, azim=azim)
        ax.axis("off")
    fig.suptitle(f"malla Poisson ({len(t):,} triángulos)")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def step_calibration(cfg, dataset_dir, out_dir, fig_dir):
    """Ejecuta la calibración estéreo y guarda sus resultados."""
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
        mono_init = cfg.get("mono_init", True)
        calibration = stereo_calib.stereo_calibrate(world, left_pts, right_pts, size, mono_init=mono_init)
        calibration["used_pairs"] = used
        save_checkerboard_figure(calib_dir, checkerboard, used, left_pts, right_pts,
                                 os.path.join(fig_dir, "00_deteccion_checkerboard.png"))

        max_error = cfg.get("max_view_error_px")
        save_per_view_errors_figure(calibration, os.path.join(fig_dir, "00_error_por_vista.png"),
                                    threshold=max_error)
        if max_error:
            keep = calibration["per_view_errors"].max(axis=1) <= max_error
            dropped = [u for u, k in zip(used, keep) if not k]
            if dropped:
                print(f"  se descartan {len(dropped)} pares con error > {max_error} px: {dropped}")
                world = [w for w, k in zip(world, keep) if k]
                left_pts = [p for p, k in zip(left_pts, keep) if k]
                right_pts = [p for p, k in zip(right_pts, keep) if k]
                used = [u for u, k in zip(used, keep) if k]
                rms_before = calibration["rms_stereo"]
                calibration = stereo_calib.stereo_calibrate(world, left_pts, right_pts, size,
                                                            mono_init=mono_init)
                calibration["used_pairs"] = used
                calibration["dropped_pairs"] = dropped
                calibration["rms_stereo_before_drop"] = rms_before
                save_per_view_errors_figure(calibration, os.path.join(fig_dir, "00b_error_por_vista_final.png"))
    stereo_calib.print_calibration(calibration)

    if cfg.get("maps_pkl"):
        maps = stereo_calib.load_pickle(resolve(cfg["maps_pkl"]))
    else:
        maps = stereo_calib.stereo_rectify(calibration, alpha=cfg.get("rectify_alpha", 0))

    stereo_calib.save_pickle(calibration, os.path.join(out_dir, "stereo_calibration.pkl"))
    stereo_calib.save_pickle(maps, os.path.join(out_dir, "stereo_maps.pkl"))

    pairs = list_stereo_pairs(calib_dir)
    if pairs:
        _, lf, rf = pairs[len(pairs) // 2]
        left, right = cv2.imread(lf), cv2.imread(rf)
        save_rectification_figure(
            left, right, *stereo_calib.rectify_pair(left, right, maps),
            os.path.join(fig_dir, "01_rectificacion_calib.png"),
        )
    return calibration, maps


def method_name(cfg):
    """Devuelve el nombre del método de disparidad configurado."""
    return "CREStereo" if cfg.get("disparity_method", "sgbm") == "crestereo" else "SGBM"


def compute_disparity(cfg, left_r, right_r, idx, inter_dir):
    """SGBM (rápido) o CREStereo (red neuronal, ~15 s por par; se cachea en intermediate/)."""
    if cfg.get("disparity_method", "sgbm") != "crestereo":
        return stereo.compute_disparity(cv2.cvtColor(left_r, cv2.COLOR_BGR2GRAY),
                                        cv2.cvtColor(right_r, cv2.COLOR_BGR2GRAY), cfg.get("sgbm"))

    cre = cfg.get("crestereo", {})
    image_hash = hashlib.md5(left_r.tobytes() + right_r.tobytes()).hexdigest()[:10]
    cache = os.path.join(inter_dir, f"disp_crestereo_{idx}_{image_hash}.npy")
    if os.path.exists(cache):
        return np.load(cache).astype(np.float32)
    disparity = stereo.compute_disparity_cre(
        left_r, right_r, resolve(cre.get("models_dir", "models")),
        shape=cre.get("shape", "1280x720"), iterations=cre.get("iterations", 10),
    )
    np.save(cache, disparity.astype(np.float16))
    return disparity


def step_reconstruction(cfg, dataset_dir, maps, fig_dir, inter_dir, comparison=None):
    """Reconstruye y fusiona las vistas del objeto."""
    print("\n[4.2] Disparidad, nube local y pose de cada captura")
    captures_dir = os.path.join(dataset_dir, "captures")
    markers = read_markers_yaml(os.path.join(captures_dir, "markers.yaml"))
    board = pose_mod.make_charuco_board(markers) if markers["type"] == "charuco" else None
    print("  objeto de referencia:", markers)

    K_rect = maps["P1"][:, :3]
    Q = maps["Q"]
    axis_len = (markers.get("square_length") or markers.get("square_size")) * 3

    all_points, all_colors = [], []
    poses_world, per_view = [], []
    icp_params = cfg.get("icp") or {}
    use_icp = icp_params.get("enabled", False)
    target = o3d.geometry.PointCloud()
    baseline_points, baseline_colors, baseline_poses = [], [], []
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(inter_dir, exist_ok=True)

    for n, (idx, lf, rf) in enumerate(list_stereo_pairs(captures_dir)):
        left, right = cv2.imread(lf), cv2.imread(rf)
        left_r, right_r = stereo_calib.rectify_pair(left, right, maps)

        disparity = compute_disparity(cfg, left_r, right_r, idx, inter_dir)

        p = pose_mod.estimate_pose(left_r, markers, K_rect, board=board)
        if p is None:
            print(f"  captura {idx}: no se detectó el objeto de referencia, se descarta")
            continue

        correction = None
        if cfg.get("board_disparity_correction") and "object_points" in p:
            correction = stereo.board_disparity_correction(
                disparity, Q, p["object_points"], p["image_points"], p["rvec"], p["tvec"])
            if correction is not None:
                disparity = stereo.apply_disparity_correction(disparity, *correction[:2])

        points, colors = stereo.disparity_to_points(
            disparity, Q, left_r, z_range=cfg.get("z_range", (50, 2000))
        )

        T_world_cam = FLIP_Z_UP @ pose_mod.pose_to_matrix(p["rvec"], p["tvec"])
        world_points = pose_mod.transform_points(points, T_world_cam)

        icp_info = None
        if use_icp:
            world_cloud = fusion.make_pcd(world_points, colors)
            source = fusion.prepare_icp_cloud(world_cloud, cfg["box_min"], cfg["box_max"], icp_params)
            if comparison is not None:
                baseline, _ = fusion.crop_box(world_cloud, cfg["box_min"], cfg["box_max"])
                baseline_points.append(np.asarray(baseline.points).copy())
                baseline_colors.append(np.asarray(baseline.colors).copy())
                baseline_poses.append(T_world_cam.copy())
            if len(target.points) == 0:
                delta = np.eye(4)
                _, icp_info = fusion.refine_with_icp(source, target, icp_params)
                if len(source.points) >= icp_params.get("min_points", 100):
                    target = o3d.geometry.PointCloud(source)
                    icp_info["reason"] = "referencia fija"
            else:
                delta, icp_info = fusion.refine_with_icp(source, target, icp_params)
                if icp_info["accepted"]:
                    source.transform(delta)
                    target = (target + source).voxel_down_sample(icp_params.get("voxel_size", 3.0))
            world_cloud.transform(delta)
            T_world_cam = delta @ T_world_cam
            world_cloud, _ = fusion.crop_box(world_cloud, cfg["box_min"], cfg["box_max"])
            world_points = np.asarray(world_cloud.points).copy()
            output_colors = np.asarray(world_cloud.colors).copy()
            icp_info["delta_T"] = delta.tolist()
            print(f"  ICP captura {idx}: {icp_info['reason']}")
        else:
            output_colors = colors

        all_points.append(world_points)
        all_colors.append(output_colors)
        poses_world.append(T_world_cam)
        per_view.append({
            "idx": idx, "points": len(points), "pose_method": p["method"],
            "pnp_points": p["num_points"], "pnp_reproj_error_px": p["reproj_error"],
            "camera_center_mm": T_world_cam[:3, 3].round(1).tolist(),
            "disparity_scale": correction[0] if correction else None,
            "disparity_offset_px": correction[1] if correction else None,
            "icp": icp_info,
        })
        fix = f" | disp. corregida: x{correction[0]:.3f} {correction[1]:+.1f}px" if correction else ""
        print(f"  captura {idx:>2}: {len(points):>8,} pts | pose {p['method']} "
              f"({p['num_points']} pts, err {p['reproj_error']:.2f}px) | "
              f"cámara en {T_world_cam[:3, 3].round(0)}{fix}")

        if n == 0:
            save_rectification_figure(left, right, left_r, right_r,
                                      os.path.join(fig_dir, "02_rectificacion_captura.png"))
            save_disparity_figure(left_r, disparity, os.path.join(fig_dir, "03_disparidad.png"),
                                  title=f"disparidad {method_name(cfg)} [px]")
            if method_name(cfg) == "CREStereo":
                sgbm = stereo.compute_disparity(cv2.cvtColor(left_r, cv2.COLOR_BGR2GRAY),
                                                cv2.cvtColor(right_r, cv2.COLOR_BGR2GRAY), cfg.get("sgbm"))
                save_disparity_comparison(left_r, sgbm, disparity,
                                          os.path.join(fig_dir, "03b_sgbm_vs_crestereo.png"))
            cv2.imwrite(os.path.join(fig_dir, "04_pose_pnp.jpg"),
                        pose_mod.draw_pose(left_r, p["rvec"], p["tvec"], K_rect, axis_len=axis_len))
            local = fusion.make_pcd(points, colors).voxel_down_sample(2.0)
            o3d.io.write_point_cloud(os.path.join(inter_dir, "local_cloud_0.ply"), local)

    if not all_points:
        raise ValueError("No hay capturas con una pose válida para reconstruir.")
    if use_icp and comparison is not None:
        comparison["before"] = fusion.make_pcd(np.vstack(baseline_points), np.vstack(baseline_colors))
        comparison["poses_before"] = baseline_poses
    pcd = fusion.make_pcd(np.vstack(all_points), np.vstack(all_colors))
    return pcd, poses_world, per_view, K_rect
