"""Comparación reproducible PnP / PnP + ICP, utilizada por la sección 8 del notebook."""
import copy
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

import fusion
import run_pipeline


def _finish(cloud, cfg, plane):
    """Mismos filtros y mismo plano de apoyo para ambas variantes."""
    cloud = cloud.voxel_down_sample(cfg["voxel_size"])
    if cfg.get("plane_margin") is not None:
        cloud = fusion.remove_plane(cloud, plane, cfg["plane_margin"])
    cloud, _ = cloud.remove_statistical_outlier(**cfg["outliers"])
    if len(cloud.points) and cfg.get("largest_cluster_eps"):
        cloud = fusion.keep_largest_cluster(cloud, cfg["largest_cluster_eps"])
    if len(cloud.points) < 4:
        raise ValueError("La nube quedó vacía o tiene muy pocos puntos después de filtrar.")
    return cloud


def _comparison_figure(before, after, path):
    """Guarda una comparación visual de las nubes antes y después de ICP."""
    clouds = [before, after]
    limits = np.vstack([np.asarray(c.points) for c in clouds])
    lo, hi = limits.min(axis=0), limits.max(axis=0)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for row, (cloud, label) in enumerate(zip(clouds, ["ChArUco", "ChArUco + ICP"])):
        pts, cols = np.asarray(cloud.points), np.asarray(cloud.colors)
        sel = np.random.default_rng(0).choice(len(pts), min(len(pts), 60000), replace=False)
        for col, (i, j, name) in enumerate([(0, 2, "frente"), (1, 2, "lateral"), (0, 1, "planta")]):
            ax = axes[row, col]
            ax.scatter(pts[sel, i], pts[sel, j], c=cols[sel], s=0.3)
            ax.set(xlim=(lo[i], hi[i]), ylim=(lo[j], hi[j]),
                   xlabel="XYZ"[i] + " [mm]", ylabel="XYZ"[j] + " [mm]",
                   title=f"{label}: {name}")
            ax.set_aspect("equal")
    fig.suptitle("Comparación con los mismos filtros y escalas")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def metrics_markdown(result):
    """Tabla por vista; diferencia explícitamente intentos y correcciones aceptadas."""
    lines = ["| Captura | Estado | Fitness antes / intento | RMSE antes / intento [mm] | Desplazamiento [mm] | Giro [°] |",
             "|---|---|---:|---:|---:|---:|"]
    def number(x):
        """Formatea una métrica opcional para la tabla."""
        return "—" if x is None else f"{x:.3f}"
    for row in result["views"]:
        m = row["icp"]
        lines.append(f"| {row['idx']} | {m['reason']} | "
                     f"{number(m['fitness_before'])} / {number(m['fitness_candidate'])} | "
                     f"{number(m['rmse_before_mm'])} / {number(m['rmse_candidate_mm'])} | "
                     f"{number(m['shift_mm'])} | {number(m['rotation_deg'])} |")
    return "\n".join(lines)


def measures_markdown(result):
    """Genera la tabla Markdown con las medidas comparadas."""
    lines = ["| Medida | ChArUco | ChArUco + ICP | Cambio |", "|---|---:|---:|---:|"]
    for key, label in [("height_mm", "Alto [mm]"), ("width_x_mm", "Ancho X [mm]"),
                       ("depth_y_mm", "Profundidad Y [mm]")]:
        a, b = result["measures_before"][key], result["measures_after"][key]
        lines.append(f"| {label} | {a:.1f} | {b:.1f} | {b-a:+.1f} |")
    if result.get("real_height_mm"):
        a, b = result["relative_height_error_before"], result["relative_height_error_after"]
        lines.append(f"| Error relativo del alto | {a:.1%} | {b:.1%} | {b-a:+.1%} |")
    lines.append(f"| Puntos | {result['points_before']:,} | {result['points_after']:,} | "
                 f"{result['points_after']-result['points_before']:+,} |")
    return "\n".join(lines)


def discussion_markdown(result):
    """Resume en Markdown los resultados del refinamiento ICP."""
    accepted = [v["icp"] for v in result["views"] if v["icp"]["accepted"]]
    n = len(result["views"])
    text = (f"Se procesaron {n} capturas con {result['method']}. ICP aceptó correcciones en "
            f"{len(accepted)} vistas. La primera vista utilizable quedó fija como referencia; "
            "en los intentos rechazados se conservó la pose del ChArUco.")
    if accepted:
        a = np.mean([v["rmse_before_mm"] for v in accepted])
        b = np.mean([v["rmse_candidate_mm"] for v in accepted])
        text += (f" En las vistas aceptadas, el RMSE medio de correspondencias pasó de {a:.2f} "
                 f"a {b:.2f} mm. Estos valores se calculan contra la referencia acumulada de cada "
                 "paso; no representan el error respecto del objeto real.")
    else:
        text += " Con estos parámetros no se aceptaron refinamientos; la salida conserva las poses iniciales."
    text += ("\n\nLa figura permite revisar si disminuyen las superficies duplicadas y si se conserva "
             "la forma del objeto. Un RMSE menor no demuestra por sí solo una reconstrucción más exacta: "
             "ICP puede alinear ruido o acumular deriva. Las dos variantes usan el mismo plano de apoyo "
             "para medir, de modo que el cambio de altura no proviene de redefinir el cero.")
    if result.get("real_height_mm"):
        before = result["relative_height_error_before"]
        after = result["relative_height_error_after"]
        closer = "base con ChArUco" if before <= after else "refinada con ICP"
        text += (f" Frente a la altura real de {result['real_height_mm']:.1f} mm, los errores "
                 f"relativos son {before:.1%} y {after:.1%}, respectivamente; la reconstrucción "
                 f"{closer} queda más cerca para esta medida.")
    else:
        text += " Falta una medida física para evaluar la exactitud métrica."
    return text


def run(cfg, dataset_dir, maps, cache_dir, output_dir, image_size):
    """Reutiliza disparidades; exporta resultados independientes en output_dir/icp/."""
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("icp", {})["enabled"] = True
    folder = os.path.join(output_dir, "icp")
    figures = os.path.join(folder, "figures")
    os.makedirs(figures, exist_ok=True)
    comparison = {}
    after, poses, views, K = run_pipeline.step_reconstruction(
        cfg, dataset_dir, maps, figures, cache_dir, comparison=comparison)
    before = comparison.pop("before")
    plane = fusion.fit_support_plane(before.voxel_down_sample(cfg["voxel_size"]))
    before = _finish(before, cfg, plane)
    after = _finish(after, cfg, plane)
    figure_path = os.path.join(figures, "08_icp_comparacion.png")
    _comparison_figure(before, after, figure_path)
    run_pipeline.save_poses_figure(after, poses, os.path.join(figures, "08b_poses_icp.png"), K, image_size)
    result = {
        "method": run_pipeline.method_name(cfg), "parameters": cfg["icp"], "views": views,
        "points_before": len(before.points), "points_after": len(after.points),
        "measures_before": fusion.measure(before, plane), "measures_after": fusion.measure(after, plane),
        "support_plane": [*plane[0].tolist(), float(plane[1])],
        "poses_before": [T.tolist() for T in comparison["poses_before"]],
        "poses_after": [T.tolist() for T in poses],
        "states": dict(Counter(v["icp"]["reason"] for v in views)),
    }
    real_height = cfg.get("real_height_mm")
    if real_height:
        result.update(
            real_height_mm=float(real_height),
            relative_height_error_before=fusion.relative_error(
                result["measures_before"]["height_mm"], real_height),
            relative_height_error_after=fusion.relative_error(
                result["measures_after"]["height_mm"], real_height),
        )
    for cloud, name in [(before, "reconstruction_before_icp.ply"), (after, "reconstruction_icp.ply")]:
        if not o3d.io.write_point_cloud(os.path.join(folder, name), cloud, write_ascii=False):
            raise OSError(f"No se pudo escribir {name}")
    if cfg.get("poisson"):
        mesh = fusion.poisson_mesh(after, **cfg["poisson"])
        if not o3d.io.write_triangle_mesh(os.path.join(folder, "reconstruction_mesh_icp.ply"), mesh):
            raise OSError("No se pudo escribir la malla ICP")
        run_pipeline.save_mesh_figure(mesh, os.path.join(figures, "08c_malla_icp.png"))
    with open(os.path.join(folder, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


if __name__ == "__main__":
    import argparse
    import yaml
    import calibration
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grupo.yaml")
    args = parser.parse_args()
    with open(run_pipeline.resolve(args.config), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    output = run_pipeline.resolve(config["output"])
    calibration_path = config.get("calibration_pkl") or os.path.join(output, "stereo_calibration.pkl")
    calibration_params = calibration.load_pickle(run_pipeline.resolve(calibration_path))
    maps = (calibration.load_pickle(run_pipeline.resolve(config["maps_pkl"])) if config.get("maps_pkl")
            else calibration.stereo_rectify(calibration_params, alpha=config.get("rectify_alpha", 0)))
    results = run(config, run_pipeline.resolve(config["dataset"]), maps,
                  os.path.join(output, "intermediate"), output, calibration_params["image_size"])
    print(discussion_markdown(results))
    print(measures_markdown(results))
