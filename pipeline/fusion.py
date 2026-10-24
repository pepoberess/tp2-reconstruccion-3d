import numpy as np
import open3d as o3d


def make_pcd(points, colors=None):
    """Construye una nube de puntos con colores opcionales."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def prepare_icp_cloud(pcd, box_min, box_max, params):
    """Copia liviana del objeto; excluye el tablero por su altura en el mundo."""
    lower = np.asarray(box_min, float).copy()
    lower[2] = max(lower[2], params.get("min_height_mm", 25.0))
    cloud, _ = crop_box(pcd, lower, box_max)
    return cloud.voxel_down_sample(params.get("voxel_size", 3.0))


def refine_with_icp(source, target, params=None):
    """Refina el registro y devuelve únicamente correcciones ICP aceptadas."""
    p = params or {}
    identity = np.eye(4)
    info = {"accepted": False, "reason": "pocos puntos", "fitness_before": None,
            "fitness_candidate": None, "rmse_before_mm": None,
            "rmse_candidate_mm": None, "overlap_rmse_before_mm": None,
            "overlap_rmse_candidate_mm": None, "shift_mm": None, "rotation_deg": None}
    minimum = p.get("min_points", 100)
    if min(len(source.points), len(target.points)) < minimum:
        return identity, info
    center = source.get_center()
    src = o3d.geometry.PointCloud(source).translate(-center)
    dst = o3d.geometry.PointCloud(target).translate(-center)
    reg = o3d.pipelines.registration
    distance = p.get("max_distance_mm", 5.0)
    before = reg.evaluate_registration(src, dst, distance, identity)
    info.update(fitness_before=float(before.fitness), rmse_before_mm=float(before.inlier_rmse))
    pairs = np.asarray(before.correspondence_set)
    if len(pairs) < minimum or before.fitness < p.get("min_fitness", 0.3):
        info["reason"] = "solapamiento insuficiente"
        return identity, info
    dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=p.get("normal_radius_mm", 10.0), max_nn=30))
    result = reg.registration_icp(
        src, dst, distance, identity, reg.TransformationEstimationPointToPlane(),
        reg.ICPConvergenceCriteria(max_iteration=p.get("max_iteration", 50)))
    T = np.asarray(result.transformation)
    if not np.isfinite(T).all():
        info["reason"] = "transformación no finita"
        return identity, info
    angle = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))
    shift = float(np.linalg.norm(T[:3, 3]))
    after = reg.evaluate_registration(src, dst, distance, T)
    a = np.asarray(src.points)[pairs[:, 0]]
    b = np.asarray(dst.points)[pairs[:, 1]]
    fixed_before = float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))
    moved_overlap = make_pcd(a @ T[:3, :3].T + T[:3, 3])
    distances_after = np.asarray(moved_overlap.compute_point_cloud_distance(dst))
    fixed_after = float(np.sqrt(np.mean(distances_after ** 2)))
    info.update(fitness_candidate=float(after.fitness), rmse_candidate_mm=float(after.inlier_rmse),
                overlap_rmse_before_mm=fixed_before, overlap_rmse_candidate_mm=fixed_after,
                shift_mm=shift, rotation_deg=angle)
    if shift > p.get("max_shift_mm", 10.0) or angle > p.get("max_rotation_deg", 3.0):
        info["reason"] = "corrección demasiado grande"
    elif after.fitness < max(p.get("min_fitness", 0.3), before.fitness - p.get("max_fitness_drop", 0.02)):
        info["reason"] = "pérdida de solapamiento"
    elif after.inlier_rmse >= before.inlier_rmse or fixed_after >= fixed_before:
        info["reason"] = "sin mejora de RMSE"
    else:
        info.update(accepted=True, reason="aceptado")
        world_T = T.copy()
        world_T[:3, 3] += center - T[:3, :3] @ center
        return world_T, info
    return identity, info


def crop_box(pcd, box_min, box_max):
    """Recorta una nube con una caja definida en coordenadas del mundo."""
    box_min, box_max = np.asarray(box_min, float), np.asarray(box_max, float)
    obb = o3d.geometry.OrientedBoundingBox(
        center=(box_min + box_max) / 2, R=np.eye(3), extent=box_max - box_min
    )
    idx = obb.get_point_indices_within_bounding_box(pcd.points)
    return pcd.select_by_index(idx), obb


def fit_support_plane(pcd, distance_threshold=3.0):
    """Estima por RANSAC el plano de apoyo orientado hacia arriba."""
    o3d.utility.random.seed(0)
    plane, _ = pcd.segment_plane(distance_threshold, ransac_n=3, num_iterations=2000)
    n, d = np.array(plane[:3]), plane[3]
    if n[2] < 0:
        n, d = -n, -d
    return n, d


def remove_plane(pcd, plane, margin):
    """Saca todo lo que esté a menos de `margin` mm por encima (o debajo) del plano."""
    n, d = plane
    dist = np.asarray(pcd.points) @ n + d
    return pcd.select_by_index(np.where(dist > margin)[0])


def optimize(pcd, box_min, box_max, voxel_size, nb_neighbors, std_ratio,
             plane_margin=None, log=print, clouds=None):
    """Aplica las etapas configuradas de limpieza de la nube."""
    stages = {"original": len(pcd.points)}

    pcd, obb = crop_box(pcd, box_min, box_max)
    stages["bounding_box"] = len(pcd.points)
    if clouds is not None:
        clouds["bounding_box"] = pcd

    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    stages["voxel"] = len(pcd.points)
    if clouds is not None:
        clouds["voxel"] = pcd

    plane = None
    if plane_margin is not None:
        plane = fit_support_plane(pcd)
        pcd = remove_plane(pcd, plane, plane_margin)
        stages["plane_removal"] = len(pcd.points)
        if clouds is not None:
            clouds["plane_removal"] = pcd
        log(f"  plano RANSAC: n={np.round(plane[0], 4)}, d={plane[1]:.2f}")

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    stages["outliers"] = len(pcd.points)
    if clouds is not None:
        clouds["outliers"] = pcd

    for name, n in stages.items():
        log(f"  {name:>13}: {n:>10,} puntos")
    return pcd, obb, plane, stages


def keep_largest_cluster(pcd, eps, min_points=20):
    """Se queda con el cluster más grande (DBSCAN)."""
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    if labels.max() < 0:
        return pcd
    largest = np.bincount(labels[labels >= 0]).argmax()
    return pcd.select_by_index(np.where(labels == largest)[0])


def poisson_mesh(pcd, depth=8, normal_radius=5.0, density_quantile=0.05, target_triangles=200_000,
                 min_component_ratio=0.02):
    """Genera y simplifica una malla mediante reconstrucción de Poisson."""
    pcd = o3d.geometry.PointCloud(pcd)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, density_quantile))

    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())

    if min_component_ratio:
        ids, counts, _ = mesh.cluster_connected_triangles()
        ids, counts = np.asarray(ids), np.asarray(counts)
        mesh.remove_triangles_by_mask(counts[ids] < min_component_ratio * counts.max())
        mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_triangles)
    mesh.compute_vertex_normals()
    return mesh


def measure(pcd, plane=None, low=0.5, high=99.5):
    """Mide la nube en milímetros usando percentiles robustos."""
    pts = np.asarray(pcd.points)
    lo = np.percentile(pts, low, axis=0)
    hi = np.percentile(pts, high, axis=0)
    aabb_dims = hi - lo

    if plane is not None:
        n, d = plane
        height = float(np.percentile(pts @ n + d, high))
    else:
        height = float(hi[2])

    obb = pcd.get_minimal_oriented_bounding_box()

    return {
        "height_mm": height,
        "height_max_mm": float((pts @ plane[0] + plane[1]).max()) if plane is not None else float(pts[:, 2].max()),
        "width_x_mm": float(aabb_dims[0]),
        "depth_y_mm": float(aabb_dims[1]),
        "aabb_dims_mm": aabb_dims.tolist(),
        "obb_extent_mm": sorted(np.asarray(obb.extent).tolist(), reverse=True),
    }


def relative_error(estimated, reference):
    """Error relativo absoluto entre una medida estimada y su valor real."""
    if reference <= 0:
        raise ValueError("La medida real debe ser mayor que cero.")
    return abs(float(estimated) - float(reference)) / float(reference)
