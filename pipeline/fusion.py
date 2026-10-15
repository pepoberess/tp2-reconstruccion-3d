"""
Paso 4.3 (fusión y optimización) y paso 5 (medición métrica) con Open3D.
"""
import numpy as np
import open3d as o3d


def make_pcd(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def crop_box(pcd, box_min, box_max):
    """
    Filtrado espacial: caja que encierra al objeto, en coordenadas del mundo.
    Como el mundo está alineado al tablero, la caja orientada coincide con una
    caja alineada a los ejes; usamos OrientedBoundingBox como pide el enunciado.
    """
    box_min, box_max = np.asarray(box_min, float), np.asarray(box_max, float)
    obb = o3d.geometry.OrientedBoundingBox(
        center=(box_min + box_max) / 2, R=np.eye(3), extent=box_max - box_min
    )
    idx = obb.get_point_indices_within_bounding_box(pcd.points)
    return pcd.select_by_index(idx), obb


def fit_support_plane(pcd, distance_threshold=3.0):
    """
    RANSAC: plano del tablero/mesa donde se apoya el objeto.
    Devuelve (n, d) con n·x + d = 0 y n apuntando hacia arriba (+Z del mundo).
    """
    o3d.utility.random.seed(0)  # RANSAC es aleatorio: semilla fija para que sea reproducible
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
             plane_margin=None, log=print):
    """bounding box -> voxel downsample -> [plano del tablero] -> statistical outlier removal."""
    stages = {"original": len(pcd.points)}

    pcd, obb = crop_box(pcd, box_min, box_max)
    stages["bounding_box"] = len(pcd.points)

    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    stages["voxel"] = len(pcd.points)

    plane = None
    if plane_margin is not None:
        plane = fit_support_plane(pcd)
        pcd = remove_plane(pcd, plane, plane_margin)
        stages["plane_removal"] = len(pcd.points)
        log(f"  plano RANSAC: n={np.round(plane[0], 4)}, d={plane[1]:.2f}")

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    stages["outliers"] = len(pcd.points)

    for name, n in stages.items():
        log(f"  {name:>13}: {n:>10,} puntos")
    return pcd, obb, plane, stages


def keep_largest_cluster(pcd, eps, min_points=20):
    """Opcional: se queda con el cluster más grande (DBSCAN), útil para sacar restos del tablero."""
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    if labels.max() < 0:
        return pcd
    largest = np.bincount(labels[labels >= 0]).argmax()
    return pcd.select_by_index(np.where(labels == largest)[0])


def measure(pcd, plane=None, low=0.5, high=99.5):
    """
    Medición métrica en el marco del mundo (mm, Z hacia arriba desde el tablero).
    Usamos percentiles en lugar de min/max para que algún punto espurio no
    infle la medida.
    """
    pts = np.asarray(pcd.points)
    lo = np.percentile(pts, low, axis=0)
    hi = np.percentile(pts, high, axis=0)
    aabb_dims = hi - lo

    if plane is not None:
        # altura = distancia máxima al plano de apoyo estimado por RANSAC
        n, d = plane
        height = float(np.percentile(pts @ n + d, high))
    else:
        # altura respecto de Z=0 (plano del tablero según la pose)
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


def camera_geometries(poses_world, K, image_size, scale=40.0):
    """Frustums + ejes de cada cámara, para visualizar en Open3D."""
    w, h = image_size
    geoms = []
    for T in poses_world:
        extrinsic = np.linalg.inv(T)  # mundo -> cámara
        frustum = o3d.geometry.LineSet.create_camera_visualization(
            w, h, K, extrinsic, scale=scale
        )
        frustum.paint_uniform_color([0.1, 0.3, 1.0])
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale * 0.6)
        axes.transform(T)
        geoms += [frustum, axes]
    return geoms
