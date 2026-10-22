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
             plane_margin=None, log=print, clouds=None):
    """
    bounding box -> voxel downsample -> [plano del tablero] -> statistical outlier removal.
    Si se pasa un dict en `clouds`, guarda ahí la nube resultante de cada etapa
    (para graficar el antes/después).
    """
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
    """
    Malla por reconstrucción de Poisson.
    1. normales por vecindad, orientadas consistentemente
    2. Poisson (profundidad del octree `depth`: más alto = más detalle)
    3. se sacan los triángulos de baja densidad (Poisson "inventa" superficie
       donde no hay puntos, sobre todo en los bordes abiertos)
    4. se sacan los pedazos sueltos de malla con menos de `min_component_ratio`
       de los triángulos del pedazo más grande (flecos que salen del ruido)
    """
    pcd = o3d.geometry.PointCloud(pcd)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, density_quantile))

    # recorta a la caja de la nube (Poisson cierra la superficie más allá de los datos)
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())

    if min_component_ratio:
        ids, counts, _ = mesh.cluster_connected_triangles()
        ids, counts = np.asarray(ids), np.asarray(counts)
        mesh.remove_triangles_by_mask(counts[ids] < min_component_ratio * counts.max())
        mesh.remove_unreferenced_vertices()
    # simplificación (quadric decimation) para que el archivo sea manejable
    if len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_triangles)
    mesh.compute_vertex_normals()
    return mesh


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
