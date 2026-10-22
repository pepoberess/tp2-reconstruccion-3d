"""
Estimamos la pose sobre la imagen izquierda RECTIFICADA, usando como intrínseca
K_rect = P1[:, :3] y distorsión nula (remap ya des-distorsionó). Así la pose
obtenida corresponde exactamente al sistema en el que reprojectImageTo3D
devuelve los puntos, y no hace falta componer con R1.
"""
import cv2
import numpy as np

import aruco
import calib


# ---------------------------------------------------------------------------
# Detección del objeto de referencia -> correspondencias 3D (mundo) / 2D (imagen)
# ---------------------------------------------------------------------------

def make_charuco_board(markers):
    return aruco.create_charuco_board(
        squares_x=markers["squares_x"],
        squares_y=markers["squares_y"],
        square_length=markers["square_length"],
        marker_length=markers["marker_length"],
        dictionary_type=getattr(cv2.aruco, markers["dictionary"]),
    )


def charuco_correspondences(image, board, min_markers=2):
    """
    Correspondencias para solvePnP a partir de un tablero ChArUco:
      - las 4 esquinas de cada marcador ArUco detectado, y
      - los corners ChArUco interpolados (más precisos, sub-pixel) si los hay.
    Con el objeto encima del tablero muchas veces se interpolan muy pocos corners
    ChArUco (necesitan marcadores vecinos visibles), por eso sumamos los marcadores.
    """
    detector = cv2.aruco.CharucoDetector(board)
    ch_corners, ch_ids, marker_corners, marker_ids = detector.detectBoard(image)
    if marker_ids is None or len(marker_ids) < min_markers:
        return None, None, (marker_corners, marker_ids)

    board_ids = list(board.getIds().flatten())
    board_obj = board.getObjPoints()
    obj, img = [], []
    for corners, marker_id in zip(marker_corners, marker_ids.flatten()):
        if marker_id not in board_ids:
            continue
        obj.extend(board_obj[board_ids.index(marker_id)])
        img.extend(corners.reshape(4, 2))

    if ch_ids is not None and len(ch_ids) > 0:
        obj.extend(board.getChessboardCorners()[ch_ids.flatten()])
        img.extend(ch_corners.reshape(-1, 2))

    obj = np.array(obj, dtype=np.float32)
    img = np.array(img, dtype=np.float32)
    return obj, img, (marker_corners, marker_ids)


def checkerboard_correspondences(image, pattern_size, square_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ok, corners = calib.detect_board(pattern_size, gray)
    if not ok:
        return None, None
    obj = calib.board_points(pattern_size).reshape(-1, 3) * square_size
    return obj, corners.reshape(-1, 2)


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------

def estimate_pose(image, markers, K, dist=None, board=None):
    """
    Devuelve dict con rvec, tvec (mundo -> cámara), error de reproyección y
    cantidad de puntos usados; o None si no se encontró el patrón.
    """
    if dist is None:
        dist = np.zeros(5)

    if markers["type"] == "charuco":
        obj, img, _ = charuco_correspondences(image, board)
    else:
        obj, img = checkerboard_correspondences(
            image, markers["pattern_size"], markers["square_size"]
        )

    if obj is None:
        if markers["type"] == "charuco":
            # fallback: método por homografía de la cátedra (aguanta mucha oclusión)
            detection = aruco.detect_charuco_markers(image, board)
            if detection is None:
                return None
            res = aruco.estimate_camera_pose_with_homography(image, board, detection, (K, dist))
            if res is None:
                return None
            _, rvec, tvec = res
            return {"rvec": rvec, "tvec": tvec, "reproj_error": np.nan,
                    "num_points": 0, "method": "homography"}
        return None

    # todos los puntos son coplanares: IPPE es el solver indicado para el caso planar
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    # refinamiento no lineal (Levenberg-Marquardt) minimizando error de reproyección
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)

    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    return {"rvec": rvec, "tvec": tvec, "reproj_error": err,
            "num_points": len(obj), "method": "pnp",
            # correspondencias usadas (sirven para corregir la disparidad con el tablero)
            "object_points": obj, "image_points": img}


def pose_to_matrix(rvec, tvec):
    """
    [R|t] mapea mundo -> cámara:  X_cam = R X_world + t.
    Devuelve T_world_cam (4x4), la inversa, que mapea cámara -> mundo:
        X_world = R^T (X_cam - t)
    """
    R, _ = cv2.Rodrigues(rvec)  # vector de Rodrigues -> matriz de rotación
    t = tvec.reshape(3)
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t
    return T


def transform_points(points, T):
    return points @ T[:3, :3].T + T[:3, 3]


def draw_pose(image, rvec, tvec, K, dist=None, axis_len=100.0):
    vis = image.copy()
    cv2.drawFrameAxes(vis, K, np.zeros(5) if dist is None else dist, rvec, tvec, axis_len, 4)
    return vis
