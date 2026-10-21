"""
Calibración estéreo a partir de imágenes de un checkerboard.

El formato del diccionario de salida es compatible con el stereo_calibration.pkl
que genera la herramienta de la cátedra (calib-stereo).
"""
import pickle
import re
from pathlib import Path

import cv2
import numpy as np


def list_stereo_pairs(folder):
    """
    Busca pares de imágenes (izquierda, derecha) en una carpeta.

    Toma todos los archivos que contienen 'left' en el nombre y busca su par
    reemplazando 'left' por 'right' (ej: calib_left_3.jpg <-> calib_right_3.jpg).

    Returns:
        lista de tuplas (left_path, right_path) ordenadas por número de imagen
    """
    folder = Path(folder)
    pairs = []
    for left in folder.glob("*left*.jpg"):
        right = left.with_name(left.name.replace("left", "right"))
        if right.exists():
            pairs.append((left, right))

    def image_number(pair):
        match = re.search(r"(\d+)", pair[0].stem)
        return int(match.group(1)) if match else -1

    return sorted(pairs, key=image_number)


def checkerboard_object_points(pattern_size, square_size):
    """
    Coordenadas 3D de las esquinas internas del checkerboard en el plano Z=0.

    Args:
        pattern_size: (puntos_por_fila, puntos_por_columna), ej (10, 7)
        square_size: lado del cuadrado (en mm)

    Returns:
        np.array (N, 3) float32
    """
    cols, rows = pattern_size
    points = np.zeros((cols * rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return points * square_size


def find_corners(image, pattern_size):
    """
    Detecta las esquinas internas del checkerboard y las refina a nivel subpixel.

    Returns:
        corners (N, 1, 2) float32, o None si no se detectó el patrón completo
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
    if not found:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def detect_stereo_corners(pairs, pattern_size):
    """
    Detecta el checkerboard en cada par. Sólo se usan los pares donde se detectó
    el patrón en ambas imágenes.

    Returns:
        dict con:
            'left_points', 'right_points': listas de corners por par válido
            'used_pairs': pares (left_path, right_path) válidos
            'rejected_pairs': pares descartados
            'image_size': (ancho, alto)
    """
    left_points, right_points = [], []
    used_pairs, rejected_pairs = [], []
    image_size = None

    for left_path, right_path in pairs:
        left_image = cv2.imread(str(left_path))
        right_image = cv2.imread(str(right_path))
        image_size = (left_image.shape[1], left_image.shape[0])

        left_corners = find_corners(left_image, pattern_size)
        right_corners = find_corners(right_image, pattern_size)

        if left_corners is None or right_corners is None:
            rejected_pairs.append((left_path, right_path))
            continue

        left_points.append(left_corners)
        right_points.append(right_corners)
        used_pairs.append((left_path, right_path))

    return {
        "left_points": left_points,
        "right_points": right_points,
        "used_pairs": used_pairs,
        "rejected_pairs": rejected_pairs,
        "image_size": image_size,
    }


def per_view_errors(object_points, image_points, K, dist, rvecs, tvecs):
    """Error de reproyección RMS (en píxeles) de cada imagen."""
    errors = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        diff = projected.reshape(-1, 2) - img.reshape(-1, 2)
        errors.append(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    return np.array(errors)


def calibrate_stereo(detection, pattern_size, square_size):
    """
    Calibra cada cámara por separado (cv.calibrateCamera) y luego estima la pose
    relativa entre ellas con cv.stereoCalibrate, dejando fijos los intrínsecos.

    Returns:
        dict con las claves del pkl de calib-stereo (left_K, left_dist, right_K,
        right_dist, R, T, E, F, image_size) más los errores de reproyección.
    """
    image_size = detection["image_size"]
    left_points = detection["left_points"]
    right_points = detection["right_points"]
    object_points = [checkerboard_object_points(pattern_size, square_size)] * len(left_points)

    # 1. calibración monocular de cada cámara
    left_rms, left_K, left_dist, left_rvecs, left_tvecs = cv2.calibrateCamera(
        object_points, left_points, image_size, None, None
    )
    right_rms, right_K, right_dist, right_rvecs, right_tvecs = cv2.calibrateCamera(
        object_points, right_points, image_size, None, None
    )

    # 2. calibración estéreo: estima R, T entre cámaras
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    stereo_rms, left_K, left_dist, right_K, right_dist, R, T, E, F = cv2.stereoCalibrate(
        object_points,
        left_points,
        right_points,
        left_K,
        left_dist,
        right_K,
        right_dist,
        image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    return {
        # mismas claves que el pkl de calib-stereo
        "left_K": left_K,
        "left_dist": left_dist,
        "right_K": right_K,
        "right_dist": right_dist,
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "image_size": image_size,
        # métricas de calidad
        "left_rms": left_rms,
        "right_rms": right_rms,
        "stereo_rms": stereo_rms,
        "left_view_errors": per_view_errors(
            object_points, left_points, left_K, left_dist, left_rvecs, left_tvecs
        ),
        "right_view_errors": per_view_errors(
            object_points, right_points, right_K, right_dist, right_rvecs, right_tvecs
        ),
    }


def save_pickle(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
