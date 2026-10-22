import contextlib
import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Parámetros por defecto de SGBM.
# Con f≈600px y baseline≈60mm: d = f*B/Z  ->  Z=250mm: d≈145, Z=800mm: d≈45.
DEFAULT_SGBM = {
    "minDisparity": 0,
    "numDisparities": 192,     # múltiplo de 16
    "blockSize": 5,
    "uniquenessRatio": 10,
    "speckleWindowSize": 200,
    "speckleRange": 2,
    "disp12MaxDiff": 1,
    "preFilterCap": 63,
    "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
}


def compute_disparity(left_rect_gray, right_rect_gray, params=None):
    """Devuelve la disparidad en píxeles (float32). Valores <= minDisparity son inválidos."""
    p = dict(DEFAULT_SGBM)
    if params:
        p.update(params)

    # P1/P2 según la recomendación de OpenCV: 8*canales*bs^2 y 32*canales*bs^2
    bs = p["blockSize"]
    p.setdefault("P1", 8 * bs * bs)
    p.setdefault("P2", 32 * bs * bs)

    sgbm = cv2.StereoSGBM_create(**p)
    # SGBM devuelve disparidad en punto fijo (x16)
    disparity = sgbm.compute(left_rect_gray, right_rect_gray).astype(np.float32) / 16.0
    return disparity


_CRE_METHOD = None
CATEDRA_DISPARITY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tp2_reconstruccion_3d", "code_examples", "stereodemo_disp_example",
)


def compute_disparity_cre(left_rect_bgr, right_rect_bgr, models_dir, shape="1280x720", iterations="10"):
    """
    Disparidad con CREStereo (red neuronal, CVPR 2022) usando el módulo `disparity`. 
    La red trabaja a `shape` y el módulo reescala la disparidad a la resolución original.
    El modelo .onnx tiene que estar en `models_dir` (ver README).
    """
    global _CRE_METHOD
    if _CRE_METHOD is None:
        if CATEDRA_DISPARITY_DIR not in sys.path:
            sys.path.insert(0, CATEDRA_DISPARITY_DIR)
        from disparity.method_cre_stereo import CREStereo
        from disparity.methods import Config

        _CRE_METHOD = CREStereo(Config(models_path=Path(models_dir)))
        _CRE_METHOD.parameters["Shape"].set_value(shape)
        _CRE_METHOD.parameters["Iterations"].set_value(str(iterations))

    from disparity.methods import InputPair

    # la calibración del InputPair no se usa para calcular la disparidad
    with contextlib.redirect_stdout(io.StringIO()):
        out = _CRE_METHOD.compute_disparity(InputPair(left_rect_bgr, right_rect_bgr, None, None))
    return out.disparity_pixels.astype(np.float32)


def disparity_to_points(disparity, Q, left_rect_bgr, min_disparity=1.0, z_range=(50, 2000)):
    """
    Usa la matriz Q de stereoRectify para pasar de (u, v, d) a (X, Y, Z).
    La nube queda en el sistema de la cámara IZQUIERDA RECTIFICADA, en mm.
    Devuelve (points Nx3, colors Nx3 RGB en [0,1]).
    """
    points_3d = cv2.reprojectImageTo3D(disparity, Q)

    Z = points_3d[:, :, 2]
    valid = (disparity > min_disparity) & np.isfinite(Z) & (Z > z_range[0]) & (Z < z_range[1])

    points = points_3d[valid].astype(np.float64)
    colors = left_rect_bgr[valid][:, ::-1].astype(np.float64) / 255.0  # BGR -> RGB
    return points, colors


def board_disparity_correction(disparity, Q, object_points, image_points, rvec, tvec,
                               min_points=20, max_scale_change=0.2):
    """
    Compara la disparidad medida con la que "debería" tener el tablero según la pose de PnP.

    Para cada esquina del ChArUco conocemos su posición 3D en la cámara (con la pose), así
    que sabemos su profundidad Z y por lo tanto su disparidad esperada (d = f·B / Z, usando Q).
    Ajustamos  d_medida = a · d_esperada + b  con dos pasadas de mínimos cuadrados (la
    segunda sin los puntos que quedan muy lejos de la recta).

    Si la cámara mira el tablero casi de frente, todas las esquinas tienen disparidades
    parecidas y la recta queda mal definida (escala y corrimiento se confunden). En ese caso
    (escala que cambia más de `max_scale_change`) corregimos solo el corrimiento: a = 1.

    Devuelve (a, b, puntos usados), o None si no hay suficientes esquinas.
    """
    R, _ = cv2.Rodrigues(rvec)
    corners_cam = object_points @ R.T + tvec.reshape(3)
    expected = (Q[2, 3] / corners_cam[:, 2] - Q[3, 3]) / Q[3, 2]

    u = image_points[:, 0].reshape(-1, 1).astype(np.float32)
    v = image_points[:, 1].reshape(-1, 1).astype(np.float32)
    measured = cv2.remap(disparity.astype(np.float32), u, v, cv2.INTER_LINEAR).ravel()

    valid = measured > 1
    for scale_free in (True, False):
        use = valid.copy()
        for _ in range(2):
            if use.sum() < min_points:
                return None
            if scale_free:
                a, b = np.polyfit(expected[use], measured[use], 1)
            else:
                a, b = 1.0, np.median(measured[use] - expected[use])
            residual = measured - (a * expected + b)
            spread = 1.4826 * np.median(np.abs(residual[use])) + 1e-6
            use &= np.abs(residual) < 3 * spread
        if use.sum() >= min_points and abs(a - 1) <= max_scale_change:
            return float(a), float(b), int(use.sum())
    return None


def apply_disparity_correction(disparity, a, b):
    """Corrige la disparidad con el ajuste de `board_disparity_correction`. Lo inválido sigue inválido."""
    corrected = (disparity - b) / a
    corrected[disparity <= 0] = 0
    return corrected.astype(np.float32)


def disparity_to_color(disparity, num_disparities=None):
    """Imagen BGR para visualizar la disparidad."""
    d = disparity.copy()
    d[d < 0] = 0
    vmax = num_disparities or max(d.max(), 1)
    d8 = np.clip(d / vmax * 255, 0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
    vis[disparity <= 0] = 0
    return vis
