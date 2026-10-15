"""
Paso 4.2 (1 y 2): disparidad sobre el par rectificado y nube de puntos local.
"""
import contextlib
import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Parámetros por defecto de SGBM (partimos de los del tutorial 6 y los ajustamos).
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
    Disparidad con CREStereo (red neuronal, CVPR 2022) usando el módulo `disparity`
    de la cátedra (tutorial 6 / code_examples). La red trabaja a `shape` y el
    módulo reescala la disparidad a la resolución original.
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


def disparity_to_color(disparity, num_disparities=None):
    """Imagen BGR para visualizar la disparidad."""
    d = disparity.copy()
    d[d < 0] = 0
    vmax = num_disparities or max(d.max(), 1)
    d8 = np.clip(d / vmax * 255, 0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
    vis[disparity <= 0] = 0
    return vis
