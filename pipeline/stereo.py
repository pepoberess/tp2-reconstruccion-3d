import contextlib
import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_SGBM = {
    "minDisparity": 0,
    "numDisparities": 192,
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

    bs = p["blockSize"]
    p.setdefault("P1", 8 * bs * bs)
    p.setdefault("P2", 32 * bs * bs)

    sgbm = cv2.StereoSGBM_create(**p)
    disparity = sgbm.compute(left_rect_gray, right_rect_gray).astype(np.float32) / 16.0
    return disparity


_CRE_METHOD = None
CATEDRA_DISPARITY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tp2_reconstruccion_3d", "code_examples", "stereodemo_disp_example",
)


def compute_disparity_cre(left_rect_bgr, right_rect_bgr, models_dir, shape="1280x720", iterations="10"):
    """Calcula la disparidad con el modelo CREStereo."""
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

    with contextlib.redirect_stdout(io.StringIO()):
        out = _CRE_METHOD.compute_disparity(InputPair(left_rect_bgr, right_rect_bgr, None, None))
    return out.disparity_pixels.astype(np.float32)


def disparity_to_points(disparity, Q, left_rect_bgr, min_disparity=1.0, z_range=(50, 2000)):
    """Convierte una disparidad en puntos 3D coloreados."""
    points_3d = cv2.reprojectImageTo3D(disparity, Q)

    Z = points_3d[:, :, 2]
    valid = (disparity > min_disparity) & np.isfinite(Z) & (Z > z_range[0]) & (Z < z_range[1])

    points = points_3d[valid].astype(np.float64)
    colors = left_rect_bgr[valid][:, ::-1].astype(np.float64) / 255.0
    return points, colors


def board_disparity_correction(disparity, Q, object_points, image_points, rvec, tvec,
                               min_points=20, max_scale_change=0.2):
    """Estima una corrección de disparidad a partir del tablero."""
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
