import pickle

import cv2
import numpy as np

import calib  # módulo de la cátedra (tutorial 6)
from dataset import list_stereo_pairs


def detect_calibration_corners(calib_dir, checkerboard, square_size_mm):
    """
    Busca el checkerboard en cada par (findChessboardCorners + cornerSubPix, vía
    calib.detect_board) y descarta los pares donde no aparece en ambas imágenes.
    """
    # puntos 3D del tablero (Z=0) en mm
    world = calib.board_points(checkerboard) * square_size_mm

    image_size = None
    world_points, left_points, right_points, used = [], [], [], []

    for idx, left_file, right_file in list_stereo_pairs(calib_dir):
        left = cv2.imread(left_file, cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(right_file, cv2.IMREAD_GRAYSCALE)
        size = (left.shape[1], left.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            raise ValueError(f"tamaño distinto en {left_file}: {size} vs {image_size}")

        ok_l, c_l = calib.detect_board(checkerboard, left)
        ok_r, c_r = calib.detect_board(checkerboard, right)
        if not (ok_l and ok_r):
            print(f"  par {idx}: checkerboard no encontrado, se descarta")
            continue

        world_points.append(world)
        left_points.append(c_l)
        right_points.append(c_r)
        used.append(idx)

    print(f"  checkerboard detectado en {len(used)} pares")
    return world_points, left_points, right_points, image_size, used


def stereo_calibrate(world_points, left_points, right_points, image_size, mono_init=True):
    """
    mono_init=True: calibra primero cada cámara por separado con calibrateCamera 
    y usa eso como inicialización de stereoCalibrate. mono_init=False: (stereoCalibrate desde cero, flags=0).
    """
    if mono_init:
        rms_l, K1, D1, _, _ = cv2.calibrateCamera(world_points, left_points, image_size, None, None)
        rms_r, K2, D2, _, _ = cv2.calibrateCamera(world_points, right_points, image_size, None, None)
        print(f"  RMS monocular: left={rms_l:.3f}px  right={rms_r:.3f}px")
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
    else:
        rms_l = rms_r = None
        K1 = D1 = K2 = D2 = None
        flags = 0

    # stereoCalibrateExtended (tutorial 6) además devuelve el error por vista
    rms, K1, D1, K2, D2, R, T, E, F, _, _, per_view = cv2.stereoCalibrateExtended(
        world_points, left_points, right_points,
        K1, D1, K2, D2, image_size, None, None,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    )
    print(f"  RMS estéreo: {rms:.3f}px")

    calibration = {
        "left_K": K1, "left_dist": D1,
        "right_K": K2, "right_dist": D2,
        "R": R, "T": T, "E": E, "F": F,
        "image_size": image_size,
        # extras para el informe
        "rms_stereo": rms, "rms_left": rms_l, "rms_right": rms_r,
        "per_view_errors": per_view,
    }
    return calibration


def stereo_rectify(calibration, alpha=0):
    """stereoRectify + initUndistortRectifyMap."""
    K1, D1 = calibration["left_K"], calibration["left_dist"]
    K2, D2 = calibration["right_K"], calibration["right_dist"]
    size = tuple(calibration["image_size"])

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, size, calibration["R"], calibration["T"], alpha=alpha
    )
    lx, ly = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_32FC1)
    rx, ry = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_32FC1)

    return {
        "left_map_x": lx, "left_map_y": ly,
        "right_map_x": rx, "right_map_y": ry,
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
        "validRoi1": roi1, "validRoi2": roi2,
    }


def rectify_pair(left, right, maps):
    left_r = cv2.remap(left, maps["left_map_x"], maps["left_map_y"], cv2.INTER_LINEAR)
    right_r = cv2.remap(right, maps["right_map_x"], maps["right_map_y"], cv2.INTER_LINEAR)
    return left_r, right_r


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def print_calibration(calibration):
    # mismo formato de impresión que el tutorial 6 (calib.np_print)
    for name in ["left_K", "left_dist", "right_K", "right_dist", "R", "T"]:
        print(f"  {name} = {calib.np_print(np.atleast_2d(calibration[name]))}")
    print(f"  baseline = {np.linalg.norm(calibration['T']):.2f} mm")
