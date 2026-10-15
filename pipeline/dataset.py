"""
Lectura de datasets estéreo con la estructura de la cátedra:

    <dataset>/
        calib/      pares *left*_N.jpg / *right*_N.jpg + checkerboard.txt
        captures/   pares *left*_N.jpg / *right*_N.jpg + markers.yaml
"""
import ast
import glob
import os
import re

import yaml


def numeric_sort(file_name):
    # mismo criterio que en el tutorial 6: ordena por el número final del nombre
    return int(file_name.split("_")[-1].split(".")[0])


def list_stereo_pairs(directory, ext="jpg"):
    """Devuelve una lista de tuplas (idx, left_path, right_path) ordenada por idx."""
    left_files = sorted(
        glob.glob(os.path.join(directory, f"*left*.{ext}")),
        key=numeric_sort,
    )
    pairs = []
    for left in left_files:
        right = os.path.join(
            os.path.dirname(left),
            os.path.basename(left).replace("left", "right"),
        )
        if not os.path.exists(right):
            print(f"warning: falta el par derecho de {left}")
            continue
        pairs.append((numeric_sort(left), left, right))
    return pairs


def read_checkerboard_txt(path):
    """Parsea el checkerboard.txt de la cátedra (sintaxis python)."""
    values = {}
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if "=" not in line:
                continue
            key, value = [s.strip() for s in line.split("=", 1)]
            values[key] = ast.literal_eval(value)
    return tuple(values["checkerboard"]), float(values["square_size_mm"])


def read_markers_yaml(path):
    """
    Lee markers.yaml. Devuelve un dict normalizado:
      charuco:      {type, dictionary, squares_x, squares_y, square_length, marker_length}
      checkerboard: {type, pattern_size, square_size}
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    marker_type = raw["MARKER_TYPE"].lower()
    if marker_type == "charuco":
        return {
            "type": "charuco",
            "dictionary": "DICT_" + raw["DICTIONARY_TYPE"],
            "squares_x": int(raw["SQUARES_X"]),
            "squares_y": int(raw["SQUARES_Y"]),
            "square_length": float(raw["SQUARE_LENGTH"]),
            "marker_length": float(raw["MARKER_LENGTH"]),
        }
    if marker_type == "checkerboard":
        # PATTERN_SIZE viene como string "(10, 7)"
        pattern = raw["PATTERN_SIZE"]
        if isinstance(pattern, str):
            pattern = tuple(int(n) for n in re.findall(r"\d+", pattern))
        return {
            "type": "checkerboard",
            "pattern_size": tuple(pattern),
            "square_size": float(raw["SQUARE_SIZE_MM"]),
        }
    raise ValueError(f"MARKER_TYPE desconocido: {raw['MARKER_TYPE']}")
