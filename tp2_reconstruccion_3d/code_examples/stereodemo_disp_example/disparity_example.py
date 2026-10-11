import cv2
import json
import numpy as np
from pathlib import Path

from disparity.methods import Calibration, InputPair, Config
from disparity.method_cre_stereo import CREStereo

# from diparity.method_cre_stereo import CREStereo
# from stereodemo.method_opencv_bm import StereoBM, StereoSGBM
# from stereodemo.methods import Calibration, InputPair, Config

models_path = Path.home() / ".cache" / "stereodemo" / "models"


calibration = Calibration(**{
    "width": 1920,
    "height": 1080,
    "baseline_meters": 60 / 1000,
    "fx": 600,
    "fy": 600,
    "cx0": 960,
    "cx1": 960,
    "cy": 536,
    "depth_range": [0.05, 20.0],
    "left_image_rect_normalized": [0, 0, 1, 1]
})


left_image = cv2.imread("images/cat_left_rect.jpg")
right_image = cv2.imread("images/cat_right_rect.jpg")

pair = InputPair(left_image, right_image, calibration, "status?")
config = Config(models_path=models_path)

method = CREStereo(config)
# method = StereoBM(config)
# medhod = StereoSGBM(config)

method.parameters["Shape"].set_value("1280x720")
method.parameters["Iterations"].set_value("10")

print("computing disparity")
disparity = method.compute_disparity(pair)


# np.savez("disparity.npz", disparity.disparity_pixels)
# cv2.imwrite("disparity.png", disparity.disparity_pixels)

dvis = disparity.disparity_pixels.copy()
dvis = 255 * (dvis - dvis.min()) / (dvis.max() - dvis.min())
dvis = dvis.astype('uint8')
cv2.imshow("disparity", dvis)

input = np.hstack((left_image, right_image))
cv2.imshow("input", input)
cv2.waitKey()