# SPDX-License-Identifier: Apache-2.0
import cv2
import numpy as np


def analyze_skin_depth(skin_rgb):
    """Estimate coarse skin depth from CIE L*.

    The project uses three broad depth categories: Light, Medium, and Deep.
    L* is used here so the depth estimate is expressed in the same perceptual
    lightness space used by ``analyze_value``.

    These thresholds are project calibration choices, not universal
    dermatological or color-analysis standards.
    """
    r, g, b = skin_rgb
    rgb_normalized = np.array(
        [[[r / 255.0, g / 255.0, b / 255.0]]],
        dtype=np.float32,
    )
    lab = cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2Lab)[0][0]
    L = float(lab[0])

    if L >= 63.0:
        return "Light"
    if L >= 53.0:
        return "Medium"
    return "Deep"
