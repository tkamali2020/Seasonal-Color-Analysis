# SPDX-License-Identifier: Apache-2.0
import cv2
import numpy as np


def analyze_value(skin_rgb):
    """Estimate perceptual lightness from CIE L*.

    Returns L* on a 0-100 scale and a coarse Light/Medium/Dark category.
    """
    r, g, b = skin_rgb
    rgb_normalized = np.array(
        [[[r / 255.0, g / 255.0, b / 255.0]]],
        dtype=np.float32,
    )
    lab = cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2Lab)[0][0]
    L = float(lab[0])

    # These thresholds correspond approximately to the former uint8-Lab
    # thresholds (160/135 on a 0-255 scale).
    if L >= 63:
        value = "Light"
    elif L >= 53:
        value = "Medium"
    else:
        value = "Dark"

    return {
        "L": round(L, 2),
        "value": value,
    }
