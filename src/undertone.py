# SPDX-License-Identifier: Apache-2.0
import cv2
import numpy as np

def analyze_undertone(skin_rgb):
    """
    Input: skin_rgb = [R, G, B] (0-255)
    Output: Undertone category (Warm / Cool / Neutral) + Metrics
    """
    r, g, b = skin_rgb

    # 1. Convert to Standard Float32 CIE LAB (Eliminates OpenCV uint8 precision errors)
    rgb_normalized = np.array([[[r / 255.0, g / 255.0, b / 255.0]]], dtype=np.float32)
    lab = cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2Lab)[0][0]

    L = float(lab[0])    # 0 to 100
    a = float(lab[1])    # Redness (+a) vs Greenness (-a)
    b_lab = float(lab[2])# Yellowness (+b) vs Blueness (-b)

    # 2. Calculate the Lab chromatic hue angle (0° to 360°).
    # This is a descriptive feature used by this project; it is not a
    # standardized clinical definition of warm/cool undertone.
    hue_rad = np.arctan2(b_lab, a)
    hue_deg = np.degrees(hue_rad) % 360

    # 3. Individual Typology Angle (ITA) - Measures skin lightness vs yellowness
    ita = np.degrees(np.arctan2(L - 50, b_lab))

    # 4. Ratio of yellowness to redness (b / a).
    # This is another heuristic feature used by the project.
    # It should not be interpreted as a universal color-analysis standard.
    ba_ratio = b_lab / (a + 1e-6)

    # ---------- Calibrated Classification ----------
    # Cool skin: High Redness relative to Yellowness (Hue < 55° or b/a ratio < 1.15)
    # Warm skin: High Yellowness relative to Redness (Hue > 64° or b/a ratio > 1.45)
    # Neutral skin: Balanced Redness and Yellowness (55° <= Hue <= 64°)

    if hue_deg < 55.0 or ba_ratio < 1.15:
        if b_lab > 17.0 and hue_deg > 52.0: 
            undertone = "Neutral"  # Catch edge-case warm undertones with high yellowness
        else:
            undertone = "Cool"

    elif hue_deg > 64.0 or ba_ratio > 1.48:
        if a > 20.0 and hue_deg < 66.0:
            undertone = "Neutral"  # Catch edge-case ruddy/flushed warm skins
        else:
            undertone = "Warm"

    else:
        # Fine-tuning based on yellowness (b_lab) for the middle range
        if b_lab < 12.5:
            undertone = "Cool"
        elif b_lab > 17.5:
            undertone = "Warm"
        else:
            undertone = "Neutral"

    return {
        "undertone": undertone,
        "hue_angle": round(hue_deg, 2),
        "ba_ratio": round(ba_ratio, 2),
        "L": round(L, 2),
        "a": round(a, 2),
        "b": round(b_lab, 2),
        "ita": round(ita, 2)
    }
