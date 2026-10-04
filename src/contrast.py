# SPDX-License-Identifier: Apache-2.0
import cv2
import numpy as np


def rgb_to_lab_float(rgb):
    """
    Convert RGB (0-255) to Standard CIE LAB float.
    L range: [0, 100], a range: [-128, 127], b range: [-128, 127]
    """
    r, g, b = rgb
    rgb_normalized = np.array([[[r / 255.0, g / 255.0, b / 255.0]]], dtype=np.float32)
    lab = cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2Lab)[0][0]
    return lab


def analyze_contrast(skin_rgb, eyebrow_rgb, eye_rgb):
    """
    Facial contrast analysis calibrated for Seasonal Color Analysis.
    """
    skin = rgb_to_lab_float(skin_rgb)
    brow = rgb_to_lab_float(eyebrow_rgb)
    eye = rgb_to_lab_float(eye_rgb)

    # 1. Delta L (Brightness difference in 0-100 scale)
    deltaL_brow = abs(skin[0] - brow[0])
    deltaL_eye = abs(skin[0] - eye[0])

    # 2. CIE76 Delta E (Euclidean distance in Lab space)
    deltaE_brow = np.linalg.norm(skin - brow)
    deltaE_eye = np.linalg.norm(skin - eye)

    # 3. Primary Score: Brightness Contrast (85% weight)
    # Eyebrows are given higher weight (65%) as they define face frame better
    brightness_score = (deltaL_brow * 0.65) + (deltaL_eye * 0.35)

    # 4. Secondary Score: Color Difference (15% weight)
    color_score = (deltaE_brow * 0.65) + (deltaE_eye * 0.35)

    # 5. Final Contrast Score (Mainly driven by Lightness delta)
    contrast_score = (brightness_score * 0.85) + (color_score * 0.15)

    # 6. Calibrated Thresholds for Realistic Distribution
    # Low Contrast: Soft Summer, Light Spring, Muted Autumn (Score < 22)
    # Medium Contrast: True Summer, Soft Autumn, Deep Autumn (22 <= Score <= 36)
    # High Contrast: Winter Seasons, Bright Spring (Score > 36)

    if contrast_score < 22.0:
        contrast = "Low"
    elif contrast_score <= 36.0:
        contrast = "Medium"
    else:
        contrast = "High"

    return {
        "score": round(float(contrast_score), 2),
        "brightness_score": round(float(brightness_score), 2),
        "color_score": round(float(color_score), 2),
        "deltaL_brow": round(float(deltaL_brow), 2),
        "deltaL_eye": round(float(deltaL_eye), 2),
        "contrast": contrast
    }
