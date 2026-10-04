# SPDX-License-Identifier: Apache-2.0
import cv2
import numpy as np

def analyze_chroma(skin_rgb):
    """
    Input : [R, G, B]
    Output: Chroma score + Category (Soft / Medium / Bright)
    """
    r, g, b = skin_rgb

    # 1. Convert RGB (0-255) to Float (0.0-1.0)
    rgb_normalized = np.array([[[r / 255.0, g / 255.0, b / 255.0]]], dtype=np.float32)

    # 2. Convert to Standard CIE Lab (Exact Floating Point)
    lab = cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2Lab)[0][0]

    # In float32 cv2.COLOR_RGB2Lab:
    # L is in [0, 100], 'a' and 'b' are in standard range (approx -128 to +127)
    a = float(lab[1])
    b = float(lab[2])

    # Calculate Chroma directly: sqrt(a^2 + b^2)
    chroma = np.sqrt(a**2 + b**2)

    # 3. New thresholds aligned with Seasonal Color Analysis
    if chroma < 18.0:
        chroma_type = "Soft"      # Soft Summer (10-18), Soft Autumn (12-20)
    elif chroma <= 27.0:
        chroma_type = "Medium"    # True Summer (18-24), True Autumn (20-28)
    else:
        chroma_type = "Bright"    # Spring (28-40), Bright Winter (30-45)

    return {
        "score": round(chroma, 2),
        "chroma": chroma_type
    }
