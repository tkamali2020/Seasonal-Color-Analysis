# SPDX-License-Identifier: Apache-2.0
import cv2
import numpy as np

# نقاط گونه، پیشانی و چانه (MediaPipe FaceMesh)
SKIN_POINTS = [
    50, 280,      # گونه
    117, 346,     # گونه پایین
    118, 347,
    10,           # پیشانی
    152           # چانه
]


def analyze_skin(img, face):

    h, w, _ = img.shape

    pixels = []

    PATCH_SIZE = 16   # 16x16

    for idx in SKIN_POINTS:

        lm = face.landmark[idx]

        x = int(lm.x * w)
        y = int(lm.y * h)

        patch = img[
            max(0, y - PATCH_SIZE // 2):min(h, y + PATCH_SIZE // 2),
            max(0, x - PATCH_SIZE // 2):min(w, x + PATCH_SIZE // 2)
        ]

        if patch.size == 0:
            continue

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        # حذف سایه، هایلایت و پیکسل‌های کم‌رنگ
        mask = (
            (V > 45) &
            (V < 245) &
            (S > 20) &
            (S < 180)
        )

        good = patch[mask]

        if len(good) > 0:
            pixels.append(good)

    if len(pixels) == 0:
        raise Exception("No valid skin pixels found.")

    pixels = np.concatenate(pixels, axis=0)

    # حذف Outlierها با Percentile
    b = pixels[:, 0]
    g = pixels[:, 1]
    r = pixels[:, 2]

    def remove_outlier(channel):
        low = np.percentile(channel, 5)
        high = np.percentile(channel, 95)
        return (channel >= low) & (channel <= high)

    mask = (
        remove_outlier(r) &
        remove_outlier(g) &
        remove_outlier(b)
    )

    pixels = pixels[mask]

    if len(pixels) == 0:
        raise ValueError("No valid skin pixels remain after outlier filtering.")

    b = int(np.median(pixels[:, 0]))
    g = int(np.median(pixels[:, 1]))
    r = int(np.median(pixels[:, 2]))

    return [r, g, b]