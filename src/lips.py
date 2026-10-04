# SPDX-License-Identifier: Apache-2.0
import cv2
import numpy as np
from sklearn.cluster import KMeans

# نقاط اصلی لب بالا و پایین
LIP_POINTS = [
    61, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 291,
    78, 95, 88, 178, 87,
    14, 317, 402, 318, 324, 308
]


def analyze_lips(img, face):

    h, w = img.shape[:2]

    pixels = []

    for idx in LIP_POINTS:

        lm = face.landmark[idx]

        x = int(lm.x * w)
        y = int(lm.y * h)

        patch = img[
            max(0, y - 3):min(h, y + 3),
            max(0, x - 3):min(w, x + 3)
        ]

        if patch.size > 0:
            pixels.append(patch.reshape(-1, 3))

    if not pixels:
        return [150, 90, 90]

    pixels = np.concatenate(pixels, axis=0)

    hsv = cv2.cvtColor(
        pixels.reshape(-1, 1, 3),
        cv2.COLOR_BGR2HSV
    ).reshape(-1, 3)

    # حذف پیکسل‌های خیلی تیره و خیلی روشن
    mask = (
        (hsv[:, 2] > 40) &
        (hsv[:, 2] < 245) &
        (hsv[:, 1] > 25)
    )

    pixels = pixels[mask]

    if len(pixels) < 20:
        return [150, 90, 90]

    # KMeans needs at least two distinct samples. If the region is nearly
    # uniform, use its median color instead of failing the whole image.
    if len(pixels) < 2 or np.unique(pixels, axis=0).shape[0] < 2:
        b, g, r = np.median(pixels, axis=0).astype(int)
        return [int(r), int(g), int(b)]

    km = KMeans(
        n_clusters=2,
        random_state=0,
        n_init=10
    )

    km.fit(pixels)

    counts = np.bincount(km.labels_)
    idx = np.argmax(counts)

    b, g, r = km.cluster_centers_[idx].astype(int)

    return [r, g, b]