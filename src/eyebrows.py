# SPDX-License-Identifier: Apache-2.0
import cv2
import numpy as np
from sklearn.cluster import KMeans

LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [336, 296, 334, 293, 300]


def analyze_eyebrows(img, face):

    h, w = img.shape[:2]

    pixels = []

    for point in LEFT_EYEBROW + RIGHT_EYEBROW:

        lm = face.landmark[point]

        x = int(lm.x * w)
        y = int(lm.y * h)

        patch = img[
            max(0, y - 4):min(h, y + 4),
            max(0, x - 4):min(w, x + 4)
        ]

        if patch.size > 0:
            pixels.append(patch.reshape(-1, 3))

    if not pixels:
        return [60, 50, 40]

    pixels = np.concatenate(pixels, axis=0)

    hsv = cv2.cvtColor(
        pixels.reshape(-1, 1, 3),
        cv2.COLOR_BGR2HSV
    ).reshape(-1, 3)

    # حذف پوست‌های خیلی روشن
    mask = (
        (hsv[:, 2] < 170) &
        (hsv[:, 1] > 20)
    )

    pixels = pixels[mask]

    if len(pixels) < 20:
        return [60, 50, 40]

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