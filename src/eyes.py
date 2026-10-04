# SPDX-License-Identifier: Apache-2.0
import cv2
import numpy as np

RIGHT_IRIS = [469, 470, 471, 472]
LEFT_IRIS = [474, 475, 476, 477]


def iris_center(face, iris_points, w, h):
    pts = np.array([
        (
            int(face.landmark[i].x * w),
            int(face.landmark[i].y * h)
        )
        for i in iris_points
    ], dtype=np.int32)

    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))

    return (cx, cy), pts


def sample_patch(img, x, y, size=2):
    h, w = img.shape[:2]

    x1 = max(x - size, 0)
    x2 = min(x + size + 1, w)
    y1 = max(y - size, 0)
    y2 = min(y + size + 1, h)

    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return None

    pixels = patch.reshape(-1, 3)

    hsv = cv2.cvtColor(
        pixels.reshape(-1, 1, 3).astype(np.uint8),
        cv2.COLOR_BGR2HSV
    ).reshape(-1, 3)

    S = hsv[:, 1]
    V = hsv[:, 2]

    # حذف بازتاب نور فلش (Glint/Reflection)
    is_glint = (V > 210) & (S < 35)
    
    valid = (
        (V > 30) &       # حذف سیاهی مردمک
        (V < 235) &      # حذف روشنی‌های بسیار شدید
        (~is_glint)      # حذف انعکاس نور فلش
    )

    pixels = pixels[valid]
    if len(pixels) == 0:
        return None

    return np.median(pixels, axis=0)


def extract_eye_hsv(img, face, iris_points):
    h, w = img.shape[:2]

    center, pts = iris_center(face, iris_points, w, h)
    cx, cy = center

    hues, sats, vals, rgbs = [], [], [], []

    for px, py in pts:
        for t in [0.35, 0.55, 0.70]:
            sx = int(px + t * (cx - px))
            sy = int(py + t * (cy - py))

            patch = sample_patch(img, sx, sy, size=2)
            if patch is None:
                continue

            b, g, r = patch.astype(np.uint8)

            hsv = cv2.cvtColor(
                np.uint8([[[b, g, r]]]),
                cv2.COLOR_BGR2HSV
            )[0][0]

            rgbs.append([r, g, b])
            hues.append(hsv[0] * 2)  # تبدیل HSV به 0-360
            sats.append(hsv[1])
            vals.append(hsv[2])

    if len(hues) == 0:
        return None

    vals_arr = np.array(vals)
    bright_indices = np.where(vals_arr >= np.median(vals_arr))[0]

    return (
        np.median(np.array(hues)[bright_indices]),
        np.median(np.array(sats)[bright_indices]),
        np.median(vals_arr[bright_indices]),
        np.median(np.array(rgbs)[bright_indices], axis=0).astype(int)
    )

def classify_eye_color(h, s, v):
    """
    منطق جدید و موازنه‌شده برای تشخیص چشم بدون خراب کردن سایر رنگ‌ها
    """
    # -------------------------------------------------------------
    # ۱. چشم‌های آبی / طوسی (Blue / Gray)
    # -------------------------------------------------------------
    # محدوده اصلی Hue برای آبی بین 150 تا 260 است.
    if 150 <= h <= 260:
        # چشم آبی حتی اگر بی‌پیدا/طوسی باشد (S کم) باز هم آبی است!
        # فقط اگر V (روشنایی) خیلی خیلی پایین باشد (تاری مطلق)، قهوه‌ای فرض می‌شود.
        if v > 30:
            return "blue"

    # -------------------------------------------------------------
    # ۲. چشم‌های سبز (Green)
    # -------------------------------------------------------------
    # سبز واقعی Hue بین 70 تا 149 دارد.
    if 70 <= h < 150:
        if s > 20:
            return "green"
        else:
            return "blue" # چشم‌های طوسی-آبی روشن گاهی در این محدوده با S پایین می‌افتند

    # -------------------------------------------------------------
    # ۳. چشم‌های قهوه‌ای و عسلی (Brown / Hazel)
    # -------------------------------------------------------------
    # چشم‌های گرم همیشه Hue زیر 45 دارند.
    if h < 45 or h > 330:  # Hue در دایره رنگی دور می‌زند (قرمز/قهوه‌ای)
        if s < 25 and v > 120:
            # عسلی روشن تحت نور
            return "hazel"
        elif h >= 25 and s >= 25:
            return "hazel"
        else:
            return "brown"

    # -------------------------------------------------------------
    # ۴. محدوده مرزی (45 تا 70 - زرد/زیتونی)
    # -------------------------------------------------------------
    if 45 <= h < 70:
        if s > 40:
            return "green"
        else:
            return "hazel"

    # حالت پیش‌فرض مطمئن
    return "brown"


# -------------------------------------------------------------
# 🎯 همان تابعی که main.py به آن نیاز دارد
# -------------------------------------------------------------
def analyze_eyes(img, face):
    right = extract_eye_hsv(img, face, RIGHT_IRIS)
    left = extract_eye_hsv(img, face, LEFT_IRIS)

    if right is None and left is None:
        return [60, 60, 60], "brown"

    if right is None:
        h, s, v, rgb = left
    elif left is None:
        h, s, v, rgb = right
    else:
        h = (right[0] + left[0]) / 2
        s = (right[1] + left[1]) / 2
        v = (right[2] + left[2]) / 2
        rgb = ((right[3] + left[3]) / 2).astype(int)

    eye_color = classify_eye_color(h, s, v)

    return rgb.tolist(), eye_color