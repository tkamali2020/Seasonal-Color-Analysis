# SPDX-License-Identifier: Apache-2.0
import csv
import json
import os

# Reduce verbose native-library logs before importing MediaPipe/OpenCV.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

import cv2

from skin import analyze_skin
from eyes import analyze_eyes
from eyebrows import analyze_eyebrows
from lips import analyze_lips
from undertone import analyze_undertone
from skin_depth import analyze_skin_depth
from contrast import analyze_contrast
from value import analyze_value
from chroma import analyze_chroma
from season_classifier import classify_season
from recommendations import get_recommendations


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "images")
RESULTS_FOLDER = os.path.join(PROJECT_ROOT, "results")
CSV_FILE = os.path.join(RESULTS_FOLDER, "results.csv")

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif")


def resize_for_analysis(img, max_size=640):
    """Resize an image without changing its aspect ratio."""
    height, width = img.shape[:2]
    if height <= max_size and width <= max_size:
        return img

    scale = max_size / max(height, width)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)


def process_image(image_path):
    """Run the complete analysis pipeline for one image."""
    # MediaPipe is imported only when the full image pipeline is executed.
    # This keeps lightweight utilities and tests usable without MediaPipe installed.
    from detector import detect_face

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    img = resize_for_analysis(img)
    face = detect_face(img)

    skin_rgb = analyze_skin(img, face)
    value = analyze_value(skin_rgb)
    chroma = analyze_chroma(skin_rgb)
    undertone = analyze_undertone(skin_rgb)
    skin_depth = analyze_skin_depth(skin_rgb)

    eye_rgb, eye_color = analyze_eyes(img, face)
    eyebrow_rgb = analyze_eyebrows(img, face)
    lip_rgb = analyze_lips(img, face)

    contrast = analyze_contrast(skin_rgb, eyebrow_rgb, eye_rgb)

    classification = classify_season(
        undertone=undertone["undertone"],
        skin_depth=skin_depth,
        contrast=contrast["contrast"],
        contrast_score=contrast["score"],
        value=value["value"],
        chroma_score=chroma["score"],
        eye_color=eye_color,
        return_details=True,
    )

    return {
        "skin_rgb": skin_rgb,
        "undertone": undertone,
        "skin_depth": skin_depth,
        "value": value,
        "chroma": chroma,
        "eye_rgb": eye_rgb,
        "eye_color": eye_color,
        "eyebrow_rgb": eyebrow_rgb,
        "lip_rgb": lip_rgb,
        "contrast": contrast,
        "classification": classification,
        "recommendations": get_recommendations(classification["season"]),
    }


def write_csv_header(writer):
    writer.writerow([
        "Image",
        "Skin RGB",
        "Undertone",
        "Undertone Hue",
        "b/a Ratio",
        "ITA",
        "Skin Depth",
        "Value",
        "L*",
        "Chroma",
        "Chroma Score",
        "Eye RGB",
        "Eye Color",
        "Eyebrow RGB",
        "Lip RGB",
        "Contrast",
        "Contrast Score",
        "Brightness Score",
        "Color Score",
        "Season",
        "Season Score",
        "Top 3 Candidates",
        "Score Breakdown",
    ])


def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    if not os.path.isdir(IMAGE_FOLDER):
        raise FileNotFoundError(f"Image folder not found: {IMAGE_FOLDER}")

    image_files = sorted(
        name for name in os.listdir(IMAGE_FOLDER)
        if name.lower().endswith(VALID_EXTENSIONS)
    )

    if not image_files:
        print(f"No supported images found in: {IMAGE_FOLDER}")
        return

    failed_images = []

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        write_csv_header(writer)

        for filename in image_files:
            print("\n" + "=" * 60)
            print(f"Processing: {filename}")

            try:
                result = process_image(os.path.join(IMAGE_FOLDER, filename))
                c = result["classification"]
                contrast = result["contrast"]
                undertone = result["undertone"]
                value = result["value"]
                chroma = result["chroma"]

                writer.writerow([
                    filename,
                    result["skin_rgb"],
                    undertone["undertone"],
                    undertone["hue_angle"],
                    undertone["ba_ratio"],
                    undertone["ita"],
                    result["skin_depth"],
                    value["value"],
                    value["L"],
                    chroma["chroma"],
                    chroma["score"],
                    result["eye_rgb"],
                    result["eye_color"],
                    result["eyebrow_rgb"],
                    result["lip_rgb"],
                    contrast["contrast"],
                    contrast["score"],
                    contrast["brightness_score"],
                    contrast["color_score"],
                    c["season"],
                    c["score"],
                    "; ".join(f"{name} ({score})" for name, score in c["top_candidates"]),
                    json.dumps(c["score_breakdown"]["contributions"], ensure_ascii=False, sort_keys=True),
                ])
                f.flush()

                print(f"Undertone: {undertone['undertone']}")
                print(f"Skin depth: {result['skin_depth']}")
                print(f"Contrast: {contrast['contrast']} ({contrast['score']:.2f})")
                print(f"Season: {c['season']} ({c['score']} pts)")
                print("Top candidates:", c["top_candidates"])

            except Exception as exc:
                failed_images.append(filename)
                print(f"Error during processing {filename}: {exc}")

    print("\n" + "=" * 60)
    if failed_images:
        print(f"Processing completed with {len(failed_images)} failure(s).")
        print("Failed images:", ", ".join(failed_images))
        print(f"Partial results saved to: {CSV_FILE}")
        return 1

    print("Processing completed successfully.")
    print(f"Results saved to: {CSV_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
