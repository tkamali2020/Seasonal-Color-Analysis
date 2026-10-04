# Seasonal Color Analysis

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

**An interpretable computer-vision pipeline for facial color analysis and rule-based 12-season classification.**

This project analyzes a facial image using facial landmarks and color measurements, then estimates a seasonal color category within the traditional 12-season personal-color framework.

> **Project status:** research/prototype implementation. The current system is rule-based and has not been trained or benchmarked as a supervised machine-learning classifier.

## What the project does

The pipeline extracts measurable facial color characteristics and combines them through an interpretable scoring system. The main outputs include:

- facial landmark detection with MediaPipe Face Mesh
- representative skin-color sampling from multiple facial regions
- HSV and CIE Lab color measurements
- skin undertone estimation
- skin-depth and perceptual-value estimates
- iris/eye-color estimation
- eyebrow and lip color extraction with guarded fallbacks for sparse/invalid samples
- facial contrast analysis
- rule-based scoring for 12 seasonal categories
- top candidate seasons and their scores
- season-specific color recommendations
- CSV export for batch processing

The 12 categories are:

`Light Spring` · `True Spring` · `Bright Spring` · `Light Summer` · `True Summer` · `Soft Summer` · `Soft Autumn` · `True Autumn` · `Deep Autumn` · `Deep Winter` · `True Winter` · `Bright Winter`

## Processing pipeline

```text
Input image
    │
    ▼
Face detection + facial landmarks
    │
    ├───────────────┬───────────────┐
    ▼               ▼               ▼
Skin analysis   Eye analysis   Brow / lip analysis
    │               │               │
    ├───────────────┴───────────────┘
    ▼
Undertone / depth / value / chroma
    │
    ▼
Facial contrast
    │
    ▼
Weighted rule-based scoring
    │
    ▼
12-season classification
    │
    ├───────────────┐
    ▼               ▼
Top candidates   Recommendations
    │
    ▼
CSV report
```

## Methodology

The main stages are:

1. **Face detection:** MediaPipe Face Mesh provides facial landmarks used to locate facial regions.
2. **Skin sampling:** multiple facial regions are sampled, with HSV-based filtering and percentile-based outlier handling used to obtain a representative skin color. Empty or fully filtered regions are handled explicitly.
3. **Undertone analysis:** facial color is summarized using CIE L*, a*, b*, hue angle, b/a ratio, and ITA-related measurements.
4. **Depth and value:** coarse categories are derived from facial color measurements using the current rule set.
5. **Eye analysis:** iris regions are sampled and summarized to obtain an approximate eye color category and RGB representation.
6. **Brow and lip analysis:** representative colors are extracted from eyebrow and lip regions.
7. **Contrast analysis:** skin, eyebrow, and eye characteristics are compared using lightness and color differences.
8. **Season classification:** the extracted characteristics are passed to a weighted, rule-based scorer for the 12 seasonal categories.
9. **Recommendations:** the selected season is mapped to example color recommendations.

For the implementation details and formulas, see [`docs/methodology.md`](docs/methodology.md).

## Important interpretation note

The classifier is **heuristic and rule-based**.

The season score is an evidence score produced by the implemented rules. It is **not** a calibrated probability, statistical confidence value, or measured prediction accuracy.

Accordingly, this repository does **not** claim a benchmark accuracy against a labeled seasonal-color dataset.

## Project structure

```text
Seasonal-Color-Analysis/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── RELEASE_CHECKLIST.md
├── docs/
│   ├── methodology.md
│   └── third_party_assets.md
├── images/
│   └── README.md
├── results/
│   └── sample_results.csv
├── src/
│   ├── main.py
│   ├── detector.py
│   ├── skin.py
│   ├── eyes.py
│   ├── eyebrows.py
│   ├── lips.py
│   ├── undertone.py
│   ├── skin_depth.py
│   ├── value.py
│   ├── chroma.py
│   ├── contrast.py
│   ├── season_data.py
│   ├── season_classifier.py
│   └── recommendations.py
└── tests/
    ├── conftest.py
    ├── test_classifier.py
    └── test_core_metrics.py
```

## Installation

Python 3.10+ is supported by the project configuration. A dedicated virtual environment is recommended.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Dependency compatibility can vary by operating system and Python version, particularly for MediaPipe. If installation fails, use a Python version for which the pinned MediaPipe release provides a compatible wheel.

## Usage

The public repository intentionally does not contain third-party sample photographs whose redistribution rights have not been verified.

To run the image pipeline, place one or more images that you own or are licensed to use and redistribute in:

```text
images/
```

Supported extensions:

```text
.jpg  .jpeg  .png  .bmp  .webp  .avif
```

Then run from the repository root:

```bash
python src/main.py
```

The script scans `images/` and writes the analysis results to:

```text
results/results.csv
```

If no supported image is present, the program exits without creating an analysis result. If one or more images fail during processing, the script continues with the remaining images, reports the failures, and returns a non-zero exit status. This makes command-line and CI failures detectable without hiding partial results.

## Output

For each processed image, the CSV report can contain:

- representative skin RGB values
- undertone and related measurements
- skin depth
- value and chroma measurements
- eye, eyebrow, and lip colors
- facial contrast and component scores
- selected season and its rule-based score
- the top three candidate seasons

Example structure:

```text
Image,Undertone,Skin Depth,Contrast,Season,Season Score,Top 3 Candidates
example.jpg,Cool,Deep,High,True Winter,...,True Winter (...); Deep Winter (...); Bright Winter (...)
```

The values above are only an **illustrative format**, not a benchmark result.

## Testing

The deterministic core logic can be tested without running the complete MediaPipe image pipeline:

```bash
pytest
```

The full image-processing workflow additionally requires the dependencies in `requirements.txt` and a suitable input image.


### Interpretable scoring

For each classification, the rule engine can expose a score breakdown for the selected season. It reports the contribution of the existing heuristic rules (undertone, skin depth, contrast, eye color, value, and chroma), and these contributions sum to the final seasonal score. This is intended to make the rule-based decision traceable; it is not statistical feature importance.

## Limitations

Results can be affected by:

- illumination and camera white balance
- strong shadows, highlights, or mixed lighting
- heavy makeup or cosmetic products
- facial occlusion
- head pose and landmark quality
- image resolution and compression
- variation in camera color rendering
- heuristic thresholds and weights in the current rule set
- the absence of a validated labeled benchmark dataset

The system should therefore be treated as an experimental computer-vision tool rather than a definitive personal-color assessment.

It is **not** a medical or diagnostic system.

## Privacy

Facial images can contain sensitive biometric information. Only process images that you have the right to use, and do not commit private, identifiable, or sensitive photographs to a public repository.

The repository does not redistribute third-party sample photographs by default. See [`docs/third_party_assets.md`](docs/third_party_assets.md) before adding external assets.

## Third-party assets

The source code license does not grant rights to photographs or other third-party material.

The repository currently contains no third-party sample photographs. If an image is added later, its source, creator, license or written permission, and required attribution should be documented in [`docs/third_party_assets.md`](docs/third_party_assets.md).

## Future work

Possible extensions include:

- illumination and white-balance normalization
- more systematic threshold calibration
- validation against a labeled seasonal-color dataset
- uncertainty analysis and calibrated probabilities
- reproducible ablation studies
- optional web or desktop interface
- expanded clothing and makeup recommendations

## License

The source code in this repository is licensed under the **Apache License, Version 2.0**. See [`LICENSE`](LICENSE) for the complete license text.

Third-party assets, if added in the future, remain subject to their own licenses or permissions.
