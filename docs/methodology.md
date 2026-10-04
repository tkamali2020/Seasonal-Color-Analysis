# Methodology

## Overview

The project estimates a 12-season personal color category from a facial image.
It uses facial landmarks and color measurements rather than a trained
machine-learning classifier.

## Processing pipeline

1. Resize the input image so its longest side is at most 640 pixels, while preserving aspect ratio.
2. Detect a single face with MediaPipe Face Mesh.
3. Sample skin regions from predefined facial landmarks and filter invalid
   pixels using HSV thresholds.
4. Estimate skin undertone using CIE L*, a*, b*, hue angle, b/a ratio, and ITA.
5. Estimate coarse skin depth and perceptual value from CIE L*. These are related lightness descriptors, so they should not be interpreted as independent biological measurements.
6. Sample iris regions and classify eye color from HSV statistics.
7. Estimate eyebrow and lip colors using landmark-based patches and K-Means.
8. Estimate facial contrast using CIE L* differences and CIE76 Delta E (Euclidean Lab color distance).
9. Apply a weighted, rule-based scoring system to the 12 seasonal categories.
10. Return the selected season, top candidates, and color recommendations.

## Important interpretation

The current season classifier is heuristic and rule-based. Its scores are
evidence scores produced by the implemented rules; they are not probabilities,
confidence values, or accuracy measurements.

The threshold values and weights should therefore be interpreted as
calibration choices of the current prototype. In particular, the undertone
hue/ratio thresholds, chroma thresholds, and contrast thresholds are heuristic
engineering rules rather than universally accepted seasonal-color standards. They require validation against
a labeled dataset before claims about predictive accuracy or clinical/scientific
performance can be made.

## Limitations

Results can be affected by illumination, camera white balance, makeup,
occlusion, facial pose, image resolution, and landmark quality.
The current implementation is intended as an interpretable prototype and
should not be treated as a professional or diagnostic assessment.

### Rule-score explainability

The seasonal classifier exposes the contribution of each existing heuristic rule to the score of the selected season. The reported breakdown separates undertone, skin depth, categorical contrast, numeric contrast adjustment, eye color, value, and chroma contributions. The contributions sum exactly to the reported seasonal score.

This breakdown is an interpretability aid for understanding the rule system; it should not be interpreted as statistical feature importance, causal importance, or learned model attribution. No classifier weights or thresholds were changed to add this functionality.
