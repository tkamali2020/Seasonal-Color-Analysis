# SPDX-License-Identifier: Apache-2.0
from season_data import SEASONS


def _score_season(
    season,
    features,
    undertone,
    skin_depth,
    contrast,
    contrast_score,
    value,
    chroma_score,
    eye_color_key,
):
    """Calculate one season score and keep the contribution of each rule."""
    contributions = {}

    # Undertone has the strongest influence in the current rule set.
    if features["undertone"] == undertone:
        contributions["undertone"] = 8
    elif undertone == "Neutral":
        contributions["undertone"] = 3
    else:
        contributions["undertone"] = -10

    # Skin depth.
    contributions["skin_depth"] = 5 if skin_depth == features["depth"] else -2

    # Contrast category and numeric contrast score.
    contributions["contrast_category"] = 4 if contrast == features["contrast"] else -1

    contrast_adjustment = 0
    if contrast_score >= 36:
        if "Winter" in season or "Bright" in season:
            contrast_adjustment += 3
        elif "Soft" in season:
            contrast_adjustment -= 2
    elif contrast_score <= 22:
        if "Summer" in season or "Soft" in season:
            contrast_adjustment += 3
        elif "Bright" in season or "Winter" in season:
            contrast_adjustment -= 2
    contributions["contrast_numeric"] = contrast_adjustment

    # Eye color.
    normalized_eye_colors = {color.strip().casefold() for color in features["eye_colors"]}
    contributions["eye_color"] = 3 if eye_color_key in normalized_eye_colors else -1

    # Value category.
    value_adjustment = 0
    if value == "Light":
        if "Light" in season:
            value_adjustment += 5
        elif "Spring" in season or "Summer" in season:
            value_adjustment += 2
        elif "Deep" in season or "Dark" in season:
            value_adjustment -= 4
    elif value in {"Dark", "Deep"}:
        if "Deep" in season or "Dark" in season:
            value_adjustment += 5
        elif "Autumn" in season or "Winter" in season:
            value_adjustment += 2
        elif "Light" in season:
            value_adjustment -= 4
    contributions["value"] = value_adjustment

    # Numeric chroma score.
    chroma_adjustment = 0
    if chroma_score >= 27:
        if "Bright" in season or "Winter" in season or "Spring" in season:
            chroma_adjustment += 5
        elif "Soft" in season:
            chroma_adjustment -= 4
    elif chroma_score <= 18:
        if "Soft" in season or "Summer" in season or "Autumn" in season:
            chroma_adjustment += 5
        elif "Bright" in season:
            chroma_adjustment -= 4
    else:
        if "True" in season:
            chroma_adjustment += 3
    contributions["chroma"] = chroma_adjustment

    return contributions, sum(contributions.values())


def calculate_score_breakdown(
    undertone,
    skin_depth,
    contrast,
    contrast_score,
    value,
    chroma_score,
    eye_color,
):
    """Return per-season rule contributions and totals.

    The returned values expose the existing heuristic rules without changing
    their weights or thresholds. This is intended for interpretability and
    debugging, not as evidence of statistical feature importance.
    """
    eye_color_key = str(eye_color).strip().casefold()
    contrast = contrast["contrast"] if isinstance(contrast, dict) else contrast

    breakdown = {}
    for season, features in SEASONS.items():
        contributions, total = _score_season(
            season,
            features,
            undertone,
            skin_depth,
            contrast,
            contrast_score,
            value,
            chroma_score,
            eye_color_key,
        )
        breakdown[season] = {
            "contributions": contributions,
            "total": total,
        }

    return breakdown


def calculate_scores(
    undertone,
    skin_depth,
    contrast,
    contrast_score,
    value,
    chroma_score,
    eye_color,
):
    """Return rule-based scores for all 12 seasonal categories.

    This is a heuristic scoring system, not a statistically trained classifier.
    """
    breakdown = calculate_score_breakdown(
        undertone,
        skin_depth,
        contrast,
        contrast_score,
        value,
        chroma_score,
        eye_color,
    )
    return {season: details["total"] for season, details in breakdown.items()}


def rank_seasons(scores):
    """Return seasons sorted by descending rule-based score."""
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def classify_season(
    undertone,
    skin_depth,
    contrast,
    contrast_score,
    value,
    chroma_score,
    eye_color,
    return_details=False,
):
    """Classify an image using the rule-based seasonal scoring system.

    Set return_details=True to obtain the selected season, score ranking,
    top candidates, and the contribution breakdown for the selected season.
    The default return value remains the season name.
    """
    breakdown = calculate_score_breakdown(
        undertone,
        skin_depth,
        contrast,
        contrast_score,
        value,
        chroma_score,
        eye_color,
    )
    scores = {season: details["total"] for season, details in breakdown.items()}
    ranked = rank_seasons(scores)
    best_match, best_score = ranked[0]

    if not return_details:
        return best_match

    return {
        "season": best_match,
        "score": best_score,
        "scores": dict(ranked),
        "top_candidates": ranked[:3],
        "score_breakdown": breakdown[best_match],
    }
