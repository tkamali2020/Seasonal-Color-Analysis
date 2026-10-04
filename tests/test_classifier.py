from season_classifier import calculate_scores, classify_season, rank_seasons
from season_data import SEASONS


def test_all_twelve_seasons_are_defined():
    assert len(SEASONS) == 12


def test_scores_are_returned_for_every_season():
    scores = calculate_scores(
        undertone="Cool",
        skin_depth="Deep",
        contrast="High",
        contrast_score=40,
        value="Dark",
        chroma_score=32,
        eye_color="blue",
    )
    assert set(scores) == set(SEASONS)
    assert all(isinstance(score, int) for score in scores.values())


def test_classifier_returns_valid_season():
    result = classify_season(
        undertone="Cool",
        skin_depth="Deep",
        contrast="High",
        contrast_score=40,
        value="Dark",
        chroma_score=32,
        eye_color="blue",
    )
    assert result in SEASONS


def test_ranking_is_deterministic():
    scores = {"B": 2, "A": 2, "C": 1}
    assert rank_seasons(scores) == [("A", 2), ("B", 2), ("C", 1)]


def test_eye_color_matching_is_case_insensitive():
    scores = calculate_scores(
        undertone="Cool",
        skin_depth="Deep",
        contrast="High",
        contrast_score=40,
        value="Dark",
        chroma_score=32,
        eye_color="dark brown",
    )
    assert scores["Dark Winter"] > scores["Bright Winter"]


def test_score_breakdown_sums_to_each_season_score():
    from season_classifier import calculate_score_breakdown

    breakdown = calculate_score_breakdown(
        undertone="Warm",
        skin_depth="Medium",
        contrast="Medium",
        contrast_score=28,
        value="Light",
        chroma_score=22,
        eye_color="hazel",
    )

    assert set(breakdown) == set(SEASONS)
    for details in breakdown.values():
        assert details["total"] == sum(details["contributions"].values())
        assert set(details["contributions"]) == {
            "undertone",
            "skin_depth",
            "contrast_category",
            "contrast_numeric",
            "eye_color",
            "value",
            "chroma",
        }


def test_classification_details_expose_selected_season_breakdown():
    result = classify_season(
        undertone="Cool",
        skin_depth="Deep",
        contrast="High",
        contrast_score=40,
        value="Dark",
        chroma_score=32,
        eye_color="blue",
        return_details=True,
    )

    assert result["score_breakdown"]["total"] == result["score"]
    assert result["score_breakdown"]["total"] == sum(
        result["score_breakdown"]["contributions"].values()
    )
