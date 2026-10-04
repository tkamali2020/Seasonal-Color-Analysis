from skin_depth import analyze_skin_depth
from value import analyze_value


def test_skin_depth_has_three_supported_categories():
    result = analyze_skin_depth([90, 70, 60])
    assert result in {"Light", "Medium", "Deep"}


def test_value_returns_expected_fields():
    result = analyze_value([180, 150, 140])
    assert set(result) == {"L", "value"}
    assert 0 <= result["L"] <= 100
    assert result["value"] in {"Light", "Medium", "Dark"}


def test_resize_for_analysis_preserves_aspect_ratio():
    import numpy as np
    from main import resize_for_analysis

    image = np.zeros((300, 900, 3), dtype=np.uint8)
    resized = resize_for_analysis(image, max_size=600)

    assert resized.shape[:2] == (200, 600)


def test_skin_depth_uses_same_lab_lightness_convention_as_value():
    from skin_depth import analyze_skin_depth

    samples = [
        ([220, 200, 190], "Light"),
        ([160, 140, 130], "Medium"),
        ([80, 65, 55], "Deep"),
    ]
    for rgb, expected in samples:
        assert analyze_skin_depth(rgb) == expected


def test_brow_and_lip_sampling_handles_landmarks_outside_image():
    from types import SimpleNamespace
    from eyebrows import analyze_eyebrows
    from lips import analyze_lips
    import numpy as np

    face = SimpleNamespace(
        landmark=[SimpleNamespace(x=2.0, y=2.0) for _ in range(478)]
    )
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    assert analyze_eyebrows(image, face) == [60, 50, 40]
    assert analyze_lips(image, face) == [150, 90, 90]
