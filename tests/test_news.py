import joblib
import pytest

from src.news import classifier

REAL_STYLE = (
    "WASHINGTON (Reuters) - The U.S. Senate on Tuesday approved a spending bill that "
    "funds federal agencies through the end of the fiscal year, sending the measure "
    "to the House of Representatives, where leaders said they expected a vote later "
    "this week. The bill passed 65 to 32 after negotiators from both parties agreed "
    "on funding levels for defense and domestic programs, aides said."
)


@pytest.fixture(scope="module")
def bundle():
    return joblib.load(classifier.MODEL_PATH)


def test_publisher_cues_are_removed() -> None:
    cleaned = classifier.clean(
        "WASHINGTON (Reuters) - Talks resumed. Featured image via Getty https://x.co/a"
    )

    assert cleaned == "Talks resumed."


def test_short_texts_are_not_scored(bundle) -> None:
    result = classifier.screen(bundle, "Aliens landed in Ohio")

    assert result["label"] == "uncertain"
    assert result["fake_probability"] is None


def test_wire_style_report_is_screened_as_likely_real(bundle) -> None:
    result = classifier.screen(bundle, REAL_STYLE)

    assert result["label"] == "likely real"
    assert 0 <= result["fake_probability"] <= classifier.REAL_BELOW


def test_saved_metrics_are_present(bundle) -> None:
    assert bundle["metrics"]["auc_roc"] > 0.9
