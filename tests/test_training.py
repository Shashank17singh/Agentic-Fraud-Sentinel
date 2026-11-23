import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.models import evaluate, train


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    features = pd.DataFrame(rng.normal(size=(400, 4)), columns=list("abcd"))
    labels = pd.Series((features["a"] + features["b"] > 0.8).astype(int))
    return features, labels


@pytest.mark.parametrize(
    "trainer",
    [
        train.train_xgboost_baseline,
        train.train_xgboost_weighted,
        train.train_lightgbm_baseline,
    ],
)
def test_baselines_learn_the_signal(
    trainer, data: tuple[pd.DataFrame, pd.Series]
) -> None:
    features, labels = data

    model = trainer(features, labels)

    assert roc_auc_score(labels, model.predict_proba(features)[:, 1]) > 0.9


def test_tuning_returns_a_fitted_model_and_its_study(
    data: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, labels = data

    model, study = train.tune_xgboost(
        features[:300], labels[:300], features[300:], labels[300:], n_trials=2
    )

    assert len(study.trials) == 2
    assert model.predict_proba(features).shape == (len(features), 2)
    assert model.get_params()["n_estimators"] == study.best_params["n_estimators"]


def test_threshold_is_the_best_f1_cut_on_the_labels_given() -> None:
    labels = pd.Series([0, 0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.6, 0.7, 0.9])

    assert evaluate.select_threshold(labels, probs) == pytest.approx(0.7)


def test_runs_are_logged_with_commit_dataset_and_params(data) -> None:
    features, labels = data
    model = train.train_xgboost_baseline(features, labels)

    result = evaluate.evaluate_model(model, features, labels, model_name="m")

    assert result["git_commit"]
    assert "IEEE-CIS" in result["dataset"]
    assert result["params"]["max_depth"] == 6


def test_weighting_uses_the_legitimate_to_fraud_ratio(data) -> None:
    features, labels = data

    model = train.train_xgboost_weighted(features, labels)

    expected = (labels == 0).sum() / (labels == 1).sum()
    assert model.get_params()["scale_pos_weight"] == pytest.approx(expected)
