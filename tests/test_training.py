import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.models import train


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    features = pd.DataFrame(rng.normal(size=(400, 4)), columns=list("abcd"))
    labels = pd.Series((features["a"] + features["b"] > 0.8).astype(int))
    return features, labels


def test_production_parameters_match_the_shipped_model() -> None:
    assert train.PRODUCTION_XGB_PARAMS["n_estimators"] == 705
    assert train.PRODUCTION_XGB_PARAMS["max_depth"] == 8
    assert train.PRODUCTION_XGB_PARAMS["eval_metric"] == "aucpr"


@pytest.mark.parametrize(
    "trainer", [train.train_xgboost_baseline, train.train_lightgbm_baseline]
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

    model, study = train.tune_xgboost(features, labels, n_trials=2)

    assert len(study.trials) == 2
    assert model.predict_proba(features).shape == (len(features), 2)
    assert model.get_params()["n_estimators"] == study.best_params["n_estimators"]
