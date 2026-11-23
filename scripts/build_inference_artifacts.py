"""
Rebuild the preprocessing the shipped model was trained with and save it into
the model bundle, so the API can turn raw transactions into model inputs.

Everything fitted here (frequency maps, medians, category codes) is
deterministic given the raw files, so it matches what the retrain notebook
fitted. The script checks that: it scores the test period and compares the
result with the metrics saved in the bundle. It also keeps a labelled sample of
test-period transactions for scripts/replay_labelled.py.

    python scripts/build_inference_artifacts.py
"""

import gc
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.features import prepare_transaction
from src.data.preprocessing import (
    fit_transform_split,
    inference_artifacts,
    load_engineered,
    time_based_split,
)

RAW = os.path.join(ROOT, "data", "raw")
MODEL_PATH = os.path.join(ROOT, "notebooks", "data", "models", "xgboost_production.pkl")
REPLAY_PATH = os.path.join(
    ROOT, "notebooks", "data", "processed", "replay_sample.parquet"
)
REPLAY_ROWS = 1000


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    cols = model.get_booster().feature_names

    df = load_engineered(
        os.path.join(RAW, "train_transaction.csv"),
        os.path.join(RAW, "train_identity.csv"),
    )
    train, test = time_based_split(df)
    del df
    gc.collect()

    # Raw test-period rows, before anything fitted touches them: what a caller
    # would send. The model never saw this period.
    replay = test.sample(REPLAY_ROWS, random_state=42).sort_values("TransactionDT")

    train, test = fit_transform_split(train, test)
    del train
    gc.collect()

    missing = sorted(set(cols) - set(test.columns))
    assert not missing, f"model columns the pipeline did not produce: {missing[:5]}"

    proba = model.predict_proba(test[cols])[:, 1]
    saved = bundle["metrics"]
    print(
        f"Test period: ROC-AUC {roc_auc_score(test['isFraud'], proba):.4f} "
        f"(saved {saved['auc_roc']:.4f}), PR-AUC "
        f"{average_precision_score(test['isFraud'], proba):.4f} "
        f"(saved {saved['auc_pr']:.4f})"
    )

    prep = inference_artifacts()

    # The API path must give the same scores as the batch pipeline.
    offline = model.predict_proba(test.loc[replay.index, cols])[:, 1]
    rows = [
        prepare_transaction(_plain(r), prep, cols)
        for r in replay.drop(columns=["isFraud"]).to_dict("records")
    ]

    online = model.predict_proba(pd.DataFrame(rows, columns=cols))[:, 1]
    gap = float(np.max(np.abs(online - offline)))
    print(f"API preparation vs batch pipeline, max score gap: {gap:.2e}")
    assert gap < 1e-4, "single-transaction preparation does not match training"

    bundle["preprocessing"] = prep
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved preprocessing into {MODEL_PATH}")

    os.makedirs(os.path.dirname(REPLAY_PATH), exist_ok=True)
    replay.to_parquet(REPLAY_PATH)
    print(
        f"Saved {len(replay)} labelled test-period rows "
        f"({int(replay['isFraud'].sum())} fraud) to {REPLAY_PATH}"
    )


def _plain(row: dict) -> dict:
    """JSON-ready values: NaN becomes None, numpy scalars become Python ones."""
    out = {}
    for k, v in row.items():
        if isinstance(v, float) and np.isnan(v):
            v = None
        elif hasattr(v, "item"):
            v = v.item()
            if isinstance(v, float) and np.isnan(v):
                v = None
        out[k] = v
    return out


if __name__ == "__main__":
    main()
