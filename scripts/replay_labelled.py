"""
Send real, labelled test-period transactions through a running API, then report
each one's true outcome to /feedback, the way chargebacks would arrive later.
/stats then shows the model's live precision, recall and false-positive rate.

Run scripts/build_inference_artifacts.py first to create the sample.

    python scripts/replay_labelled.py [API_URL]
"""

import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "notebooks", "data", "processed", "replay_sample.parquet")


def _post(url: str, body: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as reply:
        return json.load(reply)


def _plain(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def main(api: str) -> None:
    sample = pd.read_parquet(SAMPLE)
    for i, row in enumerate(sample.to_dict("records"), 1):
        tx = f"replay-{int(row['TransactionID'])}"
        features = {k: _plain(v) for k, v in row.items() if k != "isFraud"}
        _post(f"{api}/predict", {"transaction_id": tx, "features": features})
        _post(f"{api}/feedback/{tx}", {"is_fraud": bool(row["isFraud"])})
        if i % 100 == 0:
            print(f"{i}/{len(sample)} sent")

    with urllib.request.urlopen(f"{api}/stats", timeout=60) as reply:
        stats = json.load(reply)
    print(json.dumps(stats["labelled_performance"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000")
