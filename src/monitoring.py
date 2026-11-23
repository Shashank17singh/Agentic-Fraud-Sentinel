"""
Live monitoring for the fraud API: per-request log, latency, decision rates,
input and score drift, the human-review queue and simple alerts.

Build the input drift reference from the training period with:
    python -m src.monitoring data/raw/train_transaction.csv

The retrain notebook adds the model-score reference ("fraud_probability").
"""

import json
import os
import sys
import threading
from collections import Counter, deque
from datetime import datetime, timezone

import numpy as np

# shortcut: stats cover the last WINDOW requests in memory. Set a log path to
# rebuild them after a restart; on a host whose disk is wiped each deploy (Render
# free tier) they still reset, so use a database or a persistent disk there.
WINDOW = 1000
MIN_DRIFT_SAMPLE = 50

# Raw inputs tracked for drift: the highest-ranked numeric fields by mean SHAP.
DRIFT_FEATURES = [
    "TransactionAmt",
    "C1",
    "C11",
    "C13",
    "C14",
    "V70",
    "V90",
    "V258",
    "V294",
    "V317",
]

# shortcut: fixed alert limits picked as common defaults (PSI 0.2 is the usual
# "significant shift" line); tune them on real traffic.
ALERT_LIMITS = {
    "error_rate": 0.05,
    "latency_p95_ms": 2000,
    "psi": 0.2,
    "human_review_share": 0.5,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_reference(values: np.ndarray, bins: int = 10) -> dict:
    """Decile edges of training values and the share of training rows in each bin."""
    values = values[~np.isnan(values)]
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
    counts = np.bincount(
        np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1
    )
    return {"edges": edges.tolist(), "proportions": (counts / counts.sum()).tolist()}


def psi(reference: dict, values: list[float]) -> float:
    """Population stability index of live values against the training bins."""
    edges = np.asarray(reference["edges"])
    expected = np.asarray(reference["proportions"])
    counts = np.bincount(
        np.searchsorted(edges, np.asarray(values, dtype=float), side="right"),
        minlength=len(edges) + 1,
    )
    actual = counts / counts.sum()
    eps = 1e-6  # keeps empty bins from dividing by zero
    actual, expected = np.clip(actual, eps, None), np.clip(expected, eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class Monitor:
    """Thread-safe store of recent predictions and the review queue."""

    def __init__(self, reference: dict | None = None, log_path: str | None = None):
        self.reference = reference or {}
        self.log_path = log_path
        self.records: deque = deque(maxlen=WINDOW)
        self.reviews: dict[str, dict] = {}
        self.total = 0
        self.errors = 0
        self.started = _now()
        self._lock = threading.Lock()
        if log_path and os.path.exists(log_path):
            self._replay(log_path)

    def _replay(self, path: str) -> None:
        """Rebuild stats and the review queue from an earlier run's log."""
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut short by a crash
                if event.get("event") == "review":
                    case = self.reviews.get(event["transaction_id"])
                    if case:
                        case.update({k: v for k, v in event.items() if k != "event"})
                else:
                    self._add(event)

    def _append(self, event: dict) -> None:
        if not self.log_path:
            return
        try:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except OSError:
            pass  # logging must never fail a prediction

    def _add(self, rec: dict) -> None:
        self.records.append(rec)
        self.total += 1
        self.errors += rec.get("error") is not None
        if rec.get("requires_human"):
            self.reviews[rec["transaction_id"]] = {
                "transaction_id": rec["transaction_id"],
                "thread_id": rec.get("thread_id"),
                "model_decision": rec["decision"],
                "fraud_probability": rec["fraud_probability"],
                "status": "pending",
                "queued_at": rec["time"],
            }

    def record(self, **fields) -> dict:
        """
        Log one /predict call. Expected fields: transaction_id, latency_ms,
        model_version, threshold, error, and for a success fraud_probability,
        decision, requires_human, thread_id and the request's features.
        Only the features tracked for drift are kept.
        """
        features = fields.pop("features", None) or {}
        rec = {"event": "prediction", "time": _now(), **fields}
        rec["inputs"] = {
            k: features[k]
            for k in self.reference
            if k in features and _is_number(features[k])
        }
        with self._lock:
            self._add(rec)
        self._append(rec)
        return rec

    def pending_reviews(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.reviews.values() if r["status"] == "pending"]

    def pending(self, transaction_id: str) -> dict:
        """The pending case for this ID; raises KeyError if there is none."""
        with self._lock:
            case = self.reviews.get(transaction_id)
            if case is None or case["status"] != "pending":
                raise KeyError(transaction_id)
            return dict(case)

    def resolve(self, transaction_id: str, action: str, note: str = "") -> dict:
        """Record an analyst's approve or deny for a pending case."""
        if action not in ("approve", "deny"):
            raise ValueError("action must be 'approve' or 'deny'")
        with self._lock:
            case = self.reviews.get(transaction_id)
            if case is None or case["status"] != "pending":
                raise KeyError(transaction_id)
            now = datetime.now(timezone.utc)
            waited = now - datetime.fromisoformat(case["queued_at"])
            update = {
                "status": action,
                "note": note,
                "resolved_at": now.isoformat(),
                "resolution_seconds": round(waited.total_seconds(), 1),
            }
            case.update(update)
            result = dict(case)
        self._append({"event": "review", "transaction_id": transaction_id, **update})
        return result

    def stats(self) -> dict:
        with self._lock:
            records = list(self.records)
            reviews = [dict(r) for r in self.reviews.values()]
            total, errors = self.total, self.errors

        ok = [r for r in records if r.get("error") is None]
        latencies = [r["latency_ms"] for r in records]
        p50, p95, p99 = (
            [round(float(v), 1) for v in np.percentile(latencies, [50, 95, 99])]
            if latencies
            else [None] * 3
        )
        decisions = Counter(r["decision"] for r in ok)
        resolved = [r for r in reviews if r["status"] != "pending"]

        drift = {}
        for name, ref in self.reference.items():
            if name == "fraud_probability":
                values = [r["fraud_probability"] for r in ok]
            else:
                values = [r["inputs"][name] for r in ok if name in r.get("inputs", {})]
            if len(values) >= MIN_DRIFT_SAMPLE:
                drift[name] = round(psi(ref, values), 4)

        out = {
            "since": self.started,
            "window": len(records),
            "requests_total": total,
            "errors_total": errors,
            "error_rate": round(errors / total, 4) if total else 0.0,
            "latency_ms": {"p50": p50, "p95": p95, "p99": p99},
            "decision_rates": {k: round(v / len(ok), 4) for k, v in decisions.items()}
            if ok
            else {},
            "human_review_share": round(
                sum(bool(r.get("requires_human")) for r in ok) / len(ok), 4
            )
            if ok
            else 0.0,
            "mean_fraud_probability": round(
                float(np.mean([r["fraud_probability"] for r in ok])), 4
            )
            if ok
            else None,
            "review_queue": {
                "pending": sum(r["status"] == "pending" for r in reviews),
                "resolved": len(resolved),
                "mean_resolution_seconds": round(
                    float(np.mean([r["resolution_seconds"] for r in resolved])), 1
                )
                if resolved
                else None,
                # Analyst decisions are the only ground truth the service sees: the
                # share denied is the precision of the cases sent to review.
                "confirmed_fraud_rate": round(
                    sum(r["status"] == "deny" for r in resolved) / len(resolved), 4
                )
                if resolved
                else None,
            },
            "drift_psi": drift,
            "drift_tracked": sorted(self.reference),
            "models": sorted(
                {r["model_version"] for r in records if r.get("model_version")}
            ),
        }
        checks = [
            ("error_rate", out["error_rate"], ALERT_LIMITS["error_rate"]),
            ("latency_p95_ms", p95, ALERT_LIMITS["latency_p95_ms"]),
            (
                "human_review_share",
                out["human_review_share"],
                ALERT_LIMITS["human_review_share"],
            ),
        ] + [
            (f"psi {name}", value, ALERT_LIMITS["psi"]) for name, value in drift.items()
        ]
        out["alerts"] = [
            f"{name} {value} is above {limit}"
            for name, value, limit in checks
            if value is not None and value > limit
        ]
        return out


def _training_references(path: str) -> dict:
    """References for DRIFT_FEATURES from the first 80% of rows by time (training)."""
    import pandas as pd

    df = pd.read_csv(path, usecols=["TransactionDT", *DRIFT_FEATURES])
    train = df[df["TransactionDT"] <= df["TransactionDT"].quantile(0.8)]
    return {
        name: build_reference(train[name].to_numpy(dtype=float))
        for name in DRIFT_FEATURES
    }


REFERENCE_FILE = os.path.join("notebooks", "data", "models", "drift_reference.json")

if __name__ == "__main__":
    refs = _training_references(sys.argv[1])
    if os.path.exists(REFERENCE_FILE):  # keep a score reference saved by the notebook
        with open(REFERENCE_FILE, encoding="utf-8") as f:
            refs = {**json.load(f), **refs}
    with open(REFERENCE_FILE, "w", encoding="utf-8") as f:
        json.dump(refs, f, indent=1)
    print(f"Saved references for {sorted(refs)} to {REFERENCE_FILE}")
