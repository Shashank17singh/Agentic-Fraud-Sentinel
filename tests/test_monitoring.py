import numpy as np
import pytest

from src.monitoring import Monitor, build_reference, psi


def _ok(monitor: Monitor, tx: str, decision: str, human: bool, amount=50.0, ms=10.0):
    monitor.record(
        transaction_id=tx,
        latency_ms=ms,
        model_version="xgb",
        threshold=0.2,
        error=None,
        fraud_probability=0.3,
        decision=decision,
        requires_human=human,
        thread_id=f"{tx}-t",
        features={"TransactionAmt": amount, "unused": 1},
    )


def test_psi_is_near_zero_for_the_same_distribution_and_large_after_a_shift() -> None:
    rng = np.random.default_rng(0)
    reference = build_reference(rng.lognormal(3, 1, 20_000))

    assert psi(reference, rng.lognormal(3, 1, 2_000).tolist()) < 0.05
    assert psi(reference, rng.lognormal(5, 1, 2_000).tolist()) > 0.2


def test_stats_report_rates_latency_and_errors() -> None:
    monitor = Monitor()
    _ok(monitor, "a", "approve", False, ms=10)
    _ok(monitor, "b", "deny", True, ms=30)
    monitor.record(transaction_id="c", latency_ms=5, model_version="xgb", error="boom")

    stats = monitor.stats()

    assert stats["requests_total"] == 3
    assert stats["error_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert stats["decision_rates"] == {"approve": 0.5, "deny": 0.5}
    assert stats["latency_ms"]["p50"] == 10
    assert "error_rate 0.3333 is above 0.05" in stats["alerts"]


def test_flagged_cases_queue_until_an_analyst_resolves_them() -> None:
    monitor = Monitor()
    _ok(monitor, "a", "approve", False)
    _ok(monitor, "b", "flag", True)

    assert [r["transaction_id"] for r in monitor.pending_reviews()] == ["b"]
    assert monitor.pending("b")["thread_id"] == "b-t"

    resolved = monitor.resolve("b", "deny", "card reported stolen")

    assert resolved["status"] == "deny"
    assert monitor.pending_reviews() == []
    queue = monitor.stats()["review_queue"]
    assert queue["resolved"] == 1
    assert queue["confirmed_fraud_rate"] == 1.0
    with pytest.raises(KeyError):
        monitor.resolve("b", "approve")


def test_drift_is_tracked_per_feature_and_for_scores_once_there_is_enough_data() -> (
    None
):
    rng = np.random.default_rng(0)
    reference = {
        "TransactionAmt": build_reference(rng.lognormal(3, 1, 5_000)),
        "fraud_probability": build_reference(rng.uniform(0, 0.2, 5_000)),
    }
    monitor = Monitor(reference)
    for i in range(10):
        _ok(monitor, str(i), "approve", False, amount=5_000.0)
    assert monitor.stats()["drift_psi"] == {}

    for i in range(10, 60):
        _ok(monitor, str(i), "approve", False, amount=5_000.0)
    stats = monitor.stats()

    assert stats["drift_psi"]["TransactionAmt"] > 0.2
    assert stats["drift_psi"]["fraud_probability"] > 0.2
    assert any(a.startswith("psi TransactionAmt") for a in stats["alerts"])
    assert "unused" not in monitor.records[-1]["inputs"]


def test_a_restarted_monitor_rebuilds_stats_and_the_queue_from_its_log(
    tmp_path,
) -> None:
    log = tmp_path / "predictions.jsonl"
    first = Monitor(log_path=str(log))
    _ok(first, "a", "approve", False)
    _ok(first, "b", "flag", True)
    _ok(first, "c", "deny", True)
    first.resolve("c", "deny")

    restarted = Monitor(log_path=str(log))

    assert restarted.stats()["requests_total"] == 3
    assert [r["transaction_id"] for r in restarted.pending_reviews()] == ["b"]
    assert restarted.stats()["review_queue"]["resolved"] == 1


def test_true_outcomes_give_live_precision_recall_and_false_positive_rate(
    tmp_path,
) -> None:
    log = tmp_path / "monitor.jsonl"
    monitor = Monitor({}, str(log))
    # threshold 0.2, every score 0.3: all four are predicted fraud
    for tx in "abcd":
        _ok(monitor, tx, "deny", True)
    monitor.label("a", True)
    monitor.label("b", True)
    monitor.label("c", False)

    perf = monitor.stats()["labelled_performance"]

    assert perf["labelled"] == 3
    assert perf["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert perf["recall"] == 1.0
    assert perf["false_positive_rate"] == 1.0
    with pytest.raises(KeyError):
        monitor.label("never-scored", True)
    assert Monitor({}, str(log)).stats()["labelled_performance"] == perf
