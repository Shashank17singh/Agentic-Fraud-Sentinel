from fastapi.testclient import TestClient

from api.main import app, monitor

client = TestClient(app)


def _predict(tx: str) -> dict:
    reply = client.post(
        "/predict", json={"transaction_id": tx, "features": {"TransactionAmt": 25}}
    )
    assert reply.status_code == 200
    return reply.json()


def test_flagged_cases_pause_and_resume_with_the_analyst_decision() -> None:
    body = _predict("api-1")
    assert body["requires_human"] is True
    assert "[AWAITING ANALYST REVIEW]" in body["policy_reasoning"]
    assert "api-1" in [r["transaction_id"] for r in client.get("/reviews").json()]

    resolved = client.post("/reviews/api-1", json={"action": "deny", "note": "stolen"})

    report = resolved.json()["final_report"]
    assert resolved.json()["status"] == "deny"
    assert report["decision"] == "deny"
    assert "[ANALYST DENY] stolen" in report["policy_reasoning"]
    assert client.post("/reviews/api-1", json={"action": "deny"}).status_code == 404
    assert client.post("/reviews/x", json={"action": "maybe"}).status_code == 422


def test_predictions_feed_the_stats() -> None:
    _predict("api-2")

    stats = client.get("/stats").json()

    assert stats["requests_total"] >= 1
    assert stats["latency_ms"]["p50"] is not None
    assert "TransactionAmt" in stats["drift_tracked"]


def test_shap_factors_say_which_way_they_push() -> None:
    factor = _predict("api-3")["shap_top_features"][0]

    assert factor["direction"] in ("increases fraud risk", "decreases fraud risk")
    assert monitor.stats()["models"]
