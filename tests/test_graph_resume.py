import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from src.agents.graph import build_fraud_graph

STATE = {
    "transaction_id": "g1",
    "transaction_data": {"TransactionAmt": 25},
    "fraud_probability": None,
    "risk_level": None,
    "shap_explanation": None,
    "explanation_text": None,
    "decision": None,
    "policy_reasoning": None,
    "requires_human": None,
    "analyst_decision": None,
    "analyst_note": None,
    "final_report": None,
    "processing_errors": [],
}


def _graph(db: str):
    return build_fraud_graph(SqliteSaver(sqlite3.connect(db, check_same_thread=False)))


def test_a_paused_run_resumes_after_a_restart(tmp_path) -> None:
    db = str(tmp_path / "checkpoints.db")
    config = {"configurable": {"thread_id": "g1-run"}}

    first = _graph(db)
    first.invoke(STATE, config)
    assert first.get_state(config).next == ("human_review",)

    restarted = _graph(db)  # a new process would open the same file
    restarted.update_state(config, {"analyst_decision": "approve"})
    report = restarted.invoke(None, config)["final_report"]

    assert report["decision"] == "approve"
    assert "[ANALYST APPROVE]" in report["policy_reasoning"]


def test_without_a_checkpointer_the_graph_runs_straight_through() -> None:
    report = build_fraud_graph().invoke(STATE)["final_report"]

    assert report["requires_human"] is True
    assert "[QUEUED FOR HUMAN REVIEW]" in report["policy_reasoning"]
