# pyrefly: ignore [missing-import]
import json
import logging
import os
import sqlite3
import sys
import time
import uuid
from typing import Any, Literal

import joblib

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException

# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if os.path.exists(os.path.join(BASE_DIR, "data", "models", "xgboost_production.pkl")):
    DATA_DIR = os.path.join(BASE_DIR, "data")
else:
    DATA_DIR = os.path.join(BASE_DIR, "notebooks", "data")


from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agents.graph import build_fraud_graph
from src.agents.nodes import report_node
from src.chat import assistant
from src.monitoring import Monitor
from src.news import classifier as news_classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic-Fraud-Sentinel API",
    description="Multi agent fraud detection - XGBoost and LangGraph",
    version="1.0.0",
)

print("Building fraud detection graph..")
# The checkpointer lets a run pause before human review and resume after an
# analyst decides. With CHECKPOINT_DB_PATH set, paused runs are kept in SQLite and
# survive a restart; otherwise they live in memory.
CHECKPOINT_DB_PATH = os.environ.get("CHECKPOINT_DB_PATH")
if CHECKPOINT_DB_PATH:
    os.makedirs(os.path.dirname(CHECKPOINT_DB_PATH) or ".", exist_ok=True)
    checkpointer = SqliteSaver(
        sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    )
else:
    checkpointer = InMemorySaver()
fraud_graph = build_fraud_graph(checkpointer)
print("Graph ready...")
bundle = joblib.load(os.path.join(DATA_DIR, "models", "xgboost_production.pkl"))
news_bundle = joblib.load(os.path.join(DATA_DIR, "models", "news_classifier.pkl"))
MODEL_VERSION = bundle["metrics"]["model_name"]

with open(
    os.path.join(DATA_DIR, "models", "drift_reference.json"), encoding="utf-8"
) as f:
    drift_reference = json.load(f)
# Set MONITOR_LOG_PATH to append every prediction and review to a JSON-lines file,
# which also rebuilds the stats and the review queue after a restart.
monitor = Monitor(drift_reference, os.environ.get("MONITOR_LOG_PATH"))


class TransactionRequest(BaseModel):
    """
    Input schema for a transaction prediction request.
    """

    transaction_id: str
    features: dict


class PredictionResponse(BaseModel):
    """
    Output schema for a transaction prediction response.
    """

    transaction_id: str
    fraud_probability: float
    risk_level: str
    decision: str
    requires_human: bool
    explanation: str
    policy_reasoning: str
    shap_top_features: list[Any]
    errors: list[Any]

    model_config = {"arbitrary_types_allowed": True}


@app.get("/health")
def health_check():
    """
    Health check endpoint to verify API availability.

    Returns:
        dict: Basic health status and model version.
    """

    return {"status": "ok", "model": "xgboost_tuned", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest):
    """
    Main endpoint for transaction fraud prediction.

    Executes a multi-agent graph pipeline comprising:
    RiskScorer -> Explainer -> Policy -> [HumanReview|AutoApprove] -> Report.

    Args:
        request (TransactionRequest): The incoming transaction details.

    Returns:
        PredictionResponse: Structured fraud decision and explanation.
    """
    start = time.perf_counter()
    try:
        # Pass features as sent: the nodes fill gaps for the model, and the policy
        # needs to see how many were actually supplied.
        initial_state = {
            "transaction_id": request.transaction_id,
            "transaction_data": request.features,
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

        # One thread per request, so a repeated transaction ID never resumes an
        # older run.
        thread_id = f"{request.transaction_id}-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}
        result = fraud_graph.invoke(initial_state, config)

        if fraud_graph.get_state(config).next == ("human_review",):
            # Paused for an analyst: answer now from the state so far, and keep
            # the checkpoint so POST /reviews/{id} can resume the run.
            report = report_node(result)["final_report"]
            report["policy_reasoning"] += "\n[AWAITING ANALYST REVIEW]"
        else:
            report = result["final_report"]
            checkpointer.delete_thread(thread_id)

        if report is None:
            raise HTTPException(status_code=500, detail="graph produced no report")

        _log_prediction(request, start, report=report, thread_id=thread_id)
        return PredictionResponse(
            transaction_id=report["transaction_id"],
            fraud_probability=report["fraud_probability"],
            risk_level=report["risk_level"],
            decision=report["decision"],
            requires_human=report["requires_human"],
            explanation=report["explanation"] or "",
            policy_reasoning=report["policy_reasoning"] or "",
            shap_top_features=report["shap_top_features"],
            errors=report["errors"],
        )

    except HTTPException as e:
        _log_prediction(request, start, error=str(e.detail))
        raise

    except Exception as e:
        logger.exception("Prediction failed")
        _log_prediction(request, start, error=type(e).__name__)
        raise HTTPException(status_code=500, detail="Prediction failed.") from e


def _log_prediction(
    request: TransactionRequest, start: float, report=None, error=None, thread_id=None
):
    """Record one /predict call in the monitor and the service log."""
    fields = {
        "transaction_id": request.transaction_id,
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        "model_version": MODEL_VERSION,
        "threshold": bundle["threshold"],
        "error": error,
    }
    if report is not None:
        fields.update(
            fraud_probability=report["fraud_probability"],
            decision=report["decision"],
            requires_human=report["requires_human"],
            thread_id=thread_id,
            features=request.features,
        )
    logger.info("prediction %s", json.dumps(monitor.record(**fields)))


@app.get("/stats")
def get_stats():
    """Live service stats: volume, latency, decision rates, drift, review queue, alerts."""
    return monitor.stats()


@app.get("/reviews")
def list_reviews():
    """Transactions waiting for an analyst decision."""
    return monitor.pending_reviews()


class ReviewAction(BaseModel):
    """An analyst's decision on a queued transaction."""

    action: Literal["approve", "deny"]
    note: str = Field(default="", max_length=1000)


@app.post("/reviews/{transaction_id}")
def resolve_review(transaction_id: str, review: ReviewAction):
    """
    Approve or deny a queued transaction. The paused graph run resumes with the
    analyst's decision and returns its final report.
    """
    try:
        case = monitor.pending(transaction_id)
    except KeyError as e:
        raise HTTPException(
            status_code=404, detail="No pending review for this ID."
        ) from e

    final_report = None
    config = {"configurable": {"thread_id": case["thread_id"]}}
    if case["thread_id"] and fraud_graph.get_state(config).next == ("human_review",):
        fraud_graph.update_state(
            config, {"analyst_decision": review.action, "analyst_note": review.note}
        )
        final_report = fraud_graph.invoke(None, config)["final_report"]
        checkpointer.delete_thread(case["thread_id"])

    # final_report is None when the checkpoint was lost in a restart; the
    # decision is still recorded.
    resolution = monitor.resolve(transaction_id, review.action, review.note)
    return {**resolution, "final_report": final_report}


@app.get("/metrics")
def get_metrics():
    """
    Retrieve basic model metadata and performance metrics.

    Returns:
        dict: The test-set metrics saved with the production model.
    """
    metrics = bundle["metrics"]
    return {
        "model": metrics["model_name"],
        "threshold": bundle["threshold"],
        "auc_roc": metrics["auc_roc"],
        "auc_pr": metrics["auc_pr"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
    }


class TextRequest(BaseModel):
    """Input schema for the news screen and the assistant."""

    text: str = Field(min_length=1, max_length=20_000)


@app.post("/news")
def screen_news(request: TextRequest):
    """Give a preliminary fake or real label for a news article, with uncertainty."""
    return news_classifier.screen(news_bundle, request.text)


@app.get("/news/metrics")
def news_metrics():
    """Return the held-out metrics saved with the news classifier."""
    return news_bundle["metrics"]


@app.post("/chat")
def chat(request: TextRequest):
    """Answer a fraud or fake-news question from cited sources."""
    return assistant.answer(request.text)
