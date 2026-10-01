# pyrefly: ignore [missing-import]
import os
import sys
from typing import Any, List, Optional

import joblib
import pandas as pd
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
from numpy.random import logistic
# pyrefly: ignore [missing-import]
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if os.path.exists(os.path.join(BASE_DIR, "data", "models", "xgboost_production.pkl")):
    DATA_DIR = os.path.join(BASE_DIR, "data")
else:
    DATA_DIR = os.path.join(BASE_DIR, "notebooks", "data")





from src.agents.graph import build_fraud_graph
app = FastAPI(
    title="Agentic-Fraud-Sentinel API",
    description="Multi agent fraud detection - XGBoost and LangGraph",
    version="1.0.0",
)

print("Building fraud detection graph..")
fraud_graph = build_fraud_graph()
print("Graph ready...")
bundle = joblib.load(os.path.join(DATA_DIR, "models", "xgboost_production.pkl"))
FEATURE_COLS = bundle["model"].get_booster().feature_names


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
    shap_top_features: List[Any]
    errors: List[Any]

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
    try:

        feature_row = {col: request.features.get(col, 0) for col in FEATURE_COLS}

        initial_state = {
            "transaction_id": request.transaction_id,
            "transaction_data": feature_row,
            "fraud_probability": None,
            "risk_level": None,
            "shap_explanation": None,
            "explanation_text": None,
            "decision": None,
            "policy_reasoning": None,
            "requires_human": None,
            "final_report": None,
            "processing_errors": [],
        }

        result = fraud_graph.invoke(initial_state)
        report = result["final_report"]

        if report is None:
            raise HTTPException(status_code=500, detail="graph produced no report")

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

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Predcition failed: {str(e)}")


app.get("/metrics")


def get_metrics():
    """
    Retrieve basic model metadata and performance metrics.
    
    Returns:
        dict: Pre-configured model evaluation metrics.
    """

    return {
        "model": "xgboost_tuned",
        "threshold": 0.2161,
        "auc_roc": 0.9070,
        "auc_pr": 0.5758,
        "precision": 0.8335,
        "recall": 0.3942,
    }
