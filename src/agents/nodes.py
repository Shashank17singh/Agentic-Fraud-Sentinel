import os
from datetime import datetime, timezone

import joblib
import pandas as pd

from src.agents.state import FraudDetectionState
from src.data.features import prepare_transaction
from src.explainability.shap_explainer import build_explainer, explain_single_prediction

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if os.path.exists(os.path.join(BASE_DIR, "data", "models", "xgboost_production.pkl")):
    DATA_DIR = os.path.join(BASE_DIR, "data")
else:
    DATA_DIR = os.path.join(BASE_DIR, "notebooks", "data")


MODEL_PATH = os.path.join(DATA_DIR, "models", "xgboost_production.pkl")


print("Loading model bundle...")
bundle = joblib.load(MODEL_PATH)
MODEL = bundle["model"]
THRESHOLD = bundle["threshold"]
# Frequency maps, medians and category codes fitted on train (see
# src/data/preprocessing.inference_artifacts). Older bundles lack them, and
# then gaps fall back to 0.
PREPROCESSING = bundle.get("preprocessing", {})

print("Loading SHAP explainer...")
# Built from the model at startup rather than unpickled: the tree-path explainer
# needs no background data, and a pickle breaks across Python versions.
EXPLAINER = build_explainer(MODEL, None)


FEATURE_COLS = MODEL.get_booster().feature_names

# Policy rules live here, apart from training, and can be changed per deployment
# through environment variables without retraining.
# Below this share of numeric model features the score rests mostly on filled-in
# zeros, so the policy sends the case to a human instead of deciding.
MIN_FEATURE_COVERAGE = float(os.environ.get("POLICY_MIN_FEATURE_COVERAGE", "0.5"))
# Scores from this fraction of the threshold up to the threshold are "medium" risk.
MEDIUM_RISK_FRACTION = float(os.environ.get("POLICY_MEDIUM_RISK_FRACTION", "0.5"))


def _supplied_feature_count(tx_data: dict) -> int:
    """
    Count model features the caller sent. A null counts: it says the value is
    missing, which training also saw and filled with the median.
    """
    return sum(col in tx_data for col in FEATURE_COLS)


def _encode_features(tx_data: dict) -> dict:
    return prepare_transaction(tx_data, PREPROCESSING, FEATURE_COLS)


def risk_scorer_node(state: FraudDetectionState) -> dict:
    """
    Loads transaction features, run Xgboost model, returns fraud probability and a risk tier.

    Risk tiers:
    low < threshold* 0.5 -> almost certainly legitimate
    medium threshold * 0.5 to threshold -> borderline
    high >= threshold -> model says fraud
    """

    try:
        tx_data = _encode_features(state["transaction_data"])
        X = pd.DataFrame([tx_data], columns=FEATURE_COLS)
        X = X.reindex(columns=FEATURE_COLS, fill_value=0)

        fraud_prob = float(MODEL.predict_proba(X)[:, 1][0])

        low_cutoff = THRESHOLD * MEDIUM_RISK_FRACTION
        if fraud_prob >= THRESHOLD:
            risk_level = "high"
        elif fraud_prob >= low_cutoff:
            risk_level = "medium"
        else:
            risk_level = "low"

        print(
            f"[RiskScorer] tx={state['transaction_id']} "
            f"prob={fraud_prob:.4f} risk={risk_level}"
        )

        return {
            "fraud_probability": fraud_prob,
            "risk_level": risk_level,
        }

    except Exception as e:  # noqa: BLE001
        return {"processing_errors": [f"RiskScorer error: {e!s}"]}


def explainer_node(state: FraudDetectionState) -> dict:
    """Computes SHAP values for this transaction and converts them into a human readable
    explanation string. Only runs if risk level is med or high.

    """
    try:
        tx_data = _encode_features(state["transaction_data"])
        X = pd.DataFrame([tx_data], columns=FEATURE_COLS)
        X = X.reindex(columns=FEATURE_COLS, fill_value=0)

        explanation = explain_single_prediction(EXPLAINER, X, top_n=5)

        print(f"[Explainer DEBUG] explanation keys: {explanation.keys()}")
        print(f"[Explainer DEBUG] first feature item: {explanation['top_features'][0]}")

        top = explanation["top_features"]
        lines = []
        for feat in top:
            direction = "increased" if feat["shap_value"] > 0 else "decreased"
            lines.append(
                f"- {feat['feature']} (value: {feat['actual_value']}) "
                f"{direction} fraud score by {abs(feat['shap_value']):.3f}"
            )

        explanation_text = (
            f"Fraud probability: {state['fraud_probability']:.4f}\n"
            f"Key factors:\n" + "\n".join(lines)
        )

        print(
            f"[Explainer] tx={state['transaction_id']} top_feature={top[0]['feature']}"
        )

        return {
            "explanation_text": explanation_text,
            "shap_explanation": explanation,
        }

    except Exception as e:  # noqa: BLE001
        print(f"[Explainer] ERROR: {e!s}")
        import traceback

        traceback.print_exc()
        return {
            "shap_explanation": None,
            "explanation_text": f"Explanation unavailable: {e!s}",
            "processing_errors": [f"Explainer error: {e!s}"],
        }


def policy_node(state: FraudDetectionState) -> dict:
    """Applies business rules on top of the model score.
    This is deliberately rule based , not ML policy decisions
    need to be auditable and adjustable without retraining.

    Rules:
    high risk -> deny+ flag for human review
    medium risk -> flag for human review , dont auto deny
    low risk -> approve automatically
    too few features supplied -> flag for human review, whatever the score
    """

    risk_level = state["risk_level"]
    fraud_prob = state["fraud_probability"]
    supplied = _supplied_feature_count(state["transaction_data"])

    if supplied < MIN_FEATURE_COVERAGE * len(FEATURE_COLS):
        decision = "flag"
        requires_human = True
        reasoning = (
            f"Only {supplied} of {len(FEATURE_COLS)} model features were supplied, "
            f"so the fraud probability {fraud_prob:.4f} is unreliable. "
            f"Human review required."
        )

    elif risk_level == "high":
        decision = "deny"
        requires_human = True
        reasoning = (
            f"Transaction denied. Fraud probability {fraud_prob:.4f} "
            f"exceeds threshold {THRESHOLD:.4f}. Flagged for human review."
        )

    elif risk_level == "medium":
        decision = "flag"
        requires_human = True
        reasoning = (
            f"Transaction flagged for review. Fraud probability "
            f"{fraud_prob:.4f} is in borderline range. "
            f"Human review required before processing."
        )

    else:
        decision = "approve"
        requires_human = False
        reasoning = (
            f"Transaction approved. Fraud probability {fraud_prob:.4f} "
            f"is below threshold. No action required."
        )

    print(
        f"[Policy] tx={state['transaction_id']} "
        f"decision={decision} human={requires_human}"
    )

    return {
        "decision": decision,
        "requires_human": requires_human,
        "policy_reasoning": reasoning,
    }


def human_review_node(state: FraudDetectionState) -> dict:
    """
    Apply the analyst's decision. When the graph is compiled with a checkpointer
    it pauses before this node, and the API resumes it once an analyst has
    approved or denied the case. Without one (notebooks, tests) the case is only
    noted as queued.
    """
    action = state.get("analyst_decision")
    if action is None:
        print(f"[HumanReview] tx={state['transaction_id']} queued for analyst review")
        return {
            "policy_reasoning": state["policy_reasoning"]
            + "\n[QUEUED FOR HUMAN REVIEW]"
        }

    print(f"[HumanReview] tx={state['transaction_id']} analyst decided {action}")
    note = state.get("analyst_note") or ""
    return {
        "decision": action,
        "policy_reasoning": f"{state['policy_reasoning']}\n[ANALYST {action.upper()}] {note}".rstrip(),
    }


def auto_approve_node(state: FraudDetectionState) -> dict:
    """
    Low risk transactions skip human review entirely.
    Logs the auto approval decision.
    """

    print(f"[Auto Approve] tx = {state['transaction_id']}auto-approved")

    return {"policy_reasoning": state["policy_reasoning"] + "\n[AUTO_APPROVED]"}


def report_node(state: FraudDetectionState) -> dict:
    """
    Assembles the complete audit trail for this transaction. Every field from every agent ends
    up here. This is what gets returned to the API caller and logged to the monitoring dashboard.
    """

    print(
        f"[REPORT] tx = {state['transaction_id']}"
        f"decision= {state.get('decision')} complete"
    )

    shap_features = []

    if state.get("shap_explanation"):
        shap_features = state["shap_explanation"].get("top_features", [])

    report = {
        "transaction_id": state["transaction_id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fraud_probability": state["fraud_probability"],
        "risk_level": state["risk_level"],
        "decision": state["decision"],
        "requires_human": state["requires_human"],
        "policy_reasoning": state["policy_reasoning"],
        "explanation": state["explanation_text"],
        "shap_top_features": shap_features,
        "errors": state.get("processing_errors", []),
    }

    print(f"[REPORT] tx={state['transaction_id']}decision={state['decision']} complete")

    return {"final_report": report}
