from typing import TypedDict


class FraudDetectionState(TypedDict):
    """
    Shared state passed between every node in the graph. Every agent reads from
    this and writes back to it. TypedDict gives us type safety without a full dataclass.
    """

    transaction_id: str
    transaction_data: dict

    fraud_probability: float | None
    risk_level: str | None

    shap_probability: dict | None
    explanation_text: str | None

    decision: str | None
    policy_reasoning: str | None

    requires_human: bool | None

    final_report: dict | None

    processing_errors: list[str] | None
