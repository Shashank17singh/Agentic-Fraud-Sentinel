from src.agents import nodes


def _state(transaction_data: dict, risk_level: str = "high") -> dict:
    return {
        "transaction_id": "t1",
        "transaction_data": transaction_data,
        "fraud_probability": 0.9,
        "risk_level": risk_level,
    }


def test_sparse_transactions_go_to_review_instead_of_auto_deny() -> None:
    result = nodes.policy_node(_state({"TransactionAmt": 25.0, "P_emaildomain": "x"}))

    assert result["decision"] == "flag"
    assert result["requires_human"] is True
    # Both keys are model columns; the email domain is now encoded, not dropped.
    assert f"2 of {len(nodes.FEATURE_COLS)}" in result["policy_reasoning"]


def test_complete_transactions_follow_the_risk_rules() -> None:
    complete = dict.fromkeys(nodes.FEATURE_COLS, 1.0)

    assert nodes.policy_node(_state(complete, "high"))["decision"] == "deny"
    assert nodes.policy_node(_state(complete, "low"))["decision"] == "approve"
