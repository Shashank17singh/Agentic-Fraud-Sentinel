# pyrefly: ignore [missing-import]
from langgraph.graph import END, StateGraph

from src.agents.nodes import (auto_approve_node, explainer_node,
                              human_review_node, policy_node, report_node,
                              risk_scorer_node)
from src.agents.state import FraudDetectionState


def route_after_policy(state: FraudDetectionState) -> str:
    """
    Route the graph to human review or auto-approve based on policy decision.

    Args:
        state (FraudDetectionState): The current state of the transaction.

    Returns:
        str: The next node identifier.
    """
    if state["requires_human"]:
        return "go_human_review"
    else:
        return "go_auto_approve"


def build_fraud_graph():
    """
    Construct and compile the LangGraph for fraud detection.

    Returns:
        Graph: The compiled state graph.
    """
    graph = StateGraph(FraudDetectionState)

    graph.add_node("risk_scorer", risk_scorer_node)
    graph.add_node("explainer", explainer_node)
    graph.add_node("policy", policy_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("auto_approve", auto_approve_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("risk_scorer")

    graph.add_edge("risk_scorer", "explainer")
    graph.add_edge("explainer", "policy")

    graph.add_conditional_edges(
        "policy",
        route_after_policy,
        {
            "go_human_review": "human_review",
            "go_auto_approve": "auto_approve",
        },
    )

    graph.add_edge("human_review", "report")
    graph.add_edge("auto_approve", "report")

    graph.add_edge("report", END)

    app = graph.compile()
    print("Graph compiled successfully")
    return app
