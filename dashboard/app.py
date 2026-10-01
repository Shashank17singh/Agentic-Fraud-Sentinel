# pyrefly: ignore [missing-import]
import json
import os

import pandas as pd
import streamlit as st

API_URL = "https://agentic-fraud-sentinel.onrender.com"


st.set_page_config(page_title="Agentic-Fraud-Sentinel", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Fira Sans', sans-serif !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Fira Code', monospace !important;
    color: #00FF41 !important;
    text-shadow: 0 0 5px rgba(0, 255, 65, 0.4);
    text-transform: uppercase;
}

.stApp {
    background-color: #000000;
    color: #E0E0E0;
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0) !important;
}

/* Metric styling */
[data-testid="stMetricValue"] {
    font-family: 'Fira Code', monospace !important;
    color: #00FF41 !important;
    font-weight: 700;
    text-shadow: 0 0 8px rgba(0, 255, 65, 0.5);
}

[data-testid="stMetricLabel"] {
    font-family: 'Fira Code', monospace !important;
    color: #94A3B8 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

[data-testid="stMetricDelta"] {
    font-family: 'Fira Code', monospace !important;
}

/* Buttons */
.stButton > button {
    background-color: #00FF41;
    color: #0F172A;
    font-family: 'Fira Code', monospace;
    font-weight: 600;
    border: none;
    border-radius: 2px;
    text-transform: uppercase;
    transition: all 0.2s ease-in-out;
    box-shadow: 0 0 10px rgba(0, 255, 65, 0.3);
}

.stButton > button:hover {
    background-color: #1aff58;
    color: #0F172A;
    box-shadow: 0 0 15px rgba(0, 255, 65, 0.6);
    transform: translateY(-1px);
}

/* Inputs */
.stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div > div {
    background-color: #0C130E;
    color: #00FF41;
    border: 1px solid #1F1F1F;
    font-family: 'Fira Code', monospace;
    border-radius: 2px;
}
.stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus {
    border-color: #00FF41;
    box-shadow: 0 0 5px rgba(0,255,65,0.3);
}

hr {
    border-color: #1F1F1F !important;
}

.stDataFrame, .stTable {
    font-family: 'Fira Code', monospace !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("Agentic-Fraud-Sentinel Dashboard")
st.markdown("Real-time monitoring of the multi-agent fraud detection system")
st.divider()

LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "notebooks",
    "data",
    "model_results",
    "model_log.json",
)


@st.cache_data(ttl=30)
def load_log():
    """
    Load the model evaluation logs from JSON.

    Data is cached for 30 seconds for performance.

    Returns:
        pd.DataFrame: The logged model metrics.
    """
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()
    with open(LOG_PATH, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)


df = load_log()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="AUC-ROC", value="0.9070", delta="+0.0058 vs baseline")

with col2:
    st.metric(label="AUC-PR", value="0.5758", delta="0.0512 vs baseline")

with col3:
    st.metric(label="Recall", value="48.45%", delta="+16.92 vs baseline")

with col4:
    st.metric(label="Threshold", value="0.2161", delta="optimized from 0.5")


st.divider()
st.subheader("Model Version History")

if df.empty:
    st.warning("No model runs logged yet.")
else:
    display_cols = {
        "model_name": "Model",
        "auc_roc": "AUC-ROC",
        "auc_pr": "AUC-PR",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "false_positive_rate": "False Positive Rate",
        "timestamp": "Timestamp",
    }

    display_df = df[[c for c in display_cols if c in df.columns]].rename(
        columns=display_cols
    )

    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Model Metrics Comparison")

if not df.empty:
    # pyrefly: ignore [missing-import]
    import plotly.graph_objects as go

    fig = go.Figure()

    metrics = ["auc_roc", "auc_pr", "precision", "recall", "f1"]
    labels = ["AUC-ROC", "AUC-PR", "Precision", "Recall", "F1"]

    for _, row in df.iterrows():
        fig.add_trace(
            go.Bar(
                name=row["model_name"],
                x=labels,
                y=[row.get(m, 0) for m in metrics],
            )
        )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0C130E",
        paper_bgcolor="#000000",
        font=dict(family="Fira Code", color="#E0E0E0"),
        barmode="group",
        title=dict(text="Metrics across model versions", font=dict(color="#00FF41")),
        yaxis=dict(range=[0, 1], gridcolor="#1F1F1F"),
        xaxis=dict(gridcolor="#1F1F1F"),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    colors = ["#00FF41", "#00BFFF", "#FFD700", "#FF3333", "#FF00FF"]
    for i, trace in enumerate(fig.data):
        trace.marker.color = colors[i % len(colors)]

    st.plotly_chart(fig, use_container_width=True)


st.divider()
st.subheader("Live Transaction Tester")
st.markdown("Send a real transaction through the fraud detection pipeline")


col1, col2 = st.columns(2)

with col1:
    transaction_id = st.text_input("Transaction ID", value="TX_TEST_001")
    transaction_amt = st.number_input(
        "Transaction amount ", min_value=0.0, max_value=100.0, step=10.0
    )
    card1 = st.number_input("Card1", value=1000.0)
    p_email = st.selectbox(
        "P_emaildomain", ["gmail.com", "yahoo.com", "hotmail.com", "unknown"]
    )


with col2:
    c13 = st.number_input("c13", value=1.0)
    c14 = st.number_input("c14", value=1.0)
    v258 = st.number_input("v258", value=0.0)
    v317 = st.number_input("v317", value=0.0)


if st.button("Run fraud check", type="primary"):
    import requests

    features = {
        "TransactionAmt": transaction_amt,
        "card1": card1,
        "C13": c13,
        "C14": c14,
        "V258": v258,
        "V317": v317,
        "P_emaildomain": p_email,
    }

    payload = {"transaction_id": transaction_id, "features": features}

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()

            decision_color = {"approve": "✅", "flag": "⚠️", "deny": "🚨"}.get(
                result["decision"], "❓"
            )

            st.markdown(
                f"### {decision_color} Decision: `{result['decision'].upper()}`"
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fraud Probability", f"{result['fraud_probability']:.4f}")
            with col2:
                st.metric("Risk Level", result["risk_level"].upper())
            with col3:
                st.metric(
                    "Requires Human Review", "Yes" if result["requires_human"] else "No"
                )

            st.markdown("**Policy Reasoning:**")
            st.info(result["policy_reasoning"])

            st.markdown("**SHAP Explanation:**")
            st.code(result["explanation"])

        else:
            st.error(
                f"API error: {response.status_code} — "
                f"{response.json().get('detail', 'unknown error')}"
            )

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure uvicorn is running on port 8000.")


st.divider()
st.subheader("metric deep dive")

if not df.empty:
    selected_model = st.selectbox(
        "Select model version to inspect", options=df["model_name"].tolist()
    )

    row = df[df["model_name"] == selected_model].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Confusion Matrix Breakdown**")
        cm_data = {
            "": ["Predicted Legit", "Predicted Fraud"],
            "Actual Legit": [
                int(row.get("true_negatives", 0)),
                int(row.get("false_positives", 0)),
            ],
            "Actual Fraud": [
                int(row.get("false_negatives", 0)),
                int(row.get("true_positives", 0)),
            ],
        }
        st.dataframe(pd.DataFrame(cm_data).set_index(""), use_container_width=True)

    with col2:
        st.markdown("**What these numbers mean**")
        st.markdown(f"""
        - **True Negatives** `{int(row.get("true_negatives", 0)):,}` — 
          legitimate transactions correctly approved
        - **False Positives** `{int(row.get("false_positives", 0)):,}` — 
          legitimate transactions wrongly flagged
        - **False Negatives** `{int(row.get("false_negatives", 0)):,}` — 
          fraud cases missed
        - **True Positives** `{int(row.get("true_positives", 0)):,}` — 
          fraud cases correctly caught
        """)

    st.markdown("**Precision vs Recall Tradeoff**")
    st.markdown(f"""
    At threshold `{row.get("threshold", 0.5)}`:
    - For every 100 transactions flagged as fraud, 
      **{row.get("precision", 0) * 100:.1f}** are actually fraud
    - Of all actual fraud in the dataset, 
      **{row.get("recall", 0) * 100:.1f}%** were caught
    - False positive rate of **{row.get("false_positive_rate", 0) * 100:.2f}%** 
      means only **{row.get("false_positive_rate", 0) * 100:.2f}** 
      in every 100 legitimate transactions are wrongly blocked
    """)


st.divider()
