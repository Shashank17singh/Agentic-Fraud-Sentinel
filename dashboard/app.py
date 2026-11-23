# pyrefly: ignore [missing-import]
import json
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = "https://agentic-fraud-sentinel.onrender.com"


st.set_page_config(page_title="Agentic-Fraud-Sentinel", layout="wide")

CUSTOM_CSS = """
<style>
/* Hallmark · checked: 2026-10-31 · genre: modern-minimal (inferred)
 * fixed: 20 tokens lifted, focus, motion and states added · structural: left for the owner, see the audit
 * built: the project · redesign: polish · v2.0.0 */
:root {
  --color-green-400: #00ff41;
  --color-green-400-a40: rgba(0,255,65,0.4);
  --color-black: #000000;
  --color-grey-100: #e0e0e0;
  --color-black-a00: rgba(0,0,0,0);
  --color-green-400-a50: rgba(0,255,65,0.5);
  --color-blue-300: #94a3b8;
  --color-blue-800: #0f172a;
  --color-green-400-a30: rgba(0,255,65,0.3);
  --color-green-400-2: #1aff58;
  --color-green-400-a60: rgba(0,255,65,0.6);
  --color-green-800: #0c130e;
  --color-grey-800: #1f1f1f;
  --font-sans: 'Fira Sans', sans-serif !important;
  --font-mono: 'Fira Code', monospace !important;
}


@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: var(--font-sans);
}

h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-mono);
    color: var(--color-green-400) !important;
    text-shadow: 0 0 5px var(--color-green-400-a40);
}

.stApp {
    background-color: var(--color-black);
    color: var(--color-grey-100);
}

[data-testid="stHeader"] {
    background-color: var(--color-black-a00) !important;
}

/* Metric styling */
[data-testid="stMetricValue"] {
    font-family: var(--font-mono);
    color: var(--color-green-400) !important;
    font-weight: 700;
    text-shadow: 0 0 8px var(--color-green-400-a50);
}

[data-testid="stMetricLabel"] {
    font-family: var(--font-mono);
    color: var(--color-blue-300) !important;
}

[data-testid="stMetricDelta"] {
    font-family: var(--font-mono);
}

/* Buttons */
.stButton > button {
    background-color: var(--color-green-400);
    color: var(--color-blue-800);
    font-family: var(--font-mono);
    font-weight: 600;
    border: none;
    border-radius: 2px;
    transition: color 0.2s ease-in-out, background-color 0.2s ease-in-out, border-color 0.2s ease-in-out, box-shadow 0.2s ease-in-out, transform 0.2s ease-in-out, opacity 0.2s ease-in-out;
    box-shadow: 0 0 10px var(--color-green-400-a30);
}

.stButton > button:hover {
    background-color: var(--color-green-400-2);
    color: var(--color-blue-800);
    box-shadow: 0 0 15px var(--color-green-400-a60);
    transform: translateY(-1px);
}

/* Inputs */
.stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div > div {
    background-color: var(--color-green-800);
    color: var(--color-green-400);
    border: 1px solid var(--color-grey-800);
    font-family: var(--font-mono);
    border-radius: 2px;
}
.stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus {
    box-shadow: 0 0 5px var(--color-green-400-a30);
}

hr {
    border-color: var(--color-grey-800) !important;
}

.stDataFrame, .stTable {
    font-family: var(--font-mono);
}

/* Accessibility states added by the polish pass. */
:root {
  --focus-ring: #e2e8f0;
  --disabled-fill: #2a2f3a;
  --disabled-ink: #8b93a1;
}

[hidden] {
  display: none !important;
}

html,
body {
  overflow-x: clip;
}

a:focus-visible,
button:focus-visible,
input:focus-visible,
select:focus-visible,
textarea:focus-visible,
summary:focus-visible,
[tabindex]:focus-visible,
[role="button"]:focus-visible,
input:focus-visible + label {
  outline: 2px solid var(--focus-ring);
  outline-offset: 2px;
}

button:active:not(:disabled) {
  transform: translateY(1px);
}

button:disabled,
input:disabled,
select:disabled,
textarea:disabled {
  background: var(--disabled-fill);
  color: var(--disabled-ink);
  cursor: not-allowed;
}

@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    scroll-behavior: auto !important;
    transition-duration: 0.01ms !important;
  }
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

if not df.empty:
    # The last run is the production model; compare it with the latest baseline run.
    prod = df.iloc[-1]
    baselines = df[df["model_name"] == "xgb_baseline"]
    base = baselines.iloc[-1] if not baselines.empty else prod

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="AUC-ROC",
            value=f"{prod['auc_roc']:.4f}",
            delta=f"{prod['auc_roc'] - base['auc_roc']:+.4f} vs baseline",
        )
    with col2:
        st.metric(
            label="AUC-PR",
            value=f"{prod['auc_pr']:.4f}",
            delta=f"{prod['auc_pr'] - base['auc_pr']:+.4f} vs baseline",
        )
    with col3:
        st.metric(
            label="Recall",
            value=f"{prod['recall']:.2%}",
            delta=f"{(prod['recall'] - base['recall']) * 100:+.2f} pts vs baseline",
        )
    with col4:
        st.metric(
            label="Threshold",
            value=f"{prod['threshold']:.4f}",
            delta="chosen on validation",
        )


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
        font={"family": "Fira Code", "color": "#E0E0E0"},
        barmode="group",
        title={"text": "Metrics across model versions", "font": {"color": "#00FF41"}},
        yaxis={"range": [0, 1], "gridcolor": "#1F1F1F"},
        xaxis={"gridcolor": "#1F1F1F"},
        height=400,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
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
        "Transaction amount", min_value=0.0, max_value=5000.0, value=50.0, step=10.0
    )
    card1 = st.number_input("Card1", value=1000.0)


with col2:
    c13 = st.number_input("c13", value=1.0)
    c14 = st.number_input("c14", value=1.0)
    v258 = st.number_input("v258", value=0.0)
    v317 = st.number_input("v317", value=0.0)


if st.button("Run fraud check", type="primary"):
    features = {
        "TransactionAmt": transaction_amt,
        "card1": card1,
        "C13": c13,
        "C14": c14,
        "V258": v258,
        "V317": v317,
    }

    payload = {"transaction_id": transaction_id, "features": features}

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()

            st.markdown(f"### Decision: `{result['decision'].upper()}`")

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
st.subheader("News Screening")
st.caption(
    "Preliminary fake or real label from writing style (TF-IDF and logistic "
    "regression on ISOT). It does not check facts."
)
article = st.text_area("Paste a news article (at least 30 words)", height=160)
if st.button("Screen article") and article.strip():
    try:
        response = requests.post(f"{API_URL}/news", json={"text": article}, timeout=30)
        if response.status_code == 200:
            result = response.json()
            col1, col2 = st.columns(2)
            col1.metric("Label", result["label"].upper())
            prob = result["fake_probability"]
            col2.metric("Fake probability", "n/a" if prob is None else f"{prob:.2f}")
            st.info(result["note"])
        else:
            st.error(f"API error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API.")

st.divider()
st.subheader("Ask the Assistant")
st.caption(
    "Answers questions on card fraud and fake news from a small set of cited "
    "sources, and asks you to rephrase when none of them fit."
)
question = st.text_input(
    "Your question", placeholder="What should I do if my card is stolen?"
)
if st.button("Ask") and question.strip():
    try:
        response = requests.post(f"{API_URL}/chat", json={"text": question}, timeout=30)
        if response.status_code == 200:
            reply = response.json()
            st.write(reply["answer"])
            for source in reply["sources"]:
                st.markdown(f"[{source['n']}] [{source['title']}]({source['url']})")
        else:
            st.error(f"API error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API.")

st.divider()
st.subheader("Live Monitoring")
st.caption(
    "Recent /predict traffic on the API. Counts reset when the API restarts, so "
    "they cover only requests since the last deploy."
)
try:
    stats = requests.get(f"{API_URL}/stats", timeout=30).json()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Requests", stats["requests_total"])
    col2.metric("Error rate", f"{stats['error_rate']:.1%}")
    p95 = stats["latency_ms"]["p95"]
    col3.metric("p95 latency", "n/a" if p95 is None else f"{p95:.0f} ms")
    col4.metric("Awaiting review", stats["review_queue"]["pending"])
    drift = stats["drift_psi"]
    col5.metric("Max drift (PSI)", f"{max(drift.values()):.3f}" if drift else "n/a")
    if drift:
        st.caption("Drift by field (PSI, above 0.2 is a large shift)")
        st.bar_chart(pd.Series(drift, name="psi"))
    if stats["decision_rates"]:
        rates = pd.DataFrame(
            {
                "decision": list(stats["decision_rates"]),
                "share": list(stats["decision_rates"].values()),
            }
        )
        # Same bar chart, with labels kept upright so "flag" reads normally.
        st.altair_chart(
            alt.Chart(rates)
            .mark_bar()
            .encode(x=alt.X("decision:N", axis=alt.Axis(labelAngle=0)), y="share:Q"),
            width="stretch",
        )
    perf = stats.get("labelled_performance") or {}
    if perf.get("labelled"):
        st.caption(
            f"Model performance on {perf['labelled']} predictions with a known "
            f"outcome ({perf['fraud']} fraud), at the model threshold"
        )
        p1, p2, p3 = st.columns(3)
        for col, label, key in (
            (p1, "Precision", "precision"),
            (p2, "Recall", "recall"),
            (p3, "False-positive rate", "false_positive_rate"),
        ):
            col.metric(label, "n/a" if perf[key] is None else f"{perf[key]:.1%}")
    for alert in stats["alerts"]:
        st.warning(alert)
    if not stats["alerts"]:
        st.success("No alerts.")

    pending = requests.get(f"{API_URL}/reviews", timeout=30).json()
    st.markdown("**Review queue**")
    if pending:
        st.dataframe(pd.DataFrame(pending), use_container_width=True)
        case = st.selectbox("Transaction", [r["transaction_id"] for r in pending])
        note = st.text_input("Analyst note")
        approve, deny = st.columns(2)
        for column, action in ((approve, "approve"), (deny, "deny")):
            if column.button(action.capitalize(), key=f"review-{action}"):
                requests.post(
                    f"{API_URL}/reviews/{case}",
                    json={"action": action, "note": note},
                    timeout=30,
                )
                st.rerun()
    else:
        st.info("No transactions waiting for review.")
except (requests.exceptions.RequestException, KeyError, ValueError):
    st.error("Monitoring stats are not available from the API yet.")
