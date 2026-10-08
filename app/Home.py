import sys
from pathlib import Path
from textwrap import dedent

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# PROJECT SETUP
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from src.loaders import (
    load_configuration,
    load_final_summary,
    load_validation_comparison,
    load_conformal_results,
    load_shap_global
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Trustworthy Fraud AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# HTML HELPER
# =========================================================

def html(content):
    cleaned_html = "".join(
        line.strip()
        for line in dedent(content).splitlines()
        if line.strip()
    )

    st.markdown(
        cleaned_html,
        unsafe_allow_html=True
    )


# =========================================================
# DESIGN SYSTEM
# =========================================================

html("""
<style>

html, body, [class*="css"] {
    font-family:
        "Segoe UI",
        -apple-system,
        BlinkMacSystemFont,
        "Helvetica Neue",
        Arial,
        sans-serif;
}

.stApp {
    background: #FAF8F3;
    color: #1F1A17;
}

.block-container {
    max-width: 1480px;
    padding-top: 1.75rem;
    padding-left: 2.25rem;
    padding-right: 2.25rem;
    padding-bottom: 3rem;
}


/* =========================================================
   SIDEBAR
========================================================= */

[data-testid="stSidebar"] {
    background: #2B2118;
    border-right: 1px solid #4A382A;
}

[data-testid="stSidebar"] * {
    color: #F0E6D7;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFF8EC;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #D8C7A8;
}

[data-testid="stSidebar"] hr {
    border-color: #5A4635;
}


/* =========================================================
   HERO
========================================================= */

.hero {
    background:
        linear-gradient(
            135deg,
            #3A2A1F 0%,
            #2B2118 62%,
            #241B15 100%
        );

    border: 1px solid #5A4635;
    padding: 2.65rem 2.8rem;
    border-radius: 18px;
    margin-bottom: 1.7rem;

    box-shadow:
        0 2px 4px rgba(43, 33, 24, 0.08),
        0 12px 32px rgba(43, 33, 24, 0.12);
}

.hero-eyebrow {
    color: #E7D3A8;
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    margin-bottom: 0.9rem;
}

.hero-title {
    color: #FFF8EC;
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1.12;
    letter-spacing: -0.035em;
    margin-bottom: 0.85rem;
}

.hero-subtitle {
    color: #F0E6D7;
    font-size: 1rem;
    line-height: 1.68;
    max-width: 880px;
}

.hero-tech {
    margin-top: 1.2rem;
    color: #D8C7A8;
    font-size: 0.78rem;
    letter-spacing: 0.02em;
}


/* =========================================================
   KPI CARDS
========================================================= */

.metric-card {
    background: #FFFDFC;
    border: 1px solid #E4D8C8;
    border-radius: 12px;
    padding: 1.15rem 1.2rem;
    min-height: 126px;

    box-shadow:
        0 1px 2px rgba(43,33,24,0.03),
        0 4px 12px rgba(43,33,24,0.04);
}

.metric-label {
    color: #8A6A3B;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}

.metric-value {
    color: #2B2118;
    font-size: 1.95rem;
    font-weight: 700;
    letter-spacing: -0.035em;
    margin-top: 0.35rem;
}

.metric-note {
    color: #6B625B;
    font-size: 0.75rem;
    line-height: 1.4;
    margin-top: 0.28rem;
}


/* =========================================================
   SECTION HEADERS
========================================================= */

.section-header {
    margin-top: 2.7rem;
    margin-bottom: 1rem;
}

.section-title {
    color: #2B2118;
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: -0.018em;
}

.section-subtitle {
    color: #6B625B;
    font-size: 0.87rem;
    margin-top: 0.27rem;
    line-height: 1.55;
}


/* =========================================================
   RESULT PANEL
========================================================= */

.result-panel {
    background: #FFFDFC;
    border: 1px solid #E4D8C8;
    border-radius: 14px;
    padding: 1.4rem 1.5rem;

    box-shadow:
        0 2px 8px rgba(43,33,24,0.035);
}

.result-panel-title {
    color: #2B2118;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 1rem;
}

.result-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.63rem 0;
    border-bottom: 1px solid #EFE6DA;
}

.result-row:last-child {
    border-bottom: none;
}

.result-name {
    color: #6B625B;
    font-size: 0.82rem;
}

.result-value {
    color: #2B2118;
    font-size: 0.88rem;
    font-weight: 700;
}


/* =========================================================
   CONFUSION MATRIX CARDS
========================================================= */

.cm-card {
    background: #FFFDFC;
    border: 1px solid #E4D8C8;
    border-radius: 12px;
    padding: 1rem 1.05rem;
    min-height: 105px;

    box-shadow:
        0 1px 5px rgba(43,33,24,0.025);
}

.cm-label {
    color: #8A6A3B;
    font-size: 0.70rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.055em;
}

.cm-value {
    color: #2B2118;
    font-size: 1.55rem;
    font-weight: 700;
    margin-top: 0.25rem;
}

.cm-description {
    color: #6B625B;
    font-size: 0.72rem;
    margin-top: 0.15rem;
    line-height: 1.35;
}


/* =========================================================
   DECISION CARDS
========================================================= */

.decision-card {
    background: #FFFDFC;
    border: 1px solid #E4D8C8;
    border-radius: 12px;
    padding: 1rem 1.15rem;
    min-height: 110px;

    box-shadow:
        0 1px 5px rgba(43,33,24,0.025);
}

.decision-name {
    color: #8A6A3B;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.decision-value {
    color: #2B2118;
    font-size: 1.65rem;
    font-weight: 700;
    margin-top: 0.3rem;
}

.decision-percent {
    color: #6B625B;
    font-size: 0.76rem;
    margin-top: 0.2rem;
}


/* =========================================================
   INSIGHT CARDS
========================================================= */

.insight-card {
    background: #FFFDFC;
    border: 1px solid #E4D8C8;
    border-radius: 12px;
    padding: 1.05rem 1.15rem;
    min-height: 122px;

    box-shadow:
        0 1px 5px rgba(43,33,24,0.03);
}

.insight-title {
    font-size: 0.70rem;
    font-weight: 700;
    color: #8A6A3B;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.insight-value {
    font-size: 1.48rem;
    font-weight: 700;
    color: #2B2118;
    margin-top: 0.32rem;
}

.insight-text {
    font-size: 0.75rem;
    color: #6B625B;
    line-height: 1.42;
    margin-top: 0.25rem;
}


/* =========================================================
   PLOTLY PANELS
========================================================= */

div[data-testid="stPlotlyChart"] {
    background: #FFFDFC;
    border: 1px solid #E4D8C8;
    border-radius: 14px;
    padding: 0.25rem;

    box-shadow:
        0 1px 2px rgba(43,33,24,0.02),
        0 3px 10px rgba(43,33,24,0.035);
}


/* =========================================================
   FOOTER
========================================================= */

.footer {
    margin-top: 2.6rem;
    padding-top: 1.1rem;
    border-top: 1px solid #E4D8C8;
    color: #8A8178;
    font-size: 0.74rem;
}

hr {
    border-color: #E4D8C8 !important;
}

/*================================*/
.app-footer {
    margin-top: 2.5rem;
    padding-top: 1.1rem;
    padding-bottom: 0.7rem;
    border-top: 1px solid #E4D8C8;
    text-align: center;
    color: #8A8178;
    font-size: 0.74rem;
    line-height: 1.7;
}

.footer-main {
    color: #6B625B;
    text-align: center;
}

.footer-contact {
    margin-top: 0.15rem;
    text-align: center;
    padding-left: 1.8rem;
}

.footer-contact a {
    color: #8A6A3B;
    text-decoration: none;
    font-weight: 600;
}

.footer-contact a:hover {
    color: #B89146;
    text-decoration: underline;
}

.contact-label {
    color: #6B625B;
    font-weight: 600;
}

.footer-separator {
    margin: 0 0.45rem;
    color: #B8ADA1;
}
/*================================*/

</style>
""")


# =========================================================
# LOAD DATA
# =========================================================

config = load_configuration()
summary = load_final_summary()
comparison = load_validation_comparison()
conformal = load_conformal_results()
shap_global = load_shap_global()

summary_dict = dict(
    zip(summary["Metric"], summary["Value"])
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## ◈ Trustworthy Fraud AI")

    st.caption(
        "Explainable and uncertainty-aware financial fraud detection"
    )

    st.divider()

    st.markdown("### Model")

    st.write(
        f"**Primary model**  \n{config['Primary_Model']}"
    )

    st.write(
        f"**Model variant**  \n{config['Model_Variant']}"
    )

    st.write(
        f"**Decision threshold**  \n"
        f"{config['Decision_Threshold']:.3f}"
    )

    st.write(
        f"**Conformal α**  \n"
        f"{config['Primary_Conformal_Alpha']:.2f}"
    )

    st.divider()

    st.markdown("### Technology")

    st.caption("XGBoost")
    st.caption("SHAP")
    st.caption("Conformal Prediction")
    st.caption("Python · Streamlit")

    st.divider()

    st.caption("Research prototype · Portfolio project")


# =========================================================
# HERO
# =========================================================

html("""
<div class="hero">

<div class="hero-eyebrow">
Applied Machine Learning Research
</div>

<div class="hero-title">
Trustworthy Financial Fraud Detection
</div>

<div class="hero-subtitle">
An explainable and uncertainty-aware decision-support system
for detecting fraudulent financial transactions while providing
interpretable predictions and escalation of uncertain cases
for human review.
</div>

<div class="hero-tech">
XGBoost &nbsp;·&nbsp;
SHAP &nbsp;·&nbsp;
Conformal Prediction &nbsp;·&nbsp;
Human-in-the-Loop Decision Support
</div>

</div>
""")


# =========================================================
# KPI ROW
# =========================================================

k1, k2, k3, k4, k5 = st.columns(5)

kpis = [
    (
        k1,
        "PR-AUC",
        summary_dict["Test PR-AUC"],
        "Minority-class discrimination"
    ),
    (
        k2,
        "Fraud Recall",
        summary_dict["Test Recall"],
        "Actual fraud successfully detected"
    ),
    (
        k3,
        "Precision",
        summary_dict["Test Precision"],
        "Predicted fraud confirmed"
    ),
    (
        k4,
        "F1 Score",
        summary_dict["Test F1"],
        "Precision–recall balance"
    ),
    (
        k5,
        "Human Review",
        summary_dict["Conformal Human Review Rate"],
        "Transactions requiring escalation"
    )
]

for col, label, value, note in kpis:

    with col:

        html(f"""
        <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value:.1%}</div>
        <div class="metric-note">{note}</div>
        </div>
        """)


# =========================================================
# FINAL TEST EVALUATION
# =========================================================

html("""
<div class="section-header">
<div class="section-title">
Final Test Evaluation
</div>
<div class="section-subtitle">
Performance of the selected XGBoost model on the independent
held-out test set using the validation-selected decision threshold.
</div>
</div>
""")


test_left, test_right = st.columns([1.1, 1])


with test_left:

    cm1, cm2 = st.columns(2)

    with cm1:

        html(f"""
        <div class="cm-card">
        <div class="cm-label">True Positives</div>
        <div class="cm-value">
        {int(summary_dict["True Positives"]):,}
        </div>
        <div class="cm-description">
        Fraud correctly identified
        </div>
        </div>
        """)

    with cm2:

        html(f"""
        <div class="cm-card">
        <div class="cm-label">False Negatives</div>
        <div class="cm-value">
        {int(summary_dict["False Negatives"]):,}
        </div>
        <div class="cm-description">
        Fraud cases missed
        </div>
        </div>
        """)

    html("<div style='height:12px'></div>")

    cm3, cm4 = st.columns(2)

    with cm3:

        html(f"""
        <div class="cm-card">
        <div class="cm-label">True Negatives</div>
        <div class="cm-value">
        {int(summary_dict["True Negatives"]):,}
        </div>
        <div class="cm-description">
        Legitimate transactions correctly cleared
        </div>
        </div>
        """)

    with cm4:

        html(f"""
        <div class="cm-card">
        <div class="cm-label">False Positives</div>
        <div class="cm-value">
        {int(summary_dict["False Positives"]):,}
        </div>
        <div class="cm-description">
        Legitimate transactions incorrectly flagged
        </div>
        </div>
        """)


with test_right:

    html(f"""
    <div class="result-panel">

    <div class="result-panel-title">
    Held-Out Test Performance
    </div>

    <div class="result-row">
    <span class="result-name">Accuracy</span>
    <span class="result-value">
    {summary_dict["Test Accuracy"]:.2%}
    </span>
    </div>

    <div class="result-row">
    <span class="result-name">Precision</span>
    <span class="result-value">
    {summary_dict["Test Precision"]:.2%}
    </span>
    </div>

    <div class="result-row">
    <span class="result-name">Recall</span>
    <span class="result-value">
    {summary_dict["Test Recall"]:.2%}
    </span>
    </div>

    <div class="result-row">
    <span class="result-name">F1 score</span>
    <span class="result-value">
    {summary_dict["Test F1"]:.2%}
    </span>
    </div>

    <div class="result-row">
    <span class="result-name">ROC-AUC</span>
    <span class="result-value">
    {summary_dict["Test ROC-AUC"]:.3f}
    </span>
    </div>

    <div class="result-row">
    <span class="result-name">PR-AUC</span>
    <span class="result-value">
    {summary_dict["Test PR-AUC"]:.3f}
    </span>
    </div>

    </div>
    """)


# =========================================================
# MODEL BENCHMARKING
# =========================================================

html("""
<div class="section-header">
<div class="section-title">
Model Benchmarking
</div>
<div class="section-subtitle">
Validation performance across linear, ensemble and gradient-boosting approaches.
</div>
</div>
""")


model_col, shap_col = st.columns([1.25, 1])


with model_col:

    performance_long = comparison.melt(
        id_vars="Model",
        value_vars=[
            "Precision",
            "Recall",
            "F1",
            "PR_AUC"
        ],
        var_name="Metric",
        value_name="Score"
    )

    metric_labels = {
        "Precision": "Precision",
        "Recall": "Recall",
        "F1": "F1",
        "PR_AUC": "PR-AUC"
    }

    performance_long["Metric"] = (
        performance_long["Metric"]
        .map(metric_labels)
    )

    fig_model = px.bar(
        performance_long,
        y="Model",
        x="Score",
        color="Metric",
        orientation="h",
        barmode="group",
        title="Validation Performance by Model",
        color_discrete_sequence=[
            "#B89146",
            "#2F7D6D",
            "#7B5E3B",
            "#8C8073"
        ]
    )

    fig_model.update_layout(
        height=470,
        template="plotly_white",

        margin=dict(
            l=25,
            r=25,
            t=60,
            b=25
        ),

        xaxis=dict(
            title="Score",
            range=[0.60, 1.00],
            tickformat=".0%",
            gridcolor="#EFE6DA",
            zeroline=False
        ),

        yaxis=dict(
            title="",
            autorange="reversed"
        ),

        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),

        plot_bgcolor="#FFFDFC",
        paper_bgcolor="#FFFDFC",

        font=dict(
            family="Segoe UI, Arial, sans-serif",
            color="#4A4039",
            size=12
        )
    )

    fig_model.update_traces(
        marker_line_width=0
    )

    st.plotly_chart(
        fig_model,
        use_container_width=True
    )


with shap_col:

    top_shap = (
        shap_global
        .head(10)
        .sort_values(
            "Mean_Absolute_SHAP",
            ascending=True
        )
    )

    fig_shap = px.bar(
        top_shap,
        x="Mean_Absolute_SHAP",
        y="Feature",
        orientation="h",
        title="Global Model Drivers"
    )

    fig_shap.update_traces(
        marker_color="#B89146",
        marker_line_width=0
    )

    fig_shap.update_layout(
        height=470,
        template="plotly_white",

        margin=dict(
            l=25,
            r=25,
            t=60,
            b=25
        ),

        xaxis=dict(
            title="Mean absolute SHAP value",
            gridcolor="#EFE6DA",
            zeroline=False
        ),

        yaxis=dict(
            title=""
        ),

        plot_bgcolor="#FFFDFC",
        paper_bgcolor="#FFFDFC",

        font=dict(
            family="Segoe UI, Arial, sans-serif",
            color="#4A4039",
            size=12
        )
    )

    st.plotly_chart(
        fig_shap,
        use_container_width=True
    )


# =========================================================
# UNCERTAINTY AND HUMAN REVIEW
# =========================================================

html("""
<div class="section-header">
<div class="section-title">
Uncertainty & Human Review
</div>
<div class="section-subtitle">
Conformal prediction separates high-confidence automated decisions
from transactions requiring additional human oversight.
</div>
</div>
""")


confident_normal = 38380
human_review = 4125
confident_fraud = 55

total_decisions = (
    confident_normal +
    human_review +
    confident_fraud
)


d1, d2, d3 = st.columns(3)

decision_items = [
    (
        d1,
        "Confident Normal",
        confident_normal,
        confident_normal / total_decisions
    ),
    (
        d2,
        "Human Review",
        human_review,
        human_review / total_decisions
    ),
    (
        d3,
        "Confident Fraud",
        confident_fraud,
        confident_fraud / total_decisions
    )
]


for col, name, count, rate in decision_items:

    with col:

        html(f"""
        <div class="decision-card">
        <div class="decision-name">
        {name}
        </div>
        <div class="decision-value">
        {count:,}
        </div>
        <div class="decision-percent">
        {rate:.2%} of test transactions
        </div>
        </div>
        """)


html("<div style='height:14px'></div>")


coverage_col, decision_col = st.columns([1.15, 1])


with coverage_col:

    fig_conformal = go.Figure()

    fig_conformal.add_trace(
        go.Scatter(
            x=conformal["Target_Coverage"],
            y=conformal["Overall_Coverage"],
            mode="lines+markers",
            name="Overall coverage",
            line=dict(
                color="#B89146",
                width=3
            ),
            marker=dict(size=8)
        )
    )

    fig_conformal.add_trace(
        go.Scatter(
            x=conformal["Target_Coverage"],
            y=conformal["Fraud_Coverage"],
            mode="lines+markers",
            name="Fraud coverage",
            line=dict(
                color="#B5473C",
                width=2.5
            ),
            marker=dict(size=8)
        )
    )

    fig_conformal.add_trace(
        go.Scatter(
            x=conformal["Target_Coverage"],
            y=conformal["Normal_Coverage"],
            mode="lines+markers",
            name="Normal coverage",
            line=dict(
                color="#2F7D6D",
                width=2.5,
                dash="dot"
            ),
            marker=dict(size=7)
        )
    )

    fig_conformal.update_layout(
        title="Coverage Across Significance Levels",
        height=420,
        template="plotly_white",

        margin=dict(
            l=25,
            r=25,
            t=60,
            b=25
        ),

        xaxis=dict(
            title="Target coverage",
            tickformat=".0%",
            gridcolor="#EFE6DA"
        ),

        yaxis=dict(
            title="Empirical coverage",
            tickformat=".0%",
            gridcolor="#EFE6DA"
        ),

        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),

        plot_bgcolor="#FFFDFC",
        paper_bgcolor="#FFFDFC",

        font=dict(
            family="Segoe UI, Arial, sans-serif",
            color="#4A4039",
            size=12
        )
    )

    st.plotly_chart(
        fig_conformal,
        use_container_width=True
    )


with decision_col:

    decision_df = pd.DataFrame({
        "Decision": [
            "Confident Normal",
            "Human Review",
            "Confident Fraud"
        ],
        "Transactions": [
            confident_normal,
            human_review,
            confident_fraud
        ]
    })

    fig_decisions = px.bar(
        decision_df,
        y="Decision",
        x="Transactions",
        orientation="h",
        text="Transactions",
        title="Decision Volume",
        color="Decision",

        color_discrete_map={
            "Confident Normal": "#7B5E3B",
            "Human Review": "#C68A2D",
            "Confident Fraud": "#B5473C"
        }
    )

    fig_decisions.update_traces(
        texttemplate="%{x:,}",
        textposition="outside",
        cliponaxis=False,
        marker_line_width=0
    )

    fig_decisions.update_xaxes(
        type="log",
        title="Number of transactions (log scale)",
        showgrid=True,
        gridcolor="#EFE6DA"
    )

    fig_decisions.update_layout(
        height=420,
        template="plotly_white",

        margin=dict(
            l=25,
            r=50,
            t=60,
            b=25
        ),

        yaxis=dict(
            title="",
            autorange="reversed"
        ),

        showlegend=False,

        plot_bgcolor="#FFFDFC",
        paper_bgcolor="#FFFDFC",

        font=dict(
            family="Segoe UI, Arial, sans-serif",
            color="#4A4039",
            size=12
        )
    )

    st.plotly_chart(
        fig_decisions,
        use_container_width=True
    )


# =========================================================
# RESEARCH HIGHLIGHTS
# =========================================================

html("""
<div class="section-header">
<div class="section-title">
Research Highlights
</div>
<div class="section-subtitle">
Key findings from the final model and uncertainty-aware decision layer.
</div>
</div>
""")


h1, h2, h3, h4 = st.columns(4)

tp = int(summary_dict["True Positives"])
fn = int(summary_dict["False Negatives"])

fraud_total = tp + fn


highlights = [
    (
        h1,
        "Fraud Detected",
        f"{tp}/{fraud_total}",
        "Fraudulent test transactions correctly identified."
    ),

    (
        h2,
        "False Positive Alerts",
        f"{int(summary_dict['False Positives'])}",
        "Legitimate transactions incorrectly classified as fraud."
    ),

    (
        h3,
        "Confident Fraud Precision",
        f"{summary_dict['Confident Fraud Precision']:.1%}",
        "Precision among conformal confident-fraud decisions."
    ),

    (
        h4,
        "Confident Decision Accuracy",
        f"{summary_dict['Conformal Confident Accuracy']:.2%}",
        "Accuracy among transactions not escalated for review."
    )
]


for col, title, value, text in highlights:

    with col:

        html(f"""
        <div class="insight-card">
        <div class="insight-title">
        {title}
        </div>
        <div class="insight-value">
        {value}
        </div>
        <div class="insight-text">
        {text}
        </div>
        </div>
        """)


# =========================================================
# FOOTER
# =========================================================

html("""
<div class="app-footer">

    <div class="footer-main">
        <strong>Designed & Developed by Eliya Christopher Nandi · © 2026</strong>
    </div>

    <div class="footer-contact">
        <span class="contact-label">Contacts:</span>

        <a href="mailto:eliyanandi07@gmail.com">
            ✉&nbsp; eliyanandi07@gmail.com
        </a>

        <span class="footer-separator">·</span>

        <a href="https://www.linkedin.com/in/eliyanandi"
           target="_blank"
           rel="noopener noreferrer">
            in&nbsp; LinkedIn
        </a>
    </div>

</div>
""")