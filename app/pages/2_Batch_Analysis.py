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

PROJECT_DIR = Path(__file__).resolve().parents[2]

if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from src.batch_prediction import analyze_batch


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Batch Fraud Analysis",
    page_icon="◈",
    layout="wide"
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
# DESIGN
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
    padding-top: 1.6rem;
    padding-left: 2.25rem;
    padding-right: 2.25rem;
    padding-bottom: 3rem;
}


/* SIDEBAR */

[data-testid="stSidebar"] {
    background: #2B2118;
    border-right: 1px solid #4A382A;
}

[data-testid="stSidebar"] * {
    color: #F0E6D7;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #D8C7A8;
}

[data-testid="stSidebarNav"] a {
    border-radius: 8px;
    margin: 0.18rem 0.45rem;
}

[data-testid="stSidebarNav"] a:hover {
    background: #3A2A1F;
}

[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: #5A4635;
}


/* HEADER */

.page-eyebrow {
    color: #B89146;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.13em;
}

.page-title {
    color: #2B2118;
    font-size: 2.15rem;
    font-weight: 700;
    letter-spacing: -0.035em;
    margin-top: 0.45rem;
}

.page-subtitle {
    color: #6B625B;
    font-size: 0.9rem;
    line-height: 1.6;
    max-width: 940px;
    margin-top: 0.35rem;
    margin-bottom: 1.7rem;
}


/* METRIC CARDS */

.metric-card {
    background: #FFFDFC;
    border: 1px solid #E4D8C8;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    min-height: 105px;
    box-shadow: 0 1px 5px rgba(43,33,24,0.025);
}

.metric-label {
    color: #8A6A3B;
    font-size: 0.69rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.metric-value {
    color: #2B2118;
    font-size: 1.55rem;
    font-weight: 700;
    margin-top: 0.32rem;
}

.metric-note {
    color: #6B625B;
    font-size: 0.72rem;
    margin-top: 0.2rem;
}


/* SECTION */

.section-title {
    color: #2B2118;
    font-size: 1rem;
    font-weight: 700;
    margin-top: 2rem;
}

.section-subtitle {
    color: #6B625B;
    font-size: 0.77rem;
    margin-top: 0.18rem;
    margin-bottom: 0.8rem;
}


/* UPLOAD NOTE */

.upload-note {
    background: #FFF9EE;
    border: 1px solid #E7D3A8;
    border-left: 4px solid #B89146;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 1rem;
}

.upload-title {
    color: #8A6A3B;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.upload-copy {
    color: #6B625B;
    font-size: 0.78rem;
    line-height: 1.5;
    margin-top: 0.25rem;
}


/* PLOTLY */

div[data-testid="stPlotlyChart"] {
    background: #FFFDFC;
    border: 1px solid #E4D8C8;
    border-radius: 14px;
}


/* DATAFRAME */

[data-testid="stDataFrame"] {
    border: 1px solid #E4D8C8;
    border-radius: 10px;
    overflow: hidden;
}

</style>
""")


# =========================================================
# HEADER
# =========================================================

html("""
<div class="page-eyebrow">
Portfolio Risk Screening
</div>

<div class="page-title">
Batch Fraud Analysis
</div>

<div class="page-subtitle">
Upload a transaction dataset and screen multiple records using the
deployed XGBoost model and conformal uncertainty layer. Review fraud
risk, automated decisions and cases requiring additional human review.
</div>
""")


# =========================================================
# FILE UPLOAD
# =========================================================

html("""
<div class="upload-note">

<div class="upload-title">
CSV Requirements
</div>

<div class="upload-copy">
The file must contain the model features
Time, V1–V28 and Amount.
The Class column is optional.
</div>

</div>
""")


uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)


# =========================================================
# DEMO SAMPLE OPTION
# =========================================================

use_demo = st.checkbox(
    "Use a demonstration sample from the research dataset"
)


if uploaded_file is None and not use_demo:

    st.info(
        "Upload a CSV file or enable the demonstration sample."
    )

    st.stop()


# =========================================================
# LOAD INPUT
# =========================================================

if use_demo:

    @st.cache_data
    def load_demo():

        full_df = pd.read_csv(
            PROJECT_DIR / "assets" / "demo_transactions.csv"
        )

        demo = pd.concat([
            full_df[
                full_df["Class"] == 0
            ].sample(
                190,
                random_state=42
            ),

            full_df[
                full_df["Class"] == 1
            ].sample(
                10,
                random_state=42
            )

        ]).sample(
            frac=1,
            random_state=42
        ).reset_index(
            drop=True
        )

        return demo

    input_df = load_demo()

else:

    try:

        input_df = pd.read_csv(
            uploaded_file
        )

    except Exception as exc:

        st.error(
            f"Unable to read CSV file: {exc}"
        )

        st.stop()


# =========================================================
# ANALYZE
# =========================================================

try:

    with st.spinner(
        "Screening transactions..."
    ):

        results_df = analyze_batch(
            input_df
        )

except ValueError as exc:

    st.error(str(exc))

    st.stop()


# =========================================================
# SUMMARY METRICS
# =========================================================

total = len(results_df)

model_fraud = (
    results_df[
        "Model_Prediction"
    ]
    .eq("Fraud")
    .sum()
)

human_review = (
    results_df[
        "Conformal_Decision"
    ]
    .eq("Human Review")
    .sum()
)

confident_fraud = (
    results_df[
        "Conformal_Decision"
    ]
    .eq("Confident Fraud")
    .sum()
)

confident_normal = (
    results_df[
        "Conformal_Decision"
    ]
    .eq("Confident Normal")
    .sum()
)


m1, m2, m3, m4 = st.columns(4)


metrics = [

    (
        m1,
        "Transactions",
        f"{total:,}",
        "Records screened"
    ),

    (
        m2,
        "Fraud Alerts",
        f"{model_fraud:,}",
        f"{model_fraud / total:.1%} of transactions"
    ),

    (
        m3,
        "Human Review",
        f"{human_review:,}",
        f"{human_review / total:.1%} require review"
    ),

    (
        m4,
        "Confident Fraud",
        f"{confident_fraud:,}",
        f"{confident_fraud / total:.1%} of transactions"
    )
]


for col, label, value, note in metrics:

    with col:

        html(f"""
        <div class="metric-card">

        <div class="metric-label">
        {label}
        </div>

        <div class="metric-value">
        {value}
        </div>

        <div class="metric-note">
        {note}
        </div>

        </div>
        """)


# =========================================================
# CHARTS
# =========================================================

html("""
<div class="section-title">
Portfolio Risk Overview
</div>

<div class="section-subtitle">
Distribution of predicted fraud probabilities and uncertainty-aware decisions.
</div>
""")


chart_left, chart_right = st.columns(
    [1.15, 1]
)


with chart_left:

    fig_hist = px.histogram(
        results_df,
        x="Fraud_Probability",
        nbins=40,
        title="Fraud Risk Distribution"
    )

    fig_hist.update_traces(
        marker_color="#B89146"
    )

    fig_hist.update_layout(
        height=390,
        template="plotly_white",

        xaxis=dict(
            title="Fraud probability",
            tickformat=".0%",
            gridcolor="#EFE6DA"
        ),

        yaxis=dict(
            title="Transactions",
            gridcolor="#EFE6DA"
        ),

        margin=dict(
            l=35,
            r=25,
            t=55,
            b=40
        ),

        paper_bgcolor="#FFFDFC",
        plot_bgcolor="#FFFDFC",

        font=dict(
            family="Segoe UI, Arial, sans-serif",
            color="#4A4039"
        )
    )

    st.plotly_chart(
        fig_hist,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


with chart_right:

    decision_counts = (
        results_df[
            "Conformal_Decision"
        ]
        .value_counts()
        .reindex(
            [
                "Confident Normal",
                "Human Review",
                "Confident Fraud"
            ],
            fill_value=0
        )
        .reset_index()
    )

    decision_counts.columns = [
        "Decision",
        "Transactions"
    ]


    fig_decision = px.bar(
        decision_counts,
        y="Decision",
        x="Transactions",
        orientation="h",
        text="Transactions",
        title="Decision Distribution",

        color="Decision",

        color_discrete_map={
            "Confident Normal":
                "#7B5E3B",
            "Human Review":
                "#C68A2D",
            "Confident Fraud":
                "#B5473C"
        }
    )

    fig_decision.update_traces(
        textposition="outside"
    )

    fig_decision.update_layout(
        height=390,
        template="plotly_white",

        showlegend=False,

        xaxis_title="Transactions",
        yaxis_title="",

        margin=dict(
            l=35,
            r=45,
            t=55,
            b=40
        ),

        paper_bgcolor="#FFFDFC",
        plot_bgcolor="#FFFDFC",

        font=dict(
            family="Segoe UI, Arial, sans-serif",
            color="#4A4039"
        )
    )

    st.plotly_chart(
        fig_decision,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


# =========================================================
# FILTERS
# =========================================================

html("""
<div class="section-title">
Transaction Screening Results
</div>

<div class="section-subtitle">
Filter the screened portfolio by uncertainty decision or model prediction.
</div>
""")


filter1, filter2 = st.columns(2)


with filter1:

    decision_filter = st.multiselect(
        "Conformal decision",
        options=[
            "Confident Normal",
            "Human Review",
            "Confident Fraud"
        ],

        default=[
            "Confident Normal",
            "Human Review",
            "Confident Fraud"
        ]
    )


with filter2:

    prediction_filter = st.multiselect(
        "Model prediction",
        options=[
            "Legitimate",
            "Fraud"
        ],

        default=[
            "Legitimate",
            "Fraud"
        ]
    )


filtered_df = results_df[
    results_df[
        "Conformal_Decision"
    ].isin(decision_filter)
    &
    results_df[
        "Model_Prediction"
    ].isin(prediction_filter)
].copy()


# =========================================================
# DISPLAY TABLE
# =========================================================

display_columns = [
    "Transaction_ID",
    "Amount",
    "Fraud_Probability",
    "Model_Prediction",
    "Prediction_Set",
    "Conformal_Decision",
    "Recommended_Action"
]


display_df = filtered_df[
    display_columns
].copy()


display_df[
    "Fraud_Probability"
] = display_df[
    "Fraud_Probability"
].map(
    lambda x: f"{x:.2%}"
)


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    height=500
)


# =========================================================
# DOWNLOAD RESULTS
# =========================================================

download_df = results_df.copy()

csv_data = download_df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="Download screening results",
    data=csv_data,
    file_name="fraud_screening_results.csv",
    mime="text/csv"
)