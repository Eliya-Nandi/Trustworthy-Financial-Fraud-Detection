import sys
from pathlib import Path
from textwrap import dedent

import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# =========================================================
# PROJECT SETUP
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

#from src.prediction import analyze_transaction
from src.prediction import analyze_transaction
from src.explainability import explain_transaction


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Transaction Analyzer",
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
# DESIGN SYSTEM
# =========================================================

html("""
<style>

/* ---------------------------------------------------------
   GLOBAL
--------------------------------------------------------- */

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
    padding-top: 1.55rem;
    padding-left: 2.25rem;
    padding-right: 2.25rem;
    padding-bottom: 3rem;
}


/* ---------------------------------------------------------
   SIDEBAR
--------------------------------------------------------- */

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

[data-testid="stSidebar"] hr {
    border-color: #5A4635;
}


/* ---------------------------------------------------------
   PAGE HEADER
--------------------------------------------------------- */

.page-eyebrow {
    color: #B89146;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    margin-bottom: 0.55rem;
}

.page-title {
    font-size: 2.15rem;
    font-weight: 700;
    color: #2B2118;
    letter-spacing: -0.035em;
    margin-bottom: 0.4rem;
}

.page-subtitle {
    color: #6B625B;
    font-size: 0.92rem;
    line-height: 1.6;
    max-width: 980px;
    margin-bottom: 1.75rem;
}


/* ---------------------------------------------------------
   SECTION TITLE
--------------------------------------------------------- */

.section-title {
    color: #2B2118;
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

.section-subtitle {
    color: #7B7168;
    font-size: 0.78rem;
    line-height: 1.45;
    margin-bottom: 0.8rem;
}


/* ---------------------------------------------------------
   TRANSACTION SUMMARY
--------------------------------------------------------- */

.transaction-summary {
    background: #FFFDFC;
    border: 1px solid #E4D8C8;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-top: 1rem;
    box-shadow: 0 1px 5px rgba(43,33,24,0.025);
}

.summary-row {
    display: flex;
    justify-content: space-between;
    padding: 0.42rem 0;
    border-bottom: 1px solid #EFE6DA;
}

.summary-row:last-child {
    border-bottom: none;
}

.summary-label {
    color: #6B625B;
    font-size: 0.78rem;
}

.summary-value {
    color: #2B2118;
    font-size: 0.8rem;
    font-weight: 700;
}


/* ---------------------------------------------------------
   ASSESSMENT BANNER
--------------------------------------------------------- */

.assessment-banner {
    background:
        linear-gradient(
            135deg,
            #3A2A1F 0%,
            #2B2118 100%
        );

    border: 1px solid #5A4635;
    border-radius: 14px;
    padding: 1.15rem 1.25rem;
    margin-bottom: 0.8rem;
}

.assessment-label {
    color: #D8C7A8;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.assessment-value {
    color: #FFF8EC;
    font-size: 1.5rem;
    font-weight: 700;
    margin-top: 0.25rem;
}

.assessment-copy {
    color: #E7DCCF;
    font-size: 0.76rem;
    line-height: 1.45;
    margin-top: 0.25rem;
}


/* ---------------------------------------------------------
   RESULT CARDS
--------------------------------------------------------- */

.result-card {
    background: #FFFDFC;
    border: 1px solid #E4D8C8;
    border-radius: 12px;
    padding: 1.05rem 1.15rem;
    min-height: 122px;
    box-shadow: 0 1px 5px rgba(43,33,24,0.025);
}

.result-label {
    color: #8A6A3B;
    font-size: 0.69rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.065em;
}

.result-value {
    color: #2B2118;
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 0.35rem;
}

.result-note {
    color: #6B625B;
    font-size: 0.73rem;
    line-height: 1.4;
    margin-top: 0.28rem;
}


/* ---------------------------------------------------------
   RECOMMENDED ACTIONS
--------------------------------------------------------- */

.action-normal {
    background: #EEF7F3;
    border: 1px solid #CEE5DC;
    border-left: 5px solid #2F7D6D;
    padding: 1.05rem 1.15rem;
    border-radius: 10px;
}

.action-review {
    background: #FFF7E8;
    border: 1px solid #ECD9AF;
    border-left: 5px solid #C68A2D;
    padding: 1.05rem 1.15rem;
    border-radius: 10px;
}

.action-fraud {
    background: #FBEDEC;
    border: 1px solid #E9C8C4;
    border-left: 5px solid #B5473C;
    padding: 1.05rem 1.15rem;
    border-radius: 10px;
}

.action-eyebrow {
    color: #6B625B;
    font-size: 0.66rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

.action-title {
    font-weight: 700;
    color: #2B2118;
    font-size: 1rem;
    margin-top: 0.18rem;
}

.action-text {
    color: #6B625B;
    font-size: 0.8rem;
    margin-top: 0.18rem;
}


/* ---------------------------------------------------------
   FEATURE AREA
--------------------------------------------------------- */

.feature-header {
    margin-top: 1.8rem;
    color: #2B2118;
    font-size: 0.95rem;
    font-weight: 700;
}

.feature-copy {
    color: #6B625B;
    font-size: 0.76rem;
    margin-top: 0.2rem;
    margin-bottom: 0.6rem;
}


/* ---------------------------------------------------------
   STREAMLIT WIDGETS
--------------------------------------------------------- */

div[data-testid="stPlotlyChart"] {
    background: #FFFDFC;
    border: 1px solid #E4D8C8;
    border-radius: 14px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #E4D8C8;
    border-radius: 10px;
    overflow: hidden;
}

</style>
""")


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_demo_data():

    return pd.read_csv(
        PROJECT_DIR / "creditcard_clean.csv"
    )


df = load_demo_data()


# =========================================================
# PAGE HEADER
# =========================================================

html("""
<div class="page-eyebrow">
Transaction-Level Decision Support
</div>

<div class="page-title">
Transaction Risk Analyzer
</div>

<div class="page-subtitle">
Inspect an individual transaction using the deployed XGBoost model
and conformal uncertainty layer. Review the estimated fraud risk,
classification outcome, confidence status and recommended operational action.
</div>
""")


# =========================================================
# TRANSACTION SELECTION
# =========================================================

left, right = st.columns(
    [1.0, 1.25],
    gap="large"
)


with left:

    html("""
    <div class="section-title">
    Select a Transaction
    </div>

    <div class="section-subtitle">
    Choose a representative transaction from the research dataset.
    </div>
    """)


    sample_type = st.radio(
        "Transaction type",
        [
            "Legitimate sample",
            "Fraud sample",
            "Random transaction"
        ],
        horizontal=True
    )


    if sample_type == "Legitimate sample":

        subset = df[
            df["Class"] == 0
        ]

    elif sample_type == "Fraud sample":

        subset = df[
            df["Class"] == 1
        ]

    else:

        subset = df


    max_index = min(
        50,
        len(subset)
    )


    sample_position = st.slider(
        "Sample number",
        min_value=1,
        max_value=max_index,
        value=1
    )


    selected_row = subset.iloc[
        [sample_position - 1]
    ]


    true_class = int(
        selected_row["Class"].iloc[0]
    )


    amount = float(
        selected_row["Amount"].iloc[0]
    )


    time_value = float(
        selected_row["Time"].iloc[0]
    )


    true_class_text = (
        "Fraud"
        if true_class == 1
        else "Legitimate"
    )


    html(f"""
    <div class="transaction-summary">

    <div class="summary-row">
    <span class="summary-label">
    Actual class
    </span>
    <span class="summary-value">
    {true_class_text}
    </span>
    </div>

    <div class="summary-row">
    <span class="summary-label">
    Transaction amount
    </span>
    <span class="summary-value">
    {amount:,.2f}
    </span>
    </div>

    <div class="summary-row">
    <span class="summary-label">
    Transaction time
    </span>
    <span class="summary-value">
    {time_value:,.0f}
    </span>
    </div>

    </div>
    """)


# =========================================================
# ANALYZE
# =========================================================

analysis = analyze_transaction(
    selected_row
)

shap_result = explain_transaction(
    selected_row,
    top_n=8
)

top_shap = shap_result["top_features"].copy()
fraud_drivers = shap_result["fraud_drivers"].copy()
normal_drivers = shap_result["normal_drivers"].copy()


fraud_probability = float(
    analysis["Fraud_Probability"]
)


model_prediction = (
    analysis["Model_Prediction"]
)


conformal_decision = (
    analysis["Conformal_Decision"]
)


recommended_action = (
    analysis["Recommended_Action"]
)


threshold = float(
    analysis["Decision_Threshold"]
)


prediction_set = (
    analysis["Prediction_Set"]
)


# =========================================================
# RISK ASSESSMENT
# =========================================================

with right:

    if conformal_decision == "Confident Normal":

        assessment_heading = (
            "Low-risk automated decision"
        )

        assessment_copy = (
            "The transaction falls within the model's "
            "confident-normal conformal region."
        )

    elif conformal_decision == "Confident Fraud":

        assessment_heading = (
            "High-risk fraud alert"
        )

        assessment_copy = (
            "The transaction falls within the model's "
            "confident-fraud conformal region."
        )

    else:

        assessment_heading = (
            "Uncertain transaction"
        )

        assessment_copy = (
            "The conformal prediction layer does not support "
            "a confident automated decision."
        )


    html(f"""
    <div class="assessment-banner">

    <div class="assessment-label">
    Current Risk Assessment
    </div>

    <div class="assessment-value">
    {assessment_heading}
    </div>

    <div class="assessment-copy">
    {assessment_copy}
    </div>

    </div>
    """)


    fig_gauge = go.Figure(
        go.Indicator(

            mode="gauge+number",

            value=fraud_probability * 100,

            number={
                "suffix": "%",
                "valueformat": ".1f",

                "font": {
                    "size": 38,
                    "color": "#2B2118"
                }
            },

            title={
                "text": "Fraud Risk Score",

                "font": {
                    "size": 15,
                    "color": "#6B625B"
                }
            },

            gauge={

                "axis": {
                    "range": [0, 100],

                    "tickvals": [
                        0,
                        20,
                        40,
                        60,
                        80,
                        100
                    ],

                    "ticksuffix": "%",

                    "tickfont": {
                        "size": 9,
                        "color": "#6B625B"
                    },

                    "tickcolor": "#A99A8D"
                },

                "bar": {
                    "color": "#B89146",
                    "thickness": 0.28
                },

                "bgcolor": "#FFFDFC",

                "borderwidth": 0,

                "steps": [
                    {
                        "range": [0, 30],
                        "color": "#E7F2ED"
                    },

                    {
                        "range": [30, 70],
                        "color": "#FBF0D9"
                    },

                    {
                        "range": [70, 100],
                        "color": "#F5E1DE"
                    }
                ],

                "threshold": {

                    "line": {
                        "color": "#B5473C",
                        "width": 4
                    },

                    "thickness": 0.75,

                    "value": threshold * 100
                }
            }
        )
    )


    fig_gauge.update_layout(

        height=285,

        margin=dict(
            l=30,
            r=30,
            t=45,
            b=5
        ),

        paper_bgcolor="#FFFDFC",

        font=dict(
            family="Segoe UI, Arial, sans-serif"
        )
    )


    st.plotly_chart(
        fig_gauge,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


    st.caption(
        f"Red threshold marker = "
        f"{threshold:.1%} fraud classification threshold"
    )


# =========================================================
# DECISION SUMMARY
# =========================================================

html("""
<div style="height:12px"></div>

<div class="section-title">
Decision Summary
</div>

<div class="section-subtitle">
A combined view of model risk, classification,
uncertainty and observed class.
</div>
""")


r1, r2, r3, r4 = st.columns(4)


results = [

    (
        r1,
        "Fraud Probability",
        f"{fraud_probability:.2%}",
        "Model-estimated fraud score"
    ),

    (
        r2,
        "Model Prediction",
        model_prediction,
        f"Decision threshold: {threshold:.3f}"
    ),

    (
        r3,
        "Conformal Decision",
        conformal_decision,
        f"Prediction set: {prediction_set}"
    ),

    (
        r4,
        "Actual Label",
        true_class_text,
        "Ground-truth class from dataset"
    )
]


for col, label, value, note in results:

    with col:

        html(f"""
        <div class="result-card">

        <div class="result-label">
        {label}
        </div>

        <div class="result-value">
        {value}
        </div>

        <div class="result-note">
        {note}
        </div>

        </div>
        """)


# =========================================================
# RECOMMENDED ACTION
# =========================================================

html(
    "<div style='height:16px'></div>"
)


if conformal_decision == "Confident Normal":

    css_class = "action-normal"

    action_heading = (
        "Automated clearance"
    )


elif conformal_decision == "Confident Fraud":

    css_class = "action-fraud"

    action_heading = (
        "Fraud investigation recommended"
    )


else:

    css_class = "action-review"

    action_heading = (
        "Human review required"
    )


html(f"""
<div class="{css_class}">

<div class="action-eyebrow">
Recommended Action
</div>

<div class="action-title">
{action_heading}
</div>

<div class="action-text">
{recommended_action}
</div>

</div>
""")
#-----------------------------------------------------------
#===========================================================
# =========================================================
# LOCAL SHAP EXPLAINABILITY
# =========================================================

html("""
<div style="height:20px"></div>

<div class="section-title">
Why did the model make this decision?
</div>

<div class="section-subtitle">
Local SHAP values show which transaction features pushed the model
toward fraud and which features pushed the prediction toward the
legitimate class.
</div>
""")


# ---------------------------------------------------------
# Prepare chart data
# ---------------------------------------------------------

shap_chart = pd.concat([
    fraud_drivers.head(5),
    normal_drivers.head(5)
]).copy()

shap_chart = shap_chart.drop_duplicates(
    subset=["Feature"]
)

shap_chart = shap_chart.sort_values(
    "SHAP_Value",
    ascending=True
)

shap_chart["Direction"] = shap_chart[
    "SHAP_Value"
].apply(
    lambda x:
        "Toward Fraud"
        if x > 0
        else "Toward Legitimate"
)


# ---------------------------------------------------------
# SHAP contribution chart
# ---------------------------------------------------------

fig_local_shap = go.Figure()


for direction, color in [
    ("Toward Fraud", "#B5473C"),
    ("Toward Legitimate", "#2F7D6D")
]:

    subset_chart = shap_chart[
        shap_chart["Direction"] == direction
    ]

    fig_local_shap.add_trace(
        go.Bar(
            x=subset_chart["SHAP_Value"],
            y=subset_chart["Feature"],
            orientation="h",
            name=direction,
            marker_color=color,

            customdata=subset_chart[
                ["Feature_Value"]
            ],

            hovertemplate=(
                "<b>%{y}</b><br>"
                "Feature value: %{customdata[0]:.4f}<br>"
                "SHAP contribution: %{x:.4f}"
                "<extra></extra>"
            )
        )
    )


fig_local_shap.add_vline(
    x=0,
    line_width=1,
    line_color="#8C8073"
)


fig_local_shap.update_layout(

    title="Feature Contributions for Selected Transaction",

    height=430,

    template="plotly_white",

    barmode="relative",

    margin=dict(
        l=40,
        r=30,
        t=70,
        b=40
    ),

    xaxis=dict(
        title="SHAP contribution to model output",
        gridcolor="#EFE6DA",
        zeroline=False
    ),

    yaxis=dict(
        title=""
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
    fig_local_shap,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)


# ---------------------------------------------------------
# Driver summaries
# ---------------------------------------------------------

driver_left, driver_right = st.columns(2)


with driver_left:

    html("""
    <div style="
        background:#FBEDEC;
        border:1px solid #E9C8C4;
        border-radius:12px;
        padding:1rem 1.1rem;
        margin-bottom:0.6rem;
    ">
        <div style="
            color:#B5473C;
            font-size:0.72rem;
            font-weight:700;
            text-transform:uppercase;
            letter-spacing:0.06em;
        ">
            Strongest Fraud Drivers
        </div>

        <div style="
            color:#6B625B;
            font-size:0.76rem;
            margin-top:0.3rem;
        ">
            Features increasing the model's fraud score.
        </div>
    </div>
    """)

    fraud_display = (
        fraud_drivers
        .head(5)[
            [
                "Feature",
                "Feature_Value",
                "SHAP_Value"
            ]
        ]
        .copy()
    )

    fraud_display.columns = [
        "Feature",
        "Value",
        "SHAP Contribution"
    ]

    st.dataframe(
        fraud_display,
        use_container_width=True,
        hide_index=True
    )


with driver_right:

    html("""
    <div style="
        background:#EEF7F3;
        border:1px solid #CEE5DC;
        border-radius:12px;
        padding:1rem 1.1rem;
        margin-bottom:0.6rem;
    ">
        <div style="
            color:#2F7D6D;
            font-size:0.72rem;
            font-weight:700;
            text-transform:uppercase;
            letter-spacing:0.06em;
        ">
            Strongest Legitimate Drivers
        </div>

        <div style="
            color:#6B625B;
            font-size:0.76rem;
            margin-top:0.3rem;
        ">
            Features reducing the model's fraud score.
        </div>
    </div>
    """)

    normal_display = (
        normal_drivers
        .head(5)[
            [
                "Feature",
                "Feature_Value",
                "SHAP_Value"
            ]
        ]
        .copy()
    )

    normal_display.columns = [
        "Feature",
        "Value",
        "SHAP Contribution"
    ]

    st.dataframe(
        normal_display,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------
# Plain-language interpretation
# ---------------------------------------------------------

top_positive = (
    fraud_drivers.iloc[0]["Feature"]
    if len(fraud_drivers) > 0
    else None
)

top_negative = (
    normal_drivers.iloc[0]["Feature"]
    if len(normal_drivers) > 0
    else None
)


if top_positive and top_negative:

    interpretation = (
        f"The strongest feature pushing this prediction toward fraud "
        f"is {top_positive}, while {top_negative} provides the strongest "
        f"counteracting contribution toward the legitimate class."
    )

elif top_positive:

    interpretation = (
        f"The strongest local feature pushing this prediction toward "
        f"fraud is {top_positive}."
    )

elif top_negative:

    interpretation = (
        f"The strongest local feature pushing this prediction toward "
        f"the legitimate class is {top_negative}."
    )

else:

    interpretation = (
        "No dominant local feature contribution was identified."
    )


html(f"""
<div style="
    background:#FFF9EE;
    border:1px solid #E7D3A8;
    border-left:4px solid #B89146;
    border-radius:10px;
    padding:1rem 1.1rem;
    margin-top:0.8rem;
">

<div style="
    color:#8A6A3B;
    font-size:0.67rem;
    font-weight:700;
    text-transform:uppercase;
    letter-spacing:0.07em;
">
Explanation Summary
</div>

<div style="
    color:#2B2118;
    font-size:0.83rem;
    line-height:1.55;
    margin-top:0.3rem;
">
{interpretation}
</div>

<div style="
    color:#7B7168;
    font-size:0.72rem;
    line-height:1.45;
    margin-top:0.45rem;
">
SHAP values describe contributions to the model's prediction.
Because V1–V28 are anonymized transformed features, their
real-world financial meaning cannot be inferred from this dataset.
</div>

</div>
""")


# =========================================================
# TRANSACTION FEATURE PREVIEW
# =========================================================

html("""
<div class="feature-header">
Transaction Feature Vector
</div>

<div class="feature-copy">
Review the anonymized model inputs used to score the selected transaction.
</div>
""")


with st.expander(
    "View full transaction features"
):

    feature_row = (
        selected_row
        .drop(columns=["Class"])
        .T
        .reset_index()
    )


    feature_row.columns = [
        "Feature",
        "Value"
    ]


    st.dataframe(
        feature_row,
        use_container_width=True,
        hide_index=True,
        height=400
    )