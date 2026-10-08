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

from src.batch_prediction import analyze_batch
from src.explainability import explain_transaction


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Human Review Queue",
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
    max-width: 980px;
    margin-top: 0.35rem;
    margin-bottom: 1.6rem;
}


/* SUMMARY CARDS */

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


/* SECTIONS */

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


/* REVIEW PANEL */

.review-panel {
    background: #FFF7E8;
    border: 1px solid #ECD9AF;
    border-left: 5px solid #C68A2D;
    border-radius: 12px;
    padding: 1.15rem 1.2rem;
    min-height: 150px;
}

.review-label {
    color: #8A6A3B;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.review-value {
    color: #2B2118;
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 0.25rem;
}

.review-copy {
    color: #6B625B;
    font-size: 0.75rem;
    line-height: 1.45;
    margin-top: 0.45rem;
}


/* PRIORITY BADGES */

.priority-badge {
    display: inline-block;
    padding: 0.28rem 0.55rem;
    border-radius: 999px;
    font-size: 0.66rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.55rem;
}

.priority-high {
    background: #FBEDEC;
    color: #B5473C;
    border: 1px solid #E9C8C4;
}

.priority-medium {
    background: #FFF7E8;
    color: #A26E1E;
    border: 1px solid #ECD9AF;
}

.priority-low {
    background: #EEF7F3;
    color: #2F7D6D;
    border: 1px solid #CEE5DC;
}


/* REVIEW STATUS */

.review-status {
    margin-top: 0.7rem;
    background: #EEF7F3;
    border: 1px solid #CEE5DC;
    border-radius: 10px;
    padding: 0.75rem 0.9rem;
}

.review-status-label {
    color: #2F7D6D;
    font-size: 0.67rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.review-status-value {
    color: #2B2118;
    font-size: 0.85rem;
    font-weight: 700;
    margin-top: 0.2rem;
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
# LOAD DEMO DATA
# =========================================================

@st.cache_data
def load_demo_review_data():

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
    ).reset_index(drop=True)

    screened = analyze_batch(
        demo
    )

    return screened


screened_df = load_demo_review_data()


review_df = (
    screened_df[
        screened_df["Conformal_Decision"]
        == "Human Review"
    ]
    .copy()
    .reset_index(drop=True)
)


# =========================================================
# HEADER
# =========================================================

html("""
<div class="page-eyebrow">
Human-in-the-Loop Fraud Operations
</div>

<div class="page-title">
Human Review Queue
</div>

<div class="page-subtitle">
Review transactions for which the conformal prediction layer did not support
a confident automated decision. Analysts can inspect fraud risk, transaction
details and local model explanations before making a final operational decision.
</div>
""")


# =========================================================
# SUMMARY METRICS
# =========================================================

review_count = len(review_df)

high_priority = (
    review_df[
        "Fraud_Probability"
    ] >= 0.50
).sum()

medium_priority = (
    (
        review_df["Fraud_Probability"] >= 0.10
    )
    &
    (
        review_df["Fraud_Probability"] < 0.50
    )
).sum()

lower_priority = (
    review_df[
        "Fraud_Probability"
    ] < 0.10
).sum()


m1, m2, m3, m4 = st.columns(4)

metrics = [
    (
        m1,
        "Review Queue",
        f"{review_count:,}",
        "Transactions awaiting review"
    ),
    (
        m2,
        "High Priority",
        f"{high_priority:,}",
        "Fraud probability ≥ 50%"
    ),
    (
        m3,
        "Medium Priority",
        f"{medium_priority:,}",
        "Fraud probability 10–50%"
    ),
    (
        m4,
        "Lower Priority",
        f"{lower_priority:,}",
        "Fraud probability below 10%"
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
# QUEUE TABLE
# =========================================================

html("""
<div class="section-title">
Review Queue
</div>

<div class="section-subtitle">
Transactions are sorted from highest to lowest fraud probability.
</div>
""")


queue_df = (
    review_df[
        [
            "Transaction_ID",
            "Amount",
            "Fraud_Probability",
            "Model_Prediction",
            "Prediction_Set",
            "Recommended_Action"
        ]
    ]
    .sort_values(
        "Fraud_Probability",
        ascending=False
    )
    .copy()
)


queue_display = queue_df.copy()

queue_display[
    "Fraud_Probability"
] = queue_display[
    "Fraud_Probability"
].map(
    lambda x: f"{x:.2%}"
)


# Add review status column
def get_review_status(transaction_id):

    if (
        "analyst_reviews"
        in st.session_state
        and transaction_id
        in st.session_state["analyst_reviews"]
    ):
        return "Reviewed"

    return "Pending"


queue_display[
    "Review_Status"
] = queue_display[
    "Transaction_ID"
].apply(
    get_review_status
)


st.dataframe(
    queue_display,
    use_container_width=True,
    hide_index=True,
    height=350
)


# =========================================================
# SELECT TRANSACTION
# =========================================================

html("""
<div class="section-title">
Transaction Investigation
</div>

<div class="section-subtitle">
Select one review case to inspect in detail.
</div>
""")


transaction_options = (
    queue_df[
        "Transaction_ID"
    ]
    .tolist()
)


selected_transaction_id = st.selectbox(
    "Transaction ID",
    transaction_options
)


selected = (
    review_df[
        review_df["Transaction_ID"]
        == selected_transaction_id
    ]
    .iloc[[0]]
)


fraud_probability = float(
    selected[
        "Fraud_Probability"
    ].iloc[0]
)

amount = float(
    selected[
        "Amount"
    ].iloc[0]
)

model_prediction = (
    selected[
        "Model_Prediction"
    ].iloc[0]
)

prediction_set = (
    selected[
        "Prediction_Set"
    ].iloc[0]
)

recommended_action = (
    selected[
        "Recommended_Action"
    ].iloc[0]
)


# =========================================================
# PRIORITY
# =========================================================

if fraud_probability >= 0.50:

    priority_label = "High Priority"
    priority_class = "priority-high"

elif fraud_probability >= 0.10:

    priority_label = "Medium Priority"
    priority_class = "priority-medium"

else:

    priority_label = "Lower Priority"
    priority_class = "priority-low"


# =========================================================
# SAVED REVIEW
# =========================================================

saved_review = None

if "analyst_reviews" in st.session_state:

    saved_review = st.session_state[
        "analyst_reviews"
    ].get(
        selected_transaction_id
    )


# =========================================================
# TRANSACTION DETAIL
# =========================================================

detail_left, detail_right = st.columns(
    [1, 1],
    gap="large"
)


with detail_left:

    html(f"""
    <div class="review-panel">

    <div class="review-label">
    Selected Review Case
    </div>

    <div class="review-value">
    Transaction #{selected_transaction_id}
    </div>

    <div class="priority-badge {priority_class}">
    {priority_label}
    </div>

    <div class="review-copy">
    This transaction was withheld from automatic decision-making
    because its conformal prediction set was {prediction_set}.
    </div>

    </div>
    """)


    if saved_review:

        html(f"""
        <div class="review-status">

        <div class="review-status-label">
        Review Status
        </div>

        <div class="review-status-value">
        {saved_review["Decision"]}
        </div>

        </div>
        """)


    st.markdown("")

    st.write(
        f"**Fraud probability:** "
        f"{fraud_probability:.2%}"
    )

    st.write(
        f"**Transaction amount:** "
        f"{amount:,.2f}"
    )

    st.write(
        f"**Model prediction:** "
        f"{model_prediction}"
    )

    st.write(
        f"**Conformal decision:** "
        f"Human Review"
    )

    st.write(
        f"**Recommended action:** "
        f"{recommended_action}"
    )


with detail_right:

    fig_risk = go.Figure(
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
                "text": "Review-Case Fraud Risk",
                "font": {
                    "size": 15,
                    "color": "#6B625B"
                }
            },

            gauge={
                "axis": {
                    "range": [0, 100],
                    "ticksuffix": "%",
                    "tickfont": {
                        "size": 9,
                        "color": "#6B625B"
                    }
                },

                "bar": {
                    "color": "#C68A2D",
                    "thickness": 0.28
                },

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
                ]
            }
        )
    )

    fig_risk.update_layout(
        height=255,
        margin=dict(
            l=30,
            r=30,
            t=45,
            b=5
        ),
        paper_bgcolor="#FFFDFC"
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


# =========================================================
# LOCAL SHAP EXPLANATION
# =========================================================

html("""
<div class="section-title">
Why was this transaction difficult to automate?
</div>

<div class="section-subtitle">
Local SHAP values show competing model contributions for the selected review case.
</div>
""")


transaction_features = selected.drop(
    columns=[
        "Transaction_ID",
        "Fraud_Probability",
        "Model_Prediction",
        "Prediction_Set",
        "Conformal_Decision",
        "Recommended_Action"
    ],
    errors="ignore"
)


shap_result = explain_transaction(
    transaction_features,
    top_n=8
)


fraud_drivers = (
    shap_result[
        "fraud_drivers"
    ]
    .head(5)
)

normal_drivers = (
    shap_result[
        "normal_drivers"
    ]
    .head(5)
)


shap_chart = pd.concat([
    fraud_drivers,
    normal_drivers
]).copy()


shap_chart = (
    shap_chart
    .drop_duplicates(
        subset=["Feature"]
    )
    .sort_values(
        "SHAP_Value",
        ascending=True
    )
)


shap_chart[
    "Direction"
] = shap_chart[
    "SHAP_Value"
].apply(
    lambda x:
        "Toward Fraud"
        if x > 0
        else "Toward Legitimate"
)


fig_shap = go.Figure()


for direction, color in [
    (
        "Toward Fraud",
        "#B5473C"
    ),
    (
        "Toward Legitimate",
        "#2F7D6D"
    )
]:

    subset_chart = shap_chart[
        shap_chart["Direction"]
        == direction
    ]

    fig_shap.add_trace(
        go.Bar(
            x=subset_chart[
                "SHAP_Value"
            ],
            y=subset_chart[
                "Feature"
            ],
            orientation="h",
            name=direction,
            marker_color=color
        )
    )


fig_shap.add_vline(
    x=0,
    line_width=1,
    line_color="#8C8073"
)


fig_shap.update_layout(
    title="Local Feature Contributions",

    height=420,

    template="plotly_white",

    xaxis_title=(
        "SHAP contribution to model output"
    ),

    yaxis_title="",

    legend=dict(
        title="",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ),

    margin=dict(
        l=35,
        r=25,
        t=65,
        b=40
    ),

    paper_bgcolor="#FFFDFC",
    plot_bgcolor="#FFFDFC"
)


st.plotly_chart(
    fig_shap,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)


# =========================================================
# ANALYST DECISION
# =========================================================

html("""
<div class="section-title">
Analyst Decision
</div>

<div class="section-subtitle">
Record the manual disposition for this review case.
</div>
""")


analyst_decision = st.radio(
    "Decision",
    [
        "Approve as Legitimate",
        "Confirm Fraud",
        "Escalate for Further Investigation"
    ],
    horizontal=True,
    index=None,
    key=f"decision_{selected_transaction_id}"
)


analyst_note = st.text_area(
    "Analyst note",
    placeholder=(
        "Add a short justification "
        "for the review decision..."
    ),
    key=f"note_{selected_transaction_id}"
)


if st.button(
    "Save analyst decision",
    type="primary"
):

    if analyst_decision is None:

        st.warning(
            "Please select an analyst decision before saving."
        )

    else:

        if "analyst_reviews" not in st.session_state:

            st.session_state[
                "analyst_reviews"
            ] = {}


        st.session_state[
            "analyst_reviews"
        ][
            selected_transaction_id
        ] = {
            "Decision":
                analyst_decision,

            "Note":
                analyst_note
        }


        st.success(
            f"Decision saved for "
            f"Transaction #{selected_transaction_id}."
        )

        st.rerun()


# =========================================================
# SAVED REVIEW HISTORY
# =========================================================

if (
    "analyst_reviews"
    in st.session_state
    and
    st.session_state[
        "analyst_reviews"
    ]
):

    html("""
    <div class="section-title">
    Session Review History
    </div>

    <div class="section-subtitle">
    Manual decisions recorded during the current application session.
    </div>
    """)


    history_rows = []

    for transaction_id, review in (
        st.session_state[
            "analyst_reviews"
        ].items()
    ):

        history_rows.append({
            "Transaction_ID":
                transaction_id,

            "Analyst_Decision":
                review["Decision"],

            "Analyst_Note":
                review["Note"]
        })


    history_df = pd.DataFrame(
        history_rows
    )


    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )