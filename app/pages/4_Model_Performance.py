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


RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
MODELS_DIR = PROJECT_DIR / "models"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Model Performance",
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
    margin-top: 2.1rem;
}

.section-subtitle {
    color: #6B625B;
    font-size: 0.77rem;
    margin-top: 0.18rem;
    margin-bottom: 0.8rem;
}


/* INSIGHT PANEL */

.insight-panel {
    background: #FFF9EE;
    border: 1px solid #E7D3A8;
    border-left: 4px solid #B89146;
    border-radius: 10px;
    padding: 0.95rem 1rem;
}

.insight-label {
    color: #8A6A3B;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.insight-text {
    color: #4A4039;
    font-size: 0.8rem;
    line-height: 1.55;
    margin-top: 0.25rem;
}


/* IMAGE / PLOT CONTAINERS */

div[data-testid="stPlotlyChart"] {
    background: #FFFDFC;
    border: 1px solid #E4D8C8;
    border-radius: 14px;
}

[data-testid="stImage"] {
    background: #FFFDFC;
    border: 1px solid #E4D8C8;
    border-radius: 14px;
    padding: 0.4rem;
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
# LOAD DATA
# =========================================================

@st.cache_data
def load_csv(filename):

    path = RESULTS_DIR / filename

    if not path.exists():
        return None

    return pd.read_csv(path)


summary = load_csv(
    "final_research_summary.csv"
)

comparison = load_csv(
    "final_validation_model_comparison.csv"
)

shap_global = load_csv(
    "shap_global_feature_importance.csv"
)

conformal = load_csv(
    "marginal_conformal_alpha_comparison.csv"
)


if summary is None:
    st.error(
        "final_research_summary.csv was not found."
    )
    st.stop()


summary_dict = dict(
    zip(
        summary["Metric"],
        summary["Value"]
    )
)


# =========================================================
# HEADER
# =========================================================

html("""
<div class="page-eyebrow">
Research Evaluation & Diagnostics
</div>

<div class="page-title">
Model Performance & Research Insights
</div>

<div class="page-subtitle">
Explore validation benchmarking, held-out test performance, statistical
uncertainty, explainability, calibration and conformal prediction behavior
for the trustworthy financial fraud detection system.
</div>
""")


# =========================================================
# HEADLINE METRICS
# =========================================================

m1, m2, m3, m4, m5 = st.columns(5)


headline_metrics = [
    (
        m1,
        "Precision",
        f"{summary_dict['Test Precision']:.1%}",
        "Fraud alerts confirmed"
    ),

    (
        m2,
        "Fraud Recall",
        f"{summary_dict['Test Recall']:.1%}",
        "Actual fraud detected"
    ),

    (
        m3,
        "F1 Score",
        f"{summary_dict['Test F1']:.1%}",
        "Precision–recall balance"
    ),

    (
        m4,
        "ROC-AUC",
        f"{summary_dict['Test ROC-AUC']:.3f}",
        "Overall discrimination"
    ),

    (
        m5,
        "PR-AUC",
        f"{summary_dict['Test PR-AUC']:.3f}",
        "Rare-class discrimination"
    )
]


for col, label, value, note in headline_metrics:

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
# VALIDATION MODEL BENCHMARKING
# =========================================================

html("""
<div class="section-title">
Validation Model Benchmarking
</div>

<div class="section-subtitle">
Comparison of baseline, ensemble and gradient-boosting models using
the validation set before final model selection.
</div>
""")


if comparison is not None:

    model_col, table_col = st.columns(
        [1.2, 1],
        gap="large"
    )


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


        performance_long[
            "Metric"
        ] = performance_long[
            "Metric"
        ].replace({
            "PR_AUC": "PR-AUC"
        })


        fig_models = px.bar(
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


        fig_models.update_layout(
            height=460,
            template="plotly_white",

            xaxis=dict(
                title="Score",
                range=[0.60, 1.00],
                tickformat=".0%",
                gridcolor="#EFE6DA"
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

            margin=dict(
                l=25,
                r=25,
                t=60,
                b=35
            ),

            paper_bgcolor="#FFFDFC",
            plot_bgcolor="#FFFDFC",

            font=dict(
                family="Segoe UI, Arial, sans-serif",
                color="#4A4039"
            )
        )


        st.plotly_chart(
            fig_models,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )


    with table_col:

        display_comparison = (
            comparison.copy()
        )

        for col in [
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
            "PR_AUC"
        ]:

            if col in display_comparison.columns:

                display_comparison[
                    col
                ] = display_comparison[
                    col
                ].map(
                    lambda x: f"{x:.3f}"
                )


        st.dataframe(
            display_comparison,
            use_container_width=True,
            hide_index=True,
            height=355
        )


        html("""
        <div class="insight-panel">

        <div class="insight-label">
        Model Selection Insight
        </div>

        <div class="insight-text">
        Random Forest achieved the strongest validation PR-AUC and F1,
        but exhibited a larger train–validation performance gap.
        The original XGBoost model was retained as the primary system
        because it provided a strong balance of recall, F1, PR-AUC,
        ROC-AUC and generalization for the trustworthy-AI pipeline.
        </div>

        </div>
        """)


# =========================================================
# HELD-OUT TEST EVALUATION
# =========================================================

html("""
<div class="section-title">
Held-Out Test Evaluation
</div>

<div class="section-subtitle">
Final evaluation of the selected XGBoost model using the untouched test split
and the classification threshold selected on validation data.
</div>
""")


tn = int(
    summary_dict["True Negatives"]
)

fp = int(
    summary_dict["False Positives"]
)

fn = int(
    summary_dict["False Negatives"]
)

tp = int(
    summary_dict["True Positives"]
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

confusion_matrix = [
    [tn, fp],
    [fn, tp]
]


fig_cm = go.Figure(
    data=go.Heatmap(
        z=confusion_matrix,

        x=[
            "Predicted Legitimate",
            "Predicted Fraud"
        ],

        y=[
            "Actual Legitimate",
            "Actual Fraud"
        ],

        text=[
            [
                f"{tn:,}",
                f"{fp:,}"
            ],
            [
                f"{fn:,}",
                f"{tp:,}"
            ]
        ],

        texttemplate="%{text}",

        textfont=dict(
            size=20,
            color="#2B2118"
        ),

        colorscale=[
            [0.00, "#FFFDFC"],
            [0.20, "#F1E5CF"],
            [0.55, "#D7BC83"],
            [1.00, "#B89146"]
        ],

        showscale=False,

        hovertemplate=(
            "%{y}<br>"
            "%{x}<br>"
            "Transactions: %{text}"
            "<extra></extra>"
        )
    )
)


fig_cm.update_layout(
    title="Final Test Confusion Matrix",

    height=390,

    margin=dict(
        l=40,
        r=30,
        t=60,
        b=50
    ),

    xaxis=dict(
        title="Predicted Class",
        side="bottom"
    ),

    yaxis=dict(
        title="True Class",
        autorange="reversed"
    ),

    paper_bgcolor="#FFFDFC",
    plot_bgcolor="#FFFDFC",

    font=dict(
        family="Segoe UI, Arial, sans-serif",
        color="#4A4039"
    )
)


st.plotly_chart(
    fig_cm,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)


cm_col, metrics_col = st.columns(
    [1.1, 1],
    gap="large"
)


with cm_col:

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix,

            x=[
                "Predicted Legitimate",
                "Predicted Fraud"
            ],

            y=[
                "Actual Legitimate",
                "Actual Fraud"
            ],

            text=[
                [
                    f"{tn:,}",
                    f"{fp:,}"
                ],
                [
                    f"{fn:,}",
                    f"{tp:,}"
                ]
            ],

            texttemplate="%{text}",

            textfont={
                "size": 18,
                "color": "#2B2118"
            },

            colorscale=[
                [0.0, "#FFF9EE"],
                [0.5, "#E7D3A8"],
                [1.0, "#B89146"]
            ],

            showscale=False
        )
    )


    fig_cm.update_layout(
        title="Final Test Confusion Matrix",
        height=390,

        margin=dict(
            l=30,
            r=30,
            t=60,
            b=40
        ),

        xaxis_title="Predicted Class",
        yaxis_title="True Class",

        paper_bgcolor="#FFFDFC",
        plot_bgcolor="#FFFDFC",

        font=dict(
            family="Segoe UI, Arial, sans-serif",
            color="#4A4039"
        )
    )


    st.plotly_chart(
        fig_cm,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


with metrics_col:

    test_metrics_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "PR-AUC"
        ],

        "Value": [
            summary_dict[
                "Test Accuracy"
            ],

            summary_dict[
                "Test Precision"
            ],

            summary_dict[
                "Test Recall"
            ],

            summary_dict[
                "Test F1"
            ],

            summary_dict[
                "Test ROC-AUC"
            ],

            summary_dict[
                "Test PR-AUC"
            ]
        ]
    })


    fig_test = px.bar(
        test_metrics_df,
        x="Value",
        y="Metric",
        orientation="h",
        text="Value",
        title="Final Test Metrics"
    )


    fig_test.update_traces(
        marker_color="#B89146",
        texttemplate="%{x:.3f}",
        textposition="outside"
    )


    fig_test.update_layout(
        height=390,
        template="plotly_white",

        xaxis=dict(
            title="Score",
            range=[0.80, 1.01],
            gridcolor="#EFE6DA"
        ),

        yaxis=dict(
            title="",
            autorange="reversed"
        ),

        margin=dict(
            l=30,
            r=60,
            t=60,
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
        fig_test,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


html(f"""
<div class="insight-panel">

<div class="insight-label">
Test-Set Interpretation
</div>

<div class="insight-text">
The final model correctly detected {tp} of {tp + fn} fraudulent test
transactions while producing only {fp} false-positive alerts among
{tn + fp:,} legitimate transactions. Accuracy is reported for completeness,
but precision, recall, F1 and PR-AUC are more informative under the
extreme class imbalance present in this dataset.
</div>

</div>
""")


# =========================================================
# ROC & PRECISION-RECALL CURVES
# =========================================================

html("""
<div class="section-title">
Discrimination Curves
</div>

<div class="section-subtitle">
Threshold-independent evaluation of overall discrimination and rare-class
precision–recall performance.
</div>
""")


roc_path = (
    FIGURES_DIR
    / "xgboost_final_test_roc_curve.png"
)

pr_path = (
    FIGURES_DIR
    / "xgboost_final_test_precision_recall_curve.png"
)


curve1, curve2 = st.columns(2)


with curve1:

    if roc_path.exists():

        st.image(
            str(roc_path),
            caption=(
                "Final XGBoost ROC curve "
                f"(ROC-AUC = "
                f"{summary_dict['Test ROC-AUC']:.3f})"
            ),
            use_container_width=True
        )

    else:

        st.info(
            "ROC curve image was not found."
        )


with curve2:

    if pr_path.exists():

        st.image(
            str(pr_path),
            caption=(
                "Final XGBoost Precision–Recall curve "
                f"(PR-AUC = "
                f"{summary_dict['Test PR-AUC']:.3f})"
            ),
            use_container_width=True
        )

    else:

        st.info(
            "Precision–Recall curve image was not found."
        )


# =========================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =========================================================

html("""
<div class="section-title">
Statistical Uncertainty
</div>

<div class="section-subtitle">
Bootstrap confidence intervals quantify uncertainty around final held-out
test performance.
</div>
""")


bootstrap_candidates = [
    "bootstrap_confidence_intervals.csv",
    "xgboost_bootstrap_confidence_intervals.csv",
    "final_bootstrap_confidence_intervals.csv"
]


bootstrap_df = None


for filename in bootstrap_candidates:

    path = RESULTS_DIR / filename

    if path.exists():

        bootstrap_df = pd.read_csv(
            path
        )

        break


if bootstrap_df is not None:

    st.dataframe(
        bootstrap_df,
        use_container_width=True,
        hide_index=True
    )

else:

    # Use the final verified bootstrap results
    bootstrap_df = pd.DataFrame({
        "Metric": [
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
            "PR_AUC"
        ],

        "Point_Estimate": [
            0.909091,
            0.845070,
            0.875912,
            0.984725,
            0.886794
        ],

        "CI_95_Lower": [
            0.835559,
            0.754702,
            0.812488,
            0.966640,
            0.811917
        ],

        "CI_95_Upper": [
            0.972222,
            0.923106,
            0.931828,
            0.998268,
            0.948055
        ]
    })


    ci_chart = bootstrap_df.copy()

    ci_chart[
        "Lower_Error"
    ] = (
        ci_chart[
            "Point_Estimate"
        ]
        -
        ci_chart[
            "CI_95_Lower"
        ]
    )

    ci_chart[
        "Upper_Error"
    ] = (
        ci_chart[
            "CI_95_Upper"
        ]
        -
        ci_chart[
            "Point_Estimate"
        ]
    )


    fig_ci = go.Figure()


    fig_ci.add_trace(
        go.Scatter(
            x=ci_chart[
                "Point_Estimate"
            ],

            y=ci_chart[
                "Metric"
            ],

            mode="markers",

            marker=dict(
                size=11,
                color="#B89146"
            ),

            error_x=dict(
                type="data",

                symmetric=False,

                array=ci_chart[
                    "Upper_Error"
                ],

                arrayminus=ci_chart[
                    "Lower_Error"
                ],

                color="#7B5E3B",
                thickness=2
            ),

            hovertemplate=(
                "<b>%{y}</b><br>"
                "Estimate: %{x:.3f}"
                "<extra></extra>"
            )
        )
    )


    fig_ci.update_layout(
        title="95% Bootstrap Confidence Intervals",

        height=360,

        template="plotly_white",

        xaxis=dict(
            title="Metric value",
            range=[0.70, 1.01],
            gridcolor="#EFE6DA"
        ),

        yaxis=dict(
            title=""
        ),

        margin=dict(
            l=30,
            r=30,
            t=60,
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
        fig_ci,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


    with st.expander(
        "View bootstrap confidence-interval table"
    ):

        st.dataframe(
            bootstrap_df,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# GLOBAL SHAP
# =========================================================

html("""
<div class="section-title">
Global Explainability
</div>

<div class="section-subtitle">
Mean absolute SHAP values identify the variables that contributed most strongly
to the selected XGBoost model across the validation sample.
</div>
""")


if shap_global is not None:

    top_shap = (
        shap_global
        .head(12)
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
        title="Global SHAP Feature Importance"
    )


    fig_shap.update_traces(
        marker_color="#B89146"
    )


    fig_shap.update_layout(
        height=430,
        template="plotly_white",

        xaxis=dict(
            title="Mean absolute SHAP value",
            gridcolor="#EFE6DA"
        ),

        yaxis_title="",

        margin=dict(
            l=30,
            r=30,
            t=60,
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


    html("""
    <div class="insight-panel">

    <div class="insight-label">
    Explainability Note
    </div>

    <div class="insight-text">
    The anonymized V1–V28 variables are transformed features, so their
    contribution can be interpreted at the model level but should not be
    assigned unsupported real-world financial meanings.
    </div>

    </div>
    """)


# =========================================================
# CALIBRATION
# =========================================================

html("""
<div class="section-title">
Probability Calibration
</div>

<div class="section-subtitle">
Calibration diagnostics assess the reliability of predicted probability scores.
</div>
""")


reliability_path = (
    FIGURES_DIR
    / "xgboost_reliability_curve.png"
)

probability_path = (
    FIGURES_DIR
    / "xgboost_probability_distribution.png"
)


cal1, cal2 = st.columns(2)


with cal1:

    if reliability_path.exists():

        st.image(
            str(reliability_path),
            caption="XGBoost reliability curve",
            use_container_width=True
        )


with cal2:

    if probability_path.exists():

        st.image(
            str(probability_path),
            caption=(
                "Distribution of predicted fraud probabilities"
            ),
            use_container_width=True
        )


html("""
<div class="insight-panel">

<div class="insight-label">
Calibration Interpretation
</div>

<div class="insight-text">
The selected XGBoost model achieved a Brier score of approximately 0.00129
and a 10-bin Expected Calibration Error of approximately 0.00434.
These aggregate values are low, but they should be interpreted cautiously
because the dataset is dominated by legitimate transactions.
</div>

</div>
""")


# =========================================================
# CONFORMAL SENSITIVITY
# =========================================================

html("""
<div class="section-title">
Conformal Prediction Sensitivity
</div>

<div class="section-subtitle">
The significance level controls the trade-off among empirical coverage,
fraud-class coverage and human-review workload.
</div>
""")


if conformal is not None:

    conf_left, conf_right = st.columns(
        [1.2, 1],
        gap="large"
    )


    with conf_left:

        fig_cov = go.Figure()


        fig_cov.add_trace(
            go.Scatter(
                x=conformal[
                    "Target_Coverage"
                ],

                y=conformal[
                    "Overall_Coverage"
                ],

                mode="lines+markers",

                name="Overall coverage",

                line=dict(
                    color="#B89146",
                    width=3
                )
            )
        )


        fig_cov.add_trace(
            go.Scatter(
                x=conformal[
                    "Target_Coverage"
                ],

                y=conformal[
                    "Fraud_Coverage"
                ],

                mode="lines+markers",

                name="Fraud coverage",

                line=dict(
                    color="#B5473C",
                    width=2.5
                )
            )
        )


        fig_cov.add_trace(
            go.Scatter(
                x=conformal[
                    "Target_Coverage"
                ],

                y=conformal[
                    "Normal_Coverage"
                ],

                mode="lines+markers",

                name="Normal coverage",

                line=dict(
                    color="#2F7D6D",
                    width=2.5,
                    dash="dot"
                )
            )
        )


        fig_cov.update_layout(
            title="Coverage Across Alpha Levels",

            height=390,

            template="plotly_white",

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

            margin=dict(
                l=30,
                r=25,
                t=60,
                b=40
            ),

            paper_bgcolor="#FFFDFC",
            plot_bgcolor="#FFFDFC"
        )


        st.plotly_chart(
            fig_cov,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )


    with conf_right:

        fig_review = px.line(
            conformal,
            x="Target_Coverage",
            y="Human_Review_Rate",
            markers=True,
            title=(
                "Coverage vs Human Review"
            )
        )


        fig_review.update_traces(
            line=dict(
                color="#C68A2D",
                width=3
            ),

            marker=dict(
                size=9
            )
        )


        fig_review.update_layout(
            height=390,
            template="plotly_white",

            xaxis=dict(
                title="Target coverage",
                tickformat=".0%",
                gridcolor="#EFE6DA"
            ),

            yaxis=dict(
                title="Human review rate",
                tickformat=".0%",
                gridcolor="#EFE6DA"
            ),

            margin=dict(
                l=30,
                r=25,
                t=60,
                b=40
            ),

            paper_bgcolor="#FFFDFC",
            plot_bgcolor="#FFFDFC"
        )


        st.plotly_chart(
            fig_review,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )


    with st.expander(
        "View conformal sensitivity table"
    ):

        st.dataframe(
            conformal,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# FINAL RESEARCH TAKEAWAY
# =========================================================

html("""
<div class="section-title">
Research Takeaway
</div>
""")


html(f"""
<div class="insight-panel">

<div class="insight-label">
Trustworthy-AI Evaluation
</div>

<div class="insight-text">
The final system combines strong fraud discrimination
(PR-AUC {summary_dict['Test PR-AUC']:.3f}),
fraud recall of {summary_dict['Test Recall']:.1%},
transaction-level SHAP explanations and a conformal abstention mechanism.
At the primary alpha level of 0.10, approximately
{summary_dict['Conformal Human Review Rate']:.1%}
of transactions were routed to human review rather than receiving an
unsupported automated decision.
</div>

</div>
""")