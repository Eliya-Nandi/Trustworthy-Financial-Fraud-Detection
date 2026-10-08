import numpy as np
import pandas as pd
import shap

from src.loaders import load_model


# =========================================================
# EXPECTED FEATURES
# =========================================================

FEATURE_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


# =========================================================
# MODEL / EXPLAINER CACHE
# =========================================================

_model = None
_explainer = None


def get_model():
    global _model

    if _model is None:
        _model = load_model()

    return _model


def get_explainer():
    global _explainer

    if _explainer is None:
        model = get_model()
        _explainer = shap.TreeExplainer(model)

    return _explainer


# =========================================================
# VALIDATE INPUT
# =========================================================

def validate_transaction(transaction_df):

    missing_columns = [
        col
        for col in FEATURE_COLUMNS
        if col not in transaction_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    return transaction_df[FEATURE_COLUMNS].copy()


# =========================================================
# LOCAL SHAP EXPLANATION
# =========================================================

def explain_transaction(transaction_df, top_n=10):

    transaction_df = validate_transaction(
        transaction_df
    )

    explainer = get_explainer()

    shap_values = explainer.shap_values(
        transaction_df
    )

    # XGBoost binary classification typically returns
    # a 2D array: (n_samples, n_features)
    if isinstance(shap_values, list):
        shap_array = np.asarray(
            shap_values[-1]
        )
    else:
        shap_array = np.asarray(
            shap_values
        )

    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, -1]

    local_shap = shap_array[0]

    feature_values = (
        transaction_df.iloc[0].to_numpy()
    )

    explanation_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Feature_Value": feature_values,
        "SHAP_Value": local_shap
    })

    explanation_df[
        "Absolute_SHAP"
    ] = explanation_df[
        "SHAP_Value"
    ].abs()

    explanation_df = explanation_df.sort_values(
        "Absolute_SHAP",
        ascending=False
    ).reset_index(drop=True)

    top_features = (
        explanation_df
        .head(top_n)
        .copy()
    )

    fraud_drivers = (
        explanation_df[
            explanation_df["SHAP_Value"] > 0
        ]
        .head(top_n)
        .copy()
    )

    normal_drivers = (
        explanation_df[
            explanation_df["SHAP_Value"] < 0
        ]
        .head(top_n)
        .copy()
    )

    return {
        "all_features": explanation_df,
        "top_features": top_features,
        "fraud_drivers": fraud_drivers,
        "normal_drivers": normal_drivers
    }