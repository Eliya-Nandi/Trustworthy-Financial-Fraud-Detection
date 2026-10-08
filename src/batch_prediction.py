import numpy as np
import pandas as pd

from src.prediction import (
    FEATURE_COLUMNS,
    get_model,
    get_configuration
)


# =========================================================
# VALIDATE BATCH INPUT
# =========================================================

def validate_batch(df):

    missing_columns = [
        col
        for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    clean_df = df.copy()

    # Make sure model features are numeric
    for col in FEATURE_COLUMNS:
        clean_df[col] = pd.to_numeric(
            clean_df[col],
            errors="coerce"
        )

    invalid_rows = clean_df[
        FEATURE_COLUMNS
    ].isna().any(axis=1)

    if invalid_rows.any():

        bad_rows = (
            np.where(invalid_rows)[0] + 1
        ).tolist()

        raise ValueError(
            "Non-numeric or missing feature values "
            f"found in row(s): {bad_rows[:20]}"
        )

    return clean_df


# =========================================================
# BATCH ANALYSIS
# =========================================================

def analyze_batch(df):

    df = validate_batch(df)

    model = get_model()
    config = get_configuration()

    threshold = float(
        config["Decision_Threshold"]
    )

    conformal_quantile = float(
        config["Marginal_Conformal_Quantile"]
    )

    X = df[FEATURE_COLUMNS]

    # ---------------------------------------------
    # Model probabilities
    # ---------------------------------------------

    fraud_probabilities = (
        model.predict_proba(X)[:, 1]
    )

    # ---------------------------------------------
    # Standard model predictions
    # ---------------------------------------------

    model_predictions = np.where(
        fraud_probabilities >= threshold,
        "Fraud",
        "Legitimate"
    )

    # ---------------------------------------------
    # Marginal conformal prediction
    #
    # Class 0 included when:
    # P(fraud) <= q
    #
    # Class 1 included when:
    # 1 - P(fraud) <= q
    # ---------------------------------------------

    include_normal = (
        fraud_probabilities
        <= conformal_quantile
    )

    include_fraud = (
        (1 - fraud_probabilities)
        <= conformal_quantile
    )

    conformal_decisions = []
    prediction_sets = []
    actions = []

    for normal, fraud in zip(
        include_normal,
        include_fraud
    ):

        if normal and not fraud:

            prediction_sets.append("(0,)")

            conformal_decisions.append(
                "Confident Normal"
            )

            actions.append(
                "No fraud escalation required"
            )

        elif fraud and not normal:

            prediction_sets.append("(1,)")

            conformal_decisions.append(
                "Confident Fraud"
            )

            actions.append(
                "Flag transaction for fraud investigation"
            )

        elif normal and fraud:

            prediction_sets.append("(0, 1)")

            conformal_decisions.append(
                "Human Review"
            )

            actions.append(
                "Escalate transaction for manual review"
            )

        else:

            prediction_sets.append("()")

            conformal_decisions.append(
                "Human Review"
            )

            actions.append(
                "Escalate transaction for manual review"
            )

    # ---------------------------------------------
    # Build output
    # ---------------------------------------------

    result = df.copy()

    result.insert(
        0,
        "Transaction_ID",
        np.arange(1, len(result) + 1)
    )

    result["Fraud_Probability"] = (
        fraud_probabilities
    )

    result["Model_Prediction"] = (
        model_predictions
    )

    result["Prediction_Set"] = (
        prediction_sets
    )

    result["Conformal_Decision"] = (
        conformal_decisions
    )

    result["Recommended_Action"] = (
        actions
    )

    return result