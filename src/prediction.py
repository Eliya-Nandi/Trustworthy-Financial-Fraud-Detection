import pandas as pd


from src.loaders import (
    load_model,
    load_configuration
)


# =========================================================
# EXPECTED MODEL FEATURES
# =========================================================

FEATURE_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


# =========================================================
# LOAD MODEL + CONFIGURATION
# =========================================================

_model = None
_config = None


def get_model():
    global _model

    if _model is None:
        _model = load_model()

    return _model


def get_configuration():
    global _config

    if _config is None:
        _config = load_configuration()

    return _config


# =========================================================
# INPUT VALIDATION
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
# CONFORMAL PREDICTION
# =========================================================

def get_conformal_prediction(
    fraud_probability,
    conformal_quantile
):

    prediction_set = []

    # Candidate class 0:
    # nonconformity score = P(fraud)
    if fraud_probability <= conformal_quantile:
        prediction_set.append(0)

    # Candidate class 1:
    # nonconformity score = 1 - P(fraud)
    if (
        1 - fraud_probability
        <= conformal_quantile
    ):
        prediction_set.append(1)

    prediction_set = tuple(prediction_set)

    if prediction_set == (0,):

        decision = "Confident Normal"

        recommended_action = (
            "No fraud escalation required"
        )

    elif prediction_set == (1,):

        decision = "Confident Fraud"

        recommended_action = (
            "Flag transaction for fraud investigation"
        )

    else:

        decision = "Human Review"

        recommended_action = (
            "Escalate transaction for manual review"
        )

    return (
        prediction_set,
        decision,
        recommended_action
    )


# =========================================================
# SINGLE TRANSACTION PREDICTION
# =========================================================

def analyze_transaction(transaction_df):

    transaction_df = validate_transaction(
        transaction_df
    )

    model = get_model()
    config = get_configuration()

    fraud_probability = float(
        model.predict_proba(
            transaction_df
        )[0, 1]
    )

    threshold = float(
        config["Decision_Threshold"]
    )

    conformal_quantile = float(
        config[
            "Marginal_Conformal_Quantile"
        ]
    )

    model_prediction = (
        "Fraud"
        if fraud_probability >= threshold
        else "Legitimate"
    )

    prediction_set, decision, recommended_action = (
        get_conformal_prediction(
            fraud_probability,
            conformal_quantile
        )
    )

    return {
        "Fraud_Probability": fraud_probability,
        "Model_Prediction": model_prediction,
        "Decision_Threshold": threshold,
        "Conformal_Quantile": conformal_quantile,
        "Prediction_Set": prediction_set,
        "Conformal_Decision": decision,
        "Recommended_Action": recommended_action
    }