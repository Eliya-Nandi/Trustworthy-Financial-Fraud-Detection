from pathlib import Path
import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"


def load_model():
    model_path = MODELS_DIR / "final_xgboost_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found. Expected path: {model_path}"
        )

    return joblib.load(model_path)


def load_configuration():
    config_path = MODELS_DIR / "final_research_configuration.pkl"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found. Expected path: {config_path}"
        )

    return joblib.load(config_path)


def load_final_summary():
    file_path = RESULTS_DIR / "final_research_summary.csv"
    return pd.read_csv(file_path)


def load_validation_comparison():
    file_path = RESULTS_DIR / "final_validation_model_comparison.csv"
    return pd.read_csv(file_path)


def load_conformal_results():
    file_path = RESULTS_DIR / "marginal_conformal_alpha_comparison.csv"
    return pd.read_csv(file_path)


def load_human_review_results():
    file_path = RESULTS_DIR / "final_human_review_decisions.csv"
    return pd.read_csv(file_path)


def load_shap_global():
    file_path = RESULTS_DIR / "shap_global_feature_importance.csv"
    return pd.read_csv(file_path)


def load_shap_local():
    file_path = RESULTS_DIR / "shap_local_fraud_explanation.csv"
    return pd.read_csv(file_path)