# Trustworthy Financial Fraud Detection

<p align="center">
  <img src="assets/trustworthy_financial_fraud_detection.png"
       alt="Trustworthy Financial Fraud Detection"
       width="100%">
</p>

<p align="center">
  <strong>Explainable and uncertainty-aware financial fraud detection using XGBoost, SHAP, conformal prediction, and human-in-the-loop review.</strong>
</p>

---

## Project Overview

This project presents an end-to-end **trustworthy machine learning pipeline for credit-card fraud detection**.

Instead of treating fraud detection as a simple binary classification task, the system combines:

- **XGBoost** for fraud-risk prediction
- **SHAP** for global and local model explainability
- **Conformal prediction** for uncertainty-aware decisions
- **Human-in-the-loop review** for ambiguous transactions
- **Streamlit** for an interactive decision-support application

The main objective is to demonstrate how a fraud detection system can go beyond predictive performance by also providing **interpretability, uncertainty awareness, and human oversight**.

---

## Key Results

### Final Held-Out Test Performance

| Metric | Result |
|---|---:|
| Precision | **90.91%** |
| Recall | **84.51%** |
| F1 Score | **87.59%** |
| ROC-AUC | **0.985** |
| PR-AUC | **0.887** |
| True Positives | **60** |
| False Negatives | **11** |
| False Positives | **6** |
| True Negatives | **42,483** |

The final model detected **60 of 71 fraudulent transactions** in the held-out test set while producing only **6 false-positive alerts** among **42,489 legitimate transactions**.

Because the dataset is extremely imbalanced, the project emphasizes **precision, recall, F1 score, and PR-AUC** over accuracy.

### 95% Bootstrap Confidence Intervals

| Metric | Estimate | 95% CI |
|---|---:|---:|
| Precision | 0.909 | 0.836 – 0.972 |
| Recall | 0.845 | 0.755 – 0.923 |
| F1 Score | 0.876 | 0.812 – 0.932 |
| ROC-AUC | 0.985 | 0.967 – 0.998 |
| PR-AUC | 0.887 | 0.812 – 0.948 |

These intervals were estimated using **2,000 bootstrap resamples** of the final test predictions.

---

## Dataset

The project uses a public credit-card fraud detection dataset containing European cardholder transactions.

The full dataset is not included in this repository because of its size.

### Dataset Source

[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)

The deployed application uses a small demonstration subset stored in `assets/demo_transactions.csv`, while the research results were produced using the full dataset.

### Original Dataset

- **284,807 transactions**
- **492 fraudulent transactions**
- **31 columns**
- Features: `Time`, `V1–V28`, `Amount`, and `Class`
- No missing values

### After Preprocessing

- **283,726 transactions**
- **473 fraudulent transactions**
- **1,081 exact duplicate rows removed**
- Fraud prevalence: approximately **0.167%**

The variables `V1–V28` are anonymized transformed features. They are therefore treated as statistical inputs and are **not assigned unsupported real-world financial meanings**.


## Research Design

The project uses a four-way stratified split:

| Split | Transactions | Fraud |
|---|---:|---:|
| Training | 170,235 | 284 |
| Validation | 42,559 | 71 |
| Calibration | 28,372 | 47 |
| Test | 42,560 | 71 |

This separation supports distinct stages for:

1. model training
2. validation and threshold selection
3. conformal calibration
4. final held-out evaluation

The test set was reserved for final evaluation and was not used for model or threshold selection.

---

## Model Development

Three primary model families were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

### Validation Summary

The main validation results were approximately:

| Model | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.852 | 0.732 | 0.788 | 0.677 |
| Random Forest | 0.930 | 0.746 | 0.828 | 0.775 |
| XGBoost | 0.900 | 0.761 | 0.824 | 0.762 |

Random Forest achieved the strongest validation PR-AUC and F1, but it also showed a larger train-validation performance gap.

The original XGBoost model was retained as the **primary research model** because it provided a strong overall balance of:

- fraud recall
- F1 score
- PR-AUC
- ROC-AUC
- generalization behavior
- compatibility with the explainability and uncertainty pipeline

The selected XGBoost decision threshold was approximately:

```text
0.8905
```

This threshold was selected using validation data rather than the final test set.

---

## Explainable AI with SHAP

SHAP is used at both **global** and **local** levels.

### Global Explainability

Global SHAP analysis identifies which anonymized features contribute most strongly across the validation sample.

The most influential features included:

`V14`, `V4`, `V12`, `V3`, `V11`, `V10`, `V15`, `V19`, `V26`, and `V8`.

### Local Explainability

For individual transactions, the application shows:

- features pushing the prediction toward **fraud**
- features pushing the prediction toward **legitimate**
- the relative magnitude of those contributions

This provides analysts with an interpretable explanation for each model decision while avoiding unsupported semantic interpretations of the anonymized PCA-derived variables.

---

## Probability Calibration

The selected XGBoost model was also evaluated for probability calibration.

Key calibration results:

- **Brier Score:** approximately `0.00129`
- **Expected Calibration Error (10 bins):** approximately `0.00434`

These aggregate values are low, but they are interpreted cautiously because the dataset is dominated by legitimate transactions.

---

## Uncertainty-Aware Decisions with Conformal Prediction

The project adds a **marginal conformal prediction layer** on top of XGBoost fraud probabilities.

Primary significance level:

```text
alpha = 0.10
```

At this operating point:

| Measure | Result |
|---|---:|
| Target Coverage | 90% |
| Empirical Overall Coverage | **90.30%** |
| Normal-Class Coverage | **90.32%** |
| Fraud-Class Coverage | **76.06%** |
| Human Review Rate | **9.69%** |
| Confident Fraud Decisions | **55** |
| Confident Fraud Precision | **98.18%** |

Instead of forcing every transaction into an automated binary decision, the system can return:

- **Confident Normal**
- **Confident Fraud**
- **Human Review**

Transactions without a sufficiently supported singleton prediction set are routed to human review.

This abstention mechanism is a central part of the trustworthy-AI design.

---

## Conformal Sensitivity

The project also evaluates multiple significance levels to study the trade-off between coverage and review workload.

| Alpha | Target Coverage | Overall Coverage | Fraud Coverage | Human Review Rate |
|---|---:|---:|---:|---:|
| 0.05 | 95% | 95.21% | 80.28% | 4.77% |
| 0.10 | 90% | 90.30% | 76.06% | 9.69% |
| 0.20 | 80% | 79.67% | 71.83% | 20.33% |

The primary setting remains **alpha = 0.10** because it was pre-specified rather than selected from test performance.

---

## Human-in-the-Loop Review

The Streamlit application includes a dedicated analyst review queue for uncertain transactions.

Reviewers can inspect:

- fraud probability
- model prediction
- conformal decision
- transaction amount
- priority level
- local SHAP explanation
- recommended action

The analyst can then record one of three decisions:

- Approve as Legitimate
- Confirm Fraud
- Escalate for Further Investigation

This demonstrates how model uncertainty can be integrated into a practical decision-support workflow instead of being hidden from the user.

---

## Interactive Streamlit Application

The application contains five focused sections.

### 1. Home

Provides an executive overview of:

- core research results
- selected model
- final performance
- explainability
- uncertainty-aware decision routing

### 2. Transaction Analyzer

Analyzes a single transaction using:

- fraud probability
- selected decision threshold
- conformal prediction
- local SHAP explanation
- recommended action

### 3. Batch Analysis

Allows users to upload a compatible CSV file and perform batch fraud screening.

Expected model features:

```text
Time
V1
V2
...
V28
Amount
```

The `Class` column is optional.

The page returns:

- fraud probabilities
- model predictions
- conformal decisions
- review recommendations
- portfolio-level charts
- downloadable screening results

### 4. Human Review Queue

Supports investigation of transactions that the conformal layer routes to manual review.

The review queue includes:

- priority ranking
- transaction details
- fraud risk
- local SHAP explanations
- analyst notes and decisions

### 5. Model Performance

Provides a research-oriented evaluation view containing:

- model benchmarking
- confusion matrix
- ROC curve
- precision-recall curve
- bootstrap confidence intervals
- global SHAP importance
- calibration diagnostics
- conformal sensitivity analysis

---

## Demo Data

The repository includes:

```text
assets/demo_transactions.csv
```

This file exists only to support the interactive portfolio application.

It is intentionally enriched with legitimate and fraudulent examples so that the app can demonstrate:

- fraud detection
- confident fraud decisions
- confident normal decisions
- human review routing

The demo file **does not reproduce the true fraud prevalence of the original dataset** and should not be interpreted as a representative sample of real transaction traffic.

The scientific evaluation results in this repository are based on the full dataset from Kaggle.

---

## Project Structure

```text
Trustworthy-Financial-Fraud-Detection/
│
├── app/
│   ├── Home.py
│   └── pages/
│       ├── 1_Transaction_Analyzer.py
│       ├── 2_Batch_Analysis.py
│       ├── 3_Human_Review_Queue.py
│       └── 4_Model_Performance.py
│
├── assets/
│   ├── demo_transactions.csv
│   └── trustworthy_financial_fraud_detection.png
│
├── figures/
├── models/
│   ├── final_research_configuration.pkl
│   └── final_xgboost_model.pkl
│
├── results/
├── src/
│   ├── batch_prediction.py
│   ├── explainability.py
│   ├── loaders.py
│   └── prediction.py
│
├── trustworthy_fraud_detection.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Running the Application Locally

### 1. Clone the repository

```bash
git clone https://github.com/Eliya-Nandi/Trustworthy-Financial-Fraud-Detection.git
cd Trustworthy-Financial-Fraud-Detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Streamlit application

```bash
streamlit run app/Home.py
```

The application will open in your browser.

---

## Limitations & Responsible Use

This project is a **research and portfolio demonstration** and is not intended for autonomous production financial decision-making.

Important limitations include:

- the dataset is extremely imbalanced
- `V1–V28` are anonymized transformed features with no supported real-world semantic interpretation
- the dataset represents a specific historical transaction environment and may not generalize to other institutions, regions, customer populations, or fraud patterns
- model probabilities and conformal prediction sets depend on the data-generating distribution remaining sufficiently stable
- fraud-class conformal coverage is less reliable than normal-class coverage because only a small number of fraud examples are available in the calibration split
- SHAP values describe model behavior and should not be interpreted as causal explanations
- the demonstration dataset used in the web application is deliberately enriched and does not represent real fraud prevalence

In a real financial deployment, this type of model should be combined with:

- institution-specific validation
- continuous performance monitoring
- data and concept drift detection
- governance and audit procedures
- human review for uncertain or high-impact decisions
- privacy, security, legal, and regulatory controls
- retraining and recalibration when operating conditions change

The application should therefore be understood as a **decision-support research prototype**, not an autonomous fraud adjudication system.

---

## Reproducibility

The repository includes:

- the research notebook
- final model artifacts
- research figures
- final evaluation tables
- application source code
- a deployment-safe demo dataset

To fully reproduce the research from the original data, download the dataset from Kaggle:

[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)

Then place the dataset locally according to the notebook workflow.

The large raw dataset is intentionally excluded from version control.

---

## Technology Stack

- **Python**
- **XGBoost**
- **scikit-learn**
- **SHAP**
- **Conformal Prediction**
- **Pandas**
- **NumPy**
- **Plotly**
- **Streamlit**
- **Matplotlib**
- **Git**
- **GitHub**

---

## Author

**Eliya Christopher Nandi**

Email: [eliyanandi07@gmail.com](mailto:eliyanandi07@gmail.com)  
LinkedIn: [linkedin.com/in/eliyanandi](https://www.linkedin.com/in/eliyanandi)

---

<p align="center">
  <strong>Designed & Developed by Eliya Christopher Nandi · © 2026</strong>
</p>
