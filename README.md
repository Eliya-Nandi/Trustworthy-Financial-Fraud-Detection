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

## Overview

This project develops an end-to-end **trustworthy machine learning pipeline for credit-card fraud detection**. Rather than treating fraud detection as a simple binary classification problem, the system combines:

- **XGBoost** for fraud risk prediction
- **SHAP** for model explainability
- **Conformal prediction** for uncertainty-aware decisions
- **Human-in-the-loop review** for ambiguous transactions
- **Streamlit** for an interactive decision-support interface

The goal is to demonstrate how a fraud detection system can go beyond predictive accuracy by also providing **interpretability, uncertainty awareness, and human oversight**.

---

## Key Results

### Final Held-Out Test Performance

| Metric          |     Result |
| --------------- | ---------: |
| Precision       | **90.91%** |
| Recall          | **84.51%** |
| F1 Score        | **87.59%** |
| ROC-AUC         |  **0.985** |
| PR-AUC          |  **0.887** |
| True Positives  |     **60** |
| False Negatives |     **11** |
| False Positives |      **6** |
| True Negatives  | **42,483** |

The model detected **60 of 71 fraudulent transactions** in the held-out test set while producing only **6 false-positive alerts** among 42,489 legitimate transactions.

Because the dataset is extremely imbalanced, **precision, recall, F1 score, and PR-AUC** are emphasized over accuracy.

### 95% Bootstrap Confidence Intervals

| Metric    | Estimate |        95% CI |
| --------- | -------: | ------------: |
| Precision |    0.909 | 0.836 – 0.972 |
| Recall    |    0.845 | 0.755 – 0.923 |
| F1 Score  |    0.876 | 0.812 – 0.932 |
| ROC-AUC   |    0.985 | 0.967 – 0.998 |
| PR-AUC    |    0.887 | 0.812 – 0.948 |

---

## Dataset

The project uses the public **credit-card fraud detection dataset** commonly distributed through Kaggle and originally associated with European cardholder transactions.

### Original dataset

- **284,807 transactions**
- **492 fraudulent transactions**
- **31 columns**
- Features: `Time`, `V1–V28`, `Amount`, and `Class`
- No missing values

### After preprocessing

- **283,726 transactions**
- **473 fraudulent transactions**
- **1,081 exact duplicate rows removed**
- Fraud prevalence: approximately **0.167%**

The variables `V1–V28` are anonymized transformed features. They are therefore treated as statistical inputs and are **not assigned unsupported real-world financial meanings**.

The full raw dataset is intentionally **not included in this repository**. A small demonstration dataset is provided only for the deployed portfolio application.

---

## Research Workflow

The project follows a four-way stratified split:

| Split       | Transactions | Fraud |
| ----------- | -----------: | ----: |
| Training    |      170,235 |   284 |
| Validation  |       42,559 |    71 |
| Calibration |       28,372 |    47 |
| Test        |       42,560 |    71 |

This separation supports distinct stages for:

1. model training
2. validation and threshold selection
3. conformal calibration
4. final held-out evaluation

The test set was reserved for final evaluation.

---

## Model Development

Three primary model families were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

Random Forest achieved strong validation performance, but the original XGBoost configuration was retained as the **primary research model** because it provided a strong overall balance of:

- fraud recall
- F1 score
- PR-AUC
- ROC-AUC
- generalization behavior
- compatibility with the explainability and uncertainty pipeline

The selected decision threshold was determined using validation data rather than the final test set.

---

## Explainable AI with SHAP

SHAP is used at both **global** and **local** levels.

### Global explanation

Global SHAP analysis identifies which anonymized model features contribute most strongly across the validation sample.

The most influential features included:

`V14`, `V4`, `V12`, `V3`, `V11`, `V10`, `V15`, `V19`, `V26`, and `V8`.

### Local explanation

For individual transactions, the application shows:

- features pushing the prediction toward **fraud**
- features pushing the prediction toward **legitimate**
- relative contribution strength

This provides analysts with an interpretable explanation for each model decision while avoiding unsupported semantic claims about anonymized PCA-derived variables.

---

## Uncertainty-Aware Decisions with Conformal Prediction

The project adds a **marginal conformal prediction layer** on top of the XGBoost fraud probabilities.

Primary significance level:

`alpha = 0.10`

At this operating point:

| Measure                    |     Result |
| -------------------------- | ---------: |
| Target Coverage            |        90% |
| Empirical Overall Coverage | **90.30%** |
| Normal-Class Coverage      | **90.32%** |
| Fraud-Class Coverage       | **76.06%** |
| Human Review Rate          |  **9.69%** |
| Confident Fraud Decisions  |     **55** |
| Confident Fraud Precision  | **98.18%** |

Instead of forcing every transaction into an automated binary decision, the system can produce:

- **Confident Normal**
- **Confident Fraud**
- **Human Review**

Transactions without a sufficiently supported singleton prediction set are routed to human review.

This abstention mechanism is a central part of the trustworthy-AI design.

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

This demonstrates how model uncertainty can be integrated into a practical decision-support workflow rather than hidden from the user.

---

## Interactive Application

The Streamlit application contains five focused sections:

### Home

Executive overview of the trustworthy fraud detection system and core research results.

### Transaction Analyzer

Analyze a single transaction using:

- fraud probability
- optimized threshold
- conformal prediction
- local SHAP explanation
- recommended action

### Batch Analysis

Upload a compatible CSV file and perform batch fraud screening.

Expected model features:

`Time`, `V1–V28`, `Amount`

The `Class` column is optional.

### Human Review Queue

Investigate transactions that the conformal layer routes to manual review.

### Model Performance

Explore:

- model benchmarking
- confusion matrix
- ROC and precision-recall curves
- bootstrap confidence intervals
- global SHAP importance
- probability calibration
- conformal sensitivity

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
│   └── trustworthy_financial_fraud_detection_dashboard.png
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

### 3. Start Streamlit

```bash
streamlit run app/Home.py
```

The application will open in your browser.

---

## Demo Data

The repository includes:

```text
assets/demo_transactions.csv
```

This dataset is intentionally enriched with legitimate and fraudulent examples so that the application can demonstrate all parts of the workflow.

It **does not reproduce the true fraud prevalence of the original dataset** and should not be interpreted as a representative population sample.

The scientific model evaluation reported in this repository is based on the original full research dataset and the predefined held-out test split.

---

## Limitations & Responsible Use

This project is a **research and portfolio demonstration** and is not intended for autonomous production financial decision-making.

Important limitations include:

- the dataset is highly imbalanced
- `V1–V28` are anonymized transformed features with no supported real-world semantic interpretation
- the dataset represents a specific historical transaction environment and may not generalize to other institutions, regions, customer populations, or fraud patterns
- model probabilities and conformal prediction sets depend on the data-generating distribution remaining sufficiently stable
- fraud-class conformal coverage is less reliable than normal-class coverage because only a small number of fraud examples are available in the calibration split
- SHAP values describe model behavior and should not be interpreted as causal explanations
- the demonstration dataset used in the web application is deliberately enriched and does not represent real fraud prevalence

In a real financial deployment, this type of model should be combined with:

- institution-specific validation
- continuous monitoring
- drift detection
- governance and audit procedures
- human review for uncertain or high-impact decisions
- appropriate privacy, security, legal, and regulatory controls

The application is therefore presented as a **decision-support research prototype**, not an autonomous fraud adjudication system.

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
- **Git / GitHub**

---

## Author

**Eliya Christopher Nandi**

Email: [eliyanandi07@gmail.com](mailto:eliyanandi07@gmail.com)  
LinkedIn: [linkedin.com/in/eliyanandi](https://www.linkedin.com/in/eliyanandi)

---

<p align="center">
  <strong>Designed & Developed by Eliya Christopher Nandi · © 2026</strong>
</p>
