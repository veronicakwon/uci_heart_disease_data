# ❤️ Heart Disease Risk Prediction

A machine-learning project that predicts the presence of heart disease from clinical
data — with a strong emphasis on **model explainability** and **data-quality auditing**,
plus an interactive Streamlit demo.

> **Educational project — not medical advice.**

---

## Overview

Using the UCI Heart Disease dataset (920 patients pooled from four hospitals), this
project trains a gradient-boosted model to flag patients at risk of heart disease. The
work goes beyond raw accuracy: it uses **SHAP** to interrogate *why* the model makes its
predictions, which surfaced — and led to fixing — a hidden data-leakage problem.

## Key Results

| Metric | Value |
| --- | --- |
| Accuracy | **83.2%** |
| ROC-AUC | **0.903** |
| Recall @ 0.35 threshold | **0.93** (catches 93% of true cases) |
| Precision @ 0.35 threshold | **0.81** |

The decision threshold is deliberately lowered to **0.35** to prioritize *recall* — in a
clinical screening tool, missing a sick patient is worse than a false alarm.

## 🔍 Highlight: The Data-Quality Audit

The most important finding came from explainability, not accuracy. SHAP revealed the model
behaving "backwards" — e.g. *lower* cholesterol predicting *more* disease. The cause:

- One hospital (Switzerland) recorded missing cholesterol as `0` — medically impossible.
- That hospital had a 93% disease rate, so `cholesterol = 0` had quietly become a proxy for
  *"which hospital the patient came from."*
- The model was partly learning the **hospital**, not the **medicine** — a shortcut that
  would fail on patients from any new clinic.

**Fix:** recode impossible zeros as missing, and drop the hospital-site column. Accuracy
stayed essentially the same while recall *improved* — confirming the model's real skill
never depended on the artifact, and it now generalizes far better.

## Methods

- **Model:** `HistGradientBoostingClassifier` (scikit-learn) — handles missing values and
  categorical features natively, so no imputation or one-hot encoding is needed.
- **Missing data:** left in place (native handling) rather than imputed — appropriate given
  `ca` (~66%), `thal` (~53%), and `slope` (~34%) are heavily incomplete.
- **Class imbalance:** `class_weight='balanced'`.
- **Explainability:** permutation importance + SHAP (beeswarm, waterfall, dependence plots).

## Project Structure

```
healthcare project/
├── data/
│   └── heart_disease_uci.csv          # UCI Heart Disease dataset
├── notebooks/
│   ├── modeling.ipynb                 # main analysis: prep → model → evaluation → SHAP
│   └── conclusion_executive_summary.ipynb
├── projects in python/
│   └── conclusion_executive_summary.md
├── streamlit_app/
│   ├── train_and_save.py              # trains the model and saves heart_model.joblib
│   ├── heart_model.joblib             # saved model bundle (created by the script)
│   └── app.py                         # interactive risk-screener web app
└── README.md
```

## How to Run

**Requirements:** Python 3.10+. Install everything with:

```bash
pip install -r requirements.txt
```

### 1. The analysis notebook

```bash
cd notebooks
jupyter notebook modeling.ipynb
```
Run all cells to reproduce the preprocessing, model, evaluation, and SHAP explanations.

### 2. The interactive app

```bash
cd streamlit_app
python train_and_save.py     # run once to (re)create heart_model.joblib
streamlit run app.py         # launches the web app at http://localhost:8501
```
Enter a patient's values, click **Predict**, and see the risk estimate plus a per-patient
SHAP breakdown of the top contributing factors.

## Next Steps

- **Cost-based threshold optimization** — set the threshold from the real clinical cost of a
  missed case vs. a false alarm.
- **Public deployment** — host the Streamlit app (e.g. Streamlit Community Cloud) for a
  shareable link.

## Data Source

[UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) (via Kaggle).
