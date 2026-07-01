"""
Train the heart-disease model and save it to disk for the Streamlit app.
Run this ONCE (or whenever the data/model changes):

    python train_and_save.py

It reproduces the exact pipeline from the analysis notebook and writes
`heart_model.joblib`, which app.py loads.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "heart_disease_uci.csv"
THRESHOLD = 0.35  # decision threshold tuned for recall (catch sick patients)

# ----- same preprocessing as the notebook -----
df = pd.read_csv(DATA).drop(columns=["id"])
df["target"] = (df["num"] > 0).astype(int)
df = df.drop(columns=["num"])
for col in ["chol", "trestbps"]:           # recode impossible zeros -> missing
    df[col] = df[col].replace(0, np.nan)
df = df.drop(columns=["dataset"])           # drop hospital-site confounder
df["sex"] = (df["sex"] == "Male").astype(int)

categorical_cols = ["cp", "fbs", "restecg", "exang", "slope", "thal"]
for col in categorical_cols:
    df[col] = df[col].astype("category")

X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = HistGradientBoostingClassifier(
    categorical_features="from_dtype",
    class_weight="balanced",
    random_state=42,
)
model.fit(X_train, y_train)

# An empty 1-row template that carries the exact column order + category dtypes,
# so the app can build inputs that match what the model was trained on.
template = X.iloc[:0].copy()

joblib.dump(
    {
        "model": model,
        "template": template,
        "categorical_cols": categorical_cols,
        "threshold": THRESHOLD,
    },
    HERE / "heart_model.joblib",
)
print("Saved", HERE / "heart_model.joblib")
print("Categories per categorical feature:")
for c in categorical_cols:
    print(f"  {c}: {list(template[c].cat.categories)}")
