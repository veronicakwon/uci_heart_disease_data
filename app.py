# My Heart Disease Risk Screener — built by me!
#
# Run it from this folder with:   streamlit run app.py
# Streamlit re-runs this whole file top-to-bottom every time an input changes.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

st.title("Heart Disease Risk Screener")


col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Your age", 18, 100, 54)

    sex = st.selectbox("Your sex", ["Male", "Female"])

    cp = st.selectbox("Your chest pain type", ["asymptomatic", "atypical angina", "non-anginal", "typical angina"])

    trestbps = st.number_input("Your resting blood pressure", 80, 220, 130)

    chol = st.number_input("Your cholesterol level", 100, 650, 240)

    fbs = st.selectbox("Fasting blood sugar?", ["No", "Yes"])

with col2:

    restecg = st.selectbox("Ecg observation at resting condition", ["normal", "lv hypertrophy", "st-t abnormality"])

    thalch = st.number_input("Your max heart rate", 60, 220, 150)

    exang = st.selectbox("Exercise induced angina", ["No", "Yes"])

    oldpeak = st.number_input("ST depression (oldpeak)", -3.0, 7.0, 1.0, step=0.1)

    slope = st.selectbox("The slope of the peak exercise ST segment", ["upsloping", "flat", "downsloping", "Unknown"])

    ca = st.selectbox("Number of major vessels (0-3) colored by flourosopy", ["0", "1", "2", "3", "Unknown"])

    thal = st.selectbox("Thal (blood)", ["normal", "fixed defect", "reversable defect", "Unknown"])

bundle = joblib.load("heart_model.joblib")
cat_cols = bundle["categorical_cols"]
model = bundle["model"]
template = bundle["template"]
threshold = bundle["threshold"]

# if you click the predict, it will show the result
if st.button("Predict"):
    row = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs == "Yes",
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang == "Yes",
        "oldpeak": oldpeak,
        "slope": np.nan if slope == "Unknown" else slope,
        "ca": np.nan if ca == "Unknown" else int(ca),
        "thal": np.nan if thal == "Unknown" else thal
    }
    X_input = pd.DataFrame([row])[template.columns]
    for col in template.columns:
        X_input[col] = X_input[col].astype(template[col].dtype)

    prob = model.predict_proba(X_input)[0,1]
    st.metric("Probability of heart disease", f"{prob:.0%}")

    if prob >= threshold:
        st.error("⚠️ Elevated risk — recommend follow-up.")
    else:
        st.success("✅ Lower risk.")

    def to_codes(frame):
        out = frame.copy()
        for c in cat_cols:
            out[c] = out[c].cat.codes.replace(-1, np.nan)
        return out

    explainer = shap.TreeExplainer(model)
    sv = explainer(to_codes(X_input))

    contrib = pd.Series(sv.values[0], index=template.columns).sort_values(key=abs, ascending=False)
    st.subheader("Why? Top factors for this patient:")
    st.bar_chart(contrib.head(6))
    st.caption("Positive = pushed toward disease, negative = away (log-odds).")
    




