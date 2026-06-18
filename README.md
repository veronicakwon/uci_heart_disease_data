UCI Heart Disease Prediction

A machine learning project that predicts the presence of heart disease using the UCI Heart Disease dataset, with a focus on maximizing patient safety through clinical optimization of the classification threshold.


Problem Statement

Heart disease is one of the leading causes of death worldwide. Early and accurate detection is critical — but in a clinical screening context, missing a sick patient (false negative) is far more dangerous than a false alarm (false positive). This project builds a binary classifier optimized not just for overall accuracy, but for recall on the positive class, ensuring high-risk patients are flagged before leaving the clinic.


Dataset


Source: UCI Heart Disease Dataset (via Kaggle)
Size: 920 patients across 4 clinical sites (Cleveland, Hungary, Switzerland, VA Long Beach)
Target: Binary — 0 = no heart disease, 1 = heart disease present (originally a 0–4 severity scale, converted to binary)
Features: Age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, max heart rate, exercise-induced angina, ST depression, major vessels, thalassemia, and more



Approach

Preprocessing


Continuous features (e.g. cholesterol, blood pressure): imputed with median
Categorical features (e.g. slope, fbs): imputed with mode
Thalassemia (thal): Over 50% missing — rather than blindly imputing, missing values were encoded as a separate thal_Unknown category, then dropped from the final feature set as uninformative. This preserves the signal in thal_normal and thal_reversable defect without introducing bias.
Target encoding: Severity levels 1–4 collapsed into a single positive class (1 = heart disease present)
One-hot encoding applied to thal and sex


Modeling


Algorithm: Random Forest Classifier (n_estimators=100, class_weight='balanced')
Train/test split: 80/20 with random_state=42
Scaling: StandardScaler applied to all features


Threshold Tuning

The default 0.50 classification threshold yielded a Class 1 recall of only 0.78 — meaning 22% of sick patients would be missed. To address this, the threshold was manually lowered to 0.35, accepting a slightly higher false alarm rate in exchange for catching more true positives.


Results

MetricDefault Threshold (0.50)Tuned Threshold (0.35)Overall Accuracy~79%79.35%Class 1 Recall0.780.89Class 1 Precision—0.78

By tuning the threshold, the model now correctly identifies 89% of all actual heart disease cases, while maintaining strong overall accuracy. In a healthcare triage pipeline, this model functions as a reliable defensive screen that flags high-risk patients for further evaluation.


Key Findings


Maximum heart rate (thalch) and number of major vessels (ca) were the strongest predictors of heart disease
Retaining thal_normal and thal_reversable defect provided a clean contrast between healthy and high-risk states
class_weight='balanced' in the Random Forest was essential given the slight class imbalance in the dataset



How to Run

Requirements:

pip install pandas numpy matplotlib seaborn scikit-learn

Run the model:

bashpython modeling.py

Make sure heart_disease_uci.csv is in the same directory as modeling.py.


Next Steps


Hyperparameter tuning (max_depth, n_estimators) to recover Class 0 precision without compromising the 89% recall benchmark
Compare against Logistic Regression and XGBoost baselines
Add SHAP values for individual prediction explainability
Build a simple Streamlit app for interactive risk scoring
