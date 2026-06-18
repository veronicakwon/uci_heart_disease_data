#!/usr/bin/env python
# coding: utf-8

# In[163]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (downloaded from Kaggle)
df = pd.read_csv('heart_disease_uci.csv')

print("Look at the first 5 patients")
print(df.head())

print("Check for missing values and column data types")
print(df.info())
print("Number of missing values")
print(df.isna().sum())


# In[164]:


# Handling missing values

for col in ['slope', 'fbs', 'exang', 'restecg']:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in ['trestbps', 'thalch', 'chol', 'oldpeak']:
    df[col] = df[col].fillna(df[col].median())


# In[165]:


# Make missing values in 'ca' and 'thal' a separate category (because in clinical datasets, data is often missing not at random) 
df['thal'] = df['thal'].fillna('Unknown')
df['ca'] = df['ca'].fillna(-1)

print("Number of missing values: now")
# Check if every value is filled
print(df.isna().sum())


# In[166]:


# Dropping id

df = df.drop(columns = ['id'])


# In[167]:


# Make 'sex' column binary

df['sex'] = pd.get_dummies(df['sex'], drop_first = True, dtype=int)
df['sex']


# In[168]:


# Turn num column into a binary classificiation problem: 0 = no heart disease, 1 = heart disease
# Convert 1, 2, 3, 4 into 1
df['target'] = (df['num'] > 0).astype(int)
df = df.drop(columns = ['num'])
print(df['target'])

# Convert fbs column into binary classification problem: 0 = fasting blood sugar <= 120 mg/dl, 1 = fasting blood sugar > 120 mg/dl)
df['fbs'] = (df['fbs'] == True).astype(int)
print(df['fbs'])

# One-hot encode the 'thal' column
thal_dummies = pd.get_dummies(df['thal'], prefix='thal', dtype=int)

# Join the new columns back to the dataframe and drop the original 'thal'
df = pd.concat([df, thal_dummies], axis=1).drop(columns=['thal'])


# In[169]:


# Calculate correlation of all features with just 'num'
features = ["age", "sex", "trestbps", "chol", "fbs", "thalch", "target", "ca", "thal_fixed defect", "thal_normal", "thal_reversable defect", "thal_Unknown"]
target_corr = df[features].corr()['target'].drop('target').sort_values(ascending=False)

# Plot it
plt.figure(figsize=(8, 5))
sns.barplot(x=target_corr.values, y=target_corr.index, hue=target_corr.index, palette="viridis")
plt.title("Feature Correlation with Heart Disease Severity (num)", fontsize=14)
plt.xlabel("Correlation Coefficient")
plt.ylabel("Features")
plt.axvline(0, color='black', linestyle='--', linewidth=1) # Adds a center line at 0
plt.show()


# In[170]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

features.remove("target")
features.remove("thal_Unknown")
features.remove("thal_fixed defect")
# features.remove("ca")

# features.remove("thal_normal")
# features.remove("thal_reversable defect")
X = df[features]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[171]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)
accuracy = rf_model.score(X_test_scaled, y_test)
print(f"Model Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))


# In[172]:


# Notice: class 1 - recall score is only 0.78. To make this better, let's manually adjust the prediction threshold
# 1. Get the raw probabilities for the positive class (1) instead of hard predictions
y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

# 2. Lower the threshold to 0.35 (If probability >= 35%, flag as heart disease)
y_pred_adjusted = (y_prob >= 0.35).astype(int)

# 3. Check the new scores
print("--- Adjusted Threshold (0.35) ---")
print(classification_report(y_test, y_pred_adjusted))


# In[173]:


# Grab feature importances from trained Random Forest
importances = rf_model.feature_importances_
feat_importances = pd.Series(importances, index=features).sort_values(ascending=False)

# Plot it
plt.figure(figsize=(7, 4))
sns.barplot(x=feat_importances.values, y=feat_importances.index, hue=feat_importances.index, palette="mako", legend=False)
plt.title("Feature Importance in Heart Disease Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

