# ==============================================================================
# PROJECT: Customer Churn Prediction 
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    precision_score, 
    f1_score, 
    roc_auc_score, 
    classification_report, 
    confusion_matrix
)

# ------------------------------------------------------------------------------
# STEP 1: LOAD DATASET
# ------------------------------------------------------------------------------
print("=== Step 1: Loading Dataset ===")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"Dataset Shape: {df.shape[0]} Rows, {df.shape[1]} Columns")
print(df.head())

# ------------------------------------------------------------------------------
# STEP 2: DATA CLEANING & PREPROCESSING (FIXED)
# ------------------------------------------------------------------------------
print("\n=== Step 2: Data Cleaning & Preprocessing ===")

# Convert whitespace strings in TotalCharges to NaN and fill with median explicitly
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True), errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Drop customerID column
if 'customerID' in df.columns:
    df.drop(columns=['customerID'], inplace=True)

# Map target variable 'Churn' to binary integers (1 / 0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Fill any remaining numeric NaNs if present
df = df.fillna(df.median(numeric_only=True))

print("Missing Values Check:", df.isnull().sum().sum())

# ------------------------------------------------------------------------------
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------------------------
print("\n=== Step 3: Generating Visualizations (EDA) ===")

plt.figure(figsize=(14, 10))

# Graph 1: Churn Count Distribution
plt.subplot(2, 2, 1)
sns.countplot(x='Churn', data=df, palette='Set2')
plt.title('Overall Churn Count (0 = No, 1 = Yes)')

# Graph 2: Churn by Contract Type
plt.subplot(2, 2, 2)
sns.countplot(x='Contract', hue='Churn', data=df, palette='Set1')
plt.title('Churn Count by Contract Type')

# Graph 3: Tenure Distribution by Churn
plt.subplot(2, 2, 3)
sns.kdeplot(data=df, x='tenure', hue='Churn', fill=True, palette='Set1')
plt.title('Tenure Distribution by Churn Status')

# Graph 4: Monthly Charges vs Churn
plt.subplot(2, 2, 4)
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='Set3')
plt.title('Monthly Charges vs Churn')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# STEP 4: ENCODING & TRAIN-TEST SPLIT
# ------------------------------------------------------------------------------
print("\n=== Step 4: One-Hot Encoding & Train-Test Split ===")

# Generate dummy variables for categorical features
df_encoded = pd.get_dummies(df, drop_first=True)

# Separate features (X) and target variable (y)
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Perform an 80/20 stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply Feature Scaling (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------------------
# STEP 5: MODEL TRAINING & EVALUATION
# ------------------------------------------------------------------------------
print("\n=== Step 5: Model Training & Evaluation ===")

models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), True),
    "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42), False),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42), False)
}

results = []

for name, (model, use_scaling) in models.items():
    X_tr = X_train_scaled if use_scaling else X_train
    X_te = X_test_scaled if use_scaling else X_test
    
    # Train model
    model.fit(X_tr, y_train)
    
    # Make predictions
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    
    # Calculate performance metrics
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    results.append({
        "Model": name,
        "Accuracy (%)": round(acc * 100, 2),
        "Recall (%)": round(rec * 100, 2),
        "Precision (%)": round(prec * 100, 2),
        "F1-Score (%)": round(f1 * 100, 2),
        "ROC-AUC": round(auc, 4)
    })

# Output results table
results_df = pd.DataFrame(results)
print("\n", results_df.to_string(index=False))

# ------------------------------------------------------------------------------
# STEP 6: DETAILED REPORT FOR BEST MODEL (Logistic Regression)
# ------------------------------------------------------------------------------
print("\n=== Step 6: Detailed Classification Report (Logistic Regression) ===")
best_model = LogisticRegression(max_iter=1000)
best_model.fit(X_train_scaled, y_train)
y_pred_best = best_model.predict(X_test_scaled)

print("\nClassification Report:\n", classification_report(y_test, y_pred_best))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()