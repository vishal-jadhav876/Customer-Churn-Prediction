# 📊 Telco Customer Churn Prediction & Analytics

An end-to-end Data Analytics and Machine Learning project designed to predict customer churn for a telecommunications company. By analyzing customer demographics, account details, and subscribed services, this project identifies key factors driving customer attrition and builds classification models to proactively flag high-risk customers.

---

## 📌 Project Overview
Customer churn is a critical metric for subscription-based businesses. Retaining existing customers is significantly more cost-effective than acquiring new ones. 

This project performs Exploratory Data Analysis (EDA) on over **7,000+ customer records**, cleans and preprocesses categorical and numerical features, and trains multiple machine learning classifiers to predict whether a customer is likely to churn (`Yes` / `No`).

---

## 📂 Dataset Information
* **Source:** IBM Telco Customer Churn Dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
* **Total Records:** 7,043 rows
* **Total Features:** 21 columns
* **Target Variable:** `Churn` (1 = Churned, 0 = Retained)

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn`

---

## 📈 Model Performance & Evaluation

| Model | Accuracy (%) | Recall (%) | Precision (%) | F1-Score (%) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **80.70%** | **56.68%** | **65.84%** | **60.92%** | **0.8416** |
| **Gradient Boosting** | 79.84% | 51.34% | 65.31% | 57.49% | 0.8425 |
| **Random Forest** | 78.64% | 49.20% | 62.37% | 55.01% | 0.8251 |

---

## 🔑 Key Insights & Business Recommendations
1. **Contract Type:** Customers on Month-to-Month contracts have the highest churn rate. Incentivizing 1-year or 2-year long-term contracts can reduce churn significantly.
2. **Tenure Impact:** High churn is observed in the first 0–12 months of customer onboarding. Special onboarding discounts and support can improve retention.
3. **Monthly Charges:** Customers with higher monthly charges show higher churn. Offering tailored product bundles or budget-friendly plans can retain sensitive segments.

---

## ⚙️ How to Run the Project

### 1. Clone the Repository
```bash
git clone [https://github.com/vishal-jadhav876/Customer-Churn-Prediction.git](https://github.com/vishal-jadhav876/Customer-Churn-Prediction.git)
cd Customer-Churn-Prediction