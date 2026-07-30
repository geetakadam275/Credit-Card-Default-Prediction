# 🏦 Credit Card Default Prediction

## 📌 Project Overview

This project predicts whether a credit card customer is likely to default on the next month's payment using Machine Learning techniques. The project follows a complete end-to-end machine learning workflow, including data preprocessing, feature engineering, model building, and evaluation.

The project includes:

- Data Loading
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Weight of Evidence (WoE)
- Information Value (IV)
- Variance Inflation Factor (VIF)
- Logistic Regression
- Decision Tree
- Hyperparameter Tuning
- Model Evaluation

---

## 📂 Project Structure

```
Credit-Card-Default-Prediction/
│
├── Credit_Card_Default_Prediction.ipynb
├── credit_card_default.csv
├── credit_card_default_clean.csv
├── credit_card_default_feature_engineered.csv
├── credit_card_default_vif.csv
├── information_value.csv
├── Decision_Tree.png
├── Workflow.md
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📊 Dataset

The dataset contains customer demographic, payment history, bill amount, and payment amount information used to predict whether a customer will default on the next month's credit card payment.

### Target Variable

- **0 → No Default**
- **1 → Default**

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels
- Jupyter Notebook
- Git & GitHub

---

## 📈 Project Workflow

### 1. Data Loading

- Imported required libraries
- Loaded dataset
- Checked dataset shape
- Verified data types

---

### 2. Data Cleaning

- Checked missing values
- Removed duplicate records
- Verified data consistency
- Saved cleaned dataset

---

### 3. Exploratory Data Analysis (EDA)

Performed:

- Dataset overview
- Statistical summary
- Missing value analysis
- Duplicate record analysis
- Correlation analysis
- Outlier detection
- Skewness analysis

---

### 4. Feature Engineering

Applied:

- Feature selection
- Feature scaling
- Train-Test Split
- Prepared dataset for modeling

---

### 5. Weight of Evidence (WoE)

Performed:

- Variable binning
- Good and Bad distribution
- WoE calculation
- WoE transformation

---

### 6. Information Value (IV)

Performed:

- IV calculation
- Feature ranking
- Predictor strength analysis

---

### 7. Variance Inflation Factor (VIF)

Performed:

- Multicollinearity analysis
- VIF calculation
- Removed high VIF features (if required)

---

### 8. Model Building

Implemented:

- Logistic Regression
- Decision Tree Classifier

Performed:

- Model training
- Prediction
- Feature importance analysis

---

### 9. Hyperparameter Tuning

Performed:

- GridSearchCV
- Best parameter selection
- Optimized Decision Tree model

---

### 10. Model Evaluation

Evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

Also included:

- Decision Tree Visualization
- Feature Importance Analysis

---

## 📌 Machine Learning Concepts Covered

- Binary Classification
- Data Preprocessing
- Feature Engineering
- Weight of Evidence (WoE)
- Information Value (IV)
- Multicollinearity
- Variance Inflation Factor (VIF)
- Logistic Regression
- Decision Tree
- Hyperparameter Tuning
- GridSearchCV
- Model Evaluation

---

## 📊 Model Performance

The Logistic Regression and Decision Tree models were trained and evaluated using multiple classification metrics. Their performance was compared to determine the most suitable model for predicting credit card default risk.

---

## 🚀 Future Improvements

- Random Forest Classifier
- XGBoost Classifier
- LightGBM
- ROC-AUC Analysis
- Model Deployment using Streamlit or Flask
- Interactive Dashboard

---

## 👩‍💻 Author

**Geeta Kadam**

GitHub: https://github.com/geetakadam275/Credit-Card-Default-Prediction
