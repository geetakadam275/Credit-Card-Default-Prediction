# 🏦 Credit Card Default Prediction

[![Vercel Deployment](https://img.shields.io/badge/Deploy-Vercel-black?style=flat&logo=vercel)](https://vercel.com)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?style=flat&logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Model-Decision%20Tree%20(81.78%25)-orange?style=flat&logo=scikit-learn)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Backend-Flask%20WSGI-lightgrey?style=flat&logo=flask)](https://flask.palletsprojects.com)

## 📌 Project Overview

This project predicts whether a credit card customer is likely to default on their next month's payment using Machine Learning. The repository contains a complete end-to-end machine learning pipeline from exploratory data analysis and feature engineering to a live web application deployed on **Vercel Functions** and **Streamlit**.

The project includes:
- **Data Exploration & Cleaning**: Handling anomalies, distributions, and collinearity.
- **Feature Analysis**: Weight of Evidence (WoE), Information Value (IV), and Variance Inflation Factor (VIF).
- **Model Training**: Logistic Regression and Decision Tree Classifier trained on 30,000 UCI credit client records.
- **Production Web Application**: A modern interactive dashboard built with Flask and styled with responsive cards, live risk meter, and explainable risk factors.
- **Serverless API**: JSON endpoints (`POST /predict` and `GET /health`) optimized for Vercel Serverless Functions.

---

## 📂 Project Structure

```
Credit-Card-Default-Prediction/
│
├── api/
│   └── index.py                      # Vercel serverless entrypoint importing Flask app
├── templates/
│   └── index.html                    # Modern interactive fintech web UI
├── app.py                            # Flask application exporting top-level "app" for Vercel
├── streamlit_app.py                  # Streamlit web interface for local or Streamlit Cloud use
├── credit_default_model.pkl          # Trained Decision Tree model (81.78% test accuracy)
├── UCI_Credit_Card.csv               # UCI credit card clients dataset (30,000 records)
├── Credit_Card_Default_Prediction.ipynb # Jupyter notebook with full EDA & training pipeline
├── vercel.json                       # Vercel routing and rewrite configuration
├── requirements.txt                  # Production dependencies optimized for Vercel
├── Workflow.md                       # Machine learning workflow documentation
├── README.md                         # Project documentation
└── .gitignore                        # Git ignore patterns
```

---

## 📊 Dataset & Model

The dataset contains customer demographic information, credit limit, repayment history across 6 months, bill statement amounts, and previous payment amounts.

### Target Variable
- **0 → No Default (Good Customer)**
- **1 → Default (Delinquent Customer)**

### Evaluated Model: Decision Tree Classifier
- **Parameters**: `criterion='gini'`, `max_depth=5`, `random_state=42`
- **Test Accuracy**: **81.78%**
- **Top Predictive Features**:
  1. `PAY_0` (September repayment status) - 67.97% importance
  2. `PAY_2` (August repayment status) - 13.76% importance
  3. `PAY_AMT3` (July payment amount) - 4.28% importance
  4. `LIMIT_BAL` (Credit limit) - 2.19% importance

---

## 🚀 Deployment on Vercel

This repository is pre-configured to deploy on Vercel with zero extra build commands:

1. Import this repository into [Vercel](https://vercel.com/new).
2. Framework Preset: **Other** (Vercel automatically detects Python via `app.py` / `api/index.py`).
3. Root Directory: `./`
4. Click **Deploy**.

Vercel will install dependencies from `requirements.txt` and serve the application via serverless Python execution.

---

## 💻 Running Locally

### Option 1: Run the Flask Web App (Same as Vercel)

```bash
# Clone the repository
git clone https://github.com/geetakadam275/Credit-Card-Default-Prediction.git
cd Credit-Card-Default-Prediction

# Install dependencies
pip install -r requirements.txt

# Start the Flask development server
python app.py
```
Open your browser at `http://127.0.0.1:5000` to access the dashboard.

### Option 2: Run the Streamlit Interface

```bash
pip install streamlit
streamlit run streamlit_app.py
```

---

## 🔌 REST API Documentation

### 1. Predict Default Risk
- **Endpoint**: `POST /predict`
- **Headers**: `Content-Type: application/json`
- **Sample Request**:
```json
{
  "ID": 101,
  "LIMIT_BAL": 150000,
  "SEX": 2,
  "EDUCATION": 1,
  "MARRIAGE": 2,
  "AGE": 32,
  "PAY_0": -1,
  "PAY_2": -1,
  "PAY_3": -1,
  "PAY_4": -1,
  "PAY_5": -1,
  "PAY_6": -1,
  "BILL_AMT1": 5000,
  "BILL_AMT2": 4800,
  "BILL_AMT3": 5100,
  "BILL_AMT4": 4900,
  "BILL_AMT5": 5300,
  "BILL_AMT6": 4700,
  "PAY_AMT1": 5000,
  "PAY_AMT2": 4800,
  "PAY_AMT3": 5100,
  "PAY_AMT4": 4900,
  "PAY_AMT5": 5300,
  "PAY_AMT6": 4700
}
```

- **Sample Response**:
```json
{
  "confidence": 90.9,
  "insights": [
    "Recent repayment status is current or settled duly.",
    "Healthy credit utilization (3.3%).",
    "Paid total outstanding balance in full in September."
  ],
  "prediction": 0,
  "probability_default": 9.1,
  "probability_no_default": 90.9,
  "risk_category": "Low Risk",
  "risk_class": "success",
  "status_label": "No Default",
  "success": true
}
```

### 2. Healthcheck
- **Endpoint**: `GET /health`
- **Response**: `{"features_expected": 24, "model_loaded": true, "platform": "Vercel / Flask Serverless", "status": "healthy"}`

---

## 👩‍💻 Author

**Geeta Kadam**  
GitHub: [geetakadam275](https://github.com/geetakadam275)  
Repository: [Credit-Card-Default-Prediction](https://github.com/geetakadam275/Credit-Card-Default-Prediction)
