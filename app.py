import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

# Top-level WSGI application required by Vercel Python runtime
app = Flask(__name__)
application = app
handler = app

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "credit_default_model.pkl")
CSV_PATH = os.path.join(BASE_DIR, "UCI_Credit_Card.csv")

# WSGI Middleware to normalize Vercel serverless prefix rewrites
class VercelPrefixMiddleware:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path_info = environ.get("PATH_INFO", "")
        # Strip common Vercel serverless function prefixes
        prefixes = ["/api/index.py", "/api/index", "/api", "/app.py", "/app"]
        for prefix in prefixes:
            if path_info == prefix:
                environ["PATH_INFO"] = "/"
                break
            elif path_info.startswith(prefix + "/"):
                environ["PATH_INFO"] = path_info[len(prefix):]
                break
        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPrefixMiddleware(app.wsgi_app)

# 24 model features in exact training order
FEATURES = [
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

_model = None

def get_model():
    """Load cached model or train fallback if needed."""
    global _model
    if _model is not None:
        return _model

    if os.path.exists(MODEL_PATH):
        try:
            _model = joblib.load(MODEL_PATH)
            return _model
        except Exception as e:
            app.logger.warning(f"Failed loading model pickle: {e}. Attempting retraining...")

    # Fallback: train on UCI dataset if available
    if os.path.exists(CSV_PATH):
        try:
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.model_selection import train_test_split
            df = pd.read_csv(CSV_PATH)
            X = df.drop("default.payment.next.month", axis=1)
            y = df["default.payment.next.month"]
            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y
            )
            dt = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=5)
            dt.fit(X_train, y_train)
            try:
                joblib.dump(dt, MODEL_PATH)
            except Exception:
                pass
            _model = dt
            return _model
        except Exception as e:
            app.logger.error(f"Error training fallback model: {e}")

    return None

# Pre-configured sample profiles for rapid UI demonstration
SAMPLE_PROFILES = {
    "low_risk": {
        "title": "Low Risk Customer (Pristine Credit)",
        "data": {
            "ID": 101,
            "LIMIT_BAL": 200000,
            "SEX": 2,          # Female
            "EDUCATION": 1,    # Graduate
            "MARRIAGE": 2,     # Single
            "AGE": 32,
            "PAY_0": -1,       # Paid duly
            "PAY_2": -1,
            "PAY_3": -1,
            "PAY_4": -1,
            "PAY_5": -1,
            "PAY_6": -1,
            "BILL_AMT1": 5200,
            "BILL_AMT2": 4800,
            "BILL_AMT3": 5100,
            "BILL_AMT4": 4900,
            "BILL_AMT5": 5300,
            "BILL_AMT6": 4700,
            "PAY_AMT1": 5200,
            "PAY_AMT2": 4800,
            "PAY_AMT3": 5100,
            "PAY_AMT4": 4900,
            "PAY_AMT5": 5300,
            "PAY_AMT6": 4700,
        }
    },
    "high_risk": {
        "title": "High Risk Customer (Delinquent Status)",
        "data": {
            "ID": 202,
            "LIMIT_BAL": 30000,
            "SEX": 1,          # Male
            "EDUCATION": 2,    # University
            "MARRIAGE": 1,     # Married
            "AGE": 26,
            "PAY_0": 2,        # 2 months payment delay
            "PAY_2": 2,
            "PAY_3": 2,
            "PAY_4": 0,
            "PAY_5": 0,
            "PAY_6": 0,
            "BILL_AMT1": 28500,
            "BILL_AMT2": 29000,
            "BILL_AMT3": 29200,
            "BILL_AMT4": 28000,
            "BILL_AMT5": 27500,
            "BILL_AMT6": 26000,
            "PAY_AMT1": 0,
            "PAY_AMT2": 1000,
            "PAY_AMT3": 0,
            "PAY_AMT4": 500,
            "PAY_AMT5": 0,
            "PAY_AMT6": 0,
        }
    },
    "moderate_risk": {
        "title": "Moderate Risk Customer (Revolving Balance)",
        "data": {
            "ID": 303,
            "LIMIT_BAL": 80000,
            "SEX": 2,
            "EDUCATION": 2,
            "MARRIAGE": 1,
            "AGE": 41,
            "PAY_0": 1,        # 1 month delay
            "PAY_2": 0,
            "PAY_3": 0,
            "PAY_4": 0,
            "PAY_5": 0,
            "PAY_6": 0,
            "BILL_AMT1": 45000,
            "BILL_AMT2": 42000,
            "BILL_AMT3": 39000,
            "BILL_AMT4": 35000,
            "BILL_AMT5": 31000,
            "BILL_AMT6": 28000,
            "PAY_AMT1": 2000,
            "PAY_AMT2": 2000,
            "PAY_AMT3": 1800,
            "PAY_AMT4": 1500,
            "PAY_AMT5": 1500,
            "PAY_AMT6": 1200,
        }
    }
}

def extract_features(source):
    """Extract and validate 24 features from request dict/form."""
    row = []
    defaults = {
        "ID": 1,
        "LIMIT_BAL": 50000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 2,
        "AGE": 30,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 0, "BILL_AMT2": 0, "BILL_AMT3": 0, "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
        "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
    }

    for feat in FEATURES:
        val = source.get(feat)
        if val is None or val == "":
            val = defaults[feat]
        try:
            val = float(val)
        except (ValueError, TypeError):
            val = float(defaults[feat])
        row.append(val)

    return pd.DataFrame([row], columns=FEATURES)

def generate_insights(df_row, proba_default):
    """Generate risk factors and human-readable financial insights."""
    factors = []
    row = df_row.iloc[0]

    # Check repayment status
    if row["PAY_0"] >= 2:
        factors.append(f"September repayment is delayed by {int(row['PAY_0'])} months.")
    elif row["PAY_0"] == 1:
        factors.append("September repayment shows a 1-month payment delay.")
    elif row["PAY_0"] <= 0:
        factors.append("Recent repayment status is current or settled duly.")

    if row["PAY_2"] >= 2:
        factors.append(f"August repayment had a severe delay of {int(row['PAY_2'])} months.")

    # Check credit utilization
    limit = max(row["LIMIT_BAL"], 1.0)
    bill = max(row["BILL_AMT1"], 0.0)
    utilization = (bill / limit) * 100
    if utilization > 80:
        factors.append(f"High credit utilization ({utilization:.1f}% of limit NT${int(limit):,}).")
    elif utilization < 30:
        factors.append(f"Healthy credit utilization ({utilization:.1f}%).")

    # Payment vs Bill ratio
    pay = row["PAY_AMT1"]
    if bill > 0 and pay == 0:
        factors.append("Zero payment recorded towards the latest billing statement.")
    elif bill > 0 and pay >= bill:
        factors.append("Paid total outstanding balance in full in September.")

    return factors

@app.route("/", methods=["GET"])
@app.route("/api", methods=["GET"])
@app.route("/api/index", methods=["GET"])
@app.route("/api/index.py", methods=["GET"])
def index():
    """Render the dashboard UI."""
    return render_template("index.html", default_profile=SAMPLE_PROFILES["low_risk"]["data"])

@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
@app.route("/api/index/predict", methods=["POST"])
@app.route("/api/index.py/predict", methods=["POST"])
def predict():
    """Predict credit card default risk from JSON or form payload."""
    try:
        model = get_model()
        if model is None:
            return jsonify({
                "success": False,
                "error": "Model could not be loaded. Please ensure credit_default_model.pkl exists."
            }), 500

        if request.is_json:
            source = request.get_json(silent=True) or {}
        else:
            source = request.form.to_dict()

        input_df = extract_features(source)
        prediction = int(model.predict(input_df)[0])

        proba_default = 0.0
        proba_no_default = 100.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            proba_no_default = round(float(probs[0]) * 100, 2)
            proba_default = round(float(probs[1]) * 100, 2)

        risk_category = "High Risk" if prediction == 1 or proba_default >= 50 else ("Moderate Risk" if proba_default >= 25 else "Low Risk")
        status_label = "Default" if prediction == 1 else "No Default"
        risk_class = "danger" if prediction == 1 else ("warning" if risk_category == "Moderate Risk" else "success")

        insights = generate_insights(input_df, proba_default)

        result = {
            "success": True,
            "prediction": prediction,
            "status_label": status_label,
            "risk_category": risk_category,
            "risk_class": risk_class,
            "probability_default": proba_default,
            "probability_no_default": proba_no_default,
            "insights": insights,
            "confidence": max(proba_default, proba_no_default)
        }

        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify(result)

        # Standard form submission fallback
        return render_template("index.html", result=result, form_data=source, default_profile=SAMPLE_PROFILES["low_risk"]["data"])

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Get sample profiles for rapid 1-click testing."""
    return jsonify(SAMPLE_PROFILES)

@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health():
    """Vercel / monitoring healthcheck endpoint."""
    model = get_model()
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "features_expected": len(FEATURES),
        "platform": "Vercel / Flask Serverless"
    })

# Fallback: If any GET request doesn't match an exact route, gracefully serve the dashboard UI
@app.errorhandler(404)
def not_found(e):
    if request.method == "GET":
        return render_template("index.html", default_profile=SAMPLE_PROFILES["low_risk"]["data"]), 200
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

if __name__ == "__main__":
    # Local development server
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
