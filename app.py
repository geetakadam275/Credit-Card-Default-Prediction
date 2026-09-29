
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳"
)

st.title("Credit Card Default Prediction")
st.write(
    "Enter customer details to predict credit card default."
)

MODEL_PATH = "credit_default_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Trained model file not found. "
        "Please upload credit_default_model.pkl to GitHub."
    )
    st.stop()

model = load_model()

if not hasattr(model, "feature_names_in_"):
    st.error(
        "The model does not provide feature names. "
        "The input fields must be configured manually."
    )
    st.stop()

features = list(model.feature_names_in_)

st.subheader("Customer Details")

with st.form("prediction_form"):
    values = {}

    for feature in features:
        values[feature] = st.number_input(
            feature,
            value=0.0
        )

    submitted = st.form_submit_button("Predict Default Risk")

if submitted:
    input_data = pd.DataFrame(
        [values],
        columns=features
    )

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Prediction: Default")
    else:
        st.success("Prediction: No Default")
