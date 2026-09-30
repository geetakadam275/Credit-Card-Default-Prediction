import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="centered"
)

st.title("Credit Card Default Prediction")
st.write(
    "Enter customer details to predict the likelihood "
    "of credit card default using a trained Decision Tree model."
)

MODEL_PATH = "credit_default_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Trained model not found. Please upload "
        "credit_default_model.pkl to your GitHub repository."
    )
    st.stop()

model = load_model()

# Features used in your notebook's Decision Tree model
features = [
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

st.subheader("Customer Information")

with st.form("prediction_form"):

    st.markdown("### Personal Details")

    ID = st.number_input(
        "Customer ID",
        min_value=1,
        value=1,
        step=1
    )

    LIMIT_BAL = st.number_input(
        "Credit Limit (NT dollars)",
        min_value=0,
        value=50000,
        step=1000
    )

    SEX = st.selectbox(
        "Sex (1 = Male, 2 = Female)",
        options=[1, 2]
    )

    EDUCATION = st.selectbox(
        "Education",
        options=[1, 2, 3, 4, 5, 6, 0]
    )

    MARRIAGE = st.selectbox(
        "Marriage Status",
        options=[1, 2, 3, 0]
    )

    AGE = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )

    st.markdown("### Repayment Status")

    PAY_0 = st.number_input(
        "Repayment Status - September",
        value=0,
        step=1
    )

    PAY_2 = st.number_input(
        "Repayment Status - August",
        value=0,
        step=1
    )

    PAY_3 = st.number_input(
        "Repayment Status - July",
        value=0,
        step=1
    )

    PAY_4 = st.number_input(
        "Repayment Status - June",
        value=0,
        step=1
    )

    PAY_5 = st.number_input(
        "Repayment Status - May",
        value=0,
        step=1
    )

    PAY_6 = st.number_input(
        "Repayment Status - April",
        value=0,
        step=1
    )

    st.markdown("### Bill Amounts")

    BILL_AMT1 = st.number_input(
        "Bill Amount - September",
        value=0,
        step=1000
    )

    BILL_AMT2 = st.number_input(
        "Bill Amount - August",
        value=0,
        step=1000
    )

    BILL_AMT3 = st.number_input(
        "Bill Amount - July",
        value=0,
        step=1000
    )

    BILL_AMT4 = st.number_input(
        "Bill Amount - June",
        value=0,
        step=1000
    )

    BILL_AMT5 = st.number_input(
        "Bill Amount - May",
        value=0,
        step=1000
    )

    BILL_AMT6 = st.number_input(
        "Bill Amount - April",
        value=0,
        step=1000
    )

    st.markdown("### Payment Amounts")

    PAY_AMT1 = st.number_input(
        "Payment Amount - September",
        value=0,
        step=1000
    )

    PAY_AMT2 = st.number_input(
        "Payment Amount - August",
        value=0,
        step=1000
    )

    PAY_AMT3 = st.number_input(
        "Payment Amount - July",
        value=0,
        step=1000
    )

    PAY_AMT4 = st.number_input(
        "Payment Amount - June",
        value=0,
        step=1000
    )

    PAY_AMT5 = st.number_input(
        "Payment Amount - May",
        value=0,
        step=1000
    )

    PAY_AMT6 = st.number_input(
        "Payment Amount - April",
        value=0,
        step=1000
    )

    submitted = st.form_submit_button(
        "Predict Default Risk"
    )

if submitted:

    input_data = pd.DataFrame(
        [[
            ID,
            LIMIT_BAL,
            SEX,
            EDUCATION,
            MARRIAGE,
            AGE,
            PAY_0,
            PAY_2,
            PAY_3,
            PAY_4,
            PAY_5,
            PAY_6,
            BILL_AMT1,
            BILL_AMT2,
            BILL_AMT3,
            BILL_AMT4,
            BILL_AMT5,
            BILL_AMT6,
            PAY_AMT1,
            PAY_AMT2,
            PAY_AMT3,
            PAY_AMT4,
            PAY_AMT5,
            PAY_AMT6,
        ]],
        columns=features
    )

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            "Prediction: Default"
        )
    else:
        st.success(
            "Prediction: No Default"
        )

    st.caption(
        "This is a machine learning prediction, "
        "not a financial decision or guarantee."
    )
