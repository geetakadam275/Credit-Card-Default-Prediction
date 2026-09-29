import streamlit as st

st.title("My Machine Learning Project")

st.write("Welcome to my ML application!")

st.header("Enter Your Details")

age = st.number_input("Enter your age", min_value=1, max_value=100)

if st.button("Submit"):
    st.success(f"You entered age: {age}")
