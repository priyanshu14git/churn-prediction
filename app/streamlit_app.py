"""
app/streamlit_app.py
---------------------
Streamlit frontend for the Customer Churn Prediction API.

Usage (from project root, with venv activated, API already running separately):
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Predictor")
st.write(
    "Enter a customer's details below to predict their likelihood of churning, "
    "along with the top factors driving that prediction."
)

st.divider()

# --- Input form ---
with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.35, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=845.50, step=1.0)

    submitted = st.form_submit_button("Predict Churn", use_container_width=True)

# --- On submit: call the API ---
if submitted:
    payload = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    with st.spinner("Getting prediction from the model..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            st.divider()

            proba = result["churn_probability"]
            prediction = result["churn_prediction"]

            if prediction == "Yes":
                st.error(f"⚠️ High churn risk — predicted probability: **{proba:.1%}**")
            else:
                st.success(f"✅ Low churn risk — predicted probability: **{proba:.1%}**")

            st.progress(proba)

            st.subheader("Top factors driving this prediction")
            factors_df = pd.DataFrame(
                list(result["top_factors"].items()), columns=["Feature", "SHAP Impact"]
            )
            factors_df["Direction"] = factors_df["SHAP Impact"].apply(
                lambda x: "Increases churn risk" if x > 0 else "Decreases churn risk"
            )
            st.dataframe(factors_df, use_container_width=True, hide_index=True)

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the API. Make sure it's running: "
                "`uvicorn api.main:app --reload` in a separate terminal."
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")