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
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:8000/predict"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — small visual polish on top of the base theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; max-width: 1100px;}
        h1 {font-weight: 800;}
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 1rem 1rem 0.5rem 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {gap: 4px;}
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 8px 18px;
        }
        div.stButton > button, div.stFormSubmitButton > button {
            border-radius: 10px;
            font-weight: 600;
            padding: 0.6rem 1rem;
        }
        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(255,255,255,0.08);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — context / about panel
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📉 About")
    st.write(
        "This tool calls a churn-prediction API and returns the probability "
        "that a customer will churn, plus the top SHAP factors behind that "
        "prediction."
    )
    st.divider()
    st.caption("API endpoint")
    st.code(API_URL, language="text")
    st.divider()
    with st.expander("How to run the API"):
        st.code("uvicorn api.main:app --reload", language="bash")
    st.caption("Built with Streamlit")

st.title("📉 Customer Churn Predictor")
st.write(
    "Enter a customer's details below to predict their likelihood of churning, "
    "along with the top factors driving that prediction."
)
st.divider()

# ---------------------------------------------------------------------------
# Input form — organized into tabs instead of one long two-column block
# ---------------------------------------------------------------------------
with st.form("customer_form"):
    tab_profile, tab_services, tab_billing = st.tabs(
        ["👤 Profile", "🛠️ Services", "💳 Billing & Contract"]
    )

    with tab_profile:
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
        with col2:
            senior_citizen = st.selectbox(
                "Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
            )
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])

    with tab_services:
        col1, col2 = st.columns(2)
        with col1:
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

    with tab_billing:
        col1, col2 = st.columns(2)
        with col1:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
        with col2:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.35, step=0.5)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=845.50, step=1.0)

    st.write("")
    submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

# ---------------------------------------------------------------------------
# On submit: call the API
# ---------------------------------------------------------------------------
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

            # --- Headline metrics row ---
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Churn Prediction", "Yes ⚠️" if prediction == "Yes" else "No ✅")
            with m2:
                st.metric("Churn Probability", f"{proba:.1%}")
            with m3:
                risk_label = (
                    "High" if proba >= 0.66 else "Medium" if proba >= 0.33 else "Low"
                )
                st.metric("Risk Level", risk_label)

            left, right = st.columns([1, 1.4])

            # --- Gauge chart for probability ---
            with left:
                gauge_color = "#e74c3c" if proba >= 0.5 else "#2ecc71"
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=proba * 100,
                        number={"suffix": "%"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": gauge_color},
                            "steps": [
                                {"range": [0, 33], "color": "rgba(46,204,113,0.25)"},
                                {"range": [33, 66], "color": "rgba(241,196,15,0.25)"},
                                {"range": [66, 100], "color": "rgba(231,76,60,0.25)"},
                            ],
                        },
                        title={"text": "Churn Risk"},
                    )
                )
                fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)

            # --- Factor bar chart + table ---
            with right:
                st.subheader("Top factors driving this prediction")
                factors_df = pd.DataFrame(
                    list(result["top_factors"].items()), columns=["Feature", "SHAP Impact"]
                )
                factors_df["Direction"] = factors_df["SHAP Impact"].apply(
                    lambda x: "Increases churn risk" if x > 0 else "Decreases churn risk"
                )
                factors_df = factors_df.sort_values("SHAP Impact")

                bar_fig = go.Figure(
                    go.Bar(
                        x=factors_df["SHAP Impact"],
                        y=factors_df["Feature"],
                        orientation="h",
                        marker_color=[
                            "#e74c3c" if v > 0 else "#2ecc71" for v in factors_df["SHAP Impact"]
                        ],
                    )
                )
                bar_fig.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title="SHAP value (impact on prediction)",
                )
                st.plotly_chart(bar_fig, use_container_width=True)

            with st.expander("View factor details as a table"):
                st.dataframe(
                    factors_df.sort_values("SHAP Impact", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the API. Make sure it's running: "
                "`uvicorn api.main:app --reload` in a separate terminal."
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")