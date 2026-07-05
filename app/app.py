import pandas as pd

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ----------------------------
# Load Trained Pipeline
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.pkl"

pipeline = joblib.load(MODEL_PATH)

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📞",
    layout="wide"
)

st.title("📞 Customer Churn Prediction Dashboard")
st.write("Enter customer information to predict whether the customer is likely to churn.")

st.markdown("---")

# ----------------------------
# Customer Information
# ----------------------------

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox("Gender", ["Female", "Male"])

    senior = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox("Partner", ["Yes", "No"])

    dependents = st.selectbox("Dependents", ["Yes", "No"])

    tenure = st.slider("Tenure (Months)", 0, 72, 12)

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

with col2:

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=800.0
    )

# ----------------------------
# Prediction
# ----------------------------

if st.button("Predict Churn", use_container_width=True):

    input_df = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    prediction = pipeline.predict(input_df)

    probability = pipeline.predict_proba(input_df)

    churn_probability = probability[0][1]

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("🔴 Customer is likely to churn.")
    else:
        st.success("🟢 Customer is unlikely to churn.")

    st.metric(
        "Churn Probability",
        f"{churn_probability*100:.2f}%"
    )

    if churn_probability >= 0.75:
        st.error("Risk Level: High")
    elif churn_probability >= 0.50:
        st.warning("Risk Level: Medium")
    else:
        st.success("Risk Level: Low")

    st.subheader("Business Recommendation")

    if prediction[0] == 1:
        st.write("""
- Offer a discount or promotional plan.
- Recommend switching to a long-term contract.
- Contact the customer with a personalized retention offer.
- Provide priority customer support.
        """)
    else:
        st.write("""
- Continue regular customer engagement.
- Offer loyalty rewards.
- Recommend premium services where appropriate.
        """)