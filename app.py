import streamlit as st
import pandas as pd
import joblib
from monitoring import generate_prediction_report

artifacts = joblib.load("customer_churn_model.pkl")

model = artifacts["model"]
preprocessor = artifacts["preprocessor"]
threshold = artifacts["threshold"]

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="centered"
)

st.title("Customer Churn Intelligence")
st.write("Predict churn risk and recommend retention actions.")

st.subheader("Customer Details")

with st.form("customer_form"):
    
    gender = st.selectbox("Gender", ["Female", "Male"])

    senior_citizen_label = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )
    senior_citizen = 1 if senior_citizen_label == "Yes" else 0

    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
    online_security = st.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup", ["Yes", "No", "No internet service"]
    )
    device_protection = st.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech Support", ["Yes", "No", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV", ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = st.selectbox(
        "Paperless Billing", ["Yes", "No"]
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
        "Monthly Charges ($)", min_value=0.0, value=70.0
    )
    total_charges = st.number_input(
        "Total Charges ($)", min_value=0.0, value=840.0
    )

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    customer = pd.DataFrame([{
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
        "TotalCharges": total_charges
    }])

    processed_customer = preprocessor.transform(customer)
    probability = model.predict_proba(processed_customer)[0, 1]

    st.subheader("Prediction Results")
    st.metric("Churn Probability", f"{probability:.2%}")

    if probability >= threshold:
        st.error("High Risk")
    else:
        st.success("Low Risk")

    actions = []

    if contract == "Month-to-month":
        actions.append("Offer a suitable long-term contract incentive")
    if tenure <= 12:
        actions.append("Provide early-customer onboarding support")
    if tech_support == "No":
        actions.append("Offer a technical support review")
    if online_security == "No":
        actions.append("Explain available online security options")
    if not actions:
        actions.append("Offer a routine satisfaction check-in")

    st.subheader("Recommended Retention Actions")
    for action in actions:
        st.write(f"- {action}")
