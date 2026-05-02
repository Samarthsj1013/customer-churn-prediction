import streamlit as st

def render_header():
    st.set_page_config(
        page_title="Churn Prediction Dashboard",
        page_icon="🔮",
        layout="wide"
    )
    st.title("🔮 Customer Churn Prediction System")
    st.markdown("**Production-ready ML system | XGBoost | AUC: 0.9364**")
    st.divider()

def render_metric_cards(churn_prob: float, risk_level: str, prediction: int):
    col1, col2, col3, col4 = st.columns(4)

    color = {"Low": "green", "Medium": "orange", "High": "red"}[risk_level]

    with col1:
        st.metric("Churn Probability", f"{churn_prob:.1%}")
    with col2:
        st.metric("Risk Level", risk_level)
    with col3:
        st.metric("Prediction", "Will Churn" if prediction == 1 else "Will Stay")
    with col4:
        st.metric("Model AUC", "0.9364")

def render_recommendation(risk_level: str, churn_prob: float):
    recommendations = {
        "High": ("🚨 HIGH RISK", "Immediately offer a loyalty discount or contract upgrade. Assign a retention specialist.", "error"),
        "Medium": ("⚠️ MEDIUM RISK", "Send a personalized email with a special offer. Consider a free service upgrade.", "warning"),
        "Low": ("✅ LOW RISK", "Customer is stable. Continue standard engagement and monitor quarterly.", "success")
    }
    title, msg, type_ = recommendations[risk_level]
    getattr(st, type_)(f"**{title}**: {msg}")

def render_sidebar_inputs():
    st.sidebar.header("Customer Information")
    st.sidebar.subheader("Demographics")

    age = st.sidebar.slider("Age", 18, 80, 35)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    state = st.sidebar.selectbox("State", [
    "Maharashtra", "Karnataka", "Tamil Nadu", "Delhi", "Uttar Pradesh",
    "Gujarat", "Rajasthan", "West Bengal", "Telangana", "Kerala"
])

    st.sidebar.subheader("Account Details")
    tenure = st.sidebar.slider("Tenure (months)", 1, 72, 12)
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
    payment = st.sidebar.selectbox("Payment Method", ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"])

    st.sidebar.subheader("Services")
    internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber Optic", "No"])
    phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    streaming = st.sidebar.selectbox("Streaming TV", ["Yes", "No"])
    security = st.sidebar.selectbox("Online Security", ["Yes", "No"])
    support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])

    st.sidebar.subheader("Financials")
    monthly = st.sidebar.slider("Monthly Charges ($)", 20.0, 120.0, 65.0)
    total = st.sidebar.slider("Total Charges ($)", 0.0, 10000.0, monthly * tenure)
    complaints = st.sidebar.slider("Number of Complaints", 0, 10, 0)
    calls = st.sidebar.slider("Support Calls", 0, 10, 1)
    usage = st.sidebar.slider("Avg Daily Usage (GB)", 0.0, 20.0, 3.5)
    late = st.sidebar.slider("Late Payments", 0, 10, 0)
    promo = st.sidebar.selectbox("Promotion Offered", ["No", "Yes"])

    return {
        "age": age, "gender": gender, "state": state,
        "tenure_months": tenure, "contract_type": contract,
        "payment_method": payment, "internet_service": internet,
        "phone_service": phone, "streaming_tv": streaming,
        "online_security": security, "tech_support": support,
        "monthly_charges": monthly, "total_charges": total,
        "num_complaints": complaints, "num_support_calls": calls,
        "avg_daily_usage_gb": usage, "late_payments": late,
        "promotion_offered": promo
    }