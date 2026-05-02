import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.graph_objects as go
from src.models.predictor import ChurnPredictor

st.set_page_config(page_title="What-If Analysis", page_icon="🔬", layout="wide")

@st.cache_resource
def load_predictor():
    return ChurnPredictor()

st.title("🔬 What-If Analysis")
st.markdown("Simulate how changes to a customer's profile affect their churn risk. Use this to plan retention strategies.")

predictor = load_predictor()

st.subheader("Configure Base Customer")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Age", 18, 80, 35)
    tenure = st.slider("Tenure (months)", 1, 72, 12)
    monthly = st.slider("Monthly Charges (₹)", 20.0, 120.0, 75.0)
with col2:
    contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No"])
    security = st.selectbox("Online Security", ["No", "Yes"])
    tech = st.selectbox("Tech Support", ["No", "Yes"])
with col3:
    complaints = st.slider("Number of Complaints", 0, 10, 1)
    calls = st.slider("Support Calls", 0, 10, 2)
    late = st.slider("Late Payments", 0, 5, 0)
    state = st.selectbox("State", ["Maharashtra", "Karnataka", "Tamil Nadu", "Delhi",
                                    "Uttar Pradesh", "Gujarat", "Rajasthan",
                                    "West Bengal", "Telangana", "Kerala"])

base = {
    "age": age, "gender": "Male", "state": state,
    "tenure_months": tenure, "contract_type": contract,
    "payment_method": "Credit Card", "internet_service": internet,
    "phone_service": "Yes", "streaming_tv": "No",
    "online_security": security, "tech_support": tech,
    "monthly_charges": monthly, "total_charges": monthly * tenure,
    "num_complaints": complaints, "num_support_calls": calls,
    "avg_daily_usage_gb": 3.5, "late_payments": late,
    "promotion_offered": "No"
}

base_result = predictor.predict(base)
base_prob = base_result["churn_probability"]

st.divider()
col1, col2, col3 = st.columns(3)
col1.metric("Current Churn Risk", f"{base_prob:.1%}")
col2.metric("Risk Level", base_result["risk_level"])
col3.metric("Prediction", "Will Churn" if base_result["churn_prediction"] == 1 else "Will Stay")
st.divider()

st.subheader("Retention Scenario Simulator")

scenarios = {
    "📋 Upgrade to Two Year Contract": {**base, "contract_type": "Two Year"},
    "💰 Offer 20% Monthly Discount": {**base, "monthly_charges": monthly * 0.8, "total_charges": monthly * 0.8 * tenure},
    "🔒 Add Online Security": {**base, "online_security": "Yes"},
    "🛠️ Add Tech Support": {**base, "tech_support": "Yes"},
    "✅ Resolve All Complaints": {**base, "num_complaints": 0, "num_support_calls": 0},
    "🎁 Offer Promotion": {**base, "promotion_offered": "Yes"},
    "📋+🔒 Contract Upgrade + Security": {**base, "contract_type": "Two Year", "online_security": "Yes"},
    "🎯 Full Retention Package": {
        **base, "contract_type": "Two Year",
        "online_security": "Yes", "tech_support": "Yes",
        "promotion_offered": "Yes", "num_complaints": 0,
        "monthly_charges": monthly * 0.85,
        "total_charges": monthly * 0.85 * tenure
    }
}

names, probs, changes = [], [], []
for name, data in scenarios.items():
    p = predictor.predict(data)["churn_probability"]
    names.append(name)
    probs.append(p)
    changes.append(p - base_prob)

colors = ["#2ecc71" if p < base_prob else "#e74c3c" for p in probs]

fig = go.Figure()
fig.add_hline(
    y=base_prob, line_dash="dash", line_color="white", line_width=2,
    annotation_text=f"Current Risk: {base_prob:.1%}",
    annotation_position="top right"
)
fig.add_bar(
    x=names, y=probs, marker_color=colors,
    text=[f"{p:.1%}" for p in probs], textposition="outside"
)
fig.update_layout(
    title="Churn Risk Under Different Retention Scenarios",
    yaxis_title="Churn Probability",
    yaxis_tickformat=".0%",
    yaxis_range=[0, min(1.0, max(probs) * 1.3)],
    height=450,
    xaxis_tickangle=-20
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Scenario Details")
for name, prob, change in zip(names, probs, changes):
    emoji = "🟢" if change < 0 else "🔴"
    saved = abs(change)
    col1, col2, col3 = st.columns([3, 1, 1])
    col1.write(f"{emoji} **{name}**")
    col2.write(f"Risk: **{prob:.1%}**")
    col3.write(f"Change: **{change:+.1%}**")