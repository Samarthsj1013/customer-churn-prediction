import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.models.predictor import ChurnPredictor
from src.data.generator import generate_churn_data

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

@st.cache_resource
def load_predictor():
    return ChurnPredictor()

@st.cache_data
def get_analytics_data():
    predictor = load_predictor()
    df = generate_churn_data(n_samples=1000, random_state=99)
    results = []
    for _, row in df.iterrows():
        try:
            r = predictor.predict(row.to_dict())
            results.append({
                "state": row["state"],
                "contract_type": row["contract_type"],
                "tenure_months": row["tenure_months"],
                "monthly_charges": row["monthly_charges"],
                "internet_service": row["internet_service"],
                "age": row["age"],
                "num_complaints": row["num_complaints"],
                "churn_probability": r["churn_probability"],
                "risk_level": r["risk_level"],
                "actual_churn": row["churn"]
            })
        except:
            pass
    return pd.DataFrame(results)

st.title("📊 Churn Analytics Dashboard")
st.markdown("Business intelligence overview across your customer base.")

with st.spinner("Loading analytics data..."):
    df = get_analytics_data()

# KPI Row
st.subheader("Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
avg_prob = df["churn_probability"].mean()
high_risk = (df["risk_level"] == "High").sum()
med_risk = (df["risk_level"] == "Medium").sum()
revenue_at_risk = df[df["risk_level"] == "High"]["monthly_charges"].sum()
total_revenue = df["monthly_charges"].sum()

col1.metric("Avg Churn Risk", f"{avg_prob:.1%}")
col2.metric("High Risk Customers", high_risk)
col3.metric("Medium Risk", med_risk)
col4.metric("Monthly Revenue at Risk", f"₹{revenue_at_risk:,.0f}")
col5.metric("Total Monthly Revenue", f"₹{total_revenue:,.0f}")

st.divider()

# Row 1
col1, col2 = st.columns(2)
with col1:
    risk_counts = df["risk_level"].value_counts()
    fig = px.pie(
        values=risk_counts.values, names=risk_counts.index,
        title="Risk Level Distribution",
        color=risk_counts.index,
        color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    state_churn = df.groupby("state")["churn_probability"].mean().sort_values(ascending=False)
    fig = px.bar(
        x=state_churn.index, y=state_churn.values,
        title="Average Churn Risk by State",
        labels={"x": "State", "y": "Avg Churn Probability"},
        color=state_churn.values, color_continuous_scale="Reds"
    )
    fig.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

# Row 2
col1, col2 = st.columns(2)
with col1:
    contract_churn = df.groupby("contract_type")["churn_probability"].mean().sort_values(ascending=False)
    fig = px.bar(
        x=contract_churn.index, y=contract_churn.values,
        title="Churn Risk by Contract Type",
        color=contract_churn.values, color_continuous_scale="RdYlGn_r"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(
        df, x="tenure_months", y="churn_probability",
        color="risk_level",
        color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"},
        title="Tenure vs Churn Probability",
        labels={"tenure_months": "Tenure (months)", "churn_probability": "Churn Probability"},
        opacity=0.6
    )
    st.plotly_chart(fig, use_container_width=True)

# Row 3
col1, col2 = st.columns(2)
with col1:
    internet_churn = df.groupby("internet_service")["churn_probability"].mean()
    fig = px.bar(
        x=internet_churn.index, y=internet_churn.values,
        title="Churn Risk by Internet Service",
        color=internet_churn.values, color_continuous_scale="Reds"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(
        df, x="churn_probability", nbins=30,
        title="Churn Probability Distribution",
        color_discrete_sequence=["#3498db"]
    )
    st.plotly_chart(fig, use_container_width=True)