import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st

def render_gauge(churn_prob: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=churn_prob * 100,
        title={"text": "Churn Risk Score", "font": {"size": 20}},
        delta={"reference": 50},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 30], "color": "#2ecc71"},
                {"range": [30, 60], "color": "#f39c12"},
                {"range": [60, 100], "color": "#e74c3c"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": churn_prob * 100
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def render_feature_importance(feature_names: list, importances: list):
    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True).tail(15)

    fig = px.bar(
        df, x="Importance", y="Feature",
        orientation="h",
        title="Top 15 Feature Importances",
        color="Importance",
        color_continuous_scale="Blues"
    )
    fig.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def render_churn_distribution(churn_prob: float):
    labels = ["Stay", "Churn"]
    values = [1 - churn_prob, churn_prob]
    colors = ["#2ecc71", "#e74c3c"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.5,
        marker_colors=colors,
        textinfo="label+percent"
    ))
    fig.update_layout(title="Churn Probability Breakdown", height=300)
    st.plotly_chart(fig, use_container_width=True)


def render_shap_chart(feature_impacts: dict):
    items = list(feature_impacts.items())[:10]
    features = [i[0] for i in items]
    values = [i[1] for i in items]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=features,
        orientation="h",
        marker_color=colors
    ))
    fig.update_layout(
        title="SHAP Feature Impact",
        xaxis_title="Impact on Churn Probability",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)