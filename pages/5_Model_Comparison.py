import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from src.data.loader import load_processed_data

st.set_page_config(page_title="Model Comparison", page_icon="🏆", layout="wide")

@st.cache_data
def run_comparison():
    df = load_processed_data()
    X = df.drop(columns=["churn"])
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []

    # Our XGBoost
    xgb = joblib.load("models/churn_model.pkl")
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_pred = xgb.predict(X_test)
    report = classification_report(y_test, xgb_pred, output_dict=True)
    results.append({
        "Model": "XGBoost (Tuned) ⭐",
        "AUC": roc_auc_score(y_test, xgb_proba),
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1 Score": report["1"]["f1-score"],
        "Type": "Our Model"
    })

    baselines = [
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42))
    ]

    for name, model in baselines:
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        report = classification_report(y_test, pred, output_dict=True)
        results.append({
            "Model": name,
            "AUC": roc_auc_score(y_test, proba),
            "Precision": report["1"]["precision"],
            "Recall": report["1"]["recall"],
            "F1 Score": report["1"]["f1-score"],
            "Type": "Baseline"
        })

    return pd.DataFrame(results).sort_values("AUC", ascending=False).reset_index(drop=True)

st.title("🏆 Model Comparison")
st.markdown("How does our tuned XGBoost compare against other ML algorithms?")

with st.spinner("Training all models for comparison... (~30 seconds)"):
    df = run_comparison()

# AUC Bar Chart
colors = ["#f39c12" if "XGBoost" in m else "#3498db" for m in df["Model"]]
fig = go.Figure(go.Bar(
    x=df["Model"], y=df["AUC"],
    marker_color=colors,
    text=[f"{v:.4f}" for v in df["AUC"]],
    textposition="outside"
))
fig.update_layout(
    title="AUC Score Comparison (Higher is Better)",
    yaxis_title="AUC Score",
    yaxis_range=[0.5, 1.05],
    height=400
)
st.plotly_chart(fig, use_container_width=True)

# Metrics comparison
col1, col2 = st.columns(2)
with col1:
    fig2 = px.bar(
        df, x="Model", y="F1 Score",
        title="F1 Score Comparison",
        color="Type",
        color_discrete_map={"Our Model": "#f39c12", "Baseline": "#3498db"}
    )
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    fig3 = px.scatter(
        df, x="Precision", y="Recall",
        text="Model", title="Precision vs Recall",
        color="Type",
        color_discrete_map={"Our Model": "#f39c12", "Baseline": "#3498db"},
        size=[20]*len(df)
    )
    fig3.update_traces(textposition="top center")
    st.plotly_chart(fig3, use_container_width=True)

# Full metrics table
st.subheader("Full Metrics Table")
display_df = df.copy()
for col in ["AUC", "Precision", "Recall", "F1 Score"]:
    display_df[col] = display_df[col].round(4)
st.dataframe(display_df.drop(columns=["Type"]), use_container_width=True)

best = df.iloc[0]
st.success(f"🥇 Winner: **{best['Model']}** — AUC: {best['AUC']:.4f} | F1: {best['F1 Score']:.4f}")