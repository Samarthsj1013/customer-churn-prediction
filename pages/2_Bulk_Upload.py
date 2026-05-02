import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
from src.models.predictor import ChurnPredictor

st.set_page_config(page_title="Bulk Upload", page_icon="📁", layout="wide")

@st.cache_resource
def load_predictor():
    return ChurnPredictor()

st.title("📁 Bulk CSV Churn Prediction")
st.markdown("Upload a CSV file with customer data to get predictions for all customers at once.")

template_data = {
    "age": [35, 45, 28, 55],
    "gender": ["Male", "Female", "Male", "Female"],
    "state": ["Maharashtra", "Karnataka", "Delhi", "Tamil Nadu"],
    "tenure_months": [6, 48, 3, 60],
    "contract_type": ["Month-to-Month", "Two Year", "Month-to-Month", "One Year"],
    "payment_method": ["Electronic Check", "Credit Card", "Electronic Check", "Bank Transfer"],
    "internet_service": ["Fiber Optic", "DSL", "Fiber Optic", "DSL"],
    "phone_service": ["Yes", "Yes", "Yes", "Yes"],
    "streaming_tv": ["No", "Yes", "No", "Yes"],
    "online_security": ["No", "Yes", "No", "Yes"],
    "tech_support": ["No", "Yes", "No", "Yes"],
    "monthly_charges": [95.0, 35.0, 99.0, 45.0],
    "total_charges": [570.0, 1680.0, 297.0, 2700.0],
    "num_complaints": [2, 0, 3, 0],
    "num_support_calls": [4, 1, 5, 1],
    "avg_daily_usage_gb": [5.5, 2.0, 7.0, 1.5],
    "late_payments": [1, 0, 2, 0],
    "promotion_offered": ["No", "Yes", "No", "Yes"]
}
template_df = pd.DataFrame(template_data)

st.download_button(
    "📥 Download CSV Template",
    template_df.to_csv(index=False),
    "churn_template.csv",
    "text/csv",
    help="Download this template, fill it with your customer data, then upload below"
)

st.divider()
uploaded_file = st.file_uploader("📂 Upload Customer CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df)} customers successfully!")

    with st.expander("Preview uploaded data"):
        st.dataframe(df.head(), use_container_width=True)

    if st.button("🔮 Predict Churn for All Customers", type="primary", use_container_width=True):
        predictor = load_predictor()
        results = []
        progress_bar = st.progress(0)
        status = st.empty()

        for i, row in df.iterrows():
            try:
                result = predictor.predict(row.to_dict())
                results.append({
                    "Customer #": i + 1,
                    "Churn Probability": f"{result['churn_probability']:.1%}",
                    "Risk Level": result["risk_level"],
                    "Prediction": "🔴 Will Churn" if result["churn_prediction"] == 1 else "🟢 Will Stay",
                    "Recommendation": {
                        "High": "Immediate retention call needed",
                        "Medium": "Send discount offer",
                        "Low": "Standard monitoring"
                    }[result["risk_level"]]
                })
            except Exception as e:
                results.append({
                    "Customer #": i + 1,
                    "Error": str(e),
                    "Risk Level": "Unknown",
                    "Prediction": "Error"
                })
            progress_bar.progress((i + 1) / len(df))
            status.text(f"Processing customer {i+1} of {len(df)}...")

        status.empty()
        results_df = pd.DataFrame(results)

        # Summary metrics
        st.divider()
        st.subheader("📊 Prediction Summary")
        high = sum(1 for r in results if r.get("Risk Level") == "High")
        med = sum(1 for r in results if r.get("Risk Level") == "Medium")
        low = sum(1 for r in results if r.get("Risk Level") == "Low")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", len(results))
        col2.metric("🔴 High Risk", high, delta=f"{high/len(results):.0%} of total")
        col3.metric("🟡 Medium Risk", med)
        col4.metric("🟢 Low Risk", low)

        st.divider()
        st.subheader("📋 All Predictions")
        st.dataframe(results_df, use_container_width=True)

        st.download_button(
            "📥 Download All Predictions as CSV",
            results_df.to_csv(index=False),
            "churn_predictions.csv",
            "text/csv"
        )
else:
    st.info("👆 Download the template above, fill in your customer data, then upload it here.")