import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.dashboard.components import (
    render_header, render_metric_cards,
    render_recommendation, render_sidebar_inputs
)
from src.dashboard.charts import (
    render_gauge, render_feature_importance, render_churn_distribution
)
from src.models.predictor import ChurnPredictor
from src.models.explainer import ChurnExplainer
from src.features.feature_engineering import create_features

@st.cache_resource
def load_predictor():
    return ChurnPredictor()

@st.cache_resource
def load_explainer(_predictor):
    return ChurnExplainer(_predictor.model, _predictor.feature_names)

def main():
    render_header()
    customer_data = render_sidebar_inputs()

    try:
        predictor = load_predictor()
        explainer = load_explainer(predictor)
    except Exception as e:
        st.error(f"Model not loaded: {e}")
        return

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_btn = st.button("🔮 Predict Churn", type="primary", use_container_width=True)

    if predict_btn:
        with st.spinner("Analyzing customer..."):
            result = predictor.predict(customer_data)

            # Prepare data for SHAP
            df = pd.DataFrame([customer_data])
            df = create_features(df)
            cat_cols = ["gender", "state", "contract_type", "payment_method",
                        "internet_service", "phone_service", "streaming_tv",
                        "online_security", "tech_support", "promotion_offered"]
            for col in cat_cols:
                if col in predictor.encoders and col in df.columns:
                    try:
                        df[col] = predictor.encoders[col].transform(df[col].astype(str))
                    except:
                        df[col] = 0
            df = df.reindex(columns=predictor.feature_names, fill_value=0)
            df[df.columns] = predictor.scaler.transform(df)
            shap_result = explainer.explain_prediction(df)

        churn_prob = result["churn_probability"]
        risk_level = result["risk_level"]
        prediction = result["churn_prediction"]

        render_metric_cards(churn_prob, risk_level, prediction)
        st.divider()
        render_recommendation(risk_level, churn_prob)
        st.divider()

        # SHAP explanation
        st.subheader("🧠 Why this prediction?")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔴 Top Risk Factors** (pushing toward churn)")
            if shap_result["top_risk_factors"]:
                for feat, val in list(shap_result["top_risk_factors"].items())[:5]:
                    st.error(f"**{feat}**: +{val:.3f} churn risk")
            else:
                st.info("No major risk factors found")
        with col2:
            st.markdown("**🟢 Protective Factors** (keeping customer)")
            if shap_result["top_protective_factors"]:
                for feat, val in list(shap_result["top_protective_factors"].items())[:5]:
                    st.success(f"**{feat}**: {val:.3f} churn risk")
            else:
                st.info("No protective factors found")

        st.divider()

        col1, col2, col3 = st.columns(3)
        with col1:
            render_gauge(churn_prob)
        with col2:
            render_churn_distribution(churn_prob)
        with col3:
            importances = predictor.model.feature_importances_.tolist()
            render_feature_importance(predictor.feature_names, importances)

        with st.expander("📊 Full SHAP Impact Breakdown"):
            shap_df = pd.DataFrame([
                {
                    "Feature": k,
                    "SHAP Impact": round(v, 4),
                    "Direction": "Increases Churn ⬆️" if v > 0 else "Reduces Churn ⬇️"
                }
                for k, v in shap_result["feature_impacts"].items()
            ])
            st.dataframe(shap_df, use_container_width=True)

        with st.expander("View Raw Input Data"):
            st.json(customer_data)
        with st.expander("View Raw Prediction"):
            st.json(result)
    else:
        st.info("Configure customer details in the sidebar and click **Predict Churn** to get started.")

if __name__ == "__main__":
    main()