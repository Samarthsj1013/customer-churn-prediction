import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
from src.config import settings

logger = get_logger(__name__)

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoders = None
        self.feature_names = None
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            self.model = joblib.load(settings.MODEL_PATH)
            self.scaler = joblib.load(settings.SCALER_PATH)
            model_dir = Path(settings.MODEL_PATH).parent
            self.encoders = joblib.load(model_dir / "encoders.pkl")
            self.feature_names = joblib.load(model_dir / "feature_names.pkl")
            logger.info("All model artifacts loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"Model artifact not found: {e}. Train the model first.")
            raise

    def predict(self, customer_data: dict) -> dict:
        df = pd.DataFrame([customer_data])

        # Feature engineering
        df["charge_per_tenure"] = df["total_charges"] / df["tenure_months"].replace(0, 1)
        df["is_new_customer"] = (df["tenure_months"] < 12).astype(int)
        df["is_long_term"] = (df["tenure_months"] > 36).astype(int)
        df["high_monthly_charges"] = (df["monthly_charges"] > 80).astype(int)
        df["complaint_rate"] = df["num_complaints"] / df["tenure_months"].replace(0, 1)
        df["support_intensity"] = df["num_support_calls"] + df["num_complaints"] * 2
        df["risk_score"] = (
            df["num_complaints"] * 3 +
            df["late_payments"] * 2 +
            df["num_support_calls"] +
            df["high_monthly_charges"]
        )

        # Encode categoricals
        cat_cols = ["gender", "state", "contract_type", "payment_method",
                    "internet_service", "phone_service", "streaming_tv",
                    "online_security", "tech_support", "promotion_offered"]
        for col in cat_cols:
            if col in self.encoders and col in df.columns:
                try:
                    df[col] = self.encoders[col].transform(df[col].astype(str))
                except ValueError:
                    df[col] = 0

        # Align columns
        df = df.reindex(columns=self.feature_names, fill_value=0)

        # Scale
        df[df.columns] = self.scaler.transform(df)

        # Predict
        churn_proba = float(self.model.predict_proba(df)[0][1])
        churn_pred = int(churn_proba >= 0.5)

        risk_level = "Low" if churn_proba < 0.3 else "Medium" if churn_proba < 0.6 else "High"

        return {
            "churn_probability": round(churn_proba, 4),
            "churn_prediction": churn_pred,
            "risk_level": risk_level
        }