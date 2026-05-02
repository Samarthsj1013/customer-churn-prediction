import pandas as pd
import numpy as np
from src.utils.logger import get_logger
from src.utils.helpers import timer

logger = get_logger(__name__)

@timer
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating engineered features...")

    # Charge per month ratio
    df["charge_per_tenure"] = np.where(
        df["tenure_months"] > 0,
        df["total_charges"] / df["tenure_months"],
        df["monthly_charges"]
    )

    # Is new customer (less than 12 months)
    df["is_new_customer"] = (df["tenure_months"] < 12).astype(int)

    # Is long term customer (more than 36 months)
    df["is_long_term"] = (df["tenure_months"] > 36).astype(int)

    # High monthly charges flag
    df["high_monthly_charges"] = (df["monthly_charges"] > 80).astype(int)

    # Complaint ratio per tenure
    df["complaint_rate"] = np.where(
        df["tenure_months"] > 0,
        df["num_complaints"] / df["tenure_months"],
        df["num_complaints"]
    )

    # Support intensity
    df["support_intensity"] = df["num_support_calls"] + df["num_complaints"] * 2

    # Risk score (business rule based)
    df["risk_score"] = (
        df["num_complaints"] * 3 +
        df["late_payments"] * 2 +
        df["num_support_calls"] * 1 +
        df["high_monthly_charges"] * 1
    )

    logger.info(f"Created 7 new features. New shape: {df.shape}")
    return df