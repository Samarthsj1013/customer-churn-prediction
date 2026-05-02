import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.helpers import timer
from src.config import settings

logger = get_logger(__name__)

@timer
def generate_churn_data(n_samples: int = 120000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    logger.info(f"Generating {n_samples} customer records...")

    # Customer demographics
    df = pd.DataFrame()
    df["customer_id"] = [f"CUST_{i:06d}" for i in range(n_samples)]
    df["age"] = np.random.randint(18, 80, n_samples)
    df["gender"] = np.random.choice(["Male", "Female"], n_samples)
    df["state"] = np.random.choice([
    "Maharashtra", "Karnataka", "Tamil Nadu", "Delhi", "Uttar Pradesh",
    "Gujarat", "Rajasthan", "West Bengal", "Telangana", "Kerala"
], n_samples)

    # Account info
    df["tenure_months"] = np.random.randint(1, 72, n_samples)
    df["contract_type"] = np.random.choice(
        ["Month-to-Month", "One Year", "Two Year"],
        n_samples, p=[0.55, 0.25, 0.20]
    )
    df["payment_method"] = np.random.choice(
        ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"],
        n_samples
    )

    # Services
    df["internet_service"] = np.random.choice(
        ["DSL", "Fiber Optic", "No"], n_samples, p=[0.35, 0.45, 0.20]
    )
    df["phone_service"] = np.random.choice(["Yes", "No"], n_samples, p=[0.90, 0.10])
    df["streaming_tv"] = np.random.choice(["Yes", "No"], n_samples, p=[0.45, 0.55])
    df["online_security"] = np.random.choice(["Yes", "No"], n_samples, p=[0.35, 0.65])
    df["tech_support"] = np.random.choice(["Yes", "No"], n_samples, p=[0.30, 0.70])

    # Financials
    df["monthly_charges"] = np.round(np.random.uniform(20, 120, n_samples), 2)
    df["total_charges"] = np.round(
        df["monthly_charges"] * df["tenure_months"] * np.random.uniform(0.85, 1.0, n_samples), 2
    )
    df["num_complaints"] = np.random.poisson(0.5, n_samples)
    df["num_support_calls"] = np.random.poisson(1.2, n_samples)

    # Usage
    df["avg_daily_usage_gb"] = np.round(np.random.exponential(3.5, n_samples), 2)
    df["late_payments"] = np.random.poisson(0.3, n_samples)
    df["promotion_offered"] = np.random.choice(["Yes", "No"], n_samples, p=[0.25, 0.75])

    # Churn logic (realistic rules)
    churn_score = (
        (df["contract_type"] == "Month-to-Month").astype(int) * 2.5 +
        (df["internet_service"] == "Fiber Optic").astype(int) * 1.2 +
        (df["tenure_months"] < 12).astype(int) * 2.0 +
        (df["num_complaints"] > 1).astype(int) * 2.0 +
        (df["monthly_charges"] > 80).astype(int) * 1.5 +
        (df["late_payments"] > 0).astype(int) * 1.0 +
        (df["online_security"] == "No").astype(int) * 0.8 +
        (df["num_support_calls"] > 2).astype(int) * 1.2 +
        np.random.normal(0, 1, n_samples)
    )
    df["churn"] = (churn_score > 4.5).astype(int)

    logger.info(f"Churn rate: {df['churn'].mean():.2%}")
    logger.info(f"Dataset shape: {df.shape}")
    return df

@timer
def save_raw_data(df: pd.DataFrame) -> str:
    output_path = Path(settings.RAW_DATA_PATH) / "churn_raw.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Raw data saved to {output_path}")
    return str(output_path)

if __name__ == "__main__":
    df = generate_churn_data()
    save_raw_data(df)
    print(df.head())
    print(df["churn"].value_counts())