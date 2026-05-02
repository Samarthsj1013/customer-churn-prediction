import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = [
    "customer_id", "age", "gender", "tenure_months", "contract_type",
    "monthly_charges", "total_charges", "churn"
]

def validate_dataframe(df: pd.DataFrame) -> bool:
    logger.info("Validating dataframe...")
    passed = True

    # Check required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        passed = False

    # Check no empty dataframe
    if df.empty:
        logger.error("Dataframe is empty!")
        passed = False

    # Check for nulls in critical columns
    critical = ["customer_id", "churn", "monthly_charges"]
    for col in critical:
        if col in df.columns and df[col].isnull().any():
            logger.warning(f"Null values found in critical column: {col}")

    # Check churn column is binary
    if "churn" in df.columns:
        unique_vals = set(df["churn"].unique())
        if not unique_vals.issubset({0, 1}):
            logger.error(f"Churn column has unexpected values: {unique_vals}")
            passed = False

    if passed:
        logger.info("Validation passed!")
    return passed