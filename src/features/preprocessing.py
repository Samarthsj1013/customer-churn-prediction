import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.utils.logger import get_logger
from src.utils.helpers import timer

logger = get_logger(__name__)

@timer
def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = ["customer_id"]
    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing)
    logger.info(f"Dropped columns: {existing}")
    return df

@timer
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # Numeric columns — fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Filled nulls in {col} with median: {median_val}")

    # Categorical columns — fill with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info(f"Filled nulls in {col} with mode: {mode_val}")

    return df

@timer
def encode_categorical_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    encoders = {}
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.info(f"Encoded column: {col} -> {list(le.classes_)}")

    return df, encoders

@timer
def scale_numeric_features(
    df: pd.DataFrame,
    target_col: str = "churn",
    scaler: StandardScaler = None
) -> tuple[pd.DataFrame, StandardScaler]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]

    if scaler is None:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        logger.info(f"Fitted and scaled {len(numeric_cols)} numeric columns")
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
        logger.info(f"Scaled {len(numeric_cols)} numeric columns using existing scaler")

    return df, scaler