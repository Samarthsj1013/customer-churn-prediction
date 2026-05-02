import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.features.preprocessing import (
    drop_unnecessary_columns,
    handle_missing_values,
    encode_categorical_columns,
    scale_numeric_features
)
from src.features.feature_engineering import create_features
from src.data.loader import load_raw_data
from src.data.validator import validate_dataframe
from src.utils.logger import get_logger
from src.utils.helpers import timer
from src.config import settings

logger = get_logger(__name__)

@timer
def run_pipeline(save: bool = True) -> tuple[pd.DataFrame, StandardScaler, dict]:
    logger.info("Starting feature pipeline...")

    # Step 1: Load
    df = load_raw_data()

    # Step 2: Validate
    validate_dataframe(df)

    # Step 3: Feature engineering BEFORE encoding
    df = create_features(df)

    # Step 4: Drop unnecessary
    df = drop_unnecessary_columns(df)

    # Step 5: Handle missing values
    df = handle_missing_values(df)

    # Step 6: Encode categoricals
    df, encoders = encode_categorical_columns(df)

    # Step 7: Scale numeric features
    df, scaler = scale_numeric_features(df)

    if save:
        # Save processed data
        out_path = Path(settings.PROCESSED_DATA_PATH) / "churn_processed.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(f"Processed data saved to {out_path}")

        # Save scaler
        scaler_path = Path(settings.SCALER_PATH)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

        # Save encoders
        encoders_path = Path(settings.MODEL_PATH).parent / "encoders.pkl"
        joblib.dump(encoders, encoders_path)
        logger.info(f"Encoders saved to {encoders_path}")

    logger.info(f"Pipeline complete! Final shape: {df.shape}")
    return df, scaler, encoders

if __name__ == "__main__":
    df, scaler, encoders = run_pipeline()
    print(df.head())
    print(f"\nFinal columns ({len(df.columns)}): {list(df.columns)}")