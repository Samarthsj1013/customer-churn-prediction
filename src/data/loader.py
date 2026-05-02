import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.helpers import timer
from src.config import settings

logger = get_logger(__name__)

@timer
def load_raw_data() -> pd.DataFrame:
    path = Path(settings.RAW_DATA_PATH) / "churn_raw.csv"
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}. Run generator.py first.")
    df = pd.read_csv(path)
    logger.info(f"Loaded raw data: {df.shape}")
    return df

@timer
def load_processed_data() -> pd.DataFrame:
    path = Path(settings.PROCESSED_DATA_PATH) / "churn_processed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found at {path}. Run preprocessing first.")
    df = pd.read_csv(path)
    logger.info(f"Loaded processed data: {df.shape}")
    return df

def get_data_info(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "churn_rate": float(df["churn"].mean()) if "churn" in df.columns else None,
        "dtypes": df.dtypes.astype(str).to_dict()
    }