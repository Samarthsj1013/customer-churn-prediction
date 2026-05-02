from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = "sqlite:///./churn.db"

    # Model
    MODEL_PATH: str = str(BASE_DIR / "models" / "churn_model.pkl")
    SCALER_PATH: str = str(BASE_DIR / "models" / "scaler.pkl")

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = str(BASE_DIR / "logs" / "app.log")

    # Data
    RAW_DATA_PATH: str = str(BASE_DIR / "data" / "raw")
    PROCESSED_DATA_PATH: str = str(BASE_DIR / "data" / "processed")
    GENERATED_DATA_PATH: str = str(BASE_DIR / "data" / "generated")

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()