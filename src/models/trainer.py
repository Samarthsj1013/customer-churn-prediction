import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import optuna
import warnings
warnings.filterwarnings("ignore")

from src.data.loader import load_processed_data
from src.utils.logger import get_logger
from src.utils.helpers import timer
from src.config import settings

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def get_X_y(df: pd.DataFrame) -> tuple:
    X = df.drop(columns=["churn"])
    y = df["churn"]
    return X, y

@timer
def apply_smote(X_train, y_train, random_state: int = 42):
    logger.info(f"Before SMOTE - Class distribution: {dict(y_train.value_counts())}")
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE - Class distribution: {dict(pd.Series(y_resampled).value_counts())}")
    return X_resampled, y_resampled

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
        "random_state": 42,
        "eval_metric": "auc",
        "verbosity": 0
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)

@timer
def tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials: int = 30):
    logger.info(f"Starting hyperparameter tuning with {n_trials} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=False
    )
    logger.info(f"Best AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    return study.best_params

@timer
def train_model(n_trials: int = 30) -> dict:
    logger.info("Loading processed data...")
    df = load_processed_data()
    X, y = get_X_y(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Handle class imbalance
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    # Tune hyperparameters
    best_params = tune_hyperparameters(X_train_bal, y_train_bal, X_val, y_val, n_trials)

    # Train final model
    logger.info("Training final model with best parameters...")
    best_params.update({"random_state": 42, "eval_metric": "auc", "verbosity": 0})
    final_model = XGBClassifier(**best_params)
    final_model.fit(
        X_train_bal, y_train_bal,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Evaluate on test set
    test_preds = final_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_preds)
    logger.info(f"Final Test AUC: {test_auc:.4f}")

    # Save model
    model_path = Path(settings.MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save feature names
    feature_names_path = model_path.parent / "feature_names.pkl"
    joblib.dump(list(X.columns), feature_names_path)

    return {
        "model": final_model,
        "test_auc": test_auc,
        "best_params": best_params,
        "feature_names": list(X.columns),
        "X_test": X_test,
        "y_test": y_test
    }

if __name__ == "__main__":
    results = train_model(n_trials=30)
    print(f"\nTest AUC: {results['test_auc']:.4f}")