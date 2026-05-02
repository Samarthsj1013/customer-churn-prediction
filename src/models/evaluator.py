import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_model(model, X_test, y_test, save_plots: bool = True) -> dict:
    logger.info("Evaluating model...")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"AUC: {auc:.4f}")
    logger.info(f"Precision: {report['1']['precision']:.4f}")
    logger.info(f"Recall: {report['1']['recall']:.4f}")
    logger.info(f"F1: {report['1']['f1-score']:.4f}")

    results = {
        "auc": auc,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }

    if save_plots:
        _save_plots(y_test, y_pred, y_proba, cm)

    return results

def _save_plots(y_test, y_pred, y_proba, cm):
    plots_dir = Path("logs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    axes[1].plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="blue")
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(plots_dir / "model_evaluation.png", dpi=150)
    plt.close()
    logger.info("Evaluation plots saved to logs/plots/")