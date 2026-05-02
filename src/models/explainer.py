import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ChurnExplainer:
    def __init__(self, model, feature_names: list):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
        logger.info("SHAP explainer initialized")

    def explain_prediction(self, X: pd.DataFrame) -> dict:
        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values

        feature_impacts = {}
        for i, feature in enumerate(self.feature_names):
            feature_impacts[feature] = float(shap_vals[0][i])

        sorted_impacts = dict(
            sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        top_risk_factors = {
            k: v for k, v in list(sorted_impacts.items())[:5] if v > 0
        }
        top_protective = {
            k: v for k, v in list(sorted_impacts.items())[:5] if v < 0
        }

        return {
            "feature_impacts": sorted_impacts,
            "top_risk_factors": top_risk_factors,
            "top_protective_factors": top_protective,
            "base_value": float(self.explainer.expected_value 
                if not isinstance(self.explainer.expected_value, list) 
                else self.explainer.expected_value[1])
        }