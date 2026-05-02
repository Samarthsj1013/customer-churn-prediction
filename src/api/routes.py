from fastapi import APIRouter, HTTPException
from src.api.schemas import (
    CustomerInput, PredictionResponse,
    HealthResponse, BatchInput, BatchResponse
)
from src.models.predictor import ChurnPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Load predictor once at startup
try:
    predictor = ChurnPredictor()
    model_loaded = True
except Exception as e:
    logger.error(f"Failed to load predictor: {e}")
    predictor = None
    model_loaded = False

RECOMMENDATIONS = {
    "High": "Immediately offer a loyalty discount or contract upgrade. Assign a retention specialist.",
    "Medium": "Send a personalized email with a special offer. Consider a free service upgrade.",
    "Low": "Customer is stable. Continue standard engagement and monitor quarterly."
}

@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version="1.0.0"
    )

@router.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict_churn(customer: CustomerInput):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    try:
        result = predictor.predict(customer.model_dump())
        result["recommendation"] = RECOMMENDATIONS[result["risk_level"]]
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/batch", response_model=BatchResponse, tags=["Predictions"])
def predict_batch(batch: BatchInput):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        predictions = []
        for i, customer in enumerate(batch.customers):
            result = predictor.predict(customer.model_dump())
            result["customer_id"] = f"BATCH_{i+1:04d}"
            result["recommendation"] = RECOMMENDATIONS[result["risk_level"]]
            predictions.append(PredictionResponse(**result))

        return BatchResponse(
            total=len(predictions),
            predictions=predictions,
            high_risk_count=sum(1 for p in predictions if p.risk_level == "High"),
            medium_risk_count=sum(1 for p in predictions if p.risk_level == "Medium"),
            low_risk_count=sum(1 for p in predictions if p.risk_level == "Low")
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/info", tags=["Model"])
def model_info():
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "model_type": "XGBoost Classifier",
        "features": predictor.feature_names,
        "feature_count": len(predictor.feature_names),
        "auc_score": 0.9364,
        "version": "1.0.0"
    }