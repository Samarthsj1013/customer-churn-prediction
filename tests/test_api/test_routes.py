import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

SAMPLE_CUSTOMER = {
    "age": 35,
    "gender": "Male",
    "state": "CA",
    "tenure_months": 6,
    "contract_type": "Month-to-Month",
    "payment_method": "Electronic Check",
    "internet_service": "Fiber Optic",
    "phone_service": "Yes",
    "streaming_tv": "No",
    "online_security": "No",
    "tech_support": "No",
    "monthly_charges": 95.0,
    "total_charges": 570.0,
    "num_complaints": 2,
    "num_support_calls": 4,
    "avg_daily_usage_gb": 5.5,
    "late_payments": 1,
    "promotion_offered": "No"
}

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "model_loaded" in data

def test_predict_endpoint():
    response = client.post("/api/v1/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "risk_level" in data
    assert "recommendation" in data

def test_predict_invalid_input():
    response = client.post("/api/v1/predict", json={"age": "invalid"})
    assert response.status_code == 422

def test_batch_predict_endpoint():
    batch = {"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]}
    response = client.post("/api/v1/predict/batch", json=batch)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["predictions"]) == 2

def test_model_info_endpoint():
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "auc_score" in data