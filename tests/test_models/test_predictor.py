import pytest
from src.models.predictor import ChurnPredictor

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

@pytest.fixture
def predictor():
    return ChurnPredictor()

def test_predictor_loads(predictor):
    assert predictor.model is not None
    assert predictor.scaler is not None

def test_prediction_returns_dict(predictor):
    result = predictor.predict(SAMPLE_CUSTOMER)
    assert isinstance(result, dict)

def test_prediction_has_required_keys(predictor):
    result = predictor.predict(SAMPLE_CUSTOMER)
    assert "churn_probability" in result
    assert "churn_prediction" in result
    assert "risk_level" in result

def test_probability_between_0_and_1(predictor):
    result = predictor.predict(SAMPLE_CUSTOMER)
    assert 0.0 <= result["churn_probability"] <= 1.0

def test_prediction_is_binary(predictor):
    result = predictor.predict(SAMPLE_CUSTOMER)
    assert result["churn_prediction"] in [0, 1]

def test_risk_level_valid(predictor):
    result = predictor.predict(SAMPLE_CUSTOMER)
    assert result["risk_level"] in ["Low", "Medium", "High"]