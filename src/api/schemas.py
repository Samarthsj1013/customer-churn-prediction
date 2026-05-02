from pydantic import BaseModel, Field
from typing import Optional

class CustomerInput(BaseModel):
    age: int = Field(..., ge=18, le=100, example=35)
    gender: str = Field(..., example="Male")
    state: str = Field(..., example="CA")
    tenure_months: int = Field(..., ge=0, example=24)
    contract_type: str = Field(..., example="Month-to-Month")
    payment_method: str = Field(..., example="Credit Card")
    internet_service: str = Field(..., example="Fiber Optic")
    phone_service: str = Field(..., example="Yes")
    streaming_tv: str = Field(..., example="No")
    online_security: str = Field(..., example="No")
    tech_support: str = Field(..., example="No")
    monthly_charges: float = Field(..., ge=0, example=85.5)
    total_charges: float = Field(..., ge=0, example=2052.0)
    num_complaints: int = Field(..., ge=0, example=2)
    num_support_calls: int = Field(..., ge=0, example=3)
    avg_daily_usage_gb: float = Field(..., ge=0, example=4.2)
    late_payments: int = Field(..., ge=0, example=1)
    promotion_offered: str = Field(..., example="No")

    class Config:
        json_schema_extra = {
            "example": {
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
        }

class PredictionResponse(BaseModel):
    customer_id: Optional[str] = None
    churn_probability: float
    churn_prediction: int
    risk_level: str
    recommendation: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

class BatchInput(BaseModel):
    customers: list[CustomerInput]

class BatchResponse(BaseModel):
    total: int
    predictions: list[PredictionResponse]
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int