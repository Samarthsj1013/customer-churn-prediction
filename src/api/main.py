from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready API for predicting customer churn using XGBoost (AUC: 0.9364)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(router, prefix="/api/v1")

@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)