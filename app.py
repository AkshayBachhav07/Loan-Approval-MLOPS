import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Loan Approval Prediction API",
    description="End-to-End MLOps API for automated loan eligibility decisioning.",
    version="1.0.0"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.pkl")
_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=500, detail="Model artifact not found. Execute train.py first.")
        _model = joblib.load(MODEL_PATH)
    return _model

class LoanApplication(BaseModel):
    applicant_income: float = Field(..., ge=5000, le=1000000, example=65000.0)
    coapplicant_income: float = Field(..., ge=0, le=1000000, example=25000.0)
    loan_amount: float = Field(..., ge=10000, le=5000000, example=250000.0)
    credit_history: int = Field(..., ge=0, le=1, description="1: Clear History, 0: Defaulted", example=1)
    employment_status: int = Field(..., ge=0, le=1, description="1: Employed, 0: Unemployed", example=1)
    existing_loans_count: int = Field(..., ge=0, le=15, example=1)

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Loan Approval Prediction API",
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    model_loaded = os.path.exists(MODEL_PATH)
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_artifact_loaded": model_loaded
    }

@app.post("/predict")
def predict_loan_status(payload: LoanApplication):
    model = get_model()
    input_df = pd.DataFrame([payload.model_dump()])

    pred = int(model.predict(input_df)[0])
    prob = float(model.predict_proba(input_df)[0][1])

    return {
        "loan_approved": pred,
        "decision": "Approved" if pred == 1 else "Rejected",
        "approval_probability": round(prob, 4),
        "status": "Success"
    }
