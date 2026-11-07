import os
import pytest
from fastapi.testclient import TestClient

if not os.path.exists("models/model.pkl"):
    import train

from app import app

client = TestClient(app)

def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "online"

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"
    assert res.json()["model_artifact_loaded"] is True

def test_model_artifact_exists():
    assert os.path.isfile("models/model.pkl")

def test_predict_approved():
    payload = {
        "applicant_income": 85000.0,
        "coapplicant_income": 30000.0,
        "loan_amount": 150000.0,
        "credit_history": 1,
        "employment_status": 1,
        "existing_loans_count": 0
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["loan_approved"] in [0, 1]
    assert data["decision"] in ["Approved", "Rejected"]
    assert 0.0 <= data["approval_probability"] <= 1.0

def test_validation_failure():
    bad_payload = {
        "applicant_income": 85000.0,
        "coapplicant_income": 30000.0,
        "loan_amount": 150000.0,
        "credit_history": 5, # Must be 0 or 1
        "employment_status": 1,
        "existing_loans_count": 0
    }
    res = client.post("/predict", json=bad_payload)
    assert res.status_code == 422
