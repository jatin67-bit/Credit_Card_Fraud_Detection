import os
import time
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

app = FastAPI(
    title="Credit Card Fraud Detection REST API",
    description="Production-ready REST API for low-latency credit card fraud classification powered by Random Forest and SMOTE.",
    version="1.0.0"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "fraud_model.pkl")
model = None

def get_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            # Fallback dummy model for testing/CI when model file isn't pre-trained
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model file 'fraud_model.pkl' not found. Please run 'python fraud_detection.py' first."
            )
    return model

# ==========================================
# PYDANTIC DATA SCHEMAS & VALIDATION
# ==========================================
class TransactionPayload(BaseModel):
    features: List[float] = Field(
        ...,
        description="Array of 30 numerical features [Time, V1..V28, Amount] extracted from credit card transactions."
    )

    @field_validator('features')
    @classmethod
    def validate_feature_length(cls, v):
        if len(v) != 30:
            raise ValueError(f"Expected exactly 30 features, but received {len(v)} features.")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": [
                    0.0, -1.3598, -0.0727, 2.5363, 1.3781, -0.3383, 0.4623, 0.2395,
                    0.0986, 0.3637, 0.0907, -0.5516, -0.6178, -0.9913, -0.3111, 1.4681,
                    -0.4704, 0.2079, 0.0257, 0.4039, 0.2514, -0.0183, 0.2778, -0.1104,
                    0.0669, 0.1285, -0.1891, 0.1335, -0.0210, 149.62
                ]
            }
        }
    }

class PredictionResponse(BaseModel):
    is_fraud: int = Field(..., description="0 for Legitimate, 1 for Fraudulent")
    fraud_probability: float = Field(..., description="Estimated probability of fraud [0.0 to 1.0]")
    risk_level: str = Field(..., description="Risk tier: LOW, MEDIUM, or CRITICAL")
    inference_latency_ms: float = Field(..., description="Latency of the model prediction in milliseconds")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

# ==========================================
# API ENDPOINTS
# ==========================================
@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Credit Card Fraud Detection API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "predict_endpoint": "/predict"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    is_loaded = os.path.exists(MODEL_PATH)
    return {
        "status": "healthy" if is_loaded else "model_pending",
        "model_loaded": is_loaded,
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(payload: TransactionPayload):
    clf = get_model()
    start_time = time.time()

    input_data = np.array(payload.features).reshape(1, -1)
    prediction = int(clf.predict(input_data)[0])
    prob = float(clf.predict_proba(input_data)[0][1])

    latency_ms = (time.time() - start_time) * 1000

    if prob >= 0.70:
        risk = "CRITICAL"
    elif prob >= 0.35:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "is_fraud": prediction,
        "fraud_probability": round(prob, 4),
        "risk_level": risk,
        "inference_latency_ms": round(latency_ms, 2)
    }

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(payloads: List[TransactionPayload]):
    clf = get_model()
    start_time = time.time()

    input_data = np.array([p.features for p in payloads])
    predictions = clf.predict(input_data).tolist()
    probabilities = clf.predict_proba(input_data)[:, 1].tolist()

    latency_ms = (time.time() - start_time) * 1000

    results = []
    for pred, prob in zip(predictions, probabilities):
        results.append({
            "is_fraud": int(pred),
            "fraud_probability": round(float(prob), 4),
            "risk_level": "CRITICAL" if prob >= 0.70 else ("MEDIUM" if prob >= 0.35 else "LOW")
        })

    return {
        "batch_size": len(payloads),
        "total_latency_ms": round(latency_ms, 2),
        "average_latency_ms": round(latency_ms / len(payloads), 2) if payloads else 0,
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
