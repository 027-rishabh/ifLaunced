"""
SpaceX Falcon 9 Booster Landing Predictor - FastAPI Application
================================================================
REST API for predicting booster landing success.

Usage:
    uvicorn app:app --reload

API Endpoints:
    POST /predict - Make a prediction
    GET  /health  - Health check
    GET  /        - API documentation
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path
import os

app = FastAPI(
    title="SpaceX Booster Landing Predictor",
    description="ML-powered API for predicting Falcon 9 booster landing success",
    version="1.0.0"
)

# Load model on startup
MODEL_PATH = Path(__file__).parent / "src" / "models" / "best_model.pkl"

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
    print(f"✓ Model loaded successfully from {MODEL_PATH}")
else:
    model = None
    print("⚠️  Model not found. Please run 08_advanced_modeling.py first.")


class PredictionRequest(BaseModel):
    rocket_name: str = Field(..., description="Name of the rocket", example="Falcon 9")
    payload_mass: float = Field(..., description="Payload mass in kg", example=5000.0)
    orbit: str = Field(..., description="Target orbit", example="ISS")


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    confidence: str
    input_features: dict


@app.get("/")
async def root():
    """API documentation and usage"""
    return {
        "message": "SpaceX Falcon 9 Booster Landing Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Make a landing success prediction",
            "GET /health": "Check API health"
        },
        "example_request": {
            "rocket_name": "Falcon 9",
            "payload_mass": 5000.0,
            "orbit": "ISS"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict booster landing success.
    
    - **rocket_name**: Name of the rocket (e.g., "Falcon 9", "Falcon Heavy")
    - **payload_mass**: Payload mass in kg
    - **orbit**: Target orbit (e.g., "ISS", "GTO", "LEO")
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Create input DataFrame
        input_data = pd.DataFrame({
            'rocket_name': [request.rocket_name],
            'payload_mass': [request.payload_mass],
            'orbit': [request.orbit]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        return PredictionResponse(
            prediction="Success" if prediction == 1 else "Failure",
            probability=round(float(probability), 4),
            confidence=f"{max(probability, 1-probability)*100:.2f}%",
            input_features={
                "rocket_name": request.rocket_name,
                "payload_mass": request.payload_mass,
                "orbit": request.orbit
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
