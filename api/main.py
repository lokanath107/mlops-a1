from fastapi import FastAPI
import joblib
import pandas as pd

# monitoring
import logging
import time
import mlflow

from api.schemas import HeartRequest, HeartResponse

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="Heart Disease Prediction API")

# Load model
model = joblib.load("model/model.pkl")

# Prediction endpoint
@app.post("/predict", response_model=HeartResponse)
def predict(data: HeartRequest):

    # Convert Pydantic to dict and then to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # measuring latency
    start_time = time.time()

    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    latency = time.time() - start_time

    # Cloud Run / console logs
    logger.info({
        "prediction": int(prediction),
        "probability": float(proba),
        "latency_ms": round(latency * 1000, 2)
    })

    # MLflow monitoring
    mlflow.set_experiment("heart-api-monitoring")
    with mlflow.start_run():
        mlflow.log_metric("prediction", int(prediction))
        mlflow.log_metric("probability", float(proba))
        mlflow.log_metric("latency_ms", latency * 1000)

    return {
        "prediction": int(prediction),
        "probability": float(proba)
    }

# Health check
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

# Root
@app.get("/")
def root():
    return {
        "message": "Welcome to the Heart Disease Prediction API. Use /predict"
    }
