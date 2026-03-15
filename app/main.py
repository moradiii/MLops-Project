from fastapi import FastAPI, Request, HTTPException
import pandas as pd
import logging
import time
import os

from app.schemas import CustomerData, PredictionResponse
from app.model_loader import load_model

app = FastAPI(title="Customer Churn Prediction API")

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

prediction_counts = {"0": 0, "1": 0}
failed_requests = 0

model = load_model()
model_loaded = model is not None


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    logging.info(
        f"Path={request.url.path} Method={request.method} "
        f"Status={response.status_code} Duration={duration:.4f}s"
    )
    return response


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model_loaded
    }


@app.get("/metrics")
def metrics():
    return {
        "prediction_counts": prediction_counts,
        "failed_requests": failed_requests
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    global failed_requests

    try:
        input_df = pd.DataFrame([data.dict()])

        prediction_proba = model.predict_proba(input_df)[0][1]
        prediction = int(prediction_proba > 0.5)

        prediction_counts[str(prediction)] += 1

        logging.info(
            f"Prediction successful | probability={float(prediction_proba):.4f} "
            f"prediction={prediction}"
        )

        return PredictionResponse(
            churn_probability=float(prediction_proba),
            prediction=prediction
        )

    except Exception as e:
        failed_requests += 1
        logging.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")