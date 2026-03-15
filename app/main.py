import logging
from fastapi import FastAPI
import pandas as pd
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from app.schemas import CustomerData, PredictionResponse
from app.model_loader import load_model

app = FastAPI(title="Customer Churn Prediction API")

model = load_model()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):

    logger.info("Prediction request received")

    input_df = pd.DataFrame([data.dict()])

    prediction_proba = model.predict_proba(input_df)[0][1]
    prediction = int(prediction_proba > 0.5)

    logger.info(f"Prediction probability: {prediction_proba}")

    return PredictionResponse(
        churn_probability=float(prediction_proba),
        prediction=prediction
    )