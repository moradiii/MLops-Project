from pydantic import BaseModel

class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str
    PaymentMethod: str

class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: int