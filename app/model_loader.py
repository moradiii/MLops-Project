import joblib
from pathlib import Path

MODEL_PATH = Path("models/model.pkl")

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found.")
    return joblib.load(MODEL_PATH)