from src.data.feature_engineering import build_preprocessor
from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.models.train import train_model

X, y = load_data()
X_train, X_test, y_train, y_test = preprocess(X, y)

preprocessor = build_preprocessor(X_train)
model, metrics = train_model(X_train, X_test, y_train, y_test, preprocessor)
