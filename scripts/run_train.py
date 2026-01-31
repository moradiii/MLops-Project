from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.models.train import train_model

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess(X, y)
    train_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
