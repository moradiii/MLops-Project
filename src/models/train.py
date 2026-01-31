from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc:.4f}")
    return model
