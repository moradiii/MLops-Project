from sklearn.datasets import load_iris

def load_data():
    dataset = load_iris(as_frame=True)
    X = dataset.data
    y = dataset.target
    return X, y
