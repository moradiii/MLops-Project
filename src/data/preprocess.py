from sklearn.model_selection import train_test_split


def preprocess(X, y, test_size=0.2, random_state=42):
    """
    Only handles train/test split.
    Feature engineering is handled elsewhere.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
