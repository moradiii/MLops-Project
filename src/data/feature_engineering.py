import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(X: pd.DataFrame):
    X = X.copy()

    # Replace common missing markers
    X = X.replace({"N/A": np.nan, "NA": np.nan, "": np.nan})

    # Force numeric conversion for numeric-looking columns
    # (this will turn "N/A" -> NaN, and "42" -> 42)
    for col in ["duration_month", "total_bill"]:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    numerical_cols = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()


    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
