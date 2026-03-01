import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --------------------------------------------------
# 1. Load your dataset
# --------------------------------------------------
DATA_PATH = Path("data/liste_users.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Could not find {DATA_PATH}. Please ensure the file exists.")

df = pd.read_csv(DATA_PATH)

# Safety check: Ensure 'churn' exists
if 'churn' not in df.columns:
    raise ValueError("Target column 'churn' not found in the dataset!")

# --------------------------------------------------
# 2. Data Cleaning
# --------------------------------------------------
# Drop columns that are usually useless for ML (IDs, Names, Emails)
cols_to_drop = [c for c in df.columns if 'id' in c.lower() or 'name' in c.lower() or 'email' in c.lower()]
X = df.drop(columns=['churn'] + cols_to_drop, errors='ignore')
y = df["churn"]

# Identify feature types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "category"]).columns

# --------------------------------------------------
# 3. Robust Preprocessing
# --------------------------------------------------
# We add Imputers here so the model doesn't crash on missing data (NaN)
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()) # Scaling helps many models, though RF is robust to it
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# --------------------------------------------------
# 4. Pipeline & Model
# --------------------------------------------------
pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
    ]
)

# --------------------------------------------------
# 5. Train & Evaluate
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training model...")
pipeline.fit(X_train, y_train)

# Check performance before saving
y_pred = pipeline.predict(X_test)
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# 6. Save model.pkl
# --------------------------------------------------
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(pipeline, MODEL_DIR / "model.pkl")

print(f"\n✅ Model saved to {MODEL_DIR}/model.pkl")