from pathlib import Path
import pandas as pd
import yaml

DEFAULT_PATH = Path("data/raw/liste_users.yaml")
TARGET_COL = "has_retention"

def _extract_rows(content):
    # Case 1: Already a list of records
    if isinstance(content, list):
        return content

    # Case 2: Dict with a list inside (common)
    if isinstance(content, dict):
        for key in ["users", "data", "rows", "customers", "items", "records"]:
            if key in content and isinstance(content[key], list):
                return content[key]
        # fallback: first list value in the dict
        for v in content.values():
            if isinstance(v, list):
                return v

    raise ValueError(f"Unexpected YAML top-level type: {type(content)}")

def load_data(path: Path = DEFAULT_PATH, target_col: str = TARGET_COL):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Handle multi-document YAML (---)
    docs = list(yaml.safe_load_all(text))
    if len(docs) == 1:
        content = docs[0]
        rows = _extract_rows(content)
    else:
        # If multiple docs, each doc might be a record or list of records
        rows = []
        for d in docs:
            extracted = _extract_rows(d)
            rows.extend(extracted)

    df = pd.DataFrame(rows)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available columns: {list(df.columns)}"
        )

    # Convert boolean target -> 1/0
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    return X, y
