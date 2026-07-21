"""
preprocessing.py
-----------------
Load and clean the Telco Customer Churn dataset.

Usage (from project root, with venv activated):
    python src/preprocessing.py
"""

import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")
PROCESSED_DATA_PATH = Path("data/processed/churn_clean.csv")


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame."""
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Telco churn dataset.

    Known issues in this dataset:
    - 'TotalCharges' is read as a string/object because some rows have
      blank values (new customers with 0 tenure) instead of numbers.
    - 'customerID' is just an identifier, not a useful feature.
    """
    df = df.copy()

    # 1. Fix TotalCharges: convert to numeric, coercing blanks to NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # These NaNs are customers with 0 tenure (they haven't been charged yet).
    # Safest fix: fill with 0, since tenure = 0 for these rows.
    n_missing = df["TotalCharges"].isna().sum()
    print(f"Found {n_missing} missing TotalCharges values (new customers) -> filling with 0")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # 2. Drop customerID - it's an identifier, not a predictive feature
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # 3. Standardize target column to binary 0/1
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # 4. Check for duplicates
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        print(f"Found {n_dupes} duplicate rows -> dropping")
        df = df.drop_duplicates()

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical columns.
    SeniorCitizen is already 0/1 in the raw data, so we leave it as is.
    """
    df = df.copy()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    print(f"One-hot encoding {len(categorical_cols)} categorical columns: {categorical_cols}")

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded


def run_pipeline() -> pd.DataFrame:
    """Full pipeline: load -> clean -> encode -> save."""
    df = load_data()
    df = clean_data(df)

    print("\nChurn distribution (class imbalance check):")
    print(df["Churn"].value_counts(normalize=True).round(3))

    df_encoded = encode_features(df)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_encoded.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\nSaved processed data to {PROCESSED_DATA_PATH}")
    print(f"Final shape: {df_encoded.shape}")

    return df_encoded


if __name__ == "__main__":
    run_pipeline()
