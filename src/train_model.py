"""
Model Training Module for Retail Sales Prediction
Trains a Scikit-learn Linear Regression model using ColumnTransformer and Pipeline.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


CATEGORICAL_FEATURES = ["store_nbr", "family", "type", "city", "state", "season", "cluster"]
NUMERICAL_FEATURES = [
    "year", "month", "quarter", "month_number", "holiday_indicator",
    "holiday_count", "weekend_indicator", "oil_price", "transaction_count",
    "lag_1_sales", "lag_2_sales", "lag_3_sales", "rolling_mean_3_sales", "lag_12_sales"
]
TARGET_VARIABLE = "sales"


def train_linear_regression(
    df: pd.DataFrame,
    test_year_cutoff: int = 2017,
    model_save_path: str = None
) -> tuple:
    """
    Performs chronological train/test split, builds the preprocessing and Linear Regression
    pipeline, fits strictly on the training partition, and persists the artifact.
    """
    print("=" * 60)
    print("TRAINING LINEAR REGRESSION MODEL")
    print("=" * 60)

    if model_save_path is None:
        model_save_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "models", "linear_regression_model.pkl")
        )
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # 1. Chronological Train-Test Split (strictly avoiding data leakage)
    print(f"[1/4] Performing chronological split (Test Period: Year >= {test_year_cutoff})...")
    train_mask = df["year"] < test_year_cutoff
    test_mask = df["year"] >= test_year_cutoff

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    print(f"  Training set: {len(train_df):,} records ({train_df['year'].min()}-{train_df['month'].min():02d} to {train_df['year'].max()}-{train_df['month'].max():02d})")
    print(f"  Testing set:  {len(test_df):,} records ({test_df['year'].min()}-{test_df['month'].min():02d} to {test_df['year'].max()}-{test_df['month'].max():02d})")

    X_train = train_df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y_train = train_df[TARGET_VARIABLE]
    X_test = test_df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y_test = test_df[TARGET_VARIABLE]

    # 2. Build Pipeline
    print("[2/4] Constructing ColumnTransformer and Pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERICAL_FEATURES)
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ]
    )

    # 3. Fit Pipeline (strictly on training data)
    print("[3/4] Fitting Linear Regression model strictly on training data...")
    pipeline.fit(X_train, y_train)

    # 4. Save Artifacts
    print(f"[4/4] Persisting trained model pipeline to: {model_save_path}...")
    joblib.dump(pipeline, model_save_path)

    metadata = {
        "model_type": "LinearRegression",
        "target_variable": TARGET_VARIABLE,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "test_year_cutoff": test_year_cutoff,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_date_range": [f"{train_df['year'].min()}-{train_df['month'].min():02d}", f"{train_df['year'].max()}-{train_df['month'].max():02d}"],
        "test_date_range": [f"{test_df['year'].min()}-{test_df['month'].min():02d}", f"{test_df['year'].max()}-{test_df['month'].max():02d}"]
    }

    meta_path = os.path.join(os.path.dirname(model_save_path), "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Model and metadata saved successfully! Metadata:\n{json.dumps(metadata, indent=2)}")
    return pipeline, train_df, test_df, metadata


if __name__ == "__main__":
    from src.feature_engineering import run_feature_engineering
    df = run_feature_engineering()
    train_linear_regression(df)

