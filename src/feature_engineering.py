"""
Feature Engineering Module for Retail Sales Prediction
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure utils can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.helpers import get_season, count_weekend_days


def create_features(df: pd.DataFrame, drop_initial_lags: bool = True) -> pd.DataFrame:
    """
    Transforms the preprocessed monthly retail sales dataset into an enriched
    feature set ready for machine learning model training and inference.

    Features generated:
      - Calendar: year, month, quarter, season, month_number, weekend_indicator
      - Holiday: holiday_count, holiday_indicator
      - Economic & Operational: oil_price, transaction_count
      - Store attributes: city, state, type, cluster
      - Product: family
      - Lag & Rolling: lag_1_sales, lag_2_sales, lag_3_sales, rolling_mean_3_sales, lag_12_sales
    """
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    data = df.copy()

    # 1. Calendar Features
    print("[1/4] Generating calendar and seasonal features...")
    data["quarter"] = ((data["month"] - 1) // 3) + 1
    data["season"] = data["month"].apply(get_season)
    data["month_number"] = data["month"]

    # Calculate weekend days count for each (year, month)
    unique_ym = data[["year", "month"]].drop_duplicates()
    weekend_map = {
        (row["year"], row["month"]): count_weekend_days(int(row["year"]), int(row["month"]))
        for _, row in unique_ym.iterrows()
    }
    data["weekend_indicator"] = data.apply(
        lambda r: weekend_map.get((r["year"], r["month"]), 8), axis=1
    )

    # 2. Verify Holiday Features
    print("[2/4] Verifying holiday indicators...")
    if "holiday_count" not in data.columns:
        data["holiday_count"] = 0
    if "holiday_indicator" not in data.columns:
        data["holiday_indicator"] = (data["holiday_count"] > 0).astype(int)

    # 3. Lag and Rolling Window Features (strictly avoiding data leakage)
    print("[3/4] Constructing past lag and rolling window features (zero data leakage)...")
    # Sort strictly by store, family, year, month
    data = data.sort_values(["store_nbr", "family", "year", "month"]).reset_index(drop=True)

    series_group = data.groupby(["store_nbr", "family"])["sales"]

    # Past lags
    data["lag_1_sales"] = series_group.shift(1)
    data["lag_2_sales"] = series_group.shift(2)
    data["lag_3_sales"] = series_group.shift(3)

    # 3-Month Rolling Average strictly of past 3 months (lag_1, lag_2, lag_3)
    data["rolling_mean_3_sales"] = (data["lag_1_sales"] + data["lag_2_sales"] + data["lag_3_sales"]) / 3.0

    # Year-over-Year (12-month lag)
    data["lag_12_sales"] = series_group.shift(12)

    # Impute missing lag_12_sales for initial 12 months with available rolling or lag_1
    data["lag_12_sales"] = data["lag_12_sales"].fillna(data["rolling_mean_3_sales"]).fillna(data["lag_1_sales"])

    # 4. Handle initial missing lags
    print("[4/4] Filtering initial lag warm-up periods...")
    initial_rows = len(data)
    if drop_initial_lags:
        # Drop rows where 3-month lag history is not yet established (first 3 months of dataset)
        data = data.dropna(subset=["lag_1_sales", "lag_2_sales", "lag_3_sales", "rolling_mean_3_sales"]).copy()
        dropped = initial_rows - len(data)
        print(f"  Warm-up months dropped: {dropped:,} records (months 1-3 of 2013)")
        print(f"  Remaining training & testing samples: {len(data):,} records")

    return data


def run_feature_engineering(input_df: pd.DataFrame = None, save_output: bool = True) -> pd.DataFrame:
    """
    Orchestrates the feature engineering pipeline and optionally saves the output.
    """
    if input_df is None:
        from src.data_preprocessing import load_and_preprocess_data
        input_df = load_and_preprocess_data(save_output=False)

    featured_df = create_features(input_df, drop_initial_lags=True)

    if save_output:
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "featured_monthly_retail_sales.csv")
        featured_df.to_csv(out_path, index=False)
        print(f"Saved featured dataset to: {out_path}")

    return featured_df


if __name__ == "__main__":
    df = run_feature_engineering()
    print("\nFeature Engineering Complete! Columns:")
    print(list(df.columns))

