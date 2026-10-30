"""
Data Preprocessing Module for Retail Sales Prediction
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure utils can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.helpers import detect_dataset_dir


def inspect_raw_datasets(dataset_dir: str = None) -> dict:
    """
    Inspects all raw CSV files, checking shapes, null counts, duplicate records,
    and date coverage.
    """
    if dataset_dir is None:
        dataset_dir = detect_dataset_dir()

    files = [
        "train.csv", "stores.csv", "oil.csv",
        "holidays_events.csv", "transactions.csv",
        "test.csv", "sample_submission.csv"
    ]

    inspection_results = {}
    print("=" * 60)
    print("DATASET INSPECTION & INTEGRITY CHECK")
    print("=" * 60)

    for fname in files:
        fpath = os.path.join(dataset_dir, fname)
        if not os.path.exists(fpath):
            print(f"[ERROR] Required dataset file missing: {fname}")
            inspection_results[fname] = {"status": "missing"}
            continue

        # Read sample or full metadata
        df = pd.read_csv(fpath)
        null_counts = df.isnull().sum()
        null_summary = null_counts[null_counts > 0].to_dict()
        duplicate_count = int(df.duplicated().sum())

        date_min, date_max = None, None
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            date_min = str(df["date"].min().date())
            date_max = str(df["date"].max().date())

        info = {
            "status": "present",
            "shape": df.shape,
            "rows": df.shape[0],
            "cols": df.shape[1],
            "columns": list(df.columns),
            "missing_values": null_summary,
            "duplicate_rows": duplicate_count,
            "date_range": (date_min, date_max) if date_min else None
        }
        inspection_results[fname] = info

        print(f"\n[FILE] {fname}")
        print(f"  Shape: {info['rows']:,} rows, {info['cols']} columns")
        print(f"  Duplicates: {duplicate_count}")
        print(f"  Missing values: {null_summary if null_summary else 'None'}")
        if date_min:
            print(f"  Date range: {date_min} to {date_max}")

    print("\n" + "=" * 60)
    return inspection_results


def load_and_preprocess_data(dataset_dir: str = None, save_output: bool = True) -> pd.DataFrame:
    """
    Loads, cleans, aggregates daily data to monthly levels, and merges all
    associated datasets into a clean tabular DataFrame.
    """
    if dataset_dir is None:
        dataset_dir = detect_dataset_dir()

    print("\n[STEP 1/5] Loading raw datasets...")
    train_path = os.path.join(dataset_dir, "train.csv")
    stores_path = os.path.join(dataset_dir, "stores.csv")
    oil_path = os.path.join(dataset_dir, "oil.csv")
    holidays_path = os.path.join(dataset_dir, "holidays_events.csv")
    trans_path = os.path.join(dataset_dir, "transactions.csv")

    train_df = pd.read_csv(train_path)
    stores_df = pd.read_csv(stores_path)
    oil_df = pd.read_csv(oil_path)
    holidays_df = pd.read_csv(holidays_path)
    trans_df = pd.read_csv(trans_path)

    print(f"  Raw train records: {len(train_df):,}")
    print(f"  Stores records: {len(stores_df):,}")
    print(f"  Oil records: {len(oil_df):,}")
    print(f"  Holidays records: {len(holidays_df):,}")
    print(f"  Transactions records: {len(trans_df):,}")

    # 1. Date conversion
    print("\n[STEP 2/5] Parsing dates and correcting types...")
    train_df["date"] = pd.to_datetime(train_df["date"])
    train_df["year"] = train_df["date"].dt.year
    train_df["month"] = train_df["date"].dt.month

    # 2. Aggregating daily train sales to monthly sales by (year, month, store_nbr, family)
    print("\n[STEP 3/5] Aggregating daily records into monthly sales...")
    monthly_sales = train_df.groupby(
        ["year", "month", "store_nbr", "family"], as_index=False
    )[["sales", "onpromotion"]].sum()

    # Scale partial month Aug 2017 (ends on Aug 15 in train.csv) to monthly rate
    mask_aug_2017 = (monthly_sales["year"] == 2017) & (monthly_sales["month"] == 8)
    monthly_sales.loc[mask_aug_2017, "sales"] = monthly_sales.loc[mask_aug_2017, "sales"] * (31.0 / 15.0)
    monthly_sales["onpromotion"] = monthly_sales["onpromotion"].astype(float)
    monthly_sales.loc[mask_aug_2017, "onpromotion"] = monthly_sales.loc[mask_aug_2017, "onpromotion"] * (31.0 / 15.0)

    # 3. Process Oil Prices
    print("\n[STEP 4/5] Preprocessing auxiliary series (oil, transactions, holidays)...")
    oil_df["date"] = pd.to_datetime(oil_df["date"])
    min_date = train_df["date"].min()
    max_date = pd.to_datetime("2017-08-31")
    all_dates = pd.date_range(start=min_date, end=max_date, freq="D")
    oil_full = pd.DataFrame({"date": all_dates}).merge(oil_df, on="date", how="left")
    # Forward and backward fill oil prices for weekends/holidays
    oil_full["dcoilwtico"] = oil_full["dcoilwtico"].ffill().bfill()
    oil_full["year"] = oil_full["date"].dt.year
    oil_full["month"] = oil_full["date"].dt.month
    monthly_oil = oil_full.groupby(["year", "month"])["dcoilwtico"].mean().reset_index(name="oil_price")

    # 4. Process Transactions
    trans_df["date"] = pd.to_datetime(trans_df["date"])
    trans_df["year"] = trans_df["date"].dt.year
    trans_df["month"] = trans_df["date"].dt.month
    monthly_trans = trans_df.groupby(["year", "month", "store_nbr"])["transactions"].sum().reset_index(name="transaction_count")
    monthly_trans["transaction_count"] = monthly_trans["transaction_count"].astype(float)
    # Scale Aug 2017 transactions proportionally
    mask_aug_trans = (monthly_trans["year"] == 2017) & (monthly_trans["month"] == 8)
    monthly_trans.loc[mask_aug_trans, "transaction_count"] = monthly_trans.loc[mask_aug_trans, "transaction_count"] * (31.0 / 15.0)

    # 5. Process Holidays
    # Exclude transferred holidays (they were celebrated on a different day) and Work Day (undoing holidays)
    actual_holidays = holidays_df[
        (holidays_df["transferred"] == False) & (holidays_df["type"] != "Work Day")
    ].copy()
    actual_holidays["date"] = pd.to_datetime(actual_holidays["date"])
    actual_holidays["year"] = actual_holidays["date"].dt.year
    actual_holidays["month"] = actual_holidays["date"].dt.month

    # Deduplicate dates to prevent double counting if multiple events share a date
    unique_holidays = actual_holidays.drop_duplicates(subset=["date"])
    monthly_holidays = unique_holidays.groupby(["year", "month"]).size().reset_index(name="holiday_count")
    monthly_holidays["holiday_indicator"] = (monthly_holidays["holiday_count"] > 0).astype(int)

    # 6. Merge all datasets
    print("\n[STEP 5/5] Merging all preprocessed tables...")
    merged_df = monthly_sales.merge(stores_df, on="store_nbr", how="left")
    merged_df = merged_df.merge(monthly_oil, on=["year", "month"], how="left")
    merged_df = merged_df.merge(monthly_trans, on=["year", "month", "store_nbr"], how="left")
    merged_df = merged_df.merge(monthly_holidays, on=["year", "month"], how="left")

    # Fill transaction_count with 0 for stores not yet open
    merged_df["transaction_count"] = merged_df["transaction_count"].fillna(0.0)
    merged_df["holiday_count"] = merged_df["holiday_count"].fillna(0).astype(int)
    merged_df["holiday_indicator"] = merged_df["holiday_indicator"].fillna(0).astype(int)

    # Oil price fallback if any remains
    merged_df["oil_price"] = merged_df["oil_price"].ffill().bfill()

    print(f"  Preprocessed monthly records: {len(merged_df):,}")
    print(f"  Columns: {list(merged_df.columns)}")

    if save_output:
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "monthly_retail_sales.csv")
        merged_df.to_csv(out_path, index=False)
        print(f"  Saved preprocessed dataset to: {out_path}")

    return merged_df


if __name__ == "__main__":
    inspect_raw_datasets()
    df = load_and_preprocess_data()
    print("\nSample Preprocessed Data:")
    print(df.head())

