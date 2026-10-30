"""
Helper Utilities for Predicting Retail Sales Using Linear Regression
"""

import os
import calendar
import pandas as pd
import numpy as np


def detect_dataset_dir(base_dir=None):
    """
    Automatically detects the dataset directory containing the required Kaggle CSV files.
    """
    candidates = [
        os.path.join(base_dir, "dataset") if base_dir else None,
        r"D:\Predicting-Retail-Sales\dataset",
        os.path.abspath("dataset"),
        os.path.abspath("../dataset"),
        os.path.join(os.path.dirname(__file__), "..", "dataset"),
    ]
    
    required_files = [
        "train.csv", "stores.csv", "oil.csv", 
        "holidays_events.csv", "transactions.csv"
    ]
    
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            # Check if all required files exist
            if all(os.path.exists(os.path.join(candidate, f)) for f in required_files):
                return os.path.abspath(candidate)
                
    raise FileNotFoundError(
        f"Could not locate dataset directory containing required files: {required_files}.\n"
        f"Searched paths: {[c for c in candidates if c]}"
    )


def get_season(month: int) -> str:
    """
    Maps a month (1-12) to a meteorological season name.
    """
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"


def count_weekend_days(year: int, month: int) -> int:
    """
    Calculates the exact number of weekend days (Saturday and Sunday) in a given month.
    """
    _, num_days = calendar.monthrange(year, month)
    days = pd.date_range(start=f"{year}-{month:02d}-01", periods=num_days, freq="D")
    return int((days.dayofweek >= 5).sum())


def format_currency(value: float) -> str:
    """
    Formats a numeric value as a clean retail currency string ($XX,XXX.XX).
    """
    if pd.isna(value) or value is None:
        return "$0.00"
    return f"${value:,.2f}"


def format_number(value: float) -> str:
    """
    Formats an integer or float with thousands separators.
    """
    if pd.isna(value) or value is None:
        return "0"
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"

