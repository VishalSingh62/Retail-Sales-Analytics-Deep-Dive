"""
Future Sales Prediction Engine for Retail Sales Forecasting
Supports single-point prediction and recursive multi-horizon (3, 6, 12-month) forecasts.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.helpers import get_season, count_weekend_days, format_currency
from src.train_model import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


class FutureSalesPredictor:
    """
    Predictive forecasting engine powered by the trained Linear Regression pipeline.
    Handles future recursive lag computation and realistic baseline assumptions
    for operational and economic variables.
    """

    def __init__(self, model_path: str = None, historical_data_path: str = None):
        if model_path is None:
            model_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "models", "linear_regression_model.pkl")
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        self.model = joblib.load(model_path)
        self.model_path = model_path

        # Load historical dataset to establish baseline lags and store profiles
        if historical_data_path is None:
            historical_data_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "outputs", "featured_monthly_retail_sales.csv")
            )
        
        if not os.path.exists(historical_data_path):
            from src.feature_engineering import run_feature_engineering
            self.history_df = run_feature_engineering()
        else:
            self.history_df = pd.read_csv(historical_data_path)

        # Build store metadata lookup
        store_meta = self.history_df[["store_nbr", "type", "city", "state", "cluster"]].drop_duplicates()
        self.store_info = store_meta.set_index("store_nbr").to_dict(orient="index")

        # Baseline oil price (average of the last 6 recorded historical months)
        recent_oil = self.history_df[self.history_df["year"] == self.history_df["year"].max()]["oil_price"]
        self.default_oil_price = float(recent_oil.mean()) if not recent_oil.empty else 47.50

        # Store-month average transaction counts
        self.trans_lookup = self.history_df.groupby(["store_nbr", "month"])["transaction_count"].mean().to_dict()
        self.store_avg_trans = self.history_df.groupby("store_nbr")["transaction_count"].mean().to_dict()

        # Last recorded period in historical dataset
        max_record = self.history_df.sort_values(["year", "month"]).iloc[-1]
        self.last_year = int(max_record["year"])
        self.last_month = int(max_record["month"])

    def get_latest_lags(self, store_nbr: int, family: str) -> dict:
        """
        Retrieves the most recent known lag values for a given store and product family.
        """
        series = self.history_df[
            (self.history_df["store_nbr"] == store_nbr) & 
            (self.history_df["family"] == family)
        ].sort_values(["year", "month"])

        if series.empty:
            # Fallback to family average if series is unseen
            fam_data = self.history_df[self.history_df["family"] == family]
            avg_val = float(fam_data["sales"].mean()) if not fam_data.empty else 1000.0
            return {
                "lag_1_sales": avg_val,
                "lag_2_sales": avg_val,
                "lag_3_sales": avg_val,
                "rolling_mean_3_sales": avg_val,
                "lag_12_sales": avg_val
            }

        last_row = series.iloc[-1]
        sales_history = series["sales"].tolist()

        lag_1 = sales_history[-1] if len(sales_history) >= 1 else 0.0
        lag_2 = sales_history[-2] if len(sales_history) >= 2 else lag_1
        lag_3 = sales_history[-3] if len(sales_history) >= 3 else lag_2
        rolling_3 = (lag_1 + lag_2 + lag_3) / 3.0
        lag_12 = sales_history[-12] if len(sales_history) >= 12 else rolling_3

        return {
            "lag_1_sales": float(lag_1),
            "lag_2_sales": float(lag_2),
            "lag_3_sales": float(lag_3),
            "rolling_mean_3_sales": float(rolling_3),
            "lag_12_sales": float(lag_12)
        }

    def predict_single(
        self,
        store_nbr: int,
        family: str,
        year: int,
        month: int,
        season: str = None,
        holiday_indicator: int = 0,
        custom_oil_price: float = None,
        custom_transaction_count: float = None
    ) -> dict:
        """
        Generates sales prediction for a specific future month, using valid
        historical lags or recursively generated predictions.
        """
        if season is None:
            season = get_season(month)

        store_meta = self.store_info.get(store_nbr, {
            "type": "A", "city": "Quito", "state": "Pichincha", "cluster": 1
        })

        oil_price = custom_oil_price if custom_oil_price is not None else self.default_oil_price
        
        # Estimate transaction count based on historical seasonality of the store
        if custom_transaction_count is not None:
            transaction_count = custom_transaction_count
        else:
            transaction_count = self.trans_lookup.get(
                (store_nbr, month),
                self.store_avg_trans.get(store_nbr, 30000.0)
            )

        quarter = ((month - 1) // 3) + 1
        weekend_indicator = count_weekend_days(year, month)
        holiday_count = 1 if holiday_indicator else 0

        # Retrieve lags
        lags = self.get_latest_lags(store_nbr, family)

        feature_dict = {
            "store_nbr": int(store_nbr),
            "family": str(family),
            "type": str(store_meta["type"]),
            "city": str(store_meta["city"]),
            "state": str(store_meta["state"]),
            "season": str(season),
            "cluster": int(store_meta["cluster"]),
            "year": int(year),
            "month": int(month),
            "quarter": int(quarter),
            "month_number": int(month),
            "holiday_indicator": int(holiday_indicator),
            "holiday_count": int(holiday_count),
            "weekend_indicator": int(weekend_indicator),
            "oil_price": float(oil_price),
            "transaction_count": float(transaction_count),
            "lag_1_sales": lags["lag_1_sales"],
            "lag_2_sales": lags["lag_2_sales"],
            "lag_3_sales": lags["lag_3_sales"],
            "rolling_mean_3_sales": lags["rolling_mean_3_sales"],
            "lag_12_sales": lags["lag_12_sales"]
        }

        X_input = pd.DataFrame([feature_dict])[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
        pred_value = float(self.model.predict(X_input)[0])
        pred_value = max(0.0, pred_value)  # Retail sales cannot be negative

        # Historical comparison
        hist_series = self.history_df[
            (self.history_df["store_nbr"] == store_nbr) & 
            (self.history_df["family"] == family)
        ]["sales"]
        hist_avg = float(hist_series.mean()) if not hist_series.empty else pred_value
        pct_diff = ((pred_value - hist_avg) / hist_avg * 100) if hist_avg > 0 else 0.0

        assumptions = [
            f"Oil price assumed at ${oil_price:.2f}/barrel based on recent stable macroeconomic trends.",
            f"Store footfall transactions estimated at {transaction_count:,.0f} using historical monthly patterns.",
            f"Calendar computed: {weekend_indicator} weekend days and {holiday_count} holiday(s) in {calendar_month_name(month)} {year}.",
            f"Lags seeded from historical retail trajectory: Lag 1 = {format_currency(lags['lag_1_sales'])}, 3M Rolling Mean = {format_currency(lags['rolling_mean_3_sales'])}."
        ]

        return {
            "store_nbr": store_nbr,
            "family": family,
            "year": year,
            "month": month,
            "period": f"{year}-{month:02d}",
            "season": season,
            "holiday_indicator": holiday_indicator,
            "predicted_sales": round(pred_value, 2),
            "predicted_sales_formatted": format_currency(pred_value),
            "historical_average_sales": round(hist_avg, 2),
            "historical_average_formatted": format_currency(hist_avg),
            "percentage_difference_vs_avg": round(pct_diff, 2),
            "features_used": feature_dict,
            "assumptions": assumptions
        }

    def forecast_horizons(
        self,
        store_nbr: int,
        family: str,
        start_year: int = 2017,
        start_month: int = 9,
        horizons: list = [3, 6, 12]
    ) -> dict:
        """
        Recursively forecasts sales for consecutive future months up to the maximum horizon.
        Uses model predictions recursively to update lag_1, lag_2, lag_3, and rolling_3.
        """
        max_horizon = max(horizons)
        curr_year = start_year
        curr_month = start_month

        forecast_records = []
        lags = self.get_latest_lags(store_nbr, family)

        lag_1 = lags["lag_1_sales"]
        lag_2 = lags["lag_2_sales"]
        lag_3 = lags["lag_3_sales"]
        rolling_3 = lags["rolling_mean_3_sales"]
        lag_12 = lags["lag_12_sales"]

        store_meta = self.store_info.get(store_nbr, {
            "type": "A", "city": "Quito", "state": "Pichincha", "cluster": 1
        })

        for step in range(1, max_horizon + 1):
            season = get_season(curr_month)
            quarter = ((curr_month - 1) // 3) + 1
            weekend_indicator = count_weekend_days(curr_year, curr_month)
            holiday_indicator = 1 if curr_month in [1, 5, 8, 10, 11, 12] else 0
            holiday_count = 1 if holiday_indicator else 0
            oil_price = self.default_oil_price

            transaction_count = self.trans_lookup.get(
                (store_nbr, curr_month),
                self.store_avg_trans.get(store_nbr, 30000.0)
            )

            feature_dict = {
                "store_nbr": int(store_nbr),
                "family": str(family),
                "type": str(store_meta["type"]),
                "city": str(store_meta["city"]),
                "state": str(store_meta["state"]),
                "season": str(season),
                "cluster": int(store_meta["cluster"]),
                "year": int(curr_year),
                "month": int(curr_month),
                "quarter": int(quarter),
                "month_number": int(curr_month),
                "holiday_indicator": int(holiday_indicator),
                "holiday_count": int(holiday_count),
                "weekend_indicator": int(weekend_indicator),
                "oil_price": float(oil_price),
                "transaction_count": float(transaction_count),
                "lag_1_sales": float(lag_1),
                "lag_2_sales": float(lag_2),
                "lag_3_sales": float(lag_3),
                "rolling_mean_3_sales": float(rolling_3),
                "lag_12_sales": float(lag_12)
            }

            X_input = pd.DataFrame([feature_dict])[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
            pred_step = float(self.model.predict(X_input)[0])
            pred_step = max(0.0, pred_step)

            forecast_records.append({
                "step": step,
                "period": f"{curr_year}-{curr_month:02d}",
                "year": curr_year,
                "month": curr_month,
                "season": season,
                "holiday_indicator": holiday_indicator,
                "predicted_sales": round(pred_step, 2),
                "predicted_sales_formatted": format_currency(pred_step)
            })

            # Recursive update of lags for the next step
            lag_3 = lag_2
            lag_2 = lag_1
            lag_1 = pred_step
            rolling_3 = (lag_1 + lag_2 + lag_3) / 3.0
            # Advance month
            if curr_month == 12:
                curr_month = 1
                curr_year += 1
            else:
                curr_month += 1

        forecast_df = pd.DataFrame(forecast_records)

        # Calculate totals per requested horizon
        horizon_summaries = {}
        for h in horizons:
            sub = forecast_df[forecast_df["step"] <= h]
            tot_sales = float(sub["predicted_sales"].sum())
            avg_sales = float(sub["predicted_sales"].mean())
            horizon_summaries[f"{h}_month"] = {
                "horizon_months": h,
                "total_predicted_sales": round(tot_sales, 2),
                "total_predicted_sales_formatted": format_currency(tot_sales),
                "average_monthly_sales": round(avg_sales, 2),
                "average_monthly_sales_formatted": format_currency(avg_sales),
                "end_period": sub.iloc[-1]["period"]
            }

        return {
            "store_nbr": store_nbr,
            "family": family,
            "forecast_table": forecast_df,
            "horizon_summaries": horizon_summaries
        }


def calendar_month_name(month: int) -> str:
    import calendar
    return calendar.month_name[month]


if __name__ == "__main__":
    predictor = FutureSalesPredictor()
    res = predictor.predict_single(store_nbr=1, family="GROCERY I", year=2017, month=9)
    print(f"Prediction result: {res['predicted_sales_formatted']} (Diff: {res['percentage_difference_vs_avg']}%)")
    
    multi = predictor.forecast_horizons(store_nbr=1, family="GROCERY I", start_year=2017, start_month=9)
    print("\nHorizon Summaries:")
    for k, v in multi["horizon_summaries"].items():
        print(f"  {k}: {v['total_predicted_sales_formatted']} (avg: {v['average_monthly_sales_formatted']}/mo)")

