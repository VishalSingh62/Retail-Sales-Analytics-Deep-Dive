"""
Model Evaluation Module for Retail Sales Prediction
Evaluates model on holdout test partition, calculates MAE, RMSE, R², MAPE,
and creates publication-grade Actual vs Predicted charts.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.train_model import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_VARIABLE


def evaluate_model(
    model=None,
    test_df: pd.DataFrame = None,
    output_dir: str = None
) -> dict:
    """
    Evaluates the trained Linear Regression pipeline on holdout test data.
    """
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load test data first
    if test_df is None:
        featured_path = os.path.join(output_dir, "featured_monthly_retail_sales.csv")
        if os.path.exists(featured_path):
            full_df = pd.read_csv(featured_path)
        else:
            from src.feature_engineering import run_feature_engineering
            full_df = run_feature_engineering()
        test_df = full_df[full_df["year"] >= 2017].copy()

    # 2. Load model or fit on the fly if cross-environment pickle mismatch occurs
    if model is None:
        model_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "models", "linear_regression_model.pkl")
        )
        if sys.platform == "win32" and os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
            except Exception:
                model = None
        if model is None:
            from src.train_model import train_linear_regression
            featured_path = os.path.join(output_dir, "featured_monthly_retail_sales.csv")
            full_df = pd.read_csv(featured_path)
            model, _, _, _ = train_linear_regression(full_df, model_save_path=model_path)

    X_test = test_df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y_test = test_df[TARGET_VARIABLE].values

    # 3. Generate Predictions
    print("[1/3] Generating predictions on holdout test partition...")
    y_pred = model.predict(X_test)
    # Sales cannot be negative
    y_pred = np.clip(y_pred, 0, None)

    # 4. Compute Metrics
    print("[2/3] Calculating evaluation metrics...")
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))

    # MAPE on non-zero sales
    non_zero = y_test > 0
    mape_all = float(np.mean(np.abs((y_test[non_zero] - y_pred[non_zero]) / y_test[non_zero])) * 100)

    # Filtered MAPE for significant retail sales (> $100) to avoid division by tiny fractions
    sig_mask = y_test >= 100
    mape_significant = float(np.mean(np.abs((y_test[sig_mask] - y_pred[sig_mask]) / y_test[sig_mask])) * 100) if sig_mask.sum() > 0 else mape_all

    metrics = {
        "model_name": "Linear Regression",
        "primary_metric": "Mean Absolute Error (MAE)",
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2_Score": round(r2, 4),
        "MAPE_Percent": round(mape_all, 2),
        "MAPE_Significant_Percent": round(mape_significant, 2),
        "Test_Samples_Count": int(len(y_test)),
        "Test_Period": f"{test_df['year'].min()}-{test_df['month'].min():02d} to {test_df['year'].max()}-{test_df['month'].max():02d}"
    }

    print("\n" + "-" * 40)
    print("EVALUATION METRICS SUMMARY:")
    print("-" * 40)
    print(f"  Primary Metric (MAE):   ${metrics['MAE']:,.2f}")
    print(f"  RMSE:                   ${metrics['RMSE']:,.2f}")
    print(f"  R-Squared (R²):         {metrics['R2_Score']:.4f}")
    print(f"  MAPE (Sales >= $100):   {metrics['MAPE_Significant_Percent']:.2f}%")
    print(f"  MAPE (All non-zero):    {metrics['MAPE_Percent']:.2f}%")
    print(f"  Test Samples:           {metrics['Test_Samples_Count']:,}")
    print(f"  Test Period:            {metrics['Test_Period']}")
    print("-" * 40)

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved evaluation metrics to: {metrics_path}")

    # Attach predictions for visualization and dashboard
    test_eval_df = test_df.copy()
    test_eval_df["predicted_sales"] = y_pred
    test_eval_df["error"] = test_eval_df["predicted_sales"] - test_eval_df["sales"]
    test_eval_df["abs_error"] = test_eval_df["error"].abs()

    test_preds_path = os.path.join(output_dir, "test_predictions.csv")
    test_eval_df.to_csv(test_preds_path, index=False)
    print(f"Saved test predictions table to: {test_preds_path}")

    # 5. Create Actual vs Predicted Sales Visualization
    print("[3/3] Creating publication-ready Actual vs Predicted chart...")
    # Aggregate by Month to show the macro monthly trajectory
    monthly_agg = test_eval_df.groupby(["year", "month"])[["sales", "predicted_sales"]].sum().reset_index()
    monthly_agg["date_label"] = monthly_agg.apply(lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1)

    plt.figure(figsize=(12, 6), dpi=300)
    plt.plot(
        monthly_agg["date_label"],
        monthly_agg["sales"] / 1e6,
        marker="o",
        linewidth=2.8,
        color="#1565C0",
        label="Actual Retail Sales"
    )
    plt.plot(
        monthly_agg["date_label"],
        monthly_agg["predicted_sales"] / 1e6,
        marker="s",
        linewidth=2.8,
        linestyle="--",
        color="#E65100",
        label="Predicted Retail Sales (Linear Regression)"
    )

    plt.title("Actual vs Predicted Monthly Retail Sales (Holdout Test Period 2017)", pad=15, fontsize=14, fontweight="bold")
    plt.xlabel("Month", fontsize=12, fontweight="semibold")
    plt.ylabel("Total Sales (Millions USD)", fontsize=12, fontweight="semibold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, fontsize=11, loc="best")

    # Annotate summary box
    summary_text = (
        f"MAE: ${metrics['MAE']:,.2f}\n"
        f"RMSE: ${metrics['RMSE']:,.2f}\n"
        f"R²: {metrics['R2_Score']:.4f}"
    )
    plt.gca().text(
        0.03, 0.82,
        summary_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#90A4AE", alpha=0.9)
    )

    chart_path = os.path.join(output_dir, "actual_vs_predicted.png")
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Actual vs Predicted chart to: {chart_path}")

    return metrics


if __name__ == "__main__":
    evaluate_model()

