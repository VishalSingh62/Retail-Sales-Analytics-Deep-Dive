# -*- coding: utf-8 -*-
"""
Main Execution Pipeline for Predicting Retail Sales Using Linear Regression
Executes the end-to-end workflow: preprocessing, feature engineering, EDA,
model training, evaluation, and future forecasting verification.
"""

import sys

# Python version safeguard for environments where legacy Python 2 is mapped to 'python'
if sys.version_info[0] < 3:
    v_str = str(sys.version_info[0]) + "." + str(sys.version_info[1])
    sys.stderr.write(
        "\n" + "=" * 65 + "\n"
        "[ERROR] Python 3.10+ is required. You are running Python " + v_str + ".\n"
        "Your system's 'python' command is mapped to legacy Python 2.7 (C:\\Python27).\n\n"
        "To run the project with Python 3, please execute:\n"
        "    py -3 main.py\n"
        "  or:\n"
        "    python3.13 main.py\n\n"
        "To launch the dashboard:\n"
        "    py -3 -m streamlit run dashboard/app.py\n"
        "=" * 65 + "\n\n"
    )
    sys.exit(1)

import os
import argparse
import time

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_preprocessing import inspect_raw_datasets, load_and_preprocess_data
from src.feature_engineering import run_feature_engineering
from src.eda import generate_all_eda_visualizations
from src.train_model import train_linear_regression
from src.evaluate_model import evaluate_model
from src.prediction import FutureSalesPredictor
from utils.helpers import detect_dataset_dir, format_currency


def run_full_pipeline():
    start_time = time.time()
    print("=" * 70)
    print("  PREDICTING RETAIL SALES USING LINEAR REGRESSION")
    print("  End-to-End Machine Learning Pipeline Execution")
    print("=" * 70)

    # 1. Inspect and detect dataset
    dataset_dir = detect_dataset_dir()
    print("\n[INFO] Detected dataset directory: {}".format(dataset_dir))
    inspect_raw_datasets(dataset_dir)

    # 2. Preprocess Data
    print("\n" + "=" * 70)
    print("STAGE 1: DATA PREPROCESSING")
    print("=" * 70)
    preprocessed_df = load_and_preprocess_data(dataset_dir, save_output=True)

    # 3. Feature Engineering
    print("\n" + "=" * 70)
    print("STAGE 2: FEATURE ENGINEERING")
    print("=" * 70)
    featured_df = run_feature_engineering(preprocessed_df, save_output=True)

    # 4. Exploratory Data Analysis
    print("\n" + "=" * 70)
    print("STAGE 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    eda_charts = generate_all_eda_visualizations(featured_df)

    # 5. Train Linear Regression Model
    print("\n" + "=" * 70)
    print("STAGE 4: MACHINE LEARNING MODEL TRAINING")
    print("=" * 70)
    pipeline, train_df, test_df, metadata = train_linear_regression(featured_df)

    # 6. Evaluate Model
    print("\n" + "=" * 70)
    print("STAGE 5: MODEL EVALUATION")
    print("=" * 70)
    metrics = evaluate_model(model=pipeline, test_df=test_df)

    # 7. Test Future Prediction Engine
    print("\n" + "=" * 70)
    print("STAGE 6: FUTURE SALES PREDICTION VERIFICATION")
    print("=" * 70)
    predictor = FutureSalesPredictor()
    sample_pred = predictor.predict_single(store_nbr=1, family="GROCERY I", year=2017, month=9)
    print("  Test Store 1, GROCERY I (2017-09):")
    print("    Predicted Sales: {}".format(sample_pred['predicted_sales_formatted']))
    print("    Historical Average: {}".format(sample_pred['historical_average_formatted']))
    print("    Difference: {:+.2f}%".format(sample_pred['percentage_difference_vs_avg']))

    multi_forecast = predictor.forecast_horizons(store_nbr=1, family="GROCERY I", start_year=2017, start_month=9)
    print("\n  Multi-horizon Forecasts:")
    for h_name, summary in multi_forecast["horizon_summaries"].items():
        print("    {}-Month Forecast (through {}): {} (avg: {}/mo)".format(
            summary['horizon_months'], summary['end_period'],
            summary['total_predicted_sales_formatted'], summary['average_monthly_sales_formatted']
        ))

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY IN {:.1f} SECONDS".format(elapsed))
    print("=" * 70)
    print("\nSummary of Generated Artifacts:")
    print("  - Preprocessed dataset: outputs/monthly_retail_sales.csv")
    print("  - Featured dataset:     outputs/featured_monthly_retail_sales.csv")
    print("  - EDA Visualizations:   {} charts saved in outputs/eda/".format(len(eda_charts)))
    print("  - Trained Model:        models/linear_regression_model.pkl")
    print("  - Model Metadata:       models/model_metadata.json")
    print("  - Evaluation Metrics:   outputs/evaluation_metrics.json")
    print("  - Actual vs Predicted:  outputs/actual_vs_predicted.png")
    print("  - Test Predictions:     outputs/test_predictions.csv")
    print("\nTo launch the interactive dashboard:")
    print("  py -3 -m streamlit run dashboard/app.py")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retail Sales Forecasting Pipeline")
    parser.add_argument("--all", action="store_true", help="Run entire pipeline from scratch")
    args = parser.parse_args()

    run_full_pipeline()
