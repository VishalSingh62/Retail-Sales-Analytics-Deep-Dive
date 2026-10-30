# Predicting Retail Sales Using Linear Regression

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, end-to-end Machine Learning solution for forecasting monthly store-level retail sales using Scikit-Learn Linear Regression, automated feature engineering, and an interactive modern SaaS analytics dashboard built with Streamlit and Plotly.

---

## Table of Contents
1. [Project Introduction](#project-introduction)
2. [Problem Statement](#problem-statement)
3. [Project Objectives](#project-objectives)
4. [Dataset Description & Source](#dataset-description--source)
5. [Technologies Used](#technologies-used)
6. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
7. [Feature Engineering & Anti-Leakage Safeguards](#feature-engineering--anti-leakage-safeguards)
8. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
9. [Machine Learning Methodology](#machine-learning-methodology)
10. [Linear Regression In-Depth Explanation](#linear-regression-in-depth-explanation)
11. [Model Evaluation & Benchmarking](#model-evaluation--benchmarking)
12. [Interactive Streamlit Dashboard Features](#interactive-streamlit-dashboard-features)
13. [Future Sales Forecasting Engine](#future-sales-forecasting-engine)
14. [Project Folder Structure](#project-folder-structure)
15. [Installation & Setup](#installation--setup)
16. [How to Run the Project](#how-to-run-the-project)
17. [Future Improvements](#future-improvements)

---

## 1. Project Introduction
Accurate retail sales forecasting is pivotal for modern supply chain management, inventory allocation, cash flow planning, and promotional strategy. This project presents a complete, production-grade Machine Learning solution titled **"Predicting Retail Sales Using Linear Regression"**, trained on historical sales records from Corporación Favorita, one of Ecuador's premier grocery retailers.

By harmonizing daily transactional data across 54 brick-and-mortar stores and 33 product families with macroeconomic oil price movements, Ecuadorian national and regional holiday schedules, and time-lagged sales patterns, the system accurately forecasts monthly sales volumes.

---

## 2. Problem Statement
Retail demand is influenced by non-linear calendar seasonality, holiday shifts, local economic indicators (such as oil price shocks in oil-dependent economies like Ecuador), and chain-wide footfall patterns. Retail enterprises frequently face challenges with:
- **Stock-outs** of critical grocery items leading to revenue loss and customer dissatisfaction.
- **Overstocking** perishable inventory causing spoilage and increased holding costs.
- **Inaccurate aggregate planning** across distinct product families and geographic clusters.

The core challenge is to construct an interpretable, highly reliable Linear Regression forecasting pipeline that transforms granular daily observations into actionable monthly store-family demand predictions without lookahead bias or data leakage.

---

## 3. Project Objectives
- **End-to-End Automation**: Ingest, clean, validate, and merge 5 separate relational data sources into a consolidated monthly analytics schema.
- **Strict Anti-Leakage Feature Engineering**: Build rolling window averages, autoregressive lags, seasonal components, and macroeconomic indicators derived strictly from historical data.
- **Interpretable Machine Learning**: Train an Ordinary Least Squares (OLS) Linear Regression model wrapped in an automated Scikit-Learn `Pipeline` with `ColumnTransformer` preprocessing.
- **Rigorous Chronological Evaluation**: Validate model generalizations using a forward-chaining temporal holdout test partition (2017) evaluated on Mean Absolute Error (MAE), RMSE, R² Score, and MAPE.
- **Recursive Multi-Horizon Forecasting**: Enable real-time single-period predictions as well as multi-quarter forward forecasts (3-month, 6-month, and 12-month).
- **Executive SaaS Dashboard**: Deliver an interactive Streamlit web dashboard (*RetailVision*) with interactive Plotly visualizations, scenario planning tools, and dataset export capabilities.

---

## 4. Dataset Description & Source

The dataset used in this project is sourced from the official Kaggle Competition:
**[Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)**

### Relational Schema Summary:
| File Name | Records | Description | Primary Key / Merge Keys |
| :--- | :--- | :--- | :--- |
| `train.csv` | 3,000,888 | Daily historical store-family sales and promotion records (Jan 2013 – Aug 2017) | `id` (date, store_nbr, family) |
| `stores.csv` | 54 | Store metadata including city, state, store type (A-E), and geographic cluster | `store_nbr` |
| `oil.csv` | 1,218 | Daily WTI crude oil price (dcoilwtico) reflecting Ecuador's economic health | `date` |
| `holidays_events.csv` | 350 | National, regional, and local Ecuadorian holidays, events, bridges, and transfer flags | `date` |
| `transactions.csv` | 83,488 | Daily transaction (footfall) counts recorded per physical store | `date`, `store_nbr` |
| `test.csv` | 28,512 | Test set features for final 16 days of August 2017 | `id` |
| `sample_submission.csv`| 28,512 | Standard submission format template | `id` |

---

## 5. Technologies Used
- **Programming Language**: Python 3.10+
- **Data Manipulation & Analysis**: Pandas 2.2+, NumPy 2.2+
- **Machine Learning & Pipeline**: Scikit-Learn 1.6+, Joblib 1.4+
- **Data Visualization (Static EDA)**: Matplotlib 3.10+, Seaborn 0.13+
- **Interactive Web Dashboard**: Streamlit 1.30+
- **Interactive Visualizations (Web)**: Plotly Express & Plotly Graph Objects 5.15+

---

## 6. Data Preprocessing Pipeline
Implemented in `src/data_preprocessing.py`:
1. **Data Inspection**: Programmatically scans all CSV files, verifies record counts, detects null values, and screens for duplicate rows.
2. **Date Normalization**: Standardizes timestamp strings into ISO `datetime64[ns]` objects, extracting calendar year, month, and day components.
3. **Monthly Aggregation**: Aggregates 3 million granular daily sales transactions into monthly store-family observations `(year, month, store_nbr, family)`.
4. **Partial Month Normalization**: In `train.csv`, August 2017 contains 15 recorded days. The preprocessing engine scales sales, promotions, and transactions proportionally $(31.0 / 15.0)$ to monthly equivalent rates, preventing artificial volume dips in time-series analysis.
5. **Oil Price Imputation**: Daily missing oil prices (weekends and market holidays) are imputed via chronological forward-fill followed by backward-fill, then averaged to compute monthly mean crude oil prices.
6. **Holiday Event De-biasing**: Filters out transferred holidays (`transferred == True`, where festivities moved to another date) and excludes `Work Day` types that cancel out holidays. Deduplicates dates to prevent synthetic record inflation.
7. **Clean Left Joins**: Merges `stores.csv`, `oil.csv`, `transactions.csv`, and aggregated holidays without generating duplicate rows. Unopened stores receive 0 transactions.

---

## 7. Feature Engineering & Anti-Leakage Safeguards
Implemented in `src/feature_engineering.py`:

### Feature Inventory:
1. **Calendar & Seasonality**:
   - `year`: Numeric year $(2013-2017)$.
   - `month` & `month_number`: Month of the year $(1-12)$.
   - `quarter`: Business quarter $(Q1-Q4)$.
   - `season`: Meteorological season (`Winter`, `Spring`, `Summer`, `Fall`).
   - `weekend_indicator`: Total weekend days (Saturdays and Sundays) in that specific month.
2. **Holiday Dimensions**:
   - `holiday_count`: Number of official holidays occurring during the month.
   - `holiday_indicator`: Binary flag ($1$ if month has at least one holiday, else $0$).
3. **Macroeconomic & Operational Drivers**:
   - `oil_price`: Monthly average WTI crude oil price per barrel.
   - `transaction_count`: Total customer footfall transactions recorded by the store.
4. **Hierarchical Categorical Identifiers**:
   - `store_nbr`, `city`, `state`, `type`, `cluster`, `family`.
5. **Autoregressive Lag & Rolling Features (Zero Lookahead Bias)**:
   - `lag_1_sales`: Sales from the previous month $(t - 1)$.
   - `lag_2_sales`: Sales from 2 months prior $(t - 2)$.
   - `lag_3_sales`: Sales from 3 months prior $(t - 3)$.
   - `rolling_mean_3_sales`: 3-month moving average computed strictly across $(lag_1 + lag_2 + lag_3) / 3$.
   - `lag_12_sales`: Year-over-Year sales from the exact same month in the prior calendar year $(t - 12)$.

---

## 8. Exploratory Data Analysis (EDA)
The system automatically generates 10 high-resolution (300 DPI) publication-quality charts saved to `outputs/eda/`:

1. **Monthly Sales Trend (`monthly_sales_trend.png`)**: Macro historical sales trajectory across Ecuador from 2013 to 2017.
2. **Yearly Sales Comparison (`yearly_sales_comparison.png`)**: Overlaid monthly seasonal curves illustrating Q4 holiday sales surges.
3. **Holiday vs Non-Holiday Sales (`holiday_vs_non_holiday.png`)**: Average sales lift during holiday vs non-holiday months.
4. **Seasonal Sales Analysis (`seasonal_sales_analysis.png`)**: Mean sales comparison across Spring, Summer, Fall, and Winter.
5. **Sales by Product Family (`sales_by_product_family.png`)**: Comprehensive breakdown of sales across all 33 retail product lines.
6. **Sales by Store (`sales_by_store.png`)**: Chain-wide sales performance across all 54 physical retail stores.
7. **Monthly Sales Distribution (`monthly_sales_distribution.png`)**: Log-scaled distribution of monthly store-family sales.
8. **Correlation Heatmap (`correlation_heatmap.png`)**: Pearson correlation matrix between sales, autoregressive lags, oil prices, and transactions.
9. **Top 10 Performing Stores (`top_10_stores.png`)**: Highest revenue-generating retail outlets categorized by city and store type.
10. **Top 10 Product Families (`top_10_product_families.png`)**: Core volume drivers including Grocery I, Beverages, and Produce.

---

## 9. Machine Learning Methodology
- **Validation Scheme**: **Chronological Time-Series Holdout Split**. Random train-test splitting introduces severe temporal leakage in time-series forecasting. Hence, the model trains strictly on earlier history (`2013-04` through `2016-12`, 80,190 records) and is evaluated on a completely unseen future holdout period (`2017-01` through `2017-08`, 14,256 records).
- **Preprocessing Encapsulation**: A Scikit-Learn `ColumnTransformer` applies `OneHotEncoder(handle_unknown='ignore')` to categorical attributes (`store_nbr`, `family`, `type`, `city`, `state`, `season`, `cluster`) and `StandardScaler()` to all numerical predictors.
- **Fitting Rule**: The preprocessor is fitted **only on the training partition** to prevent test distribution leakage.

---

## 10. Linear Regression In-Depth Explanation
The model estimates parameters $\beta$ using Ordinary Least Squares (OLS):

$$\hat{y} = \beta_0 + \sum_{j=1}^{p} \beta_j X_j$$

Where:
- $\hat{y}$ is the predicted monthly sales for a given store and product family.
- $X_j$ represents one-hot encoded store and product family indicators, scaled macroeconomic variables (oil price, transaction count), seasonal and holiday indicators, and autoregressive lag predictors ($\text{Lag}_1, \text{Lag}_2, \text{Lag}_3, \text{Rolling}_3, \text{Lag}_{12}$).
- The loss function minimized is the Residual Sum of Squares (RSS):

$$RSS(\beta) = \sum_{i=1}^{N} \left(y_i - \hat{y}_i\right)^2$$

Linear Regression provides superior interpretability, fast training times, deterministic convergence, and robust baseline generalization when paired with autoregressive and seasonal feature engineering.

---

## 11. Model Evaluation & Benchmarking
Evaluated on the unseen 2017 holdout test set (14,256 observations):

| Evaluation Metric | Test Result | Description |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | **$1,829.07** | Primary evaluation metric (mean absolute dollar difference per store-family) |
| **Root Mean Squared Error (RMSE)** | **$5,046.73** | Standard deviation of prediction residuals (penalizes extreme deviations) |
| **Coefficient of Determination ($R^2$)**| **0.9833 (98.33%)** | Proportion of variance in retail sales explained by the regression pipeline |
| **MAPE (Sales $\ge$ $100)** | **18.20%** | Mean absolute percentage error for economically meaningful sales |

Evaluation charts and metric JSON records are persisted in `outputs/actual_vs_predicted.png` and `outputs/evaluation_metrics.json`.

---

## 12. Interactive Streamlit Dashboard Features
The dashboard (**RetailVision**) features 5 comprehensive sections:
1. **Overview Dashboard**: SaaS KPI cards (Total Sales, Average Monthly Sales, Peak Month, Store Count, Family Count, Model MAE) and macro Plotly visualizations.
2. **Sales Analytics**: Dynamic multidimensional filtering across Year, Month, Store, Product Family, and Season with interactive category breakdown charts.
3. **Machine Learning Model**: Interactive Actual vs Predicted time-series charts, metric cards, and pipeline architecture specifications.
4. **Future Sales Prediction**: Interactive prediction playground where users select scenario parameters and obtain instant forecasts, historical benchmark comparisons, and 3-, 6-, and 12-month forward forecasts.
5. **Dataset Explorer**: Schema inspection, record counters, missing value tables, interactive data table preview, and 1-click filtered CSV export.

---

## 13. Future Sales Forecasting Engine
Implemented in `src/prediction.py`:
- **Single-Month Prediction**: Accepts any scenario (Store, Family, Year, Month, Season, Holiday). Retrieves the latest known historical lags and operational estimates (seasonal store transactions and stable oil price baselines) to output immediate demand projections.
- **Multi-Period Recursive Forecasting**: Generates multi-step forecasts for **3 months, 6 months, and 12 months ahead**. At each step, previous predictions are recursively fed into $\text{Lag}_1$, advancing $\text{Lag}_2$, $\text{Lag}_3$, and updating 3-month rolling averages without requiring future actuals.

---

## 14. Project Folder Structure
```
Predicting-Retail-Sales/
│
├── dataset/
│   ├── train.csv                      # Historical daily store sales (Kaggle)
│   ├── test.csv                       # Test dataset
│   ├── stores.csv                     # Store location and type metadata
│   ├── holidays_events.csv            # Holidays and national events
│   ├── oil.csv                        # Daily crude oil prices
│   ├── transactions.csv               # Store footfall transaction counts
│   └── sample_submission.csv          # Submission sample
│
├── models/
│   ├── linear_regression_model.pkl    # Serialized Scikit-learn Pipeline
│   └── model_metadata.json            # Model specifications and parameters
│
├── outputs/
│   ├── eda/                           # 10 High-res Matplotlib/Seaborn charts
│   │   ├── monthly_sales_trend.png
│   │   ├── yearly_sales_comparison.png
│   │   ├── holiday_vs_non_holiday.png
│   │   ├── seasonal_sales_analysis.png
│   │   ├── sales_by_product_family.png
│   │   ├── sales_by_store.png
│   │   ├── monthly_sales_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── top_10_stores.png
│   │   └── top_10_product_families.png
│   ├── actual_vs_predicted.png        # Publication-grade actual vs predicted plot
│   ├── evaluation_metrics.json        # Formal metrics (MAE, RMSE, R2, MAPE)
│   ├── test_predictions.csv           # Holdout test set actuals vs predictions
│   ├── monthly_retail_sales.csv       # Preprocessed monthly sales
│   └── featured_monthly_retail_sales.csv # Final feature-engineered dataset
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py          # Data ingestion, cleaning, aggregation
│   ├── feature_engineering.py         # Lags, seasonality, calendar, macro features
│   ├── eda.py                         # 10 publication-quality visualizations
│   ├── train_model.py                 # Pipeline training & artifact serialization
│   ├── evaluate_model.py              # Performance evaluation & error analysis
│   └── prediction.py                  # Single-point & recursive multi-horizon engine
│
├── dashboard/
│   └── app.py                         # RetailVision Streamlit Web Application
│
├── utils/
│   ├── __init__.py
│   └── helpers.py                     # Dataset detection, formatting, calendar utilities
│
├── requirements.txt                   # Pinned dependency requirements
├── README.md                          # Comprehensive project documentation
└── main.py                            # End-to-end execution pipeline
```

---

## 15. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.
- Git (optional, for version tracking).

### Installation Steps
1. Navigate to the project root directory:
   ```bash
   cd D:\Predicting-Retail-Sales
   ```
2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux / macOS:
   source venv/bin/activate
   ```
3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## 16. How to Run the Project

### A. Run the End-to-End Pipeline
Execute the full preprocessing, feature engineering, EDA generation, model training, evaluation, and prediction verification pipeline via `main.py`:
```bash
python main.py
```

### B. Launch the Interactive Streamlit Dashboard
Launch the web interface locally:
```bash
streamlit run dashboard/app.py
```
Once launched, open your web browser at:
`http://localhost:8501`

---

## 17. Future Improvements
- **Ensemble Hybrid Models**: Combine Linear Regression baseline with Ridge/Lasso regularization, LightGBM, and Prophet to capture non-linear calendar interactions.
- **Store Clustering Refinements**: Incorporate geospatial distances and store square footage into cluster embeddings.
- **Automated Hyperparameter Tuning**: Integrate Optuna for automated lag window and feature selection.
- **RESTful API Service**: Expose the `FutureSalesPredictor` engine via FastAPI for containerized microservice deployment.

