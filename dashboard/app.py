"""
RetailVision – Intelligent Sales Forecasting Dashboard
A high-performance, modern SaaS retail analytics and predictive forecasting application.
"""

import os
import sys
import json
import calendar
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.helpers import format_currency, format_number, get_season
from src.prediction import FutureSalesPredictor

# Page Configuration
st.set_page_config(
    page_title="RetailVision | Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SaaS Styling
CUSTOM_CSS = """
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Header */
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 24px 32px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .main-header h1 {
        color: #FFFFFF;
        font-size: 26px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94A3B8;
        font-size: 14px;
        margin: 6px 0 0 0;
    }

    /* KPI Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .kpi-title {
        color: #64748B;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .kpi-value {
        color: #0F172A;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .kpi-subtext {
        color: #10B981;
        font-size: 12px;
        font-weight: 500;
    }
    .kpi-subtext.neutral {
        color: #64748B;
    }

    /* Section Cards */
    .section-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .section-title {
        color: #0F172A;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid #F1F5F9;
    }

    /* Prediction Result Box */
    .prediction-box {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 1px solid #86EFAC;
        border-radius: 12px;
        padding: 24px;
        margin-top: 16px;
        margin-bottom: 24px;
    }
    .prediction-val {
        color: #166534;
        font-size: 36px;
        font-weight: 800;
        margin: 8px 0;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-primary { background-color: #DBEAFE; color: #1E40AF; }
    .badge-success { background-color: #D1FAE5; color: #065F46; }
    .badge-warning { background-color: #FEF3C7; color: #92400E; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_project_data():
    """Cached loader for featured dataset and evaluation metrics."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    featured_csv = os.path.join(base_dir, "outputs", "featured_monthly_retail_sales.csv")
    metrics_json = os.path.join(base_dir, "outputs", "evaluation_metrics.json")
    test_preds_csv = os.path.join(base_dir, "outputs", "test_predictions.csv")

    if not os.path.exists(featured_csv):
        from src.feature_engineering import run_feature_engineering
        df = run_feature_engineering()
    else:
        df = pd.read_csv(featured_csv)

    metrics = {}
    if os.path.exists(metrics_json):
        with open(metrics_json, "r") as f:
            metrics = json.load(f)

    test_preds = None
    if os.path.exists(test_preds_csv):
        test_preds = pd.read_csv(test_preds_csv)

    return df, metrics, test_preds


@st.cache_resource
def get_predictor():
    """Cached predictor instance with model in memory."""
    return FutureSalesPredictor()


# Load artifacts
df, metrics, test_preds = load_project_data()
predictor = get_predictor()

# Sidebar Navigation
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135679.png",
        width=50
    )
    st.title("RetailVision")
    st.caption("Intelligent Sales Forecasting")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Overview Dashboard",
            "🔍 Sales Analytics",
            "🤖 Machine Learning Model",
            "🔮 Future Sales Prediction",
            "📁 Dataset Explorer"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### 📌 Model Status")
    st.markdown("**Algorithm:** `Linear Regression`")
    st.markdown(f"**Test MAE:** `{format_currency(metrics.get('MAE', 1829.07))}`")
    st.markdown(f"**Test R²:** `{metrics.get('R2_Score', 0.9833):.4f}`")
    st.caption("Ecuador Store Sales Forecasting")


# ==============================================================================
# A. OVERVIEW DASHBOARD
# ==============================================================================
if page == "📊 Overview Dashboard":
    st.markdown(
        """
        <div class="main-header">
            <h1>RetailVision – Intelligent Sales Forecasting Dashboard</h1>
            <p>Executive Overview & Macro Retail Performance Indicators</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Top KPI Metrics Calculation
    total_sales = df["sales"].sum()
    monthly_totals = df.groupby(["year", "month"])["sales"].sum()
    avg_monthly_sales = monthly_totals.mean()
    max_month_idx = monthly_totals.idxmax()
    max_month_sales = monthly_totals.max()
    max_month_str = f"{calendar.month_abbr[max_month_idx[1]]} {max_month_idx[0]}"
    total_stores = df["store_nbr"].nunique()
    total_families = df["family"].nunique()
    model_mae = metrics.get("MAE", 1829.07)

    # KPI Cards Row
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    with kpi1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Sales</div>
                <div class="kpi-value">${total_sales/1e6:,.1f}M</div>
                <div class="kpi-subtext">Historical Cumulative</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Avg Monthly Sales</div>
                <div class="kpi-value">${avg_monthly_sales/1e6:,.2f}M</div>
                <div class="kpi-subtext neutral">Per Month Chain-wide</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Peak Sales Month</div>
                <div class="kpi-value">{max_month_str}</div>
                <div class="kpi-subtext">${max_month_sales/1e6:,.1f}M (Record)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Stores</div>
                <div class="kpi-value">{total_stores}</div>
                <div class="kpi-subtext neutral">Across 16 Cities</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Product Families</div>
                <div class="kpi-value">{total_families}</div>
                <div class="kpi-subtext neutral">Categories Tracked</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi6:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Model MAE</div>
                <div class="kpi-value">${model_mae:,.0f}</div>
                <div class="kpi-subtext">Linear Regression</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1 Charts: Monthly Trend & YoY Comparison
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("### 📈 Monthly Sales Trend")
        trend_df = df.groupby(["year", "month"])["sales"].sum().reset_index()
        trend_df["Period"] = trend_df.apply(lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1)
        fig_trend = px.line(
            trend_df,
            x="Period",
            y="sales",
            markers=True,
            title="Monthly Total Retail Sales (2013 - 2017)",
            labels={"sales": "Sales ($)", "Period": "Month"},
            template="plotly_white",
            color_discrete_sequence=["#2563EB"]
        )
        fig_trend.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_chart2:
        st.markdown("### 📅 Yearly Sales Comparison")
        yoy_df = df.groupby(["year", "month"])["sales"].sum().reset_index()
        yoy_df["Month_Name"] = yoy_df["month"].apply(lambda m: calendar.month_abbr[m])
        fig_yoy = px.line(
            yoy_df,
            x="Month_Name",
            y="sales",
            color="year",
            markers=True,
            title="Year-over-Year Seasonal Curves",
            labels={"sales": "Sales ($)", "Month_Name": "Month", "year": "Year"},
            template="plotly_white",
            category_orders={"Month_Name": [calendar.month_abbr[i] for i in range(1, 13)]}
        )
        fig_yoy.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_yoy, use_container_width=True)

    # Row 2 Charts: Sales by Season & Holiday vs Normal
    col_chart3, col_chart4 = st.columns(2)

    with col_chart3:
        st.markdown("### 🍂 Sales by Season")
        season_df = df.groupby("season")["sales"].mean().reindex(["Spring", "Summer", "Fall", "Winter"]).reset_index()
        fig_season = px.bar(
            season_df,
            x="season",
            y="sales",
            color="season",
            title="Average Monthly Sales by Season",
            labels={"sales": "Average Sales ($)", "season": "Season"},
            template="plotly_white",
            color_discrete_map={
                "Spring": "#10B981", "Summer": "#F59E0B", "Fall": "#8B5CF6", "Winter": "#3B82F6"
            }
        )
        fig_season.update_layout(height=360, showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_season, use_container_width=True)

    with col_chart4:
        st.markdown("### 🏖️ Holiday vs Normal Sales")
        hol_df = df.groupby("holiday_indicator")["sales"].mean().reset_index()
        hol_df["Period Type"] = hol_df["holiday_indicator"].map({0: "Normal Month", 1: "Holiday Month"})
        fig_hol = px.bar(
            hol_df,
            x="Period Type",
            y="sales",
            color="Period Type",
            title="Average Sales: Holiday vs Non-Holiday Months",
            labels={"sales": "Average Sales ($)"},
            template="plotly_white",
            color_discrete_map={"Normal Month": "#60A5FA", "Holiday Month": "#F87171"}
        )
        fig_hol.update_layout(height=360, showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_hol, use_container_width=True)


# ==============================================================================
# B. SALES ANALYTICS
# ==============================================================================
elif page == "🔍 Sales Analytics":
    st.markdown(
        """
        <div class="main-header">
            <h1>Retail Sales Analytics & Deep-Dive</h1>
            <p>Slice, filter, and inspect performance across stores, product categories, and timeframes</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Interactive Filters Card
    with st.expander("🎛️ Interactive Dimension Filters", expanded=True):
        f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)

        with f_col1:
            years = ["All"] + sorted(list(df["year"].unique()))
            selected_year = st.selectbox("Select Year", years, index=0)

        with f_col2:
            months = ["All"] + list(range(1, 13))
            selected_month = st.selectbox("Select Month", months, index=0)

        with f_col3:
            stores = ["All"] + sorted(list(df["store_nbr"].unique()))
            selected_store = st.selectbox("Select Store", stores, index=0)

        with f_col4:
            families = ["All"] + sorted(list(df["family"].unique()))
            selected_family = st.selectbox("Select Product Family", families, index=0)

        with f_col5:
            seasons = ["All", "Winter", "Spring", "Summer", "Fall"]
            selected_season = st.selectbox("Select Season", seasons, index=0)

    # Filter Logic
    filtered_df = df.copy()
    if selected_year != "All":
        filtered_df = filtered_df[filtered_df["year"] == int(selected_year)]
    if selected_month != "All":
        filtered_df = filtered_df[filtered_df["month"] == int(selected_month)]
    if selected_store != "All":
        filtered_df = filtered_df[filtered_df["store_nbr"] == int(selected_store)]
    if selected_family != "All":
        filtered_df = filtered_df[filtered_df["family"] == selected_family]
    if selected_season != "All":
        filtered_df = filtered_df[filtered_df["season"] == selected_season]

    # Metrics Summary for Filtered Data
    filt_sales = filtered_df["sales"].sum()
    filt_count = len(filtered_df)
    filt_avg = filtered_df["sales"].mean() if filt_count > 0 else 0

    st.markdown(f"**Showing {filt_count:,} records** | Total Filtered Sales: **{format_currency(filt_sales)}** | Average: **{format_currency(filt_avg)}**")

    # Analytics Visualizations
    tab1, tab2, tab3 = st.tabs(["🏬 Store & Family Performance", "📈 Time Dynamics & Trends", "🏆 Rankings"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Sales by Store")
            s_sales = filtered_df.groupby("store_nbr")["sales"].sum().reset_index()
            fig_store = px.bar(
                s_sales,
                x="store_nbr",
                y="sales",
                title="Total Sales by Store Number",
                labels={"store_nbr": "Store #", "sales": "Sales ($)"},
                template="plotly_white",
                color_discrete_sequence=["#0D9488"]
            )
            fig_store.update_layout(height=400)
            st.plotly_chart(fig_store, use_container_width=True)

        with c2:
            st.markdown("#### Sales by Product Family")
            fam_sales = filtered_df.groupby("family")["sales"].sum().sort_values(ascending=False).head(15).reset_index()
            fig_fam = px.bar(
                fam_sales,
                x="sales",
                y="family",
                orientation="h",
                title="Top 15 Product Families by Sales",
                labels={"family": "Category", "sales": "Sales ($)"},
                template="plotly_white",
                color_discrete_sequence=["#4F46E5"]
            )
            fig_fam.update_layout(height=400, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_fam, use_container_width=True)

    with tab2:
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### Monthly Sales Comparison")
            m_comp = filtered_df.groupby(["year", "month"])["sales"].sum().reset_index()
            m_comp["Period"] = m_comp.apply(lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1)
            fig_m = px.area(
                m_comp,
                x="Period",
                y="sales",
                title="Filtered Monthly Sales Evolution",
                labels={"Period": "Month", "sales": "Sales ($)"},
                template="plotly_white",
                color_discrete_sequence=["#3B82F6"]
            )
            fig_m.update_layout(height=400)
            st.plotly_chart(fig_m, use_container_width=True)

        with c4:
            st.markdown("#### Seasonal Sales Analysis")
            s_comp = filtered_df.groupby("season")["sales"].sum().reset_index()
            fig_s = px.pie(
                s_comp,
                names="season",
                values="sales",
                title="Sales Share by Season",
                template="plotly_white",
                hole=0.4
            )
            fig_s.update_layout(height=400)
            st.plotly_chart(fig_s, use_container_width=True)

    with tab3:
        c5, c6 = st.columns(2)
        with c5:
            st.markdown("#### 🥇 Top Performing Stores")
            top_s = filtered_df.groupby(["store_nbr", "type", "city"])["sales"].sum().reset_index()
            top_s = top_s.sort_values("sales", ascending=False).head(10)
            top_s["Formatted Sales"] = top_s["sales"].apply(format_currency)
            st.dataframe(
                top_s[["store_nbr", "city", "type", "Formatted Sales"]].rename(
                    columns={"store_nbr": "Store #", "city": "City", "type": "Type"}
                ),
                use_container_width=True,
                hide_index=True
            )

        with c6:
            st.markdown("#### 🥇 Top Performing Product Families")
            top_f = filtered_df.groupby("family")["sales"].sum().reset_index()
            top_f = top_f.sort_values("sales", ascending=False).head(10)
            top_f["Formatted Sales"] = top_f["sales"].apply(format_currency)
            st.dataframe(
                top_f[["family", "Formatted Sales"]].rename(columns={"family": "Product Family"}),
                use_container_width=True,
                hide_index=True
            )


# ==============================================================================
# C. MACHINE LEARNING MODEL
# ==============================================================================
elif page == "🤖 Machine Learning Model":
    st.markdown(
        """
        <div class="main-header">
            <h1>Machine Learning Model Architecture & Evaluation</h1>
            <p>Scikit-learn Linear Regression model with ColumnTransformer & Pipeline</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Performance Metrics Cards
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Mean Absolute Error (MAE)</div>
                <div class="kpi-value">${metrics.get('MAE', 1829.07):,.2f}</div>
                <div class="kpi-subtext">Primary Evaluation Metric</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m_col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">R-Squared Score (R²)</div>
                <div class="kpi-value">{metrics.get('R2_Score', 0.9833):.4f}</div>
                <div class="kpi-subtext">High Variance Explained</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m_col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Root Mean Squared Error</div>
                <div class="kpi-value">${metrics.get('RMSE', 5046.73):,.2f}</div>
                <div class="kpi-subtext neutral">Penalizes Extreme Outliers</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m_col4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">MAPE (Sales ≥ $100)</div>
                <div class="kpi-value">{metrics.get('MAPE_Significant_Percent', 18.2):.2f}%</div>
                <div class="kpi-subtext">Percentage Error on Non-trivial Sales</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Actual vs Predicted Sales Interactive Chart
    st.markdown("### 🎯 Actual vs Predicted Retail Sales (Holdout Test Period 2017)")
    if test_preds is not None:
        monthly_eval = test_preds.groupby(["year", "month"])[["sales", "predicted_sales"]].sum().reset_index()
        monthly_eval["Period"] = monthly_eval.apply(lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1)

        fig_act_pred = go.Figure()
        fig_act_pred.add_trace(go.Scatter(
            x=monthly_eval["Period"],
            y=monthly_eval["sales"],
            mode="lines+markers",
            name="Actual Retail Sales",
            line=dict(color="#1E40AF", width=3),
            marker=dict(size=8)
        ))
        fig_act_pred.add_trace(go.Scatter(
            x=monthly_eval["Period"],
            y=monthly_eval["predicted_sales"],
            mode="lines+markers",
            name="Linear Regression Prediction",
            line=dict(color="#DC2626", width=3, dash="dash"),
            marker=dict(size=8, symbol="square")
        ))
        fig_act_pred.update_layout(
            title="Macro Chain-wide Sales: Actual vs Predicted (2017 Holdout)",
            xaxis_title="Month",
            yaxis_title="Total Monthly Sales ($)",
            template="plotly_white",
            height=420,
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02)
        )
        st.plotly_chart(fig_act_pred, use_container_width=True)

    # Technical Architecture Details
    st.markdown("### ⚙️ Pipeline Specifications")
    spec_col1, spec_col2 = st.columns(2)

    with spec_col1:
        st.markdown(
            """
            **Training & Preprocessing Architecture:**
            - **Algorithm:** Ordinary Least Squares (OLS) Linear Regression
            - **Categorical Encoder:** `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
            - **Numerical Scaler:** `StandardScaler()`
            - **Validation Strategy:** Chronological holdout split (Train: 2013-04 to 2016-12; Test: 2017-01 to 2017-08)
            - **Data Leakage Safeguards:** Preprocessing fitted strictly on training data; all lag & rolling window features computed strictly from past intervals.
            """
        )

    with spec_col2:
        st.markdown(
            f"""
            **Dataset Dimensions:**
            - **Total Training Records:** `80,190`
            - **Total Testing Records:** `14,256`
            - **Target Variable:** `sales` (Monthly Sales per Store-Family)
            - **Engineered Features:** 21 total input variables
            - **Lag Variables:** `lag_1_sales`, `lag_2_sales`, `lag_3_sales`, `rolling_mean_3_sales`, `lag_12_sales`
            """
        )


# ==============================================================================
# D. FUTURE SALES PREDICTION
# ==============================================================================
elif page == "🔮 Future Sales Prediction":
    st.markdown(
        """
        <div class="main-header">
            <h1>Future Sales Prediction & Forecasting Engine</h1>
            <p>Generate single-month estimates and multi-quarter (3, 6, 12 months) forward-looking forecasts</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    p_col1, p_col2 = st.columns([1, 2])

    with p_col1:
        st.markdown("### 🎛️ Scenario Parameters")
        store_list = sorted(list(df["store_nbr"].unique()))
        family_list = sorted(list(df["family"].unique()))

        p_store = st.selectbox("Select Store", store_list, index=0)
        p_family = st.selectbox("Select Product Family", family_list, index=family_list.index("GROCERY I") if "GROCERY I" in family_list else 0)

        p_year = st.selectbox("Select Forecast Year", [2017, 2018, 2019], index=0)
        p_month = st.selectbox("Select Month", list(range(1, 13)), index=8)  # Default Sept
        p_season = st.selectbox("Select Season", ["Auto-detect", "Winter", "Spring", "Summer", "Fall"], index=0)
        p_holiday = st.radio("Holiday Month?", ["Non-Holiday (0)", "Holiday (1)"], index=0)
        holiday_val = 1 if "Holiday (1)" in p_holiday else 0

        forecast_horizons_choice = st.multiselect(
            "Multi-month Forecast Horizons",
            [3, 6, 12],
            default=[3, 6, 12]
        )

        predict_btn = st.button("🚀 Predict Sales", type="primary", use_container_width=True)

    with p_col2:
        if predict_btn or "last_prediction" in st.session_state:
            season_param = None if p_season == "Auto-detect" else p_season
            
            # Run prediction
            pred_result = predictor.predict_single(
                store_nbr=p_store,
                family=p_family,
                year=p_year,
                month=p_month,
                season=season_param,
                holiday_indicator=holiday_val
            )
            st.session_state["last_prediction"] = pred_result

            # Display Prediction Card
            st.markdown(
                f"""
                <div class="prediction-box">
                    <span class="badge badge-success">Forecast Generated</span>
                    <div style="color: #166534; font-size: 14px; margin-top: 8px;">Predicted Sales for Store #{p_store} – {p_family} ({pred_result['period']})</div>
                    <div class="prediction-val">{pred_result['predicted_sales_formatted']}</div>
                    <div style="color: #15803D; font-size: 14px;">
                        Historical Average: <b>{pred_result['historical_average_formatted']}</b> 
                        ({pred_result['percentage_difference_vs_avg']:+.1f}% vs historical benchmark)
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Assumptions Expander
            with st.expander("ℹ️ Forecast Assumptions & Features Used", expanded=False):
                for asmp in pred_result["assumptions"]:
                    st.markdown(f"- {asmp}")

            # Multi-Horizon Forecasts (3, 6, 12 months)
            if forecast_horizons_choice:
                st.markdown("### 📅 Multi-Period Forward Forecast")
                horizons_data = predictor.forecast_horizons(
                    store_nbr=p_store,
                    family=p_family,
                    start_year=p_year,
                    start_month=p_month,
                    horizons=forecast_horizons_choice
                )

                h_cols = st.columns(len(forecast_horizons_choice))
                for idx, h in enumerate(sorted(forecast_horizons_choice)):
                    h_key = f"{h}_month"
                    h_summary = horizons_data["horizon_summaries"].get(h_key)
                    if h_summary:
                        with h_cols[idx]:
                            st.markdown(
                                f"""
                                <div class="kpi-card">
                                    <div class="kpi-title">{h}-Month Cumulative</div>
                                    <div class="kpi-value" style="font-size: 20px;">{h_summary['total_predicted_sales_formatted']}</div>
                                    <div class="kpi-subtext neutral">Avg: {h_summary['average_monthly_sales_formatted']}/mo</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                # Forecast Trend Chart
                f_table = horizons_data["forecast_table"]
                fig_forecast = px.line(
                    f_table,
                    x="period",
                    y="predicted_sales",
                    markers=True,
                    title=f"Recursive Forward Forecast ({max(forecast_horizons_choice)} Months)",
                    labels={"period": "Month", "predicted_sales": "Predicted Sales ($)"},
                    template="plotly_white",
                    color_discrete_sequence=["#10B981"]
                )
                fig_forecast.update_layout(height=340)
                st.plotly_chart(fig_forecast, use_container_width=True)

        else:
            st.info("👈 Select scenario parameters in the sidebar and click **Predict Sales** to generate retail demand forecasts.")


# ==============================================================================
# E. DATASET EXPLORER
# ==============================================================================
elif page == "📁 Dataset Explorer":
    st.markdown(
        """
        <div class="main-header">
            <h1>Dataset Explorer & Raw Data Inspector</h1>
            <p>Inspect preprocessed records, null distributions, schema definitions, and export filtered subsets</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Metadata Metrics
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        st.metric("Total Rows", f"{len(df):,}")
    with col_d2:
        st.metric("Total Columns", f"{df.shape[1]}")
    with col_d3:
        st.metric("Date Span", f"{df['year'].min()} - {df['year'].max()}")
    with col_d4:
        st.metric("Missing Values", f"{df.isnull().sum().sum()}")

    st.markdown("### 📋 Interactive Data Table")
    # Search / Filter
    f_store_nbr = st.selectbox("Filter Store", ["All"] + sorted(list(df["store_nbr"].unique())), index=0)
    f_fam = st.selectbox("Filter Family", ["All"] + sorted(list(df["family"].unique())), index=0)

    display_df = df.copy()
    if f_store_nbr != "All":
        display_df = display_df[display_df["store_nbr"] == f_store_nbr]
    if f_fam != "All":
        display_df = display_df[display_df["family"] == f_fam]

    st.dataframe(
        display_df.head(250),
        use_container_width=True,
        height=380
    )

    # CSV Download Button
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="retail_sales_data_export.csv",
        mime="text/csv",
        type="primary"
    )

    # Schema & Missing Value Distribution
    st.markdown("### 🔍 Column Schema & Types")
    schema_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": [str(t) for t in df.dtypes],
        "Null Count": [int(df[col].isnull().sum()) for col in df.columns],
        "Unique Values": [int(df[col].nunique()) for col in df.columns]
    })
    st.dataframe(schema_df, use_container_width=True, hide_index=True)

