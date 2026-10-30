"""
Exploratory Data Analysis (EDA) Module
Generates 10 publication-quality visualizations using Matplotlib and Seaborn.
"""

import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "semibold",
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


def generate_all_eda_visualizations(df: pd.DataFrame, output_dir: str = None) -> list:
    """
    Creates and saves all 10 required EDA visualizations.
    """
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "eda"))
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("GENERATING EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS")
    print("=" * 60)

    generated_charts = []

    # 1. Monthly Sales Trend
    print("[1/10] Monthly Sales Trend...")
    monthly_trend = df.groupby(["year", "month"])["sales"].sum().reset_index()
    monthly_trend["date_str"] = monthly_trend.apply(lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1)

    plt.figure(figsize=(14, 6))
    plt.plot(monthly_trend["date_str"], monthly_trend["sales"] / 1e6, marker="o", color="#1E88E5", linewidth=2.5, markersize=5)
    plt.title("Monthly Total Retail Sales Trend (2013 - 2017)", pad=15)
    plt.xlabel("Year - Month")
    plt.ylabel("Total Sales (Millions USD)")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    trend_path = os.path.join(output_dir, "monthly_sales_trend.png")
    plt.savefig(trend_path)
    plt.close()
    generated_charts.append(trend_path)

    # 2. Yearly Sales Comparison
    print("[2/10] Yearly Sales Comparison...")
    yearly_month = df.groupby(["year", "month"])["sales"].sum().reset_index()
    plt.figure(figsize=(12, 6))
    palette = sns.color_palette("tab10", n_colors=yearly_month["year"].nunique())
    sns.lineplot(
        data=yearly_month,
        x="month",
        y=yearly_month["sales"] / 1e6,
        hue="year",
        palette=palette,
        marker="o",
        linewidth=2.2
    )
    plt.title("Year-over-Year Monthly Sales Comparison", pad=15)
    plt.xlabel("Month of Year")
    plt.ylabel("Total Sales (Millions USD)")
    plt.xticks(range(1, 13), [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ])
    plt.legend(title="Year", frameon=True)
    yearly_path = os.path.join(output_dir, "yearly_sales_comparison.png")
    plt.savefig(yearly_path)
    plt.close()
    generated_charts.append(yearly_path)

    # 3. Holiday vs Non-Holiday Sales
    print("[3/10] Holiday vs Non-Holiday Sales...")
    holiday_summary = df.groupby("holiday_indicator")["sales"].mean().reset_index()
    holiday_summary["Type"] = holiday_summary["holiday_indicator"].map({0: "Non-Holiday Month", 1: "Holiday Month"})

    plt.figure(figsize=(8, 6))
    bars = plt.bar(holiday_summary["Type"], holiday_summary["sales"], color=["#42A5F5", "#EF5350"], width=0.45)
    plt.title("Average Sales: Holiday vs Non-Holiday Months", pad=15)
    plt.xlabel("Period Type")
    plt.ylabel("Average Sales per Store-Family ($)")
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., h + 150, f"${h:,.0f}", ha="center", va="bottom", fontweight="bold")
    holiday_path = os.path.join(output_dir, "holiday_vs_non_holiday.png")
    plt.savefig(holiday_path)
    plt.close()
    generated_charts.append(holiday_path)

    # 4. Seasonal Sales Analysis
    print("[4/10] Seasonal Sales Analysis...")
    season_order = ["Spring", "Summer", "Fall", "Winter"]
    seasonal_data = df.groupby("season")["sales"].agg(["mean", "sum"]).reindex(season_order).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    colors = ["#66BB6A", "#FFA726", "#AB47BC", "#29B6F6"]
    bars = ax1.bar(seasonal_data["season"], seasonal_data["mean"], color=colors, width=0.5, alpha=0.85)
    ax1.set_title("Seasonal Retail Sales Analysis (Average Sales)", pad=15)
    ax1.set_xlabel("Season")
    ax1.set_ylabel("Average Monthly Sales ($)")
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., h + 100, f"${h:,.0f}", ha="center", va="bottom", fontweight="bold")
    season_path = os.path.join(output_dir, "seasonal_sales_analysis.png")
    plt.savefig(season_path)
    plt.close()
    generated_charts.append(season_path)

    # 5. Sales by Product Family
    print("[5/10] Sales by Product Family...")
    family_sales = df.groupby("family")["sales"].sum().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(12, 10))
    sns.barplot(data=family_sales, y="family", x=family_sales["sales"] / 1e6, hue="family", palette="viridis", legend=False, orient="h")
    plt.title("Total Sales by Product Family (2013 - 2017)", pad=15)
    plt.xlabel("Total Sales (Millions USD)")
    plt.ylabel("Product Family")
    family_path = os.path.join(output_dir, "sales_by_product_family.png")
    plt.savefig(family_path)
    plt.close()
    generated_charts.append(family_path)

    # 6. Sales by Store
    print("[6/10] Sales by Store...")
    store_sales = df.groupby("store_nbr")["sales"].sum().reset_index()
    store_sales = store_sales.sort_values("store_nbr")
    plt.figure(figsize=(14, 6))
    plt.bar(store_sales["store_nbr"].astype(str), store_sales["sales"] / 1e6, color="#00897B", width=0.7)
    plt.title("Total Retail Sales Across All 54 Stores", pad=15)
    plt.xlabel("Store Number")
    plt.ylabel("Total Sales (Millions USD)")
    plt.xticks(rotation=90, fontsize=8)
    store_path = os.path.join(output_dir, "sales_by_store.png")
    plt.savefig(store_path)
    plt.close()
    generated_charts.append(store_path)

    # 7. Monthly Sales Distribution
    print("[7/10] Monthly Sales Distribution...")
    plt.figure(figsize=(10, 6))
    # Log transform positive sales for clear visual distribution without extreme tail skew
    positive_sales = df[df["sales"] > 0]["sales"]
    sns.histplot(positive_sales, bins=50, kde=True, color="#3949AB", log_scale=True)
    plt.title("Monthly Retail Sales Distribution (Log Scale)", pad=15)
    plt.xlabel("Monthly Sales ($ - Log Scale)")
    plt.ylabel("Record Count")
    dist_path = os.path.join(output_dir, "monthly_sales_distribution.png")
    plt.savefig(dist_path)
    plt.close()
    generated_charts.append(dist_path)

    # 8. Correlation Heatmap
    print("[8/10] Correlation Heatmap...")
    numeric_cols = [
        "sales", "oil_price", "transaction_count", "holiday_count",
        "lag_1_sales", "lag_2_sales", "lag_3_sales", "rolling_mean_3_sales", "lag_12_sales"
    ]
    corr_cols = [c for c in numeric_cols if c in df.columns]
    corr = df[corr_cols].corr()

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Correlation Heatmap: Sales, Lags & Macro Features", pad=15)
    corr_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(corr_path)
    plt.close()
    generated_charts.append(corr_path)

    # 9. Top 10 Performing Stores
    print("[9/10] Top 10 Performing Stores...")
    top_stores = df.groupby(["store_nbr", "type", "city"])["sales"].sum().reset_index()
    top_stores = top_stores.sort_values("sales", ascending=False).head(10)
    top_stores["label"] = top_stores.apply(lambda r: f"Store #{r['store_nbr']} ({r['city']}, Type {r['type']})", axis=1)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(top_stores["label"][::-1], top_stores["sales"][::-1] / 1e6, color="#5E35B1", height=0.6)
    plt.title("Top 10 Performing Retail Stores by Total Sales", pad=15)
    plt.xlabel("Total Sales (Millions USD)")
    plt.ylabel("Store")
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.5, bar.get_y() + bar.get_height() / 2., f"${w:,.1f}M", ha="left", va="center", fontweight="bold", fontsize=9)
    top_stores_path = os.path.join(output_dir, "top_10_stores.png")
    plt.savefig(top_stores_path)
    plt.close()
    generated_charts.append(top_stores_path)

    # 10. Top 10 Product Families
    print("[10/10] Top 10 Product Families...")
    top_families = df.groupby("family")["sales"].sum().sort_values(ascending=False).head(10).reset_index()

    plt.figure(figsize=(10, 6))
    bars = plt.barh(top_families["family"][::-1], top_families["sales"][::-1] / 1e6, color="#00838F", height=0.6)
    plt.title("Top 10 Best-Selling Product Families", pad=15)
    plt.xlabel("Total Sales (Millions USD)")
    plt.ylabel("Product Family")
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 1.0, bar.get_y() + bar.get_height() / 2., f"${w:,.1f}M", ha="left", va="center", fontweight="bold", fontsize=9)
    top_families_path = os.path.join(output_dir, "top_10_product_families.png")
    plt.savefig(top_families_path)
    plt.close()
    generated_charts.append(top_families_path)

    print(f"\nAll 10 EDA visualizations saved successfully in: {output_dir}")
    return generated_charts


if __name__ == "__main__":
    from src.feature_engineering import run_feature_engineering
    df = run_feature_engineering()
    generate_all_eda_visualizations(df)

