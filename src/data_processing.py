from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import COLUMN_ALIASES, DEFAULT_COST_RATIO


def read_tabular_file(file_obj) -> pd.DataFrame:
    """Read CSV or Excel content from a Streamlit upload object."""
    file_name = getattr(file_obj, "name", "") or ""
    suffix = Path(file_name).suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_obj)
    return pd.read_csv(file_obj)


def save_dataframe(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def infer_column_map(df: pd.DataFrame) -> Dict[str, str]:
    """Best-effort mapping from raw columns to the app's standardized fields."""
    columns = list(df.columns)
    normalized = {column.lower().strip(): column for column in columns}
    mapping: Dict[str, str] = {}

    for role, aliases in COLUMN_ALIASES.items():
        selected = None
        for alias in aliases:
            if alias in normalized:
                selected = normalized[alias]
                break
        if selected is None:
            for column in columns:
                column_norm = column.lower().replace(" ", "_")
                if any(alias in column_norm for alias in aliases):
                    selected = column
                    break
        if selected is not None:
            mapping[role] = selected

    return mapping


def is_date_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False

    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False

    parsed = pd.to_datetime(sample, errors="coerce")
    return float(parsed.notna().mean()) >= 0.6


def is_numeric_like(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True

    sample = pd.to_numeric(series.dropna().astype(str).head(50), errors="coerce")
    if sample.empty:
        return False
    return float(sample.notna().mean()) >= 0.8


def is_categorical_like(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return False
    return True


def generate_sample_data(seed: int = 42) -> pd.DataFrame:
    """Create a realistic retail dataset for demo use when no file is uploaded."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=720, freq="D")
    products = [
        "Smart Speaker",
        "Wireless Headphones",
        "Office Chair",
        "Gaming Mouse",
        "LED Monitor",
        "Laptop Stand",
        "Mechanical Keyboard",
        "Webcam",
        "Lipstick",
        "Foundation",
        "Moisturizer",
        "Perfume",
        "Nail Polish",
        "Face Wash",
        "Ladies Dress",
        "Ladies Skirt",
        "Ladies Jeans",
        "Ladies Handbag",
        "Ladies Sandals",
        "Ladies Blouse",
    ]
    regions = ["North", "South", "East", "West"]
    base_quantity = {
        "Smart Speaker": 18,
        "Wireless Headphones": 24,
        "Office Chair": 12,
        "Gaming Mouse": 28,
        "LED Monitor": 15,
        "Laptop Stand": 22,
        "Mechanical Keyboard": 20,
        "Webcam": 16,
        "Lipstick": 20,
        "Foundation": 16,
        "Moisturizer": 14,
        "Perfume": 8,
        "Nail Polish": 25,
        "Face Wash": 18,
        "Ladies Dress": 6,
        "Ladies Skirt": 7,
        "Ladies Jeans": 6,
        "Ladies Handbag": 5,
        "Ladies Sandals": 10,
        "Ladies Blouse": 12,
    }
    base_price = {
        "Smart Speaker": 74,
        "Wireless Headphones": 89,
        "Office Chair": 155,
        "Gaming Mouse": 42,
        "LED Monitor": 198,
        "Laptop Stand": 36,
        "Mechanical Keyboard": 96,
        "Webcam": 58,
        "Lipstick": 12.5,
        "Foundation": 18.0,
        "Moisturizer": 22.0,
        "Perfume": 45.0,
        "Nail Polish": 6.5,
        "Face Wash": 9.75,
        "Ladies Dress": 54.9,
        "Ladies Skirt": 34.2,
        "Ladies Jeans": 49.99,
        "Ladies Handbag": 79.99,
        "Ladies Sandals": 29.5,
        "Ladies Blouse": 24.75,
    }

    rows = []
    for day_index, date in enumerate(dates):
        seasonal_factor = 1.0 + 0.18 * np.sin(2 * np.pi * date.dayofyear / 365.0) + 0.08 * np.sin(2 * np.pi * date.dayofweek / 7.0)
        trend_factor = 1.0 + (day_index / max(len(dates), 1)) * 0.12
        for product in products:
            expected_units = base_quantity[product] * seasonal_factor * trend_factor
            units_sold = max(int(rng.normal(expected_units, max(2.0, expected_units * 0.18))), 0)
            unit_price = round(base_price[product] * (1.0 + rng.normal(0, 0.035)), 2)
            unit_cost = round(unit_price * (0.58 + rng.normal(0, 0.03)), 2)
            current_stock = max(int(expected_units * 3.0 + rng.normal(0, expected_units * 1.4)), 0)
            rows.append(
                {
                    "date": date,
                    "product": product,
                    "quantity": units_sold,
                    "unit_price": unit_price,
                    "unit_cost": unit_cost,
                    "stock": current_stock,
                    "region": rng.choice(regions),
                    "channel": rng.choice(["Online", "Retail", "Wholesale"]),
                }
            )

    return pd.DataFrame(rows)


def prepare_sales_dataframe(
    df: pd.DataFrame,
    column_map: Dict[str, str],
    target_column: Optional[str],
    revenue_column: Optional[str],
    price_column: Optional[str],
    cost_column: Optional[str],
    stock_column: Optional[str],
    default_cost_ratio: float = DEFAULT_COST_RATIO,
) -> pd.DataFrame:
    """Standardize the uploaded data into a single analytics-friendly table."""
    prepared = df.copy()

    rename_map: Dict[str, str] = {}
    if column_map.get("date"):
        rename_map[column_map["date"]] = "date"
    if column_map.get("product"):
        rename_map[column_map["product"]] = "product"
    if target_column:
        rename_map[target_column] = "demand_units"
    if revenue_column:
        rename_map[revenue_column] = "revenue_source"
    if price_column:
        rename_map[price_column] = "unit_price"
    if cost_column:
        rename_map[cost_column] = "unit_cost"
    if stock_column:
        rename_map[stock_column] = "stock_on_hand"

    prepared = prepared.rename(columns=rename_map)

    if "date" not in prepared.columns:
        raise ValueError("A date column is required to run the sales forecast.")

    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared = prepared.dropna(subset=["date"]).copy()
    prepared = prepared.sort_values("date").reset_index(drop=True)

    if "product" not in prepared.columns:
        prepared["product"] = "All Products"
    prepared["product"] = prepared["product"].fillna("All Products").astype(str)

    for column in ["demand_units", "revenue_source", "unit_price", "unit_cost", "stock_on_hand"]:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    if "demand_units" not in prepared.columns:
        prepared["demand_units"] = np.nan
    prepared["demand_units"] = prepared["demand_units"].fillna(0).clip(lower=0)

    if "unit_price" not in prepared.columns:
        prepared["unit_price"] = np.nan
    if "unit_cost" not in prepared.columns:
        prepared["unit_cost"] = np.nan
    if "stock_on_hand" not in prepared.columns:
        prepared["stock_on_hand"] = np.nan

    revenue_source = prepared["revenue_source"] if "revenue_source" in prepared.columns else pd.Series(np.nan, index=prepared.index)
    revenue_from_price = prepared["demand_units"] * prepared["unit_price"]
    prepared["revenue"] = revenue_source.fillna(revenue_from_price)
    prepared["revenue"] = prepared["revenue"].fillna(prepared["demand_units"])

    if prepared["unit_cost"].isna().all():
        if prepared["unit_price"].notna().any():
            prepared["unit_cost"] = prepared["unit_price"].fillna(prepared["revenue"].where(prepared["demand_units"] > 0, np.nan) / prepared["demand_units"].replace(0, np.nan)) * default_cost_ratio
        else:
            prepared["unit_cost"] = prepared["revenue"] * default_cost_ratio / prepared["demand_units"].replace(0, np.nan)
    else:
        fallback_cost = prepared["unit_price"].fillna(prepared["revenue"].where(prepared["demand_units"] > 0, np.nan) / prepared["demand_units"].replace(0, np.nan)) * default_cost_ratio
        prepared["unit_cost"] = prepared["unit_cost"].fillna(fallback_cost)

    prepared["unit_cost"] = prepared["unit_cost"].fillna(0)
    prepared["total_cost"] = prepared["demand_units"] * prepared["unit_cost"]
    prepared["profit"] = prepared["revenue"] - prepared["total_cost"]
    prepared["month_start"] = prepared["date"].dt.to_period("M").dt.to_timestamp()
    prepared["year"] = prepared["date"].dt.year
    prepared["month"] = prepared["date"].dt.month
    prepared["month_name"] = prepared["date"].dt.strftime("%b")
    prepared["day_of_week"] = prepared["date"].dt.day_name()
    prepared["week_of_year"] = prepared["date"].dt.isocalendar().week.astype(int)

    return prepared


def build_monthly_summary(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        cleaned_df.groupby("month_start", as_index=False)
        .agg(
            demand_units=("demand_units", "sum"),
            revenue=("revenue", "sum"),
            total_cost=("total_cost", "sum"),
            profit=("profit", "sum"),
            avg_unit_price=("unit_price", "mean"),
            avg_unit_cost=("unit_cost", "mean"),
            active_products=("product", "nunique"),
            ending_stock=("stock_on_hand", "last"),
        )
        .rename(columns={"month_start": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    monthly["loss"] = monthly["profit"].clip(upper=0).abs()
    monthly["profit_margin"] = np.where(monthly["revenue"] != 0, monthly["profit"] / monthly["revenue"], np.nan)
    monthly["month_label"] = monthly["date"].dt.strftime("%b %Y")
    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month

    return monthly


def build_product_summary(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        cleaned_df.groupby("product", as_index=False)
        .agg(
            demand_units=("demand_units", "sum"),
            revenue=("revenue", "sum"),
            total_cost=("total_cost", "sum"),
            profit=("profit", "sum"),
            avg_unit_price=("unit_price", "mean"),
            avg_unit_cost=("unit_cost", "mean"),
            current_stock=("stock_on_hand", "last"),
            stock_peak=("stock_on_hand", "max"),
            last_sale_date=("date", "max"),
            demand_std=("demand_units", "std"),
            demand_avg=("demand_units", "mean"),
        )
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )

    summary["current_stock"] = summary["current_stock"].fillna(0)
    summary["profit_margin"] = np.where(summary["revenue"] != 0, summary["profit"] / summary["revenue"], np.nan)
    summary["days_to_stockout"] = np.where(summary["demand_avg"] > 0, summary["current_stock"] / summary["demand_avg"] * 30.44, np.nan)
    summary["stockout_date_estimate"] = pd.to_datetime(summary["last_sale_date"]) + pd.to_timedelta(summary["days_to_stockout"], unit="D")

    return summary
