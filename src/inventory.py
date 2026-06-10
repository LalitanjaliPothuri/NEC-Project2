from __future__ import annotations

from statistics import NormalDist
from typing import Dict

import numpy as np
import pandas as pd


def build_inventory_report(
    product_summary: pd.DataFrame,
    forecast_frame: pd.DataFrame,
    service_level: float = 0.95,
    lead_time_days: int = 30,
) -> pd.DataFrame:
    """Create reorder, stockout, and overstock recommendations at the product level."""
    if product_summary.empty:
        return product_summary.copy()

    report = product_summary.copy().reset_index(drop=True)
    next_month_demand = float(forecast_frame["predicted_units"].iloc[0]) if not forecast_frame.empty else float(report["demand_units"].sum())
    total_units = float(report["demand_units"].sum()) if float(report["demand_units"].sum()) != 0 else 1.0

    z_score = NormalDist().inv_cdf(service_level)
    today = pd.Timestamp.today().normalize()

    report["demand_share"] = report["demand_units"] / total_units
    report["expected_next_month_units"] = next_month_demand * report["demand_share"]
    report["daily_avg_demand"] = report["demand_avg"].replace(0, np.nan) / 30.44
    report["daily_demand_std"] = report["demand_std"].fillna(0) / np.sqrt(30.44)
    report["safety_stock"] = z_score * report["daily_demand_std"] * np.sqrt(float(lead_time_days))
    report["reorder_point"] = report["daily_avg_demand"].fillna(0) * lead_time_days + report["safety_stock"]
    report["target_stock"] = report["expected_next_month_units"] + report["reorder_point"]
    report["recommended_order_qty"] = (report["target_stock"] - report["current_stock"]).clip(lower=0)
    report["days_until_stockout"] = np.where(
        report["daily_avg_demand"].fillna(0) > 0,
        report["current_stock"].fillna(0) / report["daily_avg_demand"],
        np.nan,
    )
    report["predicted_stockout_date"] = today + pd.to_timedelta(report["days_until_stockout"], unit="D")
    report["restock_by"] = report["predicted_stockout_date"] - pd.to_timedelta(lead_time_days, unit="D")
    report["overstock_units"] = (report["current_stock"].fillna(0) - report["target_stock"]).clip(lower=0)

    conditions = [
        report["days_until_stockout"] <= lead_time_days,
        report["overstock_units"] > 0,
        report["days_until_stockout"] <= lead_time_days * 2,
    ]
    choices = ["Critical", "Overstock", "Reorder Soon"]
    report["status"] = np.select(conditions, choices, default="Healthy")
    report["stock_coverage_days"] = report["days_until_stockout"]
    report["needs_restock"] = np.where(report["recommended_order_qty"] > 0, "Yes", "No")

    columns = [
        "product",
        "demand_units",
        "revenue",
        "profit",
        "current_stock",
        "expected_next_month_units",
        "daily_avg_demand",
        "days_until_stockout",
        "predicted_stockout_date",
        "restock_by",
        "reorder_point",
        "target_stock",
        "recommended_order_qty",
        "overstock_units",
        "status",
        "needs_restock",
    ]
    available = [column for column in columns if column in report.columns]
    return report[available + [column for column in report.columns if column not in available]]


def stock_alert_summary(inventory_report: pd.DataFrame) -> Dict[str, int]:
    if inventory_report.empty:
        return {"critical": 0, "reorder": 0, "overstock": 0, "healthy": 0}

    status = inventory_report["status"].fillna("Healthy")
    return {
        "critical": int((status == "Critical").sum()),
        "reorder": int((status == "Reorder Soon").sum()),
        "overstock": int((status == "Overstock").sum()),
        "healthy": int((status == "Healthy").sum()),
    }
