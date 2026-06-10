from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import MODELS_DIR, RANDOM_STATE


def _feature_lags(length: int) -> List[int]:
    return [1, 2, 3, 6, 12] if length >= 12 else [1, 2, 3]


def build_feature_frame(monthly_df: pd.DataFrame, target_col: str = "demand_units") -> Tuple[pd.DataFrame, List[str]]:
    frame = monthly_df[["date", target_col]].copy().sort_values("date").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    frame["quarter"] = frame["date"].dt.quarter
    frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12.0)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12.0)
    frame["trend"] = np.arange(len(frame), dtype=float)

    lags = _feature_lags(len(frame))
    for lag in lags:
        frame[f"lag_{lag}"] = frame[target_col].shift(lag)

    for window in [3, 6]:
        if len(frame) >= window:
            frame[f"rolling_mean_{window}"] = frame[target_col].shift(1).rolling(window).mean()
            frame[f"rolling_std_{window}"] = frame[target_col].shift(1).rolling(window).std().fillna(0)

    feature_cols = [column for column in frame.columns if column not in {"date", target_col}]
    usable = frame.dropna(subset=feature_cols).reset_index(drop=True)
    return usable, feature_cols


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _save_bundle(bundle: Dict, path: Path | None = None) -> Path:
    path = path or (MODELS_DIR / "sales_forecast_model.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_model_bundle(path: Path | None = None) -> Dict | None:
    path = path or (MODELS_DIR / "sales_forecast_model.joblib")
    if path.exists():
        return joblib.load(path)
    return None


def train_sales_forecast_model(
    monthly_df: pd.DataFrame,
    target_col: str = "demand_units",
    forecast_months: int = 12,
    test_months: int = 6,
) -> Dict:
    """Train a sales forecasting model and generate validation plus future predictions."""
    monthly_df = monthly_df.sort_values("date").reset_index(drop=True)

    if monthly_df.empty:
        future_dates = pd.date_range(pd.Timestamp.today().normalize() + pd.offsets.MonthBegin(1), periods=forecast_months, freq="MS")
        forecast = pd.DataFrame(
            {
                "date": future_dates,
                "predicted_units": [0.0] * len(future_dates),
                "predicted_revenue": [0.0] * len(future_dates),
            }
        )
        forecast["month_label"] = forecast["date"].dt.strftime("%b %Y")
        bundle = {
            "model": None,
            "feature_cols": [],
            "target_col": target_col,
            "feature_frame": pd.DataFrame(columns=["date", target_col]),
            "metrics": {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")},
            "validation_frame": pd.DataFrame(columns=["date", "actual", "predicted"]),
            "forecast_frame": forecast,
            "feature_importance": pd.DataFrame(columns=["feature", "importance"]),
            "price_basis": 1.0,
        }
        _save_bundle(bundle)
        return bundle

    feature_df, feature_cols = build_feature_frame(monthly_df, target_col=target_col)

    if len(feature_df) < 6:
        baseline = monthly_df[target_col].tail(3).mean() if len(monthly_df) else 0.0
        last_date = pd.to_datetime(monthly_df["date"].dropna().max(), errors="coerce")
        if pd.isna(last_date):
            last_date = pd.Timestamp.today().normalize()
        future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=forecast_months, freq="MS")
        forecast = pd.DataFrame(
            {
                "date": future_dates,
                "predicted_units": [max(float(baseline), 0.0)] * len(future_dates),
            }
        )
        price_basis = monthly_df.get("avg_unit_price", pd.Series(dtype=float)).dropna().tail(3).mean()
        forecast["predicted_revenue"] = forecast["predicted_units"] * (price_basis if pd.notna(price_basis) else 1.0)
        forecast["month_label"] = forecast["date"].dt.strftime("%b %Y")
        bundle = {
            "model": None,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "feature_frame": feature_df,
            "metrics": {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")},
            "validation_frame": pd.DataFrame(columns=["date", "actual", "predicted"]),
            "forecast_frame": forecast,
            "feature_importance": pd.DataFrame(columns=["feature", "importance"]),
            "price_basis": float(price_basis) if pd.notna(price_basis) else 1.0,
        }
        _save_bundle(bundle)
        return bundle

    test_months = max(1, min(test_months, len(feature_df) - 3))
    train_cutoff = max(len(feature_df) - test_months, 1)

    train_frame = feature_df.iloc[:train_cutoff].copy()
    test_frame = feature_df.iloc[train_cutoff:].copy()

    model = RandomForestRegressor(
        n_estimators=350,
        random_state=RANDOM_STATE,
        min_samples_leaf=2,
    )
    model.fit(train_frame[feature_cols], train_frame[target_col])

    validation_pred = model.predict(test_frame[feature_cols])
    validation_frame = test_frame[["date", target_col]].copy()
    validation_frame = validation_frame.rename(columns={target_col: "actual"})
    validation_frame["predicted"] = validation_pred
    validation_frame["abs_error"] = (validation_frame["actual"] - validation_frame["predicted"]).abs()
    validation_frame["pct_error"] = np.where(validation_frame["actual"] != 0, validation_frame["abs_error"] / validation_frame["actual"].abs() * 100.0, np.nan)

    mae = float(mean_absolute_error(validation_frame["actual"], validation_frame["predicted"]))
    rmse = float(np.sqrt(mean_squared_error(validation_frame["actual"], validation_frame["predicted"])))
    mape = _safe_mape(validation_frame["actual"].to_numpy(), validation_frame["predicted"].to_numpy())

    price_basis = monthly_df.get("avg_unit_price", pd.Series(dtype=float)).dropna().tail(3).mean()
    if pd.isna(price_basis):
        revenue_sum = float(monthly_df.get("revenue", pd.Series(dtype=float)).sum())
        units_sum = float(monthly_df[target_col].sum()) if target_col in monthly_df else 0.0
        price_basis = revenue_sum / units_sum if units_sum else 1.0

    history = monthly_df[["date", target_col]].copy().sort_values("date").reset_index(drop=True)
    forecast_rows = []
    for _ in range(forecast_months):
        next_date = history["date"].max() + pd.offsets.MonthBegin(1)
        extended = pd.concat(
            [history, pd.DataFrame({"date": [next_date], target_col: [np.nan]})],
            ignore_index=True,
        )
        forecast_features, _ = build_feature_frame(extended, target_col=target_col)
        candidate = forecast_features.iloc[[-1]] if not forecast_features.empty else pd.DataFrame()
        if candidate.empty:
            predicted_units = float(history[target_col].tail(3).mean())
        else:
            predicted_units = float(model.predict(candidate[feature_cols])[0])
        predicted_units = max(predicted_units, 0.0)
        history = pd.concat(
            [history, pd.DataFrame({"date": [next_date], target_col: [predicted_units]})],
            ignore_index=True,
        )
        forecast_rows.append(
            {
                "date": next_date,
                "predicted_units": predicted_units,
                "predicted_revenue": predicted_units * float(price_basis),
            }
        )

    forecast_frame = pd.DataFrame(forecast_rows)
    forecast_frame["month_label"] = forecast_frame["date"].dt.strftime("%b %Y")

    feature_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "feature_frame": feature_df,
        "metrics": {"mae": mae, "rmse": rmse, "mape": mape},
        "validation_frame": validation_frame,
        "forecast_frame": forecast_frame,
        "feature_importance": feature_importance,
        "price_basis": float(price_basis),
    }
    _save_bundle(bundle)
    return bundle
