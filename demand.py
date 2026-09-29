"""Demand forecasting: global LightGBM over Date x DC x SKU with recursive multi-step inference."""
from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mcp_servers.forecasting_mcp.config import MAX_HORIZON
from mcp_servers.forecasting_mcp.features import CALENDAR_FEATURES, build_calendar_features
from mcp_servers.forecasting_mcp.models import (
    InsufficientDataError, LightGBMModel, ModelArtifact, compute_metrics, make_version, metric_or_inf,
)

logger = logging.getLogger("forecasting_mcp.demand")

LAGS = (1, 7, 14, 28)
ROLL_MEAN_WINDOWS = (7, 14, 28)
ROLL_STD_WINDOWS = (7, 28)
HISTORY_WINDOW = max(LAGS + ROLL_MEAN_WINDOWS + ROLL_STD_WINDOWS)
SEASON = 7
MIN_TRAIN_DAYS = 90
EVAL_WINDOW_BOUNDS = (14, 56)
INTERVAL_QUANTILES = (0.1, 0.9)
LGB_OBJECTIVE = "tweedie"

LAG_FEATURES = (
    [f"lag_{k}" for k in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_MEAN_WINDOWS]
    + [f"roll_std_{w}" for w in ROLL_STD_WINDOWS]
)
EXOG_FEATURES = ["unit_price_usd", "price_ratio", "promo_flag", "promo_discount_pct"]
STATIC_CATEGORICAL = ["dc_id", "sku_id", "product_category", "subcategory"]
STATIC_NUMERIC = ["base_price_usd", "demand_factor"]
FEATURES = LAG_FEATURES + EXOG_FEATURES + CALENDAR_FEATURES + STATIC_CATEGORICAL + STATIC_NUMERIC

ASSUMPTIONS = [
    "Future promotions are unknown, so forecasts assume no promotion (promo_flag=0, discount=0).",
    "Future unit price = median non-promo price of the last 28 days per DC x SKU.",
    "Holiday/peak flags beyond fl.calendar reuse the latest year's month/day flags.",
    "Stockout days are excluded as training targets (censored demand) but kept as lag history.",
    "Prediction interval = forecast + 10th/90th percentile of validation residuals per DC x SKU.",
]


@dataclass
class DemandPanel:
    dates: pd.DatetimeIndex
    static: pd.DataFrame
    demand: np.ndarray
    price: np.ndarray
    promo: np.ndarray
    discount: np.ndarray
    stockout: np.ndarray
    calendar_features: pd.DataFrame


def build_panel(history: pd.DataFrame, products: pd.DataFrame, dcs: pd.DataFrame,
                calendar: pd.DataFrame) -> DemandPanel:
    """Pivot fl.demand_history into aligned (date x series) arrays."""
    if history.empty:
        raise InsufficientDataError("fl.demand_history returned no rows.")
    df = history.copy()
    df["series_key"] = df["dc_id"] + "|" + df["sku_id"]
    dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    keys = sorted(df["series_key"].unique())

    def wide(col: str) -> pd.DataFrame:
        return df.pivot(index="date", columns="series_key", values=col).reindex(index=dates, columns=keys).astype(float)

    demand = wide("demand_units")
    started = demand.notna().cummax()
    gaps = int((demand.isna() & started).to_numpy().sum())
    if gaps:
        logger.warning("Filling %d missing Date x DC x SKU demand rows with 0.", gaps)
    demand = demand.fillna(0.0).where(started)

    static = (
        df.sort_values("date").drop_duplicates("series_key", keep="last")
        .set_index("series_key").loc[keys, ["dc_id", "sku_id", "product_category"]].reset_index()
        .merge(products[["sku_id", "subcategory", "base_price_usd"]], on="sku_id", how="left")
        .merge(dcs[["dc_id", "demand_factor"]], on="dc_id", how="left")
    )
    price = wide("unit_price_usd").ffill().bfill()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median_price = np.nanmedian(price.to_numpy(), axis=0)
    static["base_price_usd"] = static["base_price_usd"].astype(float).fillna(pd.Series(median_price))
    static["demand_factor"] = static["demand_factor"].astype(float).fillna(1.0)
    price = price.fillna(pd.Series(static["base_price_usd"].to_numpy(), index=keys))

    category_sources = {
        "dc_id": dcs["dc_id"], "sku_id": products["sku_id"],
        "product_category": products["product_category"], "subcategory": products["subcategory"],
    }
    for col in STATIC_CATEGORICAL:
        cats = sorted(set(category_sources[col].dropna().astype(str)) | set(static[col].dropna().astype(str)))
        static[col] = pd.Categorical(static[col].astype(str), categories=cats)

    return DemandPanel(
        dates=dates,
        static=static,
        demand=demand.to_numpy(),
        price=price.to_numpy(),
        promo=wide("promo_flag").fillna(0).to_numpy(),
        discount=wide("promo_discount_pct").fillna(0).to_numpy(),
        stockout=wide("stockout_flag").fillna(0).to_numpy(),
        calendar_features=build_calendar_features(
            pd.date_range(dates[0], dates[-1] + pd.Timedelta(days=MAX_HORIZON), freq="D"), calendar
        ),
    )


# ---------------- feature construction (shared by training and inference) ----------------
def lag_feature_arrays(demand: np.ndarray) -> dict[str, np.ndarray]:
    """Lag/rolling features for every date; rolling windows end at t-1 so the target is never used."""
    wide = pd.DataFrame(demand)
    shifted = wide.shift(1)
    feats = {f"lag_{k}": wide.shift(k).to_numpy() for k in LAGS}
    feats.update({f"roll_mean_{w}": shifted.rolling(w).mean().to_numpy() for w in ROLL_MEAN_WINDOWS})
    feats.update({f"roll_std_{w}": shifted.rolling(w).std().to_numpy() for w in ROLL_STD_WINDOWS})
    return feats


def last_lag_features(history: np.ndarray) -> dict[str, np.ndarray]:
    """Same features as lag_feature_arrays for the day after the last row of `history`."""
    feats = {f"lag_{k}": history[-k] for k in LAGS}
    feats.update({f"roll_mean_{w}": history[-w:].mean(axis=0) for w in ROLL_MEAN_WINDOWS})
    feats.update({f"roll_std_{w}": history[-w:].std(axis=0, ddof=1) for w in ROLL_STD_WINDOWS})
    return {name: values[None, :] for name, values in feats.items()}


def _exog(panel: DemandPanel, price: np.ndarray, promo: np.ndarray, discount: np.ndarray) -> dict[str, np.ndarray]:
    base = panel.static["base_price_usd"].to_numpy(float)
    base = np.where(base > 0, base, np.nan)
    return {"unit_price_usd": price, "price_ratio": price / base[None, :], "promo_flag": promo, "promo_discount_pct": discount}


def _assemble(panel: DemandPanel, dates: pd.DatetimeIndex, arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    n_dates, n_series = len(dates), len(panel.static)
    data: dict[str, object] = {name: np.asarray(arr, dtype=float).reshape(-1) for name, arr in arrays.items()}
    cal = panel.calendar_features.loc[dates]
    for col in CALENDAR_FEATURES:
        data[col] = np.repeat(cal[col].to_numpy(), n_series)
    for col in STATIC_NUMERIC:
        data[col] = np.tile(panel.static[col].to_numpy(float), n_dates)
    for col in STATIC_CATEGORICAL:
        cat = panel.static[col].cat
        data[col] = pd.Categorical.from_codes(np.tile(cat.codes.to_numpy(), n_dates), categories=cat.categories)
    return pd.DataFrame(data)[FEATURES]


def training_frame(panel: DemandPanel, lags: dict[str, np.ndarray], start: int, end: int) -> tuple[pd.DataFrame, np.ndarray]:
    rows = slice(start, end + 1)
    arrays = {name: arr[rows] for name, arr in lags.items()}
    arrays.update(_exog(panel, panel.price[rows], panel.promo[rows], panel.discount[rows]))
    X = _assemble(panel, panel.dates[rows], arrays)
    y = panel.demand[rows].reshape(-1)
    valid = ~np.isnan(y) & (panel.stockout[rows].reshape(-1) == 0) & X[LAG_FEATURES].notna().all(axis=1).to_numpy()
    return X.loc[valid].reset_index(drop=True), y[valid]


def _future_exog(panel: DemandPanel, origin: int, steps: int) -> dict[str, np.ndarray]:
    lo = max(0, origin - 27)
    price, promo = panel.price[lo:origin + 1], panel.promo[lo:origin + 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        regular = np.nanmedian(np.where(promo == 0, price, np.nan), axis=0)
    regular = np.where(np.isnan(regular), price[-1], regular)
    regular = np.where(np.isnan(regular), panel.static["base_price_usd"].to_numpy(float), regular)
    zeros = np.zeros((steps, len(regular)))
    return _exog(panel, np.tile(regular, (steps, 1)), zeros, zeros)


def recursive_forecast(panel: DemandPanel, predict: Callable[[pd.DataFrame], np.ndarray],
                       origin: int, steps: int) -> np.ndarray:
    """Forecast `steps` days after position `origin`, feeding predictions back as lag history."""
    if origin < HISTORY_WINDOW - 1:
        raise InsufficientDataError("Not enough history before the forecast origin to build lag features.")
    history = panel.demand[origin - HISTORY_WINDOW + 1: origin + 1].copy()
    dates = pd.date_range(panel.dates[origin] + pd.Timedelta(days=1), periods=steps, freq="D")
    exog = _future_exog(panel, origin, steps)
    out = np.empty((steps, history.shape[1]))
    for i in range(steps):
        arrays = last_lag_features(history)
        arrays.update({name: values[i:i + 1] for name, values in exog.items()})
        pred = np.clip(predict(_assemble(panel, dates[i:i + 1], arrays)), 0, None)
        out[i] = pred
        history = np.vstack([history[1:], pred[None, :]])
    out[:, np.isnan(panel.demand[origin])] = np.nan
    return out


def seasonal_naive(panel: DemandPanel, origin: int, steps: int) -> np.ndarray:
    last_season = panel.demand[origin - SEASON + 1: origin + 1]
    return np.stack([last_season[i % SEASON] for i in range(steps)])


# ---------------- training ----------------
def _window_dates(panel: DemandPanel, origin: int, steps: int) -> dict[str, str]:
    return {"start": panel.dates[origin + 1].date().isoformat(), "end": panel.dates[origin + steps].date().isoformat()}


def train_demand_model(panel: DemandPanel, signature: tuple) -> ModelArtifact:
    n_days = len(panel.dates)
    window = int(np.clip(n_days // 10, *EVAL_WINDOW_BOUNDS))
    required = HISTORY_WINDOW + MIN_TRAIN_DAYS + 2 * window
    if n_days < required:
        raise InsufficientDataError(f"Demand forecasting needs at least {required} days of history; found {n_days}.")

    last = n_days - 1
    test_origin = last - window
    val_origin = test_origin - window
    first = HISTORY_WINDOW
    lags = lag_feature_arrays(panel.demand)

    # Train -> early-stop on validation (one-step), then score validation recursively.
    X_tr, y_tr = training_frame(panel, lags, first, val_origin)
    X_va, y_va = training_frame(panel, lags, val_origin + 1, test_origin)
    model = LightGBMModel(objective=LGB_OBJECTIVE).fit(X_tr, y_tr, X_va, y_va)
    best_iteration = model.best_iteration

    val_actual = panel.demand[val_origin + 1: test_origin + 1]
    val_preds = {
        "lightgbm": recursive_forecast(panel, model.predict, val_origin, window),
        "seasonal_naive": seasonal_naive(panel, val_origin, window),
    }
    val_metrics = {name: compute_metrics(val_actual, pred) for name, pred in val_preds.items()}
    selected = min(val_metrics, key=lambda name: metric_or_inf(val_metrics[name], "wape"))

    # Test period is scored only after selection; model refit on train+validation.
    X_tv, y_tv = training_frame(panel, lags, first, test_origin)
    model_tv = LightGBMModel(objective=LGB_OBJECTIVE).fit(X_tv, y_tv, num_rounds=best_iteration)
    test_actual = panel.demand[test_origin + 1:]
    test_preds = {
        "lightgbm": recursive_forecast(panel, model_tv.predict, test_origin, window),
        "seasonal_naive": seasonal_naive(panel, test_origin, window),
    }
    test_metrics = {name: compute_metrics(test_actual, pred) for name, pred in test_preds.items()}

    final_model = None
    importance: dict[str, float] = {}
    final_rows = 0
    if selected == "lightgbm":
        X_all, y_all = training_frame(panel, lags, first, last)
        final_rows = len(y_all)
        final_model = LightGBMModel(objective=LGB_OBJECTIVE).fit(X_all, y_all, num_rounds=best_iteration)
        forecast = recursive_forecast(panel, final_model.predict, last, MAX_HORIZON)
        importance = final_model.feature_importance()
    else:
        forecast = seasonal_naive(panel, last, MAX_HORIZON)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q_lo, q_hi = np.nanquantile(val_actual - val_preds[selected], INTERVAL_QUANTILES, axis=0)
    q_lo, q_hi = np.nan_to_num(q_lo), np.nan_to_num(q_hi)
    lower = np.minimum(np.clip(forecast + q_lo, 0, None), forecast)
    upper = np.maximum(forecast + q_hi, forecast)

    dc_frame = pd.DataFrame(panel.demand, index=panel.dates).T.groupby(panel.static["dc_id"].astype(str).to_numpy()).sum(min_count=1).T
    dc_daily = dc_frame.rename_axis(index="date", columns="dc_id").stack().rename("demand_units").reset_index()

    version, trained_at = make_version(selected, panel.dates[-1])
    return ModelArtifact(
        key="demand",
        model_name=selected,
        version=version,
        trained_at=trained_at,
        data_signature=signature,
        payload={
            "model": final_model,
            "static": panel.static,
            "origin_date": panel.dates[-1],
            "forecast_dates": pd.date_range(panel.dates[-1] + pd.Timedelta(days=1), periods=MAX_HORIZON, freq="D"),
            "forecast": forecast,
            "lower": lower,
            "upper": upper,
            "eval_window": window,
            "test_dates": panel.dates[test_origin + 1:],
            "test_actual": test_actual,
            "test_preds": test_preds,
            "dc_daily_demand": dc_daily,
        },
        metadata={
            "algorithm": "LightGBM global model (tweedie objective) with recursive multi-step forecasting",
            "candidates": ["lightgbm", "seasonal_naive"],
            "selected_model": selected,
            "selection_rule": "lowest WAPE on the validation period (test period not used for selection)",
            "best_iteration": best_iteration,
            "final_training_rows": final_rows,
            "series_count": len(panel.static),
            "history": {"start": panel.dates[0].date().isoformat(), "end": panel.dates[-1].date().isoformat()},
            "split": {
                "train": {"start": panel.dates[first].date().isoformat(), "end": panel.dates[val_origin].date().isoformat()},
                "validation": _window_dates(panel, val_origin, window),
                "test": _window_dates(panel, test_origin, window),
            },
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "features": FEATURES,
            "top_feature_importance_gain": importance,
            "assumptions": ASSUMPTIONS,
        },
    )


# ---------------- inference views ----------------
def series_mask(artifact: ModelArtifact, filters: dict[str, str | None]) -> np.ndarray:
    static = artifact.payload["static"]
    mask = np.ones(len(static), dtype=bool)
    for col in ("dc_id", "sku_id", "product_category"):
        if filters.get(col):
            mask &= static[col].astype(str).to_numpy() == filters[col]
    return mask


def demand_forecast_frame(artifact: ModelArtifact, mask: np.ndarray, horizon: int) -> pd.DataFrame:
    p = artifact.payload
    idx = np.flatnonzero(mask)
    static = p["static"].iloc[idx]
    dates = p["forecast_dates"][:horizon]
    n = len(dates)
    frame = pd.DataFrame({
        "forecast_date": np.repeat(dates, len(idx)),
        "horizon_days": np.repeat(np.arange(1, n + 1), len(idx)),
        "dc_id": np.tile(static["dc_id"].astype(str).to_numpy(), n),
        "sku_id": np.tile(static["sku_id"].astype(str).to_numpy(), n),
        "product_category": np.tile(static["product_category"].astype(str).to_numpy(), n),
        "predicted_demand_units": p["forecast"][:n, idx].reshape(-1),
        "lower_bound": p["lower"][:n, idx].reshape(-1),
        "upper_bound": p["upper"][:n, idx].reshape(-1),
    })
    return frame.dropna(subset=["predicted_demand_units"]).round(
        {"predicted_demand_units": 2, "lower_bound": 2, "upper_bound": 2}
    ).reset_index(drop=True)


def demand_test_metrics(artifact: ModelArtifact, mask: np.ndarray, horizon: int) -> dict:
    """Hold-out (test) metrics for the requested series, over the first min(horizon, test window) days."""
    p = artifact.payload
    days = min(horizon, p["eval_window"])
    actual = p["test_actual"][:days, mask]
    metrics = compute_metrics(actual, p["test_preds"][artifact.model_name][:days, mask])
    return {
        **metrics,
        "model": artifact.model_name,
        "evaluation": "chronological hold-out test period, recursive multi-step forecast from the test origin",
        "test_period": {"start": p["test_dates"][0].date().isoformat(), "end": p["test_dates"][days - 1].date().isoformat()},
        "evaluated_days": days,
        "baseline_seasonal_naive": compute_metrics(actual, p["test_preds"]["seasonal_naive"][:days, mask]),
    }
