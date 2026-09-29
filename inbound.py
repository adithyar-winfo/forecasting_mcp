"""Inbound forecasting per purchase order: expected received quantity and arrival delay / ETA."""
from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

try:
    from .features import build_calendar_features
    from .models import (
        InsufficientDataError, LightGBMModel, ModelArtifact, compute_metrics, make_version, metric_or_inf,
    )
except ImportError:
    from features import build_calendar_features
    from models import (
        InsufficientDataError, LightGBMModel, ModelArtifact, compute_metrics, make_version, metric_or_inf,
    )

logger = logging.getLogger("forecasting_mcp.inbound")

OUTCOMES = ("delay_days", "fill_rate", "on_time_flag")
HISTORY_GROUPS = {
    "vendor": ["vendor_id"],
    "vendor_mode": ["vendor_id", "freight_mode"],
    "lane": ["vendor_id", "dc_id", "sku_id"],
}
HISTORY_FEATURES = [
    f"{prefix}_{stat}" for prefix in HISTORY_GROUPS
    for stat in ("po_count", *(f"{c}_mean" for c in OUTCOMES))
]
CATEGORICAL = ["vendor_id", "dc_id", "sku_id", "product_category", "freight_mode"]
NUMERIC = [
    "ordered_qty", "lead_time_days", "po_day_of_week", "po_month",
    "promised_day_of_week", "promised_month", "promised_peak_season", "promised_holiday_flag",
    "base_lead_time_days", "lead_time_std_days", "daily_capacity_units", "master_on_time_rate", "master_fill_rate",
    *HISTORY_FEATURES, "trailing_demand_28d", "qty_to_daily_demand",
]
FEATURES = CATEGORICAL + NUMERIC
MIN_ARRIVED_POS = 200
MIN_SPLIT_ROWS = 30
VAL_FRACTION = TEST_FRACTION = 0.15

OUTPUT_COLUMNS = [
    "po_id", "po_date", "vendor_id", "dc_id", "sku_id", "freight_mode", "ordered_qty", "promised_arrival_date",
    "expected_fill_rate", "expected_quantity", "expected_delay_days", "expected_arrival_date",
    "on_time_probability", "delay_probability", "overdue", "horizon_days", "week_start_date",
]


# ---------------- features (everything is known at PO creation time) ----------------
def _asof_history(pos: pd.DataFrame, known: pd.DataFrame, by: list[str], prefix: str) -> pd.DataFrame:
    """Stats of POs in the same group that had already arrived before each PO's po_date."""
    count_col = f"{prefix}_po_count"
    mean_cols = {c: f"{prefix}_{c}_mean" for c in OUTCOMES}
    out = pd.DataFrame(index=pos.index)
    if known.empty:
        out[count_col] = 0.0
        for col in mean_cols.values():
            out[col] = np.nan
        return out
    k = known.sort_values("actual_arrival_date", kind="mergesort")
    grouped = k.groupby(by, sort=False)
    right = k[by + ["actual_arrival_date"]].copy()
    right["_n"] = grouped.cumcount() + 1
    for c in OUTCOMES:
        right[f"_s_{c}"] = grouped[c].cumsum().astype(float)
    left = pos[by + ["po_date"]].reset_index(names="_row").sort_values("po_date", kind="mergesort")
    merged = pd.merge_asof(
        left, right, left_on="po_date", right_on="actual_arrival_date", by=by,
        direction="backward", allow_exact_matches=False,
    ).set_index("_row").reindex(pos.index)
    out[count_col] = merged["_n"].fillna(0).astype(float)
    for c, col in mean_cols.items():
        out[col] = merged[f"_s_{c}"] / merged["_n"]
    return out


def _trailing_demand(pos: pd.DataFrame, demand_daily: pd.DataFrame) -> pd.Series:
    """Mean daily DC x SKU demand over the 28 days before po_date."""
    if demand_daily.empty:
        return pd.Series(np.nan, index=pos.index)
    d = demand_daily[["date", "dc_id", "sku_id", "demand_units"]].sort_values("date").copy()
    d["demand_units"] = d["demand_units"].astype(float)
    d["trailing_demand_28d"] = d.groupby(["dc_id", "sku_id"])["demand_units"].transform(
        lambda s: s.shift(1).rolling(28, min_periods=7).mean()
    )
    left = pos[["dc_id", "sku_id", "po_date"]].reset_index(names="_row").sort_values("po_date", kind="mergesort")
    merged = pd.merge_asof(
        left, d[["date", "dc_id", "sku_id", "trailing_demand_28d"]], left_on="po_date", right_on="date",
        by=["dc_id", "sku_id"], direction="backward",
    ).set_index("_row").reindex(pos.index)
    return merged["trailing_demand_28d"].astype(float)


def _categories(inbound: pd.DataFrame, vendors: pd.DataFrame, products: pd.DataFrame) -> dict[str, list[str]]:
    sources = {
        "vendor_id": [inbound["vendor_id"], vendors["vendor_id"]],
        "dc_id": [inbound["dc_id"]],
        "sku_id": [inbound["sku_id"], products["sku_id"]],
        "product_category": [products["product_category"]],
        "freight_mode": [inbound["freight_mode"], vendors["primary_mode"]],
    }
    return {col: sorted({str(v) for s in series for v in s.dropna()}) for col, series in sources.items()}


def build_features(pos: pd.DataFrame, known: pd.DataFrame, vendors: pd.DataFrame, products: pd.DataFrame,
                   demand_daily: pd.DataFrame, calendar: pd.DataFrame,
                   categories: dict[str, list[str]]) -> pd.DataFrame:
    X = pd.DataFrame(index=pos.index)
    X["ordered_qty"] = pos["ordered_qty"].astype(float)
    X["lead_time_days"] = pos["lead_time_days"].astype(float)
    X["po_day_of_week"] = pos["po_date"].dt.dayofweek
    X["po_month"] = pos["po_date"].dt.month
    promised = build_calendar_features(pos["promised_arrival_date"], calendar)
    X["promised_day_of_week"] = promised["day_of_week"].to_numpy()
    X["promised_month"] = promised["month"].to_numpy()
    X["promised_peak_season"] = promised["peak_season"].to_numpy()
    X["promised_holiday_flag"] = promised["holiday_flag"].to_numpy()

    vendor = pos[["vendor_id"]].merge(vendors, on="vendor_id", how="left")
    X["base_lead_time_days"] = vendor["base_lead_time_days"].to_numpy(float)
    X["lead_time_std_days"] = vendor["lead_time_std_days"].to_numpy(float)
    X["daily_capacity_units"] = vendor["daily_capacity_units"].to_numpy(float)
    X["master_on_time_rate"] = vendor["on_time_rate"].to_numpy(float)
    X["master_fill_rate"] = vendor["fill_rate"].to_numpy(float)

    for prefix, by in HISTORY_GROUPS.items():
        X = X.join(_asof_history(pos, known, by, prefix))
    X["trailing_demand_28d"] = _trailing_demand(pos, demand_daily)
    X["qty_to_daily_demand"] = X["ordered_qty"] / X["trailing_demand_28d"].where(X["trailing_demand_28d"] > 0)

    product_category = pos[["sku_id"]].merge(products[["sku_id", "product_category"]], on="sku_id", how="left")
    source = {**{c: pos[c] for c in CATEGORICAL if c in pos}, "product_category": product_category["product_category"]}
    for col in CATEGORICAL:
        X[col] = pd.Categorical(np.asarray(source[col].astype(str)), categories=categories[col])
    return X[FEATURES]


# ---------------- training ----------------
def _split_masks(po_dates: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    unique = pd.DatetimeIndex(sorted(po_dates.unique()))
    n = len(unique)
    n_test = max(1, int(round(n * TEST_FRACTION)))
    n_val = max(1, int(round(n * VAL_FRACTION)))
    if n - n_test - n_val < 1:
        raise InsufficientDataError("Not enough distinct PO dates for a chronological train/validation/test split.")
    train_end, val_end = unique[n - n_test - n_val - 1], unique[n - n_test - 1]
    tr = (po_dates <= train_end).to_numpy()
    va = ((po_dates > train_end) & (po_dates <= val_end)).to_numpy()
    te = (po_dates > val_end).to_numpy()
    if min(tr.sum(), va.sum(), te.sum()) < MIN_SPLIT_ROWS:
        raise InsufficientDataError("Not enough arrived POs in each chronological split to train inbound models.")
    fmt = lambda d: pd.Timestamp(d).date().isoformat()  # noqa: E731
    split = {
        "train": {"start": fmt(unique[0]), "end": fmt(train_end), "rows": int(tr.sum())},
        "validation": {"start": fmt(unique[n - n_test - n_val]), "end": fmt(val_end), "rows": int(va.sum())},
        "test": {"start": fmt(unique[n - n_test]), "end": fmt(unique[-1]), "rows": int(te.sum())},
    }
    return tr, va, te, split


def _fit_target(X: pd.DataFrame, y: np.ndarray, masks: tuple[np.ndarray, np.ndarray, np.ndarray],
                baseline: np.ndarray, objective: str, clip: Callable[[np.ndarray], np.ndarray],
                evaluate: Callable[[np.ndarray, np.ndarray], dict]) -> tuple[LightGBMModel | None, dict]:
    """Select LightGBM vs historical baseline on validation MAE; report test metrics for both."""
    tr, va, te = masks
    model = LightGBMModel(objective=objective).fit(X.loc[tr], y[tr], X.loc[va], y[va])
    val = {"lightgbm": evaluate(clip(model.predict(X.loc[va])), va), "historical_baseline": evaluate(clip(baseline[va]), va)}
    selected = min(val, key=lambda name: metric_or_inf(val[name], "mae"))
    tv = tr | va
    refit = LightGBMModel(objective=objective).fit(X.loc[tv], y[tv], num_rounds=model.best_iteration)
    test = {"lightgbm": evaluate(clip(refit.predict(X.loc[te])), te), "historical_baseline": evaluate(clip(baseline[te]), te)}
    final = None
    if selected == "lightgbm":
        final = LightGBMModel(objective=objective).fit(X, y, num_rounds=model.best_iteration)
    return final, {"selected_model": selected, "best_iteration": model.best_iteration,
                   "validation_metrics": val, "test_metrics": test,
                   "top_feature_importance_gain": final.feature_importance() if final else {}}


def _baseline_fill(X: pd.DataFrame, default: float) -> np.ndarray:
    return X["vendor_fill_rate_mean"].fillna(X["master_fill_rate"]).fillna(default).to_numpy(float)


def _baseline_delay(X: pd.DataFrame, default: float) -> np.ndarray:
    return X["vendor_mode_delay_days_mean"].fillna(X["vendor_delay_days_mean"]).fillna(default).to_numpy(float)


def _clip_fill(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def _clip_delay(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, None)


def arrived_pos(inbound: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    return inbound[inbound["actual_arrival_date"].notna() & (inbound["actual_arrival_date"] <= as_of)].reset_index(drop=True)


def open_pos(inbound: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    not_arrived = inbound["actual_arrival_date"].isna() | (inbound["actual_arrival_date"] > as_of)
    return inbound[(inbound["po_date"] <= as_of) & not_arrived].reset_index(drop=True)


def train_inbound_models(inbound: pd.DataFrame, vendors: pd.DataFrame, products: pd.DataFrame,
                         demand_daily: pd.DataFrame, calendar: pd.DataFrame, as_of: pd.Timestamp,
                         signature: tuple) -> ModelArtifact:
    arrived = arrived_pos(inbound, as_of)
    if len(arrived) < MIN_ARRIVED_POS:
        raise InsufficientDataError(
            f"Inbound forecasting needs at least {MIN_ARRIVED_POS} arrived POs; found {len(arrived)}."
        )
    categories = _categories(inbound, vendors, products)
    X = build_features(arrived, arrived, vendors, products, demand_daily, calendar, categories)
    tr, va, te, split = _split_masks(arrived["po_date"])

    fill_default = float(arrived.loc[tr, "fill_rate"].mean())
    delay_default = float(arrived.loc[tr, "delay_days"].mean())
    ordered = arrived["ordered_qty"].to_numpy(float)
    received = arrived["received_qty"].to_numpy(float)
    delay = arrived["delay_days"].to_numpy(float)

    fill_model, quantity_meta = _fit_target(
        X, arrived["fill_rate"].to_numpy(float), (tr, va, te), _baseline_fill(X, fill_default), "regression",
        _clip_fill, lambda pred, rows: compute_metrics(received[rows], ordered[rows] * pred),
    )
    delay_model, delay_meta = _fit_target(
        X, delay, (tr, va, te), _baseline_delay(X, delay_default), "regression",
        _clip_delay, lambda pred, rows: compute_metrics(delay[rows], pred),
    )

    on_time = arrived["on_time_flag"].to_numpy(float)
    minority = int(min(on_time.sum(), len(on_time) - on_time.sum()))
    on_time_model = None
    if minority >= MIN_SPLIT_ROWS:
        on_time_model = LightGBMModel(objective="binary").fit(X.loc[tr], on_time[tr], X.loc[va], on_time[va])
        on_time_model = LightGBMModel(objective="binary").fit(X, on_time, num_rounds=on_time_model.best_iteration)
        on_time_method = "lightgbm_classifier"
    else:
        on_time_method = "smoothed_vendor_freight_mode_rate"
        logger.info("Only %d on-time/late minority-class POs; using smoothed historical on-time rate.", minority)

    model_name = "lightgbm" if quantity_meta["selected_model"] == delay_meta["selected_model"] == "lightgbm" else (
        f"quantity:{quantity_meta['selected_model']};delay:{delay_meta['selected_model']}"
    )
    version, trained_at = make_version("inbound", as_of)
    return ModelArtifact(
        key="inbound",
        model_name=model_name,
        version=version,
        trained_at=trained_at,
        data_signature=signature,
        payload={
            "fill_model": fill_model, "delay_model": delay_model, "on_time_model": on_time_model,
            "fill_default": fill_default, "delay_default": delay_default, "categories": categories,
            "cycle_days": np.sort((arrived["actual_arrival_date"] - arrived["po_date"]).dt.days.to_numpy()),
        },
        metadata={
            "approach": "PO/event-level models (not a time series): fill rate -> expected quantity, delay days -> ETA",
            "as_of_date": as_of.date().isoformat(),
            "training_pos": int(len(arrived)),
            "split": split,
            "quantity_model": quantity_meta,
            "delay_model": delay_meta,
            "on_time_method": on_time_method,
            "on_time_minority_class_count": minority,
            "features": FEATURES,
            "assumptions": [
                "Only POs that arrived on or before the as-of date are used for training.",
                "Vendor / lane history features only use POs that arrived before each PO's po_date.",
                "Expected quantity = ordered_qty x predicted fill rate (clipped to [0, 1]).",
                "Overdue POs (promised date passed, not yet arrived) get at least (as_of + 1 - promised) days delay.",
            ],
        },
    )


# ---------------- inference ----------------
def predict_open_pos(artifact: ModelArtifact, inbound: pd.DataFrame, vendors: pd.DataFrame, products: pd.DataFrame,
                     demand_daily: pd.DataFrame, calendar: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Forecast every PO placed on/before `as_of` that has not arrived by `as_of`."""
    pos = open_pos(inbound, as_of)
    if pos.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    p = artifact.payload
    X = build_features(pos, arrived_pos(inbound, as_of), vendors, products, demand_daily, calendar, p["categories"])

    fill = _clip_fill(p["fill_model"].predict(X) if p["fill_model"] else _baseline_fill(X, p["fill_default"]))
    delay = _clip_delay(p["delay_model"].predict(X) if p["delay_model"] else _baseline_delay(X, p["delay_default"]))

    promised = pos["promised_arrival_date"]
    overdue = (promised <= as_of).to_numpy()
    min_delay = ((as_of + pd.Timedelta(days=1)) - promised).dt.days.clip(lower=0).to_numpy(float)
    delay = np.where(overdue, np.maximum(delay, min_delay), delay)
    delay_days = np.round(delay).astype(int)
    expected_arrival = promised + pd.to_timedelta(delay_days, unit="D")

    if p["on_time_model"] is not None:
        on_time = p["on_time_model"].predict(X)
    else:
        n = X["vendor_mode_po_count"].to_numpy(float)
        rate = X["vendor_mode_on_time_flag_mean"].fillna(0).to_numpy(float)
        on_time = (rate * n + 1.0) / (n + 2.0)
    on_time = np.where(overdue, 0.0, on_time)

    out = pos[["po_id", "po_date", "vendor_id", "dc_id", "sku_id", "freight_mode", "ordered_qty", "promised_arrival_date"]].copy()
    out["expected_fill_rate"] = np.round(fill, 4)
    out["expected_quantity"] = np.round(pos["ordered_qty"].to_numpy(float) * fill, 1)
    out["expected_delay_days"] = np.round(delay, 2)
    out["expected_arrival_date"] = expected_arrival
    out["on_time_probability"] = np.round(on_time, 4)
    out["delay_probability"] = np.round(1.0 - on_time, 4)
    out["overdue"] = overdue
    out["horizon_days"] = (expected_arrival - as_of).dt.days
    out["week_start_date"] = expected_arrival - pd.to_timedelta(expected_arrival.dt.dayofweek, unit="D")
    return out[OUTPUT_COLUMNS].sort_values(["expected_arrival_date", "po_id"]).reset_index(drop=True)
