"""Workload and labor planning: demand/inbound drivers -> workload -> labor hours -> headcount."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    from .models import (
        InsufficientDataError, ModelArtifact, compute_metrics, make_version,
    )
except ImportError:
    from models import (
        InsufficientDataError, ModelArtifact, compute_metrics, make_version,
    )

logger = logging.getLogger("forecasting_mcp.labor")

# Workload driver per process (matched case-insensitively).
PROCESS_DRIVERS = {"picking": "demand", "packing": "demand", "returns": "demand", "receiving": "inbound"}
TRAILING_DAYS = 90
AVAILABLE_LABOR_DAYS = 28
EVAL_DAYS = 56
MIN_LABOR_DAYS = 28
MAX_ABSENCE_RATE = 0.5
DEFAULT_HOURS_PER_HEAD = 8.0
SHIFT_KEYS = ["dc_id", "process", "shift"]

OUTPUT_COLUMNS = [
    "forecast_date", "horizon_days", "dc_id", "process", "shift", "driver", "driver_units",
    "predicted_workload_units", "predicted_productivity_uplh", "required_labor_hours", "absence_rate",
    "expected_absence_count", "adjusted_labor_hours", "required_headcount", "available_headcount", "headcount_gap",
]


def _days(n: int) -> pd.Timedelta:
    return pd.Timedelta(days=n)


def historical_drivers(dc_daily_demand: pd.DataFrame, arrived: pd.DataFrame) -> pd.DataFrame:
    """Daily DC demand units and received inbound units (by actual arrival date)."""
    demand = dc_daily_demand.groupby(["date", "dc_id"])["demand_units"].sum().astype(float).rename("demand")
    inbound = (
        arrived.rename(columns={"actual_arrival_date": "date"})
        .groupby(["date", "dc_id"])["received_qty"].sum().astype(float).rename("inbound")
    )
    return pd.concat([demand, inbound], axis=1).fillna(0.0).rename_axis(["date", "dc_id"]).reset_index()


def _daily_workload(labor: pd.DataFrame, drivers: pd.DataFrame) -> pd.DataFrame:
    daily = labor.groupby(["date", "dc_id", "process"], as_index=False)["workload_units"].sum()
    daily["workload_units"] = daily["workload_units"].astype(float)
    daily["driver"] = daily["process"].astype(str).str.lower().map(PROCESS_DRIVERS)
    daily = daily.merge(drivers, on=["date", "dc_id"], how="left")
    daily[["demand", "inbound"]] = daily[["demand", "inbound"]].fillna(0.0)
    daily["driver_units"] = np.where(daily["driver"] == "demand", daily["demand"], daily["inbound"])
    return daily


def _ratios(frame: pd.DataFrame) -> pd.DataFrame:
    """Workload units per driver unit by DC x process; falls back to the cross-DC process ratio."""
    g = frame.groupby(["dc_id", "process", "driver"], as_index=False)[["workload_units", "driver_units"]].sum()
    g["ratio"] = g["workload_units"] / g["driver_units"].where(g["driver_units"] > 0)
    by_process = frame.groupby("process")[["workload_units", "driver_units"]].sum()
    process_ratio = by_process["workload_units"] / by_process["driver_units"].where(by_process["driver_units"] > 0)
    g["ratio"] = g["ratio"].fillna(g["process"].map(process_ratio)).fillna(0.0)
    return g[["dc_id", "process", "driver", "ratio"]]


def _validate(daily: pd.DataFrame) -> dict:
    """Chronological hold-out: ratios fit before the cutoff, scored on the last EVAL_DAYS with actual drivers."""
    last, first = daily["date"].max(), daily["date"].min()
    eval_days = min(EVAL_DAYS, ((last - first).days + 1) // 4)
    if eval_days < 7:
        return {"status": "skipped", "reason": "not enough labor history for a hold-out period"}
    cutoff = last - _days(eval_days)
    fit = daily[(daily["date"] <= cutoff) & (daily["date"] > cutoff - _days(TRAILING_DAYS))]
    test = daily[daily["date"] > cutoff].merge(_ratios(fit)[["dc_id", "process", "ratio"]], on=["dc_id", "process"], how="left")
    test["pred_driver_ratio"] = test["ratio"].fillna(0.0) * test["driver_units"]
    last_week = (
        daily[(daily["date"] > cutoff - _days(7)) & (daily["date"] <= cutoff)]
        .assign(dow=lambda d: d["date"].dt.dayofweek)[["dc_id", "process", "dow", "workload_units"]]
        .rename(columns={"workload_units": "pred_seasonal_naive"})
    )
    test = test.assign(dow=test["date"].dt.dayofweek).merge(last_week, on=["dc_id", "process", "dow"], how="left")

    def score(frame: pd.DataFrame) -> dict:
        return {
            "driver_ratio": compute_metrics(frame["workload_units"], frame["pred_driver_ratio"]),
            "seasonal_naive": compute_metrics(frame["workload_units"], frame["pred_seasonal_naive"]),
        }

    return {
        "evaluation": "chronological hold-out; ratios fit on the 90 days before the test period, applied to actual drivers",
        "test_period": {"start": (cutoff + _days(1)).date().isoformat(), "end": last.date().isoformat()},
        "overall": score(test),
        "by_process": {str(proc): score(g) for proc, g in test.groupby("process")},
    }


def _shift_params(labor: pd.DataFrame, last: pd.Timestamp) -> pd.DataFrame:
    recent = labor[labor["date"] > last - _days(TRAILING_DAYS)]
    agg = recent.groupby(SHIFT_KEYS).agg(
        workload=("workload_units", "sum"), absence=("absence_count", "sum"),
        scheduled_headcount=("scheduled_headcount", "sum"), scheduled_hours=("scheduled_hours", "sum"),
    ).astype(float)
    productivity = (
        recent[recent["productivity_units_per_labor_hour"] > 0]
        .groupby(SHIFT_KEYS)["productivity_units_per_labor_hour"].mean().rename("productivity")
    )
    available = (
        labor[labor["date"] > last - _days(AVAILABLE_LABOR_DAYS)]
        .groupby(SHIFT_KEYS)["actual_headcount"].mean().rename("available_headcount")
    )
    p = agg.join(productivity).join(available).reset_index()
    total = p.groupby(["dc_id", "process"])["workload"].transform("sum")
    n_shifts = p.groupby(["dc_id", "process"])["shift"].transform("count")
    headcount = p["scheduled_headcount"].where(p["scheduled_headcount"] > 0)
    p["workload_share"] = (p["workload"] / total.where(total > 0)).fillna(1.0 / n_shifts)
    p["absence_rate"] = (p["absence"] / headcount).fillna(0.0).clip(0.0, MAX_ABSENCE_RATE)
    p["hours_per_head"] = (p["scheduled_hours"] / headcount).fillna(DEFAULT_HOURS_PER_HEAD)
    p["productivity"] = p["productivity"].fillna(p.groupby("process")["productivity"].transform("mean"))
    return p[SHIFT_KEYS + ["workload_share", "productivity", "absence_rate", "hours_per_head", "available_headcount"]]


def _mean_daily_receipts(arrived: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float]:
    if arrived.empty:
        return {}
    span = min(TRAILING_DAYS, (as_of - arrived["actual_arrival_date"].min()).days + 1)
    recent = arrived[arrived["actual_arrival_date"] > as_of - _days(span)]
    return (recent.groupby("dc_id")["received_qty"].sum().astype(float) / span).to_dict()


def fit_labor_plan(labor: pd.DataFrame, dc_daily_demand: pd.DataFrame, arrived: pd.DataFrame,
                   cycle_days: np.ndarray, as_of: pd.Timestamp, signature: tuple,
                   upstream: dict[str, str]) -> ModelArtifact:
    labor = labor[labor["date"] <= as_of]
    n_days = labor["date"].nunique()
    if n_days < MIN_LABOR_DAYS:
        raise InsufficientDataError(
            f"Labor forecasting needs at least {MIN_LABOR_DAYS} days of fl.labor_operations_history; found {n_days}."
        )
    daily = _daily_workload(labor, historical_drivers(dc_daily_demand, arrived))
    unknown = sorted(daily.loc[daily["driver"].isna(), "process"].astype(str).unique())
    if unknown:
        logger.warning("No workload driver mapping for processes %s; they are excluded.", unknown)
    daily = daily.dropna(subset=["driver"])
    last = daily["date"].max()
    ratios = _ratios(daily[daily["date"] > last - _days(TRAILING_DAYS)])

    version, trained_at = make_version("driver_ratio", last)
    return ModelArtifact(
        key="labor",
        model_name="driver_ratio",
        version=version,
        trained_at=trained_at,
        data_signature=signature,
        payload={
            "ratios": ratios,
            "shift_params": _shift_params(labor, last),
            "mean_daily_receipts": _mean_daily_receipts(arrived, as_of),
            "cycle_days": np.asarray(cycle_days),
        },
        metadata={
            "approach": "driver-based: demand/inbound forecasts -> workload (Date x DC x Process) "
                        "-> shift split -> labor hours via productivity -> absence-adjusted headcount",
            "process_drivers": PROCESS_DRIVERS,
            "excluded_processes": unknown,
            "labor_history_end": last.date().isoformat(),
            "workload_ratios": ratios.round(4).to_dict(orient="records"),
            "validation": _validate(daily),
            "upstream_models": upstream,
            "assumptions": [
                f"Ratios, shift split, productivity and absence rates use the trailing {TRAILING_DAYS} days.",
                "Picking/Packing/Returns workload is driven by forecast DC demand; Receiving by expected inbound receipts.",
                "Expected receipts = open-PO forecast arrivals + trailing daily receipts x share of POs "
                "not yet placed (empirical PO cycle-time distribution).",
                "required_labor_hours = workload / expected productivity (units per labor hour).",
                "adjusted_labor_hours = required_labor_hours / (1 - absence rate); "
                "required_headcount = ceil(adjusted hours / scheduled hours per head).",
                f"available_headcount = mean actual headcount over the last {AVAILABLE_LABOR_DAYS} days.",
            ],
        },
    )


def plan_labor(artifact: ModelArtifact, demand_by_dc: pd.DataFrame, open_receipts: pd.DataFrame,
               as_of: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """demand_by_dc / open_receipts columns: forecast_date, dc_id, units."""
    p = artifact.payload
    dates = pd.DataFrame({
        "forecast_date": pd.date_range(as_of + _days(1), periods=horizon, freq="D"),
        "horizon_days": np.arange(1, horizon + 1),
    })
    grid = (
        p["ratios"].merge(dates, how="cross")
        .merge(demand_by_dc.rename(columns={"units": "demand_driver"}), on=["forecast_date", "dc_id"], how="left")
        .merge(open_receipts.rename(columns={"units": "open_po_receipts"}), on=["forecast_date", "dc_id"], how="left")
    )
    cycle = p["cycle_days"]
    unplaced_share = (
        np.searchsorted(cycle, grid["horizon_days"].to_numpy() - 1, side="right") / len(cycle) if len(cycle) else 0.0
    )
    grid["inbound_driver"] = (
        grid["open_po_receipts"].fillna(0.0)
        + grid["dc_id"].map(p["mean_daily_receipts"]).fillna(0.0).to_numpy() * unplaced_share
    )
    grid["driver_units"] = np.where(grid["driver"] == "demand", grid["demand_driver"].fillna(0.0), grid["inbound_driver"])

    rows = grid.merge(p["shift_params"], on=["dc_id", "process"], how="inner")
    rows["predicted_workload_units"] = rows["ratio"] * rows["driver_units"] * rows["workload_share"]
    rows["predicted_productivity_uplh"] = rows["productivity"]
    rows["required_labor_hours"] = rows["predicted_workload_units"] / rows["productivity"].where(rows["productivity"] > 0)
    rows["adjusted_labor_hours"] = rows["required_labor_hours"] / (1.0 - rows["absence_rate"])
    rows["required_headcount"] = np.ceil((rows["adjusted_labor_hours"] / rows["hours_per_head"]).round(4))
    rows["expected_absence_count"] = rows["required_headcount"] * rows["absence_rate"]
    rows["headcount_gap"] = rows["required_headcount"] - rows["available_headcount"]
    rounding = {c: 2 for c in ("driver_units", "predicted_workload_units", "predicted_productivity_uplh",
                               "required_labor_hours", "adjusted_labor_hours", "available_headcount", "headcount_gap")}
    rounding.update({"absence_rate": 4, "expected_absence_count": 3})
    return rows[OUTPUT_COLUMNS].round(rounding).sort_values(["forecast_date", "dc_id", "process", "shift"]).reset_index(drop=True)
