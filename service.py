"""Supply-chain forecasting service: request validation, model caching, persistence and JSON-safe output."""
from __future__ import annotations

import logging
import math
import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from . import demand as demand_mod
    from . import inbound as inbound_mod
    from . import labor as labor_mod
    from .config import DEFAULT_MAX_ROWS, MAX_HORIZON, MAX_ROWS_LIMIT
    from .db_adapter import SupplyChainDBAdapter
    from .models import (
        ForecastError, InsufficientDataError, InvalidRequestError, ModelArtifact, ModelCache,
    )
except ImportError:
    import demand as demand_mod
    import inbound as inbound_mod
    import labor as labor_mod
    from config import DEFAULT_MAX_ROWS, MAX_HORIZON, MAX_ROWS_LIMIT
    from db_adapter import SupplyChainDBAdapter
    from models import (
        ForecastError, InsufficientDataError, InvalidRequestError, ModelArtifact, ModelCache,
    )

logger = logging.getLogger("forecasting_mcp.service")


def to_jsonable(value: Any) -> Any:
    """Recursively convert pandas/numpy/date values into JSON-compatible Python objects."""
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, pd.DataFrame):
        return to_jsonable(value.to_dict(orient="records"))
    if isinstance(value, (pd.Series, pd.Index, np.ndarray)):
        return to_jsonable(value.tolist())
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        ts = pd.Timestamp(value)
        return ts.date().isoformat() if ts.tz is None and ts == ts.normalize() else ts.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _validate_int(value: Any, name: str, low: int, high: int) -> int:
    try:
        number = int(value)
        if isinstance(value, bool) or float(value) != number:
            raise ValueError
    except (TypeError, ValueError):
        raise InvalidRequestError(f"{name} must be an integer between {low} and {high}; got {value!r}.") from None
    if not low <= number <= high:
        raise InvalidRequestError(f"{name} must be between {low} and {high}; got {number}.")
    return number


def _match(value: Any, options: Iterable[Any], label: str) -> str | None:
    """Case-insensitive match of a user filter against known values; None/blank means 'no filter'."""
    if value is None or str(value).strip() == "":
        return None
    lookup = {str(o).strip().lower(): str(o) for o in options if o is not None and not pd.isna(o)}
    key = str(value).strip().lower()
    if key not in lookup:
        valid = sorted(lookup.values())
        shown = valid[:25] + (["..."] if len(valid) > 25 else [])
        raise InvalidRequestError(f"Unknown {label} '{value}'. Valid values: {shown}")
    return lookup[key]


def _apply_filters(frame: pd.DataFrame, filters: dict[str, str | None]) -> pd.DataFrame:
    for col, value in filters.items():
        if value is not None:
            frame = frame[frame[col].astype(str) == value]
    return frame


def _active(filters: dict[str, str | None]) -> dict[str, str]:
    return {k: v for k, v in filters.items() if v is not None}


def _new_run_id() -> str:
    return str(uuid.uuid4())


class SupplyChainForecastService:
    """Demand, inbound and labor forecasting for the Foot Locker supply-chain use case."""

    def __init__(self, db: SupplyChainDBAdapter, cache: ModelCache | None = None) -> None:
        self.db = db
        self.cache = cache or ModelCache()

    # ================= public API =================
    def forecast_demand(self, dc_id: str | None = None, sku_id: str | None = None,
                        product_category: str | None = None, horizon: int = 30, persist: bool = True,
                        force_retrain: bool = False, max_rows: int = DEFAULT_MAX_ROWS) -> dict[str, Any]:
        horizon = _validate_int(horizon, "horizon", 1, MAX_HORIZON)
        max_rows = _validate_int(max_rows, "max_rows", 0, MAX_ROWS_LIMIT)
        products, dcs = self.db.get_product_master(), self.db.get_dc_master()
        filters = {
            "dc_id": _match(dc_id, dcs["dc_id"], "dc_id"),
            "sku_id": _match(sku_id, products["sku_id"], "sku_id"),
            "product_category": _match(product_category, products["product_category"], "product_category"),
        }
        artifact, cache_hit = self._demand_artifact(products, dcs, force_retrain)
        mask = demand_mod.series_mask(artifact, filters)
        if not mask.any():
            raise InvalidRequestError(f"No demand history matches filters {_active(filters)}.")

        frame = demand_mod.demand_forecast_frame(artifact, mask, horizon)
        origin = artifact.payload["origin_date"]
        run_id = _new_run_id()
        persistence = self._persist(persist, "demand_forecast", frame.assign(
            run_id=run_id, model_name=artifact.model_name, model_version=artifact.version, forecast_origin_date=origin,
        ))
        summary = frame.groupby("forecast_date", as_index=False)["predicted_demand_units"].sum().round(2)
        return {
            "type": "demand_forecast",
            "model": artifact.model_name,
            "model_version": artifact.version,
            "horizon": horizon,
            "forecast_origin_date": origin,
            "filters": _active(filters),
            "series_count": int(mask.sum()),
            "metrics": demand_mod.demand_test_metrics(artifact, mask, horizon),
            "summary_by_date": summary,
            **self._data_block(frame, max_rows),
            "run_id": run_id,
            "persistence": persistence,
            "model_info": self._model_info(artifact, cache_hit),
        }

    def forecast_inbound(self, po_id: str | None = None, vendor_id: str | None = None, dc_id: str | None = None,
                         sku_id: str | None = None, horizon: int = 30, persist: bool = True,
                         force_retrain: bool = False, max_rows: int = DEFAULT_MAX_ROWS) -> dict[str, Any]:
        horizon = _validate_int(horizon, "horizon", 1, MAX_HORIZON)
        max_rows = _validate_int(max_rows, "max_rows", 0, MAX_ROWS_LIMIT)
        artifact, cache_hit, as_of = self._inbound_artifact(force_retrain)
        vendors, products, dcs = self.db.get_vendor_master(), self.db.get_product_master(), self.db.get_dc_master()
        predictions: pd.DataFrame = artifact.payload["open_po_forecast"]
        filters = {
            "po_id": self._resolve_po(po_id, predictions, as_of),
            "vendor_id": _match(vendor_id, vendors["vendor_id"], "vendor_id"),
            "dc_id": _match(dc_id, dcs["dc_id"], "dc_id"),
            "sku_id": _match(sku_id, products["sku_id"], "sku_id"),
        }
        frame = _apply_filters(predictions, filters)
        frame = frame[frame["horizon_days"] <= horizon].reset_index(drop=True)

        run_id = _new_run_id()
        persistence = self._persist(persist, "inbound_forecast", frame.rename(columns={
            "expected_quantity": "predicted_received_qty", "expected_fill_rate": "predicted_fill_rate",
            "expected_delay_days": "predicted_delay_days", "expected_arrival_date": "predicted_arrival_date",
        }).assign(run_id=run_id, model_name=artifact.model_name, model_version=artifact.version,
                  forecast_origin_date=as_of))
        weekly = (
            frame.groupby(["week_start_date", "dc_id"], as_index=False)
            .agg(expected_quantity=("expected_quantity", "sum"), po_count=("po_id", "count"))
            .round({"expected_quantity": 1})
        )
        meta = artifact.metadata
        response = {
            "type": "inbound_forecast",
            "model": artifact.model_name,
            "model_version": artifact.version,
            "horizon": horizon,
            "forecast_origin_date": as_of,
            "filters": _active(filters),
            "open_po_count": int(len(frame)),
            "metrics": {
                "quantity": self._selected_test_metrics(meta["quantity_model"]),
                "delay_days": self._selected_test_metrics(meta["delay_model"]),
                "on_time_method": meta["on_time_method"],
            },
            "weekly_expected_receipts": weekly,
            **self._data_block(frame, max_rows),
            "run_id": run_id,
            "persistence": persistence,
            "model_info": self._model_info(artifact, cache_hit),
        }
        if frame.empty:
            response["message"] = f"No open POs are expected to arrive within {horizon} days of {as_of.date()}."
        return response

    def forecast_labor(self, dc_id: str | None = None, process: str | None = None, shift: str | None = None,
                       horizon: int = 14, persist: bool = True, force_retrain: bool = False,
                       max_rows: int = DEFAULT_MAX_ROWS) -> dict[str, Any]:
        horizon = _validate_int(horizon, "horizon", 1, MAX_HORIZON)
        max_rows = _validate_int(max_rows, "max_rows", 0, MAX_ROWS_LIMIT)
        products, dcs = self.db.get_product_master(), self.db.get_dc_master()
        demand_art, demand_hit = self._demand_artifact(products, dcs, force_retrain)
        inbound_art, inbound_hit, as_of = self._inbound_artifact(force_retrain)
        labor_art, labor_hit = self._labor_artifact(demand_art, inbound_art, as_of, force_retrain)

        params = labor_art.payload["shift_params"]
        filters = {
            "dc_id": _match(dc_id, dcs["dc_id"], "dc_id"),
            "process": _match(process, params["process"].unique(), "process"),
            "shift": _match(shift, params["shift"].unique(), "shift"),
        }
        all_series = np.ones(len(demand_art.payload["static"]), dtype=bool)
        demand_by_dc = (
            demand_mod.demand_forecast_frame(demand_art, all_series, horizon)
            .groupby(["forecast_date", "dc_id"], as_index=False)["predicted_demand_units"].sum()
            .rename(columns={"predicted_demand_units": "units"})
        )
        open_receipts = (
            inbound_art.payload["open_po_forecast"]
            .groupby(["expected_arrival_date", "dc_id"], as_index=False)["expected_quantity"].sum()
            .rename(columns={"expected_arrival_date": "forecast_date", "expected_quantity": "units"})
        )
        frame = _apply_filters(labor_mod.plan_labor(labor_art, demand_by_dc, open_receipts, as_of, horizon), filters)
        if frame.empty:
            raise InvalidRequestError(f"No labor history matches filters {_active(filters)}.")
        frame = frame.reset_index(drop=True)

        run_id = _new_run_id()
        persistence = self._persist(persist, "labor_forecast", frame.assign(
            run_id=run_id, model_name=labor_art.model_name, model_version=labor_art.version, forecast_origin_date=as_of,
        ))
        summary = (
            frame.groupby(["forecast_date", "dc_id", "process"], as_index=False)[
                ["predicted_workload_units", "required_labor_hours", "adjusted_labor_hours",
                 "required_headcount", "available_headcount"]
            ].sum().round(2)
        )
        validation = labor_art.metadata["validation"]
        return {
            "type": "labor_forecast",
            "model": labor_art.model_name,
            "model_version": labor_art.version,
            "horizon": horizon,
            "forecast_origin_date": as_of,
            "filters": _active(filters),
            "metrics": {
                "workload": validation.get("overall", {}).get("driver_ratio"),
                "baseline_seasonal_naive": validation.get("overall", {}).get("seasonal_naive"),
                "by_process": validation.get("by_process"),
                "test_period": validation.get("test_period"),
                "note": "Workload accuracy given actual drivers; end-to-end error also includes demand/inbound forecast error.",
            },
            "summary_by_date_dc_process": summary,
            **self._data_block(frame, max_rows),
            "run_id": run_id,
            "persistence": persistence,
            "model_info": {
                **self._model_info(labor_art, labor_hit),
                "upstream": {
                    "demand": {"model_version": demand_art.version, "from_cache": demand_hit},
                    "inbound": {"model_version": inbound_art.version, "from_cache": inbound_hit},
                },
            },
        }

    # ================= model artifacts =================
    def _demand_artifact(self, products: pd.DataFrame, dcs: pd.DataFrame, force: bool) -> tuple[ModelArtifact, bool]:
        signature = self.db.get_table_signature("demand_history", "date")
        if signature[1] == 0:
            raise InsufficientDataError("fl.demand_history is empty.")

        def train() -> ModelArtifact:
            panel = demand_mod.build_panel(self.db.get_demand_history(), products, dcs, self.db.get_calendar())
            return demand_mod.train_demand_model(panel, signature)

        return self.cache.get_or_train("demand", signature, train, force)

    def _inbound_artifact(self, force: bool) -> tuple[ModelArtifact, bool, pd.Timestamp]:
        inbound_sig = self.db.get_table_signature("inbound_history", "po_date")
        if inbound_sig[1] == 0:
            raise InsufficientDataError("fl.inbound_history is empty.")
        demand_sig = self.db.get_table_signature("demand_history", "date")
        # Planning as-of date = last day of demand history (falls back to the last PO date).
        as_of = pd.Timestamp(demand_sig[0] or inbound_sig[0])
        signature = (inbound_sig, demand_sig)

        def train() -> ModelArtifact:
            inbound = self.db.get_inbound_history()
            vendors, products = self.db.get_vendor_master(), self.db.get_product_master()
            calendar = self.db.get_calendar()
            demand_daily = self.db.get_demand_history()[["date", "dc_id", "sku_id", "demand_units"]]
            artifact = inbound_mod.train_inbound_models(inbound, vendors, products, demand_daily, calendar, as_of, signature)
            artifact.payload["open_po_forecast"] = inbound_mod.predict_open_pos(
                artifact, inbound, vendors, products, demand_daily, calendar, as_of
            )
            return artifact

        artifact, hit = self.cache.get_or_train("inbound", signature, train, force)
        return artifact, hit, as_of

    def _labor_artifact(self, demand_art: ModelArtifact, inbound_art: ModelArtifact, as_of: pd.Timestamp,
                        force: bool) -> tuple[ModelArtifact, bool]:
        labor_sig = self.db.get_table_signature("labor_operations_history", "date")
        if labor_sig[1] == 0:
            raise InsufficientDataError("fl.labor_operations_history is empty.")
        signature = (labor_sig, demand_art.version, inbound_art.version)

        def train() -> ModelArtifact:
            arrived = inbound_mod.arrived_pos(self.db.get_inbound_history(), as_of)
            return labor_mod.fit_labor_plan(
                self.db.get_labor_history(), demand_art.payload["dc_daily_demand"], arrived,
                inbound_art.payload["cycle_days"], as_of, signature,
                upstream={"demand": demand_art.version, "inbound": inbound_art.version},
            )

        return self.cache.get_or_train("labor", signature, train, force)

    # ================= helpers =================
    @staticmethod
    def _resolve_po(po_id: str | None, predictions: pd.DataFrame, as_of: pd.Timestamp) -> str | None:
        if po_id is None or str(po_id).strip() == "":
            return None
        key = str(po_id).strip().upper()
        if key in set(predictions["po_id"].astype(str).str.upper()):
            return key
        raise InvalidRequestError(
            f"PO '{po_id}' is not an open PO as of {as_of.date()} (unknown, already arrived, or placed later). "
            "Only open POs are forecast."
        )

    @staticmethod
    def _selected_test_metrics(meta: dict[str, Any]) -> dict[str, Any]:
        selected = meta["selected_model"]
        return {
            **meta["test_metrics"][selected],
            "model": selected,
            "evaluation": "chronological hold-out test period (latest POs by po_date)",
            "baseline_historical_mean": meta["test_metrics"]["historical_baseline"],
        }

    @staticmethod
    def _data_block(frame: pd.DataFrame, max_rows: int) -> dict[str, Any]:
        return {
            "data": frame.head(max_rows),
            "total_rows": int(len(frame)),
            "returned_rows": int(min(len(frame), max_rows)),
            "truncated": bool(len(frame) > max_rows),
        }

    @staticmethod
    def _model_info(artifact: ModelArtifact, cache_hit: bool) -> dict[str, Any]:
        return {
            "model_name": artifact.model_name,
            "model_version": artifact.version,
            "trained_at": artifact.trained_at,
            "from_cache": cache_hit,
            "data_signature": artifact.data_signature,
            **artifact.metadata,
        }

    def _persist(self, persist: bool, table: str, frame: pd.DataFrame) -> dict[str, Any]:
        target = f"{self.db.schema}.{table}"
        if not persist:
            return {"status": "skipped", "table": target}
        try:
            return {"status": "written", "table": target, "rows_written": self.db.insert_forecast_frame(table, frame)}
        except ForecastError as exc:
            logger.error("Persisting %s failed: %s", target, exc)
            return {"status": "failed", "table": target, "message": str(exc)}