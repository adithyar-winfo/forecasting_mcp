"""Foot Locker Supply Chain Forecasting MCP server (demand, inbound, labor)."""
import logging
import sys
from collections.abc import Callable
from typing import Any

from fastmcp import FastMCP

try:
    from .config import DEFAULT_MAX_ROWS, SCHEMA, get_db_config
    from .db_adapter import SupplyChainDBAdapter
    from .db_service import DBService
    from .models import DataSourceError, ForecastError
    from .service import SupplyChainForecastService, to_jsonable
except ImportError:
    from config import DEFAULT_MAX_ROWS, SCHEMA, get_db_config
    from db_adapter import SupplyChainDBAdapter
    from db_service import DBService
    from models import DataSourceError, ForecastError
    from service import SupplyChainForecastService, to_jsonable

# stdout carries the MCP stdio protocol, so logs must go to stderr.
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("forecasting_mcp")

mcp = FastMCP("forecasting_mcp")
_service: SupplyChainForecastService | None = None


def _get_service() -> SupplyChainForecastService:
    global _service
    if _service is None:
        config = get_db_config()
        try:
            db_service = DBService(config)
        except Exception as exc:
            raise DataSourceError(f"Could not initialize database service: {exc}") from exc
        _service = SupplyChainForecastService(SupplyChainDBAdapter(db_service, config, SCHEMA))
    return _service


def _run(tool: str, call: Callable[[SupplyChainForecastService], dict[str, Any]]) -> dict[str, Any]:
    try:
        return to_jsonable(call(_get_service()))
    except ForecastError as exc:
        logger.warning("%s failed (%s): %s", tool, exc.error_type, exc)
        return {"type": "error", "tool": tool, "error_type": exc.error_type, "message": str(exc)}
    except Exception as exc:
        logger.exception("%s failed unexpectedly", tool)
        return {"type": "error", "tool": tool, "error_type": "internal_error", "message": f"Unexpected error: {exc}"}


@mcp.tool()
def forecast_demand(
    dc_id: str | None = None,
    sku_id: str | None = None,
    product_category: str | None = None,
    horizon: int = 30,
    persist: bool = True,
    force_retrain: bool = False,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> dict[str, Any]:
    """Forecast daily demand units (Date x DC_ID x SKU_ID) from fl.demand_history.

    Uses a global LightGBM model with lag (1/7/14/28-day), rolling (7/14/28-day mean, 7/28-day std),
    calendar, price/promotion and product/DC features, forecast recursively day by day. The model is
    selected against a seasonal-naive baseline on a chronological validation period.

    Args:
        dc_id: Optional distribution center filter, e.g. "DC01". Omit for all DCs.
        sku_id: Optional SKU filter, e.g. "SKU001". Omit for all SKUs.
        product_category: Optional category filter, e.g. "Footwear", "Apparel". Omit for all categories.
        horizon: Days ahead to forecast, 1-90 (typical: 7, 14, 30, 60, 90). Default 30.
        persist: Write the forecast rows to fl.demand_forecast (default True).
        force_retrain: Retrain even if a cached model for the current data exists.
        max_rows: Max forecast rows returned in `data` (all rows are still persisted). Default 200.

    Returns:
        {"type": "demand_forecast", "model", "model_version", "horizon", "forecast_origin_date", "filters",
         "series_count", "metrics": {"mae", "rmse", "wape", "bias", ...hold-out test metrics},
         "summary_by_date": [{forecast_date, predicted_demand_units}],
         "data": [{forecast_date, horizon_days, dc_id, sku_id, product_category,
                   predicted_demand_units, lower_bound, upper_bound}],
         "total_rows", "truncated", "run_id", "persistence", "model_info"}
        On failure: {"type": "error", "error_type", "message"}.
    """
    return _run("forecast_demand", lambda s: s.forecast_demand(
        dc_id=dc_id, sku_id=sku_id, product_category=product_category, horizon=horizon,
        persist=persist, force_retrain=force_retrain, max_rows=max_rows,
    ))


@mcp.tool()
def forecast_inbound(
    po_id: str | None = None,
    vendor_id: str | None = None,
    dc_id: str | None = None,
    sku_id: str | None = None,
    horizon: int = 30,
    persist: bool = True,
    force_retrain: bool = False,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> dict[str, Any]:
    """Forecast expected received quantity and arrival date / delay for open purchase orders.

    Open POs are those placed on or before the planning as-of date (last day of demand history) that
    have not yet arrived. PO-level LightGBM models (selected against vendor historical-mean baselines)
    predict the fill rate (-> expected quantity = ordered_qty x fill rate) and delay days
    (-> expected arrival = promised arrival + delay), using vendor, freight mode, lead time, DC, SKU,
    vendor/lane history known at PO time, recent demand and calendar features.

    Args:
        po_id: Optional single open PO, e.g. "PO000286".
        vendor_id: Optional vendor filter, e.g. "V002".
        dc_id: Optional DC filter, e.g. "DC01".
        sku_id: Optional SKU filter, e.g. "SKU002".
        horizon: Only include POs expected to arrive within this many days, 1-90. Default 30.
        persist: Write results to fl.inbound_forecast (default True).
        force_retrain: Retrain even if a cached model for the current data exists.
        max_rows: Max PO rows returned in `data`. Default 200.

    Returns:
        {"type": "inbound_forecast", "model", "model_version", "horizon", "forecast_origin_date", "filters",
         "open_po_count", "metrics": {"quantity": {...}, "delay_days": {...}, "on_time_method"},
         "weekly_expected_receipts": [{week_start_date, dc_id, expected_quantity, po_count}],
         "data": [{po_id, po_date, vendor_id, dc_id, sku_id, freight_mode, ordered_qty, promised_arrival_date,
                   expected_fill_rate, expected_quantity, expected_delay_days, expected_arrival_date,
                   on_time_probability, delay_probability, overdue, horizon_days, week_start_date}],
         "total_rows", "truncated", "run_id", "persistence", "model_info"}
        On failure: {"type": "error", "error_type", "message"}.
    """
    return _run("forecast_inbound", lambda s: s.forecast_inbound(
        po_id=po_id, vendor_id=vendor_id, dc_id=dc_id, sku_id=sku_id, horizon=horizon,
        persist=persist, force_retrain=force_retrain, max_rows=max_rows,
    ))


@mcp.tool()
def forecast_labor(
    dc_id: str | None = None,
    process: str | None = None,
    shift: str | None = None,
    horizon: int = 14,
    persist: bool = True,
    force_retrain: bool = False,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> dict[str, Any]:
    """Forecast workload, required labor hours and headcount by Date x DC x Process x Shift.

    Chain: demand forecast (Picking, Packing, Returns) and inbound receipts forecast (Receiving)
    -> workload via historical workload-per-driver ratios -> split by shift -> labor hours
    = workload / expected productivity -> absence-adjusted hours -> required headcount, compared
    with recently available headcount.

    Args:
        dc_id: Optional DC filter, e.g. "DC01".
        process: Optional process filter: "Receiving", "Picking", "Packing" or "Returns".
        shift: Optional shift filter, e.g. "Shift_A" or "Shift_B".
        horizon: Days ahead, 1-90 (typical: 7, 14, 30). Default 14.
        persist: Write results to fl.labor_forecast (default True).
        force_retrain: Rebuild demand, inbound and labor models even if cached.
        max_rows: Max rows returned in `data`. Default 200.

    Returns:
        {"type": "labor_forecast", "model", "model_version", "horizon", "forecast_origin_date", "filters",
         "metrics": {"workload": {mae, rmse, wape, bias}, "baseline_seasonal_naive", "by_process", ...},
         "summary_by_date_dc_process": [...],
         "data": [{forecast_date, horizon_days, dc_id, process, shift, driver, driver_units,
                   predicted_workload_units, predicted_productivity_uplh, required_labor_hours, absence_rate,
                   expected_absence_count, adjusted_labor_hours, required_headcount, available_headcount,
                   headcount_gap}],
         "total_rows", "truncated", "run_id", "persistence", "model_info"}
        On failure: {"type": "error", "error_type", "message"}.
    """
    return _run("forecast_labor", lambda s: s.forecast_labor(
        dc_id=dc_id, process=process, shift=shift, horizon=horizon,
        persist=persist, force_retrain=force_retrain, max_rows=max_rows,
    ))


if __name__ == "__main__":
    mcp.run()