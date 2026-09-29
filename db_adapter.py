"""Database access for the supply-chain forecasting MCP, targeting the configured schema.

This module must not contain database-specific configuration. Connection
management and client construction are delegated to the shared `db_mcp` package
and its `DBService` instance passed into the adapter.
"""
from __future__ import annotations

import logging
import math
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from mcp_servers.db_mcp.service import DBService
from mcp_servers.forecasting_mcp.models import DataSourceError

logger = logging.getLogger("forecasting_mcp.db")

SOURCE_COLUMNS: dict[str, list[str]] = {
    "demand_history": [
        "date", "dc_id", "sku_id", "product_category", "orders", "demand_units", "unit_price_usd",
        "revenue_usd", "promo_flag", "promo_discount_pct", "stockout_flag",
    ],
    "inbound_history": [
        "po_id", "po_date", "vendor_id", "dc_id", "sku_id", "ordered_qty", "promised_arrival_date",
        "actual_arrival_date", "lead_time_days", "delay_days", "freight_mode", "received_qty",
        "fill_rate", "on_time_flag",
    ],
    "labor_operations_history": [
        "date", "dc_id", "shift", "process", "scheduled_headcount", "actual_headcount", "scheduled_hours",
        "actual_hours", "overtime_hours", "absence_count", "workload_units", "throughput_units",
        "productivity_units_per_labor_hour", "backlog_units", "utilization_pct",
    ],
    "calendar": ["date", "holiday_flag", "holiday_name", "peak_season"],
    "product_master": ["sku_id", "product_category", "subcategory", "base_price_usd", "unit_weight_kg", "unit_volume_m3"],
    "vendor_master": [
        "vendor_id", "vendor_name", "origin_region", "primary_mode", "base_lead_time_days",
        "lead_time_std_days", "daily_capacity_units", "on_time_rate", "fill_rate",
    ],
    "dc_master": [
        "dc_id", "dc_name", "region", "demand_factor", "labor_cost_factor", "storage_capacity_units",
        "receiving_capacity_units_day", "picking_capacity_units_day", "packing_capacity_units_day",
        "returns_capacity_units_day",
    ],
}

OUTPUT_COLUMNS: dict[str, list[str]] = {
    "demand_forecast": [
        "run_id", "model_name", "model_version", "forecast_origin_date", "forecast_date", "horizon_days",
        "dc_id", "sku_id", "predicted_demand_units", "lower_bound", "upper_bound",
    ],
    "inbound_forecast": [
        "run_id", "model_name", "model_version", "forecast_origin_date", "horizon_days", "po_id",
        "week_start_date", "vendor_id", "dc_id", "sku_id", "ordered_qty", "predicted_received_qty",
        "predicted_fill_rate", "predicted_delay_days", "predicted_arrival_date", "on_time_probability",
    ],
    "labor_forecast": [
        "run_id", "model_name", "model_version", "forecast_origin_date", "forecast_date", "horizon_days",
        "dc_id", "process", "shift", "predicted_workload_units", "predicted_productivity_uplh",
        "required_labor_hours", "expected_absence_count", "adjusted_labor_hours", "required_headcount",
    ],
}

_DATE_COLUMNS = {"date", "po_date", "promised_arrival_date", "actual_arrival_date"}


def _to_db_value(value: Any) -> Any:
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class SupplyChainDBAdapter:
    """Reads fl source tables and writes forecast results through the shared DBService connection."""

    def __init__(self, db_service: DBService, db_config: dict[str, Any], schema: str) -> None:
        self.db = db_service
        self._config = db_config
        self.schema = schema
        self._verified_outputs: set[str] = set()

    # ---------------- connection helpers ----------------
    def _connection(self):
        # Rely on the shared DBService to manage the underlying client/connection.
        # Forecasting MCP should not attempt to create or reconfigure DB clients.
        try:
            return self.db.client.conn
        except Exception:
            logger.warning("DB connection was not available via DBService.")
            raise

    def _read(self, sql: str, params: list | tuple | None = None) -> pd.DataFrame:
        conn = self._connection()
        try:
            df = self.db.run_query(sql, params)
            conn.commit()
        except Exception as exc:
            # Delegate to DataSourceError for upstream handling while avoiding
            # database-driver-specific exception types here.
            try:
                conn.rollback()
            except Exception:
                pass
            raise DataSourceError(f"Database query failed: {str(exc).strip()}") from exc
        for col in df.columns.intersection(list(_DATE_COLUMNS)):
            df[col] = pd.to_datetime(df[col])
        return df

    def _select(self, table: str, filters: dict[str, Any] | None = None, order_by: str | None = None) -> pd.DataFrame:
        clauses, params = [], []
        for col, value in (filters or {}).items():
            if value is not None:
                clauses.append(f"{col} = %s")
                params.append(value)
        sql = f"SELECT {', '.join(SOURCE_COLUMNS[table])} FROM {self.schema}.{table}"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        if order_by:
            sql += f" ORDER BY {order_by}"
        return self._read(sql, params or None)

    # ---------------- source tables ----------------
    def get_demand_history(self, dc_id: str | None = None, sku_id: str | None = None,
                           product_category: str | None = None) -> pd.DataFrame:
        return self._select("demand_history", {"dc_id": dc_id, "sku_id": sku_id, "product_category": product_category},
                            "date, dc_id, sku_id")

    def get_inbound_history(self, vendor_id: str | None = None, dc_id: str | None = None,
                            sku_id: str | None = None) -> pd.DataFrame:
        return self._select("inbound_history", {"vendor_id": vendor_id, "dc_id": dc_id, "sku_id": sku_id},
                            "po_date, po_id")

    def get_labor_history(self, dc_id: str | None = None, process: str | None = None,
                          shift: str | None = None) -> pd.DataFrame:
        return self._select("labor_operations_history", {"dc_id": dc_id, "process": process, "shift": shift},
                            "date, dc_id, process, shift")

    def get_calendar(self) -> pd.DataFrame:
        return self._select("calendar", order_by="date")

    def get_product_master(self) -> pd.DataFrame:
        return self._select("product_master", order_by="sku_id")

    def get_vendor_master(self) -> pd.DataFrame:
        return self._select("vendor_master", order_by="vendor_id")

    def get_dc_master(self) -> pd.DataFrame:
        return self._select("dc_master", order_by="dc_id")

    def get_table_signature(self, table: str, date_column: str) -> tuple[str | None, int]:
        """(max date, row count) used to decide whether cached models are stale."""
        if date_column not in SOURCE_COLUMNS.get(table, []):
            raise ValueError(f"Unsupported signature column {table}.{date_column}")
        df = self._read(f"SELECT MAX({date_column}) AS max_date, COUNT(*) AS row_count FROM {self.schema}.{table}")
        max_date = df["max_date"].iloc[0]
        return (None if max_date is None else pd.Timestamp(max_date).date().isoformat(), int(df["row_count"].iloc[0]))

    # ---------------- output tables ----------------
    def _verify_output_table(self, table: str) -> None:
        if table in self._verified_outputs:
            return
        df = self._read(
            "SELECT column_name FROM information_schema.columns WHERE table_schema = %s AND table_name = %s",
            (self.schema, table),
        )
        if df.empty:
            raise DataSourceError(f"Output table {self.schema}.{table} does not exist.")
        missing = sorted(set(OUTPUT_COLUMNS[table]) - set(df["column_name"]))
        if missing:
            raise DataSourceError(f"Output table {self.schema}.{table} is missing columns: {missing}")
        self._verified_outputs.add(table)

    def insert_forecast_frame(self, table: str, frame: pd.DataFrame) -> int:
        if table not in OUTPUT_COLUMNS:
            raise ValueError(f"Unsupported output table: {table}")
        if frame.empty:
            return 0
        self._verify_output_table(table)
        columns = OUTPUT_COLUMNS[table]
        rows = [tuple(_to_db_value(v) for v in row) for row in frame.reindex(columns=columns).itertuples(index=False)]
        sql = f"INSERT INTO {self.schema}.{table} ({', '.join(columns)}) VALUES %s"
        conn = self._connection()
        try:
            # Use the DBService-provided connection cursor for writes. The
            # underlying DB driver is managed by db_mcp; keep error handling
            # generic here so forecasting_mcp is driver-agnostic.
            with conn.cursor() as cur:
                # Bulk-insert: using simple executemany to avoid binding to a
                # specific DB driver helper in this package.
                placeholders = ", ".join(["%s"] * len(columns))
                insert_sql = f"INSERT INTO {self.schema}.{table} ({', '.join(columns)}) VALUES ({placeholders})"
                cur.executemany(insert_sql, rows)
            conn.commit()
        except Exception as exc:
            try:
                conn.rollback()
            except Exception:
                pass
            raise DataSourceError(f"Failed to write {self.schema}.{table}: {str(exc).strip()}") from exc
        return len(rows)