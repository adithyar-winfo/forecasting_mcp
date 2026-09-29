"""Local database service for standalone forecasting MCP deployments."""
from __future__ import annotations

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor


class PostgresClient:
    """Minimal PostgreSQL client used by forecasting_mcp when deployed standalone."""

    def __init__(self, host: str, dbname: str, user: str, password: str, port: int = 5432) -> None:
        self.conn = psycopg2.connect(
            host=host,
            dbname=dbname,
            user=user,
            password=password,
            port=port,
        )


class DBService:
    """Simple query service that returns pandas DataFrames."""

    def __init__(self, config: dict) -> None:
        self.client = PostgresClient(**config)

    def run_query(self, query: str, params=None) -> pd.DataFrame:
        conn = self.client.conn
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            if cur.description is None:
                return pd.DataFrame()
            rows = cur.fetchall()
        return pd.DataFrame(rows)
