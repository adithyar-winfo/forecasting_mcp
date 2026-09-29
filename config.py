"""Configuration for the Foot Locker supply-chain forecasting MCP."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from dotenv import load_dotenv


_here = Path(__file__).resolve()
for candidate in [
    _here.parent / ".env",          # same folder
    _here.parent.parent / ".env",   # one level up
    Path.cwd() / ".env",            # current working dir
]:
    if candidate.exists():
        load_dotenv(candidate)

SCHEMA = os.getenv("FL_SCHEMA", "fl")
if not re.fullmatch(r"[a-z_][a-z0-9_]*", SCHEMA):
    raise ValueError(f"Invalid FL_SCHEMA value: {SCHEMA!r}")

MAX_HORIZON = 90
RECOMMENDED_HORIZONS = (7, 14, 30, 60, 90)
DEFAULT_MAX_ROWS = 200
MAX_ROWS_LIMIT = 5000


def get_db_config() -> dict[str, Any]:
    """Build DB config from environment for standalone deployments.

    Supports either `DATABASE_URL` or explicit environment variables:
    `POSTGRES_HOST/DB_HOST`, `POSTGRES_DATABASE/DB_NAME`,
    `POSTGRES_USERNAME/DB_USER`, `POSTGRES_PASSWORD/DB_PASSWORD`,
    `POSTGRES_PORT/DB_PORT`.
    """
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        parsed = urlparse(database_url)
        if not (parsed.scheme.startswith("postgres") or parsed.scheme.startswith("postgresql")):
            raise ValueError("DATABASE_URL must use postgres/postgresql scheme.")
        query = parse_qs(parsed.query)
        sslmode = query.get("sslmode", [None])[0]
        return {
            "host": parsed.hostname or "localhost",
            "dbname": (parsed.path or "").lstrip("/"),
            "user": unquote(parsed.username or ""),
            "password": unquote(parsed.password or ""),
            "port": int(parsed.port or 5432),
            **({"sslmode": sslmode} if sslmode else {}),
        }

    host = os.getenv("POSTGRES_HOST") or os.getenv("DB_HOST")
    dbname = os.getenv("POSTGRES_DATABASE") or os.getenv("DB_NAME")
    user = os.getenv("POSTGRES_USERNAME") or os.getenv("DB_USER")
    password = os.getenv("POSTGRES_PASSWORD") or os.getenv("DB_PASSWORD")
    port = int(os.getenv("POSTGRES_PORT") or os.getenv("DB_PORT") or "5432")

    missing = [
        name for name, value in {
            "POSTGRES_HOST/DB_HOST": host,
            "POSTGRES_DATABASE/DB_NAME": dbname,
            "POSTGRES_USERNAME/DB_USER": user,
            "POSTGRES_PASSWORD/DB_PASSWORD": password,
        }.items() if not value
    ]
    if missing:
        raise ValueError(f"Missing DB env vars: {', '.join(missing)}")

    return {
        "host": host,
        "dbname": dbname,
        "user": user,
        "password": password,
        "port": port,
    }
