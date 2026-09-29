"""Configuration for the Foot Locker supply-chain forecasting MCP."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(Path(__file__).with_name(".env"))

SCHEMA = os.getenv("FL_SCHEMA", "fl")
if not re.fullmatch(r"[a-z_][a-z0-9_]*", SCHEMA):
    raise ValueError(f"Invalid FL_SCHEMA value: {SCHEMA!r}")

MAX_HORIZON = 90
RECOMMENDED_HORIZONS = (7, 14, 30, 60, 90)
DEFAULT_MAX_ROWS = 200
MAX_ROWS_LIMIT = 5000


def get_db_config() -> dict[str, Any]:
    """Return the shared DB config from the `db_mcp` package.

    Forecasting MCP should not embed or override database-specific configuration.
    The shared `db_mcp` config is the
    single source of truth for connection details.
    """
    from mcp_servers.db_mcp.config import config

    return dict(config)
