"""Calendar feature engineering shared by the demand, inbound and labor forecasts."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CALENDAR_FEATURES = [
    "day_of_week", "day_of_month", "week", "month", "quarter",
    "is_weekend", "month_end", "holiday_flag", "peak_season",
]
_FLAG_COLUMNS = ["holiday_flag", "peak_season"]


def build_calendar_features(dates: Any, calendar: pd.DataFrame) -> pd.DataFrame:
    """Deterministic date features plus holiday/peak flags from fl.calendar.

    Dates beyond fl.calendar reuse the latest year's flag for the same month/day.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    out = pd.DataFrame(index=idx)
    out["day_of_week"] = idx.dayofweek
    out["day_of_month"] = idx.day
    out["week"] = idx.isocalendar().week.to_numpy().astype(int)
    out["month"] = idx.month
    out["quarter"] = idx.quarter
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["month_end"] = idx.is_month_end.astype(int)

    if calendar.empty:
        for col in _FLAG_COLUMNS:
            out[col] = 0
        return out

    cal = calendar.assign(date=pd.to_datetime(calendar["date"]))
    known = cal.set_index("date")[_FLAG_COLUMNS].reindex(idx)
    by_month_day = (
        cal.assign(_m=cal["date"].dt.month, _d=cal["date"].dt.day)
        .sort_values("date")
        .drop_duplicates(["_m", "_d"], keep="last")
        .set_index(["_m", "_d"])[_FLAG_COLUMNS]
    )
    fallback = by_month_day.reindex(pd.MultiIndex.from_arrays([idx.month, idx.day]))
    for col in _FLAG_COLUMNS:
        k = known[col].to_numpy(dtype=float)
        f = fallback[col].to_numpy(dtype=float)
        out[col] = np.nan_to_num(np.where(np.isnan(k), f, k)).astype(int)
    return out
