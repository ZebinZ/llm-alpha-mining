from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def local_session_dates_yyyymmdd(
    timestamps: pd.Series,
    *,
    timezone: str,
) -> pd.Series:
    """Format local calendar dates without per-row ``strftime`` calls."""

    values = pd.Series(timestamps, copy=False)
    if not isinstance(values.dtype, pd.DatetimeTZDtype):
        raise ValueError("session-date timestamps must be timezone-aware")
    local = values.dt.tz_convert(timezone)
    valid = local.notna().to_numpy(dtype=bool)
    result: np.ndarray[Any, np.dtype[np.object_]] = np.empty(
        len(local),
        dtype=object,
    )
    result[:] = np.nan
    if valid.any():
        years = local.dt.year.to_numpy()[valid].astype(np.int64, copy=False)
        months = local.dt.month.to_numpy()[valid].astype(np.int64, copy=False)
        days = local.dt.day.to_numpy()[valid].astype(np.int64, copy=False)
        encoded = years * 10_000 + months * 100 + days
        result[valid] = encoded.astype(str)
    return pd.Series(
        result,
        index=values.index,
        name=values.name,
        dtype=object,
    )


__all__ = ["local_session_dates_yyyymmdd"]
