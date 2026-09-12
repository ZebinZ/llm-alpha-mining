from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .intraday_semantics import (
    IntradayAggregationPolicy,
    LEGACY_V2_AGGREGATION_POLICY,
    intraday_window_positions,
)


def aggregate_intraday(
    minute_factors: Mapping[str, pd.DataFrame],
    method: str = "last_valid",
    *,
    aggregation_policy: IntradayAggregationPolicy = LEGACY_V2_AGGREGATION_POLICY,
    expression_warmup_bars: int = 0,
) -> pd.DataFrame:
    """Aggregate per-day minute frames under a versioned window contract.

    The default is the historical V2 behavior.  Callers opting into V3 receive
    exact session-position windows and the same minimum valid-bar rule as the
    dense V3 runner.  ``expression_warmup_bars`` declares leading positions
    that a rolling expression cannot populate and therefore excludes them from
    the coverage denominator.
    """

    if int(expression_warmup_bars) < 0:
        raise ValueError("expression_warmup_bars must be non-negative")
    rows = {}
    for date, frame in minute_factors.items():
        frame = frame.sort_index()
        positions = intraday_window_positions(
            frame.index.to_numpy(),
            method,
            semantics_version=aggregation_policy.semantics_version,
        )
        positions = positions[positions >= int(expression_warmup_bars)]
        if len(positions) == 0:
            if aggregation_policy.minimum_valid_bar_fraction > 0:
                raise ValueError(
                    f"no theoretically valid expression bars for {method}: "
                    f"warmup_bars={expression_warmup_bars}"
                )
            rows[date] = pd.Series(np.nan, index=frame.columns, dtype=float)
            continue
        selected = frame.iloc[positions]
        counts = selected.notna().sum(axis=0)
        minimum = max(
            1,
            int(
                math.ceil(
                    len(positions)
                    * float(aggregation_policy.minimum_valid_bar_fraction)
                )
            ),
        )
        if method == "last_valid":
            value = selected.ffill().iloc[-1]
        else:
            value = selected.mean(axis=0)
        rows[date] = value.where(counts >= minimum)

    daily = pd.DataFrame.from_dict(rows, orient="index")
    daily.index.name = "date"
    daily.columns.name = "stock"
    return daily.sort_index()


def aggregate_interday(
    daily_signal: pd.DataFrame,
    method: str = "mean",
    window: int = 5,
    normalize_before_agg: str | None = None,
) -> pd.DataFrame:
    signal = daily_signal.astype(float).sort_index()
    if normalize_before_agg == "rank":
        signal = signal.rank(axis=1, pct=True)
    elif normalize_before_agg == "zscore":
        signal = signal.sub(signal.mean(axis=1), axis=0).div(signal.std(axis=1), axis=0)
    elif normalize_before_agg not in (None, "none"):
        raise ValueError(f"Unknown normalization: {normalize_before_agg}")

    if method in ("last", "last_1d"):
        return signal
    if method == "mean":
        return signal.rolling(window=window, min_periods=window).mean()
    if method == "sum":
        return signal.rolling(window=window, min_periods=window).sum()
    if method == "std":
        return signal.rolling(window=window, min_periods=window).std()
    if method == "ema":
        return signal.ewm(span=window, min_periods=window, adjust=False).mean()
    raise ValueError(f"Unknown interday aggregation method: {method}")
