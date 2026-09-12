from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


DENSE_SEMANTICS_V2 = "dense_numpy_v2_full_window_average_ties"
DENSE_SEMANTICS_V3 = (
    "dense_numpy_v3_exact_session_bars_minimum_coverage_quality_audit"
)
SUPPORTED_DENSE_SEMANTICS = frozenset({DENSE_SEMANTICS_V2, DENSE_SEMANTICS_V3})

SUPPORTED_INTRADAY_METHODS = frozenset(
    {
        "last_valid",
        "full_day_mean",
        "open_30m_mean",
        "am_mean",
        "midday_30m_mean",
        "post_lunch_30m_mean",
        "pm_mean",
        "close_30m_mean",
        "close_60m_mean",
    }
)


@dataclass(frozen=True, slots=True)
class IntradayAggregationPolicy:
    """Versioned intraday aggregation and data-quality contract.

    V2 is the immutable replay contract.  It keeps the historical inclusive
    clock masks and accepts a stock-day when at least one expression bar is
    finite.  V3 selects exact bar counts by position inside the AM/PM sessions
    and requires the configured fraction of bars that can be valid after the
    expression's declared rolling warm-up.

    Volume and stale-close thresholds are deliberately audit-only in this
    revision.  They are recorded so a later, separately frozen execution policy
    can choose gates without silently changing factor values.
    """

    semantics_version: str
    minimum_valid_bar_fraction: float
    audit_nonzero_volume: bool = True
    audit_stale_close: bool = True
    stale_close_alert_fraction: float = 0.80

    def __post_init__(self) -> None:
        if self.semantics_version not in SUPPORTED_DENSE_SEMANTICS:
            raise ValueError(
                f"unsupported dense semantics version: {self.semantics_version}"
            )
        for name, value in (
            ("minimum_valid_bar_fraction", self.minimum_valid_bar_fraction),
            ("stale_close_alert_fraction", self.stale_close_alert_fraction),
        ):
            number = float(value)
            if not np.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


LEGACY_V2_AGGREGATION_POLICY = IntradayAggregationPolicy(
    semantics_version=DENSE_SEMANTICS_V2,
    minimum_valid_bar_fraction=0.0,
    audit_nonzero_volume=False,
    audit_stale_close=False,
)

PRODUCTION_V3_AGGREGATION_POLICY = IntradayAggregationPolicy(
    semantics_version=DENSE_SEMANTICS_V3,
    minimum_valid_bar_fraction=0.80,
    audit_nonzero_volume=True,
    audit_stale_close=True,
    stale_close_alert_fraction=0.80,
)


def intraday_window_positions(
    minutes: np.ndarray,
    method: str,
    *,
    semantics_version: str,
) -> np.ndarray:
    """Return deterministic bar positions for one aggregation method.

    The V3 contract uses positions within each exchange session.  On the
    authoritative 241-row grid this maps open/midday/close windows to exactly
    30 bars and close-60 to exactly 60 bars, while never crossing the lunch
    break.  An underspecified source grid fails closed instead of shortening a
    window without disclosure.
    """

    name = str(method)
    if name not in SUPPORTED_INTRADAY_METHODS:
        raise ValueError(f"Unknown intraday aggregation method: {name}")
    version = str(semantics_version)
    if version not in SUPPORTED_DENSE_SEMANTICS:
        raise ValueError(f"unsupported dense semantics version: {version}")

    clock = np.asarray(minutes)
    if clock.ndim != 1:
        raise ValueError("minute axis must be one-dimensional")
    if len(clock) == 0:
        raise ValueError("minute axis must not be empty")
    if len(np.unique(clock)) != len(clock):
        raise ValueError("minute axis must be unique")
    if np.any(clock[1:] <= clock[:-1]):
        raise ValueError("minute axis must be strictly increasing")

    if version == DENSE_SEMANTICS_V2:
        return np.flatnonzero(_legacy_v2_clock_mask(clock, name))

    am = np.flatnonzero((clock >= 93000) & (clock <= 113000))
    pm = np.flatnonzero((clock >= 130000) & (clock <= 150000))
    full_session = np.concatenate((am, pm))
    if name in {"last_valid", "full_day_mean"}:
        positions = full_session
    elif name == "open_30m_mean":
        positions = _take_exact(am, 30, from_end=False, method=name)
    elif name == "am_mean":
        positions = am
    elif name == "midday_30m_mean":
        positions = _take_exact(am, 30, from_end=True, method=name)
    elif name == "post_lunch_30m_mean":
        positions = _take_exact(pm, 30, from_end=False, method=name)
    elif name == "pm_mean":
        positions = pm
    elif name == "close_30m_mean":
        positions = _take_exact(pm, 30, from_end=True, method=name)
    elif name == "close_60m_mean":
        positions = _take_exact(pm, 60, from_end=True, method=name)
    else:  # pragma: no cover - exhaustive guard above.
        raise AssertionError(name)
    if len(positions) == 0:
        raise ValueError(f"intraday window is empty: {name}")
    return positions.astype(np.int64, copy=False)


def consecutive_clock_pairs(minutes: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Identify adjacent selected bars that are one wall-clock minute apart."""

    clock = np.asarray(minutes)[np.asarray(positions, dtype=np.int64)]
    if len(clock) < 2:
        return np.zeros(0, dtype=bool)
    hours = clock.astype(np.int64) // 10000
    minute_of_hour = (clock.astype(np.int64) // 100) % 100
    minute_of_day = hours * 60 + minute_of_hour
    return np.diff(minute_of_day) == 1


def _take_exact(
    positions: np.ndarray,
    count: int,
    *,
    from_end: bool,
    method: str,
) -> np.ndarray:
    if len(positions) < int(count):
        raise ValueError(
            f"insufficient session bars for {method}: {len(positions)}<{int(count)}"
        )
    return positions[-int(count) :] if from_end else positions[: int(count)]


def _legacy_v2_clock_mask(minutes: np.ndarray, method: str) -> np.ndarray:
    if method in {"last_valid", "full_day_mean"}:
        return np.ones(len(minutes), dtype=bool)
    if method == "open_30m_mean":
        return (minutes >= 93000) & (minutes <= 100000)
    if method == "am_mean":
        return (minutes >= 93000) & (minutes <= 113000)
    if method == "midday_30m_mean":
        return (minutes >= 110000) & (minutes <= 113000)
    if method == "post_lunch_30m_mean":
        return (minutes >= 130000) & (minutes <= 133000)
    if method == "pm_mean":
        return minutes >= 130000
    if method == "close_30m_mean":
        return minutes >= 143000
    if method == "close_60m_mean":
        return minutes >= 140000
    raise ValueError(f"Unknown intraday aggregation method: {method}")


__all__ = [
    "DENSE_SEMANTICS_V2",
    "DENSE_SEMANTICS_V3",
    "IntradayAggregationPolicy",
    "LEGACY_V2_AGGREGATION_POLICY",
    "PRODUCTION_V3_AGGREGATION_POLICY",
    "SUPPORTED_DENSE_SEMANTICS",
    "SUPPORTED_INTRADAY_METHODS",
    "consecutive_clock_pairs",
    "intraday_window_positions",
]
