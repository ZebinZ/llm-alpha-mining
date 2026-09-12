from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ExecutionLabelPolicy:
    """Immutable timing and execution contract for weekly research labels.

    The signal is allowed to consume the complete signal-date close.  It is
    therefore not executable at that close.  The first permissible fill is the
    following trading session's official open, and the position is valued out
    at the official open following the next rebalance signal.
    """

    policy_id: str = "v5_next_open_to_next_rebalance_open_v1"
    schema_version: str = "execution-label-policy/v1"
    timezone: str = "Asia/Shanghai"
    information_cutoff_event: str = "signal_session_close"
    information_cutoff_clock: str = "15:00:00+08:00"
    order_event: str = "next_session_open_call_auction"
    order_clock: str = "09:15:00+08:00"
    entry_event: str = "next_session_official_open"
    entry_clock: str = "09:30:00+08:00"
    exit_event: str = "next_rebalance_next_session_official_open"
    exit_clock: str = "09:30:00+08:00"
    entry_delay_sessions: int = 1
    price_field: str = "frdata.stock_open"
    adjustment_field: str = "frdata.stock_adj"
    adjusted_price_formula: str = "stock_open * stock_adj"
    holding_interval: str = "half_open_[entry_open,exit_open)"
    missing_execution_observation: str = "invalidate_label"
    explicit_suspension: str = "invalidate_label_no_delayed_fill"
    amount_requirement: str = "finite_and_strictly_positive_on_entry_and_exit"
    amount_role: str = "ex_post_execution_validity_only_never_signal_universe"
    unexplained_missing_fill: str = "forbidden"
    price_limit_fill_assumption: str = "not_modelled_research_label_only"

    def __post_init__(self) -> None:
        if self.entry_delay_sessions != 1:
            raise ValueError("next-open policy requires entry_delay_sessions=1")
        if self.missing_execution_observation != "invalidate_label":
            raise ValueError("execution labels must fail closed on missing data")
        if self.unexplained_missing_fill != "forbidden":
            raise ValueError("execution labels may not fill unknown observations")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ExecutionLabelResult:
    period_returns: pd.DataFrame
    label_windows: pd.DataFrame
    execution_validity: pd.DataFrame
    diagnostics: pd.DataFrame
    policy_id: str
    policy_digest: str


_WINDOW_COLUMNS = (
    "signal_date",
    "information_cutoff",
    "order_time",
    "entry_date",
    "entry_time",
    "entry_price_field",
    "next_signal_date",
    "exit_date",
    "exit_time",
    "exit_price_field",
    "holding_interval",
    "label_start",
    "label_end",
)


def build_execution_label_windows(
    signal_dates: Sequence[object],
    calendar_dates: Sequence[object],
    *,
    policy: ExecutionLabelPolicy | None = None,
) -> pd.DataFrame:
    """Create non-overlapping next-open weekly execution intervals."""

    active_policy = policy or ExecutionLabelPolicy()
    calendar = _ordered_normalized_dates(calendar_dates, name="calendar")
    signals = _ordered_normalized_dates(signal_dates, name="signal")
    if len(signals) < 2:
        return pd.DataFrame(columns=_WINDOW_COLUMNS)

    calendar_position = {date: index for index, date in enumerate(calendar)}
    missing_signals = [date for date in signals if date not in calendar_position]
    if missing_signals:
        raise ValueError("signal_dates_outside_calendar:" + ",".join(missing_signals))

    rows: list[dict[str, str]] = []
    delay = active_policy.entry_delay_sessions
    for signal_date, next_signal_date in zip(signals[:-1], signals[1:], strict=True):
        entry_position = calendar_position[signal_date] + delay
        exit_position = calendar_position[next_signal_date] + delay
        if entry_position >= len(calendar) or exit_position >= len(calendar):
            raise ValueError(
                f"execution_boundary_outside_calendar:{signal_date}:{next_signal_date}"
            )
        entry_date = calendar[entry_position]
        exit_date = calendar[exit_position]
        if entry_date >= exit_date:
            raise ValueError(
                f"execution_window_not_positive:{signal_date}:{entry_date}:{exit_date}"
            )
        rows.append(
            {
                "signal_date": signal_date,
                "information_cutoff": _timestamp(
                    signal_date,
                    active_policy.information_cutoff_clock,
                ),
                "order_time": _timestamp(entry_date, active_policy.order_clock),
                "entry_date": entry_date,
                "entry_time": _timestamp(entry_date, active_policy.entry_clock),
                "entry_price_field": active_policy.adjusted_price_formula,
                "next_signal_date": next_signal_date,
                "exit_date": exit_date,
                "exit_time": _timestamp(exit_date, active_policy.exit_clock),
                "exit_price_field": active_policy.adjusted_price_formula,
                "holding_interval": active_policy.holding_interval,
                "label_start": entry_date,
                "label_end": exit_date,
            }
        )
    windows = pd.DataFrame(rows, columns=_WINDOW_COLUMNS)
    validate_execution_label_windows(windows)
    return windows


def compute_execution_aware_weekly_returns(
    open_prices: pd.DataFrame,
    adjustment_factors: pd.DataFrame,
    amounts: pd.DataFrame,
    explicit_suspension: pd.DataFrame,
    calendar_dates: Sequence[object],
    signal_dates: Sequence[object],
    *,
    signal_universe: pd.DataFrame | None = None,
    policy: ExecutionLabelPolicy | None = None,
) -> ExecutionLabelResult:
    """Compute adjusted open-to-open returns under a fail-closed fill policy.

    ``signal_universe`` is used only to scope diagnostics.  It never decides
    whether a raw return is computed, so future execution availability cannot
    leak into signal-date membership.
    """

    active_policy = policy or ExecutionLabelPolicy()
    windows = build_execution_label_windows(
        signal_dates,
        calendar_dates,
        policy=active_policy,
    )
    if windows.empty:
        empty = pd.DataFrame(index=pd.Index([], name="signal_date"))
        return ExecutionLabelResult(
            period_returns=empty.copy(),
            label_windows=windows,
            execution_validity=empty.astype(bool),
            diagnostics=pd.DataFrame(index=pd.Index([], name="signal_date")),
            policy_id=active_policy.policy_id,
            policy_digest=active_policy.digest,
        )

    required_dates = list(
        dict.fromkeys(windows["entry_date"].tolist() + windows["exit_date"].tolist())
    )
    opens = _normalize_numeric_panel(open_prices, required_dates, name="open_prices")
    adjustments = _normalize_numeric_panel(
        adjustment_factors,
        required_dates,
        name="adjustment_factors",
    )
    traded_amounts = _normalize_numeric_panel(
        amounts,
        required_dates,
        name="amounts",
    )
    suspensions = _normalize_boolean_panel(
        explicit_suspension,
        required_dates,
        name="explicit_suspension",
    )
    columns = (
        opens.columns.union(adjustments.columns, sort=False)
        .union(traded_amounts.columns, sort=False)
        .union(suspensions.columns, sort=False)
    )
    opens = opens.reindex(index=required_dates, columns=columns)
    adjustments = adjustments.reindex(index=required_dates, columns=columns)
    traded_amounts = traded_amounts.reindex(index=required_dates, columns=columns)
    suspensions = (
        suspensions.reindex(index=required_dates, columns=columns)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    universe = _normalize_optional_universe(
        signal_universe,
        windows["signal_date"].tolist(),
        columns,
    )

    return_rows: list[pd.Series] = []
    validity_rows: list[pd.Series] = []
    diagnostics: list[dict[str, object]] = []
    for window in windows.itertuples(index=False):
        signal_date = str(window.signal_date)
        entry_date = str(window.entry_date)
        exit_date = str(window.exit_date)
        entry_open = opens.loc[entry_date]
        exit_open = opens.loc[exit_date]
        entry_adjustment = adjustments.loc[entry_date]
        exit_adjustment = adjustments.loc[exit_date]
        entry_amount = traded_amounts.loc[entry_date]
        exit_amount = traded_amounts.loc[exit_date]
        entry_suspended = suspensions.loc[entry_date]
        exit_suspended = suspensions.loc[exit_date]

        entry_open_invalid = ~np.isfinite(entry_open) | (entry_open <= 0)
        exit_open_invalid = ~np.isfinite(exit_open) | (exit_open <= 0)
        entry_adjustment_invalid = ~np.isfinite(entry_adjustment) | (
            entry_adjustment <= 0
        )
        exit_adjustment_invalid = ~np.isfinite(exit_adjustment) | (exit_adjustment <= 0)
        entry_amount_invalid = ~np.isfinite(entry_amount) | (entry_amount <= 0)
        exit_amount_invalid = ~np.isfinite(exit_amount) | (exit_amount <= 0)
        validity = ~(
            entry_open_invalid
            | exit_open_invalid
            | entry_adjustment_invalid
            | exit_adjustment_invalid
            | entry_amount_invalid
            | exit_amount_invalid
            | entry_suspended
            | exit_suspended
        )
        entry_price = entry_open * entry_adjustment
        exit_price = exit_open * exit_adjustment
        period_return = (exit_price / entry_price - 1.0).where(validity)
        nonfinite_return = ~np.isfinite(period_return) & validity
        if nonfinite_return.any():
            validity = validity & ~nonfinite_return
            period_return = period_return.where(validity)

        scope = universe.loc[signal_date]
        return_rows.append(period_return)
        validity_rows.append(validity)
        diagnostics.append(
            {
                "signal_date": signal_date,
                "information_cutoff": str(window.information_cutoff),
                "order_time": str(window.order_time),
                "entry_date": entry_date,
                "entry_time": str(window.entry_time),
                "exit_date": exit_date,
                "exit_time": str(window.exit_time),
                "holding_interval": active_policy.holding_interval,
                "scope_stock_count": int(scope.sum()),
                "valid_label_count": int((scope & validity).sum()),
                "invalid_execution_count": int((scope & ~validity).sum()),
                "entry_open_invalid_count": int((scope & entry_open_invalid).sum()),
                "exit_open_invalid_count": int((scope & exit_open_invalid).sum()),
                "entry_adjustment_invalid_count": int(
                    (scope & entry_adjustment_invalid).sum()
                ),
                "exit_adjustment_invalid_count": int(
                    (scope & exit_adjustment_invalid).sum()
                ),
                "entry_amount_invalid_count": int((scope & entry_amount_invalid).sum()),
                "exit_amount_invalid_count": int((scope & exit_amount_invalid).sum()),
                "entry_suspended_count": int((scope & entry_suspended).sum()),
                "exit_suspended_count": int((scope & exit_suspended).sum()),
                "adjustment_changed_count": int(
                    (
                        scope
                        & ~entry_adjustment_invalid
                        & ~exit_adjustment_invalid
                        & (entry_adjustment != exit_adjustment)
                    ).sum()
                ),
                "nonfinite_return_count": int((scope & nonfinite_return).sum()),
            }
        )

    signal_index = pd.Index(windows["signal_date"].tolist(), name="signal_date")
    period_returns = pd.DataFrame(return_rows, index=signal_index, columns=columns)
    execution_validity = pd.DataFrame(
        validity_rows,
        index=signal_index,
        columns=columns,
    ).astype(bool)
    diagnostics_frame = pd.DataFrame(diagnostics).set_index("signal_date")
    return ExecutionLabelResult(
        period_returns=period_returns,
        label_windows=windows,
        execution_validity=execution_validity,
        diagnostics=diagnostics_frame,
        policy_id=active_policy.policy_id,
        policy_digest=active_policy.digest,
    )


def select_execution_labels_between(
    label_windows: pd.DataFrame,
    period_start: object,
    period_end: object,
) -> list[str]:
    """Select only labels whose full entry-to-exit interval is contained."""

    validate_execution_label_windows(label_windows)
    start = _normalize_date(period_start)
    end = _normalize_date(period_end)
    if start > end:
        raise ValueError("purge_period_reversed")
    selected = label_windows[
        (label_windows["entry_date"] >= start) & (label_windows["exit_date"] <= end)
    ]
    return selected["signal_date"].astype(str).tolist()


def validate_execution_label_windows(label_windows: pd.DataFrame) -> None:
    """Reject reversed or overlapping execution intervals."""

    required = {
        "signal_date",
        "entry_date",
        "exit_date",
        "label_start",
        "label_end",
    }
    missing = sorted(required.difference(label_windows.columns))
    if missing:
        raise ValueError("execution_windows_missing_columns:" + ",".join(missing))
    previous_exit: str | None = None
    for row in label_windows.itertuples(index=False):
        signal_date = _normalize_date(row.signal_date)
        entry_date = _normalize_date(row.entry_date)
        exit_date = _normalize_date(row.exit_date)
        if not signal_date < entry_date < exit_date:
            raise ValueError(
                f"invalid_execution_order:{signal_date}:{entry_date}:{exit_date}"
            )
        if _normalize_date(row.label_start) != entry_date:
            raise ValueError("label_start_must_equal_entry_date")
        if _normalize_date(row.label_end) != exit_date:
            raise ValueError("label_end_must_equal_exit_date")
        if previous_exit is not None and entry_date < previous_exit:
            raise ValueError(
                f"overlapping_execution_windows:{entry_date}:{previous_exit}"
            )
        previous_exit = exit_date


def _normalize_numeric_panel(
    frame: pd.DataFrame,
    dates: Sequence[str],
    *,
    name: str,
) -> pd.DataFrame:
    out = _normalize_panel_axes(frame, name=name).reindex(index=list(dates))
    return (
        out.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )


def _normalize_boolean_panel(
    frame: pd.DataFrame,
    dates: Sequence[str],
    *,
    name: str,
) -> pd.DataFrame:
    out = _normalize_panel_axes(frame, name=name).reindex(index=list(dates))
    observed = out.stack(future_stack=True).dropna()
    if not observed.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise TypeError(f"{name}_must_be_boolean")
    return out.astype("boolean").fillna(False).astype(bool)


def _normalize_optional_universe(
    frame: pd.DataFrame | None,
    dates: Sequence[str],
    columns: pd.Index,
) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(True, index=list(dates), columns=columns)
    out = _normalize_panel_axes(frame, name="signal_universe")
    observed = out.stack(future_stack=True).dropna()
    if not observed.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise TypeError("signal_universe_must_be_boolean")
    return (
        out.reindex(index=list(dates), columns=columns)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )


def _normalize_panel_axes(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    out = pd.DataFrame(frame).copy(deep=False)
    out.index = [_normalize_date(value) for value in out.index]
    out.columns = [str(value).zfill(6) for value in out.columns]
    if not out.index.is_unique or not out.columns.is_unique:
        raise ValueError(f"{name}_axes_must_be_unique")
    return out


def _ordered_normalized_dates(values: Sequence[object], *, name: str) -> list[str]:
    normalized = [_normalize_date(value) for value in values]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name}_dates_must_be_unique")
    if normalized != sorted(normalized):
        raise ValueError(f"{name}_dates_must_be_strictly_increasing")
    return normalized


def _normalize_date(value: object) -> str:
    compact = str(value).strip().replace("-", "").replace("/", "").replace(".", "")
    if len(compact) >= 8 and compact[:8].isdigit():
        return compact[:8]
    return pd.Timestamp(value).strftime("%Y%m%d")


def _timestamp(date: str, clock: str) -> str:
    return f"{date[:4]}-{date[4:6]}-{date[6:8]}T{clock}"


__all__ = [
    "ExecutionLabelPolicy",
    "ExecutionLabelResult",
    "build_execution_label_windows",
    "compute_execution_aware_weekly_returns",
    "select_execution_labels_between",
    "validate_execution_label_windows",
]
