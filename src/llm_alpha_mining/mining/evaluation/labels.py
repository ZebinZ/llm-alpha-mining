from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForwardReturnPolicy:
    """Versioned policy for missing observations inside a holding window.

    ``invalidate_window`` is deliberately fail-closed: an unexplained missing
    daily return makes that stock's period label unavailable.  Zero-filling is
    only allowed for dates explicitly marked as suspended by point-in-time
    tradability data supplied to ``compute_forward_returns``.
    """

    policy_id: str = "v5_invalidate_unexplained_missing_v1"
    missing_observation: str = "invalidate_window"
    explicit_suspension_return: float = 0.0

    def __post_init__(self) -> None:
        if self.missing_observation != "invalidate_window":
            raise ValueError("V5 only supports fail-closed unexplained missing returns")


@dataclass(frozen=True)
class ForwardReturnResult:
    period_returns: pd.DataFrame
    diagnostics: pd.DataFrame
    policy_id: str


def compute_forward_returns(
    stock_returns: pd.DataFrame,
    calendar_dates: Sequence[str],
    label_windows: pd.DataFrame,
    *,
    policy: ForwardReturnPolicy | None = None,
    explicit_suspension: pd.DataFrame | None = None,
) -> ForwardReturnResult:
    """Compound returns without silently converting unknown data gaps to zero.

    ``label_windows`` must contain ``signal_date``, ``label_start`` and
    ``label_end``.  Membership decisions remain outside this function so future
    return availability can never define the signal-date universe.
    """

    active_policy = policy or ForwardReturnPolicy()
    required = {"signal_date", "label_start", "label_end"}
    missing_columns = required.difference(label_windows.columns)
    if missing_columns:
        raise ValueError(
            "label_windows_missing_columns:" + ",".join(sorted(missing_columns))
        )

    calendar = [str(item) for item in calendar_dates]
    returns = _normalize_panel(stock_returns).reindex(index=calendar)
    suspension = None
    if explicit_suspension is not None:
        suspension = (
            _normalize_panel(explicit_suspension)
            .reindex(index=calendar, columns=returns.columns)
            .fillna(False)
            .astype(bool)
        )

    rows: list[pd.Series] = []
    diagnostics: list[dict[str, object]] = []
    signal_dates: list[str] = []
    for window in label_windows.itertuples(index=False):
        signal_date = str(window.signal_date)
        start = str(window.label_start)
        end = str(window.label_end)
        if start not in returns.index or end not in returns.index:
            raise ValueError(
                f"label_window_outside_calendar:{signal_date}:{start}:{end}"
            )
        start_position = returns.index.get_loc(start)
        end_position = returns.index.get_loc(end)
        if not isinstance(start_position, int) or not isinstance(end_position, int):
            raise ValueError("calendar_index_must_be_unique")
        if end_position < start_position:
            raise ValueError(f"label_window_reversed:{signal_date}:{start}:{end}")

        window_returns = returns.iloc[start_position : end_position + 1].copy()
        original_missing = window_returns.isna()
        explained = pd.DataFrame(
            False,
            index=window_returns.index,
            columns=window_returns.columns,
        )
        if suspension is not None:
            suspension_window = suspension.iloc[start_position : end_position + 1]
            explained = original_missing & suspension_window
            window_returns = window_returns.mask(
                explained,
                float(active_policy.explicit_suspension_return),
            )

        unexplained = window_returns.isna()
        complete = ~unexplained.any(axis=0)
        compounded = (1.0 + window_returns).prod(
            axis=0, min_count=len(window_returns)
        ) - 1.0
        compounded = compounded.where(complete)
        rows.append(compounded)
        signal_dates.append(signal_date)
        diagnostics.append(
            {
                "signal_date": signal_date,
                "label_start": start,
                "label_end": end,
                "trading_day_count": int(len(window_returns)),
                "stock_count": int(window_returns.shape[1]),
                "valid_label_count": int(complete.sum()),
                "invalid_unexplained_missing_count": int((~complete).sum()),
                "explicit_suspension_fill_count": int(explained.to_numpy().sum()),
                "raw_missing_observation_count": int(original_missing.to_numpy().sum()),
            }
        )

    period_returns = pd.DataFrame(
        rows,
        index=pd.Index(signal_dates, name="signal_date"),
    )
    diagnostics_frame = pd.DataFrame(diagnostics).set_index("signal_date")
    return ForwardReturnResult(
        period_returns=period_returns,
        diagnostics=diagnostics_frame,
        policy_id=active_policy.policy_id,
    )


def _normalize_panel(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(frame).copy()
    out.index = [_normalize_date(item) for item in out.index]
    out.columns = [str(item).zfill(6) for item in out.columns]
    if not out.index.is_unique or not out.columns.is_unique:
        raise ValueError("return_panel_axes_must_be_unique")
    values = out.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    return values.astype(float)


def _normalize_date(value: object) -> str:
    compact = str(value).replace("-", "").replace("/", "").replace(".", "")
    if len(compact) >= 8 and compact[:8].isdigit():
        return compact[:8]
    return pd.Timestamp(value).strftime("%Y%m%d")
