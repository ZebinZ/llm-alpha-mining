from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from alpha_research.core.frequency import FrequencyMode, FrequencySpec, TradingCalendar


_ALLOWED_AGGREGATIONS = frozenset(
    {"open", "high", "low", "close", "last", "first", "sum", "mean"}
)


class CalendarBarAggregator:
    """Aggregate standardized records without reading storage or creating labels."""

    def aggregate(
        self,
        frame: pd.DataFrame,
        *,
        source_frequency: FrequencySpec,
        target_frequency: FrequencySpec,
        field_methods: Mapping[str, str],
        calendar: TradingCalendar,
    ) -> pd.DataFrame:
        if (
            source_frequency.mode is not FrequencyMode.BAR
            or target_frequency.mode is not FrequencyMode.BAR
        ):
            raise ValueError("calendar bar aggregation requires bar frequencies")
        if source_frequency.calendar_id != target_frequency.calendar_id:
            raise ValueError("source and target calendars differ")
        if target_frequency.interval != "1d":
            raise NotImplementedError("Phase 1 aggregator currently targets daily bars")
        if not field_methods or any(
            method not in _ALLOWED_AGGREGATIONS for method in field_methods.values()
        ):
            raise ValueError("unknown or empty aggregation method")
        required = {"timestamp", "security", "effective_at", "known_at", *field_methods}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError("aggregation input fields missing:" + ",".join(missing))
        values = pd.DataFrame(frame).copy()
        timestamps = pd.to_datetime(values["timestamp"], errors="raise")
        if timestamps.dt.tz is None:
            raise ValueError("aggregation timestamps must be timezone-aware")
        local = timestamps.dt.tz_convert(target_frequency.timezone)
        values["session_date"] = local.dt.strftime("%Y%m%d")
        if not set(values["session_date"]).issubset(set(calendar.sessions)):
            raise ValueError("aggregation contains dates outside calendar")
        rows: list[dict[str, object]] = []
        for (date, security), group in values.groupby(
            ["session_date", "security"], sort=True, observed=True
        ):
            group = group.sort_values("timestamp", kind="stable")
            close_timestamp = pd.Timestamp(
                f"{date} 15:00:00", tz=target_frequency.timezone
            )
            row: dict[str, object] = {
                "timestamp": close_timestamp,
                "security": str(security),
                "effective_at": close_timestamp,
                "known_at": max(pd.to_datetime(group["known_at"], errors="raise")),
            }
            for field, method in field_methods.items():
                series = group[field]
                if method in {"open", "first"}:
                    row[field] = series.iloc[0]
                elif method in {"close", "last"}:
                    row[field] = series.iloc[-1]
                elif method == "high":
                    row[field] = series.max(skipna=True)
                elif method == "low":
                    row[field] = series.min(skipna=True)
                elif method == "sum":
                    row[field] = series.sum(min_count=1)
                elif method == "mean":
                    row[field] = series.mean()
            rows.append(row)
        return (
            pd.DataFrame(rows)
            .sort_values(["timestamp", "security"], kind="stable")
            .reset_index(drop=True)
        )


__all__ = ["CalendarBarAggregator"]
