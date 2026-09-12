from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import time
from enum import Enum
from typing import cast
from zoneinfo import ZoneInfo

import pandas as pd

from .hashing import hash_json


class FrequencyMode(str, Enum):
    BAR = "bar"
    EVENT = "event"
    VOLUME = "volume"


class TimestampPolicy(str, Enum):
    BAR_OPEN = "bar_open"
    BAR_CLOSE = "bar_close"
    EVENT_TIME = "event_time"


class RevisionPolicy(str, Enum):
    FIRST_SEEN = "first_seen"
    AS_OF = "as_of"
    LATEST_FOR_REPLAY = "latest_for_replay"


@dataclass(frozen=True, slots=True)
class AvailabilityLag:
    duration: str
    applies_from: str = "known_at"

    def __post_init__(self) -> None:
        if pd.Timedelta(self.duration) < pd.Timedelta(0):
            raise ValueError("availability lag must not be negative")
        if self.applies_from not in {"known_at", "effective_at"}:
            raise ValueError("availability lag reference is invalid")

    def to_dict(self) -> dict[str, str]:
        return {"duration": self.duration, "applies_from": self.applies_from}


@dataclass(frozen=True, slots=True)
class FrequencySpec:
    mode: FrequencyMode
    interval: str | None
    calendar_id: str
    timezone: str
    session_id: str
    timestamp_policy: TimestampPolicy
    sampling_policy: str

    def __post_init__(self) -> None:
        if not isinstance(self.mode, (FrequencyMode, str)):
            raise TypeError("frequency mode must be text")
        if not isinstance(self.timestamp_policy, (TimestampPolicy, str)):
            raise TypeError("frequency timestamp_policy must be text")
        object.__setattr__(self, "mode", FrequencyMode(self.mode))
        object.__setattr__(
            self, "timestamp_policy", TimestampPolicy(self.timestamp_policy)
        )
        if self.interval is not None and not isinstance(self.interval, str):
            raise TypeError("frequency interval must be text or null")
        for name in ("calendar_id", "timezone", "session_id", "sampling_policy"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"frequency {name} must be text")
        ZoneInfo(self.timezone)
        for name in ("calendar_id", "session_id", "sampling_policy"):
            if not getattr(self, name).strip():
                raise ValueError(f"frequency {name} must not be empty")
        if self.mode is FrequencyMode.BAR:
            if self.interval is None or pd.Timedelta(self.interval) <= pd.Timedelta(0):
                raise ValueError("bar frequency requires a positive interval")
            if self.timestamp_policy not in {
                TimestampPolicy.BAR_OPEN,
                TimestampPolicy.BAR_CLOSE,
            }:
                raise ValueError("bar frequency requires bar_open or bar_close policy")
        elif self.mode is FrequencyMode.EVENT:
            if self.interval is not None:
                raise ValueError("event frequency must not declare a fixed interval")
            if self.timestamp_policy is not TimestampPolicy.EVENT_TIME:
                raise ValueError("event frequency requires event_time policy")
        elif self.mode is FrequencyMode.VOLUME:
            if self.interval is not None:
                raise ValueError("volume frequency uses sampling_policy, not interval")
            if self.timestamp_policy is not TimestampPolicy.EVENT_TIME:
                raise ValueError("volume sampling requires event_time policy")

    @classmethod
    def daily(cls) -> "FrequencySpec":
        return cls(
            mode=FrequencyMode.BAR,
            interval="1d",
            calendar_id="SSE_SZSE",
            timezone="Asia/Shanghai",
            session_id="CN_A_SHARE_REGULAR",
            timestamp_policy=TimestampPolicy.BAR_CLOSE,
            sampling_policy="calendar_bar",
        )

    @classmethod
    def minute(cls, interval: str = "1min") -> "FrequencySpec":
        return cls(
            mode=FrequencyMode.BAR,
            interval=interval,
            calendar_id="SSE_SZSE",
            timezone="Asia/Shanghai",
            session_id="CN_A_SHARE_REGULAR",
            timestamp_policy=TimestampPolicy.BAR_CLOSE,
            sampling_policy="calendar_bar",
        )

    @classmethod
    def events(cls, sampling_policy: str = "all_events") -> "FrequencySpec":
        return cls(
            mode=FrequencyMode.EVENT,
            interval=None,
            calendar_id="SSE_SZSE",
            timezone="Asia/Shanghai",
            session_id="CN_A_SHARE_REGULAR",
            timestamp_policy=TimestampPolicy.EVENT_TIME,
            sampling_policy=sampling_policy,
        )

    @classmethod
    def volume_bars(cls, sampling_policy: str) -> "FrequencySpec":
        return cls(
            mode=FrequencyMode.VOLUME,
            interval=None,
            calendar_id="SSE_SZSE",
            timezone="Asia/Shanghai",
            session_id="CN_A_SHARE_REGULAR",
            timestamp_policy=TimestampPolicy.EVENT_TIME,
            sampling_policy=sampling_policy,
        )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "interval": self.interval,
            "calendar_id": self.calendar_id,
            "timezone": self.timezone,
            "session_id": self.session_id,
            "timestamp_policy": self.timestamp_policy.value,
            "sampling_policy": self.sampling_policy,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FrequencySpec":
        expected = {
            "mode",
            "interval",
            "calendar_id",
            "timezone",
            "session_id",
            "timestamp_policy",
            "sampling_policy",
        }
        if set(value) != expected:
            raise ValueError("FrequencySpec wire fields differ")
        interval = value["interval"]
        if interval is not None and not isinstance(interval, str):
            raise TypeError("FrequencySpec interval must be text or null")
        return cls(
            mode=FrequencyMode(_wire_text(value["mode"], name="mode")),
            interval=interval,
            calendar_id=_wire_text(value["calendar_id"], name="calendar_id"),
            timezone=_wire_text(value["timezone"], name="timezone"),
            session_id=_wire_text(value["session_id"], name="session_id"),
            timestamp_policy=TimestampPolicy(
                _wire_text(value["timestamp_policy"], name="timestamp_policy")
            ),
            sampling_policy=_wire_text(
                value["sampling_policy"], name="sampling_policy"
            ),
        )


@dataclass(frozen=True, slots=True)
class AvailabilitySpec:
    effective_at_field: str
    known_at_field: str
    publication_lag: AvailabilityLag | str
    revision_policy: RevisionPolicy

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "revision_policy", RevisionPolicy(self.revision_policy)
        )
        if isinstance(self.publication_lag, str):
            object.__setattr__(
                self, "publication_lag", AvailabilityLag(self.publication_lag)
            )
        elif not isinstance(self.publication_lag, AvailabilityLag):
            raise TypeError("publication_lag must be AvailabilityLag or duration text")
        if not self.effective_at_field or not self.known_at_field:
            raise ValueError("availability fields must not be empty")
        if self.effective_at_field == self.known_at_field:
            raise ValueError("effective_at and known_at fields must differ")

    @classmethod
    def market_data(cls) -> "AvailabilitySpec":
        return cls(
            effective_at_field="effective_at",
            known_at_field="known_at",
            publication_lag="0s",
            revision_policy=RevisionPolicy.FIRST_SEEN,
        )

    def to_dict(self) -> dict[str, object]:
        lag = self._normalized_lag()
        return {
            "effective_at_field": self.effective_at_field,
            "known_at_field": self.known_at_field,
            "publication_lag": lag.to_dict(),
            "revision_policy": self.revision_policy.value,
        }

    def visible_as_of(
        self,
        frame: pd.DataFrame,
        *,
        as_of: pd.Timestamp,
        allow_latest_replay: bool = False,
        revision_keys: tuple[str, ...] = (),
        resolve_revisions: bool = True,
    ) -> pd.DataFrame:
        cutoff = _aware_timestamp(as_of, name="as_of")
        if (
            self.revision_policy is RevisionPolicy.LATEST_FOR_REPLAY
            and not allow_latest_replay
        ):
            raise PermissionError("latest_for_replay availability is forbidden")
        if self.effective_at_field not in frame or self.known_at_field not in frame:
            raise ValueError("availability columns are missing")
        effective = pd.to_datetime(frame[self.effective_at_field], errors="raise")
        known = pd.to_datetime(frame[self.known_at_field], errors="raise")
        if effective.dt.tz is None or known.dt.tz is None:
            raise ValueError("availability timestamps must be timezone-aware")
        normalized_lag = self._normalized_lag()
        lag = pd.Timedelta(normalized_lag.duration)
        lag_origin = known if normalized_lag.applies_from == "known_at" else effective
        visible = (
            (lag_origin + lag <= cutoff) & (known <= cutoff) & (effective <= cutoff)
        )
        selected = frame.loc[visible].copy()
        grouping = [*revision_keys, self.effective_at_field]
        if resolve_revisions and selected.duplicated(grouping, keep=False).any():
            if not revision_keys:
                raise ValueError("unknown_revision_identity_keys")
            selected = selected.assign(__known_at_sort=known.loc[selected.index])
            selected = selected.sort_values(
                grouping + ["__known_at_sort"], kind="stable"
            )
            keep = (
                "first" if self.revision_policy is RevisionPolicy.FIRST_SEEN else "last"
            )
            selected = selected.drop_duplicates(grouping, keep=keep).drop(
                columns="__known_at_sort"
            )
        return selected

    def _normalized_lag(self) -> AvailabilityLag:
        if not isinstance(self.publication_lag, AvailabilityLag):  # pragma: no cover
            raise RuntimeError("availability lag was not normalized")
        return self.publication_lag


@dataclass(frozen=True, slots=True)
class SessionSpec:
    session_id: str
    timezone: str
    intervals: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        ZoneInfo(self.timezone)
        if not self.session_id or not self.intervals:
            raise ValueError("session must define an id and intervals")
        previous_end: time | None = None
        for start, end in self.intervals:
            start_time = time.fromisoformat(start)
            end_time = time.fromisoformat(end)
            if start_time >= end_time:
                raise ValueError("session interval start must precede end")
            if previous_end is not None and start_time < previous_end:
                raise ValueError("session intervals must be sorted and non-overlapping")
            previous_end = end_time

    @classmethod
    def cn_a_share_regular(cls) -> "SessionSpec":
        return cls(
            session_id="CN_A_SHARE_REGULAR",
            timezone="Asia/Shanghai",
            intervals=(("09:30:00", "11:30:00"), ("13:00:00", "15:00:00")),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "SessionSpec":
        expected = {"session_id", "timezone", "intervals"}
        if set(value) != expected:
            raise ValueError("session mapping fields differ")
        raw_intervals = value["intervals"]
        if not isinstance(raw_intervals, list):
            raise TypeError("session intervals must be a list")
        intervals: list[tuple[str, str]] = []
        for item in raw_intervals:
            if not isinstance(item, list) or len(item) != 2:
                raise TypeError("session interval must be a two-item list")
            intervals.append((str(item[0]), str(item[1])))
        return cls(
            session_id=str(value["session_id"]),
            timezone=str(value["timezone"]),
            intervals=tuple(intervals),
        )

    def contains_local_time(self, value: time) -> bool:
        return any(
            time.fromisoformat(start) <= value <= time.fromisoformat(end)
            for start, end in self.intervals
        )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "timezone": self.timezone,
            "intervals": [list(item) for item in self.intervals],
        }


@dataclass(frozen=True, slots=True)
class TradingCalendar:
    calendar_id: str
    timezone: str
    sessions: tuple[str, ...]
    session: SessionSpec

    def __post_init__(self) -> None:
        ZoneInfo(self.timezone)
        if self.calendar_id == "" or self.session.timezone != self.timezone:
            raise ValueError("calendar/session timezone mismatch")
        normalized = tuple(_date_text(item) for item in self.sessions)
        if normalized != tuple(sorted(set(normalized))):
            raise ValueError("calendar sessions must be sorted and unique")
        object.__setattr__(self, "sessions", normalized)

    def assert_timestamp(
        self, value: pd.Timestamp, *, allow_daily_close: bool = True
    ) -> None:
        timestamp = _aware_timestamp(value, name="timestamp")
        local = timestamp.tz_convert(self.timezone)
        date_text = local.strftime("%Y%m%d")
        if date_text not in set(self.sessions):
            raise ValueError(f"timestamp_not_in_trading_calendar:{date_text}")
        if not self.session.contains_local_time(local.time()):
            if not (allow_daily_close and local.time() == time(15, 0)):
                raise ValueError(f"timestamp_outside_session:{local.isoformat()}")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TradingCalendar":
        expected = {"calendar_id", "timezone", "sessions", "session"}
        if set(value) != expected:
            raise ValueError("trading calendar mapping fields differ")
        raw_sessions = value["sessions"]
        raw_session = value["session"]
        if not isinstance(raw_sessions, list):
            raise TypeError("trading calendar sessions must be a list")
        if not isinstance(raw_session, Mapping):
            raise TypeError("trading calendar session must be an object")
        return cls(
            calendar_id=str(value["calendar_id"]),
            timezone=str(value["timezone"]),
            sessions=tuple(str(item) for item in raw_sessions),
            session=SessionSpec.from_mapping(raw_session),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "calendar_id": self.calendar_id,
            "timezone": self.timezone,
            "sessions": list(self.sessions),
            "session": self.session.to_dict(),
        }


def _aware_timestamp(value: object, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp


def _date_text(value: object) -> str:
    text = str(value).strip().replace("-", "").replace("/", "").replace(".", "")
    if text.isdigit() and len(text) >= 8:
        return text[:8]
    return cast(str, pd.Timestamp(value).strftime("%Y%m%d"))


def _wire_text(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"FrequencySpec {name} must be text")
    return value


__all__ = [
    "AvailabilityLag",
    "AvailabilitySpec",
    "FrequencyMode",
    "FrequencySpec",
    "RevisionPolicy",
    "SessionSpec",
    "TimestampPolicy",
    "TradingCalendar",
]
