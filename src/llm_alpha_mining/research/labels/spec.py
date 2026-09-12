from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math
from typing import cast

from llm_alpha_mining.research.core.frequency import FrequencySpec
from llm_alpha_mining.research.core.hashing import hash_json, require_sha256


class LabelTask(str, Enum):
    RETURN = "return"
    EXCESS_RETURN = "excess_return"
    DIRECTION = "direction"
    VOLATILITY = "volatility"


class EntryEvent(str, Enum):
    NEXT_SESSION_OPEN = "next_session_open"
    SIGNAL_SESSION_CLOSE = "signal_session_close"


class PriceObservationEvent(str, Enum):
    """Economic event represented by a price field.

    This is intentionally separate from :class:`EntryEvent`: an entry event is
    relative to a signal, while this enum describes the field itself.  Keeping
    both identities prevents a label called ``next_session_open`` from silently
    consuming a close-price column.
    """

    SESSION_OPEN = "session_open"
    SESSION_CLOSE = "session_close"


@dataclass(frozen=True, slots=True)
class LabelSpec:
    label_id: str
    version: str
    task: LabelTask | str
    horizon_sessions: int
    information_cutoff_event: str
    entry_event: EntryEvent | str
    entry_lag_sessions: int
    price_field: str
    price_observation_event: PriceObservationEvent | str
    adjustment_field: str | None
    amount_field: str | None
    require_execution_tradability: bool
    benchmark_id: str | None
    benchmark_snapshot_id: str | None
    direction_threshold: float
    annualize_volatility: bool
    missing_policy: str
    dataset_id: str
    snapshot_id: str
    schema_hash: str
    availability_hash: str
    security_contract_hash: str
    frequency: FrequencySpec
    schema_version: str = "label-spec/v2"

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str):
            raise TypeError("label schema_version must be text")
        if self.schema_version != "label-spec/v2":
            raise ValueError("unsupported LabelSpec schema")
        if not isinstance(self.task, (LabelTask, str)):
            raise TypeError("label task must be text")
        if not isinstance(self.entry_event, (EntryEvent, str)):
            raise TypeError("label entry_event must be text")
        if not isinstance(self.price_observation_event, (PriceObservationEvent, str)):
            raise TypeError("label price_observation_event must be text")
        object.__setattr__(self, "task", LabelTask(self.task))
        object.__setattr__(self, "entry_event", EntryEvent(self.entry_event))
        object.__setattr__(
            self,
            "price_observation_event",
            PriceObservationEvent(self.price_observation_event),
        )
        for name in (
            "label_id",
            "version",
            "information_cutoff_event",
            "price_field",
            "dataset_id",
        ):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"label {name} must be text")
            if not getattr(self, name).strip():
                raise ValueError(f"label {name} must not be empty")
        for name in (
            "adjustment_field",
            "amount_field",
            "benchmark_id",
            "benchmark_snapshot_id",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"label {name} must be text or null")
        for name in ("require_execution_tradability", "annualize_volatility"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"label {name} must be boolean")
        if (
            not isinstance(self.direction_threshold, (int, float))
            or isinstance(self.direction_threshold, bool)
            or not math.isfinite(float(self.direction_threshold))
        ):
            raise TypeError("label direction_threshold must be a finite number")
        object.__setattr__(self, "direction_threshold", float(self.direction_threshold))
        if not isinstance(self.frequency, FrequencySpec):
            raise TypeError("label frequency must be FrequencySpec")
        if self.information_cutoff_event not in {
            "signal_session_open",
            "signal_session_close",
        }:
            raise ValueError(
                "labels require an explicit signal-session open or close cutoff"
            )
        if (
            not isinstance(self.horizon_sessions, int)
            or isinstance(self.horizon_sessions, bool)
            or self.horizon_sessions <= 0
        ):
            raise ValueError("label horizon_sessions must be positive")
        if (
            not isinstance(self.entry_lag_sessions, int)
            or isinstance(self.entry_lag_sessions, bool)
            or self.entry_lag_sessions < 0
        ):
            raise ValueError("label entry_lag_sessions must be non-negative")
        if self.entry_event is EntryEvent.NEXT_SESSION_OPEN:
            if self.entry_lag_sessions < 1:
                raise ValueError("next-session-open labels require positive entry lag")
            if self.task is LabelTask.VOLATILITY:
                raise ValueError("volatility labels require signal-session-close entry")
            if self.price_observation_event is not PriceObservationEvent.SESSION_OPEN:
                raise ValueError(
                    "next-session-open labels require a session-open price observation"
                )
        if self.entry_event is EntryEvent.SIGNAL_SESSION_CLOSE:
            if self.entry_lag_sessions != 0:
                raise ValueError("signal-close labels require zero entry lag")
            if self.task is not LabelTask.VOLATILITY:
                raise ValueError(
                    "return/direction labels require next-session-open entry"
                )
            if self.price_observation_event is not PriceObservationEvent.SESSION_CLOSE:
                raise ValueError(
                    "signal-close labels require a session-close price observation"
                )
        if self.task is LabelTask.EXCESS_RETURN:
            if not self.benchmark_id or not self.benchmark_snapshot_id:
                raise ValueError("excess-return labels require a benchmark binding")
            require_sha256(
                self.benchmark_snapshot_id, name="label benchmark_snapshot_id"
            )
        elif self.benchmark_id is not None or self.benchmark_snapshot_id is not None:
            raise ValueError("only excess-return labels may bind a benchmark")
        if self.task is not LabelTask.DIRECTION and self.direction_threshold != 0.0:
            raise ValueError("direction_threshold only applies to direction labels")
        if self.task is not LabelTask.VOLATILITY and self.annualize_volatility:
            raise ValueError("annualize_volatility only applies to volatility labels")
        if self.missing_policy != "invalidate_label":
            raise ValueError("labels must fail closed on missing observations")
        if self.require_execution_tradability and self.task is LabelTask.VOLATILITY:
            raise ValueError(
                "realized-volatility labels do not use execution tradability"
            )
        if self.require_execution_tradability and self.amount_field is None:
            raise ValueError("execution-aware labels require an amount field")
        require_sha256(self.snapshot_id, name="label snapshot_id")
        require_sha256(self.schema_hash, name="label schema_hash")
        require_sha256(self.availability_hash, name="label availability_hash")
        require_sha256(self.security_contract_hash, name="label security_contract_hash")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        task = self.task
        event = self.entry_event
        price_event = self.price_observation_event
        if (
            not isinstance(task, LabelTask)
            or not isinstance(event, EntryEvent)
            or not isinstance(price_event, PriceObservationEvent)
        ):  # pragma: no cover
            raise RuntimeError("LabelSpec enums were not normalized")
        return {
            "schema_version": self.schema_version,
            "label_id": self.label_id,
            "version": self.version,
            "task": task.value,
            "horizon_sessions": self.horizon_sessions,
            "information_cutoff_event": self.information_cutoff_event,
            "entry_event": event.value,
            "entry_lag_sessions": self.entry_lag_sessions,
            "price_field": self.price_field,
            "price_observation_event": price_event.value,
            "adjustment_field": self.adjustment_field,
            "amount_field": self.amount_field,
            "require_execution_tradability": self.require_execution_tradability,
            "benchmark_id": self.benchmark_id,
            "benchmark_snapshot_id": self.benchmark_snapshot_id,
            "direction_threshold": self.direction_threshold,
            "annualize_volatility": self.annualize_volatility,
            "missing_policy": self.missing_policy,
            "dataset_id": self.dataset_id,
            "snapshot_id": self.snapshot_id,
            "schema_hash": self.schema_hash,
            "availability_hash": self.availability_hash,
            "security_contract_hash": self.security_contract_hash,
            "frequency": self.frequency.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "LabelSpec":
        expected = {
            "schema_version",
            "label_id",
            "version",
            "task",
            "horizon_sessions",
            "information_cutoff_event",
            "entry_event",
            "entry_lag_sessions",
            "price_field",
            "price_observation_event",
            "adjustment_field",
            "amount_field",
            "require_execution_tradability",
            "benchmark_id",
            "benchmark_snapshot_id",
            "direction_threshold",
            "annualize_volatility",
            "missing_policy",
            "dataset_id",
            "snapshot_id",
            "schema_hash",
            "availability_hash",
            "security_contract_hash",
            "frequency",
        }
        if set(value) != expected:
            raise ValueError("LabelSpec wire fields differ")
        frequency = value["frequency"]
        if not isinstance(frequency, Mapping):
            raise TypeError("LabelSpec frequency must be an object")
        return cls(
            schema_version=_wire_text(value["schema_version"], "schema_version"),
            label_id=_wire_text(value["label_id"], "label_id"),
            version=_wire_text(value["version"], "version"),
            task=LabelTask(_wire_text(value["task"], "task")),
            horizon_sessions=_wire_integer(
                value["horizon_sessions"], "horizon_sessions"
            ),
            information_cutoff_event=_wire_text(
                value["information_cutoff_event"], "information_cutoff_event"
            ),
            entry_event=EntryEvent(_wire_text(value["entry_event"], "entry_event")),
            entry_lag_sessions=_wire_integer(
                value["entry_lag_sessions"], "entry_lag_sessions"
            ),
            price_field=_wire_text(value["price_field"], "price_field"),
            price_observation_event=PriceObservationEvent(
                _wire_text(value["price_observation_event"], "price_observation_event")
            ),
            adjustment_field=_wire_optional_text(
                value["adjustment_field"], "adjustment_field"
            ),
            amount_field=_wire_optional_text(value["amount_field"], "amount_field"),
            require_execution_tradability=_wire_boolean(
                value["require_execution_tradability"],
                "require_execution_tradability",
            ),
            benchmark_id=_wire_optional_text(value["benchmark_id"], "benchmark_id"),
            benchmark_snapshot_id=_wire_optional_text(
                value["benchmark_snapshot_id"], "benchmark_snapshot_id"
            ),
            direction_threshold=_wire_number(
                value["direction_threshold"], "direction_threshold"
            ),
            annualize_volatility=_wire_boolean(
                value["annualize_volatility"], "annualize_volatility"
            ),
            missing_policy=_wire_text(value["missing_policy"], "missing_policy"),
            dataset_id=_wire_text(value["dataset_id"], "dataset_id"),
            snapshot_id=_wire_text(value["snapshot_id"], "snapshot_id"),
            schema_hash=_wire_text(value["schema_hash"], "schema_hash"),
            availability_hash=_wire_text(
                value["availability_hash"], "availability_hash"
            ),
            security_contract_hash=_wire_text(
                value["security_contract_hash"], "security_contract_hash"
            ),
            frequency=FrequencySpec.from_mapping(frequency),
        )


def _wire_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"LabelSpec {name} must be text")
    return value


def _wire_optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _wire_text(value, name)


def _wire_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"LabelSpec {name} must be an integer")
    return value


def _wire_boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"LabelSpec {name} must be boolean")
    return value


def _wire_number(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise TypeError(f"LabelSpec {name} must be a finite number")
    return float(value)


__all__ = ["EntryEvent", "LabelSpec", "LabelTask", "PriceObservationEvent"]
