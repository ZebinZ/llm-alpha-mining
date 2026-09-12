from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import time
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import cast
from zoneinfo import ZoneInfo

import pandas as pd

from alpha_research.core.frequency import TradingCalendar
from alpha_research.core.hashing import hash_json


class SourceKind(str, Enum):
    SNAPSHOT = "snapshot"
    ORDER = "order"
    TRADE = "trade"


class AvailabilityMode(str, Enum):
    EVENT_TIME_LAG = "event_time_lag"
    STATIC_POST_CLOSE = "static_post_close"


class DataCapability(str, Enum):
    POST_CLOSE_DAILY_AGGREGATION = "post_close_daily_aggregation"
    SNAPSHOT_BOOK_STATE = "snapshot_book_state"
    BASIC_ORDER_FLOW = "basic_order_flow"
    ACTIVE_TRADE_ORDER_PROJECTION = "active_trade_order_projection"
    BASIC_TRADE_FLOW = "basic_trade_flow"
    DIRECTIONAL_FLOW = "directional_flow"
    CANCEL_FLOW = "cancel_flow"
    ORDER_LIFECYCLE = "order_lifecycle"
    EXACT_COMBINED_SEQUENCE = "exact_combined_sequence"
    FULL_LOB_RECONSTRUCTION = "full_lob_reconstruction"
    TRUE_INTRADAY_EXECUTION = "true_intraday_execution"


class CapabilityStatus(str, Enum):
    VERIFIED = "verified"
    RESEARCH_ONLY = "research_only"
    UNVERIFIED = "unverified"
    UNSUPPORTED = "unsupported"


class IntendedUse(str, Enum):
    RESEARCH = "research"
    PRODUCTION = "production"


class EvidenceKind(str, Enum):
    TEACHER_CONFIRMATION = "teacher_confirmation"
    EXCHANGE_DOCUMENTATION = "exchange_documentation"
    PUBLIC_CONNECTOR_DOCUMENTATION = "public_connector_documentation"
    LOCAL_DATA_AUDIT = "local_data_audit"


class SemanticAdmissionError(PermissionError):
    pass


class ExecutionTimingError(PermissionError):
    pass


@dataclass(frozen=True, slots=True)
class EnumValueSemantic:
    """Versioned interpretation of one provider enum value.

    Raw numeric values are deliberately represented as strings in the
    configuration.  This keeps JSON round-trips exact and prevents ``1`` and
    ``1.0`` from becoming two apparently different provider contracts.
    """

    raw_value: str
    meaning: str

    def __post_init__(self) -> None:
        raw = str(self.raw_value).strip()
        meaning = str(self.meaning).strip()
        if not raw or not meaning:
            raise ValueError("enum value semantics must not be empty")
        try:
            integer = int(raw)
        except ValueError as exc:
            raise ValueError("provider enum values must be integer text") from exc
        if str(integer) != raw:
            raise ValueError("provider enum values must use canonical integer text")
        object.__setattr__(self, "raw_value", raw)
        object.__setattr__(self, "meaning", meaning)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "EnumValueSemantic":
        _strict_keys(
            value,
            required={"raw_value", "meaning"},
            optional=set(),
            path="provider_semantics.profiles[].enum_semantics[].values[]",
        )
        return cls(raw_value=str(value["raw_value"]), meaning=str(value["meaning"]))

    def to_dict(self) -> dict[str, str]:
        return {"raw_value": self.raw_value, "meaning": self.meaning}


@dataclass(frozen=True, slots=True)
class EnumFieldSemantics:
    """Closed value vocabulary for one physical provider field."""

    source_field: str
    values: tuple[EnumValueSemantic, ...]

    def __post_init__(self) -> None:
        field = str(self.source_field).strip()
        if not field or not self.values:
            raise ValueError("enum field semantics require a field and values")
        raw_values = [item.raw_value for item in self.values]
        meanings = [item.meaning for item in self.values]
        if len(raw_values) != len(set(raw_values)):
            raise ValueError("provider enum raw values must be unique")
        if any(not item for item in meanings):  # pragma: no cover - child invariant
            raise ValueError("provider enum meanings must not be empty")
        object.__setattr__(self, "source_field", field)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "EnumFieldSemantics":
        _strict_keys(
            value,
            required={"source_field", "values"},
            optional=set(),
            path="provider_semantics.profiles[].enum_semantics[]",
        )
        return cls(
            source_field=str(value["source_field"]),
            values=tuple(
                EnumValueSemantic.from_mapping(item)
                for item in _objects(
                    value["values"],
                    path="provider_semantics.profiles[].enum_semantics[].values",
                )
            ),
        )

    @property
    def value_map(self) -> Mapping[int, str]:
        return MappingProxyType(
            {int(item.raw_value): item.meaning for item in self.values}
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "source_field": self.source_field,
            "values": [item.to_dict() for item in self.values],
        }


@dataclass(frozen=True, slots=True)
class SemanticEvidence:
    evidence_id: str
    kind: EvidenceKind | str
    claim: str
    reference: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", EvidenceKind(self.kind))
        for name in ("evidence_id", "claim", "reference"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"semantic evidence {name} must not be empty")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "SemanticEvidence":
        _strict_keys(
            value,
            required={"evidence_id", "kind", "claim", "reference"},
            optional=set(),
            path="provider_semantics.profiles[].evidence[]",
        )
        return cls(
            evidence_id=str(value["evidence_id"]),
            kind=str(value["kind"]),
            claim=str(value["claim"]),
            reference=str(value["reference"]),
        )

    def to_dict(self) -> dict[str, str]:
        kind = self.kind
        if not isinstance(kind, EvidenceKind):  # pragma: no cover
            raise RuntimeError("semantic evidence kind was not normalized")
        return {
            "evidence_id": self.evidence_id,
            "kind": kind.value,
            "claim": self.claim,
            "reference": self.reference,
        }


@dataclass(frozen=True, slots=True)
class CapabilityAssessment:
    capability: DataCapability | str
    status: CapabilityStatus | str
    rationale: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "capability", DataCapability(self.capability))
        object.__setattr__(self, "status", CapabilityStatus(self.status))
        if not self.rationale.strip():
            raise ValueError("capability rationale must not be empty")
        evidence = tuple(str(item).strip() for item in self.evidence_ids)
        if any(not item for item in evidence) or len(evidence) != len(set(evidence)):
            raise ValueError("capability evidence ids must be non-empty and unique")
        object.__setattr__(self, "evidence_ids", evidence)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CapabilityAssessment":
        _strict_keys(
            value,
            required={"capability", "status", "rationale", "evidence_ids"},
            optional=set(),
            path="provider_semantics.profiles[].capabilities[]",
        )
        return cls(
            capability=str(value["capability"]),
            status=str(value["status"]),
            rationale=str(value["rationale"]),
            evidence_ids=_strings(
                value["evidence_ids"], path="capability.evidence_ids"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        capability = self.capability
        status = self.status
        if not isinstance(capability, DataCapability):  # pragma: no cover
            raise RuntimeError("data capability was not normalized")
        if not isinstance(status, CapabilityStatus):  # pragma: no cover
            raise RuntimeError("capability status was not normalized")
        return {
            "capability": capability.value,
            "status": status.value,
            "rationale": self.rationale,
            "evidence_ids": list(self.evidence_ids),
        }

    def permits(self, intended_use: IntendedUse | str) -> bool:
        use = IntendedUse(intended_use)
        status = CapabilityStatus(self.status)
        if use is IntendedUse.RESEARCH:
            return status in {CapabilityStatus.VERIFIED, CapabilityStatus.RESEARCH_ONLY}
        return status is CapabilityStatus.VERIFIED


@dataclass(frozen=True, slots=True)
class SourceAvailabilityPolicy:
    mode: AvailabilityMode | str
    timezone: str
    provenance: str
    exact_delivery_time_known: bool
    event_time_lag: str = "0s"
    conservative_known_time: str | None = None
    earliest_execution_session_offset: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", AvailabilityMode(self.mode))
        ZoneInfo(self.timezone)
        if not self.provenance.strip():
            raise ValueError("availability provenance must not be empty")
        if not isinstance(self.exact_delivery_time_known, bool):
            raise TypeError("exact_delivery_time_known must be boolean")
        if (
            not isinstance(self.earliest_execution_session_offset, int)
            or self.earliest_execution_session_offset < 0
        ):
            raise ValueError("earliest execution session offset must be nonnegative")
        mode = AvailabilityMode(self.mode)
        if mode is AvailabilityMode.EVENT_TIME_LAG:
            if pd.Timedelta(self.event_time_lag) < pd.Timedelta(0):
                raise ValueError("event-time availability lag must not be negative")
            if self.conservative_known_time is not None:
                raise ValueError("event-time availability cannot declare an EOD clock")
        else:
            if self.exact_delivery_time_known:
                raise ValueError(
                    "static post-close policy cannot claim an exact delivery time"
                )
            if self.conservative_known_time is None:
                raise ValueError(
                    "static post-close policy requires a conservative known time"
                )
            clock = time.fromisoformat(self.conservative_known_time)
            if clock <= time(15, 0):
                raise ValueError(
                    "static post-close known time must be after market close"
                )
            if pd.Timedelta(self.event_time_lag) != pd.Timedelta(0):
                raise ValueError(
                    "static post-close policy does not use an event-time lag"
                )
            if self.earliest_execution_session_offset < 1:
                raise ValueError(
                    "static post-close data cannot be executable in the same session"
                )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "SourceAvailabilityPolicy":
        _strict_keys(
            value,
            required={
                "mode",
                "timezone",
                "provenance",
                "exact_delivery_time_known",
                "event_time_lag",
                "conservative_known_time",
                "earliest_execution_session_offset",
            },
            optional=set(),
            path="provider_semantics.profiles[].availability",
        )
        exact = value["exact_delivery_time_known"]
        if not isinstance(exact, bool):
            raise TypeError("availability exact_delivery_time_known must be boolean")
        offset = value["earliest_execution_session_offset"]
        if not isinstance(offset, int) or isinstance(offset, bool):
            raise TypeError("availability session offset must be an integer")
        clock = value["conservative_known_time"]
        return cls(
            mode=str(value["mode"]),
            timezone=str(value["timezone"]),
            provenance=str(value["provenance"]),
            exact_delivery_time_known=exact,
            event_time_lag=str(value["event_time_lag"]),
            conservative_known_time=None if clock is None else str(clock),
            earliest_execution_session_offset=offset,
        )

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        mode = self.mode
        if not isinstance(mode, AvailabilityMode):  # pragma: no cover
            raise RuntimeError("availability mode was not normalized")
        return {
            "mode": mode.value,
            "timezone": self.timezone,
            "provenance": self.provenance,
            "exact_delivery_time_known": self.exact_delivery_time_known,
            "event_time_lag": self.event_time_lag,
            "conservative_known_time": self.conservative_known_time,
            "earliest_execution_session_offset": self.earliest_execution_session_offset,
        }

    def knowledge_times(
        self,
        event_times: pd.Series,
        *,
        trading_dates: pd.Series | Sequence[str] | None = None,
    ) -> pd.Series:
        events = pd.to_datetime(event_times, errors="raise")
        if events.dt.tz is None:
            raise ValueError("availability event times must be timezone-aware")
        events = events.dt.tz_convert(self.timezone)
        mode = AvailabilityMode(self.mode)
        if mode is AvailabilityMode.EVENT_TIME_LAG:
            return events + pd.Timedelta(self.event_time_lag)
        if trading_dates is None:
            dates = events.dt.strftime("%Y%m%d")
        else:
            dates = pd.Series(trading_dates, index=events.index, dtype="string")
            dates = dates.map(_date_text)
        clock = cast(str, self.conservative_known_time)
        clock_format = "%H:%M:%S.%f" if "." in clock else "%H:%M:%S"
        parsed = pd.to_datetime(
            dates.astype(str) + " " + clock,
            format=f"%Y%m%d {clock_format}",
            errors="raise",
        )
        known = pd.Series(parsed, index=events.index).dt.tz_localize(
            self.timezone,
            ambiguous="raise",
            nonexistent="raise",
        )
        if (known < events).any():
            raise ValueError("conservative post-close knowledge time precedes an event")
        return known

    def earliest_execution_at(
        self,
        trading_date: object,
        *,
        calendar: TradingCalendar,
    ) -> pd.Timestamp:
        if calendar.timezone != self.timezone:
            raise ValueError(
                "availability policy and trading calendar timezones differ"
            )
        date_text = _date_text(trading_date)
        try:
            position = calendar.sessions.index(date_text)
        except ValueError as exc:
            raise ExecutionTimingError(
                f"source_session_not_in_calendar:{date_text}"
            ) from exc
        target = position + self.earliest_execution_session_offset
        if target >= len(calendar.sessions):
            raise ExecutionTimingError(
                f"future_execution_session_unavailable:{date_text}"
            )
        first_open = calendar.session.intervals[0][0]
        return pd.Timestamp(
            f"{calendar.sessions[target]} {first_open}", tz=self.timezone
        )

    def assert_execution_allowed(
        self,
        trading_date: object,
        *,
        entry_at: object,
        calendar: TradingCalendar,
    ) -> None:
        entry = pd.Timestamp(entry_at)
        if entry.tzinfo is None:
            raise ValueError("entry time must be timezone-aware")
        earliest = self.earliest_execution_at(trading_date, calendar=calendar)
        if entry.tz_convert(self.timezone) < earliest:
            raise ExecutionTimingError(
                "entry_precedes_static_data_execution_boundary:"
                f"entry={entry.isoformat()}:earliest={earliest.isoformat()}"
            )


@dataclass(frozen=True, slots=True)
class ProviderRegimeSpec:
    profile_id: str
    provider_id: str
    venue: str
    source_kind: SourceKind | str
    effective_from: str
    effective_to: str
    observed_fields: tuple[str, ...]
    availability: SourceAvailabilityPolicy
    capabilities: tuple[CapabilityAssessment, ...]
    evidence: tuple[SemanticEvidence, ...]
    enum_semantics: tuple[EnumFieldSemantics, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_kind", SourceKind(self.source_kind))
        for name in ("profile_id", "provider_id", "venue"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"provider regime {name} must not be empty")
        start = _date_text(self.effective_from)
        end = _date_text(self.effective_to)
        if start > end:
            raise ValueError("provider regime start must not exceed end")
        object.__setattr__(self, "effective_from", start)
        object.__setattr__(self, "effective_to", end)
        fields = tuple(str(item).strip() for item in self.observed_fields)
        if (
            not fields
            or any(not item for item in fields)
            or len(fields) != len(set(fields))
        ):
            raise ValueError(
                "provider regime observed fields must be non-empty and unique"
            )
        object.__setattr__(self, "observed_fields", fields)
        enum_fields = [item.source_field for item in self.enum_semantics]
        if len(enum_fields) != len(set(enum_fields)):
            raise ValueError("provider enum semantic fields must be unique")
        unknown_enum_fields = sorted(set(enum_fields).difference(fields))
        if unknown_enum_fields:
            raise ValueError(
                "provider enum semantics reference unobserved fields:"
                + ",".join(unknown_enum_fields)
            )
        capability_names = [
            DataCapability(item.capability) for item in self.capabilities
        ]
        if not capability_names or len(capability_names) != len(set(capability_names)):
            raise ValueError(
                "provider regime capabilities must be non-empty and unique"
            )
        evidence_ids = [item.evidence_id for item in self.evidence]
        if not evidence_ids or len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("provider regime evidence must be non-empty and unique")
        unknown_evidence = sorted(
            {
                evidence_id
                for capability in self.capabilities
                for evidence_id in capability.evidence_ids
            }.difference(evidence_ids)
        )
        if unknown_evidence:
            raise ValueError(
                "capabilities reference unknown evidence:" + ",".join(unknown_evidence)
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ProviderRegimeSpec":
        _strict_keys(
            value,
            required={
                "profile_id",
                "provider_id",
                "venue",
                "source_kind",
                "effective_from",
                "effective_to",
                "observed_fields",
                "availability",
                "capabilities",
                "evidence",
            },
            optional={"enum_semantics"},
            path="provider_semantics.profiles[]",
        )
        availability = value["availability"]
        if not isinstance(availability, Mapping):
            raise TypeError("provider regime availability must be an object")
        return cls(
            profile_id=str(value["profile_id"]),
            provider_id=str(value["provider_id"]),
            venue=str(value["venue"]),
            source_kind=str(value["source_kind"]),
            effective_from=str(value["effective_from"]),
            effective_to=str(value["effective_to"]),
            observed_fields=_strings(value["observed_fields"], path="observed_fields"),
            availability=SourceAvailabilityPolicy.from_mapping(availability),
            capabilities=tuple(
                CapabilityAssessment.from_mapping(item)
                for item in _objects(value["capabilities"], path="capabilities")
            ),
            evidence=tuple(
                SemanticEvidence.from_mapping(item)
                for item in _objects(value["evidence"], path="evidence")
            ),
            enum_semantics=tuple(
                EnumFieldSemantics.from_mapping(item)
                for item in _objects(
                    value.get("enum_semantics", []), path="enum_semantics"
                )
            ),
        )

    @property
    def capability_map(self) -> Mapping[DataCapability, CapabilityAssessment]:
        return MappingProxyType(
            {DataCapability(item.capability): item for item in self.capabilities}
        )

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    @property
    def enum_semantics_map(self) -> Mapping[str, Mapping[int, str]]:
        return MappingProxyType(
            {item.source_field: item.value_map for item in self.enum_semantics}
        )

    def enum_meaning(self, source_field: str, raw_value: int) -> str:
        field = self.enum_semantics_map.get(source_field)
        if field is None:
            raise SemanticAdmissionError(
                f"provider_enum_field_unregistered:{self.profile_id}:{source_field}"
            )
        meaning = field.get(int(raw_value))
        if meaning is None:
            raise SemanticAdmissionError(
                f"provider_enum_value_unregistered:{self.profile_id}:"
                f"{source_field}={raw_value}"
            )
        return meaning

    def to_dict(self) -> dict[str, object]:
        source_kind = self.source_kind
        if not isinstance(source_kind, SourceKind):  # pragma: no cover
            raise RuntimeError("source kind was not normalized")
        return {
            "profile_id": self.profile_id,
            "provider_id": self.provider_id,
            "venue": self.venue,
            "source_kind": source_kind.value,
            "effective_from": self.effective_from,
            "effective_to": self.effective_to,
            "observed_fields": list(self.observed_fields),
            "availability": self.availability.to_dict(),
            "capabilities": [item.to_dict() for item in self.capabilities],
            "evidence": [item.to_dict() for item in self.evidence],
            "enum_semantics": [item.to_dict() for item in self.enum_semantics],
        }

    def active_on(self, trading_date: object) -> bool:
        date_text = _date_text(trading_date)
        return self.effective_from <= date_text <= self.effective_to

    def require_capabilities(
        self,
        capabilities: Sequence[DataCapability | str],
        *,
        intended_use: IntendedUse | str,
    ) -> None:
        use = IntendedUse(intended_use)
        missing: list[str] = []
        rejected: list[str] = []
        for item in capabilities:
            capability = DataCapability(item)
            assessment = self.capability_map.get(capability)
            if assessment is None:
                missing.append(capability.value)
            elif not assessment.permits(use):
                rejected.append(
                    f"{capability.value}={CapabilityStatus(assessment.status).value}"
                )
        if missing or rejected:
            raise SemanticAdmissionError(
                f"provider_capability_rejected:{self.profile_id}:{use.value}:"
                f"missing={','.join(sorted(missing))}:"
                f"rejected={','.join(sorted(rejected))}"
            )


@dataclass(frozen=True, slots=True)
class ProviderSemanticCatalog:
    schema_version: str
    catalog_id: str
    profiles: tuple[ProviderRegimeSpec, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "provider-semantics/v1":
            raise ValueError("unsupported provider semantics schema_version")
        if not self.catalog_id.strip() or not self.profiles:
            raise ValueError("provider semantics catalog id and profiles are required")
        profile_ids = [item.profile_id for item in self.profiles]
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("provider semantic profile ids must be unique")
        groups: dict[tuple[str, str, SourceKind], list[ProviderRegimeSpec]] = {}
        for profile in self.profiles:
            key = (
                profile.provider_id,
                profile.venue,
                SourceKind(profile.source_kind),
            )
            groups.setdefault(key, []).append(profile)
        for key, members in groups.items():
            ordered = sorted(members, key=lambda item: item.effective_from)
            for left, right in zip(ordered, ordered[1:], strict=False):
                if right.effective_from <= left.effective_to:
                    raise ValueError(
                        "provider semantic regimes overlap:"
                        f"{key}:{left.profile_id}:{right.profile_id}"
                    )

    @classmethod
    def load_json(cls, path: str | Path) -> "ProviderSemanticCatalog":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("provider semantics configuration must be an object")
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ProviderSemanticCatalog":
        _strict_keys(
            value,
            required={"schema_version", "catalog_id", "profiles"},
            optional=set(),
            path="provider_semantics",
        )
        return cls(
            schema_version=str(value["schema_version"]),
            catalog_id=str(value["catalog_id"]),
            profiles=tuple(
                ProviderRegimeSpec.from_mapping(item)
                for item in _objects(
                    value["profiles"], path="provider_semantics.profiles"
                )
            ),
        )

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "catalog_id": self.catalog_id,
            "profiles": [item.to_dict() for item in self.profiles],
        }

    def resolve(
        self,
        *,
        provider_id: str,
        venue: str,
        source_kind: SourceKind | str,
        trading_date: object,
    ) -> ProviderRegimeSpec:
        kind = SourceKind(source_kind)
        matches = [
            item
            for item in self.profiles
            if item.provider_id == provider_id
            and item.venue == venue
            and SourceKind(item.source_kind) is kind
            and item.active_on(trading_date)
        ]
        if len(matches) != 1:
            raise SemanticAdmissionError(
                "provider_regime_resolution_failed:"
                f"provider={provider_id}:venue={venue}:source={kind.value}:"
                f"date={_date_text(trading_date)}:matches={len(matches)}"
            )
        return matches[0]


def _date_text(value: object) -> str:
    text = str(value).strip().replace("-", "").replace("/", "").replace(".", "")
    if not text.isdigit() or len(text) != 8:
        raise ValueError(f"trading date must be YYYYMMDD:{value}")
    parsed = pd.Timestamp(text)
    result = parsed.strftime("%Y%m%d")
    if result != text:
        raise ValueError(f"invalid trading date:{value}")
    return cast(str, result)


def _strict_keys(
    value: Mapping[str, object],
    *,
    required: set[str],
    optional: set[str],
    path: str,
) -> None:
    actual = set(value)
    missing = sorted(required.difference(actual))
    unknown = sorted(actual.difference(required | optional))
    if missing or unknown:
        raise ValueError(f"{path} schema differs:missing={missing},unknown={unknown}")


def _objects(value: object, *, path: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise TypeError(f"{path} must be a list of objects")
    return tuple(cast(Mapping[str, object], item) for item in value)


def _strings(value: object, *, path: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{path} must be a list of strings")
    if not all(isinstance(item, str) for item in value):
        raise TypeError(f"{path} must contain only strings")
    result = tuple(cast(str, item) for item in value)
    if any(not item.strip() for item in result):
        raise ValueError(f"{path} must not contain empty strings")
    return result


__all__ = [
    "AvailabilityMode",
    "CapabilityAssessment",
    "CapabilityStatus",
    "DataCapability",
    "EvidenceKind",
    "EnumFieldSemantics",
    "EnumValueSemantic",
    "ExecutionTimingError",
    "IntendedUse",
    "ProviderRegimeSpec",
    "ProviderSemanticCatalog",
    "SemanticAdmissionError",
    "SemanticEvidence",
    "SourceAvailabilityPolicy",
    "SourceKind",
]
