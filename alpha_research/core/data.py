from __future__ import annotations

import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, cast

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)

from .frequency import AvailabilitySpec, FrequencySpec
from .hashing import hash_file, hash_frame, hash_json, require_sha256


class FieldRole(str, Enum):
    KEY = "key"
    FEATURE = "feature"
    MARKET = "market"
    STATUS = "status"
    AVAILABILITY = "availability"
    LABEL = "label"


class DataRole(str, Enum):
    RESEARCH = "research"
    TRAIN = "train"
    VALIDATION = "validation"
    LOCAL_TEST = "local_test"
    EXTERNAL_HOLDOUT = "external_holdout"
    LIVE = "live"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    name: str
    dtype: str
    role: FieldRole
    nullable: bool = True
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    applicability_field: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", FieldRole(self.role))
        if not self.name or self.dtype not in {
            "float",
            "integer",
            "number",
            "string",
            "boolean",
            "datetime_tz",
        }:
            raise ValueError(f"invalid field specification:{self.name}:{self.dtype}")
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError(f"field range is inverted:{self.name}")
        if self.role is FieldRole.LABEL:
            raise ValueError("data adapters cannot expose label-role fields")
        if self.applicability_field == self.name:
            raise ValueError("field applicability cannot reference itself")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "role": self.role.value,
            "nullable": self.nullable,
            "unit": self.unit,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "applicability_field": self.applicability_field,
        }


@dataclass(frozen=True, slots=True)
class DataSchema:
    schema_id: str
    version: str
    fields: tuple[FieldSpec, ...]
    timestamp_field: str = "timestamp"
    security_field: str = "security"
    key_fields: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.schema_id or not self.version or not self.fields:
            raise ValueError("schema id, version, and fields are required")
        names = tuple(field.name for field in self.fields)
        if len(set(names)) != len(names):
            raise ValueError("schema field names must be unique")
        if self.timestamp_field not in names or self.security_field not in names:
            raise ValueError("schema must include timestamp and security fields")
        normalized_keys = (
            (self.timestamp_field, self.security_field)
            if self.key_fields is None
            else tuple(self.key_fields)
        )
        if not normalized_keys or len(set(normalized_keys)) != len(normalized_keys):
            raise ValueError("schema key fields must be non-empty and unique")
        missing_keys = sorted(set(normalized_keys).difference(names))
        if missing_keys:
            raise ValueError("schema key fields are missing:" + ",".join(missing_keys))
        if (
            self.timestamp_field not in normalized_keys
            or self.security_field not in normalized_keys
        ):
            raise ValueError("schema keys must include timestamp and security fields")
        for name in normalized_keys:
            if self.field_map[name].role is not FieldRole.KEY:
                raise ValueError(f"schema key field must have key role:{name}")
        object.__setattr__(self, "key_fields", normalized_keys)
        for field in self.fields:
            if (
                field.applicability_field is not None
                and field.applicability_field not in names
            ):
                raise ValueError(
                    f"field applicability reference is missing:{field.name}:{field.applicability_field}"
                )

    @property
    def field_map(self) -> Mapping[str, FieldSpec]:
        return MappingProxyType({field.name: field for field in self.fields})

    @property
    def feature_fields(self) -> tuple[str, ...]:
        return tuple(
            field.name
            for field in self.fields
            if field.role in {FieldRole.FEATURE, FieldRole.MARKET, FieldRole.STATUS}
        )

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "version": self.version,
            "timestamp_field": self.timestamp_field,
            "security_field": self.security_field,
            "key_fields": list(self.key_fields or ()),
            "fields": [field.to_dict() for field in self.fields],
        }

    def validate_frame(
        self,
        frame: pd.DataFrame,
        *,
        requested_fields: tuple[str, ...],
    ) -> None:
        required = {
            *(self.key_fields or ()),
            "effective_at",
            "known_at",
            *requested_fields,
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError("batch_missing_schema_fields:" + ",".join(missing))
        unknown = sorted(set(requested_fields).difference(self.field_map))
        if unknown:
            raise ValueError("request_unknown_schema_fields:" + ",".join(unknown))
        for name in required:
            field = self.field_map.get(name)
            if field is None:
                raise ValueError(f"batch_metadata_field_not_in_schema:{name}")
            series = frame[name]
            if not _dtype_matches(series, field.dtype):
                raise TypeError(
                    f"batch_field_dtype_mismatch:{name}:{series.dtype}:{field.dtype}"
                )
            if not field.nullable and series.isna().any():
                raise ValueError(f"batch_nonnullable_field_has_missing:{name}")


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    dataset_id: str
    version: str
    owner: str
    description: str
    storage_format: str
    schema_hash: str
    frequency: FrequencySpec
    availability: AvailabilitySpec

    def __post_init__(self) -> None:
        for name in (
            "dataset_id",
            "version",
            "owner",
            "description",
            "storage_format",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"dataset {name} must not be empty")
        require_sha256(self.schema_hash, name="schema_hash")

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "owner": self.owner,
            "description": self.description,
            "storage_format": self.storage_format,
            "schema_hash": self.schema_hash,
            "frequency": self.frequency.to_dict(),
            "availability": self.availability.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SourceAsset:
    logical_name: str
    uri: str
    sha256: str
    size_bytes: int
    mtime_ns: int
    mode: int

    def __post_init__(self) -> None:
        if not self.logical_name or not self.uri or self.size_bytes < 0:
            raise ValueError("invalid source asset")
        require_sha256(self.sha256, name="source asset sha256")
        if not stat.S_ISREG(self.mode):
            raise ValueError("source asset must be a regular file")

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        logical_name: str,
        expected_sha256: str | None = None,
    ) -> "SourceAsset":
        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise ValueError(
                f"source asset must be a regular non-symlink file:{source}"
            )
        before = source.stat()
        digest = hash_file(source)
        after = source.stat()
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after:
            raise RuntimeError(f"source asset changed while hashing:{source}")
        if expected_sha256 is not None and digest != require_sha256(
            expected_sha256, name=f"expected digest for {logical_name}"
        ):
            raise ValueError(f"source asset digest mismatch:{logical_name}")
        return cls(
            logical_name=logical_name,
            uri=str(source.resolve()),
            sha256=digest,
            size_bytes=after.st_size,
            mtime_ns=after.st_mtime_ns,
            mode=after.st_mode,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "logical_name": self.logical_name,
            "uri": self.uri,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "mode": self.mode,
        }

    def identity_payload(self) -> dict[str, object]:
        """Stable content identity; filesystem location is provenance, not identity."""
        return {
            "logical_name": self.logical_name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class DatasetSnapshot:
    snapshot_id: str
    dataset_id: str
    dataset_version: str
    dataset_spec_hash: str
    schema_hash: str
    source_assets: tuple[SourceAsset, ...]
    source_vintage_at: str
    created_at: str

    def __post_init__(self) -> None:
        require_sha256(self.snapshot_id, name="snapshot_id")
        require_sha256(self.dataset_spec_hash, name="dataset_spec_hash")
        require_sha256(self.schema_hash, name="schema_hash")
        _aware(self.source_vintage_at, name="source_vintage_at")
        _aware(self.created_at, name="created_at")
        names = [asset.logical_name for asset in self.source_assets]
        if not self.dataset_id or not self.dataset_version or not self.source_assets:
            raise ValueError("snapshot dataset identity and assets are required")
        if len(names) != len(set(names)):
            raise ValueError("snapshot source asset names must be unique")
        if self.snapshot_id != hash_json(self.identity_payload()):
            raise ValueError("snapshot id does not match content manifest")

    @classmethod
    def create(
        cls,
        *,
        dataset: DatasetSpec,
        schema: DataSchema,
        source_assets: tuple[SourceAsset, ...],
        source_vintage_at: object,
        created_at: object,
    ) -> "DatasetSnapshot":
        if dataset.schema_hash != schema.content_hash:
            raise ValueError("dataset schema hash differs from supplied schema")
        payload = {
            "dataset_id": dataset.dataset_id,
            "dataset_version": dataset.version,
            "dataset_spec_hash": dataset.content_hash,
            "schema_hash": schema.content_hash,
            "source_content": [
                asset.identity_payload()
                for asset in sorted(source_assets, key=lambda item: item.logical_name)
            ],
            "source_vintage_at": _aware(
                source_vintage_at, name="source_vintage_at"
            ).isoformat(),
        }
        return cls(
            snapshot_id=hash_json(payload),
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.version,
            dataset_spec_hash=dataset.content_hash,
            schema_hash=schema.content_hash,
            source_assets=tuple(
                sorted(source_assets, key=lambda item: item.logical_name)
            ),
            source_vintage_at=payload["source_vintage_at"],
            created_at=_aware(created_at, name="created_at").isoformat(),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "dataset_spec_hash": self.dataset_spec_hash,
            "schema_hash": self.schema_hash,
            "source_content": [
                asset.identity_payload() for asset in self.source_assets
            ],
            "source_vintage_at": self.source_vintage_at,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.manifest_dict(),
            **self.materialization_payload(),
        }

    def manifest_dict(self) -> dict[str, object]:
        """Location-independent immutable content manifest."""
        return {**self.identity_payload(), "snapshot_id": self.snapshot_id}

    def materialization_payload(self) -> dict[str, object]:
        """Auditable physical replica used to read the content snapshot."""
        return {
            "source_assets": [asset.to_dict() for asset in self.source_assets],
            "created_at": self.created_at,
        }

    @property
    def materialization_hash(self) -> str:
        return hash_json(self.materialization_payload())


@dataclass(frozen=True, slots=True)
class DataRequest:
    dataset_id: str
    snapshot_id: str
    fields: tuple[str, ...]
    start: str
    end: str
    as_of: str
    role: DataRole
    frequency: FrequencySpec
    securities: tuple[str, ...] | None = None
    batch_rows: int = 250_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", DataRole(self.role))
        require_sha256(self.snapshot_id, name="request snapshot_id")
        if not self.dataset_id or not self.fields:
            raise ValueError("request dataset and fields are required")
        if len(self.fields) != len(set(self.fields)):
            raise ValueError("request fields must be unique")
        start = _aware(self.start, name="request start")
        end = _aware(self.end, name="request end")
        as_of = _aware(self.as_of, name="request as_of")
        if start > end or as_of < start:
            raise ValueError("request time range/as_of is invalid")
        object.__setattr__(self, "start", start.isoformat())
        object.__setattr__(self, "end", end.isoformat())
        object.__setattr__(self, "as_of", as_of.isoformat())
        if self.securities is not None:
            normalized = tuple(_security(item) for item in self.securities)
            if len(normalized) != len(set(normalized)):
                raise ValueError("request securities must be unique")
            object.__setattr__(self, "securities", normalized)
        if not isinstance(self.batch_rows, int) or self.batch_rows <= 0:
            raise ValueError("batch_rows must be a positive integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "snapshot_id": self.snapshot_id,
            "fields": list(self.fields),
            "start": self.start,
            "end": self.end,
            "as_of": self.as_of,
            "role": self.role.value,
            "frequency": self.frequency.to_dict(),
            "securities": (
                list(self.securities) if self.securities is not None else None
            ),
            "batch_rows": self.batch_rows,
        }


@dataclass(frozen=True, slots=True)
class DataBatch:
    batch_id: str
    sequence: int
    snapshot_id: str
    schema_hash: str
    request_hash: str
    frequency_hash: str
    calendar_id: str
    timezone: str
    availability: AvailabilitySpec
    as_of: str
    requested_fields: tuple[str, ...]
    frame: pd.DataFrame

    def __post_init__(self) -> None:
        for name in (
            "batch_id",
            "snapshot_id",
            "schema_hash",
            "request_hash",
            "frequency_hash",
        ):
            require_sha256(str(getattr(self, name)), name=name)
        if self.sequence < 0 or not self.calendar_id or not self.timezone:
            raise ValueError("invalid batch sequence/calendar/timezone")
        _aware(self.as_of, name="batch as_of")
        copied = pd.DataFrame(self.frame).copy(deep=True)
        object.__setattr__(self, "frame", copied)
        expected = hash_json(
            {
                "sequence": self.sequence,
                "snapshot_id": self.snapshot_id,
                "schema_hash": self.schema_hash,
                "request_hash": self.request_hash,
                "frequency_hash": self.frequency_hash,
                "as_of": self.as_of,
                "requested_fields": list(self.requested_fields),
                "frame_hash": hash_frame(copied),
            }
        )
        if self.batch_id != expected:
            raise ValueError("batch id does not match content")

    def verify_content(self) -> None:
        expected = hash_json(
            {
                "sequence": self.sequence,
                "snapshot_id": self.snapshot_id,
                "schema_hash": self.schema_hash,
                "request_hash": self.request_hash,
                "frequency_hash": self.frequency_hash,
                "as_of": self.as_of,
                "requested_fields": list(self.requested_fields),
                "frame_hash": hash_frame(self.frame),
            }
        )
        if expected != self.batch_id:
            raise RuntimeError("data batch content changed after construction")

    @classmethod
    def create(
        cls,
        *,
        sequence: int,
        snapshot: DatasetSnapshot,
        schema: DataSchema,
        request: DataRequest,
        availability: AvailabilitySpec,
        frame: pd.DataFrame,
    ) -> "DataBatch":
        if snapshot.snapshot_id != request.snapshot_id:
            raise ValueError("request snapshot does not match adapter snapshot")
        if request.dataset_id != snapshot.dataset_id:
            raise ValueError("request dataset does not match snapshot")
        if request.frequency.content_hash == "":  # pragma: no cover
            raise ValueError("frequency hash is empty")
        schema.validate_frame(frame, requested_fields=request.fields)
        visible = availability.visible_as_of(
            frame,
            as_of=pd.Timestamp(request.as_of),
            allow_latest_replay=request.role is DataRole.RESEARCH,
            resolve_revisions=False,
        )
        if len(visible) != len(frame):
            raise ValueError("adapter emitted records unavailable at request as_of")
        request_hash = hash_json(request.to_dict())
        payload = {
            "sequence": sequence,
            "snapshot_id": snapshot.snapshot_id,
            "schema_hash": schema.content_hash,
            "request_hash": request_hash,
            "frequency_hash": request.frequency.content_hash,
            "as_of": request.as_of,
            "requested_fields": list(request.fields),
            "frame_hash": hash_frame(frame),
        }
        return cls(
            batch_id=hash_json(payload),
            sequence=sequence,
            snapshot_id=snapshot.snapshot_id,
            schema_hash=schema.content_hash,
            request_hash=request_hash,
            frequency_hash=request.frequency.content_hash,
            calendar_id=request.frequency.calendar_id,
            timezone=request.frequency.timezone,
            availability=availability,
            as_of=request.as_of,
            requested_fields=request.fields,
            frame=frame,
        )

    def to_descriptor(self) -> dict[str, object]:
        self.verify_content()
        return {
            "batch_id": self.batch_id,
            "sequence": self.sequence,
            "snapshot_id": self.snapshot_id,
            "schema_hash": self.schema_hash,
            "request_hash": self.request_hash,
            "frequency_hash": self.frequency_hash,
            "calendar_id": self.calendar_id,
            "timezone": self.timezone,
            "availability": self.availability.to_dict(),
            "as_of": self.as_of,
            "requested_fields": list(self.requested_fields),
            "row_count": len(self.frame),
            "frame_hash": hash_frame(self.frame),
        }


def standard_market_schema(
    *,
    schema_id: str,
    version: str,
    feature_fields: Mapping[str, str],
    applicability: Mapping[str, str] | None = None,
    bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    units: Mapping[str, str] | None = None,
) -> DataSchema:
    fields = [
        FieldSpec("timestamp", "datetime_tz", FieldRole.KEY, nullable=False),
        FieldSpec("security", "string", FieldRole.KEY, nullable=False),
        FieldSpec(
            "effective_at", "datetime_tz", FieldRole.AVAILABILITY, nullable=False
        ),
        FieldSpec("known_at", "datetime_tz", FieldRole.AVAILABILITY, nullable=False),
    ]
    applicability = dict(applicability or {})
    bounds = dict(bounds or {})
    units = dict(units or {})
    unknown_constraints = sorted(
        (set(applicability) | set(bounds) | set(units)).difference(feature_fields)
    )
    if unknown_constraints:
        raise ValueError(
            "schema constraints reference unknown features:"
            + ",".join(unknown_constraints)
        )
    fields.extend(
        FieldSpec(
            name,
            dtype,
            FieldRole.MARKET,
            unit=units.get(name),
            minimum=bounds.get(name, (None, None))[0],
            maximum=bounds.get(name, (None, None))[1],
            applicability_field=applicability.get(name),
        )
        for name, dtype in feature_fields.items()
    )
    return DataSchema(schema_id=schema_id, version=version, fields=tuple(fields))


def _dtype_matches(series: pd.Series, expected: str) -> bool:
    if expected == "float":
        return cast(bool, is_float_dtype(series.dtype))
    if expected == "integer":
        return cast(
            bool,
            is_integer_dtype(series.dtype) and not is_bool_dtype(series.dtype),
        )
    if expected == "number":
        return cast(
            bool,
            is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype),
        )
    if expected == "string":
        return cast(bool, is_string_dtype(series.dtype) or series.dtype == object)
    if expected == "boolean":
        return cast(bool, is_bool_dtype(series.dtype))
    if expected == "datetime_tz":
        return (
            is_datetime64_any_dtype(series.dtype)
            and getattr(series.dt, "tz", None) is not None
        )
    return False


def _aware(value: object, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp


def _security(value: object) -> str:
    text = str(value).strip()
    return text.zfill(6) if text.isdigit() and len(text) <= 6 else text


__all__ = [
    "DataBatch",
    "DataRequest",
    "DataRole",
    "DataSchema",
    "DatasetSnapshot",
    "DatasetSpec",
    "FieldRole",
    "FieldSpec",
    "SourceAsset",
    "standard_market_schema",
]
