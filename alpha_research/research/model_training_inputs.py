"""Strict, content-addressed inputs for governed model training.

This module intentionally contains no artifact resolver and performs no file
I/O.  ``ModelTrainingInputManifestV2`` is the immutable wire contract;
``LoadedModelTrainingInputs`` verifies objects that a later, separately
governed resolver has already loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
import pyarrow as pa
import pyarrow.parquet as pq

from alpha_research.core.frequency import TradingCalendar
from alpha_research.core.hashing import (
    canonical_json_bytes,
    hash_frame,
    hash_json,
    require_sha256,
)
from alpha_research.labels import LabelResult
from alpha_research.models.selection_spec import NestedPurgedSelectionSpec
from alpha_research.research.lineage import (
    ModelFeatureBinding,
    ResearchScientificLineageManifestV2,
)
from alpha_research.research.spec import ResearchStage
from alpha_research.validation import (
    ValidationReceipt,
    ValidationReceiptVerifier,
    ValidationSpec,
    validation_calendar_hash,
)
from factor_production.v5.artifacts import ArtifactRecord

if TYPE_CHECKING:
    from alpha_research.experiments import ExperimentSpec
    from alpha_research.research.execution import (
        ScientificStageRequest,
    )


_DATAFRAME_REFERENCE_SCHEMA = "model-training-dataframe-artifact-reference/v2"
_INPUT_MANIFEST_SCHEMA = "model-training-input-manifest/v2"
SCORING_ELIGIBILITY_SOURCE_SCHEMA = "scoring-eligibility-source/v2"
_PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
_MAXIMUM_MANIFEST_WIRE_BYTES = 16 * 1024 * 1024
_SAFE_ERROR_CODE = re.compile(r"[a-z][a-z0-9_]*\Z")
_FRAME_ROLES = frozenset(
    {
        "label_diagnostics",
        "label_validity",
        "label_values",
        "label_windows",
        "model_feature",
        "scoring_eligibility",
    }
)
_MODEL_TRAINING_PARENTS = frozenset(
    {
        ResearchStage.FACTOR_GENERATION,
        ResearchStage.LABEL_BUILDING,
        ResearchStage.VALIDATION_SPLIT,
    }
)
_ELIGIBILITY_SOURCE_STAGE = ResearchStage.FACTOR_GENERATION


class ModelTrainingInputError(ValueError):
    """Stable fail-closed error raised by model-training input contracts."""

    def __init__(self, code: str, detail: str) -> None:
        if _SAFE_ERROR_CODE.fullmatch(code) is None:
            raise ValueError("model-training input error code is invalid")
        self.code = code
        self.detail = str(detail)
        super().__init__(f"{code}:{self.detail}")


def _error(code: str, detail: str) -> ModelTrainingInputError:
    return ModelTrainingInputError(code, detail)


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise _error("invalid_hash", f"{name} must be text")
    try:
        return cast(str, require_sha256(value, name=name))
    except ValueError as exc:
        raise _error("invalid_hash", str(exc)) from exc


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise _error("invalid_text", f"{name} must be non-empty stripped text")
    return value


def _integer(value: object, *, name: str, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise _error("invalid_integer", f"{name} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise _error("invalid_integer", f"{name} must be {qualifier}")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise _error("invalid_boolean", f"{name} must be boolean")
    return value


def _object(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise _error("invalid_object", f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _object_array(value: object, *, name: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) and all(isinstance(key, str) for key in item)
        for item in value
    ):
        raise _error("invalid_array", f"{name} must be an object array")
    return tuple(cast(Mapping[str, object], item) for item in value)


def _exact_fields(
    value: Mapping[str, object], expected: frozenset[str], *, name: str
) -> None:
    fields = frozenset(value)
    if fields != expected:
        missing = ",".join(sorted(expected - fields)) or "-"
        extra = ",".join(sorted(fields - expected)) or "-"
        raise _error(
            "wire_fields_differ",
            f"{name} fields differ; missing={missing}; extra={extra}",
        )


def _stage_name(stage: ResearchStage) -> str:
    return str(stage.value)


def _scalar_identity(value: object) -> Mapping[str, object]:
    if value is None:
        return {"type": "none", "value": None}
    if isinstance(value, pd.Timestamp):
        return {
            "type": "pandas.Timestamp",
            "timezone": None if value.tz is None else str(value.tz),
            "value": value.isoformat(),
        }
    if isinstance(value, np.generic):
        return {
            "type": f"numpy.{value.dtype}",
            "value": repr(value.item()),
        }
    if isinstance(value, float) and not np.isfinite(value):
        return {"type": "float", "value": repr(value)}
    if isinstance(value, (str, int, float, bool)):
        return {"type": type(value).__name__, "value": value}
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": repr(value),
    }


def _dtype_identity(dtype: object) -> Mapping[str, object]:
    descriptor: dict[str, object] = {
        "class": f"{type(dtype).__module__}.{type(dtype).__qualname__}",
        "text": str(dtype),
    }
    if isinstance(dtype, pd.CategoricalDtype):
        categories = dtype.categories
        descriptor["ordered"] = bool(dtype.ordered)
        descriptor["categories_dtype"] = str(categories.dtype)
        descriptor["categories_hash"] = hashlib.sha256(
            pd.util.hash_pandas_object(
                categories, index=False, categorize=False
            ).values.tobytes()
        ).hexdigest()
    storage = getattr(dtype, "storage", None)
    if storage is not None:
        descriptor["storage"] = str(storage)
    pyarrow_dtype = getattr(dtype, "pyarrow_dtype", None)
    if pyarrow_dtype is not None:
        descriptor["pyarrow_dtype"] = str(pyarrow_dtype)
    timezone = getattr(dtype, "tz", None)
    if timezone is not None:
        descriptor["timezone"] = str(timezone)
    return descriptor


def _axis_hash(axis: pd.Index) -> str:
    if isinstance(axis, pd.MultiIndex):
        dtype_identity: object = [_dtype_identity(level.dtype) for level in axis.levels]
        names = [_scalar_identity(name) for name in axis.names]
        axis_kind = "multi"
    else:
        dtype_identity = _dtype_identity(axis.dtype)
        names = [_scalar_identity(axis.name)]
        axis_kind = "single"
    metadata = {
        "schema_version": "pandas-semantic-axis-identity/v1",
        "axis_kind": axis_kind,
        "dtype_identity": dtype_identity,
        "names": names,
        "nlevels": axis.nlevels,
        "length": len(axis),
    }
    digest = hashlib.sha256(canonical_json_bytes(metadata))
    try:
        digest.update(
            pd.util.hash_pandas_object(
                axis, index=False, categorize=False
            ).values.tobytes()
        )
    except (TypeError, ValueError) as exc:
        raise _error("invalid_frame_axis", "frame axis cannot be hashed") from exc
    return digest.hexdigest()


def _dtypes_hash(frame: pd.DataFrame) -> str:
    return cast(
        str,
        hash_json(
            {
                "schema_version": "pandas-dtypes-identity/v1",
                "dtypes": [_dtype_identity(dtype) for dtype in frame.dtypes],
            }
        ),
    )


def strong_hash_frame(frame: pd.DataFrame) -> str:
    """Hash values plus exact typed pandas axes and dtype metadata."""

    value = pd.DataFrame(frame)
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema_version": "strong-pandas-frame-identity/v1",
                "index_hash": _axis_hash(value.index),
                "columns_hash": _axis_hash(value.columns),
                "dtypes_hash": _dtypes_hash(value),
                "shape": [len(value.index), len(value.columns)],
            }
        )
    )
    try:
        digest.update(
            pd.util.hash_pandas_object(
                value, index=True, categorize=False
            ).values.tobytes()
        )
    except (TypeError, ValueError) as exc:
        raise _error("invalid_frame", "frame values cannot be hashed") from exc
    return digest.hexdigest()


def _parquet_uncompressed_bytes(parquet: pq.ParquetFile) -> int:
    metadata = parquet.metadata
    if metadata is None:
        raise _error("invalid_parquet", "parquet metadata is missing")
    total = 0
    for row_group_offset in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_offset)
        for column_offset in range(row_group.num_columns):
            total += int(row_group.column(column_offset).total_uncompressed_size)
    if total <= 0:
        raise _error("invalid_parquet", "parquet uncompressed size is invalid")
    return total


@dataclass(frozen=True, slots=True)
class ScoringEligibilityPolicyV1:
    """Strict point-in-time policy; label-dependent masks are forbidden."""

    policy_id: str
    version: str
    universe_membership_hash: str
    availability_policy_hash: str
    timestamp_policy_hash: str
    source_data_hashes: tuple[str, ...]
    requires_label_data: bool = False
    schema_version: str = "scoring-eligibility-policy/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "scoring-eligibility-policy/v1":
            raise _error("unsupported_schema", "unsupported eligibility policy")
        object.__setattr__(self, "policy_id", _text(self.policy_id, name="policy_id"))
        object.__setattr__(self, "version", _text(self.version, name="policy version"))
        for name in (
            "universe_membership_hash",
            "availability_policy_hash",
            "timestamp_policy_hash",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=f"eligibility {name}"),
            )
        sources = tuple(self.source_data_hashes)
        if not sources or sources != tuple(sorted(set(sources))):
            raise _error(
                "invalid_eligibility_policy",
                "eligibility source hashes must be non-empty, sorted and unique",
            )
        for offset, digest in enumerate(sources):
            _sha256(digest, name=f"eligibility source_data_hashes:{offset}")
        object.__setattr__(self, "source_data_hashes", sources)
        if _boolean(self.requires_label_data, name="requires_label_data") is not False:
            raise _error(
                "invalid_eligibility_policy",
                "eligibility policy may not depend on labels",
            )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "version": self.version,
            "universe_membership_hash": self.universe_membership_hash,
            "availability_policy_hash": self.availability_policy_hash,
            "timestamp_policy_hash": self.timestamp_policy_hash,
            "source_data_hashes": list(self.source_data_hashes),
            "requires_label_data": self.requires_label_data,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ScoringEligibilityPolicyV1":
        expected = frozenset(
            {
                "schema_version",
                "policy_id",
                "version",
                "universe_membership_hash",
                "availability_policy_hash",
                "timestamp_policy_hash",
                "source_data_hashes",
                "requires_label_data",
            }
        )
        _exact_fields(value, expected, name="ScoringEligibilityPolicyV1")
        raw_sources = value["source_data_hashes"]
        if not isinstance(raw_sources, list) or not all(
            isinstance(item, str) for item in raw_sources
        ):
            raise _error(
                "invalid_eligibility_policy", "source_data_hashes must be text array"
            )
        return cls(
            schema_version=_text(value["schema_version"], name="schema_version"),
            policy_id=_text(value["policy_id"], name="policy_id"),
            version=_text(value["version"], name="version"),
            universe_membership_hash=_sha256(
                value["universe_membership_hash"], name="universe_membership_hash"
            ),
            availability_policy_hash=_sha256(
                value["availability_policy_hash"], name="availability_policy_hash"
            ),
            timestamp_policy_hash=_sha256(
                value["timestamp_policy_hash"], name="timestamp_policy_hash"
            ),
            source_data_hashes=tuple(raw_sources),
            requires_label_data=_boolean(
                value["requires_label_data"], name="requires_label_data"
            ),
        )


@dataclass(frozen=True, slots=True)
class DataFrameArtifactReferenceV2:
    """Identity and shape metadata for one already-published parquet frame."""

    logical_name: str
    location: str
    payload_sha256: str
    size_bytes: int
    role: str
    frame_hash: str
    row_count: int
    column_count: int
    index_hash: str
    columns_hash: str
    dtypes_hash: str
    parquet_uncompressed_bytes: int
    media_type: str = _PARQUET_MEDIA_TYPE
    schema_version: str = _DATAFRAME_REFERENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _DATAFRAME_REFERENCE_SCHEMA:
            raise _error(
                "unsupported_schema", "unsupported DataFrame artifact reference schema"
            )
        logical_name = _text(self.logical_name, name="artifact logical_name")
        location = _text(self.location, name="artifact location")
        role = _text(self.role, name="artifact role")
        if role not in _FRAME_ROLES:
            raise _error("invalid_artifact_role", f"unsupported frame role:{role}")
        if self.media_type != _PARQUET_MEDIA_TYPE:
            raise _error(
                "invalid_media_type", "model-training frame artifact must be parquet"
            )
        payload_sha256 = _sha256(self.payload_sha256, name="artifact payload_sha256")
        frame_hash = _sha256(self.frame_hash, name="artifact frame_hash")
        index_hash = _sha256(self.index_hash, name="artifact index_hash")
        columns_hash = _sha256(self.columns_hash, name="artifact columns_hash")
        dtypes_hash = _sha256(self.dtypes_hash, name="artifact dtypes_hash")
        size_bytes = _integer(
            self.size_bytes, name="artifact size_bytes", positive=True
        )
        row_count = _integer(self.row_count, name="artifact row_count", positive=True)
        column_count = _integer(
            self.column_count, name="artifact column_count", positive=True
        )
        uncompressed_bytes = _integer(
            self.parquet_uncompressed_bytes,
            name="artifact parquet_uncompressed_bytes",
            positive=True,
        )
        try:
            record = ArtifactRecord(
                logical_name=logical_name,
                location=location,
                sha256=payload_sha256,
                size_bytes=size_bytes,
                media_type=self.media_type,
                role=role,
            )
        except ValueError as exc:
            raise _error("unsafe_artifact_reference", str(exc)) from exc
        object.__setattr__(self, "logical_name", record.logical_name)
        object.__setattr__(self, "location", record.location)
        object.__setattr__(self, "payload_sha256", payload_sha256)
        object.__setattr__(self, "size_bytes", size_bytes)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "frame_hash", frame_hash)
        object.__setattr__(self, "row_count", row_count)
        object.__setattr__(self, "column_count", column_count)
        object.__setattr__(self, "index_hash", index_hash)
        object.__setattr__(self, "columns_hash", columns_hash)
        object.__setattr__(self, "dtypes_hash", dtypes_hash)
        object.__setattr__(self, "parquet_uncompressed_bytes", uncompressed_bytes)

    @classmethod
    def bind_frame(
        cls,
        *,
        record: ArtifactRecord,
        frame: pd.DataFrame,
        parquet_payload: bytes,
    ) -> "DataFrameArtifactReferenceV2":
        """Bind a frame only after verifying its exact parquet payload/footer."""

        if not isinstance(record, ArtifactRecord):
            raise _error("invalid_artifact_reference", "record type differs")
        if not isinstance(parquet_payload, bytes) or not parquet_payload:
            raise _error("invalid_parquet", "parquet payload must be non-empty bytes")
        if (
            len(parquet_payload) != record.size_bytes
            or hashlib.sha256(parquet_payload).hexdigest() != record.sha256
        ):
            raise _error("artifact_payload_mismatch", "parquet record differs")
        try:
            parquet = pq.ParquetFile(pa.BufferReader(parquet_payload))
            decoded = parquet.read().to_pandas()
        except (pa.ArrowException, OSError, ValueError) as exc:
            raise _error(
                "invalid_parquet", "parquet payload cannot be decoded"
            ) from exc
        value = pd.DataFrame(frame)
        if value.empty:
            raise _error("invalid_frame", "referenced frame must be non-empty")
        if strong_hash_frame(decoded) != strong_hash_frame(value):
            raise _error("artifact_payload_mismatch", "parquet frame differs")
        return cls(
            logical_name=record.logical_name,
            location=record.location,
            payload_sha256=record.sha256,
            size_bytes=record.size_bytes,
            media_type=record.media_type,
            role=record.role,
            frame_hash=strong_hash_frame(value),
            row_count=len(value.index),
            column_count=len(value.columns),
            index_hash=_axis_hash(value.index),
            columns_hash=_axis_hash(value.columns),
            dtypes_hash=_dtypes_hash(value),
            parquet_uncompressed_bytes=_parquet_uncompressed_bytes(parquet),
        )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_artifact_record(self) -> ArtifactRecord:
        """Return the existing artifact-store reference without reading it."""

        return ArtifactRecord(
            logical_name=self.logical_name,
            location=self.location,
            sha256=self.payload_sha256,
            size_bytes=self.size_bytes,
            media_type=self.media_type,
            role=self.role,
        )

    def decode_parquet_payload(
        self,
        payload: bytes,
        *,
        maximum_uncompressed_bytes: int,
    ) -> pd.DataFrame:
        """Preflight the footer, enforce budget, then decode and verify identity."""

        limit = _integer(
            maximum_uncompressed_bytes,
            name="maximum_uncompressed_bytes",
            positive=True,
        )
        if (
            not isinstance(payload, bytes)
            or len(payload) != self.size_bytes
            or hashlib.sha256(payload).hexdigest() != self.payload_sha256
        ):
            raise _error("artifact_payload_mismatch", "parquet payload differs")
        try:
            parquet = pq.ParquetFile(pa.BufferReader(payload))
            observed_uncompressed = _parquet_uncompressed_bytes(parquet)
        except (pa.ArrowException, OSError, ValueError) as exc:
            raise _error("invalid_parquet", "parquet footer is invalid") from exc
        if observed_uncompressed != self.parquet_uncompressed_bytes:
            raise _error(
                "parquet_metadata_mismatch", "parquet uncompressed size differs"
            )
        if observed_uncompressed > limit:
            raise _error("resource_budget_exceeded", "parquet frame exceeds budget")
        metadata = parquet.metadata
        if metadata is None or metadata.num_rows != self.row_count:
            raise _error("parquet_metadata_mismatch", "parquet row count differs")
        try:
            frame = parquet.read().to_pandas()
        except (pa.ArrowException, OSError, ValueError) as exc:
            raise _error("invalid_parquet", "parquet frame cannot be decoded") from exc
        _verify_loaded_reference(frame, self, name=self.logical_name)
        return frame

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "logical_name": self.logical_name,
            "location": self.location,
            "payload_sha256": self.payload_sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "role": self.role,
            "frame_hash": self.frame_hash,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "index_hash": self.index_hash,
            "columns_hash": self.columns_hash,
            "dtypes_hash": self.dtypes_hash,
            "parquet_uncompressed_bytes": self.parquet_uncompressed_bytes,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "DataFrameArtifactReferenceV2":
        expected = frozenset(
            {
                "schema_version",
                "logical_name",
                "location",
                "payload_sha256",
                "size_bytes",
                "media_type",
                "role",
                "frame_hash",
                "row_count",
                "column_count",
                "index_hash",
                "columns_hash",
                "dtypes_hash",
                "parquet_uncompressed_bytes",
            }
        )
        _exact_fields(value, expected, name="DataFrameArtifactReferenceV2")
        return cls(
            schema_version=_text(value["schema_version"], name="schema_version"),
            logical_name=_text(value["logical_name"], name="logical_name"),
            location=_text(value["location"], name="location"),
            payload_sha256=_sha256(value["payload_sha256"], name="payload_sha256"),
            size_bytes=_integer(value["size_bytes"], name="size_bytes"),
            media_type=_text(value["media_type"], name="media_type"),
            role=_text(value["role"], name="role"),
            frame_hash=_sha256(value["frame_hash"], name="frame_hash"),
            row_count=_integer(value["row_count"], name="row_count"),
            column_count=_integer(value["column_count"], name="column_count"),
            index_hash=_sha256(value["index_hash"], name="index_hash"),
            columns_hash=_sha256(value["columns_hash"], name="columns_hash"),
            dtypes_hash=_sha256(value["dtypes_hash"], name="dtypes_hash"),
            parquet_uncompressed_bytes=_integer(
                value["parquet_uncompressed_bytes"],
                name="parquet_uncompressed_bytes",
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelTrainingInputManifestV2:
    """Canonical manifest for every scientific input to model training."""

    research_run_spec_hash: str
    experiment_spec_hash: str
    scientific_lineage_manifest_hash: str
    model_training_contract_hash: str
    parent_artifact_hashes: Mapping[ResearchStage | str, str]
    selection_spec_hash: str
    selection_spec: NestedPurgedSelectionSpec
    feature_bindings: tuple[ModelFeatureBinding, ...]
    feature_frames: Mapping[str, DataFrameArtifactReferenceV2]
    label_spec_hash: str
    label_view_hash: str
    label_benchmark_hash: str | None
    label_values_hash: str
    label_windows_hash: str
    label_validity_hash: str
    label_diagnostics_hash: str
    label_values: DataFrameArtifactReferenceV2
    label_windows: DataFrameArtifactReferenceV2
    label_validity: DataFrameArtifactReferenceV2
    label_diagnostics: DataFrameArtifactReferenceV2
    validation_spec_hash: str
    validation_spec: ValidationSpec
    validation_receipt_hash: str
    validation_receipt: ValidationReceipt
    trading_calendar_content_hash: str
    trading_calendar: TradingCalendar
    scoring_eligibility_hash: str
    scoring_eligibility_policy: ScoringEligibilityPolicyV1
    scoring_eligibility_policy_hash: str
    scoring_eligibility_source_stage: ResearchStage | str
    scoring_eligibility_source_artifact_hash: str
    scoring_eligibility: DataFrameArtifactReferenceV2
    research_only: bool = True
    production_ready: bool = False
    schema_version: str = _INPUT_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _INPUT_MANIFEST_SCHEMA:
            raise _error(
                "unsupported_schema", "unsupported model-training input schema"
            )
        for name in (
            "research_run_spec_hash",
            "experiment_spec_hash",
            "scientific_lineage_manifest_hash",
            "model_training_contract_hash",
            "selection_spec_hash",
            "label_spec_hash",
            "label_view_hash",
            "label_values_hash",
            "label_windows_hash",
            "label_validity_hash",
            "label_diagnostics_hash",
            "validation_spec_hash",
            "validation_receipt_hash",
            "trading_calendar_content_hash",
            "scoring_eligibility_hash",
            "scoring_eligibility_policy_hash",
            "scoring_eligibility_source_artifact_hash",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=f"model-training {name}"),
            )
        if self.label_benchmark_hash is not None:
            object.__setattr__(
                self,
                "label_benchmark_hash",
                _sha256(
                    self.label_benchmark_hash,
                    name="model-training label_benchmark_hash",
                ),
            )
        parents = self._normalize_parents(self.parent_artifact_hashes)
        object.__setattr__(self, "parent_artifact_hashes", parents)
        if not isinstance(self.selection_spec, NestedPurgedSelectionSpec):
            raise _error(
                "invalid_nested_contract", "selection_spec has an invalid type"
            )
        if self.selection_spec.content_hash != self.selection_spec_hash:
            raise _error(
                "selection_spec_hash_mismatch",
                "selection spec payload differs from bound hash",
            )
        if self.selection_spec.outer_validation_spec_hash != self.validation_spec_hash:
            raise _error(
                "selection_validation_mismatch",
                "selection spec points to a different outer validation spec",
            )
        bindings = tuple(self.feature_bindings)
        if not bindings or not all(
            isinstance(item, ModelFeatureBinding) for item in bindings
        ):
            raise _error(
                "invalid_feature_bindings", "feature bindings must be non-empty"
            )
        binding_names = tuple(item.feature_name for item in bindings)
        if binding_names != tuple(sorted(set(binding_names))):
            raise _error(
                "invalid_feature_bindings",
                "feature bindings must be sorted by unique feature name",
            )
        object.__setattr__(self, "feature_bindings", bindings)
        frames = self._normalize_feature_frames(self.feature_frames)
        object.__setattr__(self, "feature_frames", frames)
        expected_features = tuple(
            sorted(
                {
                    feature_name
                    for candidate in self.selection_spec.candidates
                    for feature_name in candidate.feature_names
                }
            )
        )
        if binding_names != expected_features or tuple(frames) != expected_features:
            raise _error(
                "feature_set_mismatch",
                "selection, lineage bindings and feature artifacts differ",
            )
        for binding in bindings:
            reference = frames[binding.feature_name]
            if reference.role != "model_feature":
                raise _error(
                    "artifact_role_mismatch",
                    f"feature {binding.feature_name} role differs",
                )
        self._validate_label_contracts()
        self._validate_validation_contracts()
        source_stage = self._normalize_eligibility_source(
            self.scoring_eligibility_source_stage
        )
        object.__setattr__(self, "scoring_eligibility_source_stage", source_stage)
        policy = self.scoring_eligibility_policy
        if not isinstance(policy, ScoringEligibilityPolicyV1):
            raise _error(
                "invalid_eligibility_policy", "eligibility policy type differs"
            )
        if policy.content_hash != self.scoring_eligibility_policy_hash:
            raise _error(
                "eligibility_policy_hash_mismatch",
                "eligibility policy payload differs from bound hash",
            )
        parents = cast(Mapping[ResearchStage, str], self.parent_artifact_hashes)
        if (
            self.scoring_eligibility_source_artifact_hash
            != parents[_ELIGIBILITY_SOURCE_STAGE]
        ):
            raise _error(
                "eligibility_source_artifact_mismatch",
                "eligibility source must be the exact factor-generation parent",
            )
        self._require_reference(
            self.scoring_eligibility,
            role="scoring_eligibility",
            name="scoring eligibility",
        )
        self._validate_panel_shapes()
        self._validate_panel_axes()
        if _boolean(self.research_only, name="research_only") is not True:
            raise _error(
                "invalid_release_flags", "model-training inputs must be research-only"
            )
        if _boolean(self.production_ready, name="production_ready") is not False:
            raise _error(
                "invalid_release_flags",
                "B1 model-training inputs may not be production-ready",
            )

    @staticmethod
    def _normalize_parents(
        value: Mapping[ResearchStage | str, str],
    ) -> Mapping[ResearchStage, str]:
        if not isinstance(value, Mapping):
            raise _error(
                "invalid_parent_artifacts", "parent artifact hashes must be a mapping"
            )
        parents: dict[ResearchStage, str] = {}
        for raw_stage, raw_hash in value.items():
            try:
                stage = ResearchStage(raw_stage)
            except (TypeError, ValueError) as exc:
                raise _error(
                    "invalid_parent_artifacts", f"unknown parent stage:{raw_stage}"
                ) from exc
            if stage in parents:
                raise _error(
                    "invalid_parent_artifacts", f"duplicate parent stage:{stage.value}"
                )
            parents[stage] = _sha256(raw_hash, name=f"parent artifact:{stage.value}")
        if frozenset(parents) != _MODEL_TRAINING_PARENTS:
            raise _error(
                "invalid_parent_artifacts",
                "model training requires factor_generation, label_building and "
                "validation_split parents",
            )
        return MappingProxyType(
            dict(sorted(parents.items(), key=lambda item: _stage_name(item[0])))
        )

    @staticmethod
    def _normalize_feature_frames(
        value: Mapping[str, DataFrameArtifactReferenceV2],
    ) -> Mapping[str, DataFrameArtifactReferenceV2]:
        if not isinstance(value, Mapping) or not all(
            isinstance(name, str)
            and name
            and name == name.strip()
            and isinstance(reference, DataFrameArtifactReferenceV2)
            for name, reference in value.items()
        ):
            raise _error(
                "invalid_feature_artifacts", "feature artifacts have invalid types"
            )
        return MappingProxyType(dict(sorted(value.items())))

    @staticmethod
    def _normalize_eligibility_source(value: ResearchStage | str) -> ResearchStage:
        try:
            stage = ResearchStage(value)
        except (TypeError, ValueError) as exc:
            raise _error(
                "invalid_eligibility_source", "eligibility source stage is invalid"
            ) from exc
        if stage is not _ELIGIBILITY_SOURCE_STAGE:
            raise _error(
                "invalid_eligibility_source",
                "eligibility must originate from the direct factor-generation parent",
            )
        return stage

    @staticmethod
    def _require_reference(
        reference: object, *, role: str, name: str
    ) -> DataFrameArtifactReferenceV2:
        if not isinstance(reference, DataFrameArtifactReferenceV2):
            raise _error("invalid_artifact_reference", f"{name} reference type differs")
        if reference.role != role:
            raise _error("artifact_role_mismatch", f"{name} role differs")
        return reference

    def _validate_label_contracts(self) -> None:
        self._require_reference(
            self.label_values,
            role="label_values",
            name="label values",
        )
        self._require_reference(
            self.label_windows,
            role="label_windows",
            name="label windows",
        )
        self._require_reference(
            self.label_validity,
            role="label_validity",
            name="label validity",
        )
        self._require_reference(
            self.label_diagnostics,
            role="label_diagnostics",
            name="label diagnostics",
        )

    def _validate_validation_contracts(self) -> None:
        if not isinstance(self.validation_spec, ValidationSpec):
            raise _error(
                "invalid_nested_contract", "validation_spec has an invalid type"
            )
        if self.validation_spec.content_hash != self.validation_spec_hash:
            raise _error(
                "validation_spec_hash_mismatch",
                "validation spec payload differs from bound hash",
            )
        if not isinstance(self.validation_receipt, ValidationReceipt):
            raise _error(
                "invalid_nested_contract", "validation_receipt has an invalid type"
            )
        receipt = self.validation_receipt
        if receipt.content_hash != self.validation_receipt_hash:
            raise _error(
                "validation_receipt_hash_mismatch",
                "validation receipt payload differs from bound hash",
            )
        receipt_bindings = {
            "validation_spec_hash": self.validation_spec_hash,
            "label_spec_hash": self.label_spec_hash,
            "labels_hash": self.label_values_hash,
            "windows_hash": self.label_windows_hash,
        }
        for name, expected in receipt_bindings.items():
            if getattr(receipt, name) != expected:
                raise _error(
                    "validation_receipt_binding_mismatch",
                    f"validation receipt {name} differs",
                )
        if not isinstance(self.trading_calendar, TradingCalendar):
            raise _error(
                "invalid_nested_contract", "trading_calendar has an invalid type"
            )
        if self.trading_calendar.content_hash != self.trading_calendar_content_hash:
            raise _error(
                "calendar_hash_mismatch",
                "trading calendar payload differs from bound content hash",
            )
        if receipt.calendar_hash != validation_calendar_hash(self.trading_calendar):
            raise _error(
                "validation_calendar_hash_mismatch",
                "validation receipt calendar differs from the embedded calendar",
            )

    def _validate_panel_shapes(self) -> None:
        panels = [
            *self.feature_frames.values(),
            self.label_values,
            self.label_validity,
            self.scoring_eligibility,
        ]
        shapes = {(item.row_count, item.column_count) for item in panels}
        if len(shapes) != 1:
            raise _error(
                "panel_shape_mismatch",
                "feature, label, validity and eligibility panel shapes differ",
            )

    def _validate_panel_axes(self) -> None:
        """Reject cross-panel axis drift before any parquet payload is decoded."""

        panels = [
            *self.feature_frames.values(),
            self.label_values,
            self.label_validity,
            self.scoring_eligibility,
        ]
        if (
            len({item.index_hash for item in panels}) != 1
            or len({item.columns_hash for item in panels}) != 1
        ):
            raise _error(
                "panel_axis_mismatch",
                "feature, label, validity and eligibility panel axes differ",
            )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def _validate_request_and_lineage(
        self,
        *,
        request: "ScientificStageRequest",
        scientific_lineage_manifest: ResearchScientificLineageManifestV2,
        experiment_spec: "ExperimentSpec",
    ) -> None:
        """Internal resolver hook; it is not an authorization receipt."""

        from alpha_research.experiments import ExperimentSpec
        from alpha_research.research.execution import ScientificStageRequest

        if not isinstance(request, ScientificStageRequest):
            raise _error("request_binding_mismatch", "request type differs")
        if request.stage is not ResearchStage.MODEL_TRAINING:
            raise _error("request_binding_mismatch", "request stage differs")
        if request.research_run_spec_hash != self.research_run_spec_hash:
            raise _error("request_binding_mismatch", "research run differs")
        if request.experiment_spec_hash != self.experiment_spec_hash:
            raise _error("request_binding_mismatch", "experiment differs")
        if not isinstance(experiment_spec, ExperimentSpec):
            raise _error("request_binding_mismatch", "experiment type differs")
        if experiment_spec.content_hash != request.experiment_spec_hash:
            raise _error("request_binding_mismatch", "registered experiment differs")
        if request.contract.content_hash != self.model_training_contract_hash:
            raise _error("request_binding_mismatch", "stage contract differs")
        request_parents = {
            ResearchStage(stage): digest
            for stage, digest in request.parent_artifact_hashes.items()
        }
        if request_parents != dict(self.parent_artifact_hashes):
            raise _error("request_binding_mismatch", "parent artifacts differ")
        if not isinstance(
            scientific_lineage_manifest, ResearchScientificLineageManifestV2
        ):
            raise _error("lineage_binding_mismatch", "V2 lineage is required")
        lineage = scientific_lineage_manifest
        if lineage.content_hash != self.scientific_lineage_manifest_hash:
            raise _error("lineage_binding_mismatch", "lineage identity differs")
        if lineage.research_run_spec_hash != self.research_run_spec_hash:
            raise _error("lineage_binding_mismatch", "lineage run differs")
        if lineage.nested_selection_spec_hash != self.selection_spec_hash:
            raise _error("lineage_binding_mismatch", "selection identity differs")
        if lineage.model_feature_bindings != self.feature_bindings:
            raise _error("lineage_binding_mismatch", "feature bindings differ")
        if request.contract.component_bindings.get("model") != lineage.model_spec_hash:
            raise _error("lineage_binding_mismatch", "model specification differs")
        if experiment_spec.scientific_lineage_manifest_hash != lineage.content_hash:
            raise _error("lineage_binding_mismatch", "experiment lineage differs")
        if experiment_spec.model_spec_hash != lineage.model_spec_hash:
            raise _error("lineage_binding_mismatch", "experiment model differs")
        if experiment_spec.label_spec_hash != self.label_spec_hash:
            raise _error("request_binding_mismatch", "experiment label differs")
        if experiment_spec.validation_spec_hash != self.validation_spec_hash:
            raise _error("request_binding_mismatch", "experiment validation differs")
        experiment_factors = tuple(experiment_spec.factor_spec_hashes)
        lineage_factors = tuple(
            item.factor_spec_hash for item in lineage.factor_logic_bindings
        )
        if set(experiment_factors) != set(lineage_factors):
            raise _error("lineage_binding_mismatch", "experiment factors differ")
        expected_components = {
            **{
                f"factor:{offset}": digest
                for offset, digest in enumerate(experiment_factors)
            },
            "label": experiment_spec.label_spec_hash,
            "validation": experiment_spec.validation_spec_hash,
            "model": lineage.model_spec_hash,
        }
        if dict(request.contract.component_bindings) != dict(
            sorted(expected_components.items())
        ):
            raise _error("request_binding_mismatch", "contract components differ")

    def _validate_eligibility_source_payload(
        self, payload: Mapping[str, object]
    ) -> None:
        raw_binding = payload.get("scoring_eligibility_source")
        binding = _object(raw_binding, name="scoring eligibility source binding")
        _exact_fields(
            binding,
            frozenset(
                {
                    "schema_version",
                    "policy",
                    "policy_hash",
                    "frame_hash",
                    "artifact_reference",
                    "artifact_reference_hash",
                }
            ),
            name="scoring eligibility source binding",
        )
        if binding["schema_version"] != SCORING_ELIGIBILITY_SOURCE_SCHEMA:
            raise _error("eligibility_provenance_mismatch", "source schema differs")
        try:
            source_policy = ScoringEligibilityPolicyV1.from_mapping(
                _object(binding["policy"], name="source eligibility policy")
            )
            source_reference = DataFrameArtifactReferenceV2.from_mapping(
                _object(
                    binding["artifact_reference"],
                    name="source eligibility artifact reference",
                )
            )
        except ModelTrainingInputError as exc:
            raise _error(
                "eligibility_provenance_mismatch",
                "source policy or artifact reference is invalid",
            ) from exc
        expected = {
            "policy_hash": self.scoring_eligibility_policy_hash,
            "frame_hash": self.scoring_eligibility_hash,
            "artifact_reference_hash": self.scoring_eligibility.content_hash,
        }
        for name, value in expected.items():
            if binding[name] != value:
                raise _error(
                    "eligibility_provenance_mismatch", f"source {name} differs"
                )
        if source_policy.content_hash != self.scoring_eligibility_policy_hash:
            raise _error(
                "eligibility_provenance_mismatch", "source policy hash differs"
            )
        if source_policy != self.scoring_eligibility_policy:
            raise _error(
                "eligibility_provenance_mismatch", "source policy payload differs"
            )
        if source_reference != self.scoring_eligibility:
            raise _error(
                "eligibility_provenance_mismatch",
                "source artifact reference differs",
            )

    def to_wire_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        source_stage = self.scoring_eligibility_source_stage
        if not isinstance(source_stage, ResearchStage):  # pragma: no cover
            raise RuntimeError("eligibility source stage was not normalized")
        parents = cast(Mapping[ResearchStage, str], self.parent_artifact_hashes)
        return {
            "schema_version": self.schema_version,
            "research_run_spec_hash": self.research_run_spec_hash,
            "experiment_spec_hash": self.experiment_spec_hash,
            "scientific_lineage_manifest_hash": (self.scientific_lineage_manifest_hash),
            "model_training_contract_hash": self.model_training_contract_hash,
            "parent_artifact_hashes": {
                _stage_name(stage): digest for stage, digest in parents.items()
            },
            "selection_spec_hash": self.selection_spec_hash,
            "selection_spec": self.selection_spec.to_dict(),
            "feature_bindings": [item.to_dict() for item in self.feature_bindings],
            "feature_frames": {
                name: reference.to_dict()
                for name, reference in self.feature_frames.items()
            },
            "label_spec_hash": self.label_spec_hash,
            "label_view_hash": self.label_view_hash,
            "label_benchmark_hash": self.label_benchmark_hash,
            "label_values_hash": self.label_values_hash,
            "label_windows_hash": self.label_windows_hash,
            "label_validity_hash": self.label_validity_hash,
            "label_diagnostics_hash": self.label_diagnostics_hash,
            "label_values": self.label_values.to_dict(),
            "label_windows": self.label_windows.to_dict(),
            "label_validity": self.label_validity.to_dict(),
            "label_diagnostics": self.label_diagnostics.to_dict(),
            "validation_spec_hash": self.validation_spec_hash,
            "validation_spec": self.validation_spec.to_dict(),
            "validation_receipt_hash": self.validation_receipt_hash,
            "validation_receipt": self.validation_receipt.to_dict(),
            "trading_calendar_content_hash": self.trading_calendar_content_hash,
            "trading_calendar": self.trading_calendar.to_dict(),
            "scoring_eligibility_hash": self.scoring_eligibility_hash,
            "scoring_eligibility_policy": self.scoring_eligibility_policy.to_dict(),
            "scoring_eligibility_policy_hash": (self.scoring_eligibility_policy_hash),
            "scoring_eligibility_source_stage": source_stage.value,
            "scoring_eligibility_source_artifact_hash": (
                self.scoring_eligibility_source_artifact_hash
            ),
            "scoring_eligibility": self.scoring_eligibility.to_dict(),
            "research_only": self.research_only,
            "production_ready": self.production_ready,
        }

    @classmethod
    def from_wire_bytes(cls, payload: bytes) -> "ModelTrainingInputManifestV2":
        if not isinstance(payload, bytes):
            raise _error("invalid_wire", "manifest wire payload must be bytes")
        if not payload or len(payload) > _MAXIMUM_MANIFEST_WIRE_BYTES:
            raise _error("invalid_wire", "manifest wire payload size is invalid")
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _error(
                "invalid_wire", "manifest wire payload is invalid JSON"
            ) from exc
        value = _object(decoded, name="model-training input manifest")
        if canonical_json_bytes(value) != payload:
            raise _error("noncanonical_wire", "manifest wire payload is not canonical")
        result = cls.from_mapping(value)
        if result.to_wire_bytes() != payload:  # pragma: no cover - defensive
            raise _error("wire_identity_mismatch", "manifest wire identity differs")
        return result

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "ModelTrainingInputManifestV2":
        expected = frozenset(
            {
                "schema_version",
                "research_run_spec_hash",
                "experiment_spec_hash",
                "scientific_lineage_manifest_hash",
                "model_training_contract_hash",
                "parent_artifact_hashes",
                "selection_spec_hash",
                "selection_spec",
                "feature_bindings",
                "feature_frames",
                "label_spec_hash",
                "label_view_hash",
                "label_benchmark_hash",
                "label_values_hash",
                "label_windows_hash",
                "label_validity_hash",
                "label_diagnostics_hash",
                "label_values",
                "label_windows",
                "label_validity",
                "label_diagnostics",
                "validation_spec_hash",
                "validation_spec",
                "validation_receipt_hash",
                "validation_receipt",
                "trading_calendar_content_hash",
                "trading_calendar",
                "scoring_eligibility_hash",
                "scoring_eligibility_policy",
                "scoring_eligibility_policy_hash",
                "scoring_eligibility_source_stage",
                "scoring_eligibility_source_artifact_hash",
                "scoring_eligibility",
                "research_only",
                "production_ready",
            }
        )
        _exact_fields(value, expected, name="ModelTrainingInputManifestV2")
        raw_parents = _object(value["parent_artifact_hashes"], name="parents")
        raw_frames = _object(value["feature_frames"], name="feature_frames")
        try:
            selection_spec = NestedPurgedSelectionSpec.from_mapping(
                _object(value["selection_spec"], name="selection_spec")
            )
            feature_bindings = tuple(
                ModelFeatureBinding.from_mapping(item)
                for item in _object_array(
                    value["feature_bindings"], name="feature_bindings"
                )
            )
            feature_frames = {
                name: DataFrameArtifactReferenceV2.from_mapping(
                    _object(reference, name=f"feature_frames.{name}")
                )
                for name, reference in raw_frames.items()
            }
            validation_spec = ValidationSpec.from_mapping(
                _object(value["validation_spec"], name="validation_spec")
            )
            validation_receipt = ValidationReceipt.from_mapping(
                _object(value["validation_receipt"], name="validation_receipt")
            )
            calendar = TradingCalendar.from_mapping(
                _object(value["trading_calendar"], name="trading_calendar")
            )
        except ModelTrainingInputError:
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise _error("invalid_nested_contract", str(exc)) from exc
        benchmark = value["label_benchmark_hash"]
        if benchmark is not None and not isinstance(benchmark, str):
            raise _error("invalid_hash", "label_benchmark_hash must be text or null")
        return cls(
            schema_version=_text(value["schema_version"], name="schema_version"),
            research_run_spec_hash=_sha256(
                value["research_run_spec_hash"], name="research_run_spec_hash"
            ),
            experiment_spec_hash=_sha256(
                value["experiment_spec_hash"], name="experiment_spec_hash"
            ),
            scientific_lineage_manifest_hash=_sha256(
                value["scientific_lineage_manifest_hash"],
                name="scientific_lineage_manifest_hash",
            ),
            model_training_contract_hash=_sha256(
                value["model_training_contract_hash"],
                name="model_training_contract_hash",
            ),
            parent_artifact_hashes={
                stage: _sha256(digest, name=f"parent_artifact_hashes.{stage}")
                for stage, digest in raw_parents.items()
            },
            selection_spec_hash=_sha256(
                value["selection_spec_hash"], name="selection_spec_hash"
            ),
            selection_spec=selection_spec,
            feature_bindings=feature_bindings,
            feature_frames=feature_frames,
            label_spec_hash=_sha256(value["label_spec_hash"], name="label_spec_hash"),
            label_view_hash=_sha256(value["label_view_hash"], name="label_view_hash"),
            label_benchmark_hash=benchmark,
            label_values_hash=_sha256(
                value["label_values_hash"], name="label_values_hash"
            ),
            label_windows_hash=_sha256(
                value["label_windows_hash"], name="label_windows_hash"
            ),
            label_validity_hash=_sha256(
                value["label_validity_hash"], name="label_validity_hash"
            ),
            label_diagnostics_hash=_sha256(
                value["label_diagnostics_hash"], name="label_diagnostics_hash"
            ),
            label_values=DataFrameArtifactReferenceV2.from_mapping(
                _object(value["label_values"], name="label_values")
            ),
            label_windows=DataFrameArtifactReferenceV2.from_mapping(
                _object(value["label_windows"], name="label_windows")
            ),
            label_validity=DataFrameArtifactReferenceV2.from_mapping(
                _object(value["label_validity"], name="label_validity")
            ),
            label_diagnostics=DataFrameArtifactReferenceV2.from_mapping(
                _object(value["label_diagnostics"], name="label_diagnostics")
            ),
            validation_spec_hash=_sha256(
                value["validation_spec_hash"], name="validation_spec_hash"
            ),
            validation_spec=validation_spec,
            validation_receipt_hash=_sha256(
                value["validation_receipt_hash"], name="validation_receipt_hash"
            ),
            validation_receipt=validation_receipt,
            trading_calendar_content_hash=_sha256(
                value["trading_calendar_content_hash"],
                name="trading_calendar_content_hash",
            ),
            trading_calendar=calendar,
            scoring_eligibility_hash=_sha256(
                value["scoring_eligibility_hash"],
                name="scoring_eligibility_hash",
            ),
            scoring_eligibility_policy=ScoringEligibilityPolicyV1.from_mapping(
                _object(
                    value["scoring_eligibility_policy"],
                    name="scoring_eligibility_policy",
                )
            ),
            scoring_eligibility_policy_hash=_sha256(
                value["scoring_eligibility_policy_hash"],
                name="scoring_eligibility_policy_hash",
            ),
            scoring_eligibility_source_stage=_text(
                value["scoring_eligibility_source_stage"],
                name="scoring_eligibility_source_stage",
            ),
            scoring_eligibility_source_artifact_hash=_sha256(
                value["scoring_eligibility_source_artifact_hash"],
                name="scoring_eligibility_source_artifact_hash",
            ),
            scoring_eligibility=DataFrameArtifactReferenceV2.from_mapping(
                _object(value["scoring_eligibility"], name="scoring_eligibility")
            ),
            research_only=_boolean(value["research_only"], name="research_only"),
            production_ready=_boolean(
                value["production_ready"], name="production_ready"
            ),
        )


@dataclass(frozen=True, slots=True)
class LoadedModelTrainingInputs:
    """Verified in-memory frames resolved from one immutable input manifest."""

    manifest: ModelTrainingInputManifestV2
    feature_frames: Mapping[str, pd.DataFrame]
    label_result: LabelResult
    scoring_eligibility: pd.DataFrame

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, ModelTrainingInputManifestV2):
            raise _error("invalid_loaded_inputs", "manifest type differs")
        if not isinstance(self.feature_frames, Mapping) or not all(
            isinstance(name, str) and isinstance(frame, pd.DataFrame)
            for name, frame in self.feature_frames.items()
        ):
            raise _error("invalid_loaded_inputs", "feature frame types differ")
        if set(self.feature_frames) != set(self.manifest.feature_frames):
            raise _error(
                "feature_set_mismatch", "loaded feature names differ from manifest"
            )
        frames = MappingProxyType(
            {
                name: pd.DataFrame(self.feature_frames[name]).copy(deep=True)
                for name in sorted(self.feature_frames)
            }
        )
        if not isinstance(self.label_result, LabelResult):
            raise _error("invalid_loaded_inputs", "label result type differs")
        labels = LabelResult(
            label_spec_hash=self.label_result.label_spec_hash,
            label_view_hash=self.label_result.label_view_hash,
            benchmark_hash=self.label_result.benchmark_hash,
            labels_hash=self.label_result.labels_hash,
            windows_hash=self.label_result.windows_hash,
            validity_hash=self.label_result.validity_hash,
            diagnostics_hash=self.label_result.diagnostics_hash,
            labels=self.label_result.labels,
            label_windows=self.label_result.label_windows,
            validity=self.label_result.validity,
            diagnostics=self.label_result.diagnostics,
        )
        if not isinstance(self.scoring_eligibility, pd.DataFrame):
            raise _error("invalid_loaded_inputs", "scoring eligibility type differs")
        eligibility = pd.DataFrame(self.scoring_eligibility).copy(deep=True)
        _require_complete_boolean_panel(eligibility, name="scoring eligibility")
        object.__setattr__(self, "feature_frames", frames)
        object.__setattr__(self, "label_result", labels)
        object.__setattr__(self, "scoring_eligibility", eligibility)
        self.verify_content()

    @property
    def selection_spec(self) -> NestedPurgedSelectionSpec:
        return self.manifest.selection_spec

    @property
    def validation_spec(self) -> ValidationSpec:
        return self.manifest.validation_spec

    @property
    def validation_receipt(self) -> ValidationReceipt:
        return self.manifest.validation_receipt

    @property
    def trading_calendar(self) -> TradingCalendar:
        return self.manifest.trading_calendar

    def verify_content(self) -> None:
        """Recompute all in-memory hashes, axes and the purge receipt."""

        manifest = self.manifest
        feature_bindings = {
            binding.feature_name: binding for binding in manifest.feature_bindings
        }
        for name, frame in self.feature_frames.items():
            _verify_loaded_reference(
                frame, manifest.feature_frames[name], name=f"feature:{name}"
            )
            if hash_frame(frame) != feature_bindings[name].signal_hash:
                raise _error(
                    "feature_lineage_hash_mismatch",
                    f"feature:{name} differs from its scientific lineage signal",
                )
        labels = self.label_result
        labels.verify_content()
        label_identity = {
            "label_spec_hash": labels.label_spec_hash,
            "label_view_hash": labels.label_view_hash,
            "label_benchmark_hash": labels.benchmark_hash,
            "label_values_hash": labels.labels_hash,
            "label_windows_hash": labels.windows_hash,
            "label_validity_hash": labels.validity_hash,
            "label_diagnostics_hash": labels.diagnostics_hash,
        }
        for name, value in label_identity.items():
            if getattr(manifest, name) != value:
                raise _error("loaded_label_binding_mismatch", f"loaded {name} differs")
        _verify_loaded_reference(
            labels.labels, manifest.label_values, name="label values"
        )
        _verify_loaded_reference(
            labels.label_windows, manifest.label_windows, name="label windows"
        )
        _verify_loaded_reference(
            labels.validity, manifest.label_validity, name="label validity"
        )
        _verify_loaded_reference(
            labels.diagnostics,
            manifest.label_diagnostics,
            name="label diagnostics",
        )
        _verify_loaded_reference(
            self.scoring_eligibility,
            manifest.scoring_eligibility,
            name="scoring eligibility",
        )
        if hash_frame(self.scoring_eligibility) != manifest.scoring_eligibility_hash:
            raise _error(
                "eligibility_scientific_hash_mismatch",
                "scoring eligibility differs from its scientific identity",
            )
        _require_complete_boolean_panel(
            self.scoring_eligibility, name="scoring eligibility"
        )
        for name, frame in self.feature_frames.items():
            _require_same_axes(frame, labels.labels, name=f"feature:{name}")
        _require_same_axes(labels.validity, labels.labels, name="label validity")
        _require_same_axes(
            self.scoring_eligibility, labels.labels, name="scoring eligibility"
        )
        _require_panel_domain(
            labels.labels,
            eligibility=self.scoring_eligibility & labels.validity,
            name="label values",
            calendar=manifest.trading_calendar,
        )
        for name, frame in self.feature_frames.items():
            _require_panel_domain(
                frame,
                eligibility=self.scoring_eligibility,
                name=f"feature:{name}",
                calendar=manifest.trading_calendar,
            )
        try:
            ValidationReceiptVerifier().verify(
                manifest.validation_receipt,
                spec=manifest.validation_spec,
                labels=labels,
                calendar=manifest.trading_calendar,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise _error("validation_receipt_recomputation_mismatch", str(exc)) from exc


def _verify_loaded_reference(
    frame: pd.DataFrame, reference: DataFrameArtifactReferenceV2, *, name: str
) -> None:
    if strong_hash_frame(frame) != reference.frame_hash:
        raise _error("loaded_frame_hash_mismatch", f"{name} hash differs")
    if frame.shape != (reference.row_count, reference.column_count):
        raise _error("loaded_frame_shape_mismatch", f"{name} shape differs")
    if _axis_hash(frame.index) != reference.index_hash:
        raise _error("loaded_frame_schema_mismatch", f"{name} index differs")
    if _axis_hash(frame.columns) != reference.columns_hash:
        raise _error("loaded_frame_schema_mismatch", f"{name} columns differ")
    if _dtypes_hash(frame) != reference.dtypes_hash:
        raise _error("loaded_frame_schema_mismatch", f"{name} dtypes differ")


def _require_same_axes(
    value: pd.DataFrame, expected: pd.DataFrame, *, name: str
) -> None:
    if not value.index.equals(expected.index) or not value.columns.equals(
        expected.columns
    ):
        raise _error("loaded_panel_axes_mismatch", f"{name} axes differ")


def _require_complete_boolean_panel(frame: pd.DataFrame, *, name: str) -> None:
    if frame.empty or frame.isna().any(axis=None):
        raise _error("invalid_boolean_panel", f"{name} must be complete and non-empty")
    if not all(is_bool_dtype(dtype) for dtype in frame.dtypes):
        raise _error("invalid_boolean_panel", f"{name} must contain only booleans")


def _require_panel_domain(
    frame: pd.DataFrame,
    *,
    eligibility: pd.DataFrame,
    name: str,
    calendar: TradingCalendar,
) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise _error("invalid_panel_domain", f"{name} index must be DatetimeIndex")
    if frame.index.tz is None or str(frame.index.tz) != calendar.timezone:
        raise _error("invalid_panel_domain", f"{name} timezone differs")
    if (
        frame.index.hasnans
        or not frame.index.is_unique
        or not frame.index.is_monotonic_increasing
    ):
        raise _error(
            "invalid_panel_domain",
            f"{name} timestamps must be complete, sorted and unique",
        )
    try:
        for timestamp in frame.index.unique():
            calendar.assert_timestamp(pd.Timestamp(timestamp), allow_daily_close=True)
    except ValueError as exc:
        raise _error(
            "invalid_panel_domain", f"{name} timestamp is outside the calendar"
        ) from exc
    if not frame.columns.is_unique:
        raise _error("invalid_panel_domain", f"{name} securities must be unique")
    if any(
        is_bool_dtype(dtype) or is_complex_dtype(dtype) or not is_numeric_dtype(dtype)
        for dtype in frame.dtypes
    ):
        raise _error("invalid_panel_domain", f"{name} must be numeric")
    try:
        values = frame.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise _error("invalid_panel_domain", f"{name} cannot be numeric") from exc
    mask = eligibility.to_numpy(dtype=bool)
    if not np.isfinite(values[mask]).all():
        raise _error("invalid_panel_domain", f"{name} has non-finite eligible values")


__all__ = [
    "SCORING_ELIGIBILITY_SOURCE_SCHEMA",
    "DataFrameArtifactReferenceV2",
    "LoadedModelTrainingInputs",
    "ModelTrainingInputError",
    "ModelTrainingInputManifestV2",
    "ScoringEligibilityPolicyV1",
    "strong_hash_frame",
]
