"""Strict, validation-only input contracts for governed research readiness.

This module is deliberately limited to immutable identities.  It performs no
artifact I/O and grants no authority by itself.  A later trusted resolver must
recompute frame and membership hashes, verify the registered parent attempts,
and create :class:`AuthorityBoundResearchReadinessInputs` through its private
constructor.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import Final, Mapping, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_complex_dtype, is_numeric_dtype

from alpha_research.core.data import DataRole
from alpha_research.core.hashing import canonical_json_bytes, hash_json, require_sha256
from alpha_research.experiments.spec import DataPartition
from alpha_research.models.nested_outer_evaluation_store import (
    OuterEvaluationCompletion,
)
from alpha_research.research.spec import ResearchStage
from alpha_research.research.model_training_inputs import strong_hash_frame
from factor_production.v5.artifacts import ArtifactRecord


_PLAN_SCHEMA: Final = "research-readiness-plan/v2"
_FRAME_REFERENCE_SCHEMA: Final = "research-readiness-frame-reference/v1"
_SCORE_BUNDLE_SCHEMA: Final = "research-readiness-score-artifacts/v1"
_PARENT_BINDING_SCHEMA: Final = "research-readiness-parent-attempt-binding/v1"
_INPUT_MANIFEST_SCHEMA: Final = "research-readiness-input-manifest/v2"
_AUTHORITY_RECEIPT_SCHEMA: Final = "research-readiness-authority-receipt/v2"
_CANDIDATE_SCORE_REPRESENTATION: Final = "cross_sectional_rank"
_PARQUET_MEDIA_TYPE: Final = "application/vnd.apache.parquet"
_MAXIMUM_WIRE_BYTES: Final = 16 * 1024 * 1024
_DIRECT_PARENTS: Final = (
    ResearchStage.MODEL_TRAINING,
    ResearchStage.SCORE_CONSTRUCTION,
    ResearchStage.BACKTEST,
)
_FRAME_ROLES: Final = frozenset(
    {
        "candidate_score",
        "baseline_score",
        "perturbation_score",
        "label_values",
        "label_validity",
        "scoring_eligibility",
    }
)


class ResearchReadinessInputError(ValueError):
    """Stable fail-closed error for readiness input contracts."""

    def __init__(self, code: str, detail: str) -> None:
        if not isinstance(code, str) or re.fullmatch(r"[a-z][a-z0-9_]*", code) is None:
            raise ValueError("readiness input error code is unsafe")
        self.code = code
        self.detail = str(detail)
        super().__init__(f"{self.code}:{self.detail}")


def _error(code: str, detail: str) -> ResearchReadinessInputError:
    return ResearchReadinessInputError(code, detail)


def _digest(value: object, *, name: str) -> str:
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


def _positive_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise _error("invalid_integer", f"{name} must be a positive integer")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise _error("invalid_boolean", f"{name} must be boolean")
    return value


def _object(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise _error("invalid_object", f"{name} must be an object")
    return cast(Mapping[str, object], value)


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


def _parse_wire(payload: bytes, *, name: str) -> Mapping[str, object]:
    if (
        not isinstance(payload, bytes)
        or not payload
        or len(payload) > _MAXIMUM_WIRE_BYTES
    ):
        raise _error("invalid_wire", f"{name} wire size differs")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error("invalid_wire", f"{name} wire is invalid JSON") from exc
    value = _object(decoded, name=name)
    if canonical_json_bytes(value) != payload:
        raise _error("noncanonical_wire", f"{name} wire is not canonical")
    return value


def _hash_mapping(value: object, *, name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in value.items()
    ):
        raise _error("invalid_hash_mapping", f"{name} must be a text mapping")
    normalized: dict[str, str] = {}
    for key, item in value.items():
        clean_key = _text(key, name=f"{name} key")
        if clean_key in normalized:
            raise _error("duplicate_definition", f"{name} keys repeat")
        normalized[clean_key] = _digest(item, name=f"{name}:{clean_key}")
    if not normalized:
        raise _error("invalid_hash_mapping", f"{name} must not be empty")
    return MappingProxyType(dict(sorted(normalized.items())))


def _research_boundary(research_only: object, production_ready: object) -> None:
    if _boolean(research_only, name="research_only") is not True:
        raise _error("invalid_release_flags", "readiness inputs must be research-only")
    if _boolean(production_ready, name="production_ready") is not False:
        raise _error(
            "invalid_release_flags", "readiness inputs may not be production-ready"
        )


@dataclass(frozen=True, slots=True)
class ResearchReadinessPlanV1:
    """Preregistered validation-only readiness experiment identity.

    The historical class name is retained to avoid a broad Python API rename,
    but its canonical wire schema is v2.  V2 keeps the registered data
    partition identity separate from the outer-validation split receipt.
    """

    plan_id: str
    version: str
    readiness_spec_hash: str
    validation_spec_hash: str
    nested_selection_spec_hash: str
    scoring_eligibility_policy_hash: str
    validation_partition_hash: str
    outer_validation_receipt_hash: str
    validation_window_start: str
    validation_window_end: str
    candidate_definition_hash: str
    baseline_definition_hash: str
    perturbation_definition_hashes: Mapping[str, str]
    candidate_score_representation: str = _CANDIDATE_SCORE_REPRESENTATION
    partition: DataPartition | str = DataPartition.VALIDATION
    data_role: DataRole | str = DataRole.VALIDATION
    research_only: bool = True
    production_ready: bool = False
    schema_version: str = _PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _PLAN_SCHEMA:
            raise _error("unsupported_schema", "unsupported readiness plan schema")
        object.__setattr__(self, "plan_id", _text(self.plan_id, name="plan_id"))
        object.__setattr__(self, "version", _text(self.version, name="version"))
        for name in (
            "readiness_spec_hash",
            "validation_spec_hash",
            "nested_selection_spec_hash",
            "scoring_eligibility_policy_hash",
            "validation_partition_hash",
            "outer_validation_receipt_hash",
            "candidate_definition_hash",
            "baseline_definition_hash",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        representation = _text(
            self.candidate_score_representation,
            name="candidate_score_representation",
        )
        if representation != _CANDIDATE_SCORE_REPRESENTATION:
            raise _error(
                "invalid_score_representation",
                "candidate scores must be cross-sectional ranks",
            )
        object.__setattr__(self, "candidate_score_representation", representation)
        try:
            partition = DataPartition(self.partition)
            role = DataRole(self.data_role)
        except (TypeError, ValueError) as exc:
            raise _error("invalid_partition", "readiness partition is invalid") from exc
        if partition is not DataPartition.VALIDATION or role is not DataRole.VALIDATION:
            raise _error(
                "invalid_partition",
                "readiness research may consume only the registered validation partition",
            )
        object.__setattr__(self, "partition", partition)
        object.__setattr__(self, "data_role", role)
        try:
            start = pd.Timestamp(self.validation_window_start)
            end = pd.Timestamp(self.validation_window_end)
        except (TypeError, ValueError) as exc:
            raise _error("invalid_partition", "validation window is invalid") from exc
        if start.tzinfo is None or end.tzinfo is None or start >= end:
            raise _error(
                "invalid_partition", "validation window must be timezone-aware and ordered"
            )
        object.__setattr__(self, "validation_window_start", start.isoformat())
        object.__setattr__(self, "validation_window_end", end.isoformat())
        perturbations = _hash_mapping(
            self.perturbation_definition_hashes,
            name="perturbation_definition_hashes",
        )
        definitions = (
            self.candidate_definition_hash,
            self.baseline_definition_hash,
            *perturbations.values(),
        )
        if len(definitions) != len(set(definitions)):
            raise _error(
                "duplicate_definition",
                "candidate, baseline and perturbation definitions must be distinct",
            )
        object.__setattr__(self, "perturbation_definition_hashes", perturbations)
        if self.validation_partition_hash == self.outer_validation_receipt_hash:
            raise _error(
                "identity_domain_collision",
                "validation data partition and outer validation receipt must differ",
            )
        _research_boundary(self.research_only, self.production_ready)

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "version": self.version,
            "readiness_spec_hash": self.readiness_spec_hash,
            "validation_spec_hash": self.validation_spec_hash,
            "nested_selection_spec_hash": self.nested_selection_spec_hash,
            "scoring_eligibility_policy_hash": self.scoring_eligibility_policy_hash,
            "partition": DataPartition(self.partition).value,
            "data_role": DataRole(self.data_role).value,
            "validation_partition_hash": self.validation_partition_hash,
            "outer_validation_receipt_hash": self.outer_validation_receipt_hash,
            "validation_window_start": self.validation_window_start,
            "validation_window_end": self.validation_window_end,
            "candidate_definition_hash": self.candidate_definition_hash,
            "baseline_definition_hash": self.baseline_definition_hash,
            "perturbation_definition_hashes": dict(
                self.perturbation_definition_hashes
            ),
            "candidate_score_representation": self.candidate_score_representation,
            "research_only": self.research_only,
            "production_ready": self.production_ready,
        }

    def to_wire_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self.to_dict()))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ResearchReadinessPlanV1":
        _exact_fields(
            value,
            frozenset(
                {
                    "schema_version",
                    "plan_id",
                    "version",
                    "readiness_spec_hash",
                    "validation_spec_hash",
                    "nested_selection_spec_hash",
                    "scoring_eligibility_policy_hash",
                    "partition",
                    "data_role",
                    "validation_partition_hash",
                    "outer_validation_receipt_hash",
                    "validation_window_start",
                    "validation_window_end",
                    "candidate_definition_hash",
                    "baseline_definition_hash",
                    "perturbation_definition_hashes",
                    "candidate_score_representation",
                    "research_only",
                    "production_ready",
                }
            ),
            name="ResearchReadinessPlanV1",
        )
        return cls(
            schema_version=cast(str, value["schema_version"]),
            plan_id=cast(str, value["plan_id"]),
            version=cast(str, value["version"]),
            readiness_spec_hash=cast(str, value["readiness_spec_hash"]),
            validation_spec_hash=cast(str, value["validation_spec_hash"]),
            nested_selection_spec_hash=cast(
                str, value["nested_selection_spec_hash"]
            ),
            scoring_eligibility_policy_hash=cast(
                str, value["scoring_eligibility_policy_hash"]
            ),
            partition=cast(str, value["partition"]),
            data_role=cast(str, value["data_role"]),
            validation_partition_hash=cast(str, value["validation_partition_hash"]),
            outer_validation_receipt_hash=cast(
                str, value["outer_validation_receipt_hash"]
            ),
            validation_window_start=cast(str, value["validation_window_start"]),
            validation_window_end=cast(str, value["validation_window_end"]),
            candidate_definition_hash=cast(str, value["candidate_definition_hash"]),
            baseline_definition_hash=cast(str, value["baseline_definition_hash"]),
            perturbation_definition_hashes=cast(
                Mapping[str, str], value["perturbation_definition_hashes"]
            ),
            candidate_score_representation=cast(
                str, value["candidate_score_representation"]
            ),
            research_only=cast(bool, value["research_only"]),
            production_ready=cast(bool, value["production_ready"]),
        )

    @classmethod
    def from_wire_bytes(cls, payload: bytes) -> "ResearchReadinessPlanV1":
        return cls.from_mapping(_parse_wire(payload, name="readiness plan"))


@dataclass(frozen=True, slots=True)
class ResearchReadinessFrameReferenceV1:
    """Strict parquet identity with readiness-specific semantic roles.

    ``member_set_hash`` is an explicit resolver-recomputed hash of non-missing
    observations.  The reference cannot derive it without loading the parquet;
    that verification belongs to the future trusted resolver.
    """

    logical_name: str
    location: str
    payload_sha256: str
    size_bytes: int
    semantic_role: str
    frame_hash: str
    row_count: int
    column_count: int
    index_hash: str
    columns_hash: str
    dtypes_hash: str
    member_set_hash: str
    parquet_uncompressed_bytes: int
    media_type: str = _PARQUET_MEDIA_TYPE
    schema_version: str = _FRAME_REFERENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _FRAME_REFERENCE_SCHEMA:
            raise _error("unsupported_schema", "unsupported readiness frame reference")
        logical_name = _text(self.logical_name, name="frame logical_name")
        location = _text(self.location, name="frame location")
        path = PurePosixPath(location)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise _error("unsafe_artifact_reference", "frame location is unsafe")
        role = _text(self.semantic_role, name="frame semantic_role")
        if role not in _FRAME_ROLES:
            raise _error("invalid_artifact_role", f"unsupported readiness role:{role}")
        if self.media_type != _PARQUET_MEDIA_TYPE:
            raise _error("invalid_media_type", "readiness frame must be parquet")
        for name in (
            "payload_sha256",
            "frame_hash",
            "index_hash",
            "columns_hash",
            "dtypes_hash",
            "member_set_hash",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        for name in (
            "size_bytes",
            "row_count",
            "column_count",
            "parquet_uncompressed_bytes",
        ):
            object.__setattr__(
                self, name, _positive_integer(getattr(self, name), name=name)
            )
        try:
            ArtifactRecord(
                logical_name=logical_name,
                location=path.as_posix(),
                sha256=self.payload_sha256,
                size_bytes=self.size_bytes,
                media_type=self.media_type,
                role=role,
            )
        except ValueError as exc:
            raise _error("unsafe_artifact_reference", str(exc)) from exc
        object.__setattr__(self, "logical_name", logical_name)
        object.__setattr__(self, "location", path.as_posix())
        object.__setattr__(self, "semantic_role", role)

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "logical_name": self.logical_name,
            "location": self.location,
            "payload_sha256": self.payload_sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "semantic_role": self.semantic_role,
            "frame_hash": self.frame_hash,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "index_hash": self.index_hash,
            "columns_hash": self.columns_hash,
            "dtypes_hash": self.dtypes_hash,
            "member_set_hash": self.member_set_hash,
            "parquet_uncompressed_bytes": self.parquet_uncompressed_bytes,
        }

    @classmethod
    def bind_verified_frame(
        cls,
        *,
        record: ArtifactRecord,
        frame: pd.DataFrame,
        semantic_role: str,
        parquet_uncompressed_bytes: int,
    ) -> "ResearchReadinessFrameReferenceV1":
        """Bind one already-decoded parquet frame to its immutable artifact."""

        if type(record) is not ArtifactRecord:
            raise _error("invalid_artifact_reference", "artifact record type differs")
        if type(frame) is not pd.DataFrame:
            raise _error("invalid_frame", "readiness frame type differs")
        value = frame.copy(deep=True)
        _validate_readiness_frame_semantics(value, role=semantic_role)
        return cls(
            logical_name=record.logical_name,
            location=record.location,
            payload_sha256=record.sha256,
            size_bytes=record.size_bytes,
            media_type=record.media_type,
            semantic_role=semantic_role,
            frame_hash=strong_hash_frame(value),
            row_count=len(value.index),
            column_count=len(value.columns),
            index_hash=_readiness_axis_hash(value.index),
            columns_hash=_readiness_axis_hash(value.columns),
            dtypes_hash=_readiness_dtypes_hash(value),
            member_set_hash=_readiness_member_set_hash(value),
            parquet_uncompressed_bytes=parquet_uncompressed_bytes,
        )

    def verify_frame(self, frame: pd.DataFrame) -> None:
        """Recompute the strong frame, schema and member-set identities."""

        if type(frame) is not pd.DataFrame:
            raise _error("invalid_frame", "readiness frame type differs")
        _validate_readiness_frame_semantics(frame, role=self.semantic_role)
        if (
            frame.shape != (self.row_count, self.column_count)
            or strong_hash_frame(frame) != self.frame_hash
            or _readiness_axis_hash(frame.index) != self.index_hash
            or _readiness_axis_hash(frame.columns) != self.columns_hash
            or _readiness_dtypes_hash(frame) != self.dtypes_hash
            or _readiness_member_set_hash(frame) != self.member_set_hash
        ):
            raise _error("frame_identity_mismatch", "decoded frame identity differs")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "ResearchReadinessFrameReferenceV1":
        _exact_fields(
            value,
            frozenset(
                {
                    "schema_version",
                    "logical_name",
                    "location",
                    "payload_sha256",
                    "size_bytes",
                    "media_type",
                    "semantic_role",
                    "frame_hash",
                    "row_count",
                    "column_count",
                    "index_hash",
                    "columns_hash",
                    "dtypes_hash",
                    "member_set_hash",
                    "parquet_uncompressed_bytes",
                }
            ),
            name="ResearchReadinessFrameReferenceV1",
        )
        return cls(
            schema_version=cast(str, value["schema_version"]),
            logical_name=cast(str, value["logical_name"]),
            location=cast(str, value["location"]),
            payload_sha256=cast(str, value["payload_sha256"]),
            size_bytes=cast(int, value["size_bytes"]),
            media_type=cast(str, value["media_type"]),
            semantic_role=cast(str, value["semantic_role"]),
            frame_hash=cast(str, value["frame_hash"]),
            row_count=cast(int, value["row_count"]),
            column_count=cast(int, value["column_count"]),
            index_hash=cast(str, value["index_hash"]),
            columns_hash=cast(str, value["columns_hash"]),
            dtypes_hash=cast(str, value["dtypes_hash"]),
            member_set_hash=cast(str, value["member_set_hash"]),
            parquet_uncompressed_bytes=cast(
                int, value["parquet_uncompressed_bytes"]
            ),
        )


@dataclass(frozen=True, slots=True)
class ResearchReadinessScoreArtifactsV1:
    """Exact candidate, baseline and preregistered perturbation score frames."""

    candidate_definition_hash: str
    candidate_scores: ResearchReadinessFrameReferenceV1
    baseline_definition_hash: str
    baseline_scores: ResearchReadinessFrameReferenceV1
    perturbation_definition_hashes: Mapping[str, str]
    perturbation_scores: Mapping[str, ResearchReadinessFrameReferenceV1]
    schema_version: str = _SCORE_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _SCORE_BUNDLE_SCHEMA:
            raise _error("unsupported_schema", "unsupported readiness score bundle")
        candidate_hash = _digest(
            self.candidate_definition_hash, name="candidate_definition_hash"
        )
        baseline_hash = _digest(
            self.baseline_definition_hash, name="baseline_definition_hash"
        )
        object.__setattr__(self, "candidate_definition_hash", candidate_hash)
        object.__setattr__(self, "baseline_definition_hash", baseline_hash)
        if not isinstance(self.candidate_scores, ResearchReadinessFrameReferenceV1):
            raise _error("invalid_score_bundle", "candidate score reference differs")
        if not isinstance(self.baseline_scores, ResearchReadinessFrameReferenceV1):
            raise _error("invalid_score_bundle", "baseline score reference differs")
        if self.candidate_scores.semantic_role != "candidate_score":
            raise _error("artifact_role_mismatch", "candidate score role differs")
        if self.baseline_scores.semantic_role != "baseline_score":
            raise _error("artifact_role_mismatch", "baseline score role differs")
        definitions = _hash_mapping(
            self.perturbation_definition_hashes,
            name="perturbation_definition_hashes",
        )
        raw_scores = self.perturbation_scores
        if not isinstance(raw_scores, Mapping) or not all(
            isinstance(name, str)
            and isinstance(reference, ResearchReadinessFrameReferenceV1)
            for name, reference in raw_scores.items()
        ):
            raise _error("invalid_score_bundle", "perturbation scores are invalid")
        scores = dict(sorted(raw_scores.items()))
        if tuple(scores) != tuple(definitions):
            raise _error(
                "perturbation_set_mismatch",
                "perturbation score names differ from registered definitions",
            )
        if any(item.semantic_role != "perturbation_score" for item in scores.values()):
            raise _error("artifact_role_mismatch", "perturbation score role differs")
        all_definitions = (candidate_hash, baseline_hash, *definitions.values())
        if len(all_definitions) != len(set(all_definitions)):
            raise _error("duplicate_definition", "score definitions repeat")
        references = (self.candidate_scores, self.baseline_scores, *scores.values())
        axes = {
            (item.row_count, item.column_count, item.index_hash, item.columns_hash)
            for item in references
        }
        if len(axes) != 1:
            raise _error("panel_axis_mismatch", "score frame axes differ")
        if len({item.member_set_hash for item in references}) != 1:
            raise _error(
                "member_set_mismatch", "score prediction member sets differ"
            )
        identities = tuple(item.frame_hash for item in references)
        if len(identities) != len(set(identities)):
            raise _error("duplicate_score", "score frame identities repeat")
        logical_names = tuple(item.logical_name for item in references)
        if len(logical_names) != len(set(logical_names)):
            raise _error("duplicate_artifact", "score logical names repeat")
        object.__setattr__(self, "perturbation_definition_hashes", definitions)
        object.__setattr__(self, "perturbation_scores", MappingProxyType(scores))

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "candidate_definition_hash": self.candidate_definition_hash,
            "candidate_scores": self.candidate_scores.to_dict(),
            "baseline_definition_hash": self.baseline_definition_hash,
            "baseline_scores": self.baseline_scores.to_dict(),
            "perturbation_definition_hashes": dict(
                self.perturbation_definition_hashes
            ),
            "perturbation_scores": {
                name: reference.to_dict()
                for name, reference in self.perturbation_scores.items()
            },
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "ResearchReadinessScoreArtifactsV1":
        _exact_fields(
            value,
            frozenset(
                {
                    "schema_version",
                    "candidate_definition_hash",
                    "candidate_scores",
                    "baseline_definition_hash",
                    "baseline_scores",
                    "perturbation_definition_hashes",
                    "perturbation_scores",
                }
            ),
            name="ResearchReadinessScoreArtifactsV1",
        )
        raw_perturbations = _object(
            value["perturbation_scores"], name="perturbation_scores"
        )
        return cls(
            schema_version=cast(str, value["schema_version"]),
            candidate_definition_hash=cast(str, value["candidate_definition_hash"]),
            candidate_scores=ResearchReadinessFrameReferenceV1.from_mapping(
                _object(value["candidate_scores"], name="candidate_scores")
            ),
            baseline_definition_hash=cast(str, value["baseline_definition_hash"]),
            baseline_scores=ResearchReadinessFrameReferenceV1.from_mapping(
                _object(value["baseline_scores"], name="baseline_scores")
            ),
            perturbation_definition_hashes=cast(
                Mapping[str, str], value["perturbation_definition_hashes"]
            ),
            perturbation_scores={
                name: ResearchReadinessFrameReferenceV1.from_mapping(
                    _object(reference, name=f"perturbation_scores:{name}")
                )
                for name, reference in raw_perturbations.items()
            },
        )


@dataclass(frozen=True, slots=True)
class ResearchReadinessParentAttemptBindingV1:
    """Exact registry/runtime/checkpoint identity for one direct parent."""

    stage: ResearchStage | str
    artifact_hash: str
    artifact_descriptor_hash: str
    attempt_id: int
    attempt_number: int
    input_hash: str
    checkpoint_hash: str
    checkpoint_location: str
    stage_contract_hash: str
    schema_version: str = _PARENT_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _PARENT_BINDING_SCHEMA:
            raise _error("unsupported_schema", "unsupported readiness parent binding")
        try:
            stage = ResearchStage(self.stage)
        except (TypeError, ValueError) as exc:
            raise _error("invalid_parent_attempt", "parent stage is invalid") from exc
        if stage not in _DIRECT_PARENTS:
            raise _error("invalid_parent_attempt", "parent stage is not direct")
        object.__setattr__(self, "stage", stage)
        for name in (
            "artifact_hash",
            "artifact_descriptor_hash",
            "input_hash",
            "checkpoint_hash",
            "stage_contract_hash",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        for name in ("attempt_id", "attempt_number"):
            object.__setattr__(
                self, name, _positive_integer(getattr(self, name), name=name)
            )
        location = _text(self.checkpoint_location, name="checkpoint_location")
        path = PurePosixPath(location)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise _error("invalid_parent_attempt", "checkpoint location is unsafe")
        object.__setattr__(self, "checkpoint_location", path.as_posix())
        if self.checkpoint_hash != self.artifact_hash:
            raise _error("invalid_parent_attempt", "checkpoint artifact differs")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "stage": ResearchStage(self.stage).value,
            "artifact_hash": self.artifact_hash,
            "artifact_descriptor_hash": self.artifact_descriptor_hash,
            "attempt_id": self.attempt_id,
            "attempt_number": self.attempt_number,
            "input_hash": self.input_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "checkpoint_location": self.checkpoint_location,
            "stage_contract_hash": self.stage_contract_hash,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "ResearchReadinessParentAttemptBindingV1":
        _exact_fields(
            value,
            frozenset(
                {
                    "schema_version",
                    "stage",
                    "artifact_hash",
                    "artifact_descriptor_hash",
                    "attempt_id",
                    "attempt_number",
                    "input_hash",
                    "checkpoint_hash",
                    "checkpoint_location",
                    "stage_contract_hash",
                }
            ),
            name="ResearchReadinessParentAttemptBindingV1",
        )
        return cls(
            schema_version=cast(str, value["schema_version"]),
            stage=cast(str, value["stage"]),
            artifact_hash=cast(str, value["artifact_hash"]),
            artifact_descriptor_hash=cast(
                str, value["artifact_descriptor_hash"]
            ),
            attempt_id=cast(int, value["attempt_id"]),
            attempt_number=cast(int, value["attempt_number"]),
            input_hash=cast(str, value["input_hash"]),
            checkpoint_hash=cast(str, value["checkpoint_hash"]),
            checkpoint_location=cast(str, value["checkpoint_location"]),
            stage_contract_hash=cast(str, value["stage_contract_hash"]),
        )


def _normalize_parent_hashes(
    value: object, *, name: str
) -> Mapping[ResearchStage, str]:
    if not isinstance(value, Mapping):
        raise _error("invalid_parent_artifacts", f"{name} must be a mapping")
    result: dict[ResearchStage, str] = {}
    for raw_stage, raw_hash in value.items():
        try:
            stage = ResearchStage(raw_stage)
        except (TypeError, ValueError) as exc:
            raise _error("invalid_parent_artifacts", "parent stage is invalid") from exc
        if stage in result:
            raise _error("invalid_parent_artifacts", "parent stage repeats")
        result[stage] = _digest(raw_hash, name=f"{name}:{stage.value}")
    ordered = {stage: result[stage] for stage in _DIRECT_PARENTS if stage in result}
    if tuple(ordered) != _DIRECT_PARENTS or len(result) != len(_DIRECT_PARENTS):
        raise _error(
            "invalid_parent_artifacts",
            "readiness requires model_training, score_construction and backtest parents",
        )
    return MappingProxyType(ordered)


@dataclass(frozen=True, slots=True)
class ResearchReadinessInputManifestV1:
    """Canonical identity of every scientific input to readiness evaluation.

    The historical class name is retained, while the canonical wire schema is
    v2 so v1 manifests that conflated partition and receipt identities cannot
    be reopened implicitly.
    """

    research_run_spec_hash: str
    experiment_spec_hash: str
    scientific_lineage_manifest_hash: str
    robustness_contract_hash: str
    readiness_plan_hash: str
    readiness_plan: ResearchReadinessPlanV1
    parent_artifact_hashes: Mapping[ResearchStage | str, str]
    phase_one_manifest_hash: str
    outer_evaluation_completion_hash: str
    outer_evaluation_completion: OuterEvaluationCompletion
    outer_evaluation_result_hash: str
    outer_execution_snapshot_hash: str
    score_artifacts_hash: str
    score_artifacts: ResearchReadinessScoreArtifactsV1
    label_spec_hash: str
    outer_validation_receipt_hash: str
    label_values_hash: str
    label_values: ResearchReadinessFrameReferenceV1
    label_validity_hash: str
    label_validity: ResearchReadinessFrameReferenceV1
    scoring_eligibility_hash: str
    scoring_eligibility: ResearchReadinessFrameReferenceV1
    research_only: bool = True
    production_ready: bool = False
    schema_version: str = _INPUT_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _INPUT_MANIFEST_SCHEMA:
            raise _error("unsupported_schema", "unsupported readiness input manifest")
        for name in (
            "research_run_spec_hash",
            "experiment_spec_hash",
            "scientific_lineage_manifest_hash",
            "robustness_contract_hash",
            "readiness_plan_hash",
            "phase_one_manifest_hash",
            "outer_evaluation_completion_hash",
            "outer_evaluation_result_hash",
            "outer_execution_snapshot_hash",
            "score_artifacts_hash",
            "label_spec_hash",
            "outer_validation_receipt_hash",
            "label_values_hash",
            "label_validity_hash",
            "scoring_eligibility_hash",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if not isinstance(self.readiness_plan, ResearchReadinessPlanV1):
            raise _error("invalid_manifest", "readiness plan type differs")
        if self.readiness_plan.content_hash != self.readiness_plan_hash:
            raise _error("plan_hash_mismatch", "readiness plan payload differs")
        parents = _normalize_parent_hashes(
            self.parent_artifact_hashes, name="parent_artifact_hashes"
        )
        object.__setattr__(self, "parent_artifact_hashes", parents)
        if not isinstance(self.outer_evaluation_completion, OuterEvaluationCompletion):
            raise _error("invalid_outer_completion", "outer completion type differs")
        completion = self.outer_evaluation_completion
        if completion.content_hash != self.outer_evaluation_completion_hash:
            raise _error("outer_completion_hash_mismatch", "outer completion differs")
        if (
            completion.manifest_hash != self.phase_one_manifest_hash
            or completion.result_hash != self.outer_evaluation_result_hash
            or completion.execution_snapshot_hash != self.outer_execution_snapshot_hash
        ):
            raise _error(
                "outer_completion_lineage_mismatch",
                "phase-one, result or execution snapshot identity differs",
            )
        if (
            completion.evaluation_partition_hash
            != self.readiness_plan.outer_validation_receipt_hash
            or self.outer_validation_receipt_hash
            != self.readiness_plan.outer_validation_receipt_hash
        ):
            raise _error(
                "outer_validation_receipt_mismatch",
                "outer completion and manifest are not bound to the planned receipt",
            )
        if not isinstance(self.score_artifacts, ResearchReadinessScoreArtifactsV1):
            raise _error("invalid_manifest", "score artifacts type differs")
        if self.score_artifacts.content_hash != self.score_artifacts_hash:
            raise _error("score_bundle_hash_mismatch", "score artifact bundle differs")
        if (
            self.score_artifacts.candidate_definition_hash
            != self.readiness_plan.candidate_definition_hash
            or self.score_artifacts.baseline_definition_hash
            != self.readiness_plan.baseline_definition_hash
            or dict(self.score_artifacts.perturbation_definition_hashes)
            != dict(self.readiness_plan.perturbation_definition_hashes)
        ):
            raise _error(
                "score_definition_mismatch", "score definitions differ from the plan"
            )
        references = (
            ("label_values", self.label_values),
            ("label_validity", self.label_validity),
            ("scoring_eligibility", self.scoring_eligibility),
        )
        for role, reference in references:
            if not isinstance(reference, ResearchReadinessFrameReferenceV1):
                raise _error("invalid_manifest", f"{role} reference type differs")
            if reference.semantic_role != role:
                raise _error("artifact_role_mismatch", f"{role} role differs")
        expected_hashes = {
            "label_values_hash": self.label_values.frame_hash,
            "label_validity_hash": self.label_validity.frame_hash,
            "scoring_eligibility_hash": self.scoring_eligibility.frame_hash,
        }
        for name, expected in expected_hashes.items():
            if getattr(self, name) != expected:
                raise _error("frame_hash_mismatch", f"{name} differs from its reference")
        score_references = (
            self.score_artifacts.candidate_scores,
            self.score_artifacts.baseline_scores,
            *self.score_artifacts.perturbation_scores.values(),
        )
        all_references = (*score_references, *(item for _, item in references))
        axes = {
            (item.row_count, item.column_count, item.index_hash, item.columns_hash)
            for item in all_references
        }
        if len(axes) != 1:
            raise _error("panel_axis_mismatch", "readiness input axes differ")
        logical_names = tuple(item.logical_name for item in all_references)
        if len(logical_names) != len(set(logical_names)):
            raise _error("duplicate_artifact", "readiness logical names repeat")
        _research_boundary(self.research_only, self.production_ready)

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "research_run_spec_hash": self.research_run_spec_hash,
            "experiment_spec_hash": self.experiment_spec_hash,
            "scientific_lineage_manifest_hash": self.scientific_lineage_manifest_hash,
            "robustness_contract_hash": self.robustness_contract_hash,
            "readiness_plan_hash": self.readiness_plan_hash,
            "readiness_plan": self.readiness_plan.to_dict(),
            "parent_artifact_hashes": {
                ResearchStage(stage).value: digest
                for stage, digest in self.parent_artifact_hashes.items()
            },
            "phase_one_manifest_hash": self.phase_one_manifest_hash,
            "outer_evaluation_completion_hash": self.outer_evaluation_completion_hash,
            "outer_evaluation_completion": self.outer_evaluation_completion.to_dict(),
            "outer_evaluation_result_hash": self.outer_evaluation_result_hash,
            "outer_execution_snapshot_hash": self.outer_execution_snapshot_hash,
            "score_artifacts_hash": self.score_artifacts_hash,
            "score_artifacts": self.score_artifacts.to_dict(),
            "label_spec_hash": self.label_spec_hash,
            "outer_validation_receipt_hash": self.outer_validation_receipt_hash,
            "label_values_hash": self.label_values_hash,
            "label_values": self.label_values.to_dict(),
            "label_validity_hash": self.label_validity_hash,
            "label_validity": self.label_validity.to_dict(),
            "scoring_eligibility_hash": self.scoring_eligibility_hash,
            "scoring_eligibility": self.scoring_eligibility.to_dict(),
            "research_only": self.research_only,
            "production_ready": self.production_ready,
        }

    def to_wire_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self.to_dict()))

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "ResearchReadinessInputManifestV1":
        _exact_fields(
            value,
            frozenset(
                {
                    "schema_version",
                    "research_run_spec_hash",
                    "experiment_spec_hash",
                    "scientific_lineage_manifest_hash",
                    "robustness_contract_hash",
                    "readiness_plan_hash",
                    "readiness_plan",
                    "parent_artifact_hashes",
                    "phase_one_manifest_hash",
                    "outer_evaluation_completion_hash",
                    "outer_evaluation_completion",
                    "outer_evaluation_result_hash",
                    "outer_execution_snapshot_hash",
                    "score_artifacts_hash",
                    "score_artifacts",
                    "label_spec_hash",
                    "outer_validation_receipt_hash",
                    "label_values_hash",
                    "label_values",
                    "label_validity_hash",
                    "label_validity",
                    "scoring_eligibility_hash",
                    "scoring_eligibility",
                    "research_only",
                    "production_ready",
                }
            ),
            name="ResearchReadinessInputManifestV1",
        )
        parents = _object(value["parent_artifact_hashes"], name="parents")
        return cls(
            schema_version=cast(str, value["schema_version"]),
            research_run_spec_hash=cast(str, value["research_run_spec_hash"]),
            experiment_spec_hash=cast(str, value["experiment_spec_hash"]),
            scientific_lineage_manifest_hash=cast(
                str, value["scientific_lineage_manifest_hash"]
            ),
            robustness_contract_hash=cast(str, value["robustness_contract_hash"]),
            readiness_plan_hash=cast(str, value["readiness_plan_hash"]),
            readiness_plan=ResearchReadinessPlanV1.from_mapping(
                _object(value["readiness_plan"], name="readiness_plan")
            ),
            parent_artifact_hashes=cast(Mapping[str, str], parents),
            phase_one_manifest_hash=cast(str, value["phase_one_manifest_hash"]),
            outer_evaluation_completion_hash=cast(
                str, value["outer_evaluation_completion_hash"]
            ),
            outer_evaluation_completion=OuterEvaluationCompletion.from_mapping(
                _object(value["outer_evaluation_completion"], name="outer_completion")
            ),
            outer_evaluation_result_hash=cast(
                str, value["outer_evaluation_result_hash"]
            ),
            outer_execution_snapshot_hash=cast(
                str, value["outer_execution_snapshot_hash"]
            ),
            score_artifacts_hash=cast(str, value["score_artifacts_hash"]),
            score_artifacts=ResearchReadinessScoreArtifactsV1.from_mapping(
                _object(value["score_artifacts"], name="score_artifacts")
            ),
            label_spec_hash=cast(str, value["label_spec_hash"]),
            outer_validation_receipt_hash=cast(
                str, value["outer_validation_receipt_hash"]
            ),
            label_values_hash=cast(str, value["label_values_hash"]),
            label_values=ResearchReadinessFrameReferenceV1.from_mapping(
                _object(value["label_values"], name="label_values")
            ),
            label_validity_hash=cast(str, value["label_validity_hash"]),
            label_validity=ResearchReadinessFrameReferenceV1.from_mapping(
                _object(value["label_validity"], name="label_validity")
            ),
            scoring_eligibility_hash=cast(str, value["scoring_eligibility_hash"]),
            scoring_eligibility=ResearchReadinessFrameReferenceV1.from_mapping(
                _object(value["scoring_eligibility"], name="scoring_eligibility")
            ),
            research_only=cast(bool, value["research_only"]),
            production_ready=cast(bool, value["production_ready"]),
        )

    @classmethod
    def from_wire_bytes(cls, payload: bytes) -> "ResearchReadinessInputManifestV1":
        return cls.from_mapping(_parse_wire(payload, name="readiness input manifest"))


@dataclass(frozen=True, slots=True, init=False)
class LoadedResearchReadinessInputs:
    """Privately owned validation panels with defensive-copy accessors.

    Pandas objects remain mutable even inside a frozen dataclass.  Keeping the
    authoritative copies in private slots and returning a deep copy on every
    public access prevents a downstream caller from mutating the resolver's
    evidence after verification.
    """

    manifest: ResearchReadinessInputManifestV1
    _candidate_scores: pd.DataFrame
    _baseline_scores: pd.DataFrame
    _perturbation_scores: Mapping[str, pd.DataFrame]
    _labels: pd.DataFrame
    _label_validity: pd.DataFrame
    _scoring_eligibility: pd.DataFrame

    def __init__(
        self,
        *,
        manifest: ResearchReadinessInputManifestV1,
        candidate_scores: pd.DataFrame,
        baseline_scores: pd.DataFrame,
        perturbation_scores: Mapping[str, pd.DataFrame],
        labels: pd.DataFrame,
        label_validity: pd.DataFrame,
        scoring_eligibility: pd.DataFrame,
    ) -> None:
        if type(manifest) is not ResearchReadinessInputManifestV1:
            raise _error("invalid_loaded_inputs", "manifest type differs")
        frames = {
            "candidate_scores": candidate_scores,
            "baseline_scores": baseline_scores,
            "labels": labels,
            "label_validity": label_validity,
            "scoring_eligibility": scoring_eligibility,
        }
        if any(type(frame) is not pd.DataFrame for frame in frames.values()):
            raise _error("invalid_loaded_inputs", "loaded frame type differs")
        if not isinstance(perturbation_scores, Mapping) or not all(
            isinstance(name, str) and type(frame) is pd.DataFrame
            for name, frame in perturbation_scores.items()
        ):
            raise _error("invalid_loaded_inputs", "perturbation score types differ")
        if set(perturbation_scores) != set(
            manifest.score_artifacts.perturbation_scores
        ):
            raise _error(
                "perturbation_set_mismatch",
                "loaded perturbations differ from the manifest",
            )
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(
            self, "_candidate_scores", candidate_scores.copy(deep=True)
        )
        object.__setattr__(
            self, "_baseline_scores", baseline_scores.copy(deep=True)
        )
        object.__setattr__(self, "_labels", labels.copy(deep=True))
        object.__setattr__(
            self, "_label_validity", label_validity.copy(deep=True)
        )
        object.__setattr__(
            self, "_scoring_eligibility", scoring_eligibility.copy(deep=True)
        )
        object.__setattr__(
            self,
            "_perturbation_scores",
            MappingProxyType(
                {
                    name: perturbation_scores[name].copy(deep=True)
                    for name in sorted(perturbation_scores)
                }
            ),
        )
        self.verify_content()

    @property
    def candidate_scores(self) -> pd.DataFrame:
        return self._candidate_scores.copy(deep=True)

    @property
    def baseline_scores(self) -> pd.DataFrame:
        return self._baseline_scores.copy(deep=True)

    @property
    def perturbation_scores(self) -> Mapping[str, pd.DataFrame]:
        return MappingProxyType(
            {
                name: frame.copy(deep=True)
                for name, frame in self._perturbation_scores.items()
            }
        )

    @property
    def labels(self) -> pd.DataFrame:
        return self._labels.copy(deep=True)

    @property
    def label_validity(self) -> pd.DataFrame:
        return self._label_validity.copy(deep=True)

    @property
    def scoring_eligibility(self) -> pd.DataFrame:
        return self._scoring_eligibility.copy(deep=True)

    def verify_content(self) -> None:
        """Recompute every frame identity and validation-only domain invariant."""

        manifest = self.manifest
        references = manifest.score_artifacts
        references.candidate_scores.verify_frame(self._candidate_scores)
        references.baseline_scores.verify_frame(self._baseline_scores)
        for name, frame in self._perturbation_scores.items():
            references.perturbation_scores[name].verify_frame(frame)
        manifest.label_values.verify_frame(self._labels)
        manifest.label_validity.verify_frame(self._label_validity)
        manifest.scoring_eligibility.verify_frame(self._scoring_eligibility)

        panels = (
            self._candidate_scores,
            self._baseline_scores,
            *self._perturbation_scores.values(),
            self._labels,
            self._label_validity,
            self._scoring_eligibility,
        )
        first = panels[0]
        if any(
            not frame.index.equals(first.index)
            or not frame.columns.equals(first.columns)
            for frame in panels[1:]
        ):
            raise _error("panel_axis_mismatch", "loaded readiness axes differ")
        start = pd.Timestamp(manifest.readiness_plan.validation_window_start)
        end = pd.Timestamp(manifest.readiness_plan.validation_window_end)
        if not isinstance(first.index, pd.DatetimeIndex) or (
            (first.index < start).any() or (first.index >= end).any()
        ):
            raise _error(
                "validation_window_mismatch",
                "frame timestamps leave the validation window",
            )
        score_frames = (
            self._candidate_scores,
            self._baseline_scores,
            *self._perturbation_scores.values(),
        )
        if len({_readiness_member_set_hash(frame) for frame in score_frames}) != 1:
            raise _error(
                "member_set_mismatch", "loaded prediction member sets differ"
            )
        ranked = self._candidate_scores.rank(axis=1, method="average")
        active_candidate = self._candidate_scores.notna()
        if not self._candidate_scores.where(active_candidate).equals(
            ranked.where(active_candidate)
        ):
            raise _error(
                "score_representation_mismatch",
                "candidate scores are not canonical cross-sectional ranks",
            )
        active = (
            self._label_validity.to_numpy(dtype=bool)
            & self._scoring_eligibility.to_numpy(dtype=bool)
            & self._candidate_scores.notna().to_numpy(dtype=bool)
            & self._labels.notna().to_numpy(dtype=bool)
        )
        if not np.any(active):
            raise _error(
                "empty_evaluation_membership", "no valid evaluation member remains"
            )

    @property
    def loaded_memory_bytes(self) -> int:
        self.verify_content()
        return sum(
            int(frame.memory_usage(index=True, deep=True).sum())
            for frame in (
                self._candidate_scores,
                self._baseline_scores,
                *self._perturbation_scores.values(),
                self._labels,
                self._label_validity,
                self._scoring_eligibility,
            )
        )

    @property
    def content_hash(self) -> str:
        """Re-verify and hash the exact resolver-owned frame identities."""

        self.verify_content()
        return cast(
            str,
            hash_json(
                {
                    "schema_version": "loaded-research-readiness-inputs/v1",
                    "manifest_hash": self.manifest.content_hash,
                    "candidate_score_hash": strong_hash_frame(
                        self._candidate_scores
                    ),
                    "baseline_score_hash": strong_hash_frame(
                        self._baseline_scores
                    ),
                    "perturbation_score_hashes": {
                        name: strong_hash_frame(frame)
                        for name, frame in self._perturbation_scores.items()
                    },
                    "label_values_hash": strong_hash_frame(self._labels),
                    "label_validity_hash": strong_hash_frame(
                        self._label_validity
                    ),
                    "scoring_eligibility_hash": strong_hash_frame(
                        self._scoring_eligibility
                    ),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ResearchReadinessAuthorityReceiptV1:
    """Deterministic evidence emitted only by a trusted resolver.

    The historical Python class name is retained, while the strict wire schema
    is v2.  ``authoritative_frame_read_bytes`` accounts only for the encoded
    scientific frames opened by the two required authority re-derivation
    passes.  It is deliberately not presented as operating-system total I/O;
    registry and other small metadata reads are outside this governed metric.
    """

    manifest_hash: str
    request_binding_hash: str
    research_run_spec_hash: str
    experiment_spec_hash: str
    scientific_lineage_manifest_hash: str
    current_attempt_id: int
    current_attempt_number: int
    current_attempt_input_hash: str
    parent_payload_hashes: Mapping[ResearchStage | str, str]
    parent_descriptor_hashes: Mapping[ResearchStage | str, str]
    parent_attempt_bindings: Mapping[
        ResearchStage | str, ResearchReadinessParentAttemptBindingV1
    ]
    phase_one_manifest_hash: str
    outer_evaluation_completion_hash: str
    outer_evaluation_result_hash: str
    outer_execution_snapshot_hash: str
    authoritative_frame_read_bytes: int
    total_compressed_bytes: int
    total_parquet_uncompressed_bytes: int
    total_loaded_memory_bytes: int
    research_only: bool = True
    production_ready: bool = False
    schema_version: str = _AUTHORITY_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _AUTHORITY_RECEIPT_SCHEMA:
            raise _error("unsupported_schema", "unsupported readiness authority receipt")
        for name in (
            "manifest_hash",
            "request_binding_hash",
            "research_run_spec_hash",
            "experiment_spec_hash",
            "scientific_lineage_manifest_hash",
            "current_attempt_input_hash",
            "phase_one_manifest_hash",
            "outer_evaluation_completion_hash",
            "outer_evaluation_result_hash",
            "outer_execution_snapshot_hash",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        for name in ("current_attempt_id", "current_attempt_number"):
            object.__setattr__(
                self, name, _positive_integer(getattr(self, name), name=name)
            )
        for name in (
            "authoritative_frame_read_bytes",
            "total_compressed_bytes",
            "total_parquet_uncompressed_bytes",
            "total_loaded_memory_bytes",
        ):
            object.__setattr__(
                self, name, _positive_integer(getattr(self, name), name=name)
            )
        payloads = _normalize_parent_hashes(
            self.parent_payload_hashes, name="parent_payload_hashes"
        )
        descriptors = _normalize_parent_hashes(
            self.parent_descriptor_hashes, name="parent_descriptor_hashes"
        )
        raw_bindings = self.parent_attempt_bindings
        if not isinstance(raw_bindings, Mapping):
            raise _error("invalid_authority_receipt", "parent bindings must be a mapping")
        bindings: dict[ResearchStage, ResearchReadinessParentAttemptBindingV1] = {}
        for raw_stage, binding in raw_bindings.items():
            try:
                stage = ResearchStage(raw_stage)
            except (TypeError, ValueError) as exc:
                raise _error("invalid_authority_receipt", "parent stage is invalid") from exc
            if (
                not isinstance(binding, ResearchReadinessParentAttemptBindingV1)
                or binding.stage is not stage
                or stage in bindings
            ):
                raise _error("invalid_authority_receipt", "parent binding differs")
            bindings[stage] = binding
        ordered = {stage: bindings[stage] for stage in _DIRECT_PARENTS if stage in bindings}
        if tuple(ordered) != _DIRECT_PARENTS or len(bindings) != len(_DIRECT_PARENTS):
            raise _error("invalid_authority_receipt", "parent binding set differs")
        for stage, binding in ordered.items():
            if (
                binding.artifact_hash != payloads[stage]
                or binding.artifact_descriptor_hash != descriptors[stage]
            ):
                raise _error("invalid_authority_receipt", "parent identity differs")
        object.__setattr__(self, "parent_payload_hashes", payloads)
        object.__setattr__(self, "parent_descriptor_hashes", descriptors)
        object.__setattr__(self, "parent_attempt_bindings", MappingProxyType(ordered))
        _research_boundary(self.research_only, self.production_ready)

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_hash": self.manifest_hash,
            "request_binding_hash": self.request_binding_hash,
            "research_run_spec_hash": self.research_run_spec_hash,
            "experiment_spec_hash": self.experiment_spec_hash,
            "scientific_lineage_manifest_hash": self.scientific_lineage_manifest_hash,
            "current_attempt_id": self.current_attempt_id,
            "current_attempt_number": self.current_attempt_number,
            "current_attempt_input_hash": self.current_attempt_input_hash,
            "parent_payload_hashes": {
                ResearchStage(stage).value: digest
                for stage, digest in self.parent_payload_hashes.items()
            },
            "parent_descriptor_hashes": {
                ResearchStage(stage).value: digest
                for stage, digest in self.parent_descriptor_hashes.items()
            },
            "parent_attempt_bindings": {
                ResearchStage(stage).value: binding.to_dict()
                for stage, binding in self.parent_attempt_bindings.items()
            },
            "phase_one_manifest_hash": self.phase_one_manifest_hash,
            "outer_evaluation_completion_hash": self.outer_evaluation_completion_hash,
            "outer_evaluation_result_hash": self.outer_evaluation_result_hash,
            "outer_execution_snapshot_hash": self.outer_execution_snapshot_hash,
            "authoritative_frame_read_bytes": self.authoritative_frame_read_bytes,
            "total_compressed_bytes": self.total_compressed_bytes,
            "total_parquet_uncompressed_bytes": (
                self.total_parquet_uncompressed_bytes
            ),
            "total_loaded_memory_bytes": self.total_loaded_memory_bytes,
            "research_only": self.research_only,
            "production_ready": self.production_ready,
        }

    def to_wire_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self.to_dict()))

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "ResearchReadinessAuthorityReceiptV1":
        _exact_fields(
            value,
            frozenset(
                {
                    "schema_version",
                    "manifest_hash",
                    "request_binding_hash",
                    "research_run_spec_hash",
                    "experiment_spec_hash",
                    "scientific_lineage_manifest_hash",
                    "current_attempt_id",
                    "current_attempt_number",
                    "current_attempt_input_hash",
                    "parent_payload_hashes",
                    "parent_descriptor_hashes",
                    "parent_attempt_bindings",
                    "phase_one_manifest_hash",
                    "outer_evaluation_completion_hash",
                    "outer_evaluation_result_hash",
                    "outer_execution_snapshot_hash",
                    "authoritative_frame_read_bytes",
                    "total_compressed_bytes",
                    "total_parquet_uncompressed_bytes",
                    "total_loaded_memory_bytes",
                    "research_only",
                    "production_ready",
                }
            ),
            name="ResearchReadinessAuthorityReceiptV1",
        )
        raw_bindings = _object(
            value["parent_attempt_bindings"], name="parent_attempt_bindings"
        )
        return cls(
            schema_version=cast(str, value["schema_version"]),
            manifest_hash=cast(str, value["manifest_hash"]),
            request_binding_hash=cast(str, value["request_binding_hash"]),
            research_run_spec_hash=cast(str, value["research_run_spec_hash"]),
            experiment_spec_hash=cast(str, value["experiment_spec_hash"]),
            scientific_lineage_manifest_hash=cast(
                str, value["scientific_lineage_manifest_hash"]
            ),
            current_attempt_id=cast(int, value["current_attempt_id"]),
            current_attempt_number=cast(int, value["current_attempt_number"]),
            current_attempt_input_hash=cast(str, value["current_attempt_input_hash"]),
            parent_payload_hashes=cast(
                Mapping[str, str], value["parent_payload_hashes"]
            ),
            parent_descriptor_hashes=cast(
                Mapping[str, str], value["parent_descriptor_hashes"]
            ),
            parent_attempt_bindings={
                name: ResearchReadinessParentAttemptBindingV1.from_mapping(
                    _object(binding, name=f"parent_attempt_bindings:{name}")
                )
                for name, binding in raw_bindings.items()
            },
            phase_one_manifest_hash=cast(str, value["phase_one_manifest_hash"]),
            outer_evaluation_completion_hash=cast(
                str, value["outer_evaluation_completion_hash"]
            ),
            outer_evaluation_result_hash=cast(
                str, value["outer_evaluation_result_hash"]
            ),
            outer_execution_snapshot_hash=cast(
                str, value["outer_execution_snapshot_hash"]
            ),
            authoritative_frame_read_bytes=cast(
                int, value["authoritative_frame_read_bytes"]
            ),
            total_compressed_bytes=cast(int, value["total_compressed_bytes"]),
            total_parquet_uncompressed_bytes=cast(
                int, value["total_parquet_uncompressed_bytes"]
            ),
            total_loaded_memory_bytes=cast(
                int, value["total_loaded_memory_bytes"]
            ),
            research_only=cast(bool, value["research_only"]),
            production_ready=cast(bool, value["production_ready"]),
        )

    @classmethod
    def from_wire_bytes(cls, payload: bytes) -> "ResearchReadinessAuthorityReceiptV1":
        return cls.from_mapping(_parse_wire(payload, name="readiness authority receipt"))


_RESOLVER_CAPABILITY_CONSTRUCTOR_GUARD: Final = object()


class _ResearchReadinessResolutionSeal:
    """One-shot, process-local proof issued by one resolver instance."""

    __slots__ = ("_binding_hash", "_consumed", "_owner_nonce")

    def __init__(
        self,
        *,
        constructor_guard: object,
        owner_nonce: object,
        binding_hash: str,
    ) -> None:
        if constructor_guard is not _RESOLVER_CAPABILITY_CONSTRUCTOR_GUARD:
            raise TypeError("readiness resolution seals are resolver-private")
        self._owner_nonce = owner_nonce
        self._binding_hash = _digest(binding_hash, name="resolution binding_hash")
        self._consumed = False


class _ResearchReadinessResolverCapability:
    """Per-resolver capability used only for the final verified handoff."""

    __slots__ = ("_nonce",)

    def __init__(self, *, constructor_guard: object) -> None:
        if constructor_guard is not _RESOLVER_CAPABILITY_CONSTRUCTOR_GUARD:
            raise TypeError("readiness resolver capabilities are private")
        self._nonce = object()

    def issue(self, binding_hash: str) -> _ResearchReadinessResolutionSeal:
        return _ResearchReadinessResolutionSeal(
            constructor_guard=_RESOLVER_CAPABILITY_CONSTRUCTOR_GUARD,
            owner_nonce=self._nonce,
            binding_hash=binding_hash,
        )

    def consume(
        self,
        seal: _ResearchReadinessResolutionSeal,
        *,
        expected_binding_hash: str,
    ) -> None:
        if (
            type(seal) is not _ResearchReadinessResolutionSeal
            or seal._owner_nonce is not self._nonce
            or seal._consumed
            or seal._binding_hash != expected_binding_hash
        ):
            raise _error(
                "invalid_resolver_capability", "resolver capability seal differs"
            )
        seal._consumed = True


def _new_research_readiness_resolver_capability(
) -> _ResearchReadinessResolverCapability:
    """Mint a capability for one exact trusted resolver instance."""

    return _ResearchReadinessResolverCapability(
        constructor_guard=_RESOLVER_CAPABILITY_CONSTRUCTOR_GUARD
    )


def _authority_handoff_hash(
    loaded: LoadedResearchReadinessInputs,
    receipt: ResearchReadinessAuthorityReceiptV1,
) -> str:
    if type(loaded) is not LoadedResearchReadinessInputs or type(
        receipt
    ) is not ResearchReadinessAuthorityReceiptV1:
        raise _error("invalid_authority_binding", "handoff types differ")
    return cast(
        str,
        hash_json(
            {
                "schema_version": "research-readiness-authority-handoff/v1",
                "loaded_inputs_hash": loaded.content_hash,
                "authority_receipt_hash": receipt.content_hash,
            }
        ),
    )


@dataclass(frozen=True, slots=True, init=False)
class AuthorityBoundResearchReadinessInputs:
    """Process-local evidence bundle constructible only by trusted resolution."""

    loaded: LoadedResearchReadinessInputs
    authority_receipt: ResearchReadinessAuthorityReceiptV1

    @classmethod
    def _from_verified_resolution(
        cls,
        *,
        loaded: LoadedResearchReadinessInputs,
        authority_receipt: ResearchReadinessAuthorityReceiptV1,
        resolver_capability: _ResearchReadinessResolverCapability,
        resolution_seal: _ResearchReadinessResolutionSeal,
    ) -> "AuthorityBoundResearchReadinessInputs":
        if type(resolver_capability) is not _ResearchReadinessResolverCapability:
            raise _error(
                "invalid_resolver_capability", "resolver capability is required"
            )
        binding_hash = _authority_handoff_hash(loaded, authority_receipt)
        resolver_capability.consume(
            resolution_seal,
            expected_binding_hash=binding_hash,
        )
        value = object.__new__(cls)
        object.__setattr__(value, "loaded", loaded)
        object.__setattr__(value, "authority_receipt", authority_receipt)
        value._verify_binding()
        return value

    def _verify_binding(self) -> None:
        if type(self.loaded) is not LoadedResearchReadinessInputs:
            raise _error("invalid_authority_binding", "loaded inputs type differs")
        if not isinstance(
            self.authority_receipt, ResearchReadinessAuthorityReceiptV1
        ):
            raise _error("invalid_authority_binding", "authority receipt type differs")
        self.loaded.verify_content()
        manifest = self.loaded.manifest
        receipt = self.authority_receipt
        expected = {
            "manifest_hash": manifest.content_hash,
            "research_run_spec_hash": manifest.research_run_spec_hash,
            "experiment_spec_hash": manifest.experiment_spec_hash,
            "scientific_lineage_manifest_hash": (
                manifest.scientific_lineage_manifest_hash
            ),
            "phase_one_manifest_hash": manifest.phase_one_manifest_hash,
            "outer_evaluation_completion_hash": (
                manifest.outer_evaluation_completion_hash
            ),
            "outer_evaluation_result_hash": manifest.outer_evaluation_result_hash,
            "outer_execution_snapshot_hash": manifest.outer_execution_snapshot_hash,
        }
        for name, digest in expected.items():
            if getattr(receipt, name) != digest:
                raise _error("invalid_authority_binding", f"receipt {name} differs")
        if dict(receipt.parent_payload_hashes) != dict(manifest.parent_artifact_hashes):
            raise _error("invalid_authority_binding", "parent artifacts differ")
        references = _readiness_manifest_references(manifest)
        if receipt.total_compressed_bytes != sum(
            reference.size_bytes for reference in references
        ):
            raise _error("invalid_authority_binding", "compressed byte total differs")
        if receipt.total_parquet_uncompressed_bytes != sum(
            reference.parquet_uncompressed_bytes for reference in references
        ):
            raise _error(
                "invalid_authority_binding", "parquet byte total differs"
            )
        if receipt.total_loaded_memory_bytes != self.loaded.loaded_memory_bytes:
            raise _error("invalid_authority_binding", "loaded byte total differs")
        _research_boundary(manifest.research_only, manifest.production_ready)
        _research_boundary(receipt.research_only, receipt.production_ready)

    @property
    def manifest(self) -> ResearchReadinessInputManifestV1:
        return self.loaded.manifest

    @property
    def content_hash(self) -> str:
        self._verify_binding()
        return cast(
            str,
            hash_json(
                {
                    "schema_version": "authority-bound-research-readiness-inputs/v1",
                    "manifest_hash": self.loaded.manifest.content_hash,
                    "loaded_inputs_hash": self.loaded.content_hash,
                    "authority_receipt_hash": self.authority_receipt.content_hash,
                }
            ),
        )


def _validate_readiness_frame_semantics(frame: pd.DataFrame, *, role: str) -> None:
    if frame.empty or not frame.index.is_unique or not frame.columns.is_unique:
        raise _error("invalid_frame", f"{role} frame axes are invalid")
    if isinstance(frame.index, pd.DatetimeIndex):
        if (
            frame.index.tz is None
            or frame.index.hasnans
            or not frame.index.is_monotonic_increasing
        ):
            raise _error("invalid_frame", f"{role} timestamp index is invalid")
    if role in {"label_validity", "scoring_eligibility"}:
        if any(dtype != np.dtype(bool) for dtype in frame.dtypes) or frame.isna().any().any():
            raise _error("invalid_boolean_mask", f"{role} must be exact non-null bool")
        return
    if any(
        not is_numeric_dtype(dtype) or is_complex_dtype(dtype)
        for dtype in frame.dtypes
    ):
        raise _error("invalid_numeric_frame", f"{role} must be real numeric")
    values = frame.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise _error("invalid_numeric_frame", f"{role} contains infinity")


def _readiness_member_set_hash(frame: pd.DataFrame) -> str:
    return cast(str, strong_hash_frame(frame.notna().astype(bool)))


def _readiness_axis_hash(axis: pd.Index) -> str:
    identity: dict[str, object] = {
        "schema_version": "readiness-axis-identity/v1",
        "class": f"{type(axis).__module__}.{type(axis).__qualname__}",
        "names": [str(name) if name is not None else None for name in axis.names],
        "dtypes": (
            [str(level.dtype) for level in axis.levels]
            if isinstance(axis, pd.MultiIndex)
            else [str(axis.dtype)]
        ),
        "length": len(axis),
    }
    digest = hashlib.sha256(canonical_json_bytes(identity))
    try:
        digest.update(
            pd.util.hash_pandas_object(axis, index=False, categorize=False)
            .values.tobytes()
        )
    except (TypeError, ValueError) as exc:
        raise _error("invalid_frame_axis", "frame axis cannot be hashed") from exc
    return digest.hexdigest()


def _readiness_dtypes_hash(frame: pd.DataFrame) -> str:
    return cast(
        str,
        hash_json(
            {
                "schema_version": "readiness-dtypes-identity/v1",
                "dtypes": [
                    {
                        "class": f"{type(dtype).__module__}.{type(dtype).__qualname__}",
                        "text": str(dtype),
                    }
                    for dtype in frame.dtypes
                ],
            }
        ),
    )


def _readiness_manifest_references(
    manifest: ResearchReadinessInputManifestV1,
) -> tuple[ResearchReadinessFrameReferenceV1, ...]:
    return (
        manifest.score_artifacts.candidate_scores,
        manifest.score_artifacts.baseline_scores,
        *manifest.score_artifacts.perturbation_scores.values(),
        manifest.label_values,
        manifest.label_validity,
        manifest.scoring_eligibility,
    )


__all__ = [
    "AuthorityBoundResearchReadinessInputs",
    "LoadedResearchReadinessInputs",
    "ResearchReadinessAuthorityReceiptV1",
    "ResearchReadinessFrameReferenceV1",
    "ResearchReadinessInputError",
    "ResearchReadinessInputManifestV1",
    "ResearchReadinessParentAttemptBindingV1",
    "ResearchReadinessPlanV1",
    "ResearchReadinessScoreArtifactsV1",
]
