"""Durable phase-one seal for nested model selection.

The manifest deliberately contains inner-selection evidence only.  In
particular, it has no field capable of carrying an outer-fold score, model
result, prediction, or label value.  A later trusted execution boundary may
therefore use the content hash as the immutable input to a one-shot outer
evaluation.  The document still contains cryptographic commitments to source
data and is control-plane evidence, not an Agent-visible or public artifact.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

from alpha_research.core.hashing import (
    canonical_json_bytes,
    hash_json,
    require_sha256,
)
from alpha_research.core.immutable_json import (
    load_immutable_json_document,
    write_immutable_json_document,
)
from alpha_research.models.selection_result import OuterFoldSelectionReceipt
from alpha_research.models.selection_spec import NestedPurgedSelectionSpec
from alpha_research.models.model_execution_governance import (
    ModelExecutionContext,
    ModelFitAttemptOutcome,
    ModelFitAttemptSnapshot,
)
from alpha_research.validation import ValidationReceipt, ValidationSpec


_SCHEMA_VERSION: Final[str] = "nested-selection-phase-one-manifest/v3"
_REFERENCE_SCHEMA_VERSION: Final[str] = (
    "nested-selection-phase-one-manifest-reference/v3"
)
_MANIFEST_SUFFIX: Final[str] = ".nested-selection-phase-one-manifest.json"
_MEDIA_TYPE: Final[str] = (
    "application/vnd.alpha-research.nested-selection-phase-one-manifest+json"
)
_MAXIMUM_MANIFEST_BYTES: Final[int] = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class NestedSelectionPhaseOneManifest:
    """Immutable evidence that every inner selection finished before phase two."""

    selection_spec_hash: str
    selection_spec: NestedPurgedSelectionSpec
    outer_validation_spec_hash: str
    outer_validation_spec: ValidationSpec
    outer_validation_receipt_hash: str
    outer_validation_receipt: ValidationReceipt
    source_feature_hashes: Mapping[str, str]
    source_label_values_hash: str
    source_label_validity_hash: str
    source_scoring_eligibility_hash: str
    selections: tuple[OuterFoldSelectionReceipt, ...]
    required_inner_fold_evaluations: int
    consumed_inner_fold_evaluations: int
    inner_execution_snapshot_hash: str
    inner_execution_snapshot: ModelFitAttemptSnapshot
    planned_outer_fold_evaluations: int
    research_only: bool = True
    production_ready: bool = False
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("unsupported nested selection phase-one manifest schema")
        if not isinstance(self.selection_spec, NestedPurgedSelectionSpec):
            raise TypeError("phase-one selection specification type differs")
        if not isinstance(self.outer_validation_spec, ValidationSpec):
            raise TypeError("phase-one outer validation specification type differs")
        if not isinstance(self.outer_validation_receipt, ValidationReceipt):
            raise TypeError("phase-one outer validation receipt type differs")
        for name in (
            "selection_spec_hash",
            "outer_validation_spec_hash",
            "outer_validation_receipt_hash",
            "source_label_values_hash",
            "source_label_validity_hash",
            "source_scoring_eligibility_hash",
            "inner_execution_snapshot_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"phase-one {name}")
        if self.selection_spec.content_hash != self.selection_spec_hash:
            raise ValueError("phase-one selection specification hash differs")
        if (
            self.outer_validation_spec.content_hash != self.outer_validation_spec_hash
            or self.selection_spec.outer_validation_spec_hash
            != self.outer_validation_spec_hash
        ):
            raise ValueError("phase-one outer validation specification hash differs")
        if (
            self.outer_validation_receipt.content_hash
            != self.outer_validation_receipt_hash
            or self.outer_validation_receipt.validation_spec_hash
            != self.outer_validation_spec_hash
        ):
            raise ValueError("phase-one outer validation receipt hash differs")
        if self.source_label_values_hash != self.outer_validation_receipt.labels_hash:
            raise ValueError("phase-one source label values hash differs")

        features = _hash_mapping(self.source_feature_hashes, "source features")
        expected_feature_names = tuple(
            sorted(
                {
                    name
                    for candidate in self.selection_spec.candidates
                    for name in candidate.feature_names
                }
            )
        )
        if tuple(features) != expected_feature_names:
            raise ValueError("phase-one source feature contract differs")
        object.__setattr__(self, "source_feature_hashes", MappingProxyType(features))

        selections = tuple(self.selections)
        if not selections or not all(
            isinstance(item, OuterFoldSelectionReceipt) for item in selections
        ):
            raise TypeError("phase-one selections have invalid types")
        selection_fold_ids = tuple(item.outer_fold_id for item in selections)
        plan_fold_ids = tuple(
            item.outer_fold_id for item in self.selection_spec.outer_plans
        )
        spec_fold_ids = tuple(item.fold_id for item in self.outer_validation_spec.folds)
        receipt_fold_ids = tuple(
            item.fold_id for item in self.outer_validation_receipt.folds
        )
        if len(set(selection_fold_ids)) != len(selection_fold_ids):
            raise ValueError("phase-one selection outer fold ids must be unique")
        if not (
            selection_fold_ids == plan_fold_ids == spec_fold_ids == receipt_fold_ids
        ):
            raise ValueError("phase-one outer fold contract/order differs")
        for selection, fold_receipt in zip(
            selections,
            self.outer_validation_receipt.folds,
            strict=True,
        ):
            if (
                selection.selection_spec_hash != self.selection_spec_hash
                or selection.selection_spec.content_hash != self.selection_spec_hash
                or selection.inner_validation_receipt.label_spec_hash
                != self.outer_validation_receipt.label_spec_hash
            ):
                raise ValueError("phase-one selection lineage differs")
            training_slice = selection.outer_training_slice
            if (
                training_slice.outer_validation_spec_hash
                != self.outer_validation_spec_hash
                or training_slice.outer_fold_receipt_hash != fold_receipt.content_hash
                or training_slice.outer_fold_receipt.content_hash
                != fold_receipt.content_hash
            ):
                raise ValueError("phase-one outer training membership differs")
        object.__setattr__(self, "selections", selections)

        if not isinstance(self.inner_execution_snapshot, ModelFitAttemptSnapshot):
            raise TypeError("phase-one inner execution snapshot type differs")
        if (
            self.inner_execution_snapshot.content_hash
            != self.inner_execution_snapshot_hash
        ):
            raise ValueError("phase-one inner execution snapshot hash differs")

        for name in (
            "required_inner_fold_evaluations",
            "consumed_inner_fold_evaluations",
            "planned_outer_fold_evaluations",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"phase-one {name} must be positive")
        expected_inner = sum(
            len(self.selection_spec.candidates) * len(plan.inner_validation_spec.folds)
            for plan in self.selection_spec.outer_plans
        )
        observed_inner = sum(item.consumed_inner_evaluations for item in selections)
        expected_outer = len(self.selection_spec.outer_plans)
        if (
            self.required_inner_fold_evaluations != expected_inner
            or self.consumed_inner_fold_evaluations != observed_inner
            or observed_inner != expected_inner
            or self.planned_outer_fold_evaluations != expected_outer
            or expected_outer != self.selection_spec.maximum_outer_evaluations
            or expected_inner + expected_outer
            != self.selection_spec.required_fold_evaluations
            or expected_inner + expected_outer
            > self.selection_spec.maximum_fold_evaluations
        ):
            raise ValueError("phase-one fold evaluation accounting differs")
        _verify_inner_execution_snapshot(self)
        if (
            not isinstance(self.research_only, bool)
            or not self.research_only
            or not isinstance(self.production_ready, bool)
            or self.production_ready
        ):
            raise ValueError("phase-one assurance boundary differs")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "selection_spec_hash": self.selection_spec_hash,
            "selection_spec": self.selection_spec.to_dict(),
            "outer_validation_spec_hash": self.outer_validation_spec_hash,
            "outer_validation_spec": self.outer_validation_spec.to_dict(),
            "outer_validation_receipt_hash": self.outer_validation_receipt_hash,
            "outer_validation_receipt": self.outer_validation_receipt.to_dict(),
            "source_feature_hashes": dict(self.source_feature_hashes),
            "source_label_values_hash": self.source_label_values_hash,
            "source_label_validity_hash": self.source_label_validity_hash,
            "source_scoring_eligibility_hash": (self.source_scoring_eligibility_hash),
            "selections": [item.to_dict() for item in self.selections],
            "required_inner_fold_evaluations": (self.required_inner_fold_evaluations),
            "consumed_inner_fold_evaluations": (self.consumed_inner_fold_evaluations),
            "inner_execution_snapshot_hash": self.inner_execution_snapshot_hash,
            "inner_execution_snapshot": self.inner_execution_snapshot.to_dict(),
            "planned_outer_fold_evaluations": (self.planned_outer_fold_evaluations),
            "research_only": self.research_only,
            "production_ready": self.production_ready,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> NestedSelectionPhaseOneManifest:
        expected = {
            "schema_version",
            "selection_spec_hash",
            "selection_spec",
            "outer_validation_spec_hash",
            "outer_validation_spec",
            "outer_validation_receipt_hash",
            "outer_validation_receipt",
            "source_feature_hashes",
            "source_label_values_hash",
            "source_label_validity_hash",
            "source_scoring_eligibility_hash",
            "selections",
            "required_inner_fold_evaluations",
            "consumed_inner_fold_evaluations",
            "inner_execution_snapshot_hash",
            "inner_execution_snapshot",
            "planned_outer_fold_evaluations",
            "research_only",
            "production_ready",
        }
        if set(value) != expected:
            raise ValueError("NestedSelectionPhaseOneManifest wire fields differ")
        raw_selections = _mapping_list(value["selections"], "selections")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            selection_spec_hash=_text(
                value["selection_spec_hash"], "selection_spec_hash"
            ),
            selection_spec=NestedPurgedSelectionSpec.from_mapping(
                _mapping(value["selection_spec"], "selection_spec")
            ),
            outer_validation_spec_hash=_text(
                value["outer_validation_spec_hash"],
                "outer_validation_spec_hash",
            ),
            outer_validation_spec=ValidationSpec.from_mapping(
                _mapping(value["outer_validation_spec"], "outer_validation_spec")
            ),
            outer_validation_receipt_hash=_text(
                value["outer_validation_receipt_hash"],
                "outer_validation_receipt_hash",
            ),
            outer_validation_receipt=ValidationReceipt.from_mapping(
                _mapping(
                    value["outer_validation_receipt"],
                    "outer_validation_receipt",
                )
            ),
            source_feature_hashes=_string_mapping(
                value["source_feature_hashes"], "source_feature_hashes"
            ),
            source_label_values_hash=_text(
                value["source_label_values_hash"], "source_label_values_hash"
            ),
            source_label_validity_hash=_text(
                value["source_label_validity_hash"],
                "source_label_validity_hash",
            ),
            source_scoring_eligibility_hash=_text(
                value["source_scoring_eligibility_hash"],
                "source_scoring_eligibility_hash",
            ),
            selections=tuple(
                OuterFoldSelectionReceipt.from_mapping(item) for item in raw_selections
            ),
            required_inner_fold_evaluations=_integer(
                value["required_inner_fold_evaluations"],
                "required_inner_fold_evaluations",
            ),
            consumed_inner_fold_evaluations=_integer(
                value["consumed_inner_fold_evaluations"],
                "consumed_inner_fold_evaluations",
            ),
            inner_execution_snapshot_hash=_text(
                value["inner_execution_snapshot_hash"],
                "inner_execution_snapshot_hash",
            ),
            inner_execution_snapshot=ModelFitAttemptSnapshot.from_mapping(
                _mapping(
                    value["inner_execution_snapshot"],
                    "inner_execution_snapshot",
                )
            ),
            planned_outer_fold_evaluations=_integer(
                value["planned_outer_fold_evaluations"],
                "planned_outer_fold_evaluations",
            ),
            research_only=_boolean(value["research_only"], "research_only"),
            production_ready=_boolean(value["production_ready"], "production_ready"),
        )


def nested_inner_selection_execution_context(
    *,
    selection_spec_hash: str,
    outer_validation_receipt_hash: str,
    source_feature_hashes: Mapping[str, str],
    source_label_values_hash: str,
    source_label_validity_hash: str,
    source_scoring_eligibility_hash: str,
) -> ModelExecutionContext:
    """Derive the immutable execution identity for all inner fold fits."""

    hashes = _hash_mapping(source_feature_hashes, "inner execution source features")
    for name, value in (
        ("selection_spec_hash", selection_spec_hash),
        ("outer_validation_receipt_hash", outer_validation_receipt_hash),
        ("source_label_values_hash", source_label_values_hash),
        ("source_label_validity_hash", source_label_validity_hash),
        ("source_scoring_eligibility_hash", source_scoring_eligibility_hash),
    ):
        require_sha256(value, name=f"inner execution {name}")
    scope_hash = cast(
        str,
        hash_json(
            {
                "schema_version": "nested-inner-selection-execution-scope/v1",
                "selection_spec_hash": selection_spec_hash,
                "outer_validation_receipt_hash": outer_validation_receipt_hash,
                "source_feature_hashes": hashes,
                "source_label_values_hash": source_label_values_hash,
                "source_label_validity_hash": source_label_validity_hash,
                "source_scoring_eligibility_hash": (source_scoring_eligibility_hash),
            }
        ),
    )
    return ModelExecutionContext(
        execution_id=(
            f"nested-inner:{selection_spec_hash}:{outer_validation_receipt_hash}"
        ),
        execution_scope_hash=scope_hash,
    )


def _verify_inner_execution_snapshot(
    manifest: NestedSelectionPhaseOneManifest,
) -> None:
    snapshot = manifest.inner_execution_snapshot
    expected_context = nested_inner_selection_execution_context(
        selection_spec_hash=manifest.selection_spec_hash,
        outer_validation_receipt_hash=manifest.outer_validation_receipt_hash,
        source_feature_hashes=manifest.source_feature_hashes,
        source_label_values_hash=manifest.source_label_values_hash,
        source_label_validity_hash=manifest.source_label_validity_hash,
        source_scoring_eligibility_hash=manifest.source_scoring_eligibility_hash,
    )
    if (
        snapshot.execution_context != expected_context
        or snapshot.execution_context_hash != expected_context.content_hash
    ):
        raise ValueError("phase-one inner execution context differs")
    expected_count = manifest.required_inner_fold_evaluations
    if (
        snapshot.maximum_attempts != expected_count
        or snapshot.consumed_attempt_count != expected_count
        or snapshot.terminal_attempt_count != expected_count
        or snapshot.open_attempt_count != 0
        or snapshot.remaining_attempt_count != 0
    ):
        raise ValueError("phase-one inner execution budget evidence differs")
    if any(
        terminal.outcome is not ModelFitAttemptOutcome.SUCCEEDED
        or terminal.failure_type is not None
        for terminal in snapshot.terminals
    ):
        raise ValueError("phase-one inner execution contains failed attempts")
    expected_attempts = tuple(
        (
            score.materialized_model_spec_hash,
            score.inner_validation_receipt_hash,
            fold.fold_id,
        )
        for selection in manifest.selections
        for score in selection.candidate_scores
        for fold in score.fold_scores
    )
    if len(expected_attempts) != expected_count:
        raise ValueError("phase-one inner execution expected attempt count differs")
    for attempt, expected in zip(snapshot.attempts, expected_attempts, strict=True):
        model_spec_hash, validation_receipt_hash, fold_id = expected
        if (
            attempt.fold_attempt_ordinal != 1
            or attempt.model_spec_hash != model_spec_hash
            or attempt.validation_receipt_hash != validation_receipt_hash
            or attempt.fold_id != fold_id
        ):
            raise ValueError("phase-one inner execution attempt lineage differs")


@dataclass(frozen=True, slots=True)
class NestedSelectionPhaseOneManifestReference:
    manifest_id: str
    document_sha256: str
    filename: str
    size_bytes: int
    media_type: str = _MEDIA_TYPE
    schema_version: str = _REFERENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _REFERENCE_SCHEMA_VERSION:
            raise ValueError("unsupported phase-one manifest reference schema")
        require_sha256(self.manifest_id, name="phase-one manifest id")
        require_sha256(self.document_sha256, name="phase-one manifest document sha256")
        if self.filename != f"{self.manifest_id}{_MANIFEST_SUFFIX}":
            raise ValueError("phase-one manifest filename differs")
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes <= 0
            or self.size_bytes > _MAXIMUM_MANIFEST_BYTES
        ):
            raise ValueError("phase-one manifest size is invalid")
        if self.media_type != _MEDIA_TYPE:
            raise ValueError("phase-one manifest media type differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "document_sha256": self.document_sha256,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> NestedSelectionPhaseOneManifestReference:
        expected = {
            "schema_version",
            "manifest_id",
            "document_sha256",
            "filename",
            "size_bytes",
            "media_type",
        }
        if set(value) != expected:
            raise ValueError(
                "NestedSelectionPhaseOneManifestReference wire fields differ"
            )
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            manifest_id=_text(value["manifest_id"], "manifest_id"),
            document_sha256=_text(value["document_sha256"], "document_sha256"),
            filename=_text(value["filename"], "filename"),
            size_bytes=_integer(value["size_bytes"], "size_bytes"),
            media_type=_text(value["media_type"], "media_type"),
        )


def publish_nested_selection_phase_one_manifest(
    manifest: NestedSelectionPhaseOneManifest,
    output_root: str | Path,
) -> NestedSelectionPhaseOneManifestReference:
    """Publish one canonical, content-addressed, immutable manifest."""

    if not isinstance(manifest, NestedSelectionPhaseOneManifest):
        raise TypeError("phase-one manifest type differs")
    manifest_id = manifest.content_hash
    filename = f"{manifest_id}{_MANIFEST_SUFFIX}"
    wire = manifest.to_dict()
    payload = canonical_json_bytes(wire) + b"\n"
    if len(payload) > _MAXIMUM_MANIFEST_BYTES:
        raise ValueError("phase-one manifest exceeds serialized size limit")
    write_immutable_json_document(
        output_root,
        filename=filename,
        value=wire,
    )
    return NestedSelectionPhaseOneManifestReference(
        manifest_id=manifest_id,
        document_sha256=hashlib.sha256(payload).hexdigest(),
        filename=filename,
        size_bytes=len(payload),
    )


def load_nested_selection_phase_one_manifest(
    path: str | Path,
    reference: NestedSelectionPhaseOneManifestReference,
) -> NestedSelectionPhaseOneManifest:
    """Load a manifest only when filename, bytes, wire contract and ID agree."""

    if not isinstance(reference, NestedSelectionPhaseOneManifestReference):
        raise TypeError("phase-one manifest reference type differs")
    target = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    value = load_immutable_json_document(
        target,
        expected_filename=reference.filename,
        expected_document_sha256=reference.document_sha256,
        maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
    )
    payload_size = len(canonical_json_bytes(dict(value))) + 1
    if payload_size != reference.size_bytes:
        raise ValueError("phase-one manifest size differs")
    manifest = NestedSelectionPhaseOneManifest.from_mapping(value)
    if manifest.content_hash != reference.manifest_id:
        raise ValueError("phase-one manifest content hash differs")
    return manifest


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object")
    return MappingProxyType(dict(value))


def _mapping_list(value: object, name: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) and all(isinstance(key, str) for key in item)
        for item in value
    ):
        raise TypeError(f"{name} must be an object array")
    return tuple(MappingProxyType(dict(item)) for item in value)


def _string_mapping(value: object, name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise TypeError(f"{name} must be a text mapping")
    return MappingProxyType(dict(sorted(value.items())))


def _hash_mapping(value: Mapping[str, str], name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise TypeError(f"{name} must be a text mapping")
    hashes = dict(sorted(value.items()))
    if not hashes or any(not key.strip() for key in hashes):
        raise ValueError(f"{name} must not be empty")
    for key, item in hashes.items():
        require_sha256(item, name=f"{name}:{key}")
    return hashes


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


__all__ = [
    "NestedSelectionPhaseOneManifest",
    "NestedSelectionPhaseOneManifestReference",
    "load_nested_selection_phase_one_manifest",
    "publish_nested_selection_phase_one_manifest",
]
