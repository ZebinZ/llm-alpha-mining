"""Trusted, registry-backed resolver for model-training inputs.

Only this module turns a self-describing input manifest into an authority-bound
object that a scientific model implementation may consume.  It reloads the
registered ExperimentSpec, verifies all three parent stage artifacts from the
content-addressed store, and enforces parquet resource limits before decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Callable, Mapping, cast

import pandas as pd

from alpha_research.core.hashing import canonical_json_bytes, hash_json, require_sha256
from alpha_research.experiments.controller import stage_input_hash
from alpha_research.experiments.registry import ExperimentRegistry
from alpha_research.experiments.spec import ExperimentSpec
from alpha_research.labels import LabelResult
from alpha_research.models.selection_spec import NestedPurgedSelectionSpec
from alpha_research.orchestration import AttemptStatus, ExperimentRuntime
from alpha_research.research.artifacts import StageArtifactDescriptor, StageContract
from alpha_research.research.execution import (
    ScientificStageRequest,
    ScientificStageResult,
)
from alpha_research.research.lineage import ResearchScientificLineageManifestV2
from alpha_research.research.model_training_inputs import (
    DataFrameArtifactReferenceV2,
    LoadedModelTrainingInputs,
    ModelTrainingInputError,
    ModelTrainingInputManifestV2,
)
from alpha_research.research.model_training_upstream_bundles import (
    FactorModelTrainingArtifactsV1,
    LabelModelTrainingArtifactsV1,
    ValidationModelTrainingArtifactsV1,
)
from alpha_research.research.spec import ResearchRunSpec, ResearchStage
from factor_production.v5.artifacts.manifest import (
    ArtifactError as ContentArtifactError,
)
from factor_production.v5.artifacts.manifest import (
    ArtifactRecord as ContentArtifactRecord,
)
from factor_production.v5.artifacts.manifest import ArtifactStore


_PARENT_STAGES = (
    ResearchStage.FACTOR_GENERATION,
    ResearchStage.LABEL_BUILDING,
    ResearchStage.VALIDATION_SPLIT,
)
_GOVERNED_BODY_FIELDS = frozenset(
    {
        "schema_version",
        "bridge_hash",
        "stage_contract_hash",
        "role",
        "agent_context_hash",
        "agent_task",
        "agent_decision",
        "agent_result",
        "scientific_result",
        "memory_record",
        "memory_record_hash",
    }
)
# Arrow decoding, pandas materialization, validation scratch arrays and the
# immutable hand-off copies coexist transiently.  This conservative multiplier
# is a fail-closed in-process preflight; production isolation must still impose
# an operating-system RSS limit.
_PARQUET_WORKING_SET_MULTIPLIER = 6
_MAXIMUM_AUTHORITY_RECEIPT_WIRE_BYTES = 1024 * 1024


def _error(code: str, detail: str) -> ModelTrainingInputError:
    return ModelTrainingInputError(code, detail)


@dataclass(frozen=True, slots=True)
class GovernedParentAttemptBindingV1:
    """Runtime, registry and checkpoint identity for one direct parent."""

    stage: ResearchStage | str
    artifact_hash: str
    artifact_descriptor_hash: str
    attempt_id: int
    attempt_number: int
    input_hash: str
    checkpoint_hash: str
    checkpoint_location: str
    stage_contract_hash: str
    schema_version: str = "governed-parent-attempt-binding/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "governed-parent-attempt-binding/v1":
            raise _error("unsupported_schema", "unsupported parent attempt binding")
        try:
            stage = ResearchStage(self.stage)
        except (TypeError, ValueError) as exc:
            raise _error("invalid_parent_attempt", "parent stage is invalid") from exc
        if stage not in _PARENT_STAGES:
            raise _error("invalid_parent_attempt", "parent stage is not direct")
        object.__setattr__(self, "stage", stage)
        for name in (
            "artifact_hash",
            "artifact_descriptor_hash",
            "input_hash",
            "checkpoint_hash",
            "stage_contract_hash",
        ):
            try:
                object.__setattr__(
                    self,
                    name,
                    require_sha256(str(getattr(self, name)), name=name),
                )
            except ValueError as exc:
                raise _error("invalid_parent_attempt", str(exc)) from exc
        for name in ("attempt_id", "attempt_number"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise _error("invalid_parent_attempt", f"{name} must be positive")
        location = self.checkpoint_location
        if (
            not isinstance(location, str)
            or not location
            or location != location.strip()
        ):
            raise _error("invalid_parent_attempt", "checkpoint location is invalid")
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
        stage = self.stage
        if not isinstance(stage, ResearchStage):  # pragma: no cover
            raise RuntimeError("parent stage was not normalized")
        return {
            "schema_version": self.schema_version,
            "stage": stage.value,
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
    ) -> "GovernedParentAttemptBindingV1":
        expected = frozenset(
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
        )
        if frozenset(value) != expected:
            raise _error("invalid_parent_attempt", "parent binding fields differ")
        try:
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
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ModelTrainingInputError):
                raise
            raise _error("invalid_parent_attempt", "parent binding is invalid") from exc


@dataclass(frozen=True, slots=True)
class ModelTrainingInputAuthorityReceiptV2:
    """Deterministic evidence emitted only after trusted resolution succeeds."""

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
        ResearchStage | str, GovernedParentAttemptBindingV1
    ]
    total_compressed_bytes: int
    total_parquet_uncompressed_bytes: int
    total_loaded_memory_bytes: int
    schema_version: str = "model-training-input-authority-receipt/v2"

    def __post_init__(self) -> None:
        if self.schema_version != "model-training-input-authority-receipt/v2":
            raise _error("unsupported_schema", "unsupported authority receipt")
        for name in (
            "manifest_hash",
            "request_binding_hash",
            "research_run_spec_hash",
            "experiment_spec_hash",
            "scientific_lineage_manifest_hash",
            "current_attempt_input_hash",
        ):
            try:
                require_sha256(str(getattr(self, name)), name=name)
            except ValueError as exc:
                raise _error("invalid_authority_receipt", str(exc)) from exc
        for name in ("current_attempt_id", "current_attempt_number"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise _error(
                    "invalid_authority_receipt", f"{name} must be positive"
                )
        for name in ("parent_payload_hashes", "parent_descriptor_hashes"):
            raw = getattr(self, name)
            if not isinstance(raw, Mapping):
                raise _error("invalid_authority_receipt", f"{name} must be a mapping")
            normalized: dict[ResearchStage, str] = {}
            for stage, digest in raw.items():
                normalized[ResearchStage(stage)] = require_sha256(
                    digest, name=f"{name}:{ResearchStage(stage).value}"
                )
            if tuple(normalized) != _PARENT_STAGES:
                normalized = dict(
                    sorted(normalized.items(), key=lambda item: item[0].value)
                )
            if tuple(normalized) != _PARENT_STAGES:
                raise _error("invalid_authority_receipt", f"{name} stages differ")
            object.__setattr__(self, name, MappingProxyType(normalized))
        raw_bindings = self.parent_attempt_bindings
        if not isinstance(raw_bindings, Mapping):
            raise _error(
                "invalid_authority_receipt", "parent_attempt_bindings must be a mapping"
            )
        bindings: dict[ResearchStage, GovernedParentAttemptBindingV1] = {}
        for raw_stage, binding in raw_bindings.items():
            stage = ResearchStage(raw_stage)
            if (
                not isinstance(binding, GovernedParentAttemptBindingV1)
                or binding.stage is not stage
            ):
                raise _error(
                    "invalid_authority_receipt", "parent attempt binding differs"
                )
            bindings[stage] = binding
        bindings = dict(sorted(bindings.items(), key=lambda item: item[0].value))
        if tuple(bindings) != _PARENT_STAGES:
            raise _error(
                "invalid_authority_receipt", "parent attempt binding stages differ"
            )
        for stage, binding in bindings.items():
            if (
                binding.artifact_hash != self.parent_payload_hashes[stage]
                or binding.artifact_descriptor_hash
                != self.parent_descriptor_hashes[stage]
            ):
                raise _error(
                    "invalid_authority_receipt", "parent attempt identity differs"
                )
        object.__setattr__(
            self, "parent_attempt_bindings", MappingProxyType(bindings)
        )
        for name in (
            "total_compressed_bytes",
            "total_parquet_uncompressed_bytes",
            "total_loaded_memory_bytes",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise _error("invalid_authority_receipt", f"{name} must be positive")

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
            "scientific_lineage_manifest_hash": (
                self.scientific_lineage_manifest_hash
            ),
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
            "total_compressed_bytes": self.total_compressed_bytes,
            "total_parquet_uncompressed_bytes": (
                self.total_parquet_uncompressed_bytes
            ),
            "total_loaded_memory_bytes": self.total_loaded_memory_bytes,
        }

    def to_wire_bytes(self) -> bytes:
        """Return the exact canonical receipt representation."""

        return cast(bytes, canonical_json_bytes(self.to_dict()))

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "ModelTrainingInputAuthorityReceiptV2":
        expected = frozenset(
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
                "total_compressed_bytes",
                "total_parquet_uncompressed_bytes",
                "total_loaded_memory_bytes",
            }
        )
        if frozenset(value) != expected:
            raise _error("invalid_authority_receipt", "receipt fields differ")
        parent_payloads = value["parent_payload_hashes"]
        parent_descriptors = value["parent_descriptor_hashes"]
        parent_bindings = value["parent_attempt_bindings"]
        if not isinstance(parent_payloads, Mapping) or not isinstance(
            parent_descriptors, Mapping
        ) or not isinstance(parent_bindings, Mapping):
            raise _error(
                "invalid_authority_receipt", "receipt parent mappings differ"
            )
        scalar_hashes = {
            name: value[name]
            for name in (
                "manifest_hash",
                "request_binding_hash",
                "research_run_spec_hash",
                "experiment_spec_hash",
                "scientific_lineage_manifest_hash",
                "current_attempt_input_hash",
            )
        }
        if not all(isinstance(item, str) for item in scalar_hashes.values()):
            raise _error("invalid_authority_receipt", "receipt hashes differ")
        totals = {
            name: value[name]
            for name in (
                "total_compressed_bytes",
                "total_parquet_uncompressed_bytes",
                "total_loaded_memory_bytes",
            )
        }
        if not all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in totals.values()
        ):
            raise _error("invalid_authority_receipt", "receipt totals differ")
        attempt_identity = {
            name: value[name]
            for name in ("current_attempt_id", "current_attempt_number")
        }
        if not all(
            isinstance(item, int)
            and not isinstance(item, bool)
            and item > 0
            for item in attempt_identity.values()
        ):
            raise _error(
                "invalid_authority_receipt", "receipt attempt identity differs"
            )
        return cls(
            schema_version=str(value["schema_version"]),
            manifest_hash=cast(str, scalar_hashes["manifest_hash"]),
            request_binding_hash=cast(str, scalar_hashes["request_binding_hash"]),
            research_run_spec_hash=cast(
                str, scalar_hashes["research_run_spec_hash"]
            ),
            experiment_spec_hash=cast(str, scalar_hashes["experiment_spec_hash"]),
            scientific_lineage_manifest_hash=cast(
                str, scalar_hashes["scientific_lineage_manifest_hash"]
            ),
            current_attempt_id=cast(int, attempt_identity["current_attempt_id"]),
            current_attempt_number=cast(
                int, attempt_identity["current_attempt_number"]
            ),
            current_attempt_input_hash=cast(
                str, scalar_hashes["current_attempt_input_hash"]
            ),
            parent_payload_hashes=cast(Mapping[str, str], parent_payloads),
            parent_descriptor_hashes=cast(Mapping[str, str], parent_descriptors),
            parent_attempt_bindings={
                str(stage): GovernedParentAttemptBindingV1.from_mapping(
                    cast(Mapping[str, object], binding)
                )
                for stage, binding in parent_bindings.items()
                if isinstance(stage, str) and isinstance(binding, Mapping)
            },
            total_compressed_bytes=cast(int, totals["total_compressed_bytes"]),
            total_parquet_uncompressed_bytes=cast(
                int, totals["total_parquet_uncompressed_bytes"]
            ),
            total_loaded_memory_bytes=cast(
                int, totals["total_loaded_memory_bytes"]
            ),
        )

    @classmethod
    def from_wire_bytes(
        cls, payload: bytes
    ) -> "ModelTrainingInputAuthorityReceiptV2":
        if (
            not isinstance(payload, bytes)
            or not payload
            or len(payload) > _MAXIMUM_AUTHORITY_RECEIPT_WIRE_BYTES
        ):
            raise _error("invalid_authority_receipt", "receipt wire size differs")
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _error("invalid_authority_receipt", "receipt wire is invalid") from exc
        if not isinstance(decoded, Mapping) or not all(
            isinstance(key, str) for key in decoded
        ):
            raise _error("invalid_authority_receipt", "receipt wire is not an object")
        mapping = cast(Mapping[str, object], decoded)
        if canonical_json_bytes(mapping) != payload:
            raise _error("noncanonical_wire", "authority receipt wire is not canonical")
        return cls.from_mapping(mapping)


@dataclass(frozen=True, slots=True, init=False)
class AuthorityBoundModelTrainingInputs:
    """Evidence bundle constructible only through trusted resolution.

    This is a process-local misuse guard, not a Python security boundary.  A
    model-stage implementation must invoke :class:`ModelTrainingInputResolver`
    itself rather than accept this object from an untrusted caller.
    """

    loaded: LoadedModelTrainingInputs
    authority_receipt: ModelTrainingInputAuthorityReceiptV2

    @classmethod
    def _from_verified_resolution(
        cls,
        *,
        loaded: LoadedModelTrainingInputs,
        authority_receipt: ModelTrainingInputAuthorityReceiptV2,
    ) -> "AuthorityBoundModelTrainingInputs":
        value = object.__new__(cls)
        object.__setattr__(value, "loaded", loaded)
        object.__setattr__(value, "authority_receipt", authority_receipt)
        value._verify_binding()
        return value

    def _verify_binding(self) -> None:
        if not isinstance(self.loaded, LoadedModelTrainingInputs):
            raise _error("invalid_authority_binding", "loaded inputs type differs")
        if not isinstance(
            self.authority_receipt, ModelTrainingInputAuthorityReceiptV2
        ):
            raise _error("invalid_authority_binding", "authority receipt type differs")
        if self.authority_receipt.manifest_hash != self.loaded.manifest.content_hash:
            raise _error("invalid_authority_binding", "manifest identity differs")
        manifest = self.loaded.manifest
        receipt = self.authority_receipt
        expected = {
            "research_run_spec_hash": manifest.research_run_spec_hash,
            "experiment_spec_hash": manifest.experiment_spec_hash,
            "scientific_lineage_manifest_hash": (
                manifest.scientific_lineage_manifest_hash
            ),
        }
        for name, digest in expected.items():
            if getattr(receipt, name) != digest:
                raise _error(
                    "invalid_authority_binding", f"receipt {name} differs"
                )
        if dict(receipt.parent_payload_hashes) != dict(
            manifest.parent_artifact_hashes
        ):
            raise _error(
                "invalid_authority_binding", "receipt parent artifacts differ"
            )
        if tuple(receipt.parent_attempt_bindings) != _PARENT_STAGES:
            raise _error(
                "invalid_authority_binding", "receipt parent attempts differ"
            )
        self.loaded.verify_content()


@dataclass(frozen=True, slots=True)
class _VerifiedParent:
    payload_hash: str
    descriptor_hash: str
    attempt_binding: GovernedParentAttemptBindingV1 | None
    result: ScientificStageResult


@dataclass(frozen=True, slots=True)
class _VerifiedRuntimeAuthority:
    """Direct parents plus the exact running model-attempt identity."""

    parents: Mapping[ResearchStage, _VerifiedParent]
    current_attempt_id: int
    current_attempt_number: int
    current_attempt_input_hash: str


class ModelTrainingInputResolver:
    """Reload, verify, budget-check and authority-bind one model input bundle."""

    def __init__(
        self,
        *,
        registry: ExperimentRegistry,
        artifact_store: ArtifactStore,
        runtime: ExperimentRuntime,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if type(registry) is not ExperimentRegistry:
            raise TypeError("resolver requires an exact ExperimentRegistry")
        if type(artifact_store) is not ArtifactStore:
            raise TypeError("resolver requires an exact ArtifactStore")
        if type(runtime) is not ExperimentRuntime:
            raise TypeError("resolver requires an exact ExperimentRuntime")
        if clock is not None and not callable(clock):
            raise TypeError("resolver clock must be callable or None")
        self.registry = registry
        self.artifact_store = artifact_store
        self.runtime = runtime
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def assemble_manifest(
        self,
        *,
        request: ScientificStageRequest,
        run_spec: ResearchRunSpec,
        scientific_lineage_manifest: ResearchScientificLineageManifestV2,
        selection_spec: NestedPurgedSelectionSpec,
    ) -> ModelTrainingInputManifestV2:
        """Construct the only accepted manifest from authoritative parents."""

        if not isinstance(request, ScientificStageRequest):
            raise _error("request_binding_mismatch", "request type differs")
        if not isinstance(run_spec, ResearchRunSpec):
            raise _error("request_binding_mismatch", "run specification type differs")
        if not isinstance(selection_spec, NestedPurgedSelectionSpec):
            raise _error("invalid_nested_contract", "selection specification differs")
        if run_spec.content_hash != request.research_run_spec_hash:
            raise _error("request_binding_mismatch", "run specification differs")
        experiment_spec = self.registry.load_experiment_spec(
            request.experiment_spec_hash
        )
        self._validate_context_authority(
            request=request,
            run_spec=run_spec,
            experiment_spec=experiment_spec,
            scientific_lineage_manifest=scientific_lineage_manifest,
            selection_spec=selection_spec,
        )
        runtime_authority = self._load_runtime_authoritative_parents(
            experiment_spec=experiment_spec,
            run_spec=run_spec,
            request=request,
        )
        parents = runtime_authority.parents
        try:
            factor = FactorModelTrainingArtifactsV1.from_result_payload(
                parents[ResearchStage.FACTOR_GENERATION].result.result_payload
            )
            label = LabelModelTrainingArtifactsV1.from_result_payload(
                parents[ResearchStage.LABEL_BUILDING].result.result_payload
            )
            validation = ValidationModelTrainingArtifactsV1.from_result_payload(
                parents[ResearchStage.VALIDATION_SPLIT].result.result_payload
            )
        except ModelTrainingInputError as exc:
            raise _error(
                "parent_binding_mismatch", "typed upstream bundle is invalid"
            ) from exc
        manifest = ModelTrainingInputManifestV2(
            research_run_spec_hash=run_spec.content_hash,
            experiment_spec_hash=experiment_spec.content_hash,
            scientific_lineage_manifest_hash=scientific_lineage_manifest.content_hash,
            model_training_contract_hash=request.contract.content_hash,
            parent_artifact_hashes={
                stage: parents[stage].payload_hash for stage in _PARENT_STAGES
            },
            selection_spec_hash=selection_spec.content_hash,
            selection_spec=selection_spec,
            feature_bindings=scientific_lineage_manifest.model_feature_bindings,
            feature_frames=factor.feature_artifacts,
            label_spec_hash=label.label_spec_hash,
            label_view_hash=label.label_view_hash,
            label_benchmark_hash=label.label_benchmark_hash,
            label_values_hash=label.label_values_hash,
            label_windows_hash=label.label_windows_hash,
            label_validity_hash=label.label_validity_hash,
            label_diagnostics_hash=label.label_diagnostics_hash,
            label_values=label.label_values,
            label_windows=label.label_windows,
            label_validity=label.label_validity,
            label_diagnostics=label.label_diagnostics,
            validation_spec_hash=validation.validation_spec.content_hash,
            validation_spec=validation.validation_spec,
            validation_receipt_hash=validation.validation_receipt.content_hash,
            validation_receipt=validation.validation_receipt,
            trading_calendar_content_hash=validation.trading_calendar.content_hash,
            trading_calendar=validation.trading_calendar,
            scoring_eligibility_hash=factor.scoring_eligibility_hash,
            scoring_eligibility_policy=factor.scoring_eligibility_policy,
            scoring_eligibility_policy_hash=(
                factor.scoring_eligibility_policy.content_hash
            ),
            scoring_eligibility_source_stage=ResearchStage.FACTOR_GENERATION,
            scoring_eligibility_source_artifact_hash=parents[
                ResearchStage.FACTOR_GENERATION
            ].payload_hash,
            scoring_eligibility=factor.scoring_eligibility,
        )
        self._validate_registered_authority(
            manifest=manifest,
            request=request,
            run_spec=run_spec,
            experiment_spec=experiment_spec,
            scientific_lineage_manifest=scientific_lineage_manifest,
        )
        self._validate_eligibility_policy_authority(
            manifest=manifest,
            run_spec=run_spec,
        )
        self._validate_parent_results(manifest=manifest, parents=parents)
        return manifest

    @staticmethod
    def _validate_context_authority(
        *,
        request: ScientificStageRequest,
        run_spec: ResearchRunSpec,
        experiment_spec: ExperimentSpec,
        scientific_lineage_manifest: ResearchScientificLineageManifestV2,
        selection_spec: NestedPurgedSelectionSpec,
    ) -> None:
        scientific_lineage_manifest.validate_for(run_spec)
        if experiment_spec.content_hash != request.experiment_spec_hash:
            raise _error("request_binding_mismatch", "registered experiment differs")
        if tuple(experiment_spec.factor_spec_hashes) != tuple(
            run_spec.factor_spec_hashes
        ):
            raise _error("request_binding_mismatch", "run/experiment factors differ")
        for name in ("label_spec_hash", "validation_spec_hash", "model_spec_hash"):
            if getattr(experiment_spec, name) != getattr(run_spec, name):
                raise _error(
                    "request_binding_mismatch", f"run/experiment {name} differs"
                )
        if request.contract.content_hash != StageContract.for_run(
            run_spec, ResearchStage.MODEL_TRAINING
        ).content_hash:
            raise _error("request_binding_mismatch", "model stage contract differs")
        lineage = scientific_lineage_manifest
        if (
            experiment_spec.scientific_lineage_manifest_hash != lineage.content_hash
            or lineage.research_run_spec_hash != run_spec.content_hash
            or lineage.nested_selection_spec_hash != selection_spec.content_hash
            or experiment_spec.model_spec_hash != lineage.model_spec_hash
        ):
            raise _error("lineage_binding_mismatch", "lineage authority differs")

    def resolve(
        self,
        *,
        manifest: ModelTrainingInputManifestV2,
        request: ScientificStageRequest,
        run_spec: ResearchRunSpec,
        scientific_lineage_manifest: ResearchScientificLineageManifestV2,
    ) -> AuthorityBoundModelTrainingInputs:
        if not isinstance(manifest, ModelTrainingInputManifestV2):
            raise _error("invalid_manifest", "manifest type differs")
        if not isinstance(request, ScientificStageRequest):
            raise _error("request_binding_mismatch", "request type differs")
        if not isinstance(run_spec, ResearchRunSpec):
            raise _error("request_binding_mismatch", "run specification type differs")
        if run_spec.content_hash != request.research_run_spec_hash:
            raise _error("request_binding_mismatch", "run specification differs")
        registered_spec = self.registry.load_experiment_spec(
            request.experiment_spec_hash
        )
        if self.runtime.spec.content_hash != registered_spec.content_hash:
            raise _error(
                "parent_runtime_authority_mismatch", "runtime experiment differs"
            )
        self._validate_registered_authority(
            manifest=manifest,
            request=request,
            run_spec=run_spec,
            experiment_spec=registered_spec,
            scientific_lineage_manifest=scientific_lineage_manifest,
        )
        self._validate_eligibility_policy_authority(
            manifest=manifest,
            run_spec=run_spec,
        )
        runtime_authority = self._load_runtime_authoritative_parents(
            experiment_spec=registered_spec,
            run_spec=run_spec,
            request=request,
        )
        parents = runtime_authority.parents
        self._validate_parent_results(manifest=manifest, parents=parents)
        loaded, resource_totals = self._load_frames(
            manifest=manifest,
            experiment_spec=registered_spec,
        )
        receipt = ModelTrainingInputAuthorityReceiptV2(
            manifest_hash=manifest.content_hash,
            request_binding_hash=_request_binding_hash(request),
            research_run_spec_hash=run_spec.content_hash,
            experiment_spec_hash=registered_spec.content_hash,
            scientific_lineage_manifest_hash=(
                scientific_lineage_manifest.content_hash
            ),
            current_attempt_id=runtime_authority.current_attempt_id,
            current_attempt_number=runtime_authority.current_attempt_number,
            current_attempt_input_hash=(
                runtime_authority.current_attempt_input_hash
            ),
            parent_payload_hashes={
                stage: parents[stage].payload_hash for stage in _PARENT_STAGES
            },
            parent_descriptor_hashes={
                stage: parents[stage].descriptor_hash for stage in _PARENT_STAGES
            },
            parent_attempt_bindings={
                stage: self._required_attempt_binding(parents[stage])
                for stage in _PARENT_STAGES
            },
            total_compressed_bytes=resource_totals[0],
            total_parquet_uncompressed_bytes=resource_totals[1],
            total_loaded_memory_bytes=resource_totals[2],
        )
        return AuthorityBoundModelTrainingInputs._from_verified_resolution(
            loaded=loaded,
            authority_receipt=receipt,
        )

    @staticmethod
    def _required_attempt_binding(
        parent: _VerifiedParent,
    ) -> GovernedParentAttemptBindingV1:
        binding = parent.attempt_binding
        if binding is None:  # pragma: no cover - guarded by runtime loader.
            raise RuntimeError("direct parent lacks an attempt binding")
        return binding

    def _load_runtime_authoritative_parents(
        self,
        *,
        experiment_spec: ExperimentSpec,
        run_spec: ResearchRunSpec,
        request: ScientificStageRequest,
    ) -> _VerifiedRuntimeAuthority:
        """Rebuild controller order and cross-check runtime/registry/checkpoints."""

        if tuple(experiment_spec.stages) != tuple(
            stage.value for stage in run_spec.enabled_stages
        ):
            raise _error(
                "parent_runtime_authority_mismatch", "runtime stage topology differs"
            )
        model_stage = ResearchStage.MODEL_TRAINING.value
        try:
            model_index = experiment_spec.stages.index(model_stage)
        except ValueError as exc:  # pragma: no cover - run contract already checks.
            raise _error(
                "parent_runtime_authority_mismatch", "model stage is not enabled"
            ) from exc
        results: dict[str, str] = {}
        direct: dict[ResearchStage, _VerifiedParent] = {}
        for stage_name in experiment_spec.stages[:model_index]:
            stage = ResearchStage(stage_name)
            attempt = self.runtime.successful_attempt(stage_name)
            if (
                attempt is None
                or attempt.status is not AttemptStatus.SUCCEEDED
                or attempt.experiment_spec_hash != experiment_spec.content_hash
                or attempt.stage != stage_name
                or attempt.result_hash is None
            ):
                raise _error(
                    "parent_runtime_authority_mismatch",
                    f"{stage_name} has no authoritative success",
                )
            expected_input_hash = stage_input_hash(
                experiment_spec_hash=experiment_spec.content_hash,
                stage=stage_name,
                prior_result_hashes=results,
            )
            if attempt.input_hash != expected_input_hash:
                raise _error(
                    "parent_runtime_authority_mismatch",
                    f"{stage_name} input identity differs",
                )
            verified = self._load_parent(
                experiment_spec=experiment_spec,
                run_spec=run_spec,
                stage=stage,
                artifact_hash=attempt.result_hash,
                expected_attempt_id=attempt.attempt_id,
                expected_attempt_number=attempt.attempt_number,
                expected_input_hash=expected_input_hash,
                expected_parent_hashes={
                    ResearchStage(parent): results[ResearchStage(parent).value]
                    for parent in StageContract.for_run(
                        run_spec, stage
                    ).required_parent_stages
                },
            )
            results[stage_name] = attempt.result_hash
            if stage in _PARENT_STAGES:
                direct[stage] = verified
        if tuple(sorted(direct, key=lambda item: item.value)) != _PARENT_STAGES:
            raise _error(
                "parent_runtime_authority_mismatch", "direct parent stages differ"
            )
        requested_parents = {
            ResearchStage(stage): digest
            for stage, digest in request.parent_artifact_hashes.items()
        }
        if requested_parents != {
            stage: direct[stage].payload_hash for stage in _PARENT_STAGES
        }:
            raise _error(
                "parent_runtime_authority_mismatch", "request parents differ"
            )
        try:
            current = self.runtime.get_attempt(request.attempt_id)
        except (KeyError, ValueError) as exc:
            raise _error(
                "parent_runtime_authority_mismatch", "model attempt is missing"
            ) from exc
        expected_model_input = stage_input_hash(
            experiment_spec_hash=experiment_spec.content_hash,
            stage=model_stage,
            prior_result_hashes=results,
        )
        if (
            current.status is not AttemptStatus.RUNNING
            or current.experiment_spec_hash != experiment_spec.content_hash
            or current.stage != model_stage
            or current.attempt_id != request.attempt_id
            or current.attempt_number != request.attempt_number
            or current.input_hash != expected_model_input
        ):
            raise _error(
                "parent_runtime_authority_mismatch", "model attempt authority differs"
            )
        try:
            lease_expires_at = datetime.fromisoformat(
                current.lease_expires_at.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise _error(
                "parent_runtime_authority_mismatch",
                "model attempt lease timestamp differs",
            ) from exc
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise _error(
                "parent_runtime_authority_mismatch",
                "model attempt authority clock differs",
            )
        if lease_expires_at.tzinfo is None or lease_expires_at.astimezone(
            timezone.utc
        ) <= now.astimezone(timezone.utc):
            raise _error(
                "parent_runtime_authority_mismatch",
                "model attempt lease is expired",
            )
        return _VerifiedRuntimeAuthority(
            parents=MappingProxyType(direct),
            current_attempt_id=current.attempt_id,
            current_attempt_number=current.attempt_number,
            current_attempt_input_hash=current.input_hash,
        )

    @staticmethod
    def _validate_registered_authority(
        *,
        manifest: ModelTrainingInputManifestV2,
        request: ScientificStageRequest,
        run_spec: ResearchRunSpec,
        experiment_spec: ExperimentSpec,
        scientific_lineage_manifest: ResearchScientificLineageManifestV2,
    ) -> None:
        scientific_lineage_manifest.validate_for(run_spec)
        if tuple(experiment_spec.factor_spec_hashes) != tuple(
            run_spec.factor_spec_hashes
        ):
            raise _error("request_binding_mismatch", "run/experiment factors differ")
        for name in ("label_spec_hash", "validation_spec_hash", "model_spec_hash"):
            if getattr(experiment_spec, name) != getattr(run_spec, name):
                raise _error(
                    "request_binding_mismatch", f"run/experiment {name} differs"
                )
        expected_contract = StageContract.for_run(
            run_spec, ResearchStage.MODEL_TRAINING
        )
        if request.contract.content_hash != expected_contract.content_hash:
            raise _error("request_binding_mismatch", "model stage contract differs")
        manifest._validate_request_and_lineage(
            request=request,
            scientific_lineage_manifest=scientific_lineage_manifest,
            experiment_spec=experiment_spec,
        )

    def _load_parent(
        self,
        *,
        experiment_spec: ExperimentSpec,
        run_spec: ResearchRunSpec,
        stage: ResearchStage,
        artifact_hash: str,
        expected_attempt_id: int,
        expected_attempt_number: int,
        expected_input_hash: str,
        expected_parent_hashes: Mapping[ResearchStage, str],
    ) -> _VerifiedParent:
        descriptor = dict(
            self.registry.load_artifact_descriptor(
                experiment_spec.content_hash, artifact_hash
            )
        )
        expected_fields = {
            "artifact_hash",
            "logical_name",
            "kind",
            "location",
            "media_type",
            "size_bytes",
            "stage",
            "attempt_id",
            "input_hash",
        }
        if set(descriptor) != expected_fields:
            raise _error("parent_artifact_mismatch", "registry descriptor fields differ")
        if (
            descriptor["artifact_hash"] != artifact_hash
            or descriptor["stage"] != stage.value
            or descriptor["kind"] != "governed_research_stage"
            or descriptor["media_type"] != "application/json"
            or descriptor["attempt_id"] != expected_attempt_id
            or descriptor["input_hash"] != expected_input_hash
        ):
            raise _error("parent_artifact_mismatch", f"{stage.value} descriptor differs")
        attempt_id = descriptor["attempt_id"]
        input_hash = descriptor["input_hash"]
        if (
            not isinstance(attempt_id, int)
            or isinstance(attempt_id, bool)
            or attempt_id <= 0
            or not isinstance(input_hash, str)
        ):
            raise _error(
                "parent_artifact_mismatch",
                f"{stage.value} attempt/input descriptor differs",
            )
        try:
            require_sha256(input_hash, name=f"{stage.value} parent input_hash")
        except ValueError as exc:
            raise _error(
                "parent_artifact_mismatch",
                f"{stage.value} input identity is invalid",
            ) from exc
        checkpoints = self.runtime.list_checkpoints(expected_attempt_id)
        if (
            len(checkpoints) != 1
            or checkpoints[0].checkpoint_name != "stage_result"
            or checkpoints[0].artifact_hash != artifact_hash
            or checkpoints[0].location != descriptor["location"]
        ):
            raise _error(
                "parent_runtime_authority_mismatch",
                f"{stage.value} checkpoint differs",
            )
        try:
            record = ContentArtifactRecord(
                logical_name=str(descriptor["logical_name"]),
                location=str(descriptor["location"]),
                sha256=artifact_hash,
                size_bytes=int(cast(int, descriptor["size_bytes"])),
                media_type="application/json",
                role="experiment_stage_result",
            )
            payload = self.artifact_store.read_bytes(record)
            contract = StageContract.for_run(run_spec, stage)
            typed_descriptor = StageArtifactDescriptor.from_payload(contract, payload)
        except (ContentArtifactError, OSError, TypeError, ValueError) as exc:
            raise _error(
                "parent_artifact_mismatch", f"{stage.value} payload verification failed"
            ) from exc
        if typed_descriptor.artifact_hash != artifact_hash:
            raise _error("parent_artifact_mismatch", f"{stage.value} hash differs")
        if dict(typed_descriptor.parent_artifact_hashes) != dict(
            expected_parent_hashes
        ):
            raise _error(
                "parent_runtime_authority_mismatch",
                f"{stage.value} payload parents differ",
            )
        body = _governed_body(payload)
        scientific_value = body["scientific_result"]
        if not isinstance(scientific_value, Mapping):
            raise _error("parent_artifact_mismatch", "scientific result is not an object")
        try:
            result = ScientificStageResult.from_mapping(scientific_value)
        except (TypeError, ValueError) as exc:
            raise _error("parent_artifact_mismatch", "scientific result is invalid") from exc
        if result.stage is not stage:
            raise _error("parent_artifact_mismatch", "scientific result stage differs")
        memory = body["memory_record"]
        if (
            not isinstance(memory, Mapping)
            or memory.get("result_hash") != result.content_hash
            or body["memory_record_hash"] != hash_json(dict(memory))
        ):
            raise _error("parent_artifact_mismatch", "memory/result binding differs")
        return _VerifiedParent(
            payload_hash=artifact_hash,
            descriptor_hash=cast(str, hash_json(descriptor)),
            attempt_binding=(
                GovernedParentAttemptBindingV1(
                    stage=stage,
                    artifact_hash=artifact_hash,
                    artifact_descriptor_hash=cast(str, hash_json(descriptor)),
                    attempt_id=expected_attempt_id,
                    attempt_number=expected_attempt_number,
                    input_hash=expected_input_hash,
                    checkpoint_hash=checkpoints[0].artifact_hash,
                    checkpoint_location=checkpoints[0].location,
                    stage_contract_hash=StageContract.for_run(
                        run_spec, stage
                    ).content_hash,
                )
                if stage in _PARENT_STAGES
                else None
            ),
            result=result,
        )

    @staticmethod
    def _validate_eligibility_policy_authority(
        *,
        manifest: ModelTrainingInputManifestV2,
        run_spec: ResearchRunSpec,
    ) -> None:
        """Bind eligibility sources to the exact label-free factor contract.

        A policy's ``requires_label_data`` flag is descriptive rather than an
        authority proof.  The trusted resolver therefore derives the only
        permissible source identities from the registered factor-generation
        contract and rejects self-declared, label/validation/protected inputs.
        """

        factor_contract = StageContract.for_run(
            run_spec, ResearchStage.FACTOR_GENERATION
        )
        allowed_sources = tuple(
            sorted(
                {
                    digest
                    for role, digest in factor_contract.component_bindings.items()
                    if role.startswith("data:")
                }
            )
        )
        policy = manifest.scoring_eligibility_policy
        if policy.source_data_hashes != allowed_sources:
            raise _error(
                "eligibility_policy_authority_mismatch",
                "eligibility sources differ from the factor-stage data authority",
            )
        all_bindings = dict(run_spec.component_bindings())
        forbidden_roles = {
            role
            for role in all_bindings
            if role in {"label", "validation", "evaluation", "model"}
            or role
            in {
                "data:partition:test",
                "data:partition:holdout",
            }
        }
        forbidden_hashes = {all_bindings[role] for role in forbidden_roles}
        semantic_hashes = {
            policy.universe_membership_hash,
            policy.availability_policy_hash,
            policy.timestamp_policy_hash,
        }
        if semantic_hashes & forbidden_hashes:
            raise _error(
                "eligibility_policy_authority_mismatch",
                "eligibility policy semantics reference a forbidden component",
            )

    @staticmethod
    def _validate_parent_results(
        *,
        manifest: ModelTrainingInputManifestV2,
        parents: Mapping[ResearchStage, _VerifiedParent],
    ) -> None:
        factor_payload = parents[ResearchStage.FACTOR_GENERATION].result.result_payload
        raw_features = factor_payload.get("model_feature_artifacts")
        if not isinstance(raw_features, Mapping):
            raise _error("parent_binding_mismatch", "factor feature references missing")
        try:
            feature_references = {
                str(name): DataFrameArtifactReferenceV2.from_mapping(reference)
                for name, reference in raw_features.items()
                if isinstance(reference, Mapping)
            }
        except (TypeError, ValueError) as exc:
            raise _error("parent_binding_mismatch", "factor feature references invalid") from exc
        if len(feature_references) != len(raw_features) or feature_references != dict(
            manifest.feature_frames
        ):
            raise _error("parent_binding_mismatch", "factor feature references differ")
        manifest._validate_eligibility_source_payload(factor_payload)

        label_payload = parents[ResearchStage.LABEL_BUILDING].result.result_payload
        label_bundle = label_payload.get("model_label_artifacts")
        if not isinstance(label_bundle, Mapping):
            raise _error("parent_binding_mismatch", "label artifact bundle missing")
        expected_label_bundle = {
            "schema_version": "model-label-artifacts/v1",
            "label_spec_hash": manifest.label_spec_hash,
            "label_view_hash": manifest.label_view_hash,
            "label_benchmark_hash": manifest.label_benchmark_hash,
            "label_values_hash": manifest.label_values_hash,
            "label_windows_hash": manifest.label_windows_hash,
            "label_validity_hash": manifest.label_validity_hash,
            "label_diagnostics_hash": manifest.label_diagnostics_hash,
            "label_values": manifest.label_values.to_dict(),
            "label_windows": manifest.label_windows.to_dict(),
            "label_validity": manifest.label_validity.to_dict(),
            "label_diagnostics": manifest.label_diagnostics.to_dict(),
        }
        if dict(label_bundle) != expected_label_bundle:
            raise _error("parent_binding_mismatch", "label artifact bundle differs")

        validation_payload = parents[
            ResearchStage.VALIDATION_SPLIT
        ].result.result_payload
        validation_bundle = validation_payload.get("model_validation_artifacts")
        if not isinstance(validation_bundle, Mapping):
            raise _error("parent_binding_mismatch", "validation bundle missing")
        expected_validation_bundle = {
            "schema_version": "model-validation-artifacts/v1",
            "validation_spec_hash": manifest.validation_spec_hash,
            "validation_spec": manifest.validation_spec.to_dict(),
            "validation_receipt_hash": manifest.validation_receipt_hash,
            "validation_receipt": manifest.validation_receipt.to_dict(),
            "trading_calendar_content_hash": manifest.trading_calendar_content_hash,
            "trading_calendar": manifest.trading_calendar.to_dict(),
        }
        if dict(validation_bundle) != expected_validation_bundle:
            raise _error("parent_binding_mismatch", "validation bundle differs")

    def _load_frames(
        self,
        *,
        manifest: ModelTrainingInputManifestV2,
        experiment_spec: ExperimentSpec,
    ) -> tuple[LoadedModelTrainingInputs, tuple[int, int, int]]:
        references = {
            **{
                f"feature:{name}": reference
                for name, reference in manifest.feature_frames.items()
            },
            "label_values": manifest.label_values,
            "label_windows": manifest.label_windows,
            "label_validity": manifest.label_validity,
            "label_diagnostics": manifest.label_diagnostics,
            "scoring_eligibility": manifest.scoring_eligibility,
        }
        memory_limit = experiment_spec.resource_budget.maximum_peak_memory_bytes
        declared_compressed = sum(item.size_bytes for item in references.values())
        declared_uncompressed = sum(
            item.parquet_uncompressed_bytes for item in references.values()
        )
        conservative_working_set = (
            declared_compressed
            + _PARQUET_WORKING_SET_MULTIPLIER * declared_uncompressed
        )
        if (
            declared_compressed > memory_limit
            or declared_uncompressed > memory_limit
            or conservative_working_set > memory_limit
        ):
            raise _error(
                "resource_budget_exceeded",
                "conservative parquet working set exceeds budget",
            )
        frames: dict[str, pd.DataFrame] = {}
        total_compressed = 0
        total_uncompressed = 0
        total_loaded = 0
        for name, reference in references.items():
            try:
                payload = self.artifact_store.read_bytes(reference.to_artifact_record())
            except (ContentArtifactError, OSError, TypeError, ValueError) as exc:
                raise _error("artifact_payload_mismatch", f"{name} reload failed") from exc
            total_compressed += len(payload)
            frame = reference.decode_parquet_payload(
                payload,
                maximum_uncompressed_bytes=memory_limit - total_uncompressed,
            )
            total_uncompressed += reference.parquet_uncompressed_bytes
            total_loaded += int(frame.memory_usage(index=True, deep=True).sum())
            retained_working_set = (
                total_loaded * 2
                + len(payload)
                + _PARQUET_WORKING_SET_MULTIPLIER
                * reference.parquet_uncompressed_bytes
            )
            if (
                total_compressed > memory_limit
                or total_uncompressed > memory_limit
                or total_loaded > memory_limit
                or retained_working_set > memory_limit
            ):
                raise _error("resource_budget_exceeded", "loaded bundle exceeds budget")
            frames[name] = frame
        label_result = LabelResult(
            label_spec_hash=manifest.label_spec_hash,
            label_view_hash=manifest.label_view_hash,
            benchmark_hash=manifest.label_benchmark_hash,
            labels_hash=manifest.label_values_hash,
            windows_hash=manifest.label_windows_hash,
            validity_hash=manifest.label_validity_hash,
            diagnostics_hash=manifest.label_diagnostics_hash,
            labels=frames["label_values"],
            label_windows=frames["label_windows"],
            validity=frames["label_validity"],
            diagnostics=frames["label_diagnostics"],
        )
        loaded = LoadedModelTrainingInputs(
            manifest=manifest,
            feature_frames={
                name.removeprefix("feature:"): frame
                for name, frame in frames.items()
                if name.startswith("feature:")
            },
            label_result=label_result,
            scoring_eligibility=frames["scoring_eligibility"],
        )
        return loaded, (total_compressed, total_uncompressed, total_loaded)


def _governed_body(payload: bytes) -> Mapping[str, object]:
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, Mapping):  # pragma: no cover - descriptor checked.
        raise _error("parent_artifact_mismatch", "stage envelope is invalid")
    body = decoded.get("payload")
    if not isinstance(body, Mapping) or set(body) != _GOVERNED_BODY_FIELDS:
        raise _error("parent_artifact_mismatch", "governed stage body fields differ")
    if body.get("schema_version") != "governed-agent-stage/v1":
        raise _error("parent_artifact_mismatch", "governed stage body schema differs")
    return cast(Mapping[str, object], body)


def _request_binding_hash(request: ScientificStageRequest) -> str:
    return cast(
        str,
        hash_json(
            {
                "schema_version": "scientific-stage-request-binding/v1",
                "research_run_spec_hash": request.research_run_spec_hash,
                "experiment_spec_hash": request.experiment_spec_hash,
                "stage": ResearchStage(request.stage).value,
                "attempt_id": request.attempt_id,
                "attempt_number": request.attempt_number,
                "contract_hash": request.contract.content_hash,
                "parent_artifact_hashes": {
                    ResearchStage(stage).value: digest
                    for stage, digest in request.parent_artifact_hashes.items()
                },
                "agent_decision_hash": request.agent_decision.content_hash,
            }
        ),
    )


__all__ = [
    "AuthorityBoundModelTrainingInputs",
    "GovernedParentAttemptBindingV1",
    "ModelTrainingInputAuthorityReceiptV2",
    "ModelTrainingInputResolver",
]
