"""Concrete, resolver-mandatory nested-purged model-training stage.

The factory in this module is the production research composition path for
``ResearchStage.MODEL_TRAINING``.  It does not accept a caller-authored input
manifest: every invocation reconstructs one from registered parent results,
cross-checks the live runtime attempt chain, reloads every content-addressed
frame, and only then exposes data to the Agent-only inner-selection facet.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
import resource
import sys
import time
from typing import Callable, Mapping, cast

from alpha_research.core.hashing import hash_json, require_sha256
from alpha_research.experiments import ExperimentSpec
from alpha_research.experiments.registry import ExperimentRegistry
from alpha_research.models.nested_selection import (
    verify_nested_selection_phase_one_manifest_inputs,
)
from alpha_research.models.selection_spec import NestedPurgedSelectionSpec
from alpha_research.orchestration import ExperimentRuntime, reported_usage
from alpha_research.research.execution import (
    JsonValue,
    ScientificStageImplementation,
    ScientificStageRequest,
    ScientificStageResult,
)
from alpha_research.research.lineage import ResearchScientificLineageManifestV2
from alpha_research.research.model_training_input_resolver import (
    AuthorityBoundModelTrainingInputs,
    ModelTrainingInputAuthorityReceiptV2,
    ModelTrainingInputResolver,
)
from alpha_research.research.model_training_inputs import ModelTrainingInputManifestV2
from alpha_research.research.nested_selection_control import (
    AgentNestedSelectionController,
    VerifiedNestedSelectionPhaseOne,
)
from alpha_research.research.spec import ResearchRunSpec, ResearchStage
from factor_production.v5.artifacts import ArtifactRecord, ArtifactStore


_MANIFEST_MEDIA_TYPE = (
    "application/vnd.alpha-research.model-training-input-manifest+json"
)
_RECEIPT_MEDIA_TYPE = (
    "application/vnd.alpha-research.model-training-input-authority-receipt+json"
)


@dataclass(frozen=True, slots=True)
class ModelTrainingInputDocumentReferenceV1:
    """Strict content-addressed reference embedded in the model-stage result."""

    document_kind: str
    logical_name: str
    location: str
    payload_sha256: str
    size_bytes: int
    content_hash: str
    media_type: str
    schema_version: str = "model-training-input-document-reference/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "model-training-input-document-reference/v1":
            raise ValueError("unsupported model-training document reference")
        if self.document_kind not in {"input_manifest", "authority_receipt"}:
            raise ValueError("model-training document kind is invalid")
        if (
            not isinstance(self.logical_name, str)
            or not self.logical_name
            or self.logical_name != self.logical_name.strip()
        ):
            raise ValueError("model-training document logical name is invalid")
        path = PurePosixPath(self.location)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("model-training document location is unsafe")
        object.__setattr__(self, "location", path.as_posix())
        payload_hash = require_sha256(
            self.payload_sha256, name="model-training document payload_sha256"
        )
        content_hash = require_sha256(
            self.content_hash, name="model-training document content_hash"
        )
        if payload_hash != content_hash:
            raise ValueError("model-training document content identity differs")
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes <= 0
        ):
            raise ValueError("model-training document size must be positive")
        expected_media = {
            "input_manifest": _MANIFEST_MEDIA_TYPE,
            "authority_receipt": _RECEIPT_MEDIA_TYPE,
        }[self.document_kind]
        if self.media_type != expected_media:
            raise ValueError("model-training document media type differs")

    @property
    def reference_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "document_kind": self.document_kind,
            "logical_name": self.logical_name,
            "location": self.location,
            "payload_sha256": self.payload_sha256,
            "size_bytes": self.size_bytes,
            "content_hash": self.content_hash,
            "media_type": self.media_type,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "ModelTrainingInputDocumentReferenceV1":
        expected = frozenset(
            {
                "schema_version",
                "document_kind",
                "logical_name",
                "location",
                "payload_sha256",
                "size_bytes",
                "content_hash",
                "media_type",
            }
        )
        if frozenset(value) != expected:
            raise ValueError("model-training document reference fields differ")
        return cls(
            schema_version=cast(str, value["schema_version"]),
            document_kind=cast(str, value["document_kind"]),
            logical_name=cast(str, value["logical_name"]),
            location=cast(str, value["location"]),
            payload_sha256=cast(str, value["payload_sha256"]),
            size_bytes=cast(int, value["size_bytes"]),
            content_hash=cast(str, value["content_hash"]),
            media_type=cast(str, value["media_type"]),
        )

    def to_artifact_record(self) -> ArtifactRecord:
        return ArtifactRecord(
            logical_name=self.logical_name,
            location=self.location,
            sha256=self.payload_sha256,
            size_bytes=self.size_bytes,
            media_type=self.media_type,
            role="model_training_control",
        )

    @classmethod
    def bind_record(
        cls,
        *,
        document_kind: str,
        record: ArtifactRecord,
        content_hash: str,
    ) -> "ModelTrainingInputDocumentReferenceV1":
        return cls(
            document_kind=document_kind,
            logical_name=record.logical_name,
            location=record.location,
            payload_sha256=record.sha256,
            size_bytes=record.size_bytes,
            content_hash=content_hash,
            media_type=record.media_type,
        )


class ResolvedNestedPurgedModelTrainingFactory:
    """Bind the Agent inner selector behind trusted input reconstruction."""

    _implementation_hash: str
    _implementation_id: str
    _lineage: ResearchScientificLineageManifestV2
    _run_spec: ResearchRunSpec
    _selection_spec: NestedPurgedSelectionSpec

    __slots__ = (
        "_implementation_hash",
        "_implementation_id",
        "_lineage",
        "_run_spec",
        "_selection_spec",
    )

    def __init__(
        self,
        *,
        run_spec: ResearchRunSpec,
        scientific_lineage_manifest: ResearchScientificLineageManifestV2,
        selection_spec: NestedPurgedSelectionSpec,
        implementation_id: str,
        implementation_hash: str,
    ) -> None:
        if not isinstance(run_spec, ResearchRunSpec):
            raise TypeError("model-training factory run specification differs")
        if not isinstance(
            scientific_lineage_manifest, ResearchScientificLineageManifestV2
        ):
            raise TypeError("model-training factory requires V2 scientific lineage")
        if not isinstance(selection_spec, NestedPurgedSelectionSpec):
            raise TypeError("model-training factory selection specification differs")
        scientific_lineage_manifest.validate_for(run_spec)
        if scientific_lineage_manifest.nested_selection_spec_hash != (
            selection_spec.content_hash
        ):
            raise ValueError("model-training factory selection lineage differs")
        if (
            not isinstance(implementation_id, str)
            or not 1 <= len(implementation_id) <= 128
            or not implementation_id[0].isalnum()
            or any(
                character
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
                for character in implementation_id
            )
        ):
            raise ValueError("model-training implementation_id is invalid")
        if implementation_id.lower() in {"noop", "placeholder", "stub", "mock"}:
            raise ValueError("placeholder model-training implementation is forbidden")
        require_sha256(implementation_hash, name="model-training implementation_hash")
        object.__setattr__(self, "_run_spec", run_spec)
        object.__setattr__(self, "_lineage", scientific_lineage_manifest)
        object.__setattr__(self, "_selection_spec", selection_spec)
        object.__setattr__(self, "_implementation_id", implementation_id)
        object.__setattr__(self, "_implementation_hash", implementation_hash)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ResolvedNestedPurgedModelTrainingFactory is immutable")

    @property
    def selection_spec_hash(self) -> str:
        return cast(str, self._selection_spec.content_hash)

    def bind(
        self,
        *,
        nested_selection: AgentNestedSelectionController,
        run_spec: ResearchRunSpec,
        scientific_lineage_manifest: ResearchScientificLineageManifestV2,
        experiment_spec: ExperimentSpec,
        registry: ExperimentRegistry,
        runtime: ExperimentRuntime,
        workspace_root: str | Path,
        clock: Callable[[], datetime] | None = None,
    ) -> ScientificStageImplementation:
        if type(nested_selection) is not AgentNestedSelectionController:
            raise TypeError("nested_selection must be an exact Agent controller")
        if not isinstance(run_spec, ResearchRunSpec):
            raise TypeError("model-training bound run specification differs")
        if not isinstance(
            scientific_lineage_manifest, ResearchScientificLineageManifestV2
        ):
            raise TypeError("model-training bound scientific lineage differs")
        if run_spec.content_hash != self._run_spec.content_hash:
            raise ValueError("model-training bound run specification differs")
        if scientific_lineage_manifest.content_hash != self._lineage.content_hash:
            raise ValueError("model-training bound scientific lineage differs")
        if type(experiment_spec) is not ExperimentSpec:
            raise TypeError("model-training experiment specification differs")
        if type(registry) is not ExperimentRegistry:
            raise TypeError("model-training registry authority differs")
        if type(runtime) is not ExperimentRuntime:
            raise TypeError("model-training runtime authority differs")
        if runtime.spec.content_hash != experiment_spec.content_hash:
            raise ValueError("model-training runtime experiment differs")
        if not isinstance(workspace_root, (str, Path)):
            raise TypeError("model-training workspace root differs")
        if clock is not None and not callable(clock):
            raise TypeError("model-training clock must be callable or None")
        selection_spec = self._selection_spec
        implementation_id = self._implementation_id
        implementation_hash = self._implementation_hash
        if selection_spec.content_hash != scientific_lineage_manifest.nested_selection_spec_hash:
            raise ValueError("model-training bound selection lineage differs")
        return _ResolvedNestedPurgedModelTrainingImplementation(
            selection_spec=selection_spec,
            implementation_id=implementation_id,
            implementation_hash=implementation_hash,
            nested_selection=nested_selection,
            run_spec=run_spec,
            scientific_lineage_manifest=scientific_lineage_manifest,
            experiment_spec_hash=experiment_spec.content_hash,
            registry=registry,
            runtime=runtime,
            workspace_root=Path(workspace_root).resolve(),
            clock=clock,
        )


class _ResolvedNestedPurgedModelTrainingImplementation:
    __slots__ = (
        "_experiment_spec_hash",
        "_implementation_hash",
        "_implementation_id",
        "_lineage",
        "_nested_selection",
        "_nested_selection_authority_hash",
        "_registry",
        "_run_spec",
        "_runtime",
        "_selection_spec",
        "_clock",
        "_workspace_root",
    )

    def __init__(
        self,
        *,
        selection_spec: NestedPurgedSelectionSpec,
        implementation_id: str,
        implementation_hash: str,
        nested_selection: AgentNestedSelectionController,
        run_spec: ResearchRunSpec,
        scientific_lineage_manifest: ResearchScientificLineageManifestV2,
        experiment_spec_hash: str,
        registry: ExperimentRegistry,
        runtime: ExperimentRuntime,
        workspace_root: Path,
        clock: Callable[[], datetime] | None,
    ) -> None:
        self._selection_spec = selection_spec
        self._implementation_id = implementation_id
        self._implementation_hash = implementation_hash
        self._nested_selection = nested_selection
        self._nested_selection_authority_hash = require_sha256(
            nested_selection.authority_hash,
            name="model-training nested-selection authority hash",
        )
        self._run_spec = run_spec
        self._lineage = scientific_lineage_manifest
        self._experiment_spec_hash = require_sha256(
            experiment_spec_hash,
            name="model-training experiment specification hash",
        )
        self._registry = registry
        self._runtime = runtime
        self._clock = clock
        self._workspace_root = workspace_root

    def __call__(self, request: ScientificStageRequest) -> ScientificStageResult:
        if ResearchStage(request.stage) is not ResearchStage.MODEL_TRAINING:
            raise ValueError("resolved model-training stage differs")
        if request.experiment_spec_hash != self._experiment_spec_hash:
            raise ValueError("resolved model-training experiment differs")
        started_wall = time.perf_counter()
        started_cpu = time.process_time()
        artifact_store = ArtifactStore(self._workspace_root)
        resolver = ModelTrainingInputResolver(
            registry=self._registry,
            artifact_store=artifact_store,
            runtime=self._runtime,
            clock=self._clock,
        )
        manifest = resolver.assemble_manifest(
            request=request,
            run_spec=self._run_spec,
            scientific_lineage_manifest=self._lineage,
            selection_spec=self._selection_spec,
        )
        manifest_reference = _publish_manifest(artifact_store, manifest)
        authority_bound = resolver.resolve(
            manifest=manifest,
            request=request,
            run_spec=self._run_spec,
            scientific_lineage_manifest=self._lineage,
        )
        receipt_reference = _publish_authority_receipt(
            artifact_store, authority_bound.authority_receipt
        )
        phase_one = self._select_inner(
            authority_bound=authority_bound,
        )
        phase_one_reference = phase_one.reference
        phase_one_manifest = phase_one.manifest
        receipt = authority_bound.authority_receipt
        parent_bindings = {
            ResearchStage(stage).value: binding.to_dict()
            for stage, binding in receipt.parent_attempt_bindings.items()
        }
        payload: dict[str, JsonValue] = {
            "phase_one_manifest_reference": cast(
                dict[str, JsonValue], phase_one_reference.to_dict()
            ),
            "verified_phase_one_manifest_hash": phase_one_manifest.content_hash,
            "verified_phase_one_seal_hash": phase_one.content_hash,
            "nested_selection_authority_hash": (self._nested_selection_authority_hash),
            "model_training_input_manifest_hash": manifest.content_hash,
            "model_training_input_manifest_reference": cast(
                dict[str, JsonValue], manifest_reference.to_dict()
            ),
            "model_training_input_authority_receipt_hash": receipt.content_hash,
            "model_training_input_authority_receipt_reference": cast(
                dict[str, JsonValue], receipt_reference.to_dict()
            ),
            "model_training_parent_attempt_bindings": cast(
                dict[str, JsonValue], parent_bindings
            ),
            "research_only": True,
            "production_ready": False,
        }
        evidence = tuple(
            dict.fromkeys(
                (
                    manifest.content_hash,
                    manifest_reference.reference_hash,
                    receipt.content_hash,
                    receipt_reference.reference_hash,
                    phase_one_reference.manifest_id,
                    phase_one_reference.document_sha256,
                    phase_one.content_hash,
                    self._nested_selection_authority_hash,
                    *(
                        binding.content_hash
                        for binding in receipt.parent_attempt_bindings.values()
                    ),
                )
            )
        )
        loaded = authority_bound.loaded
        return ScientificStageResult(
            stage=ResearchStage.MODEL_TRAINING,
            implementation_id=self._implementation_id,
            implementation_hash=self._implementation_hash,
            result_payload=payload,
            evidence_hashes=evidence,
            sanitized_feedback_codes=(),
            usage=reported_usage(
                wall_seconds=max(0.0, time.perf_counter() - started_wall),
                cpu_seconds=max(0.0, time.process_time() - started_cpu),
                peak_memory_bytes=_peak_rss_bytes(),
                disk_write_bytes=(
                    manifest_reference.size_bytes
                    + receipt_reference.size_bytes
                    + phase_one_reference.size_bytes
                ),
            ),
            input_rows=len(loaded.label_result.labels.index),
            output_rows=len(self._selection_spec.outer_plans),
            symbols=len(loaded.label_result.labels.columns),
            disk_read_bytes=receipt.total_compressed_bytes,
            cache_hits=0,
            cache_misses=0,
            worker_count=1,
        )

    def _select_inner(
        self,
        *,
        authority_bound: AuthorityBoundModelTrainingInputs,
    ) -> VerifiedNestedSelectionPhaseOne:
        loaded = authority_bound.loaded
        if (
            self._nested_selection.authority_hash
            != self._nested_selection_authority_hash
        ):
            raise ValueError("nested selection authority changed before execution")
        verified = AgentNestedSelectionController._select_inner_verified(
            self._nested_selection,
            self._selection_spec,
            feature_signals=loaded.feature_frames,
            labels=loaded.label_result,
            outer_validation_spec=loaded.manifest.validation_spec,
            outer_validation_receipt=loaded.manifest.validation_receipt,
            calendar=loaded.manifest.trading_calendar,
            scoring_eligibility=loaded.scoring_eligibility,
        )
        if (
            verified.authority_hash != self._nested_selection_authority_hash
            or self._nested_selection.authority_hash
            != self._nested_selection_authority_hash
        ):
            raise ValueError("nested selection authority binding differs")
        restored = AgentNestedSelectionController._reload_phase_one_manifest(
            self._nested_selection,
            verified.reference,
        )
        if restored != verified.manifest:
            raise ValueError("nested selection verified manifest differs")
        loaded.verify_content()
        verify_nested_selection_phase_one_manifest_inputs(
            restored,
            spec=self._selection_spec,
            feature_signals=loaded.feature_frames,
            labels=loaded.label_result,
            outer_validation_spec=loaded.manifest.validation_spec,
            outer_validation_receipt=loaded.manifest.validation_receipt,
            calendar=loaded.manifest.trading_calendar,
            scoring_eligibility=loaded.scoring_eligibility,
        )
        return verified


def _publish_manifest(
    store: ArtifactStore,
    manifest: ModelTrainingInputManifestV2,
) -> ModelTrainingInputDocumentReferenceV1:
    payload = manifest.to_wire_bytes()
    record = store.put_bytes(
        f"model_training_input_manifest_{manifest.content_hash[:16]}",
        payload,
        media_type=_MANIFEST_MEDIA_TYPE,
        role="model_training_control",
    )
    reference = ModelTrainingInputDocumentReferenceV1.bind_record(
        document_kind="input_manifest",
        record=record,
        content_hash=manifest.content_hash,
    )
    restored = ModelTrainingInputManifestV2.from_wire_bytes(
        store.read_bytes(reference.to_artifact_record())
    )
    if restored != manifest:
        raise ValueError("published model-training manifest differs")
    return reference


def _publish_authority_receipt(
    store: ArtifactStore,
    receipt: ModelTrainingInputAuthorityReceiptV2,
) -> ModelTrainingInputDocumentReferenceV1:
    payload = receipt.to_wire_bytes()
    record = store.put_bytes(
        f"model_training_authority_receipt_{receipt.content_hash[:16]}",
        payload,
        media_type=_RECEIPT_MEDIA_TYPE,
        role="model_training_control",
    )
    reference = ModelTrainingInputDocumentReferenceV1.bind_record(
        document_kind="authority_receipt",
        record=record,
        content_hash=receipt.content_hash,
    )
    restored = ModelTrainingInputAuthorityReceiptV2.from_wire_bytes(
        store.read_bytes(reference.to_artifact_record())
    )
    if restored != receipt:
        raise ValueError("published model-training authority receipt differs")
    return reference


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


__all__ = [
    "ModelTrainingInputDocumentReferenceV1",
    "ResolvedNestedPurgedModelTrainingFactory",
]
