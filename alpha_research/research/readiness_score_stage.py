"""Authority-bound SCORE_CONSTRUCTION stage for ML-F controlled recipes.

The public factory freezes the run, scientific lineage, preregistered readiness
plan and closed recipe set before execution.  Its bound callable accepts only a
``ScientificStageRequest``.  It reconstructs the authoritative MODEL_TRAINING
parent from the registry/runtime/checkpoint chain, obtains an already-completed
outer evaluation through the read-only audit facet, publishes the closed score
recipes, and immediately reopens every artifact before returning.

No public API in this module accepts score frames, labels, or a
``CompletedOuterEvaluation``.  Outputs are validation-only research evidence;
they cannot authorize protected-data access or production release.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import resource
import sqlite3
import sys
import time
from types import MappingProxyType
from typing import Final, cast, final

from alpha_research.core.hashing import hash_json, require_sha256
from alpha_research.experiments import DataPartition, ExperimentSpec
from alpha_research.experiments.controller import stage_input_hash
from alpha_research.experiments.registry import ExperimentRegistry
from alpha_research.models.nested_outer_evaluation_store import (
    CompletedOuterEvaluation,
)
from alpha_research.orchestration import AttemptStatus, ExperimentRuntime, reported_usage
from alpha_research.research.artifacts import StageArtifactDescriptor, StageContract
from alpha_research.research.execution import (
    AgentStageDecision,
    JsonValue,
    ScientificStageImplementation,
    ScientificStageRequest,
    ScientificStageResult,
)
from alpha_research.research.lineage import ResearchScientificLineageManifestV2
from alpha_research.research.nested_selection_control import (
    NestedOuterEvaluationAuditReader,
)
from alpha_research.research.readiness_input_resolver import (
    ResearchReadinessInputResolver,
    _VerifiedParent,
    _governed_body,
)
from alpha_research.research.readiness_inputs import (
    ResearchReadinessInputError,
    ResearchReadinessPlanV1,
)
from alpha_research.research.readiness_upstream_bundles import (
    ResearchReadinessScoreConstructionResultV2,
    ResearchReadinessScoreRecipeSetV2,
    ResearchReadinessScoreUpstreamProducer,
)
from alpha_research.research.spec import ResearchRunSpec, ResearchStage
from factor_production.v5.artifacts import ArtifactRecord, ArtifactStore
from factor_production.v5.artifacts.manifest import ArtifactError


_BOUND_SCORE_STAGE_TOKEN: Final = object()
_DIRECT_SCORE_PARENTS: Final = (
    ResearchStage.FACTOR_EVALUATION,
    ResearchStage.MODEL_TRAINING,
)


class ResearchReadinessScoreStageError(RuntimeError):
    """Stable, sanitized failure from the controlled score stage."""

    def __init__(self, code: str, detail: str) -> None:
        if (
            not isinstance(code, str)
            or not code
            or not code[0].isalpha()
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in code)
        ):
            raise ValueError("research readiness score stage error code is unsafe")
        self.code = code
        self.detail = str(detail)
        super().__init__(f"{self.code}:{self.detail}")


def _stage_error(code: str, detail: str) -> ResearchReadinessScoreStageError:
    return ResearchReadinessScoreStageError(code, detail)


@dataclass(frozen=True, slots=True)
class ResearchReadinessScorePreflightEvidence:
    """Hash-only proof that authoritative outer evidence can be reopened."""

    model_parent_hash: str
    validation_partition_hash: str
    outer_validation_receipt_hash: str
    phase_one_manifest_hash: str
    outer_evaluation_completion_hash: str
    outer_evaluation_result_hash: str
    outer_execution_snapshot_hash: str
    research_only: bool = True
    production_ready: bool = False
    schema_version: str = "research-readiness-score-preflight/v2"

    def __post_init__(self) -> None:
        if self.schema_version != "research-readiness-score-preflight/v2":
            raise ValueError("unsupported readiness score preflight schema")
        if self.research_only is not True or self.production_ready is not False:
            raise ValueError("readiness score preflight must remain research-only")
        for name in (
            "model_parent_hash",
            "validation_partition_hash",
            "outer_validation_receipt_hash",
            "phase_one_manifest_hash",
            "outer_evaluation_completion_hash",
            "outer_evaluation_result_hash",
            "outer_execution_snapshot_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"score preflight {name}")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_parent_hash": self.model_parent_hash,
            "validation_partition_hash": self.validation_partition_hash,
            "outer_validation_receipt_hash": self.outer_validation_receipt_hash,
            "phase_one_manifest_hash": self.phase_one_manifest_hash,
            "outer_evaluation_completion_hash": (
                self.outer_evaluation_completion_hash
            ),
            "outer_evaluation_result_hash": self.outer_evaluation_result_hash,
            "outer_execution_snapshot_hash": self.outer_execution_snapshot_hash,
            "research_only": self.research_only,
            "production_ready": self.production_ready,
        }


@dataclass(frozen=True, slots=True)
class _ScoreVerifiedParent:
    """SCORE-local evidence for a direct parent outside readiness bindings."""

    payload_hash: str
    descriptor_hash: str
    result: ScientificStageResult


@dataclass(frozen=True, slots=True)
class _ScoreRuntimeAuthority:
    parents: Mapping[ResearchStage, _ScoreVerifiedParent | _VerifiedParent]
    current_attempt_id: int
    current_attempt_number: int
    current_attempt_input_hash: str
    current_worker_id: str
    current_lease_token_hash: str


@dataclass(frozen=True, slots=True)
class _ResolvedScoreInputs:
    authority: _ScoreRuntimeAuthority
    completed_outer_evaluation: CompletedOuterEvaluation


@final
class ResolvedResearchReadinessScoreFactory:
    """Freeze one exact, closed ML-F score construction before binding authority."""

    _implementation_hash: str
    _implementation_id: str
    _lineage: ResearchScientificLineageManifestV2
    _plan: ResearchReadinessPlanV1
    _recipes: ResearchReadinessScoreRecipeSetV2
    _run_spec: ResearchRunSpec

    __slots__ = (
        "_implementation_hash",
        "_implementation_id",
        "_lineage",
        "_plan",
        "_recipes",
        "_run_spec",
    )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("ResolvedResearchReadinessScoreFactory cannot be subclassed")

    def __init__(
        self,
        *,
        run_spec: ResearchRunSpec,
        scientific_lineage_manifest: ResearchScientificLineageManifestV2,
        readiness_plan: ResearchReadinessPlanV1,
        implementation_id: str,
        implementation_hash: str,
    ) -> None:
        if type(run_spec) is not ResearchRunSpec:
            raise TypeError("score factory run specification differs")
        if type(scientific_lineage_manifest) is not ResearchScientificLineageManifestV2:
            raise TypeError("score factory requires exact V2 scientific lineage")
        if type(readiness_plan) is not ResearchReadinessPlanV1:
            raise TypeError("score factory readiness plan differs")
        run_snapshot = ResearchRunSpec.from_mapping(run_spec.to_dict())
        lineage_snapshot = ResearchScientificLineageManifestV2.from_mapping(
            scientific_lineage_manifest.to_dict()
        )
        plan_snapshot = ResearchReadinessPlanV1.from_wire_bytes(
            readiness_plan.to_wire_bytes()
        )
        recipes = ResearchReadinessScoreRecipeSetV2.from_plan(plan_snapshot)
        _validate_scientific_bindings(
            run_spec=run_snapshot,
            scientific_lineage_manifest=lineage_snapshot,
            readiness_plan=plan_snapshot,
            recipes=recipes,
        )
        _validate_implementation_identity(implementation_id, implementation_hash)
        object.__setattr__(self, "_run_spec", run_snapshot)
        object.__setattr__(self, "_lineage", lineage_snapshot)
        object.__setattr__(self, "_plan", plan_snapshot)
        object.__setattr__(self, "_recipes", recipes)
        object.__setattr__(self, "_implementation_id", implementation_id)
        object.__setattr__(self, "_implementation_hash", implementation_hash)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("ResolvedResearchReadinessScoreFactory is immutable")

    @property
    def score_spec_hash(self) -> str:
        return cast(str, self._recipes.content_hash)

    @property
    def readiness_plan_hash(self) -> str:
        return cast(str, self._plan.content_hash)

    @property
    def nested_selection_spec_hash(self) -> str:
        return cast(str, self._recipes.nested_selection_spec_hash)

    def preflight_outer_completion(
        self,
        *,
        experiment_spec: ExperimentSpec,
        registry: ExperimentRegistry,
        runtime: ExperimentRuntime,
        artifact_store: ArtifactStore,
        outer_audit_reader: NestedOuterEvaluationAuditReader,
    ) -> ResearchReadinessScorePreflightEvidence:
        """Read and verify frozen MODEL/outer evidence without creating an attempt.

        This is an Assembly-side readiness check.  It neither reserves nor
        starts SCORE, and exposes only immutable evidence hashes, never frames,
        labels, predictions, or metrics.
        """

        _validate_preflight_authorities(
            experiment_spec=experiment_spec,
            registry=registry,
            runtime=runtime,
            artifact_store=artifact_store,
            outer_audit_reader=outer_audit_reader,
            run_spec=self._run_spec,
            scientific_lineage_manifest=self._lineage,
        )
        return _preflight_outer_completion(
            experiment_spec=experiment_spec,
            registry=registry,
            runtime=runtime,
            artifact_store=artifact_store,
            outer_audit_reader=outer_audit_reader,
            run_spec=self._run_spec,
            scientific_lineage_manifest=self._lineage,
            readiness_plan=self._plan,
        )

    def bind(
        self,
        *,
        run_spec: ResearchRunSpec,
        scientific_lineage_manifest: ResearchScientificLineageManifestV2,
        experiment_spec: ExperimentSpec,
        registry: ExperimentRegistry,
        runtime: ExperimentRuntime,
        artifact_store: ArtifactStore,
        outer_audit_reader: NestedOuterEvaluationAuditReader,
        worker_id: str,
        lease_token: str,
        clock: Callable[[], datetime] | None = None,
    ) -> ScientificStageImplementation:
        """Bind only Assembly-owned runtime and audit capabilities."""

        if type(run_spec) is not ResearchRunSpec:
            raise TypeError("score bound run specification differs")
        if type(scientific_lineage_manifest) is not ResearchScientificLineageManifestV2:
            raise TypeError("score bound scientific lineage differs")
        if run_spec.content_hash != self._run_spec.content_hash:
            raise ValueError("score bound run specification differs")
        if scientific_lineage_manifest.content_hash != self._lineage.content_hash:
            raise ValueError("score bound scientific lineage differs")
        if type(experiment_spec) is not ExperimentSpec:
            raise TypeError("score experiment specification differs")
        if type(registry) is not ExperimentRegistry:
            raise TypeError("score registry authority differs")
        if type(runtime) is not ExperimentRuntime:
            raise TypeError("score runtime authority differs")
        if type(artifact_store) is not ArtifactStore:
            raise TypeError("score artifact store authority differs")
        if type(outer_audit_reader) is not NestedOuterEvaluationAuditReader:
            raise TypeError("score outer audit authority differs")
        if clock is not None and not callable(clock):
            raise TypeError("score clock must be callable or None")
        _validate_worker_identity(worker_id, lease_token)
        if runtime.spec.content_hash != experiment_spec.content_hash:
            raise ValueError("score runtime experiment differs")
        if tuple(experiment_spec.stages) != tuple(
            stage.value for stage in self._run_spec.enabled_stages
        ):
            raise ValueError("score experiment stage topology differs")
        if (
            experiment_spec.scientific_lineage_manifest_hash
            != self._lineage.content_hash
        ):
            raise ValueError("score experiment scientific lineage differs")
        return _ResolvedResearchReadinessScoreImplementation(
            _token=_BOUND_SCORE_STAGE_TOKEN,
            run_spec=ResearchRunSpec.from_mapping(self._run_spec.to_dict()),
            scientific_lineage_manifest=(
                ResearchScientificLineageManifestV2.from_mapping(
                    self._lineage.to_dict()
                )
            ),
            readiness_plan=ResearchReadinessPlanV1.from_wire_bytes(
                self._plan.to_wire_bytes()
            ),
            implementation_id=self._implementation_id,
            implementation_hash=self._implementation_hash,
            experiment_spec=experiment_spec,
            registry=registry,
            runtime=runtime,
            artifact_store=artifact_store,
            outer_audit_reader=outer_audit_reader,
            worker_id=worker_id,
            lease_token=lease_token,
            clock=clock,
        )


@final
class _ResolvedResearchReadinessScoreImplementation:
    _artifact_store: ArtifactStore
    _clock: Callable[[], datetime] | None
    _experiment_spec: ExperimentSpec
    _implementation_hash: str
    _implementation_id: str
    _lease_token: str
    _lineage: ResearchScientificLineageManifestV2
    _outer_audit_reader: NestedOuterEvaluationAuditReader
    _plan: ResearchReadinessPlanV1
    _registry: ExperimentRegistry
    _run_spec: ResearchRunSpec
    _runtime: ExperimentRuntime
    _worker_id: str

    __slots__ = (
        "_artifact_store",
        "_clock",
        "_experiment_spec",
        "_implementation_hash",
        "_implementation_id",
        "_lease_token",
        "_lineage",
        "_outer_audit_reader",
        "_plan",
        "_registry",
        "_run_spec",
        "_runtime",
        "_worker_id",
    )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("resolved readiness score stage cannot be subclassed")

    def __init__(
        self,
        *,
        _token: object,
        run_spec: ResearchRunSpec,
        scientific_lineage_manifest: ResearchScientificLineageManifestV2,
        readiness_plan: ResearchReadinessPlanV1,
        implementation_id: str,
        implementation_hash: str,
        experiment_spec: ExperimentSpec,
        registry: ExperimentRegistry,
        runtime: ExperimentRuntime,
        artifact_store: ArtifactStore,
        outer_audit_reader: NestedOuterEvaluationAuditReader,
        worker_id: str,
        lease_token: str,
        clock: Callable[[], datetime] | None,
    ) -> None:
        if _token is not _BOUND_SCORE_STAGE_TOKEN:
            raise TypeError("readiness score stage requires factory authority")
        object.__setattr__(self, "_run_spec", run_spec)
        object.__setattr__(self, "_lineage", scientific_lineage_manifest)
        object.__setattr__(self, "_plan", readiness_plan)
        object.__setattr__(self, "_implementation_id", implementation_id)
        object.__setattr__(self, "_implementation_hash", implementation_hash)
        object.__setattr__(self, "_experiment_spec", experiment_spec)
        object.__setattr__(self, "_registry", registry)
        object.__setattr__(self, "_runtime", runtime)
        object.__setattr__(self, "_artifact_store", artifact_store)
        object.__setattr__(self, "_outer_audit_reader", outer_audit_reader)
        object.__setattr__(self, "_worker_id", worker_id)
        object.__setattr__(self, "_lease_token", lease_token)
        object.__setattr__(self, "_clock", clock)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("resolved readiness score stage is immutable")

    def __call__(self, request: ScientificStageRequest) -> ScientificStageResult:
        _validate_request(
            request,
            run_spec=self._run_spec,
            experiment_spec_hash=self._experiment_spec.content_hash,
        )
        started_wall = time.perf_counter()
        started_cpu = time.process_time()
        resolver = _AuthoritativeScoreInputResolver(
            registry=self._registry,
            runtime=self._runtime,
            artifact_store=self._artifact_store,
            outer_audit_reader=self._outer_audit_reader,
            worker_id=self._worker_id,
            lease_token=self._lease_token,
            clock=self._clock,
        )
        resolved = resolver.resolve(
            request=request,
            run_spec=self._run_spec,
            scientific_lineage_manifest=self._lineage,
            readiness_plan=self._plan,
        )
        # The outer-evaluation reload performed by ``resolve`` can be long.
        # Recheck the exact running SCORE lease immediately before publishing
        # any content-addressed artifacts.
        resolver.revalidate(
            experiment_spec_hash=self._experiment_spec.content_hash,
            authority=resolved.authority,
        )
        budget = self._experiment_spec.resource_budget
        memory_budget = budget.maximum_peak_memory_bytes
        producer = ResearchReadinessScoreUpstreamProducer(self._artifact_store)
        try:
            published = producer.publish(
                readiness_plan=self._plan,
                completed_outer_evaluation=resolved.completed_outer_evaluation,
                maximum_compressed_bytes=min(
                    memory_budget, budget.maximum_disk_write_bytes
                ),
                maximum_uncompressed_bytes=memory_budget,
                maximum_loaded_memory_bytes=memory_budget,
                maximum_rows=max(1, memory_budget // 8),
                maximum_columns=max(1, memory_budget // 8),
            )
            reopened = producer.reopen(
                publication_manifest_reference=(
                    published.publication_manifest_reference
                ),
                maximum_compressed_bytes=min(
                    memory_budget, budget.maximum_disk_write_bytes
                ),
                maximum_uncompressed_bytes=memory_budget,
                maximum_loaded_memory_bytes=memory_budget,
                maximum_rows=max(1, memory_budget // 8),
                maximum_columns=max(1, memory_budget // 8),
            )
        except ResearchReadinessInputError as exc:
            raise _stage_error(
                f"publication_{exc.code}",
                "controlled score publication or replay failed",
            ) from exc
        if reopened != published:
            raise _stage_error(
                "score_replay_identity_mismatch",
                "controlled score replay differs",
            )
        # If authority was lost after a CAS write, those bytes remain an
        # unreferenced orphan only: no governed stage result is returned and
        # Assembly cannot checkpoint/promote them as the authoritative output.
        resolver.revalidate(
            experiment_spec_hash=self._experiment_spec.content_hash,
            authority=resolved.authority,
        )
        return _scientific_result(
            implementation_id=self._implementation_id,
            implementation_hash=self._implementation_hash,
            published=reopened,
            parent_hashes=tuple(
                resolved.authority.parents[stage].payload_hash
                for stage in _DIRECT_SCORE_PARENTS
            ),
            started_wall=started_wall,
            started_cpu=started_cpu,
        )


class _AuthoritativeScoreInputResolver:
    """Internal SCORE authority resolver; it exposes no scientific frames."""

    def __init__(
        self,
        *,
        registry: ExperimentRegistry,
        runtime: ExperimentRuntime,
        artifact_store: ArtifactStore,
        outer_audit_reader: NestedOuterEvaluationAuditReader,
        worker_id: str,
        lease_token: str,
        clock: Callable[[], datetime] | None,
    ) -> None:
        self.registry = registry
        self.runtime = runtime
        self.artifact_store = artifact_store
        self.outer_audit_reader = outer_audit_reader
        self.worker_id = worker_id
        self.lease_token = lease_token
        self.lease_token_hash = hashlib.sha256(lease_token.encode("utf-8")).hexdigest()
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def resolve(
        self,
        *,
        request: ScientificStageRequest,
        run_spec: ResearchRunSpec,
        scientific_lineage_manifest: ResearchScientificLineageManifestV2,
        readiness_plan: ResearchReadinessPlanV1,
    ) -> _ResolvedScoreInputs:
        try:
            registered = self.registry.load_experiment_spec(
                request.experiment_spec_hash
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise _stage_error(
                "request_binding_mismatch", "registered experiment is unavailable"
            ) from exc
        if (
            registered.content_hash != request.experiment_spec_hash
            or registered.content_hash != self.runtime.spec.content_hash
        ):
            raise _stage_error(
                "request_binding_mismatch", "experiment authority differs"
            )
        shared = ResearchReadinessInputResolver(
            registry=self.registry,
            artifact_store=self.artifact_store,
            runtime=self.runtime,
            outer_audit_reader=self.outer_audit_reader,
            worker_id=self.worker_id,
            lease_token=self.lease_token,
            clock=self.clock,
        )
        authority = self._load_runtime_authority(
            registered=registered,
            run_spec=run_spec,
            request=request,
            shared=shared,
        )
        try:
            model_parent = authority.parents[ResearchStage.MODEL_TRAINING]
            if type(model_parent) is not _VerifiedParent:
                raise ResearchReadinessInputError(
                    "model_parent_mismatch", "model authority type differs"
                )
            model = shared._load_model_source(
                parent=model_parent,
                experiment_spec=registered,
                request=request,
                run_spec=run_spec,
                scientific_lineage_manifest=scientific_lineage_manifest,
                readiness_plan=readiness_plan,
            )
        except ResearchReadinessInputError as exc:
            code = (
                "outer_completion_unavailable"
                if exc.code == "outer_audit_mismatch"
                else f"input_{exc.code}"
            )
            raise _stage_error(code, "model or outer evidence is unavailable") from exc
        self.revalidate(
            experiment_spec_hash=registered.content_hash,
            authority=authority,
        )
        return _ResolvedScoreInputs(
            authority=authority,
            completed_outer_evaluation=model.completed,
        )

    def _load_runtime_authority(
        self,
        *,
        registered: ExperimentSpec,
        run_spec: ResearchRunSpec,
        request: ScientificStageRequest,
        shared: ResearchReadinessInputResolver,
    ) -> _ScoreRuntimeAuthority:
        if tuple(registered.stages) != tuple(
            stage.value for stage in run_spec.enabled_stages
        ):
            raise _stage_error(
                "runtime_authority_mismatch", "stage topology differs"
            )
        score_name = ResearchStage.SCORE_CONSTRUCTION.value
        try:
            score_index = registered.stages.index(score_name)
        except ValueError as exc:
            raise _stage_error(
                "runtime_authority_mismatch", "score stage is unavailable"
            ) from exc
        prior_results: dict[str, str] = {}
        parents: dict[ResearchStage, _ScoreVerifiedParent | _VerifiedParent] = {}
        for stage_name in registered.stages[:score_index]:
            stage = ResearchStage(stage_name)
            attempt = self.runtime.successful_attempt(stage_name)
            if (
                attempt is None
                or attempt.status is not AttemptStatus.SUCCEEDED
                or attempt.experiment_spec_hash != registered.content_hash
                or attempt.stage != stage_name
                or attempt.result_hash is None
            ):
                raise _stage_error(
                    "runtime_authority_mismatch",
                    f"{stage_name} has no authoritative success",
                )
            expected_input = stage_input_hash(
                experiment_spec_hash=registered.content_hash,
                stage=stage_name,
                prior_result_hashes=prior_results,
            )
            if attempt.input_hash != expected_input:
                raise _stage_error(
                    "runtime_authority_mismatch", f"{stage_name} input differs"
                )
            if stage in _DIRECT_SCORE_PARENTS:
                try:
                    expected_parent_hashes = {
                        ResearchStage(parent): prior_results[
                            ResearchStage(parent).value
                        ]
                        for parent in StageContract.for_run(
                            run_spec, stage
                        ).required_parent_stages
                    }
                    if stage is ResearchStage.MODEL_TRAINING:
                        parents[stage] = shared._load_parent(
                            experiment_spec=registered,
                            run_spec=run_spec,
                            stage=stage,
                            artifact_hash=attempt.result_hash,
                            expected_attempt_id=attempt.attempt_id,
                            expected_attempt_number=attempt.attempt_number,
                            expected_input_hash=expected_input,
                            expected_parent_hashes=expected_parent_hashes,
                        )
                    else:
                        parents[stage] = self._load_factor_evaluation_parent(
                            experiment_spec=registered,
                            run_spec=run_spec,
                            artifact_hash=attempt.result_hash,
                            expected_attempt_id=attempt.attempt_id,
                            expected_input_hash=expected_input,
                            expected_parent_hashes=expected_parent_hashes,
                        )
                except ResearchReadinessInputError as exc:
                    raise _stage_error(
                        f"input_{exc.code}", "score parent artifact differs"
                    ) from exc
            prior_results[stage_name] = attempt.result_hash
        if tuple(stage for stage in _DIRECT_SCORE_PARENTS if stage in parents) != (
            _DIRECT_SCORE_PARENTS
        ):
            raise _stage_error(
                "runtime_authority_mismatch", "score direct parents differ"
            )
        requested = {
            ResearchStage(stage): digest
            for stage, digest in request.parent_artifact_hashes.items()
        }
        if requested != {
            stage: parents[stage].payload_hash for stage in _DIRECT_SCORE_PARENTS
        }:
            raise _stage_error(
                "runtime_authority_mismatch", "score request parents differ"
            )
        authority = _ScoreRuntimeAuthority(
            parents=MappingProxyType(parents),
            current_attempt_id=request.attempt_id,
            current_attempt_number=request.attempt_number,
            current_attempt_input_hash=stage_input_hash(
                experiment_spec_hash=registered.content_hash,
                stage=score_name,
                prior_result_hashes=prior_results,
            ),
            current_worker_id=self.worker_id,
            current_lease_token_hash=self.lease_token_hash,
        )
        self.revalidate(
            experiment_spec_hash=registered.content_hash,
            authority=authority,
        )
        return authority

    def _load_factor_evaluation_parent(
        self,
        *,
        experiment_spec: ExperimentSpec,
        run_spec: ResearchRunSpec,
        artifact_hash: str,
        expected_attempt_id: int,
        expected_input_hash: str,
        expected_parent_hashes: Mapping[ResearchStage, str],
    ) -> _ScoreVerifiedParent:
        """Reload the other SCORE parent without widening readiness types.

        ``ResearchReadinessParentAttemptBindingV1`` deliberately admits only
        ROBUSTNESS parents.  SCORE also has a FACTOR_EVALUATION parent, so this
        method performs the same immutable registry/checkpoint/CAS/envelope
        verification but returns a narrow SCORE-local evidence record.
        """

        stage = ResearchStage.FACTOR_EVALUATION
        try:
            descriptor = dict(
                self.registry.load_artifact_descriptor(
                    experiment_spec.content_hash, artifact_hash
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ResearchReadinessInputError(
                "parent_artifact_mismatch", "factor evaluation is unavailable"
            ) from exc
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
        if set(descriptor) != expected_fields or (
            descriptor.get("artifact_hash") != artifact_hash
            or descriptor.get("stage") != stage.value
            or descriptor.get("kind") != "governed_research_stage"
            or descriptor.get("media_type") != "application/json"
            or descriptor.get("attempt_id") != expected_attempt_id
            or descriptor.get("input_hash") != expected_input_hash
        ):
            raise ResearchReadinessInputError(
                "parent_artifact_mismatch", "factor evaluation descriptor differs"
            )
        checkpoints = self.runtime.list_checkpoints(expected_attempt_id)
        if (
            len(checkpoints) != 1
            or checkpoints[0].checkpoint_name != "stage_result"
            or checkpoints[0].artifact_hash != artifact_hash
            or checkpoints[0].location != descriptor["location"]
        ):
            raise ResearchReadinessInputError(
                "runtime_authority_mismatch", "factor evaluation checkpoint differs"
            )
        try:
            record = ArtifactRecord(
                logical_name=cast(str, descriptor["logical_name"]),
                location=cast(str, descriptor["location"]),
                sha256=artifact_hash,
                size_bytes=cast(int, descriptor["size_bytes"]),
                media_type="application/json",
                role="experiment_stage_result",
            )
            payload = self.artifact_store.read_bytes(record)
            contract = StageContract.for_run(run_spec, stage)
            typed_descriptor = StageArtifactDescriptor.from_payload(contract, payload)
            body = _governed_body(payload)
            scientific = ScientificStageResult.from_mapping(
                cast(Mapping[str, object], body["scientific_result"])
            )
        except (
            ArtifactError,
            OSError,
            ResearchReadinessInputError,
            TypeError,
            ValueError,
        ) as exc:
            raise ResearchReadinessInputError(
                "parent_artifact_mismatch", "factor evaluation payload differs"
            ) from exc
        if (
            typed_descriptor.artifact_hash != artifact_hash
            or dict(typed_descriptor.parent_artifact_hashes)
            != dict(expected_parent_hashes)
            or scientific.stage is not stage
        ):
            raise ResearchReadinessInputError(
                "parent_artifact_mismatch", "factor evaluation lineage differs"
            )
        memory = body["memory_record"]
        if (
            not isinstance(memory, Mapping)
            or memory.get("result_hash") != scientific.content_hash
            or body["memory_record_hash"] != hash_json(dict(memory))
        ):
            raise ResearchReadinessInputError(
                "parent_artifact_mismatch", "factor evaluation memory differs"
            )
        return _ScoreVerifiedParent(
            payload_hash=artifact_hash,
            descriptor_hash=cast(str, hash_json(descriptor)),
            result=scientific,
        )

    def revalidate(
        self,
        *,
        experiment_spec_hash: str,
        authority: _ScoreRuntimeAuthority,
    ) -> None:
        """Recheck the running SCORE lease and kill switch after every long read."""

        if (
            authority.current_worker_id != self.worker_id
            or authority.current_lease_token_hash != self.lease_token_hash
        ):
            raise _stage_error(
                "runtime_authority_mismatch",
                "current score lease owner, token, state or kill switch differs",
            )
        try:
            row = self.runtime.connection.execute(
                """SELECT
                       a.attempt_id,a.experiment_spec_hash,a.stage,a.attempt_number,
                       a.status,a.worker_id,a.lease_token_hash,a.input_hash,
                       a.started_at,a.lease_expires_at,a.finished_at,a.result_hash,
                       a.failure_code,c.state AS runtime_control_state
                   FROM stage_attempts AS a
                   CROSS JOIN runtime_control AS c
                   WHERE a.attempt_id=? AND c.singleton=1""",
                (authority.current_attempt_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise _stage_error(
                "runtime_authority_mismatch", "score authority reload failed"
            ) from exc
        expected = {
            "attempt_id": authority.current_attempt_id,
            "experiment_spec_hash": experiment_spec_hash,
            "stage": ResearchStage.SCORE_CONSTRUCTION.value,
            "attempt_number": authority.current_attempt_number,
            "status": AttemptStatus.RUNNING.value,
            "worker_id": authority.current_worker_id,
            "lease_token_hash": authority.current_lease_token_hash,
            "input_hash": authority.current_attempt_input_hash,
            "runtime_control_state": "running",
        }
        if row is None or any(row[name] != value for name, value in expected.items()):
            raise _stage_error(
                "runtime_authority_mismatch",
                "current score lease owner, token, state or kill switch differs",
            )
        if (
            row["started_at"] is None
            or row["finished_at"] is not None
            or row["result_hash"] is not None
            or row["failure_code"] is not None
        ):
            raise _stage_error(
                "runtime_authority_mismatch", "current score attempt state differs"
            )
        try:
            lease = datetime.fromisoformat(
                str(row["lease_expires_at"]).replace("Z", "+00:00")
            )
        except (TypeError, ValueError) as exc:
            raise _stage_error(
                "runtime_authority_mismatch", "score lease timestamp differs"
            ) from exc
        now = self.clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise _stage_error("runtime_authority_mismatch", "score clock differs")
        if lease.tzinfo is None or lease.astimezone(timezone.utc) <= now.astimezone(
            timezone.utc
        ):
            raise _stage_error(
                "runtime_authority_mismatch", "current score lease is expired"
            )


def _validate_scientific_bindings(
    *,
    run_spec: ResearchRunSpec,
    scientific_lineage_manifest: ResearchScientificLineageManifestV2,
    readiness_plan: ResearchReadinessPlanV1,
    recipes: ResearchReadinessScoreRecipeSetV2,
) -> None:
    scientific_lineage_manifest.validate_for(run_spec)
    required = {
        ResearchStage.MODEL_TRAINING,
        ResearchStage.FACTOR_EVALUATION,
        ResearchStage.SCORE_CONSTRUCTION,
    }
    if not required.issubset(run_spec.enabled_stages):
        raise ValueError("controlled readiness score stages are not enabled")
    if run_spec.score_spec_hash != recipes.content_hash:
        raise ValueError("score specification differs from the closed recipe set")
    if readiness_plan.nested_selection_spec_hash != recipes.nested_selection_spec_hash:
        raise ValueError("score recipe nested-selection identity differs")
    if (
        readiness_plan.nested_selection_spec_hash
        != scientific_lineage_manifest.nested_selection_spec_hash
    ):
        raise ValueError("score plan nested-selection lineage differs")
    if readiness_plan.validation_spec_hash != run_spec.validation_spec_hash:
        raise ValueError("score plan validation specification differs")
    if readiness_plan.readiness_spec_hash != run_spec.robustness_spec_hash:
        raise ValueError("score plan readiness specification differs")
    try:
        validation_partition_hash = run_spec.data_partition_hashes[
            DataPartition.VALIDATION
        ]
        validation_partition_window = run_spec.data_partition_windows[
            DataPartition.VALIDATION
        ]
    except KeyError as exc:
        raise ValueError("score validation partition is unavailable") from exc
    if (
        readiness_plan.validation_partition_hash
        != validation_partition_hash
        or (
            readiness_plan.validation_window_start,
            readiness_plan.validation_window_end,
        )
        != validation_partition_window
    ):
        raise ValueError("score plan validation partition differs")


def _validate_preflight_authorities(
    *,
    experiment_spec: ExperimentSpec,
    registry: ExperimentRegistry,
    runtime: ExperimentRuntime,
    artifact_store: ArtifactStore,
    outer_audit_reader: NestedOuterEvaluationAuditReader,
    run_spec: ResearchRunSpec,
    scientific_lineage_manifest: ResearchScientificLineageManifestV2,
) -> None:
    if type(experiment_spec) is not ExperimentSpec:
        raise TypeError("score preflight experiment specification differs")
    if type(registry) is not ExperimentRegistry:
        raise TypeError("score preflight registry authority differs")
    if type(runtime) is not ExperimentRuntime:
        raise TypeError("score preflight runtime authority differs")
    if type(artifact_store) is not ArtifactStore:
        raise TypeError("score preflight artifact store authority differs")
    if type(outer_audit_reader) is not NestedOuterEvaluationAuditReader:
        raise TypeError("score preflight outer audit authority differs")
    if runtime.spec.content_hash != experiment_spec.content_hash:
        raise ValueError("score preflight runtime experiment differs")
    if tuple(experiment_spec.stages) != tuple(
        stage.value for stage in run_spec.enabled_stages
    ):
        raise ValueError("score preflight stage topology differs")
    if (
        experiment_spec.scientific_lineage_manifest_hash
        != scientific_lineage_manifest.content_hash
    ):
        raise ValueError("score preflight scientific lineage differs")


def _preflight_outer_completion(
    *,
    experiment_spec: ExperimentSpec,
    registry: ExperimentRegistry,
    runtime: ExperimentRuntime,
    artifact_store: ArtifactStore,
    outer_audit_reader: NestedOuterEvaluationAuditReader,
    run_spec: ResearchRunSpec,
    scientific_lineage_manifest: ResearchScientificLineageManifestV2,
    readiness_plan: ResearchReadinessPlanV1,
) -> ResearchReadinessScorePreflightEvidence:
    try:
        registered = registry.load_experiment_spec(experiment_spec.content_hash)
    except (KeyError, TypeError, ValueError) as exc:
        raise _stage_error(
            "preflight_authority_mismatch", "registered experiment is unavailable"
        ) from exc
    if (
        registered.content_hash != experiment_spec.content_hash
        or registered.content_hash != runtime.spec.content_hash
    ):
        raise _stage_error(
            "preflight_authority_mismatch", "experiment authority differs"
        )
    shared = ResearchReadinessInputResolver(
        registry=registry,
        artifact_store=artifact_store,
        runtime=runtime,
        outer_audit_reader=outer_audit_reader,
        # These identities are never used to reserve, renew, or revalidate an
        # attempt.  They satisfy the reused read-only resolver constructor.
        worker_id="readiness-score-preflight",
        lease_token="readiness-score-preflight-readonly",
    )
    try:
        model_index = registered.stages.index(ResearchStage.MODEL_TRAINING.value)
    except ValueError as exc:
        raise _stage_error(
            "preflight_authority_mismatch", "model stage is unavailable"
        ) from exc
    prior_results: dict[str, str] = {}
    model_parent: _VerifiedParent | None = None
    model_attempt_id = 0
    model_attempt_number = 0
    model_parent_hashes: dict[ResearchStage, str] = {}
    for stage_name in registered.stages[: model_index + 1]:
        stage = ResearchStage(stage_name)
        attempt = runtime.successful_attempt(stage_name)
        if (
            attempt is None
            or attempt.status is not AttemptStatus.SUCCEEDED
            or attempt.experiment_spec_hash != registered.content_hash
            or attempt.stage != stage_name
            or attempt.result_hash is None
        ):
            raise _stage_error(
                "preflight_authority_mismatch",
                f"{stage_name} has no authoritative success",
            )
        expected_input = stage_input_hash(
            experiment_spec_hash=registered.content_hash,
            stage=stage_name,
            prior_result_hashes=prior_results,
        )
        if attempt.input_hash != expected_input:
            raise _stage_error(
                "preflight_authority_mismatch", f"{stage_name} input differs"
            )
        if stage is ResearchStage.MODEL_TRAINING:
            model_parent_hashes = {
                ResearchStage(parent): prior_results[ResearchStage(parent).value]
                for parent in StageContract.for_run(
                    run_spec, ResearchStage.MODEL_TRAINING
                ).required_parent_stages
            }
            try:
                model_parent = shared._load_parent(
                    experiment_spec=registered,
                    run_spec=run_spec,
                    stage=ResearchStage.MODEL_TRAINING,
                    artifact_hash=attempt.result_hash,
                    expected_attempt_id=attempt.attempt_id,
                    expected_attempt_number=attempt.attempt_number,
                    expected_input_hash=expected_input,
                    expected_parent_hashes=model_parent_hashes,
                )
            except ResearchReadinessInputError as exc:
                raise _stage_error(
                    f"preflight_{exc.code}", "model parent is unavailable"
                ) from exc
            model_attempt_id = attempt.attempt_id
            model_attempt_number = attempt.attempt_number
        prior_results[stage_name] = attempt.result_hash
    if model_parent is None:
        raise _stage_error(
            "preflight_authority_mismatch", "model parent is unavailable"
        )
    historical_request = ScientificStageRequest(
        research_run_spec_hash=run_spec.content_hash,
        experiment_spec_hash=registered.content_hash,
        stage=ResearchStage.MODEL_TRAINING,
        attempt_id=model_attempt_id,
        attempt_number=model_attempt_number,
        contract=StageContract.for_run(run_spec, ResearchStage.MODEL_TRAINING),
        parent_artifact_hashes=cast(
            Mapping[ResearchStage | str, str], model_parent_hashes
        ),
        agent_decision=AgentStageDecision(
            status="succeeded",
            decision_code="preflight_existing_outer_completion",
            directive={
                "nested_selection_spec_hash": (
                    readiness_plan.nested_selection_spec_hash
                )
            },
        ),
    )
    try:
        source = shared._load_model_source(
            parent=model_parent,
            experiment_spec=registered,
            request=historical_request,
            run_spec=run_spec,
            scientific_lineage_manifest=scientific_lineage_manifest,
            readiness_plan=readiness_plan,
        )
    except ResearchReadinessInputError as exc:
        code = (
            "preflight_outer_completion_unavailable"
            if exc.code == "outer_audit_mismatch"
            else f"preflight_{exc.code}"
        )
        raise _stage_error(code, "model or outer evidence is unavailable") from exc
    completed = source.completed
    return ResearchReadinessScorePreflightEvidence(
        model_parent_hash=model_parent.payload_hash,
        validation_partition_hash=readiness_plan.validation_partition_hash,
        outer_validation_receipt_hash=(
            readiness_plan.outer_validation_receipt_hash
        ),
        phase_one_manifest_hash=completed.completion.manifest_hash,
        outer_evaluation_completion_hash=completed.completion.content_hash,
        outer_evaluation_result_hash=completed.result.content_hash,
        outer_execution_snapshot_hash=completed.execution_snapshot.content_hash,
    )


def _validate_request(
    request: ScientificStageRequest,
    *,
    run_spec: ResearchRunSpec,
    experiment_spec_hash: str,
) -> None:
    if type(request) is not ScientificStageRequest:
        raise _stage_error("request_binding_mismatch", "score request type differs")
    if ResearchStage(request.stage) is not ResearchStage.SCORE_CONSTRUCTION:
        raise _stage_error("request_binding_mismatch", "score request stage differs")
    if request.research_run_spec_hash != run_spec.content_hash:
        raise _stage_error("request_binding_mismatch", "score request run differs")
    if request.experiment_spec_hash != experiment_spec_hash:
        raise _stage_error("request_binding_mismatch", "score request experiment differs")
    expected_contract = StageContract.for_run(
        run_spec, ResearchStage.SCORE_CONSTRUCTION
    )
    if (
        request.contract.content_hash != expected_contract.content_hash
        or tuple(request.contract.required_parent_stages) != _DIRECT_SCORE_PARENTS
    ):
        raise _stage_error("request_binding_mismatch", "score contract differs")


def _validate_implementation_identity(
    implementation_id: str, implementation_hash: str
) -> None:
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
        raise ValueError("readiness score implementation_id is invalid")
    if implementation_id.lower() in {"noop", "placeholder", "stub", "mock"}:
        raise ValueError("placeholder readiness score implementation is forbidden")
    require_sha256(implementation_hash, name="readiness score implementation_hash")


def _validate_worker_identity(worker_id: object, lease_token: object) -> None:
    if not isinstance(worker_id, str) or not 1 <= len(worker_id) <= 128:
        raise TypeError("readiness score worker_id is invalid")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-")
    if not worker_id[0].isalnum() or any(character not in allowed for character in worker_id):
        raise ValueError("readiness score worker_id is invalid")
    if not isinstance(lease_token, str) or len(lease_token) < 16:
        raise ValueError("readiness score lease_token must contain at least 16 characters")


def _scientific_result(
    *,
    implementation_id: str,
    implementation_hash: str,
    published: ResearchReadinessScoreConstructionResultV2,
    parent_hashes: tuple[str, ...],
    started_wall: float,
    started_cpu: float,
) -> ScientificStageResult:
    manifest = published.publication_manifest
    score_references = (
        manifest.score_artifacts.candidate_scores,
        manifest.score_artifacts.baseline_scores,
        *manifest.score_artifacts.perturbation_scores.values(),
    )
    records = (
        published.publication_manifest_reference,
        *manifest.definition_artifacts.values(),
        *manifest.execution_receipt_artifacts.values(),
    )
    evidence = tuple(
        dict.fromkeys(
            (
                *parent_hashes,
                *published.evidence_hashes,
                published.publication_manifest_reference.sha256,
                *(reference.content_hash for reference in score_references),
                *(reference.frame_hash for reference in score_references),
                *(reference.payload_sha256 for reference in score_references),
                *(record.sha256 for record in records),
                *(
                    digest
                    for digest in manifest.outer_evaluation_binding.values()
                    if len(digest) == 64
                ),
            )
        )
    )
    payload = cast(Mapping[str, JsonValue], dict(published.result_payload))
    disk_bytes = sum(record.size_bytes for record in records) + sum(
        reference.size_bytes for reference in score_references
    )
    return ScientificStageResult(
        stage=ResearchStage.SCORE_CONSTRUCTION,
        implementation_id=implementation_id,
        implementation_hash=implementation_hash,
        result_payload=payload,
        evidence_hashes=evidence,
        sanitized_feedback_codes=("controlled_score_recipes_v2", "research_only"),
        usage=reported_usage(
            wall_seconds=max(0.0, time.perf_counter() - started_wall),
            cpu_seconds=max(0.0, time.process_time() - started_cpu),
            peak_memory_bytes=_peak_rss_bytes(),
            disk_write_bytes=disk_bytes,
        ),
        input_rows=manifest.score_artifacts.candidate_scores.row_count,
        output_rows=manifest.score_artifacts.candidate_scores.row_count,
        symbols=manifest.score_artifacts.candidate_scores.column_count,
        disk_read_bytes=disk_bytes,
        cache_hits=0,
        cache_misses=0,
        worker_count=1,
    )


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


__all__ = [
    "ResearchReadinessScorePreflightEvidence",
    "ResearchReadinessScoreStageError",
    "ResolvedResearchReadinessScoreFactory",
]
