"""Trusted resolver for governed research-readiness inputs.

The resolver accepts no protected partition and never mutates runtime state.
It reconstructs direct-parent authority from the registry/runtime/checkpoint
chain, consumes an already completed outer evaluation through the read-only
audit facet, and verifies every parquet payload before emitting an authority
binding.  Validation-only MODEL slices are new content-addressed derived
evidence; the immutable full-source identities remain bound by the MODEL parent
and sealed outer result.  The returned binding is research evidence, not
production approval.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import sqlite3
from types import MappingProxyType
from typing import Callable, Final, Mapping, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alpha_research.core.data import DataRole
from alpha_research.core.hashing import hash_json
from alpha_research.experiments.controller import stage_input_hash
from alpha_research.experiments.registry import ExperimentRegistry
from alpha_research.experiments.spec import DataPartition, ExperimentSpec
from alpha_research.models.nested_outer_evaluation_store import (
    CompletedOuterEvaluation,
    MAXIMUM_OUTER_EVALUATION_EXECUTION_SNAPSHOT_BYTES,
    MAXIMUM_OUTER_EVALUATION_RESULT_BYTES,
)
from alpha_research.models.nested_selection_manifest import (
    NestedSelectionPhaseOneManifestReference,
)
from alpha_research.orchestration import AttemptStatus, ExperimentRuntime
from alpha_research.research.artifacts import StageArtifactDescriptor, StageContract
from alpha_research.research.execution import (
    ScientificStageRequest,
    ScientificStageResult,
)
from alpha_research.research.lineage import ResearchScientificLineageManifestV2
from alpha_research.research.model_training_input_resolver import (
    ModelTrainingInputAuthorityReceiptV2,
)
from alpha_research.research.model_training_inputs import ModelTrainingInputManifestV2
from alpha_research.research.model_training_stage import (
    ModelTrainingInputDocumentReferenceV1,
)
from alpha_research.research.nested_selection_control import (
    NestedOuterEvaluationAuditReader,
)
from alpha_research.research.readiness_inputs import (
    AuthorityBoundResearchReadinessInputs,
    LoadedResearchReadinessInputs,
    ResearchReadinessAuthorityReceiptV1,
    ResearchReadinessFrameReferenceV1,
    ResearchReadinessInputError,
    ResearchReadinessInputManifestV1,
    ResearchReadinessParentAttemptBindingV1,
    ResearchReadinessPlanV1,
    ResearchReadinessScoreArtifactsV1,
    _authority_handoff_hash,
    _new_research_readiness_resolver_capability,
)
from alpha_research.research.readiness_upstream_bundles import (
    ResearchReadinessScoreUpstreamProducer,
)
from alpha_research.research.spec import ResearchRunSpec, ResearchStage
from factor_production.v5.artifacts.manifest import (
    ArtifactError as ContentArtifactError,
)
from factor_production.v5.artifacts.manifest import (
    ArtifactRecord as ContentArtifactRecord,
)
from factor_production.v5.artifacts.manifest import ArtifactStore


_DIRECT_PARENTS: Final = (
    ResearchStage.MODEL_TRAINING,
    ResearchStage.SCORE_CONSTRUCTION,
    ResearchStage.BACKTEST,
)
# BACKTEST remains a direct DAG dependency so the resolver still verifies its
# successful attempt, checkpoint, StageContract, input hash, and CAS payload.
# Its caller-authored scientific payload is deliberately not evidence for the
# ML-F decision; only MODEL_TRAINING and SCORE_CONSTRUCTION supply scientific
# inputs to the readiness runner.
_GOVERNED_BODY_FIELDS: Final = frozenset(
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
_MODEL_RESULT_FIELDS: Final = frozenset(
    {
        "phase_one_manifest_reference",
        "verified_phase_one_manifest_hash",
        "verified_phase_one_seal_hash",
        "nested_selection_authority_hash",
        "model_training_input_manifest_hash",
        "model_training_input_manifest_reference",
        "model_training_input_authority_receipt_hash",
        "model_training_input_authority_receipt_reference",
        "model_training_parent_attempt_bindings",
        "research_only",
        "production_ready",
    }
)
_SCORE_RESULT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "research_readiness_score_artifacts",
        "research_readiness_score_artifacts_hash",
        "score_recipe_publication_manifest_reference",
        "score_recipe_publication_manifest_hash",
        "outer_evaluation_binding",
        "research_only",
        "production_ready",
    }
)
_OUTER_BINDING_FIELDS: Final = frozenset(
    {
        "schema_version",
        "candidate_score_representation",
        "validation_partition_hash",
        "outer_validation_receipt_hash",
        "phase_one_manifest_hash",
        "outer_evaluation_completion_hash",
        "outer_evaluation_result_hash",
        "outer_execution_snapshot_hash",
    }
)
_PARQUET_WORKING_SET_MULTIPLIER: Final = 6
_SCORE_RESULT_SCHEMA: Final = "research-readiness-score-construction-result/v2"
_SCORE_PUBLICATION_REFERENCE_FIELDS: Final = frozenset(
    {
        "logical_name",
        "location",
        "sha256",
        "size_bytes",
        "media_type",
        "role",
    }
)
_NUMERIC_SCORE_BYTES: Final = 8
_AUTHORITATIVE_FRAME_REDERIVATION_PASSES: Final = 2


def _error(code: str, detail: str) -> ResearchReadinessInputError:
    return ResearchReadinessInputError(code, detail)


def _authoritative_frame_rederivation_read_bytes(
    *,
    manifest: ModelTrainingInputManifestV2,
    scores: ResearchReadinessScoreArtifactsV1,
) -> int:
    """Return governed encoded bytes for the two authority re-derivations.

    Assembly and resolution each independently open the outer candidate score
    plus the three full MODEL target/mask frames.  This content-addressed
    scientific-artifact metric intentionally excludes registry, SQLite, and
    other small metadata I/O, so it must not be described as total OS I/O.
    """

    if type(manifest) is not ModelTrainingInputManifestV2:
        raise _error("invalid_manifest", "model source manifest type differs")
    if type(scores) is not ResearchReadinessScoreArtifactsV1:
        raise _error("score_parent_mismatch", "score artifact type differs")
    one_pass = sum(
        cast(int, reference.size_bytes)
        for reference in (
            scores.candidate_scores,
            manifest.label_values,
            manifest.label_validity,
            manifest.scoring_eligibility,
        )
    )
    return _AUTHORITATIVE_FRAME_REDERIVATION_PASSES * one_pass


@dataclass(frozen=True, slots=True)
class _VerifiedParent:
    payload_hash: str
    descriptor_hash: str
    attempt_binding: ResearchReadinessParentAttemptBindingV1
    result: ScientificStageResult


@dataclass(frozen=True, slots=True)
class _RuntimeAuthority:
    parents: Mapping[ResearchStage, _VerifiedParent]
    current_attempt_id: int
    current_attempt_number: int
    current_attempt_input_hash: str
    current_worker_id: str
    current_lease_token_hash: str


@dataclass(frozen=True, slots=True)
class _ModelSource:
    manifest: ModelTrainingInputManifestV2
    phase_one_reference: NestedSelectionPhaseOneManifestReference
    completed: CompletedOuterEvaluation


@dataclass(frozen=True, slots=True)
class _ScoreReopenBudget:
    maximum_compressed_bytes: int
    maximum_uncompressed_bytes: int
    maximum_loaded_memory_bytes: int
    maximum_rows: int
    maximum_columns: int


@dataclass(frozen=True, slots=True)
class _PreparedOuterModelSource:
    """Validated derived evidence prepared before any CAS write."""

    role: str
    logical_name: str
    frame: pd.DataFrame
    payload: bytes
    parquet_uncompressed_bytes: int


class ResearchReadinessInputResolver:
    """Reconstruct and verify one validation-only readiness input bundle."""

    def __init__(
        self,
        *,
        registry: ExperimentRegistry,
        artifact_store: ArtifactStore,
        runtime: ExperimentRuntime,
        outer_audit_reader: NestedOuterEvaluationAuditReader,
        worker_id: str,
        lease_token: str,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if type(registry) is not ExperimentRegistry:
            raise TypeError("readiness resolver requires an exact ExperimentRegistry")
        if type(artifact_store) is not ArtifactStore:
            raise TypeError("readiness resolver requires an exact ArtifactStore")
        if type(runtime) is not ExperimentRuntime:
            raise TypeError("readiness resolver requires an exact ExperimentRuntime")
        if type(outer_audit_reader) is not NestedOuterEvaluationAuditReader:
            raise TypeError("readiness resolver requires an exact outer audit reader")
        self._worker_id = _runtime_code(worker_id, name="worker_id")
        self._lease_token_hash = _lease_token_hash(lease_token)
        if clock is not None and not callable(clock):
            raise TypeError("readiness resolver clock must be callable or None")
        self.registry = registry
        self.artifact_store = artifact_store
        self.runtime = runtime
        self.outer_audit_reader = outer_audit_reader
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.__resolver_capability = _new_research_readiness_resolver_capability()

    def assemble_manifest(
        self,
        *,
        request: ScientificStageRequest,
        run_spec: ResearchRunSpec,
        scientific_lineage_manifest: ResearchScientificLineageManifestV2,
        readiness_plan: ResearchReadinessPlanV1,
    ) -> ResearchReadinessInputManifestV1:
        """Build the manifest only from registered parents and audited evidence."""

        registered = self._validate_context(
            request=request,
            run_spec=run_spec,
            scientific_lineage_manifest=scientific_lineage_manifest,
            readiness_plan=readiness_plan,
        )
        authority = self._load_runtime_authority(
            experiment_spec=registered,
            run_spec=run_spec,
            request=request,
        )
        model = self._load_model_source(
            parent=authority.parents[ResearchStage.MODEL_TRAINING],
            experiment_spec=registered,
            request=request,
            run_spec=run_spec,
            scientific_lineage_manifest=scientific_lineage_manifest,
            readiness_plan=readiness_plan,
        )
        scores = self._load_score_bundle(
            authority.parents[ResearchStage.SCORE_CONSTRUCTION],
            experiment_spec=registered,
            readiness_plan=readiness_plan,
            completed=model.completed,
        )
        prepared_sources = self._prepare_outer_model_source_evidence(
            manifest=model.manifest,
            scores=scores,
            completed=model.completed,
            maximum_peak_memory_bytes=(
                registered.resource_budget.maximum_peak_memory_bytes
            ),
            maximum_disk_write_bytes=(
                registered.resource_budget.maximum_disk_write_bytes
            ),
        )
        # Selection and serialization may be non-trivial on a large validation
        # panel.  Re-establish the exact running authority before publishing
        # any derived CAS bytes.
        self._revalidate_current_authority(
            experiment_spec_hash=registered.content_hash,
            authority=authority,
        )
        label_values, label_validity, eligibility = (
            self._publish_outer_model_source_evidence(prepared_sources)
        )
        # A lease loss after CAS publication leaves only unreferenced content;
        # no manifest or governed result may be released under stale authority.
        self._revalidate_current_authority(
            experiment_spec_hash=registered.content_hash,
            authority=authority,
        )
        completion = model.completed.completion
        return ResearchReadinessInputManifestV1(
            research_run_spec_hash=run_spec.content_hash,
            experiment_spec_hash=registered.content_hash,
            scientific_lineage_manifest_hash=scientific_lineage_manifest.content_hash,
            robustness_contract_hash=request.contract.content_hash,
            readiness_plan_hash=readiness_plan.content_hash,
            readiness_plan=readiness_plan,
            parent_artifact_hashes={
                stage: authority.parents[stage].payload_hash
                for stage in _DIRECT_PARENTS
            },
            phase_one_manifest_hash=completion.manifest_hash,
            outer_evaluation_completion_hash=completion.content_hash,
            outer_evaluation_completion=completion,
            outer_evaluation_result_hash=completion.result_hash,
            outer_execution_snapshot_hash=completion.execution_snapshot_hash,
            score_artifacts_hash=scores.content_hash,
            score_artifacts=scores,
            label_spec_hash=model.manifest.label_spec_hash,
            outer_validation_receipt_hash=(
                model.manifest.validation_receipt_hash
            ),
            label_values_hash=label_values.frame_hash,
            label_values=label_values,
            label_validity_hash=label_validity.frame_hash,
            label_validity=label_validity,
            scoring_eligibility_hash=eligibility.frame_hash,
            scoring_eligibility=eligibility,
        )

    def resolve(
        self,
        *,
        manifest: ResearchReadinessInputManifestV1,
        request: ScientificStageRequest,
        run_spec: ResearchRunSpec,
        scientific_lineage_manifest: ResearchScientificLineageManifestV2,
    ) -> AuthorityBoundResearchReadinessInputs:
        """Reload every authority and frame before creating a process-local seal."""

        if not isinstance(manifest, ResearchReadinessInputManifestV1):
            raise _error("invalid_manifest", "readiness manifest type differs")
        registered = self._validate_context(
            request=request,
            run_spec=run_spec,
            scientific_lineage_manifest=scientific_lineage_manifest,
            readiness_plan=manifest.readiness_plan,
        )
        self._validate_manifest_context(
            manifest=manifest,
            request=request,
            run_spec=run_spec,
            experiment_spec=registered,
            scientific_lineage_manifest=scientific_lineage_manifest,
        )
        authority = self._load_runtime_authority(
            experiment_spec=registered,
            run_spec=run_spec,
            request=request,
        )
        model = self._load_model_source(
            parent=authority.parents[ResearchStage.MODEL_TRAINING],
            experiment_spec=registered,
            request=request,
            run_spec=run_spec,
            scientific_lineage_manifest=scientific_lineage_manifest,
            readiness_plan=manifest.readiness_plan,
        )
        scores = self._load_score_bundle(
            authority.parents[ResearchStage.SCORE_CONSTRUCTION],
            experiment_spec=registered,
            readiness_plan=manifest.readiness_plan,
            completed=model.completed,
        )
        self._validate_parent_derived_manifest(
            manifest=manifest,
            authority=authority,
            model=model,
            scores=scores,
        )
        loaded, resource_totals = self._load_and_verify_all_frames(
            manifest,
            maximum_peak_memory_bytes=(
                registered.resource_budget.maximum_peak_memory_bytes
            ),
        )
        remaining_memory = _remaining_memory_budget(
            maximum_peak_memory_bytes=(
                registered.resource_budget.maximum_peak_memory_bytes
            ),
            retained_loaded_memory_bytes=resource_totals[2],
        )
        authoritative_sources = self._derive_outer_model_source_frames(
            manifest=model.manifest,
            scores=scores,
            completed=model.completed,
            maximum_peak_memory_bytes=remaining_memory,
        )
        authoritative_memory = sum(
            int(frame.memory_usage(index=True, deep=True).sum())
            for frame in authoritative_sources.values()
        )
        if authoritative_memory > remaining_memory:
            raise _error(
                "resource_budget_exceeded",
                "combined loaded and authoritative frames exceed budget",
            )
        _verify_derived_outer_model_sources(
            loaded=loaded,
            authoritative_sources=authoritative_sources,
        )
        self._validate_outer_rank_evidence(loaded=loaded, completed=model.completed)
        self._revalidate_current_authority(
            experiment_spec_hash=registered.content_hash,
            authority=authority,
        )
        completion = model.completed.completion
        receipt = ResearchReadinessAuthorityReceiptV1(
            manifest_hash=manifest.content_hash,
            request_binding_hash=_request_binding_hash(request),
            research_run_spec_hash=run_spec.content_hash,
            experiment_spec_hash=registered.content_hash,
            scientific_lineage_manifest_hash=scientific_lineage_manifest.content_hash,
            current_attempt_id=authority.current_attempt_id,
            current_attempt_number=authority.current_attempt_number,
            current_attempt_input_hash=authority.current_attempt_input_hash,
            parent_payload_hashes={
                stage: authority.parents[stage].payload_hash
                for stage in _DIRECT_PARENTS
            },
            parent_descriptor_hashes={
                stage: authority.parents[stage].descriptor_hash
                for stage in _DIRECT_PARENTS
            },
            parent_attempt_bindings={
                stage: authority.parents[stage].attempt_binding
                for stage in _DIRECT_PARENTS
            },
            phase_one_manifest_hash=completion.manifest_hash,
            outer_evaluation_completion_hash=completion.content_hash,
            outer_evaluation_result_hash=completion.result_hash,
            outer_execution_snapshot_hash=completion.execution_snapshot_hash,
            authoritative_frame_read_bytes=(
                _authoritative_frame_rederivation_read_bytes(
                    manifest=model.manifest,
                    scores=scores,
                )
            ),
            total_compressed_bytes=resource_totals[0],
            total_parquet_uncompressed_bytes=resource_totals[1],
            total_loaded_memory_bytes=resource_totals[2],
        )
        handoff_hash = _authority_handoff_hash(loaded, receipt)
        resolution_seal = self.__resolver_capability.issue(handoff_hash)
        return AuthorityBoundResearchReadinessInputs._from_verified_resolution(
            loaded=loaded,
            authority_receipt=receipt,
            resolver_capability=self.__resolver_capability,
            resolution_seal=resolution_seal,
        )

    def _validate_context(
        self,
        *,
        request: ScientificStageRequest,
        run_spec: ResearchRunSpec,
        scientific_lineage_manifest: ResearchScientificLineageManifestV2,
        readiness_plan: ResearchReadinessPlanV1,
    ) -> ExperimentSpec:
        if not isinstance(request, ScientificStageRequest):
            raise _error("request_binding_mismatch", "request type differs")
        if request.stage is not ResearchStage.ROBUSTNESS:
            raise _error("request_binding_mismatch", "request stage differs")
        if not isinstance(run_spec, ResearchRunSpec):
            raise _error("request_binding_mismatch", "run specification type differs")
        if not isinstance(
            scientific_lineage_manifest, ResearchScientificLineageManifestV2
        ):
            raise _error("lineage_binding_mismatch", "V2 scientific lineage is required")
        if not isinstance(readiness_plan, ResearchReadinessPlanV1):
            raise _error("plan_binding_mismatch", "readiness plan type differs")
        if run_spec.content_hash != request.research_run_spec_hash:
            raise _error("request_binding_mismatch", "run specification differs")
        try:
            registered = self.registry.load_experiment_spec(
                request.experiment_spec_hash
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise _error("request_binding_mismatch", "registered experiment missing") from exc
        if (
            registered.content_hash != request.experiment_spec_hash
            or self.runtime.spec.content_hash != registered.content_hash
        ):
            raise _error("request_binding_mismatch", "experiment authority differs")
        try:
            scientific_lineage_manifest.validate_for(run_spec)
        except (TypeError, ValueError) as exc:
            raise _error(
                "lineage_binding_mismatch", "scientific lineage validation failed"
            ) from exc
        if (
            scientific_lineage_manifest.content_hash
            != registered.scientific_lineage_manifest_hash
            or scientific_lineage_manifest.research_run_spec_hash
            != run_spec.content_hash
        ):
            raise _error("lineage_binding_mismatch", "scientific lineage differs")
        expected_contract = StageContract.for_run(run_spec, ResearchStage.ROBUSTNESS)
        if (
            request.contract.content_hash != expected_contract.content_hash
            or tuple(request.contract.required_parent_stages) != _DIRECT_PARENTS
        ):
            raise _error("request_binding_mismatch", "robustness contract differs")
        protected = {
            "data:partition:test",
            "data:partition:holdout",
        }.intersection(request.contract.component_bindings)
        if protected:
            raise _error("protected_partition_forbidden", "protected data is bound")
        self._validate_plan(run_spec=run_spec, plan=readiness_plan)
        return registered

    @staticmethod
    def _validate_plan(
        *, run_spec: ResearchRunSpec, plan: ResearchReadinessPlanV1
    ) -> None:
        robustness_hash = run_spec.robustness_spec_hash
        if robustness_hash is None or plan.readiness_spec_hash != robustness_hash:
            raise _error("plan_binding_mismatch", "readiness specification differs")
        try:
            partition_hash = run_spec.data_partition_hashes[DataPartition.VALIDATION]
            partition_role = run_spec.data_partition_roles[DataPartition.VALIDATION]
            window = run_spec.data_partition_windows[DataPartition.VALIDATION]
        except KeyError as exc:  # pragma: no cover - run contract requires validation.
            raise _error("plan_binding_mismatch", "validation partition is missing") from exc
        if (
            plan.partition is not DataPartition.VALIDATION
            or plan.data_role is not DataRole(partition_role)
            or plan.validation_partition_hash != partition_hash
            or plan.validation_spec_hash != run_spec.validation_spec_hash
            or (plan.validation_window_start, plan.validation_window_end) != tuple(window)
        ):
            raise _error("plan_binding_mismatch", "validation partition differs")

    @staticmethod
    def _validate_manifest_context(
        *,
        manifest: ResearchReadinessInputManifestV1,
        request: ScientificStageRequest,
        run_spec: ResearchRunSpec,
        experiment_spec: ExperimentSpec,
        scientific_lineage_manifest: ResearchScientificLineageManifestV2,
    ) -> None:
        expected = {
            "research_run_spec_hash": run_spec.content_hash,
            "experiment_spec_hash": experiment_spec.content_hash,
            "scientific_lineage_manifest_hash": scientific_lineage_manifest.content_hash,
            "robustness_contract_hash": request.contract.content_hash,
        }
        for name, digest in expected.items():
            if getattr(manifest, name) != digest:
                raise _error("manifest_binding_mismatch", f"manifest {name} differs")
        request_parents = {
            ResearchStage(stage): digest
            for stage, digest in request.parent_artifact_hashes.items()
        }
        if request_parents != dict(manifest.parent_artifact_hashes):
            raise _error("manifest_binding_mismatch", "manifest parents differ")

    def _load_runtime_authority(
        self,
        *,
        experiment_spec: ExperimentSpec,
        run_spec: ResearchRunSpec,
        request: ScientificStageRequest,
    ) -> _RuntimeAuthority:
        if tuple(experiment_spec.stages) != tuple(
            stage.value for stage in run_spec.enabled_stages
        ):
            raise _error("runtime_authority_mismatch", "stage topology differs")
        robustness_name = ResearchStage.ROBUSTNESS.value
        try:
            robustness_index = experiment_spec.stages.index(robustness_name)
        except ValueError as exc:
            raise _error("runtime_authority_mismatch", "robustness is not enabled") from exc
        prior_results: dict[str, str] = {}
        direct: dict[ResearchStage, _VerifiedParent] = {}
        for stage_name in experiment_spec.stages[:robustness_index]:
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
                    "runtime_authority_mismatch",
                    f"{stage_name} has no authoritative success",
                )
            expected_input = stage_input_hash(
                experiment_spec_hash=experiment_spec.content_hash,
                stage=stage_name,
                prior_result_hashes=prior_results,
            )
            if attempt.input_hash != expected_input:
                raise _error("runtime_authority_mismatch", f"{stage_name} input differs")
            if stage in _DIRECT_PARENTS:
                direct[stage] = self._load_parent(
                    experiment_spec=experiment_spec,
                    run_spec=run_spec,
                    stage=stage,
                    artifact_hash=attempt.result_hash,
                    expected_attempt_id=attempt.attempt_id,
                    expected_attempt_number=attempt.attempt_number,
                    expected_input_hash=expected_input,
                    expected_parent_hashes={
                        ResearchStage(parent): prior_results[
                            ResearchStage(parent).value
                        ]
                        for parent in StageContract.for_run(
                            run_spec, stage
                        ).required_parent_stages
                    },
                )
            prior_results[stage_name] = attempt.result_hash
        if tuple(stage for stage in _DIRECT_PARENTS if stage in direct) != _DIRECT_PARENTS:
            raise _error("runtime_authority_mismatch", "direct parents differ")
        requested = {
            ResearchStage(stage): digest
            for stage, digest in request.parent_artifact_hashes.items()
        }
        if requested != {
            stage: direct[stage].payload_hash for stage in _DIRECT_PARENTS
        }:
            raise _error("runtime_authority_mismatch", "request parents differ")
        expected_input = stage_input_hash(
            experiment_spec_hash=experiment_spec.content_hash,
            stage=robustness_name,
            prior_result_hashes=prior_results,
        )
        authority = _RuntimeAuthority(
            parents=MappingProxyType(direct),
            current_attempt_id=request.attempt_id,
            current_attempt_number=request.attempt_number,
            current_attempt_input_hash=expected_input,
            current_worker_id=self._worker_id,
            current_lease_token_hash=self._lease_token_hash,
        )
        self._revalidate_current_authority(
            experiment_spec_hash=experiment_spec.content_hash,
            authority=authority,
        )
        return authority

    def _revalidate_current_authority(
        self,
        *,
        experiment_spec_hash: str,
        authority: _RuntimeAuthority,
    ) -> None:
        """Atomically recheck the running lease and kill switch without mutation."""

        if (
            authority.current_worker_id != self._worker_id
            or authority.current_lease_token_hash != self._lease_token_hash
        ):
            raise _error(
                "runtime_authority_mismatch",
                "current lease owner, token, state or kill switch differs",
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
            raise _error(
                "runtime_authority_mismatch", "current authority reload failed"
            ) from exc
        if row is None:
            raise _error("runtime_authority_mismatch", "current attempt is missing")
        expected = {
            "attempt_id": authority.current_attempt_id,
            "experiment_spec_hash": experiment_spec_hash,
            "stage": ResearchStage.ROBUSTNESS.value,
            "attempt_number": authority.current_attempt_number,
            "status": AttemptStatus.RUNNING.value,
            "worker_id": authority.current_worker_id,
            "lease_token_hash": authority.current_lease_token_hash,
            "input_hash": authority.current_attempt_input_hash,
            "runtime_control_state": "running",
        }
        if any(row[name] != value for name, value in expected.items()) or (
            row["started_at"] is None
            or row["finished_at"] is not None
            or row["result_hash"] is not None
            or row["failure_code"] is not None
        ):
            raise _error(
                "runtime_authority_mismatch",
                "current lease owner, token, state or kill switch differs",
            )
        try:
            lease = datetime.fromisoformat(
                str(row["lease_expires_at"]).replace("Z", "+00:00")
            )
        except (TypeError, ValueError) as exc:
            raise _error("runtime_authority_mismatch", "lease timestamp differs") from exc
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise _error("runtime_authority_mismatch", "authority clock differs")
        if lease.tzinfo is None or lease.astimezone(timezone.utc) <= now.astimezone(
            timezone.utc
        ):
            raise _error("runtime_authority_mismatch", "current lease is expired")

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
        try:
            descriptor = dict(
                self.registry.load_artifact_descriptor(
                    experiment_spec.content_hash, artifact_hash
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise _error("parent_artifact_mismatch", f"{stage.value} is unavailable") from exc
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
            raise _error("parent_artifact_mismatch", f"{stage.value} descriptor differs")
        checkpoints = self.runtime.list_checkpoints(expected_attempt_id)
        if (
            len(checkpoints) != 1
            or checkpoints[0].checkpoint_name != "stage_result"
            or checkpoints[0].artifact_hash != artifact_hash
            or checkpoints[0].location != descriptor["location"]
        ):
            raise _error("runtime_authority_mismatch", f"{stage.value} checkpoint differs")
        try:
            record = ContentArtifactRecord(
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
        except (ContentArtifactError, OSError, TypeError, ValueError) as exc:
            raise _error("parent_artifact_mismatch", f"{stage.value} payload differs") from exc
        if (
            typed_descriptor.artifact_hash != artifact_hash
            or dict(typed_descriptor.parent_artifact_hashes)
            != dict(expected_parent_hashes)
            or scientific.stage is not stage
        ):
            raise _error("parent_artifact_mismatch", f"{stage.value} lineage differs")
        memory = body["memory_record"]
        if (
            not isinstance(memory, Mapping)
            or memory.get("result_hash") != scientific.content_hash
            or body["memory_record_hash"] != hash_json(dict(memory))
        ):
            raise _error("parent_artifact_mismatch", f"{stage.value} memory differs")
        descriptor_hash = cast(str, hash_json(descriptor))
        return _VerifiedParent(
            payload_hash=artifact_hash,
            descriptor_hash=descriptor_hash,
            attempt_binding=ResearchReadinessParentAttemptBindingV1(
                stage=stage,
                artifact_hash=artifact_hash,
                artifact_descriptor_hash=descriptor_hash,
                attempt_id=expected_attempt_id,
                attempt_number=expected_attempt_number,
                input_hash=expected_input_hash,
                checkpoint_hash=checkpoints[0].artifact_hash,
                checkpoint_location=checkpoints[0].location,
                stage_contract_hash=StageContract.for_run(run_spec, stage).content_hash,
            ),
            result=scientific,
        )

    def _load_model_source(
        self,
        *,
        parent: _VerifiedParent,
        experiment_spec: ExperimentSpec,
        request: ScientificStageRequest,
        run_spec: ResearchRunSpec,
        scientific_lineage_manifest: ResearchScientificLineageManifestV2,
        readiness_plan: ResearchReadinessPlanV1,
    ) -> _ModelSource:
        payload = parent.result.result_payload
        if frozenset(payload) != _MODEL_RESULT_FIELDS:
            raise _error("model_parent_mismatch", "model result fields differ")
        if payload["research_only"] is not True or payload["production_ready"] is not False:
            raise _error("model_parent_mismatch", "model release flags differ")
        try:
            phase_reference = NestedSelectionPhaseOneManifestReference.from_mapping(
                _mapping(payload["phase_one_manifest_reference"], "phase reference")
            )
            manifest_reference = ModelTrainingInputDocumentReferenceV1.from_mapping(
                _mapping(
                    payload["model_training_input_manifest_reference"],
                    "model input manifest reference",
                )
            )
            receipt_reference = ModelTrainingInputDocumentReferenceV1.from_mapping(
                _mapping(
                    payload["model_training_input_authority_receipt_reference"],
                    "model input receipt reference",
                )
            )
            model_manifest = ModelTrainingInputManifestV2.from_wire_bytes(
                self.artifact_store.read_bytes(manifest_reference.to_artifact_record())
            )
            model_receipt = ModelTrainingInputAuthorityReceiptV2.from_wire_bytes(
                self.artifact_store.read_bytes(receipt_reference.to_artifact_record())
            )
            published_parent_bindings = _mapping(
                payload["model_training_parent_attempt_bindings"],
                "model parent attempt bindings",
            )
        except (ContentArtifactError, OSError, TypeError, ValueError) as exc:
            raise _error("model_parent_mismatch", "model control documents differ") from exc
        self._validate_model_parent_attempt_bindings(
            published=published_parent_bindings,
            authority_receipt=model_receipt,
        )
        if (
            payload["verified_phase_one_manifest_hash"] != phase_reference.manifest_id
            or payload["model_training_input_manifest_hash"]
            != model_manifest.content_hash
            or payload["model_training_input_authority_receipt_hash"]
            != model_receipt.content_hash
            or model_receipt.manifest_hash != model_manifest.content_hash
            or model_manifest.research_run_spec_hash != run_spec.content_hash
            or model_manifest.experiment_spec_hash != request.experiment_spec_hash
            or model_manifest.scientific_lineage_manifest_hash
            != scientific_lineage_manifest.content_hash
            or model_manifest.validation_spec_hash
            != readiness_plan.validation_spec_hash
            or model_manifest.selection_spec_hash
            != readiness_plan.nested_selection_spec_hash
            or scientific_lineage_manifest.nested_selection_spec_hash
            != readiness_plan.nested_selection_spec_hash
            or model_manifest.scoring_eligibility_policy_hash
            != readiness_plan.scoring_eligibility_policy_hash
            or model_receipt.current_attempt_id != parent.attempt_binding.attempt_id
            or model_receipt.current_attempt_number
            != parent.attempt_binding.attempt_number
            or model_receipt.current_attempt_input_hash
            != parent.attempt_binding.input_hash
        ):
            raise _error("model_parent_mismatch", "model authority binding differs")
        if payload["nested_selection_authority_hash"] != self.outer_audit_reader.authority_hash:
            raise _error("outer_audit_mismatch", "outer audit authority differs")
        expected_seal = hash_json(
            {
                "schema_version": "verified-nested-selection-phase-one/v1",
                "authority_hash": self.outer_audit_reader.authority_hash,
                "reference": phase_reference.to_dict(),
                "manifest_hash": phase_reference.manifest_id,
            }
        )
        if payload["verified_phase_one_seal_hash"] != expected_seal:
            raise _error("outer_audit_mismatch", "phase-one seal differs")
        completed = self._load_completed_outer_evaluation(
            phase_reference,
            experiment_spec=experiment_spec,
        )
        result = completed.result
        if (
            completed.completion.manifest_hash != phase_reference.manifest_id
            or result.source_label_values_hash != model_manifest.label_values_hash
            or result.source_label_validity_hash != model_manifest.label_validity_hash
            or result.source_scoring_eligibility_hash
            != model_manifest.scoring_eligibility_hash
            or result.outer_validation_receipt_hash
            != model_manifest.validation_receipt_hash
            or model_manifest.validation_receipt_hash
            != readiness_plan.outer_validation_receipt_hash
            or completed.completion.evaluation_partition_hash
            != readiness_plan.outer_validation_receipt_hash
        ):
            raise _error("outer_audit_mismatch", "outer source lineage differs")
        return _ModelSource(
            manifest=model_manifest,
            phase_one_reference=phase_reference,
            completed=completed,
        )

    def _load_completed_outer_evaluation(
        self,
        phase_reference: NestedSelectionPhaseOneManifestReference,
        *,
        experiment_spec: ExperimentSpec,
    ) -> CompletedOuterEvaluation:
        registered = self._registered_outer_evidence_spec(experiment_spec)
        result_limit, execution_snapshot_limit = _outer_evidence_read_limits(
            registered
        )
        try:
            completed = NestedOuterEvaluationAuditReader.read_completed(
                self.outer_audit_reader,
                phase_reference,
                maximum_result_bytes=result_limit,
                maximum_execution_snapshot_bytes=execution_snapshot_limit,
            )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            raise _error("outer_audit_mismatch", "outer completion is unavailable") from exc
        if not isinstance(completed, CompletedOuterEvaluation):
            raise _error("outer_audit_mismatch", "outer completion type differs")
        return completed

    def _registered_outer_evidence_spec(
        self,
        experiment_spec: ExperimentSpec,
    ) -> ExperimentSpec:
        """Re-establish budget authority before every outer evidence read."""

        if type(experiment_spec) is not ExperimentSpec:
            raise _error("resource_budget_exceeded", "experiment budget differs")
        try:
            registered = self.registry.load_experiment_spec(
                experiment_spec.content_hash
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise _error(
                "resource_budget_exceeded",
                "registered experiment budget is unavailable",
            ) from exc
        if (
            registered.content_hash != experiment_spec.content_hash
            or registered.content_hash != self.runtime.spec.content_hash
        ):
            raise _error(
                "resource_budget_exceeded",
                "registered experiment budget authority differs",
            )
        return registered

    @staticmethod
    def _validate_model_parent_attempt_bindings(
        *,
        published: Mapping[str, object],
        authority_receipt: ModelTrainingInputAuthorityReceiptV2,
    ) -> None:
        expected = {
            ResearchStage(stage).value: binding.to_dict()
            for stage, binding in authority_receipt.parent_attempt_bindings.items()
        }
        if dict(published) != expected:
            raise _error(
                "model_parent_mismatch",
                "published model parent attempt bindings differ",
            )

    def _load_score_bundle(
        self,
        parent: _VerifiedParent,
        *,
        experiment_spec: ExperimentSpec,
        readiness_plan: ResearchReadinessPlanV1,
        completed: CompletedOuterEvaluation,
    ) -> ResearchReadinessScoreArtifactsV1:
        payload = parent.result.result_payload
        if frozenset(payload) != _SCORE_RESULT_FIELDS:
            raise _error("score_parent_mismatch", "score result fields differ")
        if (
            payload["schema_version"] != _SCORE_RESULT_SCHEMA
            or payload["research_only"] is not True
            or payload["production_ready"] is not False
        ):
            raise _error("score_parent_mismatch", "score release flags differ")
        try:
            publication_reference = _score_publication_reference(
                payload["score_recipe_publication_manifest_reference"]
            )
        except (ContentArtifactError, TypeError, ValueError) as exc:
            raise _error(
                "score_parent_mismatch", "score publication reference differs"
            ) from exc
        if (
            payload["score_recipe_publication_manifest_hash"]
            != publication_reference.sha256
        ):
            raise _error("score_parent_mismatch", "score publication hash differs")
        budget = self._score_reopen_budget(
            experiment_spec=experiment_spec,
            readiness_plan=readiness_plan,
        )
        reopened = ResearchReadinessScoreUpstreamProducer(
            self.artifact_store
        ).reopen(
            publication_manifest_reference=publication_reference,
            maximum_compressed_bytes=budget.maximum_compressed_bytes,
            maximum_uncompressed_bytes=budget.maximum_uncompressed_bytes,
            maximum_loaded_memory_bytes=budget.maximum_loaded_memory_bytes,
            maximum_rows=budget.maximum_rows,
            maximum_columns=budget.maximum_columns,
        )
        if dict(reopened.result_payload) != dict(payload):
            raise _error("score_parent_mismatch", "reopened score result differs")
        # ``readiness_upstream_bundles`` is a skipped-import boundary in the
        # isolated strict-mypy check, so preserve the exact runtime type here.
        scores = cast(ResearchReadinessScoreArtifactsV1, reopened.score_artifacts)
        publication = reopened.publication_manifest
        binding = publication.outer_evaluation_binding
        if frozenset(binding) != _OUTER_BINDING_FIELDS or binding.get(
            "schema_version"
        ) != "research-readiness-outer-score-binding/v2":
            raise _error("score_parent_mismatch", "outer score binding fields differ")
        completion = completed.completion
        expected = {
            "candidate_score_representation": (
                readiness_plan.candidate_score_representation
            ),
            "validation_partition_hash": readiness_plan.validation_partition_hash,
            "outer_validation_receipt_hash": (
                readiness_plan.outer_validation_receipt_hash
            ),
            "phase_one_manifest_hash": completion.manifest_hash,
            "outer_evaluation_completion_hash": completion.content_hash,
            "outer_evaluation_result_hash": completion.result_hash,
            "outer_execution_snapshot_hash": completion.execution_snapshot_hash,
        }
        if any(binding.get(name) != digest for name, digest in expected.items()):
            raise _error("inner_score_forbidden", "scores are not outer-evaluation bound")
        expected_definitions = {
            "outer_selected_candidate_rank": readiness_plan.candidate_definition_hash,
            "one_session_lagged_candidate_rank": (
                readiness_plan.baseline_definition_hash
            ),
            **dict(readiness_plan.perturbation_definition_hashes),
        }
        if (
            payload["research_readiness_score_artifacts_hash"]
            != scores.content_hash
            or publication.content_hash != publication_reference.sha256
            or publication.plan_hash != readiness_plan.content_hash
            or publication.nested_selection_spec_hash
            != readiness_plan.nested_selection_spec_hash
            or dict(publication.definition_hashes) != expected_definitions
        ):
            raise _error("score_parent_mismatch", "score bundle hash differs")
        if (
            scores.candidate_definition_hash
            != readiness_plan.candidate_definition_hash
            or scores.baseline_definition_hash
            != readiness_plan.baseline_definition_hash
            or dict(scores.perturbation_definition_hashes)
            != dict(readiness_plan.perturbation_definition_hashes)
        ):
            raise _error("score_parent_mismatch", "score definitions differ")
        return scores

    @staticmethod
    def _score_reopen_budget(
        *,
        experiment_spec: ExperimentSpec,
        readiness_plan: ResearchReadinessPlanV1,
    ) -> _ScoreReopenBudget:
        """Map only the registered experiment budget into finite reopen limits."""

        if type(experiment_spec) is not ExperimentSpec:
            raise _error("resource_budget_exceeded", "experiment budget differs")
        frame_count = 2 + len(readiness_plan.perturbation_definition_hashes)
        peak = experiment_spec.resource_budget.maximum_peak_memory_bytes
        disk = experiment_spec.resource_budget.maximum_disk_write_bytes
        maximum_axis = peak // (_NUMERIC_SCORE_BYTES * frame_count)
        if maximum_axis <= 0:
            raise _error(
                "resource_budget_exceeded",
                "registered memory budget cannot admit one score cell",
            )
        return _ScoreReopenBudget(
            maximum_compressed_bytes=disk,
            maximum_uncompressed_bytes=peak,
            maximum_loaded_memory_bytes=peak,
            maximum_rows=maximum_axis,
            maximum_columns=maximum_axis,
        )

    def _prepare_outer_model_source_evidence(
        self,
        *,
        manifest: ModelTrainingInputManifestV2,
        scores: ResearchReadinessScoreArtifactsV1,
        completed: CompletedOuterEvaluation,
        maximum_peak_memory_bytes: int,
        maximum_disk_write_bytes: int,
    ) -> tuple[_PreparedOuterModelSource, ...]:
        selected = self._derive_outer_model_source_frames(
            manifest=manifest,
            scores=scores,
            completed=completed,
            maximum_peak_memory_bytes=maximum_peak_memory_bytes,
        )
        sources = (
            ("label_values", manifest.label_values),
            ("label_validity", manifest.label_validity),
            ("scoring_eligibility", manifest.scoring_eligibility),
        )
        candidate_reference = scores.candidate_scores
        prepared: list[_PreparedOuterModelSource] = []
        total_encoded = 0
        total_uncompressed = 0
        selected_memory = sum(
            int(frame.memory_usage(index=True, deep=True).sum())
            for frame in selected.values()
        )
        source_references = dict(sources)
        for role in ("label_values", "label_validity", "scoring_eligibility"):
            frame = selected[role]
            payload, parquet_uncompressed = _derived_parquet_payload(frame, role=role)
            total_encoded += len(payload)
            total_uncompressed += parquet_uncompressed
            source_hash = source_references[role].frame_hash
            prepared.append(
                _PreparedOuterModelSource(
                    role=role,
                    logical_name=(
                        f"readiness.derived.outer.{role}."
                        f"{source_hash[:16]}.{candidate_reference.frame_hash[:16]}"
                    ),
                    frame=frame,
                    payload=payload,
                    parquet_uncompressed_bytes=parquet_uncompressed,
                )
            )
        if (
            total_encoded > maximum_disk_write_bytes
            or total_encoded > maximum_peak_memory_bytes
            or total_uncompressed > maximum_peak_memory_bytes
            or selected_memory
            + total_encoded
            + _PARQUET_WORKING_SET_MULTIPLIER * total_uncompressed
            > maximum_peak_memory_bytes
        ):
            raise _error(
                "resource_budget_exceeded",
                "derived outer source frames exceed budget",
            )
        return tuple(prepared)

    def _derive_outer_model_source_frames(
        self,
        *,
        manifest: ModelTrainingInputManifestV2,
        scores: ResearchReadinessScoreArtifactsV1,
        completed: CompletedOuterEvaluation,
        maximum_peak_memory_bytes: int,
    ) -> Mapping[str, pd.DataFrame]:
        sources = (
            ("label_values", manifest.label_values),
            ("label_validity", manifest.label_validity),
            ("scoring_eligibility", manifest.scoring_eligibility),
        )
        candidate_reference = scores.candidate_scores
        compressed = candidate_reference.size_bytes + sum(
            item.size_bytes for _, item in sources
        )
        uncompressed = candidate_reference.parquet_uncompressed_bytes + sum(
            item.parquet_uncompressed_bytes for _, item in sources
        )
        if (
            compressed > maximum_peak_memory_bytes
            or uncompressed > maximum_peak_memory_bytes
            or compressed + _PARQUET_WORKING_SET_MULTIPLIER * uncompressed
            > maximum_peak_memory_bytes
        ):
            raise _error("resource_budget_exceeded", "model source frames exceed budget")
        candidate = self._decode_readiness_reference(
            candidate_reference,
            maximum_uncompressed_bytes=maximum_peak_memory_bytes,
        )
        decoded_sources: dict[str, pd.DataFrame] = {}
        for role, reference in sources:
            try:
                payload = self.artifact_store.read_bytes(reference.to_artifact_record())
                frame = reference.decode_parquet_payload(
                    payload,
                    maximum_uncompressed_bytes=maximum_peak_memory_bytes,
                )
            except (ContentArtifactError, OSError, TypeError, ValueError) as exc:
                raise _error("artifact_payload_mismatch", f"{role} reload failed") from exc
            decoded_sources[role] = frame
        return _select_outer_model_source_frames(
            candidate=candidate,
            label_values=decoded_sources["label_values"],
            label_validity=decoded_sources["label_validity"],
            scoring_eligibility=decoded_sources["scoring_eligibility"],
            completed=completed,
        )

    def _publish_outer_model_source_evidence(
        self,
        prepared: tuple[_PreparedOuterModelSource, ...],
    ) -> tuple[
        ResearchReadinessFrameReferenceV1,
        ResearchReadinessFrameReferenceV1,
        ResearchReadinessFrameReferenceV1,
    ]:
        expected_roles = (
            "label_values",
            "label_validity",
            "scoring_eligibility",
        )
        if tuple(item.role for item in prepared) != expected_roles:
            raise _error("invalid_frame", "derived source roles differ")
        results: list[ResearchReadinessFrameReferenceV1] = []
        for item in prepared:
            try:
                record = self.artifact_store.put_bytes(
                    item.logical_name,
                    item.payload,
                    media_type="application/vnd.apache.parquet",
                    role=item.role,
                )
                reference = ResearchReadinessFrameReferenceV1.bind_verified_frame(
                    record=record,
                    frame=item.frame,
                    semantic_role=item.role,
                    parquet_uncompressed_bytes=item.parquet_uncompressed_bytes,
                )
                restored = self._decode_readiness_reference(
                    reference,
                    maximum_uncompressed_bytes=item.parquet_uncompressed_bytes,
                )
            except (ContentArtifactError, OSError, TypeError, ValueError) as exc:
                raise _error(
                    "artifact_payload_mismatch",
                    f"derived {item.role} publication failed",
                ) from exc
            if not restored.equals(item.frame):
                raise _error(
                    "artifact_payload_mismatch",
                    f"derived {item.role} replay differs",
                )
            results.append(reference)
        return results[0], results[1], results[2]

    def _load_and_verify_all_frames(
        self,
        manifest: ResearchReadinessInputManifestV1,
        *,
        maximum_peak_memory_bytes: int,
    ) -> tuple[LoadedResearchReadinessInputs, tuple[int, int, int]]:
        references = {
            "candidate_score": manifest.score_artifacts.candidate_scores,
            "baseline_score": manifest.score_artifacts.baseline_scores,
            **{
                f"perturbation_score:{name}": reference
                for name, reference in manifest.score_artifacts.perturbation_scores.items()
            },
            "label_values": manifest.label_values,
            "label_validity": manifest.label_validity,
            "scoring_eligibility": manifest.scoring_eligibility,
        }
        compressed = sum(item.size_bytes for item in references.values())
        uncompressed = sum(
            item.parquet_uncompressed_bytes for item in references.values()
        )
        if (
            compressed > maximum_peak_memory_bytes
            or uncompressed > maximum_peak_memory_bytes
            or compressed + _PARQUET_WORKING_SET_MULTIPLIER * uncompressed
            > maximum_peak_memory_bytes
        ):
            raise _error("resource_budget_exceeded", "parquet working set exceeds budget")
        frames: dict[str, pd.DataFrame] = {}
        loaded_bytes = 0
        for name, reference in references.items():
            frame = self._decode_readiness_reference(
                reference,
                maximum_uncompressed_bytes=maximum_peak_memory_bytes,
            )
            loaded_bytes += int(frame.memory_usage(index=True, deep=True).sum())
            if loaded_bytes > maximum_peak_memory_bytes:
                raise _error("resource_budget_exceeded", "loaded frames exceed budget")
            frames[name] = frame
        loaded = LoadedResearchReadinessInputs(
            manifest=manifest,
            candidate_scores=frames["candidate_score"],
            baseline_scores=frames["baseline_score"],
            perturbation_scores={
                name.removeprefix("perturbation_score:"): frame
                for name, frame in frames.items()
                if name.startswith("perturbation_score:")
            },
            labels=frames["label_values"],
            label_validity=frames["label_validity"],
            scoring_eligibility=frames["scoring_eligibility"],
        )
        if loaded.loaded_memory_bytes != loaded_bytes:
            raise _error("resource_budget_exceeded", "loaded memory accounting differs")
        return loaded, (compressed, uncompressed, loaded_bytes)

    def _decode_readiness_reference(
        self,
        reference: ResearchReadinessFrameReferenceV1,
        *,
        maximum_uncompressed_bytes: int,
    ) -> pd.DataFrame:
        try:
            record = ContentArtifactRecord(
                logical_name=reference.logical_name,
                location=reference.location,
                sha256=reference.payload_sha256,
                size_bytes=reference.size_bytes,
                media_type=reference.media_type,
                role=reference.semantic_role,
            )
            payload = self.artifact_store.read_bytes(record)
        except (ContentArtifactError, OSError, TypeError, ValueError) as exc:
            raise _error(
                "artifact_payload_mismatch", f"{reference.logical_name} reload failed"
            ) from exc
        if (
            len(payload) != reference.size_bytes
            or hashlib.sha256(payload).hexdigest() != reference.payload_sha256
        ):
            raise _error("artifact_payload_mismatch", "parquet payload differs")
        try:
            parquet = pq.ParquetFile(pa.BufferReader(payload))
            observed_uncompressed = _parquet_uncompressed_bytes(parquet)
        except (pa.ArrowException, OSError, ValueError) as exc:
            raise _error("invalid_parquet", "parquet footer is invalid") from exc
        if (
            observed_uncompressed != reference.parquet_uncompressed_bytes
            or observed_uncompressed > maximum_uncompressed_bytes
        ):
            code = (
                "resource_budget_exceeded"
                if observed_uncompressed > maximum_uncompressed_bytes
                else "parquet_metadata_mismatch"
            )
            raise _error(code, "parquet uncompressed size differs")
        metadata = parquet.metadata
        if metadata is None or metadata.num_rows != reference.row_count:
            raise _error("parquet_metadata_mismatch", "parquet row count differs")
        try:
            frame = parquet.read().to_pandas()
        except (pa.ArrowException, OSError, ValueError) as exc:
            raise _error("invalid_parquet", "parquet frame cannot be decoded") from exc
        reference.verify_frame(frame)
        return frame

    @staticmethod
    def _validate_outer_rank_evidence(
        *,
        loaded: LoadedResearchReadinessInputs,
        completed: CompletedOuterEvaluation,
    ) -> None:
        """Rebuild the candidate panel only from sealed outer-fold rank vectors."""

        _validate_outer_rank_frames(
            candidate=loaded.candidate_scores,
            labels=loaded.labels,
            validity=loaded.label_validity,
            eligibility=loaded.scoring_eligibility,
            completed=completed,
        )

    @staticmethod
    def _validate_parent_derived_manifest(
        *,
        manifest: ResearchReadinessInputManifestV1,
        authority: _RuntimeAuthority,
        model: _ModelSource,
        scores: ResearchReadinessScoreArtifactsV1,
    ) -> None:
        completion = model.completed.completion
        expected_hashes = {
            "phase_one_manifest_hash": completion.manifest_hash,
            "outer_evaluation_completion_hash": completion.content_hash,
            "outer_evaluation_result_hash": completion.result_hash,
            "outer_execution_snapshot_hash": completion.execution_snapshot_hash,
            "score_artifacts_hash": scores.content_hash,
            "label_spec_hash": model.manifest.label_spec_hash,
            "outer_validation_receipt_hash": (
                model.manifest.validation_receipt_hash
            ),
        }
        if any(getattr(manifest, name) != value for name, value in expected_hashes.items()):
            raise _error("parent_binding_mismatch", "manifest parent lineage differs")
        if manifest.score_artifacts != scores:
            raise _error("parent_binding_mismatch", "manifest score bundle differs")
        if dict(manifest.parent_artifact_hashes) != {
            stage: authority.parents[stage].payload_hash for stage in _DIRECT_PARENTS
        }:
            raise _error("parent_binding_mismatch", "manifest parent artifacts differ")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise _error("invalid_parent_payload", f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _score_publication_reference(value: object) -> ContentArtifactRecord:
    raw = _mapping(value, "score publication reference")
    if frozenset(raw) != _SCORE_PUBLICATION_REFERENCE_FIELDS:
        raise _error(
            "score_parent_mismatch", "score publication reference fields differ"
        )
    size = raw["size_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise _error("score_parent_mismatch", "score publication size differs")
    textual = {
        name: raw[name]
        for name in ("logical_name", "location", "sha256", "media_type", "role")
    }
    if not all(isinstance(item, str) and item for item in textual.values()):
        raise _error("score_parent_mismatch", "score publication fields differ")
    return ContentArtifactRecord(
        logical_name=cast(str, raw["logical_name"]),
        location=cast(str, raw["location"]),
        sha256=cast(str, raw["sha256"]),
        size_bytes=size,
        media_type=cast(str, raw["media_type"]),
        role=cast(str, raw["role"]),
    )


def _governed_body(payload: bytes) -> Mapping[str, object]:
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, Mapping):
        raise _error("parent_artifact_mismatch", "stage envelope is invalid")
    body = decoded.get("payload")
    if not isinstance(body, Mapping) or frozenset(body) != _GOVERNED_BODY_FIELDS:
        raise _error("parent_artifact_mismatch", "governed body fields differ")
    if body.get("schema_version") != "governed-agent-stage/v1":
        raise _error("parent_artifact_mismatch", "governed body schema differs")
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


def _remaining_memory_budget(
    *,
    maximum_peak_memory_bytes: int,
    retained_loaded_memory_bytes: int,
) -> int:
    if (
        not isinstance(maximum_peak_memory_bytes, int)
        or isinstance(maximum_peak_memory_bytes, bool)
        or maximum_peak_memory_bytes <= 0
        or not isinstance(retained_loaded_memory_bytes, int)
        or isinstance(retained_loaded_memory_bytes, bool)
        or retained_loaded_memory_bytes < 0
        or retained_loaded_memory_bytes >= maximum_peak_memory_bytes
    ):
        raise _error(
            "resource_budget_exceeded",
            "retained frames exhaust registered memory budget",
        )
    return maximum_peak_memory_bytes - retained_loaded_memory_bytes


def _strict_axis_equal(left: pd.Index, right: pd.Index) -> bool:
    return (
        type(left) is type(right)
        and left.equals(right)
        and tuple(left.names) == tuple(right.names)
        and str(left.dtype) == str(right.dtype)
    )


def _sealed_outer_axes(
    completed: CompletedOuterEvaluation,
) -> tuple[pd.DatetimeIndex, pd.Index]:
    """Derive the one allowed score axis from sealed cohort/prediction vectors."""

    seen: set[pd.Timestamp] = set()
    members: set[str] = set()
    for evaluation in completed.result.outer_evaluations:
        score = evaluation.outer_score
        cohorts = score.scoring_cohort_evidence.rank_vectors
        predictions = score.prediction_rank_evidence.rank_vectors
        if len(cohorts) != len(predictions):
            raise _error("outer_score_mismatch", "outer rank vector counts differ")
        for cohort, prediction in zip(cohorts, predictions, strict=True):
            try:
                timestamp = pd.Timestamp(cohort.signal_timestamp)
            except (TypeError, ValueError) as exc:
                raise _error(
                    "outer_score_mismatch", "outer rank timestamp differs"
                ) from exc
            cohort_members = tuple(cohort.security_members)
            if (
                timestamp.tzinfo is None
                or prediction.signal_timestamp != cohort.signal_timestamp
                or tuple(prediction.security_members) != cohort_members
                or not cohort_members
                or cohort_members != tuple(sorted(set(cohort_members)))
                or not all(
                    isinstance(member, str) and member for member in cohort_members
                )
                or len(cohort.target_ranks) != len(cohort_members)
                or len(prediction.prediction_ranks) != len(cohort_members)
            ):
                raise _error(
                    "outer_score_mismatch", "outer vector axes or values differ"
                )
            canonical_timestamp = timestamp.tz_convert("UTC")
            if canonical_timestamp in seen:
                raise _error(
                    "outer_score_mismatch", "outer rank timestamps overlap"
                )
            seen.add(canonical_timestamp)
            members.update(cohort_members)
    if not seen or not members:
        raise _error("outer_score_mismatch", "outer rank evidence is empty")
    return (
        pd.DatetimeIndex(sorted(seen)),
        pd.Index(sorted(members)),
    )


def _validate_exact_outer_score_axes(
    candidate: pd.DataFrame,
    *,
    completed: CompletedOuterEvaluation,
) -> None:
    expected_index, expected_columns = _sealed_outer_axes(completed)
    if (
        type(candidate) is not pd.DataFrame
        or not isinstance(candidate.index, pd.DatetimeIndex)
        or candidate.index.tz is None
        or not candidate.index.is_unique
        or not candidate.index.is_monotonic_increasing
        or not candidate.columns.is_unique
        or not _strict_axis_equal(candidate.index, expected_index)
        or not _strict_axis_equal(candidate.columns, expected_columns)
    ):
        raise _error(
            "outer_score_mismatch",
            "score axes are not the exact sealed outer-evaluation axes",
        )


def _select_outer_model_source_frames(
    *,
    candidate: pd.DataFrame,
    label_values: pd.DataFrame,
    label_validity: pd.DataFrame,
    scoring_eligibility: pd.DataFrame,
    completed: CompletedOuterEvaluation,
) -> Mapping[str, pd.DataFrame]:
    """Select an order-preserving MODEL subset without fill or ambiguous reindex."""

    _validate_exact_outer_score_axes(candidate, completed=completed)
    sources = {
        "label_values": label_values,
        "label_validity": label_validity,
        "scoring_eligibility": scoring_eligibility,
    }
    first = label_values
    if (
        any(type(frame) is not pd.DataFrame for frame in sources.values())
        or not isinstance(first.index, pd.DatetimeIndex)
        or first.index.tz is None
        or not first.index.is_unique
        or not first.index.is_monotonic_increasing
        or not first.columns.is_unique
        or any(
            not _strict_axis_equal(frame.index, first.index)
            or not _strict_axis_equal(frame.columns, first.columns)
            for frame in sources.values()
        )
    ):
        raise _error(
            "model_source_axis_mismatch", "MODEL source panel axes differ"
        )
    source_index_utc = first.index.tz_convert("UTC")
    row_positions = source_index_utc.get_indexer(candidate.index)
    column_positions = first.columns.get_indexer(candidate.columns)
    if (
        any(int(position) < 0 for position in row_positions)
        or any(int(position) < 0 for position in column_positions)
        or tuple(int(position) for position in row_positions)
        != tuple(sorted(int(position) for position in row_positions))
        or tuple(int(position) for position in column_positions)
        != tuple(sorted(int(position) for position in column_positions))
    ):
        raise _error(
            "model_source_axis_mismatch",
            "outer score axes are not an order-preserving MODEL subset",
        )
    selected: dict[str, pd.DataFrame] = {}
    for role, source in sources.items():
        frame = source.iloc[
            list(int(position) for position in row_positions),
            list(int(position) for position in column_positions),
        ].copy(deep=True)
        # The score publication defines the canonical validation-only axis
        # representation (UTC, no inherited MODEL axis names).
        frame.index = candidate.index.copy()
        frame.columns = candidate.columns.copy()
        if (
            not _strict_axis_equal(frame.index, candidate.index)
            or not _strict_axis_equal(frame.columns, candidate.columns)
        ):
            raise _error(
                "model_source_axis_mismatch", f"derived {role} axes differ"
            )
        selected[role] = frame
    _validate_outer_rank_frames(
        candidate=candidate,
        labels=selected["label_values"],
        validity=selected["label_validity"],
        eligibility=selected["scoring_eligibility"],
        completed=completed,
    )
    return MappingProxyType(selected)


def derive_outer_model_source_frames_v1(
    *,
    candidate: pd.DataFrame,
    label_values: pd.DataFrame,
    label_validity: pd.DataFrame,
    scoring_eligibility: pd.DataFrame,
    completed: CompletedOuterEvaluation,
) -> Mapping[str, pd.DataFrame]:
    """Derive the exact validation-only MODEL sources for a sealed outer score.

    This is the public, pure replay boundary shared by readiness materialization
    and downstream lineage proofs.  Callers remain responsible for fresh-loading
    the immutable inputs; this function proves the typed axes, the
    order-preserving MODEL subset, and every sealed cohort/target/prediction rank
    before returning deep-copied frames.
    """

    if type(completed) is not CompletedOuterEvaluation:
        raise TypeError("outer model source derivation requires exact completion")
    if any(
        type(frame) is not pd.DataFrame
        for frame in (
            candidate,
            label_values,
            label_validity,
            scoring_eligibility,
        )
    ):
        raise TypeError("outer model source derivation requires exact DataFrames")
    return _select_outer_model_source_frames(
        candidate=candidate,
        label_values=label_values,
        label_validity=label_validity,
        scoring_eligibility=scoring_eligibility,
        completed=completed,
    )


def _validate_outer_rank_frames(
    *,
    candidate: pd.DataFrame,
    labels: pd.DataFrame,
    validity: pd.DataFrame,
    eligibility: pd.DataFrame,
    completed: CompletedOuterEvaluation,
) -> None:
    """Prove exact score values and PIT membership against sealed outer vectors."""

    _validate_exact_outer_score_axes(candidate, completed=completed)
    if not all(isinstance(item, str) for item in candidate.columns):
        raise _error("outer_score_mismatch", "candidate security axis is not textual")
    if any(
        not _strict_axis_equal(frame.index, candidate.index)
        or not _strict_axis_equal(frame.columns, candidate.columns)
        for frame in (labels, validity, eligibility)
    ):
        raise _error("outer_score_mismatch", "outer evidence panel axes differ")
    rebuilt = pd.DataFrame(
        float("nan"),
        index=candidate.index.copy(),
        columns=candidate.columns.copy(),
        dtype=float,
    )
    seen_timestamps: set[pd.Timestamp] = set()
    result = completed.result
    for evaluation in result.outer_evaluations:
        score = evaluation.outer_score
        cohort_vectors = score.scoring_cohort_evidence.rank_vectors
        prediction_vectors = score.prediction_rank_evidence.rank_vectors
        if len(cohort_vectors) != len(prediction_vectors):
            raise _error("outer_score_mismatch", "outer rank vector counts differ")
        if (
            score.scoring_cohort_evidence.label_values_hash
            != result.source_label_values_hash
            or score.scoring_cohort_evidence.label_validity_hash
            != result.source_label_validity_hash
            or score.scoring_cohort_evidence.eligibility_hash
            != result.source_scoring_eligibility_hash
        ):
            raise _error("outer_score_mismatch", "outer target lineage differs")
        for cohort, prediction in zip(
            cohort_vectors, prediction_vectors, strict=True
        ):
            timestamp = pd.Timestamp(cohort.signal_timestamp).tz_convert("UTC")
            if (
                prediction.signal_timestamp != cohort.signal_timestamp
                or prediction.security_members != cohort.security_members
                or timestamp in seen_timestamps
                or timestamp not in candidate.index
            ):
                raise _error(
                    "outer_score_mismatch", "outer vector axes overlap or differ"
                )
            seen_timestamps.add(timestamp)
            members = cohort.security_members
            if any(member not in candidate.columns for member in members):
                raise _error("outer_score_mismatch", "outer vector member is unknown")
            active = (
                validity.loc[timestamp]
                & eligibility.loc[timestamp]
                & labels.loc[timestamp].notna()
                & candidate.loc[timestamp].notna()
            )
            expected_members = tuple(
                sorted(cast(str, member) for member in active.index[active])
            )
            if members != expected_members:
                raise _error(
                    "outer_score_mismatch", "outer scoring membership differs"
                )
            expected_targets = tuple(
                float(value)
                for value in labels.loc[timestamp, list(members)]
                .rank(method="average")
                .tolist()
            )
            if cohort.target_ranks != expected_targets:
                raise _error("outer_score_mismatch", "outer target ranks differ")
            observed_predictions = tuple(
                float(value)
                for value in candidate.loc[timestamp, list(members)].tolist()
            )
            if prediction.prediction_ranks != observed_predictions:
                raise _error("outer_score_mismatch", "outer prediction ranks differ")
            rebuilt.loc[timestamp, list(members)] = list(
                prediction.prediction_ranks
            )
    if not seen_timestamps or not rebuilt.equals(candidate.astype(float)):
        raise _error(
            "outer_score_mismatch",
            "candidate panel is not exactly reconstructed from outer evidence",
        )


def _verify_derived_outer_model_sources(
    *,
    loaded: LoadedResearchReadinessInputs,
    authoritative_sources: Mapping[str, pd.DataFrame],
) -> None:
    """Reject a caller-supplied same-axis slice not derived from MODEL authority."""

    actual = {
        "label_values": loaded.labels,
        "label_validity": loaded.label_validity,
        "scoring_eligibility": loaded.scoring_eligibility,
    }
    if tuple(authoritative_sources) != tuple(actual):
        raise _error(
            "derived_source_mismatch", "authoritative derived source roles differ"
        )
    for role, expected in authoritative_sources.items():
        observed = actual[role]
        if (
            not _strict_axis_equal(observed.index, expected.index)
            or not _strict_axis_equal(observed.columns, expected.columns)
            or not observed.equals(expected)
        ):
            raise _error(
                "derived_source_mismatch",
                f"{role} is not the exact MODEL-derived outer slice",
            )


def _derived_parquet_payload(frame: pd.DataFrame, *, role: str) -> tuple[bytes, int]:
    buffer = io.BytesIO()
    try:
        frame.to_parquet(buffer, engine="pyarrow", compression="zstd", index=True)
        payload = buffer.getvalue()
        parquet = pq.ParquetFile(pa.BufferReader(payload))
        uncompressed = _parquet_uncompressed_bytes(parquet)
    except (pa.ArrowException, OSError, TypeError, ValueError) as exc:
        raise _error(
            "invalid_parquet", f"derived {role} parquet serialization failed"
        ) from exc
    if not payload or parquet.metadata is None or parquet.metadata.num_rows != len(frame):
        raise _error("invalid_parquet", f"derived {role} parquet metadata differs")
    return payload, uncompressed


def _outer_evidence_read_limits(experiment_spec: ExperimentSpec) -> tuple[int, int]:
    """Derive immutable outer-document limits from registered experiment state.

    These are per-document encoded-byte limits.  The store independently caps
    them at its global hard maxima and rejects the completion metadata before
    opening either large document.
    """

    if type(experiment_spec) is not ExperimentSpec:
        raise _error("resource_budget_exceeded", "experiment budget differs")
    peak = experiment_spec.resource_budget.maximum_peak_memory_bytes
    return (
        min(peak, MAXIMUM_OUTER_EVALUATION_RESULT_BYTES),
        min(peak, MAXIMUM_OUTER_EVALUATION_EXECUTION_SNAPSHOT_BYTES),
    )


def _parquet_uncompressed_bytes(parquet: pq.ParquetFile) -> int:
    metadata = parquet.metadata
    if metadata is None:
        raise _error("invalid_parquet", "parquet metadata is missing")
    total = sum(
        int(metadata.row_group(row).column(column).total_uncompressed_size)
        for row in range(metadata.num_row_groups)
        for column in range(metadata.row_group(row).num_columns)
    )
    if total <= 0:
        raise _error("invalid_parquet", "parquet uncompressed size is invalid")
    return total


def _runtime_code(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        raise TypeError(f"readiness resolver {name} is invalid")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-")
    if not value[0].isalnum() or any(character not in allowed for character in value):
        raise ValueError(f"readiness resolver {name} is invalid")
    return value


def _lease_token_hash(value: object) -> str:
    if not isinstance(value, str) or len(value) < 16:
        raise ValueError(
            "readiness resolver lease_token must contain at least 16 characters"
        )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "ResearchReadinessInputResolver",
    "derive_outer_model_source_frames_v1",
]
