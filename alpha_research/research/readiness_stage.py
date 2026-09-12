"""Resolver-bound ML-F research-readiness stage orchestration.

The public factory deliberately accepts no caller-authored data frames or input
manifest.  Every invocation asks :class:`ResearchReadinessInputResolver` to
reconstruct validation-only inputs from the registered parent-attempt chain,
runs the deterministic readiness checks, persists all evidence, and replays it
before returning a deliberately small, metric-free scientific payload.  The
governed Assembly wires this stage only after the preceding SCORE_CONSTRUCTION
stage has executed the closed, preregistered baseline and perturbation recipes;
content addressing alone is not treated as scientific authority.

This stage can only open the deep-learning *research* gate.  It cannot authorize
hidden-OOS access, production release, or portfolio deployment.
"""

from __future__ import annotations

from datetime import datetime
import resource
import sys
import time
from typing import Callable, Final, cast, final

from alpha_research.core.hashing import require_sha256
from alpha_research.experiments import ExperimentSpec
from alpha_research.experiments.registry import ExperimentRegistry
from alpha_research.orchestration import ExperimentRuntime, reported_usage
from alpha_research.research.execution import (
    JsonValue,
    ScientificStageImplementation,
    ScientificStageRequest,
    ScientificStageResult,
)
from alpha_research.research.lineage import (
    ResearchScientificLineageManifestV2,
)
from alpha_research.research.nested_selection_control import (
    NestedOuterEvaluationAuditReader,
)
from alpha_research.research.readiness_bundle_document import (
    ResearchReadinessBundleDocumentError,
    ResearchReadinessBundleLoaderV1,
    ResearchReadinessBundleProducerV1,
)
from alpha_research.research.readiness_input_resolver import (
    ResearchReadinessInputResolver,
    _RuntimeAuthority,
)
from alpha_research.research.readiness_inputs import (
    ResearchReadinessAuthorityReceiptV1,
    ResearchReadinessInputError,
    ResearchReadinessPlanV1,
)
from alpha_research.research.spec import ResearchRunSpec, ResearchStage
from alpha_research.robustness.readiness import (
    DLReadinessVerdict,
    ResearchReadinessRunner,
    ResearchReadinessSpec,
)
from factor_production.v5.artifacts import ArtifactStore
from factor_production.v5.artifacts.manifest import ArtifactError


_BOUND_STAGE_TOKEN: Final = object()
_DECISION_CODES: Final = {
    DLReadinessVerdict.GO_RESEARCH_ONLY: "dl_research_go",
    DLReadinessVerdict.NO_GO: "no_go",
    DLReadinessVerdict.INCONCLUSIVE: "inconclusive",
}
_DECISION_SCOPE: Final = "dl_research_only"
_BACKTEST_DEPENDENCY_ROLE: Final = "topology_only"


class ResearchReadinessStageError(RuntimeError):
    """Stable, fail-closed error emitted by the authority-bound stage."""

    def __init__(self, code: str, detail: str) -> None:
        if (
            not isinstance(code, str)
            or not code
            or not code[0].isalpha()
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in code)
        ):
            raise ValueError("research readiness stage error code is unsafe")
        self.code = code
        self.detail = str(detail)
        super().__init__(f"{self.code}:{self.detail}")


def _stage_error(code: str, detail: str) -> ResearchReadinessStageError:
    return ResearchReadinessStageError(code, detail)


@final
class ResolvedResearchReadinessFactory:
    """Freeze one exact ML-F plan before binding runtime authorities."""

    __slots__ = (
        "_implementation_hash",
        "_implementation_id",
        "_lineage",
        "_readiness_plan",
        "_readiness_spec",
        "_run_spec",
    )

    _implementation_hash: str
    _implementation_id: str
    _lineage: ResearchScientificLineageManifestV2
    _readiness_plan: ResearchReadinessPlanV1
    _readiness_spec: ResearchReadinessSpec
    _run_spec: ResearchRunSpec

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("ResolvedResearchReadinessFactory cannot be subclassed")

    def __init__(
        self,
        *,
        run_spec: ResearchRunSpec,
        scientific_lineage_manifest: ResearchScientificLineageManifestV2,
        readiness_spec: ResearchReadinessSpec,
        readiness_plan: ResearchReadinessPlanV1,
        implementation_id: str,
        implementation_hash: str,
    ) -> None:
        if type(run_spec) is not ResearchRunSpec:
            raise TypeError("readiness factory run specification differs")
        if type(scientific_lineage_manifest) is not ResearchScientificLineageManifestV2:
            raise TypeError("readiness factory requires exact V2 scientific lineage")
        if type(readiness_spec) is not ResearchReadinessSpec:
            raise TypeError("readiness factory specification differs")
        if type(readiness_plan) is not ResearchReadinessPlanV1:
            raise TypeError("readiness factory plan differs")
        run_snapshot = ResearchRunSpec.from_mapping(run_spec.to_dict())
        lineage_snapshot = ResearchScientificLineageManifestV2.from_mapping(
            scientific_lineage_manifest.to_dict()
        )
        plan_snapshot = ResearchReadinessPlanV1.from_wire_bytes(
            readiness_plan.to_wire_bytes()
        )
        spec_snapshot = _snapshot_readiness_spec(readiness_spec)
        _validate_scientific_bindings(
            run_spec=run_snapshot,
            scientific_lineage_manifest=lineage_snapshot,
            readiness_spec=spec_snapshot,
            readiness_plan=plan_snapshot,
        )
        _validate_implementation_identity(implementation_id, implementation_hash)
        object.__setattr__(self, "_run_spec", run_snapshot)
        object.__setattr__(self, "_lineage", lineage_snapshot)
        object.__setattr__(self, "_readiness_spec", spec_snapshot)
        object.__setattr__(self, "_readiness_plan", plan_snapshot)
        object.__setattr__(self, "_implementation_id", implementation_id)
        object.__setattr__(self, "_implementation_hash", implementation_hash)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("ResolvedResearchReadinessFactory is immutable")

    @property
    def readiness_spec_hash(self) -> str:
        return cast(str, self._readiness_spec.content_hash)

    @property
    def readiness_plan_hash(self) -> str:
        return cast(str, self._readiness_plan.content_hash)

    def bind(
        self,
        *,
        readiness_plan: ResearchReadinessPlanV1,
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
        if type(readiness_plan) is not ResearchReadinessPlanV1:
            raise TypeError("readiness bound plan differs")
        if type(run_spec) is not ResearchRunSpec:
            raise TypeError("readiness bound run specification differs")
        if type(scientific_lineage_manifest) is not ResearchScientificLineageManifestV2:
            raise TypeError("readiness bound scientific lineage differs")
        if type(experiment_spec) is not ExperimentSpec:
            raise TypeError("readiness experiment specification differs")
        if type(registry) is not ExperimentRegistry:
            raise TypeError("readiness registry authority differs")
        if type(runtime) is not ExperimentRuntime:
            raise TypeError("readiness runtime authority differs")
        if type(artifact_store) is not ArtifactStore:
            raise TypeError("readiness artifact store authority differs")
        if type(outer_audit_reader) is not NestedOuterEvaluationAuditReader:
            raise TypeError("readiness outer audit authority differs")
        if clock is not None and not callable(clock):
            raise TypeError("readiness clock must be callable or None")
        _validate_worker_identity(worker_id, lease_token)
        if readiness_plan.content_hash != self._readiness_plan.content_hash:
            raise ValueError("readiness bound plan differs")
        if run_spec.content_hash != self._run_spec.content_hash:
            raise ValueError("readiness bound run specification differs")
        if scientific_lineage_manifest.content_hash != self._lineage.content_hash:
            raise ValueError("readiness bound scientific lineage differs")
        if runtime.spec.content_hash != experiment_spec.content_hash:
            raise ValueError("readiness runtime experiment differs")
        return _ResolvedResearchReadinessImplementation(
            _token=_BOUND_STAGE_TOKEN,
            run_spec=ResearchRunSpec.from_mapping(self._run_spec.to_dict()),
            scientific_lineage_manifest=(
                ResearchScientificLineageManifestV2.from_mapping(
                    self._lineage.to_dict()
                )
            ),
            readiness_spec=_snapshot_readiness_spec(self._readiness_spec),
            readiness_plan=ResearchReadinessPlanV1.from_wire_bytes(
                self._readiness_plan.to_wire_bytes()
            ),
            implementation_id=self._implementation_id,
            implementation_hash=self._implementation_hash,
            experiment_spec_hash=experiment_spec.content_hash,
            registry=registry,
            runtime=runtime,
            artifact_store=artifact_store,
            outer_audit_reader=outer_audit_reader,
            worker_id=worker_id,
            lease_token=lease_token,
            clock=clock,
        )


@final
class _ResolvedResearchReadinessImplementation:
    """Private immutable stage implementation constructible only by the factory."""

    __slots__ = (
        "_artifact_store",
        "_clock",
        "_experiment_spec_hash",
        "_implementation_hash",
        "_implementation_id",
        "_lineage",
        "_lease_token",
        "_outer_audit_reader",
        "_readiness_plan",
        "_readiness_spec",
        "_registry",
        "_run_spec",
        "_runtime",
        "_worker_id",
    )

    _artifact_store: ArtifactStore
    _clock: Callable[[], datetime] | None
    _experiment_spec_hash: str
    _implementation_hash: str
    _implementation_id: str
    _lineage: ResearchScientificLineageManifestV2
    _lease_token: str
    _outer_audit_reader: NestedOuterEvaluationAuditReader
    _readiness_plan: ResearchReadinessPlanV1
    _readiness_spec: ResearchReadinessSpec
    _registry: ExperimentRegistry
    _run_spec: ResearchRunSpec
    _runtime: ExperimentRuntime
    _worker_id: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("resolved research readiness stage cannot be subclassed")

    def __init__(
        self,
        *,
        _token: object,
        run_spec: ResearchRunSpec,
        scientific_lineage_manifest: ResearchScientificLineageManifestV2,
        readiness_spec: ResearchReadinessSpec,
        readiness_plan: ResearchReadinessPlanV1,
        implementation_id: str,
        implementation_hash: str,
        experiment_spec_hash: str,
        registry: ExperimentRegistry,
        runtime: ExperimentRuntime,
        artifact_store: ArtifactStore,
        outer_audit_reader: NestedOuterEvaluationAuditReader,
        worker_id: str,
        lease_token: str,
        clock: Callable[[], datetime] | None,
    ) -> None:
        if _token is not _BOUND_STAGE_TOKEN:
            raise TypeError("readiness stage requires factory authority")
        object.__setattr__(self, "_run_spec", run_spec)
        object.__setattr__(self, "_lineage", scientific_lineage_manifest)
        object.__setattr__(self, "_readiness_spec", readiness_spec)
        object.__setattr__(self, "_readiness_plan", readiness_plan)
        object.__setattr__(self, "_implementation_id", implementation_id)
        object.__setattr__(self, "_implementation_hash", implementation_hash)
        object.__setattr__(
            self,
            "_experiment_spec_hash",
            require_sha256(
                experiment_spec_hash,
                name="readiness experiment specification hash",
            ),
        )
        object.__setattr__(self, "_registry", registry)
        object.__setattr__(self, "_runtime", runtime)
        object.__setattr__(self, "_artifact_store", artifact_store)
        object.__setattr__(self, "_outer_audit_reader", outer_audit_reader)
        object.__setattr__(self, "_worker_id", worker_id)
        object.__setattr__(self, "_lease_token", lease_token)
        object.__setattr__(self, "_clock", clock)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("resolved research readiness stage is immutable")

    def __call__(self, request: ScientificStageRequest) -> ScientificStageResult:
        _validate_request(
            request,
            research_run_spec_hash=self._run_spec.content_hash,
            experiment_spec_hash=self._experiment_spec_hash,
        )
        started_wall = time.perf_counter()
        started_cpu = time.process_time()
        resolver = ResearchReadinessInputResolver(
            registry=self._registry,
            artifact_store=self._artifact_store,
            runtime=self._runtime,
            outer_audit_reader=self._outer_audit_reader,
            worker_id=self._worker_id,
            lease_token=self._lease_token,
            clock=self._clock,
        )
        try:
            manifest = resolver.assemble_manifest(
                request=request,
                run_spec=self._run_spec,
                scientific_lineage_manifest=self._lineage,
                readiness_plan=self._readiness_plan,
            )
            authority_bound = resolver.resolve(
                manifest=manifest,
                request=request,
                run_spec=self._run_spec,
                scientific_lineage_manifest=self._lineage,
            )
        except ResearchReadinessInputError as exc:
            raise _stage_error(
                f"input_{exc.code}", "readiness input authority resolution failed"
            ) from exc

        loaded = authority_bound.loaded
        receipt = authority_bound.authority_receipt
        try:
            bundle = ResearchReadinessRunner().run(
                self._readiness_spec,
                candidate_evidence_hash=(
                    manifest.score_artifacts.candidate_scores.frame_hash
                ),
                baseline_evidence_hash=(
                    manifest.score_artifacts.baseline_scores.frame_hash
                ),
                outer_validation_receipt_hash=(
                    manifest.outer_validation_receipt_hash
                ),
                candidate_scores=loaded.candidate_scores,
                baseline_scores=loaded.baseline_scores,
                labels=loaded.labels,
                label_validity=loaded.label_validity,
                scoring_eligibility=loaded.scoring_eligibility,
                perturbation_scores=loaded.perturbation_scores,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise _stage_error(
                "readiness_computation_failed",
                "research readiness computation failed",
            ) from exc

        try:
            document, document_record = ResearchReadinessBundleProducerV1(
                self._artifact_store
            ).publish(bundle, authority_bound=authority_bound)
            loader = ResearchReadinessBundleLoaderV1(self._artifact_store)
            restored_document = loader.load_document(document_record)
            # Rebuild the bundle from the already authenticated document.  A
            # second load_bundle(record) would reopen the same document bytes
            # without adding authority or replay coverage.
            restored_bundle = loader.rebuild(restored_document)
            if (
                restored_document != document
                or restored_document.content_hash != document.content_hash
                or restored_bundle.content_hash != bundle.content_hash
            ):
                raise _stage_error(
                    "readiness_replay_identity_mismatch",
                    "replayed readiness evidence differs",
                )
        except ResearchReadinessStageError:
            raise
        except ResearchReadinessBundleDocumentError as exc:
            raise _stage_error(
                f"bundle_{exc.code}", "readiness bundle persistence or replay failed"
            ) from exc
        except (ArtifactError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise _stage_error(
                "readiness_bundle_replay_failed",
                "readiness bundle persistence or replay failed",
            ) from exc

        verdict = restored_bundle.verdict
        if not isinstance(verdict, DLReadinessVerdict):  # pragma: no cover
            raise _stage_error("invalid_readiness_decision", "readiness verdict differs")
        manifest_reference = restored_document.input_manifest_reference
        receipt_reference = restored_document.authority_receipt_reference
        manifest_record = manifest_reference.artifact
        receipt_record = receipt_reference.artifact
        payload: dict[str, JsonValue] = {
            "readiness_decision": _DECISION_CODES[verdict],
            "decision_scope": _DECISION_SCOPE,
            "economic_backtest_assessed": False,
            "backtest_dependency_role": _BACKTEST_DEPENDENCY_ROLE,
            "readiness_bundle_hash": restored_bundle.content_hash,
            "readiness_bundle_document_hash": restored_document.content_hash,
            "readiness_input_manifest_hash": manifest.content_hash,
            "readiness_authority_receipt_hash": receipt.content_hash,
            "research_only": True,
            "production_ready": False,
        }
        evidence = tuple(
            dict.fromkeys(
                (
                    manifest.content_hash,
                    manifest_reference.reference_hash,
                    manifest_record.sha256,
                    receipt.content_hash,
                    receipt_reference.reference_hash,
                    receipt_record.sha256,
                    authority_bound.content_hash,
                    restored_bundle.content_hash,
                    restored_document.content_hash,
                    document_record.sha256,
                    *(
                        reference.content_hash
                        for reference in restored_document.evidence_artifacts.values()
                    ),
                    *(
                        reference.artifact.sha256
                        for reference in restored_document.evidence_artifacts.values()
                    ),
                )
            )
        )
        output_records = (
            manifest_record,
            receipt_record,
            document_record,
            *(
                reference.artifact
                for reference in restored_document.evidence_artifacts.values()
            ),
        )
        derived_outer_write_bytes = sum(
            reference.size_bytes
            for reference in (
                manifest.label_values,
                manifest.label_validity,
                manifest.scoring_eligibility,
            )
        )
        score_frame_bytes = sum(
            reference.size_bytes
            for reference in (
                manifest.score_artifacts.candidate_scores,
                manifest.score_artifacts.baseline_scores,
                *manifest.score_artifacts.perturbation_scores.values(),
            )
        )
        output_replay_bytes = sum(record.size_bytes for record in output_records)
        # These are governed, content-addressed scientific-artifact bytes, not
        # an OS-level I/O counter.  Registry/SQLite and other small metadata I/O
        # are intentionally outside this explicit accounting boundary.  SCORE
        # frames are reopened once during assembly and once during resolution;
        # derived frames are replayed after publication; and the final input set
        # is reloaded by both the resolver and the bundle replay.
        governed_scientific_artifact_read_bytes = (
            receipt.authoritative_frame_read_bytes
            + 2 * score_frame_bytes
            + derived_outer_write_bytes
            + 2 * receipt.total_compressed_bytes
            + output_replay_bytes
        )
        _revalidate_resolved_authority(
            resolver=resolver,
            receipt=receipt,
            request=request,
            experiment_spec_hash=self._experiment_spec_hash,
        )
        return ScientificStageResult(
            stage=ResearchStage.ROBUSTNESS,
            implementation_id=self._implementation_id,
            implementation_hash=self._implementation_hash,
            result_payload=payload,
            evidence_hashes=evidence,
            sanitized_feedback_codes=restored_bundle.reason_codes,
            usage=reported_usage(
                wall_seconds=max(0.0, time.perf_counter() - started_wall),
                cpu_seconds=max(0.0, time.process_time() - started_cpu),
                peak_memory_bytes=_peak_rss_bytes(),
                disk_write_bytes=(
                    derived_outer_write_bytes + output_replay_bytes
                ),
            ),
            input_rows=len(loaded.labels.index),
            output_rows=1,
            symbols=len(loaded.labels.columns),
            disk_read_bytes=governed_scientific_artifact_read_bytes,
            cache_hits=0,
            cache_misses=0,
            worker_count=1,
        )


def _revalidate_resolved_authority(
    *,
    resolver: ResearchReadinessInputResolver,
    receipt: ResearchReadinessAuthorityReceiptV1,
    request: ScientificStageRequest,
    experiment_spec_hash: str,
) -> None:
    """Recheck the exact resolved attempt immediately before result release."""

    if (
        receipt.experiment_spec_hash != experiment_spec_hash
        or receipt.current_attempt_id != request.attempt_id
        or receipt.current_attempt_number != request.attempt_number
    ):
        raise _stage_error(
            "final_authority_binding_mismatch",
            "resolved readiness authority differs from the bound experiment",
        )
    authority = _RuntimeAuthority(
        parents={},
        current_attempt_id=receipt.current_attempt_id,
        current_attempt_number=receipt.current_attempt_number,
        current_attempt_input_hash=receipt.current_attempt_input_hash,
        current_worker_id=resolver._worker_id,
        current_lease_token_hash=resolver._lease_token_hash,
    )
    try:
        resolver._revalidate_current_authority(
            experiment_spec_hash=experiment_spec_hash,
            authority=authority,
        )
    except ResearchReadinessInputError as exc:
        raise _stage_error(
            "final_authority_revalidation_failed",
            "readiness authority changed before result release",
        ) from exc


def _snapshot_readiness_spec(spec: ResearchReadinessSpec) -> ResearchReadinessSpec:
    return ResearchReadinessSpec(
        readiness_id=spec.readiness_id,
        version=spec.version,
        pseudo_label_trials=spec.pseudo_label_trials,
        random_seed=spec.random_seed,
        signal_lag_sessions=tuple(spec.signal_lag_sessions),
        required_perturbation_names=tuple(spec.required_perturbation_names),
        minimum_cross_sectional_observations=(
            spec.minimum_cross_sectional_observations
        ),
        minimum_valid_dates=spec.minimum_valid_dates,
        maximum_pseudo_label_pvalue=spec.maximum_pseudo_label_pvalue,
        minimum_observed_rank_ic=spec.minimum_observed_rank_ic,
        minimum_lag_absolute_gap=spec.minimum_lag_absolute_gap,
        minimum_stability_retention=spec.minimum_stability_retention,
        minimum_incremental_rank_ic=spec.minimum_incremental_rank_ic,
        minimum_incremental_positive_date_fraction=(
            spec.minimum_incremental_positive_date_fraction
        ),
    )


def _validate_scientific_bindings(
    *,
    run_spec: ResearchRunSpec,
    scientific_lineage_manifest: ResearchScientificLineageManifestV2,
    readiness_spec: ResearchReadinessSpec,
    readiness_plan: ResearchReadinessPlanV1,
) -> None:
    scientific_lineage_manifest.validate_for(run_spec)
    if run_spec.robustness_spec_hash != readiness_spec.content_hash:
        raise ValueError("readiness specification differs from the run contract")
    if readiness_plan.readiness_spec_hash != readiness_spec.content_hash:
        raise ValueError("readiness plan specification differs")
    if readiness_plan.validation_spec_hash != run_spec.validation_spec_hash:
        raise ValueError("readiness plan validation specification differs")
    if (
        readiness_plan.nested_selection_spec_hash
        != scientific_lineage_manifest.nested_selection_spec_hash
    ):
        raise ValueError("readiness plan nested selection differs")
    if tuple(readiness_plan.perturbation_definition_hashes) != (
        readiness_spec.required_perturbation_names
    ):
        raise ValueError("readiness plan perturbation contract differs")


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
        raise ValueError("readiness implementation_id is invalid")
    if implementation_id.lower() in {"noop", "placeholder", "stub", "mock"}:
        raise ValueError("placeholder readiness implementation is forbidden")
    require_sha256(implementation_hash, name="readiness implementation_hash")


def _validate_worker_identity(worker_id: object, lease_token: object) -> None:
    if not isinstance(worker_id, str) or not 1 <= len(worker_id) <= 128:
        raise TypeError("readiness worker_id is invalid")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-")
    if not worker_id[0].isalnum() or any(
        character not in allowed for character in worker_id
    ):
        raise ValueError("readiness worker_id is invalid")
    if not isinstance(lease_token, str) or len(lease_token) < 16:
        raise ValueError("readiness lease_token must contain at least 16 characters")


def _validate_request(
    request: ScientificStageRequest,
    *,
    research_run_spec_hash: str,
    experiment_spec_hash: str,
) -> None:
    if type(request) is not ScientificStageRequest:
        raise _stage_error("request_binding_mismatch", "readiness request type differs")
    if ResearchStage(request.stage) is not ResearchStage.ROBUSTNESS:
        raise _stage_error("request_binding_mismatch", "readiness request stage differs")
    if request.research_run_spec_hash != research_run_spec_hash:
        raise _stage_error("request_binding_mismatch", "readiness request run differs")
    if request.experiment_spec_hash != experiment_spec_hash:
        raise _stage_error(
            "request_binding_mismatch", "readiness request experiment differs"
        )


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


__all__ = [
    "ResearchReadinessStageError",
    "ResolvedResearchReadinessFactory",
]
