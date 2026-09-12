from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from alpha_research.core.hashing import hash_json, require_sha256
from alpha_research.evaluation import (
    EvaluationSpec,
    FactorEvaluationReport,
    FactorEvaluationSuite,
)
from alpha_research.experiments import DataPartition
from alpha_research.factors import FactorEngine, FactorResult, FactorSpec
from alpha_research.factors.view import FactorDataView
from alpha_research.high_frequency.provider_semantics import (
    IntendedUse,
    ProviderSemanticCatalog,
)
from alpha_research.labels import LabelResult, LabelSpec, LabelTask
from alpha_research.market_logic.compiler import MarketLogicCompiler
from alpha_research.market_logic.audit_artifact import (
    ValidationAuditArtifactReceipt,
    ValidationEvaluationAuditStore,
)
from alpha_research.market_logic.evidence import (
    CostResilienceCategory,
    CoverageCategory,
    EvidenceReasonCode,
    EvidenceVerdict,
    StabilityCategory,
)
from alpha_research.market_logic.evidence_adapter import (
    EvidenceClassificationPolicy,
    LocalAdaptiveValidationEvidenceAdapter,
)
from alpha_research.market_logic.factor_binding import (
    FactorExecutionPreflightReceipt,
    LogicFactorBindingReceipt,
    MarketLogicFactorBinder,
)
from alpha_research.market_logic.registry import (
    LogicFactorBinding,
    MarketLogicRegistry,
)
from alpha_research.market_logic.spec import HorizonUnit, MarketLogicSpec
from alpha_research.validation import ValidationReceipt
from factor_production.v5.domain.enums import ScoreVisibility
from factor_production.v5.dsl import OperatorRegistry


_PORTABLE_ID = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_PORTABLE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")


class WorkflowFailure(str, Enum):
    """Closed failure codes that cannot reveal validation metric values."""

    INVALID_INPUT = "invalid_input"
    LINEAGE_FAILED = "lineage_failed"
    LOGIC_COMPILATION_FAILED = "logic_compilation_failed"
    FACTOR_BINDING_FAILED = "factor_binding_failed"
    FACTOR_EXECUTION_FAILED = "factor_execution_failed"
    EVALUATION_FAILED = "evaluation_failed"
    AUDIT_PERSISTENCE_FAILED = "audit_persistence_failed"
    EVIDENCE_ADAPTATION_FAILED = "evidence_adaptation_failed"
    REGISTRY_WRITE_FAILED = "registry_write_failed"


class LogicValidationWorkflowError(RuntimeError):
    """Fail-closed workflow error whose message is a metric-free code."""

    def __init__(self, code: WorkflowFailure) -> None:
        self.code = WorkflowFailure(code)
        super().__init__(self.code.value)


class DataReadiness(str, Enum):
    STRICT_PIT_INPUT_READY = "strict_pit_input_ready"


class AuditArtifactStatus(str, Enum):
    """Whether the sealed exact report was persisted by this workflow slice."""

    NOT_PERSISTED = "not_persisted"
    PERSISTED = "persisted"


class RegistrationStatus(str, Enum):
    REGISTERED = "registered"


@dataclass(frozen=True, slots=True)
class LogicValidationWorkflowSpec:
    """Governance inputs for one local, adaptive-validation experiment.

    This workflow deliberately has no research/degraded execution option.  A
    caller may inspect degraded data elsewhere, but it cannot create a
    retrieval-visible market-logic result through this path.
    """

    workflow_id: str
    version: str
    fold_id: str
    source_partition: DataPartition | str = DataPartition.VALIDATION
    score_visibility: ScoreVisibility | str = ScoreVisibility.LOCAL_RESEARCH
    intended_use: IntendedUse | str = IntendedUse.RESEARCH
    schema_version: str = "single-market-logic-validation-workflow/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "single-market-logic-validation-workflow/v1":
            raise ValueError("unsupported logic-validation workflow schema")
        if not _PORTABLE_ID.fullmatch(self.workflow_id):
            raise ValueError("invalid logic-validation workflow identifier")
        if not _PORTABLE_VERSION.fullmatch(self.version):
            raise ValueError("invalid logic-validation workflow version")
        if not self.fold_id.strip() or len(self.fold_id) > 128:
            raise ValueError("logic-validation fold identifier is invalid")
        partition = DataPartition(self.source_partition)
        visibility = ScoreVisibility(self.score_visibility)
        intended_use = IntendedUse(self.intended_use)
        if partition is not DataPartition.VALIDATION:
            raise ValueError("logic validation accepts only validation partition")
        if visibility is not ScoreVisibility.LOCAL_RESEARCH:
            raise ValueError("logic validation accepts only local-research visibility")
        if intended_use is not IntendedUse.RESEARCH:
            raise ValueError("logic validation is a governed research workflow")
        object.__setattr__(self, "source_partition", partition)
        object.__setattr__(self, "score_visibility", visibility)
        object.__setattr__(self, "intended_use", intended_use)

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "workflow_id": self.workflow_id,
            "version": self.version,
            "fold_id": self.fold_id,
            "source_partition": DataPartition(self.source_partition).value,
            "score_visibility": ScoreVisibility(self.score_visibility).value,
            "intended_use": IntendedUse(self.intended_use).value,
            "require_point_in_time": True,
            "require_production_ready": True,
        }


@dataclass(frozen=True, slots=True)
class LogicFactorCandidateIdentity:
    """Recomputable identity of the factor implementation of one logic.

    ``candidate_hash`` is not a score or a loose proposal identifier.  It is
    the hash of the exact logic, FactorSpec, and trusted binding receipt.
    """

    market_logic_hash: str
    factor_spec_hash: str
    logic_factor_binding_hash: str
    schema_version: str = "logic-factor-candidate-identity/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "logic-factor-candidate-identity/v1":
            raise ValueError("unsupported logic-factor candidate identity schema")
        for name in (
            "market_logic_hash",
            "factor_spec_hash",
            "logic_factor_binding_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"candidate {name}")

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "market_logic_hash": self.market_logic_hash,
            "factor_spec_hash": self.factor_spec_hash,
            "logic_factor_binding_hash": self.logic_factor_binding_hash,
        }


@dataclass(frozen=True, slots=True)
class LogicValidationExperimentIdentity:
    """Recomputable, result-free identity of one validation experiment.

    The identity binds all inputs and the realized factor signal, but no IC,
    return, date-level result, or categorical verdict.  Reclassification under
    another evidence policy therefore preserves experiment identity while
    producing a different evidence derivation hash.
    """

    workflow_spec_hash: str
    candidate_hash: str
    factor_preflight_hash: str
    factor_view_hash: str
    factor_signal_hash: str
    label_spec_hash: str
    label_result_hash: str
    validation_receipt_hash: str
    evaluation_spec_hash: str
    fold_id: str
    schema_version: str = "single-market-logic-validation-experiment/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "single-market-logic-validation-experiment/v1":
            raise ValueError("unsupported logic-validation experiment identity schema")
        for name in (
            "workflow_spec_hash",
            "candidate_hash",
            "factor_preflight_hash",
            "factor_view_hash",
            "factor_signal_hash",
            "label_spec_hash",
            "label_result_hash",
            "validation_receipt_hash",
            "evaluation_spec_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"experiment {name}")
        if not self.fold_id.strip() or len(self.fold_id) > 128:
            raise ValueError("experiment fold identifier is invalid")

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "workflow_spec_hash": self.workflow_spec_hash,
            "candidate_hash": self.candidate_hash,
            "factor_preflight_hash": self.factor_preflight_hash,
            "factor_view_hash": self.factor_view_hash,
            "factor_signal_hash": self.factor_signal_hash,
            "label_spec_hash": self.label_spec_hash,
            "label_result_hash": self.label_result_hash,
            "validation_receipt_hash": self.validation_receipt_hash,
            "evaluation_spec_hash": self.evaluation_spec_hash,
            "fold_id": self.fold_id,
        }


@dataclass(frozen=True, slots=True)
class LogicValidationWorkflowReceipt:
    """Public, metric-free outcome of a completed validation workflow.

    ``evaluation_artifact_hash`` addresses the exact report used for
    classification.  Version 1 preserves the historical in-memory-only
    receipt.  Version 2 additionally binds the write-once audit-store receipt,
    physical artifact digest, safe relative location, and size.  Neither
    version exposes evaluation metrics.
    """

    workflow_spec_hash: str
    market_logic_hash: str
    logic_constraint_hash: str
    factor_spec_hash: str
    factor_view_hash: str
    logic_factor_binding_hash: str
    factor_preflight_hash: str
    candidate_hash: str
    factor_signal_hash: str
    label_spec_hash: str
    label_result_hash: str
    validation_receipt_hash: str
    evaluation_spec_hash: str
    experiment_hash: str
    evaluation_artifact_hash: str
    evidence_artifact_hash: str
    registry_binding_hash: str
    registry_evidence_hash: str
    data_readiness: DataReadiness | str
    audit_artifact_status: AuditArtifactStatus | str
    registration_status: RegistrationStatus | str
    verdict: EvidenceVerdict | str
    directional_stability: StabilityCategory | str
    regime_stability: StabilityCategory | str
    cost_resilience: CostResilienceCategory | str
    coverage: CoverageCategory | str
    reason_codes: tuple[EvidenceReasonCode | str, ...]
    audit_artifact_receipt_hash: str | None = None
    audit_artifact_file_hash: str | None = None
    audit_artifact_binding_hash: str | None = None
    audit_artifact_location: str | None = None
    audit_artifact_size_bytes: int | None = None
    schema_version: str = "logic-validation-workflow-receipt/v1"

    def __post_init__(self) -> None:
        if self.schema_version not in {
            "logic-validation-workflow-receipt/v1",
            "logic-validation-workflow-receipt/v2",
        }:
            raise ValueError("unsupported logic-validation workflow receipt schema")
        for name in (
            "workflow_spec_hash",
            "market_logic_hash",
            "logic_constraint_hash",
            "factor_spec_hash",
            "factor_view_hash",
            "logic_factor_binding_hash",
            "factor_preflight_hash",
            "candidate_hash",
            "factor_signal_hash",
            "label_spec_hash",
            "label_result_hash",
            "validation_receipt_hash",
            "evaluation_spec_hash",
            "experiment_hash",
            "evaluation_artifact_hash",
            "evidence_artifact_hash",
            "registry_binding_hash",
            "registry_evidence_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"workflow receipt {name}")
        reasons = tuple(
            sorted(
                (EvidenceReasonCode(item) for item in self.reason_codes),
                key=lambda item: item.value,
            )
        )
        if not reasons or len(reasons) != len(set(reasons)):
            raise ValueError(
                "workflow receipt reason codes must be non-empty and unique"
            )
        object.__setattr__(self, "data_readiness", DataReadiness(self.data_readiness))
        object.__setattr__(
            self,
            "audit_artifact_status",
            AuditArtifactStatus(self.audit_artifact_status),
        )
        object.__setattr__(
            self, "registration_status", RegistrationStatus(self.registration_status)
        )
        object.__setattr__(self, "verdict", EvidenceVerdict(self.verdict))
        object.__setattr__(
            self, "directional_stability", StabilityCategory(self.directional_stability)
        )
        object.__setattr__(
            self, "regime_stability", StabilityCategory(self.regime_stability)
        )
        object.__setattr__(
            self, "cost_resilience", CostResilienceCategory(self.cost_resilience)
        )
        object.__setattr__(self, "coverage", CoverageCategory(self.coverage))
        object.__setattr__(self, "reason_codes", reasons)
        audit_status = AuditArtifactStatus(self.audit_artifact_status)
        audit_values = (
            self.audit_artifact_receipt_hash,
            self.audit_artifact_file_hash,
            self.audit_artifact_binding_hash,
            self.audit_artifact_location,
            self.audit_artifact_size_bytes,
        )
        if audit_status is AuditArtifactStatus.NOT_PERSISTED:
            if self.schema_version != "logic-validation-workflow-receipt/v1":
                raise ValueError("non-persisted workflow receipt must use v1")
            if any(item is not None for item in audit_values):
                raise ValueError(
                    "non-persisted workflow receipt has audit artifact fields"
                )
        else:
            if self.schema_version != "logic-validation-workflow-receipt/v2":
                raise ValueError("persisted workflow receipt must use v2")
            if any(item is None for item in audit_values):
                raise ValueError(
                    "persisted workflow receipt lacks audit artifact fields"
                )
            for name in (
                "audit_artifact_receipt_hash",
                "audit_artifact_file_hash",
                "audit_artifact_binding_hash",
            ):
                require_sha256(
                    str(getattr(self, name)), name=f"workflow receipt {name}"
                )
            if (
                not isinstance(self.audit_artifact_location, str)
                or not self.audit_artifact_location.strip()
                or self.audit_artifact_location.startswith("/")
                or ".." in self.audit_artifact_location.split("/")
            ):
                raise ValueError("persisted workflow audit location is unsafe")
            if (
                not isinstance(self.audit_artifact_size_bytes, int)
                or isinstance(self.audit_artifact_size_bytes, bool)
                or self.audit_artifact_size_bytes <= 0
            ):
                raise ValueError("persisted workflow audit size is invalid")
            try:
                audit_receipt = ValidationAuditArtifactReceipt(
                    evaluation_report_hash=self.evaluation_artifact_hash,
                    artifact_hash=str(self.audit_artifact_file_hash),
                    binding_hash=str(self.audit_artifact_binding_hash),
                    experiment_hash=self.experiment_hash,
                    logic_hash=self.market_logic_hash,
                    relative_location=str(self.audit_artifact_location),
                    size_bytes=self.audit_artifact_size_bytes,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "persisted workflow audit artifact binding is invalid"
                ) from exc
            if audit_receipt.content_hash != self.audit_artifact_receipt_hash:
                raise ValueError(
                    "persisted workflow audit artifact receipt hash differs"
                )

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "workflow_spec_hash": self.workflow_spec_hash,
            "market_logic_hash": self.market_logic_hash,
            "logic_constraint_hash": self.logic_constraint_hash,
            "factor_spec_hash": self.factor_spec_hash,
            "factor_view_hash": self.factor_view_hash,
            "logic_factor_binding_hash": self.logic_factor_binding_hash,
            "factor_preflight_hash": self.factor_preflight_hash,
            "candidate_hash": self.candidate_hash,
            "factor_signal_hash": self.factor_signal_hash,
            "label_spec_hash": self.label_spec_hash,
            "label_result_hash": self.label_result_hash,
            "validation_receipt_hash": self.validation_receipt_hash,
            "evaluation_spec_hash": self.evaluation_spec_hash,
            "experiment_hash": self.experiment_hash,
            "evaluation_artifact_hash": self.evaluation_artifact_hash,
            "evidence_artifact_hash": self.evidence_artifact_hash,
            "registry_binding_hash": self.registry_binding_hash,
            "registry_evidence_hash": self.registry_evidence_hash,
            "data_readiness": DataReadiness(self.data_readiness).value,
            "audit_artifact_status": AuditArtifactStatus(
                self.audit_artifact_status
            ).value,
            "registration_status": RegistrationStatus(self.registration_status).value,
            "verdict": EvidenceVerdict(self.verdict).value,
            "directional_stability": StabilityCategory(
                self.directional_stability
            ).value,
            "regime_stability": StabilityCategory(self.regime_stability).value,
            "cost_resilience": CostResilienceCategory(self.cost_resilience).value,
            "coverage": CoverageCategory(self.coverage).value,
            "reason_codes": [
                EvidenceReasonCode(item).value for item in self.reason_codes
            ],
        }
        if self.schema_version == "logic-validation-workflow-receipt/v2":
            payload["audit_artifact_receipt_hash"] = self.audit_artifact_receipt_hash
            payload["audit_artifact_file_hash"] = self.audit_artifact_file_hash
            payload["audit_artifact_binding_hash"] = self.audit_artifact_binding_hash
            payload["audit_artifact_location"] = self.audit_artifact_location
            payload["audit_artifact_size_bytes"] = self.audit_artifact_size_bytes
        return payload


class SingleMarketLogicValidationWorkflow:
    """Strict MarketLogic -> Factor -> validation evidence -> registry path.

    Exact evaluation frames and numbers never enter the public receipt or the
    research-memory registry.  All computation and evidence adaptation finish
    before the first append.  Registry writes then follow the explicit,
    retry-safe order ``logic -> binding -> evidence``; each registry method is
    individually transactional and content addressed.
    """

    def run(
        self,
        *,
        workflow_spec: LogicValidationWorkflowSpec,
        logic: MarketLogicSpec,
        factor: FactorSpec,
        factor_view: FactorDataView,
        label_spec: LabelSpec,
        labels: LabelResult,
        validation: ValidationReceipt,
        evaluation_spec: EvaluationSpec,
        compiler: MarketLogicCompiler,
        semantic_catalog: ProviderSemanticCatalog,
        evidence_policy: EvidenceClassificationPolicy,
        registry: MarketLogicRegistry,
        audit_store: ValidationEvaluationAuditStore | None = None,
        registry_parent_logic_hash: str | None = None,
    ) -> LogicValidationWorkflowReceipt:
        self._validate_types(
            workflow_spec=workflow_spec,
            logic=logic,
            factor=factor,
            factor_view=factor_view,
            label_spec=label_spec,
            labels=labels,
            validation=validation,
            evaluation_spec=evaluation_spec,
            compiler=compiler,
            semantic_catalog=semantic_catalog,
            evidence_policy=evidence_policy,
            registry=registry,
        )
        if audit_store is not None and not isinstance(
            audit_store, ValidationEvaluationAuditStore
        ):
            raise LogicValidationWorkflowError(WorkflowFailure.INVALID_INPUT)
        if (
            registry_parent_logic_hash is not None
            or logic.provenance.parent_logic_hashes
            or factor.provenance.parent_factor_hashes
        ):
            # The current registry models a single version-parent edge, while
            # evolutionary provenance may be many-to-many.  Until an explicit
            # parent-logic -> parent-factor mapping contract exists, accepting
            # either side would turn the binder's shape check into false
            # lineage assurance.  This workflow is genesis-only by design.
            raise LogicValidationWorkflowError(WorkflowFailure.LINEAGE_FAILED)
        try:
            self._validate_label_lineage(
                logic=logic,
                factor=factor,
                factor_view=factor_view,
                label_spec=label_spec,
                labels=labels,
                validation=validation,
                workflow_spec=workflow_spec,
            )
        except Exception:
            raise LogicValidationWorkflowError(WorkflowFailure.LINEAGE_FAILED) from None

        try:
            constraint = compiler.compile(
                logic,
                semantic_catalog=semantic_catalog,
                intended_use=workflow_spec.intended_use,
            )
        except Exception:
            raise LogicValidationWorkflowError(
                WorkflowFailure.LOGIC_COMPILATION_FAILED
            ) from None

        binder = MarketLogicFactorBinder()
        try:
            runtime_registry = OperatorRegistry.dataframe_pit_v3(
                factor_view.cross_section_mask,
                factor_view.history_mask,
            )
            initial_binding = binder.bind(
                logic=logic,
                constraint=constraint,
                factor=factor,
                compiler_policy=compiler.policy,
                semantic_catalog=semantic_catalog,
                data_profile_hash=logic.data_profile_hash,
                provider_regime_hashes=logic.eligible_provider_regime_hashes,
                operator_registry=runtime_registry,
                available_fields=factor_view.fields,
                intended_use=workflow_spec.intended_use,
            )
            binding, preflight = binder.preflight(
                logic=logic,
                constraint=constraint,
                factor=factor,
                compiler_policy=compiler.policy,
                semantic_catalog=semantic_catalog,
                data_profile_hash=logic.data_profile_hash,
                provider_regime_hashes=logic.eligible_provider_regime_hashes,
                view=factor_view,
                dataset_id=factor.dataset_id,
                intended_use=workflow_spec.intended_use,
                require_point_in_time=True,
                require_production_ready=True,
                expected_binding_hash=initial_binding.content_hash,
            )
        except Exception:
            raise LogicValidationWorkflowError(
                WorkflowFailure.FACTOR_BINDING_FAILED
            ) from None

        try:
            factor_result = FactorEngine(require_point_in_time=True).evaluate(
                factor, factor_view
            )
            if not factor_result.admission_eligible:
                raise ValueError("strict factor execution is not admission eligible")
        except Exception:
            raise LogicValidationWorkflowError(
                WorkflowFailure.FACTOR_EXECUTION_FAILED
            ) from None

        candidate = _candidate_identity(logic, factor, binding)
        experiment = _experiment_identity(
            workflow_spec=workflow_spec,
            candidate=candidate,
            preflight=preflight,
            factor_result=factor_result,
            label_spec=label_spec,
            labels=labels,
            validation=validation,
            evaluation_spec=evaluation_spec,
        )
        try:
            report = FactorEvaluationSuite().evaluate(
                evaluation_spec,
                factor,
                factor_result,
                label_spec,
                labels,
                validation,
                fold_id=workflow_spec.fold_id,
            )
            _verify_report_lineage(
                report=report,
                evaluation_spec=evaluation_spec,
                factor=factor,
                factor_result=factor_result,
                label_spec=label_spec,
                labels=labels,
                validation=validation,
                fold_id=workflow_spec.fold_id,
            )
        except Exception:
            raise LogicValidationWorkflowError(
                WorkflowFailure.EVALUATION_FAILED
            ) from None

        audit_receipt: ValidationAuditArtifactReceipt | None = None
        if audit_store is not None:
            try:
                audit_receipt = audit_store.publish(
                    report,
                    experiment_hash=experiment.content_hash,
                    logic_hash=logic.content_hash,
                    source_partition=workflow_spec.source_partition,
                )
                persisted_report = audit_store.load_for_audit(audit_receipt)
                if persisted_report.content_hash != report.content_hash:
                    raise ValueError("persisted evaluation report differs")
            except Exception:
                raise LogicValidationWorkflowError(
                    WorkflowFailure.AUDIT_PERSISTENCE_FAILED
                ) from None

        try:
            evidence = LocalAdaptiveValidationEvidenceAdapter(evidence_policy).adapt(
                report,
                evaluation_artifact_hash=report.content_hash,
                experiment_hash=experiment.content_hash,
                logic_hash=logic.content_hash,
                source_partition=workflow_spec.source_partition,
                score_visibility=workflow_spec.score_visibility,
            )
        except Exception:
            raise LogicValidationWorkflowError(
                WorkflowFailure.EVIDENCE_ADAPTATION_FAILED
            ) from None

        registry_binding = LogicFactorBinding(
            logic_hash=logic.content_hash,
            candidate_hash=candidate.content_hash,
            factor_hash=factor.content_hash,
            experiment_hash=experiment.content_hash,
        )
        try:
            logic_record = registry.register_logic(
                logic.content_hash,
                logic_id=logic.logic_id,
                version=logic.version,
                parent_logic_hash=registry_parent_logic_hash,
                metadata={
                    "schema_version": "logic-validation-registry-metadata/v1",
                    "logic_constraint_hash": constraint.content_hash,
                    "compiler_policy_hash": compiler.policy.content_hash,
                    "semantic_catalog_hash": semantic_catalog.content_hash,
                    "data_profile_hash": logic.data_profile_hash,
                    "workflow_spec_hash": workflow_spec.content_hash,
                },
            )
            binding_record = registry.register_binding(registry_binding)
            evidence_record = registry.register_evidence(evidence)
        except Exception:
            raise LogicValidationWorkflowError(
                WorkflowFailure.REGISTRY_WRITE_FAILED
            ) from None
        if logic_record.logic_hash != logic.content_hash:
            raise LogicValidationWorkflowError(WorkflowFailure.REGISTRY_WRITE_FAILED)

        persisted = audit_receipt is not None
        return LogicValidationWorkflowReceipt(
            workflow_spec_hash=workflow_spec.content_hash,
            market_logic_hash=logic.content_hash,
            logic_constraint_hash=constraint.content_hash,
            factor_spec_hash=factor.content_hash,
            factor_view_hash=factor_view.view_hash,
            logic_factor_binding_hash=binding.content_hash,
            factor_preflight_hash=preflight.content_hash,
            candidate_hash=candidate.content_hash,
            factor_signal_hash=factor_result.signal_hash,
            label_spec_hash=label_spec.content_hash,
            label_result_hash=label_result_hash(labels),
            validation_receipt_hash=validation.content_hash,
            evaluation_spec_hash=evaluation_spec.content_hash,
            experiment_hash=experiment.content_hash,
            evaluation_artifact_hash=report.content_hash,
            evidence_artifact_hash=evidence.evidence_artifact_hash,
            registry_binding_hash=binding_record.binding_hash,
            registry_evidence_hash=evidence_record.evidence_hash,
            data_readiness=DataReadiness.STRICT_PIT_INPUT_READY,
            audit_artifact_status=(
                AuditArtifactStatus.PERSISTED
                if persisted
                else AuditArtifactStatus.NOT_PERSISTED
            ),
            registration_status=RegistrationStatus.REGISTERED,
            verdict=evidence.verdict,
            directional_stability=evidence.directional_stability,
            regime_stability=evidence.regime_stability,
            cost_resilience=evidence.cost_resilience,
            coverage=evidence.coverage,
            reason_codes=evidence.reason_codes,
            audit_artifact_receipt_hash=(
                audit_receipt.content_hash if audit_receipt is not None else None
            ),
            audit_artifact_file_hash=(
                audit_receipt.artifact_hash if audit_receipt is not None else None
            ),
            audit_artifact_binding_hash=(
                audit_receipt.binding_hash if audit_receipt is not None else None
            ),
            audit_artifact_location=(
                audit_receipt.relative_location if audit_receipt is not None else None
            ),
            audit_artifact_size_bytes=(
                audit_receipt.size_bytes if audit_receipt is not None else None
            ),
            schema_version=(
                "logic-validation-workflow-receipt/v2"
                if persisted
                else "logic-validation-workflow-receipt/v1"
            ),
        )

    @staticmethod
    def _validate_types(**values: object) -> None:
        expected = {
            "workflow_spec": LogicValidationWorkflowSpec,
            "logic": MarketLogicSpec,
            "factor": FactorSpec,
            "factor_view": FactorDataView,
            "label_spec": LabelSpec,
            "labels": LabelResult,
            "validation": ValidationReceipt,
            "evaluation_spec": EvaluationSpec,
            "compiler": MarketLogicCompiler,
            "semantic_catalog": ProviderSemanticCatalog,
            "evidence_policy": EvidenceClassificationPolicy,
            "registry": MarketLogicRegistry,
        }
        if any(not isinstance(values[name], kind) for name, kind in expected.items()):
            raise LogicValidationWorkflowError(WorkflowFailure.INVALID_INPUT)

    @staticmethod
    def _validate_label_lineage(
        *,
        logic: MarketLogicSpec,
        factor: FactorSpec,
        factor_view: FactorDataView,
        label_spec: LabelSpec,
        labels: LabelResult,
        validation: ValidationReceipt,
        workflow_spec: LogicValidationWorkflowSpec,
    ) -> None:
        factor_view.verify_content()
        labels.verify_content()
        expected_task = {
            "forward_return": LabelTask.RETURN,
            "forward_excess_return": LabelTask.EXCESS_RETURN,
            "forward_direction": LabelTask.DIRECTION,
            "forward_volatility": LabelTask.VOLATILITY,
        }.get(logic.hypothesis.belief.target)
        if expected_task is None or label_spec.task is not expected_task:
            raise ValueError("prediction target and label task differ")
        horizon = logic.hypothesis.belief.horizon
        if HorizonUnit(horizon.unit) is not HorizonUnit.SESSIONS:
            raise ValueError("prediction horizon unit is unsupported")
        if horizon.value != label_spec.horizon_sessions:
            raise ValueError("prediction and label horizons differ")
        if labels.label_spec_hash != label_spec.content_hash:
            raise ValueError("label result specification binding differs")
        if validation.label_spec_hash != label_spec.content_hash:
            raise ValueError("validation label specification binding differs")
        if validation.labels_hash != labels.labels_hash:
            raise ValueError("validation label values binding differs")
        if validation.windows_hash != labels.windows_hash:
            raise ValueError("validation label windows binding differs")
        if label_spec.security_contract_hash != factor.security_contract_hash:
            raise ValueError("factor and label security contracts differ")
        if factor.security_contract_hash != factor_view.security_contract_hash:
            raise ValueError("factor view security contract differs")
        if not factor_view.production_ready:
            raise ValueError("factor view is not production-contract ready")
        if not factor_view.strict_point_in_time or not factor_view.has_row_availability:
            raise ValueError("factor view lacks strict point-in-time availability")
        if workflow_spec.fold_id not in {item.fold_id for item in validation.folds}:
            raise ValueError("validation fold is unavailable")
        for digest, name in (
            (labels.label_view_hash, "label view hash"),
            (labels.labels_hash, "label values hash"),
            (labels.windows_hash, "label windows hash"),
            (labels.validity_hash, "label validity hash"),
            (labels.diagnostics_hash, "label diagnostics hash"),
            (validation.validation_spec_hash, "validation spec hash"),
            (validation.calendar_hash, "validation calendar hash"),
        ):
            require_sha256(digest, name=name)


def label_result_hash(labels: LabelResult) -> str:
    """Content address of the complete sealed LabelResult descriptor."""

    return labels.content_hash


def _candidate_identity(
    logic: MarketLogicSpec,
    factor: FactorSpec,
    binding: LogicFactorBindingReceipt,
) -> LogicFactorCandidateIdentity:
    if binding.market_logic_hash != logic.content_hash:
        raise LogicValidationWorkflowError(WorkflowFailure.FACTOR_BINDING_FAILED)
    if binding.factor_spec_hash != factor.content_hash:
        raise LogicValidationWorkflowError(WorkflowFailure.FACTOR_BINDING_FAILED)
    return LogicFactorCandidateIdentity(
        market_logic_hash=logic.content_hash,
        factor_spec_hash=factor.content_hash,
        logic_factor_binding_hash=binding.content_hash,
    )


def _experiment_identity(
    *,
    workflow_spec: LogicValidationWorkflowSpec,
    candidate: LogicFactorCandidateIdentity,
    preflight: FactorExecutionPreflightReceipt,
    factor_result: FactorResult,
    label_spec: LabelSpec,
    labels: LabelResult,
    validation: ValidationReceipt,
    evaluation_spec: EvaluationSpec,
) -> LogicValidationExperimentIdentity:
    return LogicValidationExperimentIdentity(
        workflow_spec_hash=workflow_spec.content_hash,
        candidate_hash=candidate.content_hash,
        factor_preflight_hash=preflight.content_hash,
        factor_view_hash=factor_result.factor_view_hash,
        factor_signal_hash=factor_result.signal_hash,
        label_spec_hash=label_spec.content_hash,
        label_result_hash=label_result_hash(labels),
        validation_receipt_hash=validation.content_hash,
        evaluation_spec_hash=evaluation_spec.content_hash,
        fold_id=workflow_spec.fold_id,
    )


def _verify_report_lineage(
    *,
    report: FactorEvaluationReport,
    evaluation_spec: EvaluationSpec,
    factor: FactorSpec,
    factor_result: FactorResult,
    label_spec: LabelSpec,
    labels: LabelResult,
    validation: ValidationReceipt,
    fold_id: str,
) -> None:
    observed = {
        "evaluation_spec": report.evaluation_spec_hash,
        "factor_spec": report.factor_spec_hash,
        "factor_signal": report.factor_signal_hash,
        "label_spec": report.label_spec_hash,
        "label_values": report.label_values_hash,
        "validation_receipt": report.validation_receipt_hash,
    }
    expected = {
        "evaluation_spec": evaluation_spec.content_hash,
        "factor_spec": factor.content_hash,
        "factor_signal": factor_result.signal_hash,
        "label_spec": label_spec.content_hash,
        "label_values": labels.labels_hash,
        "validation_receipt": validation.content_hash,
    }
    if observed != expected or report.fold_id != fold_id:
        raise ValueError("evaluation report lineage differs")
    if report.reference_factor_hashes or report.decay_label_result_hashes:
        raise ValueError("single-logic workflow produced undeclared evaluation inputs")


__all__ = [
    "AuditArtifactStatus",
    "DataReadiness",
    "LogicFactorCandidateIdentity",
    "LogicValidationExperimentIdentity",
    "LogicValidationWorkflowError",
    "LogicValidationWorkflowReceipt",
    "LogicValidationWorkflowSpec",
    "RegistrationStatus",
    "SingleMarketLogicValidationWorkflow",
    "WorkflowFailure",
    "label_result_hash",
]
