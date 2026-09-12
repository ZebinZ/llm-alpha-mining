"""The single, fail-closed assembly boundary for governed research.

The module deliberately contains no scientific implementation discovery and no
``module:callable`` loader.  Planning is read-only.  Execution accepts only
objects that the caller has already selected and injected explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from types import MappingProxyType
from typing import NoReturn, Protocol, cast
from urllib.parse import quote

import pandas as pd

from alpha_research.agents import AgentRole
from alpha_research.core.data import DataRequest, DatasetSnapshot
from alpha_research.core.hashing import (
    canonical_json_bytes,
    hash_json,
    require_sha256,
)
from alpha_research.experiments import (
    ApprovalGrant,
    DataPartition,
    ExperimentProfile,
    ExperimentRegistry,
    ExperimentRunSummary,
    ExperimentSpec,
    ResourceBudget,
    RetryPolicy,
)
from alpha_research.experiments.registry import (
    verify_experiment_registry_connection,
)
from alpha_research.evaluation import EvaluationSpec
from alpha_research.factors import FactorRegistry
from alpha_research.factors.registry import FACTOR_REGISTRY_SCHEMA
from alpha_research.factors.spec import FactorSpec
from alpha_research.labels import LabelSpec
from alpha_research.market_logic import MarketLogicRegistry
from alpha_research.market_logic.registry import (
    MARKET_LOGIC_REGISTRY_SCHEMA,
    LogicFactorBinding,
)
from alpha_research.models import ModelRegistry
from alpha_research.models.registry import MODEL_REGISTRY_SCHEMA
from alpha_research.models.nested_selection_manifest import (
    NestedSelectionPhaseOneManifestReference,
)
from alpha_research.models.spec import ModelSpec
from alpha_research.data.quality import DataQualityPolicy
from alpha_research.observability import TelemetryStore
from alpha_research.orchestration import ExperimentRuntime
from alpha_research.research.execution import (
    AgentContextPolicy,
    AgentRoleAdapter,
    JsonValue,
    ProtectedTerminalEvaluationGateway,
    ProtectedTerminalEvaluator,
    ResearchExecutionBindings,
    ScientificStageImplementation,
    ScientificStageRequest,
    ScientificStageResult,
    _run_governed_research,
    _run_research_only_g1,
    bridge_research_run,
    build_governed_stage_handlers,
)
from alpha_research.research.lineage import (
    ResearchScientificLineageManifest,
    ResearchScientificLineageManifestV1,
    ResearchScientificLineageManifestV2,
)
from alpha_research.research.model_training_stage import (
    ResolvedNestedPurgedModelTrainingFactory,
)
from alpha_research.research.nested_selection_control import (
    AgentNestedSelectionController,
    NestedOuterEvaluationAuditReader,
    NestedSelectionResearchAuthority,
)
from alpha_research.research.readiness_inputs import ResearchReadinessPlanV1
from alpha_research.research.readiness_score_stage import (
    ResearchReadinessScoreStageError,
    ResolvedResearchReadinessScoreFactory,
)
from alpha_research.research.readiness_stage import (
    ResolvedResearchReadinessFactory,
)
from alpha_research.research.spec import ResearchRunSpec, ResearchStage
from alpha_research.validation import ValidationSpec
from factor_production.v5.artifacts import ArtifactStore
from factor_production.v5.data.security_contract import (
    SecurityDataContract,
    security_data_contract_hash,
)


_PLAN_SCHEMA = "research-assembly-plan/v2"
_CHECK_SCHEMA = "research-registry-binding-check/v1"
_MAX_JSON_BYTES = 4 * 1024 * 1024
_EXPERIMENT_REGISTRY_SCHEMA = "experiment-registry/v1"
_PRESENT = "registered"
_LINEAGE_READY = frozenset({"not_required", "validated"})
_LINEAGE_STATUSES = _LINEAGE_READY | frozenset(
    {
        "required_missing",
        "registry_binding_invalid",
        "model_feature_mismatch",
        "model_selection_missing",
        "validation_deferred",
    }
)


def _requires_resolved_research_readiness(run_spec: ResearchRunSpec) -> bool:
    """Whether ROBUSTNESS belongs to the governed model-readiness path."""

    return (
        run_spec.model_spec_hash is not None
        and run_spec.score_spec_hash is not None
        and run_spec.robustness_spec_hash is not None
    )


_COMPONENT_TYPE_BY_ROLE = MappingProxyType(
    {
        "label": "label",
        "validation": "validation",
        "evaluation": "evaluation",
        "model": "model",
        "score": "factor",
        "portfolio": "portfolio",
        "risk": "portfolio",
        "cost": "cost",
        "execution": "cost",
        "backtest": "evaluation",
        "robustness": "robustness",
        "report": "evaluation",
        "admission": "evaluation",
        "code": "code",
        "environment": "environment",
        "llm_policy": "llm_policy",
        "llm_transport": "llm_transport",
        "scientific_lineage": "scientific_lineage",
    }
)


class ResearchAssemblyError(RuntimeError):
    """Stable assembly failure raised before governed execution starts."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}:{self.detail}")


@dataclass(frozen=True, slots=True)
class RegistryBindingCheck:
    registry: str
    role: str
    expected_hash: str
    status: str
    required: bool
    expected_component_type: str | None = None
    observed_component_type: str | None = None
    schema_version: str = _CHECK_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _CHECK_SCHEMA:
            raise ValueError("unsupported registry binding check schema")
        if self.registry not in {"experiment", "factor", "model", "market_logic"}:
            raise ValueError("unsupported research registry")
        if not self.role:
            raise ValueError("registry binding role must not be empty")
        require_sha256(self.expected_hash, name="registry binding expected hash")
        if self.status not in {
            _PRESENT,
            "not_registered",
            "component_type_mismatch",
            "stored_payload_invalid",
            "cross_binding_mismatch",
            "factor_lifecycle_forbidden",
            "registry_integrity_failed",
            "registry_not_configured",
        }:
            raise ValueError("unsupported registry binding status")

    @property
    def blocking(self) -> bool:
        return self.required and self.status != _PRESENT

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "registry": self.registry,
            "role": self.role,
            "expected_hash": self.expected_hash,
            "status": self.status,
            "required": self.required,
            "expected_component_type": self.expected_component_type,
            "observed_component_type": self.observed_component_type,
        }


@dataclass(frozen=True, slots=True)
class ResearchAssemblyPlan:
    research_run_spec_hash: str
    execution_bindings_hash: str
    experiment_spec: ExperimentSpec
    registry_checks: tuple[RegistryBindingCheck, ...]
    registry_configuration: Mapping[str, bool]
    scientific_lineage_manifest_hash: str | None
    scientific_lineage_status: str
    plan_id: str = field(init=False)
    schema_version: str = _PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _PLAN_SCHEMA:
            raise ValueError("unsupported research assembly plan schema")
        require_sha256(
            self.research_run_spec_hash, name="assembly research run spec hash"
        )
        require_sha256(
            self.execution_bindings_hash, name="assembly execution bindings hash"
        )
        if self.scientific_lineage_manifest_hash is not None:
            require_sha256(
                self.scientific_lineage_manifest_hash,
                name="assembly scientific lineage manifest hash",
            )
        if self.scientific_lineage_status not in _LINEAGE_STATUSES:
            raise ValueError("unsupported scientific lineage status")
        if (
            self.scientific_lineage_status in {"validated", "validation_deferred"}
            and self.scientific_lineage_manifest_hash is None
        ):
            raise ValueError("scientific lineage status requires a manifest hash")
        if (
            self.scientific_lineage_status == "not_required"
            and self.scientific_lineage_manifest_hash is not None
        ):
            raise ValueError("not-required scientific lineage cannot bind a manifest")
        checks = tuple(
            sorted(
                self.registry_checks,
                key=lambda item: (
                    item.registry,
                    item.role,
                    item.expected_hash,
                    item.status,
                ),
            )
        )
        if len(
            {(item.registry, item.role, item.expected_hash) for item in checks}
        ) != len(checks):
            raise ValueError("registry binding checks contain duplicate identities")
        configuration = {
            name: bool(value)
            for name, value in sorted(self.registry_configuration.items())
        }
        if set(configuration) != {
            "experiment",
            "factor",
            "market_logic",
            "model",
        }:
            raise ValueError("research registry configuration fields differ")
        if configuration["experiment"] is not True:
            raise ValueError("ExperimentRegistry is mandatory")
        object.__setattr__(self, "registry_checks", checks)
        object.__setattr__(
            self, "registry_configuration", MappingProxyType(configuration)
        )
        object.__setattr__(self, "plan_id", hash_json(self.identity_payload()))

    @property
    def blocking_checks(self) -> tuple[RegistryBindingCheck, ...]:
        return tuple(item for item in self.registry_checks if item.blocking)

    @property
    def missing_registry_bindings(self) -> tuple[RegistryBindingCheck, ...]:
        return self.blocking_checks

    @property
    def ready_to_register(self) -> bool:
        return (
            not self.blocking_checks
            and self.scientific_lineage_status in _LINEAGE_READY
        )

    @property
    def decision(self) -> str:
        if self.ready_to_register:
            return "READY_TO_REGISTER"
        if self.blocking_checks:
            return "BLOCKED_REGISTRY_BINDINGS"
        return "BLOCKED_SCIENTIFIC_LINEAGE"

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "research_run_spec_hash": self.research_run_spec_hash,
            "execution_bindings_hash": self.execution_bindings_hash,
            "scientific_lineage_manifest_hash": (self.scientific_lineage_manifest_hash),
            "scientific_lineage_status": self.scientific_lineage_status,
            "experiment_spec_hash": self.experiment_spec.content_hash,
            "experiment_spec": self.experiment_spec.to_dict(),
            "registry_checks": [item.to_dict() for item in self.registry_checks],
            "registry_configuration": dict(self.registry_configuration),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "decision": self.decision,
            "ready_to_register": self.ready_to_register,
            "missing_registry_bindings": [
                item.to_dict() for item in self.blocking_checks
            ],
            **self.identity_payload(),
            "assurance_boundary": {
                "read_only": True,
                "writes_performed": False,
                "data_opened": False,
                "scientific_code_executed": False,
                "dynamic_callable_imports": False,
                "experiment_registered": False,
            },
        }

    def require_ready(self) -> None:
        if self.blocking_checks:
            roles = ",".join(
                f"{item.registry}:{item.role}:{item.status}"
                for item in self.blocking_checks
            )
            raise ResearchAssemblyError("REGISTRY_BINDINGS_BLOCKED", roles)
        if self.scientific_lineage_status not in _LINEAGE_READY:
            raise ResearchAssemblyError(
                "SCIENTIFIC_LINEAGE_BLOCKED",
                self.scientific_lineage_status,
            )


class ModelTrainingScientificImplementationFactory(Protocol):
    """Pure binder for the Agent-only nested-selection capability.

    The Assembly injects the exact experiment authority used by the controller.
    A factory must not retain a caller-created Registry, Runtime, or workspace
    from construction because those clones are not authoritative for this run.
    ``bind`` itself must remain side-effect free; execution starts only after
    the atomic experiment registration below.
    """

    @property
    def selection_spec_hash(self) -> str: ...

    def bind(
        self,
        *,
        nested_selection: AgentNestedSelectionController,
        run_spec: ResearchRunSpec,
        scientific_lineage_manifest: ResearchScientificLineageManifest,
        experiment_spec: ExperimentSpec,
        registry: ExperimentRegistry,
        runtime: ExperimentRuntime,
        workspace_root: str | Path,
        clock: Callable[[], datetime] | None = None,
    ) -> ScientificStageImplementation: ...


class _AuthorityBoundModelTrainingImplementation:
    """Seal a model-stage result to the operator authority and inner manifest."""

    __slots__ = ("_authority_hash", "_delegate")

    def __init__(
        self,
        *,
        authority_hash: str,
        delegate: ScientificStageImplementation,
    ) -> None:
        self._authority_hash = require_sha256(
            authority_hash,
            name="model training nested-selection authority hash",
        )
        self._delegate = delegate

    def __call__(self, request: ScientificStageRequest) -> ScientificStageResult:
        if ResearchStage(request.stage) is not ResearchStage.MODEL_TRAINING:
            raise ResearchAssemblyError(
                "MODEL_TRAINING_STAGE_MISMATCH",
                ResearchStage(request.stage).value,
            )
        result = self._delegate(request)
        if not isinstance(result, ScientificStageResult):
            raise TypeError(
                "model training implementation must return ScientificStageResult"
            )
        if ResearchStage(result.stage) is not ResearchStage.MODEL_TRAINING:
            raise ResearchAssemblyError(
                "MODEL_TRAINING_STAGE_MISMATCH",
                ResearchStage(result.stage).value,
            )
        payload: dict[str, JsonValue] = dict(result.result_payload)
        supplied_authority = payload.get("nested_selection_authority_hash")
        if (
            supplied_authority is not None
            and supplied_authority != self._authority_hash
        ):
            raise ResearchAssemblyError(
                "MODEL_TRAINING_AUTHORITY_BINDING_MISMATCH",
                "nested_selection_authority_hash",
            )
        reference_value = payload.get("phase_one_manifest_reference")
        if not isinstance(reference_value, Mapping):
            raise ResearchAssemblyError(
                "MODEL_TRAINING_MANIFEST_REFERENCE_MISSING",
                "phase_one_manifest_reference",
            )
        reference = NestedSelectionPhaseOneManifestReference.from_mapping(
            cast(Mapping[str, object], reference_value)
        )
        payload["nested_selection_authority_hash"] = self._authority_hash
        reference_payload: dict[str, JsonValue] = {
            "schema_version": reference.schema_version,
            "manifest_id": reference.manifest_id,
            "document_sha256": reference.document_sha256,
            "filename": reference.filename,
            "size_bytes": reference.size_bytes,
            "media_type": reference.media_type,
        }
        payload["phase_one_manifest_reference"] = reference_payload
        evidence_hashes = tuple(
            dict.fromkeys(
                (
                    *result.evidence_hashes,
                    reference.manifest_id,
                    reference.document_sha256,
                )
            )
        )
        return replace(
            result,
            result_payload=payload,
            evidence_hashes=evidence_hashes,
        )


@dataclass(frozen=True, slots=True)
class GovernedResearchImplementations:
    """The caller's explicit, typed implementation allowlist.

    Values are concrete objects, never import strings.  Exact role and stage
    coverage is checked before ExperimentSpec registration.
    """

    role_adapters: Mapping[AgentRole | str, AgentRoleAdapter]
    scientific_implementations: Mapping[
        ResearchStage | str, ScientificStageImplementation
    ]
    terminal_evaluator: ProtectedTerminalEvaluator
    terminal_evaluator_hash: str
    model_training_factory: ModelTrainingScientificImplementationFactory | None = None
    controlled_score_factory: ResolvedResearchReadinessScoreFactory | None = None
    research_readiness_factory: ResolvedResearchReadinessFactory | None = None
    research_readiness_plan: ResearchReadinessPlanV1 | None = None

    def __post_init__(self) -> None:
        self._validate_and_normalize_common_bindings()
        if isinstance(self.terminal_evaluator, str) or not callable(
            self.terminal_evaluator
        ):
            raise ResearchAssemblyError(
                "UNTYPED_IMPLEMENTATION_BINDING", "terminal_evaluator"
            )
        require_sha256(
            self.terminal_evaluator_hash,
            name="terminal_evaluator_hash",
        )

    def _validate_and_normalize_common_bindings(self) -> None:
        adapters = {
            AgentRole(role): value for role, value in self.role_adapters.items()
        }
        science = {
            ResearchStage(stage): value
            for stage, value in self.scientific_implementations.items()
        }
        if ResearchStage.MODEL_TRAINING in science:
            raise ResearchAssemblyError(
                "MODEL_TRAINING_DIRECT_BINDING_FORBIDDEN",
                "use model_training_factory with the operator-bound Agent facet",
            )
        if set(adapters) != set(AgentRole):
            missing = ",".join(
                sorted(role.value for role in set(AgentRole).difference(adapters))
            )
            raise ResearchAssemblyError("ROLE_ADAPTER_ALLOWLIST_INCOMPLETE", missing)
        for name, values in (
            ("role_adapter", tuple(adapters.values())),
            ("scientific_implementation", tuple(science.values())),
        ):
            if any(isinstance(value, str) or not callable(value) for value in values):
                raise ResearchAssemblyError("UNTYPED_IMPLEMENTATION_BINDING", name)
        factory = self.model_training_factory
        if factory is not None:
            if (
                type(cast(object, factory))
                is not ResolvedNestedPurgedModelTrainingFactory
            ):
                raise ResearchAssemblyError(
                    "MODEL_TRAINING_FACTORY_UNTRUSTED",
                    type(factory).__name__,
                )
            bind = getattr(factory, "bind", None)
            if isinstance(factory, str) or not callable(bind):
                raise ResearchAssemblyError(
                    "UNTYPED_IMPLEMENTATION_BINDING",
                    "model_training_factory",
                )
            require_sha256(
                factory.selection_spec_hash,
                name="model training factory selection_spec_hash",
            )
        score_factory = self.controlled_score_factory
        if score_factory is not None:
            # Exact type first: do not evaluate attacker-controlled properties
            # while establishing the score-construction authority boundary.
            if type(score_factory) is not ResolvedResearchReadinessScoreFactory:
                raise ResearchAssemblyError(
                    "CONTROLLED_SCORE_FACTORY_UNTRUSTED",
                    type(score_factory).__name__,
                )
            require_sha256(
                score_factory.score_spec_hash,
                name="controlled score factory score_spec_hash",
            )
            require_sha256(
                score_factory.readiness_plan_hash,
                name="controlled score factory readiness_plan_hash",
            )
            require_sha256(
                score_factory.nested_selection_spec_hash,
                name="controlled score factory nested_selection_spec_hash",
            )
        readiness_factory = self.research_readiness_factory
        if readiness_factory is not None:
            # The exact-type check must precede every property/method access: a
            # duck-typed object could otherwise execute caller code while the
            # Assembly is still validating its authority boundary.
            if type(readiness_factory) is not ResolvedResearchReadinessFactory:
                raise ResearchAssemblyError(
                    "RESEARCH_READINESS_FACTORY_UNTRUSTED",
                    type(readiness_factory).__name__,
                )
            require_sha256(
                readiness_factory.readiness_spec_hash,
                name="research readiness factory readiness_spec_hash",
            )
            require_sha256(
                readiness_factory.readiness_plan_hash,
                name="research readiness factory readiness_plan_hash",
            )
        readiness_plan = self.research_readiness_plan
        if (
            readiness_plan is not None
            and type(readiness_plan) is not ResearchReadinessPlanV1
        ):
            raise ResearchAssemblyError(
                "RESEARCH_READINESS_PLAN_UNTRUSTED",
                type(readiness_plan).__name__,
            )
        if (readiness_factory is None) != (readiness_plan is None):
            raise ResearchAssemblyError(
                "RESEARCH_READINESS_BINDING_INCOMPLETE",
                "factory_and_plan_must_be_supplied_together",
            )
        if (
            readiness_factory is not None
            and readiness_plan is not None
            and readiness_factory.readiness_plan_hash != readiness_plan.content_hash
        ):
            raise ResearchAssemblyError(
                "RESEARCH_READINESS_PLAN_MISMATCH",
                readiness_plan.content_hash,
            )
        if score_factory is not None and readiness_plan is None:
            raise ResearchAssemblyError(
                "CONTROLLED_SCORE_BINDING_INCOMPLETE",
                "readiness_plan_is_required",
            )
        if (
            score_factory is not None
            and readiness_plan is not None
            and score_factory.readiness_plan_hash != readiness_plan.content_hash
        ):
            raise ResearchAssemblyError(
                "CONTROLLED_SCORE_PLAN_MISMATCH",
                readiness_plan.content_hash,
            )
        if (
            score_factory is not None
            and readiness_plan is not None
            and score_factory.nested_selection_spec_hash
            != readiness_plan.nested_selection_spec_hash
        ):
            raise ResearchAssemblyError(
                "CONTROLLED_SCORE_NESTED_SELECTION_MISMATCH",
                readiness_plan.nested_selection_spec_hash,
            )
        object.__setattr__(self, "role_adapters", MappingProxyType(dict(adapters)))
        object.__setattr__(
            self, "scientific_implementations", MappingProxyType(dict(science))
        )

    def _validate_scientific_bindings_for(self, run_spec: ResearchRunSpec) -> None:
        readiness_managed = _requires_resolved_research_readiness(run_spec)
        if (
            readiness_managed
            and ResearchStage.SCORE_CONSTRUCTION in self.scientific_implementations
        ):
            raise ResearchAssemblyError(
                "CONTROLLED_SCORE_DIRECT_BINDING_FORBIDDEN",
                "use the exact controlled_score_factory for governed model research",
            )
        if (
            readiness_managed
            and ResearchStage.ROBUSTNESS in self.scientific_implementations
        ):
            raise ResearchAssemblyError(
                "RESEARCH_READINESS_DIRECT_BINDING_FORBIDDEN",
                "use research_readiness_factory for governed model research",
            )
        excluded = {ResearchStage.ADMISSION, ResearchStage.MODEL_TRAINING}
        if readiness_managed:
            excluded.add(ResearchStage.SCORE_CONSTRUCTION)
            excluded.add(ResearchStage.ROBUSTNESS)
        expected = {stage for stage in run_spec.enabled_stages if stage not in excluded}
        observed = {ResearchStage(stage) for stage in self.scientific_implementations}
        if observed != expected:
            missing = ",".join(
                sorted(stage.value for stage in expected.difference(observed))
            )
            extra = ",".join(
                sorted(stage.value for stage in observed.difference(expected))
            )
            raise ResearchAssemblyError(
                "SCIENTIFIC_ALLOWLIST_MISMATCH",
                f"missing={missing};extra={extra}",
            )
        model_training_enabled = ResearchStage.MODEL_TRAINING in run_spec.enabled_stages
        if model_training_enabled and self.model_training_factory is None:
            raise ResearchAssemblyError(
                "MODEL_TRAINING_FACTORY_REQUIRED",
                "model_training",
            )
        if not model_training_enabled and self.model_training_factory is not None:
            raise ResearchAssemblyError(
                "MODEL_TRAINING_FACTORY_UNEXPECTED",
                "model_training",
            )
        if readiness_managed:
            score_factory = self.controlled_score_factory
            if score_factory is None:
                raise ResearchAssemblyError(
                    "CONTROLLED_SCORE_FACTORY_REQUIRED",
                    "score_construction",
                )
            if score_factory.score_spec_hash != run_spec.score_spec_hash:
                raise ResearchAssemblyError(
                    "CONTROLLED_SCORE_SPEC_MISMATCH",
                    str(run_spec.score_spec_hash),
                )
        elif self.controlled_score_factory is not None:
            raise ResearchAssemblyError(
                "CONTROLLED_SCORE_FACTORY_UNEXPECTED",
                "score_construction",
            )
        if readiness_managed:
            if self.research_readiness_factory is None:
                raise ResearchAssemblyError(
                    "RESEARCH_READINESS_FACTORY_REQUIRED",
                    "robustness",
                )
            if self.research_readiness_plan is None:
                raise ResearchAssemblyError(
                    "RESEARCH_READINESS_PLAN_REQUIRED",
                    "robustness",
                )
            if (
                self.research_readiness_factory.readiness_spec_hash
                != run_spec.robustness_spec_hash
            ):
                raise ResearchAssemblyError(
                    "RESEARCH_READINESS_SPEC_MISMATCH",
                    str(run_spec.robustness_spec_hash),
                )
            plan = self.research_readiness_plan
            score_factory = self.controlled_score_factory
            if score_factory is None:  # pragma: no cover - guarded above.
                raise RuntimeError("controlled score factory disappeared")
            if score_factory.readiness_plan_hash != plan.content_hash:
                raise ResearchAssemblyError(
                    "CONTROLLED_SCORE_PLAN_MISMATCH",
                    plan.content_hash,
                )
            if (
                score_factory.nested_selection_spec_hash
                != plan.nested_selection_spec_hash
            ):
                raise ResearchAssemblyError(
                    "CONTROLLED_SCORE_NESTED_SELECTION_MISMATCH",
                    plan.nested_selection_spec_hash,
                )
            if plan.readiness_spec_hash != run_spec.robustness_spec_hash:
                raise ResearchAssemblyError(
                    "RESEARCH_READINESS_PLAN_SPEC_MISMATCH",
                    str(run_spec.robustness_spec_hash),
                )
            if plan.validation_spec_hash != run_spec.validation_spec_hash:
                raise ResearchAssemblyError(
                    "RESEARCH_READINESS_VALIDATION_SPEC_MISMATCH",
                    run_spec.validation_spec_hash,
                )
        elif self.research_readiness_factory is not None:
            raise ResearchAssemblyError(
                "RESEARCH_READINESS_FACTORY_UNEXPECTED",
                "robustness",
            )

    def validate_for(self, run_spec: ResearchRunSpec) -> None:
        self._validate_scientific_bindings_for(run_spec)
        if ResearchStage.ADMISSION not in run_spec.enabled_stages:
            raise ResearchAssemblyError(
                "TERMINAL_ADMISSION_REQUIRED",
                "governed execution requires an admission stage",
            )

    def validate_research_readiness_lineage(
        self,
        *,
        run_spec: ResearchRunSpec,
        scientific_lineage_manifest: ResearchScientificLineageManifestV2,
    ) -> None:
        """Validate the preregistered readiness plan against run lineage."""

        if not _requires_resolved_research_readiness(run_spec):
            return
        plan = self.research_readiness_plan
        factory = self.research_readiness_factory
        score_factory = self.controlled_score_factory
        if (
            factory is None or plan is None or score_factory is None
        ):  # pragma: no cover - validate_for guards.
            raise RuntimeError("research readiness binding disappeared")
        if plan.content_hash != factory.readiness_plan_hash:
            raise ResearchAssemblyError(
                "RESEARCH_READINESS_PLAN_MISMATCH",
                plan.content_hash,
            )
        if (
            plan.nested_selection_spec_hash
            != scientific_lineage_manifest.nested_selection_spec_hash
        ):
            raise ResearchAssemblyError(
                "RESEARCH_READINESS_NESTED_SELECTION_MISMATCH",
                scientific_lineage_manifest.nested_selection_spec_hash,
            )
        if (
            score_factory.nested_selection_spec_hash
            != scientific_lineage_manifest.nested_selection_spec_hash
        ):
            raise ResearchAssemblyError(
                "CONTROLLED_SCORE_NESTED_SELECTION_MISMATCH",
                scientific_lineage_manifest.nested_selection_spec_hash,
            )
        validation_partition = run_spec.data_partition_hashes[DataPartition.VALIDATION]
        if plan.validation_partition_hash != validation_partition:
            raise ResearchAssemblyError(
                "RESEARCH_READINESS_VALIDATION_PARTITION_MISMATCH",
                validation_partition,
            )
        validation_window = run_spec.data_partition_windows[DataPartition.VALIDATION]
        if (
            plan.validation_window_start,
            plan.validation_window_end,
        ) != validation_window:
            raise ResearchAssemblyError(
                "RESEARCH_READINESS_VALIDATION_WINDOW_MISMATCH",
                DataPartition.VALIDATION.value,
            )

    def bind_research_readiness(
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
        """Bind ML-F only to Assembly-owned registry/runtime authorities."""

        factory = self.research_readiness_factory
        plan = self.research_readiness_plan
        if factory is None or plan is None:
            raise ResearchAssemblyError(
                "RESEARCH_READINESS_FACTORY_REQUIRED",
                "robustness",
            )
        implementation = factory.bind(
            readiness_plan=plan,
            run_spec=run_spec,
            scientific_lineage_manifest=scientific_lineage_manifest,
            experiment_spec=experiment_spec,
            registry=registry,
            runtime=runtime,
            artifact_store=artifact_store,
            outer_audit_reader=outer_audit_reader,
            worker_id=worker_id,
            lease_token=lease_token,
            clock=clock,
        )
        if isinstance(implementation, str) or not callable(implementation):
            raise ResearchAssemblyError(
                "UNTYPED_IMPLEMENTATION_BINDING",
                "research_readiness_implementation",
            )
        return implementation

    def bind_controlled_score(
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
        """Bind the exact controlled SCORE implementation to run authorities."""

        factory = self.controlled_score_factory
        if factory is None:
            raise ResearchAssemblyError(
                "CONTROLLED_SCORE_FACTORY_REQUIRED",
                "score_construction",
            )
        implementation = factory.bind(
            run_spec=run_spec,
            scientific_lineage_manifest=scientific_lineage_manifest,
            experiment_spec=experiment_spec,
            registry=registry,
            runtime=runtime,
            artifact_store=artifact_store,
            outer_audit_reader=outer_audit_reader,
            worker_id=worker_id,
            lease_token=lease_token,
            clock=clock,
        )
        if isinstance(implementation, str) or not callable(implementation):
            raise ResearchAssemblyError(
                "UNTYPED_IMPLEMENTATION_BINDING",
                "controlled_score_implementation",
            )
        return implementation

    def bind_model_training(
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
        """Bind the model stage to this Assembly's exact execution authority."""

        if type(nested_selection) is not AgentNestedSelectionController:
            raise TypeError("nested_selection must be an exact Agent controller")
        if not isinstance(run_spec, ResearchRunSpec):
            raise TypeError("run_spec must be ResearchRunSpec")
        if not isinstance(
            scientific_lineage_manifest, ResearchScientificLineageManifestV2
        ):
            raise TypeError(
                "scientific_lineage_manifest must be ResearchScientificLineageManifestV2"
            )
        if not isinstance(experiment_spec, ExperimentSpec):
            raise TypeError("experiment_spec must be ExperimentSpec")
        if not isinstance(registry, ExperimentRegistry):
            raise TypeError("registry must be ExperimentRegistry")
        if not isinstance(runtime, ExperimentRuntime):
            raise TypeError("runtime must be ExperimentRuntime")
        if runtime.spec.content_hash != experiment_spec.content_hash:
            raise ResearchAssemblyError(
                "RUNTIME_SPEC_MISMATCH", experiment_spec.content_hash
            )
        if clock is not None and not callable(clock):
            raise TypeError("model-training clock must be callable or None")
        factory = self.model_training_factory
        if factory is None:
            raise ResearchAssemblyError(
                "MODEL_TRAINING_FACTORY_REQUIRED",
                "model_training",
            )
        implementation = factory.bind(
            nested_selection=nested_selection,
            run_spec=run_spec,
            scientific_lineage_manifest=scientific_lineage_manifest,
            experiment_spec=experiment_spec,
            registry=registry,
            runtime=runtime,
            workspace_root=workspace_root,
            clock=clock,
        )
        if isinstance(implementation, str) or not callable(implementation):
            raise ResearchAssemblyError(
                "UNTYPED_IMPLEMENTATION_BINDING",
                "model_training_implementation",
            )
        return _AuthorityBoundModelTrainingImplementation(
            authority_hash=nested_selection.authority_hash,
            delegate=implementation,
        )

    def model_training_selection_spec_hash(self) -> str:
        factory = self.model_training_factory
        if factory is None:
            raise ResearchAssemblyError(
                "MODEL_TRAINING_FACTORY_REQUIRED",
                "model_training",
            )
        return cast(
            str,
            require_sha256(
                factory.selection_spec_hash,
                name="model training factory selection_spec_hash",
            ),
        )


@dataclass(frozen=True, slots=True)
class ResearchOnlyG1Implementations(GovernedResearchImplementations):
    """Typed implementation allowlist with no G2 terminal capability.

    The terminal fields are removed from the constructor and frozen to
    ``None``.  Production execution rejects this subtype by exact type, so a
    research-only bundle cannot be smuggled into the protected path.
    """

    # The subtype intentionally removes these production capabilities from its
    # constructor and narrows their immutable values to None.
    terminal_evaluator: None = field(default=None, init=False)  # type: ignore[assignment]
    terminal_evaluator_hash: None = field(default=None, init=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._validate_and_normalize_common_bindings()
        if (
            self.terminal_evaluator is not None
            or self.terminal_evaluator_hash is not None
        ):
            raise ResearchAssemblyError(
                "RESEARCH_ONLY_TERMINAL_CAPABILITY_FORBIDDEN",
                "terminal_evaluator must be None",
            )

    def validate_for(self, run_spec: ResearchRunSpec) -> None:
        if (
            ExperimentProfile(run_spec.profile)
            is ExperimentProfile.PRODUCTION_CANDIDATE
        ):
            raise ResearchAssemblyError(
                "RESEARCH_ONLY_PROFILE_FORBIDDEN",
                ExperimentProfile.PRODUCTION_CANDIDATE.value,
            )
        if ResearchStage.ADMISSION in run_spec.enabled_stages:
            raise ResearchAssemblyError(
                "RESEARCH_ONLY_ADMISSION_FORBIDDEN",
                ResearchStage.ADMISSION.value,
            )
        protected = {DataPartition.TEST, DataPartition.HOLDOUT}.intersection(
            run_spec.data_partition_hashes
        )
        if protected:
            raise ResearchAssemblyError(
                "RESEARCH_ONLY_PROTECTED_PARTITION_FORBIDDEN",
                ",".join(sorted(partition.value for partition in protected)),
            )
        self._validate_scientific_bindings_for(run_spec)


@dataclass(frozen=True, slots=True)
class ResearchOnlyG1SemanticAuthorities:
    """Exact semantic authorities required by the research-only G1 path.

    Experiment-registry component descriptors remain useful lineage indices,
    but they are not semantic authorities: a descriptor containing only a
    caller-chosen ``content_hash`` must never authorize execution.  This bundle
    carries the already-selected typed contracts and the live factor registry
    from which factor definitions are independently reloaded.
    """

    factor_registry: FactorRegistry
    dataset_snapshot: DatasetSnapshot
    data_request: DataRequest
    data_quality_policy: DataQualityPolicy
    label_spec: LabelSpec
    validation_spec: ValidationSpec
    evaluation_spec: EvaluationSpec
    security_contract: SecurityDataContract
    sealed_partition_hashes: Mapping[DataPartition | str, str]
    schema_version: str = "research-only-g1-semantic-authorities/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "research-only-g1-semantic-authorities/v1":
            raise ValueError("unsupported research-only G1 authority schema")
        expected_types = (
            (self.factor_registry, FactorRegistry, "factor_registry"),
            (self.dataset_snapshot, DatasetSnapshot, "dataset_snapshot"),
            (self.data_request, DataRequest, "data_request"),
            (self.data_quality_policy, DataQualityPolicy, "data_quality_policy"),
            (self.label_spec, LabelSpec, "label_spec"),
            (self.validation_spec, ValidationSpec, "validation_spec"),
            (self.evaluation_spec, EvaluationSpec, "evaluation_spec"),
            (self.security_contract, SecurityDataContract, "security_contract"),
        )
        for value, expected, name in expected_types:
            if type(value) is not expected:
                raise TypeError(f"{name} must be an exact {expected.__name__}")
        if not isinstance(self.sealed_partition_hashes, Mapping):
            raise TypeError("sealed_partition_hashes must be a mapping")
        partitions: dict[DataPartition, str] = {}
        for raw_partition, raw_digest in self.sealed_partition_hashes.items():
            partition = DataPartition(raw_partition)
            if partition in partitions:
                raise ValueError("sealed partition aliases are forbidden")
            if type(raw_digest) is not str:
                raise TypeError("sealed partition hashes must be text")
            partitions[partition] = require_sha256(
                raw_digest,
                name=f"sealed partition:{partition.value}",
            )
        if not partitions:
            raise ValueError("sealed_partition_hashes must not be empty")
        object.__setattr__(
            self,
            "sealed_partition_hashes",
            MappingProxyType(
                {
                    partition: partitions[partition]
                    for partition in sorted(
                        partitions,
                        key=lambda value: str(value.value),
                    )
                }
            ),
        )


class _RegistryLookup(Protocol):
    configuration: Mapping[str, bool]

    def experiment_integrity_ok(self) -> bool: ...

    def experiment_component(self, digest: str) -> tuple[str, bool] | None: ...

    def factor_spec(self, digest: str) -> tuple[FactorSpec, str] | str | None: ...

    def model_spec(self, digest: str) -> ModelSpec | str | None: ...

    def market_logic(self, digest: str) -> str | None: ...

    def market_logic_binding(self, digest: str) -> LogicFactorBinding | str | None: ...


class _LiveRegistryLookup:
    def __init__(
        self,
        *,
        experiment_registry: ExperimentRegistry,
        factor_registry: FactorRegistry | None,
        model_registry: ModelRegistry | None,
        market_logic_registry: MarketLogicRegistry | None,
    ) -> None:
        self.experiment_registry = experiment_registry
        self.factor_registry = factor_registry
        self.model_registry = model_registry
        self.market_logic_registry = market_logic_registry
        self.configuration: Mapping[str, bool] = MappingProxyType(
            {
                "experiment": True,
                "factor": factor_registry is not None,
                "market_logic": market_logic_registry is not None,
                "model": model_registry is not None,
            }
        )

    def experiment_integrity_ok(self) -> bool:
        return cast(bool, self.experiment_registry.verify_integrity() == ())

    def experiment_component(self, digest: str) -> tuple[str, bool] | None:
        row = self.experiment_registry.connection.execute(
            """SELECT component_type,descriptor_json FROM components
               WHERE component_hash=?""",
            (digest,),
        ).fetchone()
        return _component_row(row, digest)

    def factor_spec(self, digest: str) -> tuple[FactorSpec, str] | str | None:
        if self.factor_registry is None:
            return None
        try:
            return (
                self.factor_registry.get(digest),
                self.factor_registry.current_state(digest),
            )
        except KeyError:
            return "not_registered"
        except (TypeError, ValueError, RuntimeError, sqlite3.DatabaseError):
            return "stored_payload_invalid"

    def model_spec(self, digest: str) -> ModelSpec | str | None:
        if self.model_registry is None:
            return None
        try:
            return self.model_registry.get_spec(digest)
        except KeyError:
            return "not_registered"
        except (TypeError, ValueError, RuntimeError, sqlite3.DatabaseError):
            return "stored_payload_invalid"

    def market_logic(self, digest: str) -> str | None:
        if self.market_logic_registry is None:
            return None
        try:
            record = self.market_logic_registry.get_logic(digest)
            return (
                _PRESENT
                if record.logic_hash == digest
                and hash_json(dict(record.metadata)) == record.metadata_hash
                else "stored_payload_invalid"
            )
        except KeyError:
            return "not_registered"
        except (TypeError, ValueError, RuntimeError, sqlite3.DatabaseError):
            return "stored_payload_invalid"

    def market_logic_binding(self, digest: str) -> LogicFactorBinding | str | None:
        if self.market_logic_registry is None:
            return None
        try:
            record = self.market_logic_registry.read_binding_for_audit(digest)
            return (
                record.binding
                if record.binding_hash == digest
                and record.binding.content_hash == digest
                else "stored_payload_invalid"
            )
        except KeyError:
            return "not_registered"
        except (TypeError, ValueError, RuntimeError, sqlite3.DatabaseError):
            return "stored_payload_invalid"


@dataclass(frozen=True, slots=True)
class ReadOnlyRegistryPaths:
    experiment: Path
    factor: Path | None = None
    model: Path | None = None
    market_logic: Path | None = None

    def __post_init__(self) -> None:
        for name in ("experiment", "factor", "model", "market_logic"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))


class ReadOnlyResearchRegistryLookup:
    """SQLite lookup adapter that cannot create or mutate registry files."""

    def __init__(self, paths: ReadOnlyRegistryPaths) -> None:
        self.paths = paths
        self._connections: dict[str, sqlite3.Connection] = {}
        self.configuration: Mapping[str, bool] = MappingProxyType(
            {
                "experiment": True,
                "factor": paths.factor is not None,
                "market_logic": paths.market_logic is not None,
                "model": paths.model is not None,
            }
        )

    def __enter__(self) -> "ReadOnlyResearchRegistryLookup":
        expected = {
            "experiment": (self.paths.experiment, _EXPERIMENT_REGISTRY_SCHEMA),
            "factor": (self.paths.factor, FACTOR_REGISTRY_SCHEMA),
            "model": (self.paths.model, MODEL_REGISTRY_SCHEMA),
            "market_logic": (
                self.paths.market_logic,
                MARKET_LOGIC_REGISTRY_SCHEMA,
            ),
        }
        try:
            for name, (path, schema) in expected.items():
                if path is not None:
                    connection = _open_read_only_registry(path)
                    _require_registry_schema(connection, schema, name=name)
                    self._connections[name] = connection
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        for connection in self._connections.values():
            connection.close()
        self._connections.clear()

    def experiment_integrity_ok(self) -> bool:
        return not verify_experiment_registry_connection(
            self._connections["experiment"]
        )

    def experiment_component(self, digest: str) -> tuple[str, bool] | None:
        row = (
            self._connections["experiment"]
            .execute(
                """SELECT component_type,descriptor_json FROM components
               WHERE component_hash=?""",
                (digest,),
            )
            .fetchone()
        )
        return _component_row(row, digest)

    def factor_spec(self, digest: str) -> tuple[FactorSpec, str] | str | None:
        connection = self._connections.get("factor")
        if connection is None:
            return None
        row = connection.execute(
            "SELECT payload_json FROM factors WHERE factor_hash=?", (digest,)
        ).fetchone()
        if row is None:
            return "not_registered"
        try:
            payload = _json_object_text(row["payload_json"])
            spec = FactorSpec.from_mapping(payload)
            if spec.content_hash != digest:
                return "stored_payload_invalid"
            state_row = connection.execute(
                """SELECT state FROM factor_lifecycle_events
                   WHERE factor_hash=? ORDER BY event_id DESC LIMIT 1""",
                (digest,),
            ).fetchone()
            if state_row is None:
                return "stored_payload_invalid"
            return spec, str(state_row["state"])
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            return "stored_payload_invalid"

    def model_spec(self, digest: str) -> ModelSpec | str | None:
        connection = self._connections.get("model")
        if connection is None:
            return None
        row = connection.execute(
            "SELECT payload_json FROM models WHERE model_hash=?", (digest,)
        ).fetchone()
        if row is None:
            return "not_registered"
        try:
            spec = ModelSpec.from_mapping(_json_object_text(row["payload_json"]))
            return spec if spec.content_hash == digest else "stored_payload_invalid"
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            return "stored_payload_invalid"

    def market_logic(self, digest: str) -> str | None:
        connection = self._connections.get("market_logic")
        if connection is None:
            return None
        row = connection.execute(
            """SELECT logic_hash,metadata_hash,metadata_json
               FROM logic_versions WHERE logic_hash=?""",
            (digest,),
        ).fetchone()
        if row is None:
            return "not_registered"
        try:
            metadata = _json_object_text(row["metadata_json"])
            return (
                _PRESENT
                if row["logic_hash"] == digest
                and hash_json(metadata) == row["metadata_hash"]
                else "stored_payload_invalid"
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return "stored_payload_invalid"

    def market_logic_binding(self, digest: str) -> LogicFactorBinding | str | None:
        connection = self._connections.get("market_logic")
        if connection is None:
            return None
        row = connection.execute(
            """SELECT binding_hash,payload_json
               FROM logic_factor_bindings WHERE binding_hash=?""",
            (digest,),
        ).fetchone()
        if row is None:
            return "not_registered"
        try:
            binding = LogicFactorBinding.from_dict(
                _json_object_text(row["payload_json"])
            )
            return (
                binding
                if row["binding_hash"] == digest and binding.content_hash == digest
                else "stored_payload_invalid"
            )
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            return "stored_payload_invalid"


class ResearchRegistryAssembly:
    """Compose existing registries; never create another registry/controller."""

    def __init__(
        self,
        *,
        experiment_registry: ExperimentRegistry,
        factor_registry: FactorRegistry | None = None,
        model_registry: ModelRegistry | None = None,
        market_logic_registry: MarketLogicRegistry | None = None,
        nested_selection_authority: NestedSelectionResearchAuthority | None = None,
    ) -> None:
        if not isinstance(experiment_registry, ExperimentRegistry):
            raise TypeError("experiment_registry must be ExperimentRegistry")
        for value, expected, name in (
            (factor_registry, FactorRegistry, "factor_registry"),
            (model_registry, ModelRegistry, "model_registry"),
            (market_logic_registry, MarketLogicRegistry, "market_logic_registry"),
        ):
            if value is not None and not isinstance(value, expected):
                raise TypeError(f"{name} has an invalid registry type")
        if (
            nested_selection_authority is not None
            and type(nested_selection_authority) is not NestedSelectionResearchAuthority
        ):
            raise TypeError(
                "nested_selection_authority must be NestedSelectionResearchAuthority"
            )
        self.experiment_registry = experiment_registry
        self.factor_registry = factor_registry
        self.model_registry = model_registry
        self.market_logic_registry = market_logic_registry
        self._nested_selection_authority = nested_selection_authority
        self._lookup = _LiveRegistryLookup(
            experiment_registry=experiment_registry,
            factor_registry=factor_registry,
            model_registry=model_registry,
            market_logic_registry=market_logic_registry,
        )

    def plan(
        self,
        run_spec: ResearchRunSpec,
        execution_bindings: ResearchExecutionBindings,
        *,
        market_logic_hashes: Sequence[str] = (),
        scientific_lineage_manifest: ResearchScientificLineageManifest | None = None,
    ) -> ResearchAssemblyPlan:
        return _build_plan(
            run_spec,
            execution_bindings,
            lookup=self._lookup,
            market_logic_hashes=market_logic_hashes,
            scientific_lineage_manifest=scientific_lineage_manifest,
        )

    def execute(
        self,
        *,
        run_spec: ResearchRunSpec,
        execution_bindings: ResearchExecutionBindings,
        agent_policy: AgentContextPolicy,
        scientific_lineage_manifest: ResearchScientificLineageManifest | None = None,
        implementations: GovernedResearchImplementations,
        protected_evaluation_approval: ApprovalGrant,
        runtime: ExperimentRuntime,
        telemetry: TelemetryStore,
        workspace_root: str | Path,
        worker_id: str,
        lease_token: str,
        lease_seconds: int = 60,
        clock: Callable[[], datetime] | None = None,
        registered_at: datetime | None = None,
        stop_after_stage: ResearchStage | str | None = None,
    ) -> ExperimentRunSummary:
        """Register the bridged ExperimentSpec and call the existing controller.

        Every validation that can fail without writing is performed before the
        atomic experiment/locked-test-approval registration.  The terminal
        evaluator is reachable only through a durable one-time gateway.
        """

        if not isinstance(agent_policy, AgentContextPolicy):
            raise TypeError("agent_policy must be AgentContextPolicy")
        if type(implementations) is not GovernedResearchImplementations:
            raise TypeError(
                "implementations must be an exact GovernedResearchImplementations"
            )
        if not isinstance(protected_evaluation_approval, ApprovalGrant):
            raise TypeError("protected_evaluation_approval must be ApprovalGrant")
        implementations.validate_for(run_spec)
        model_training_enabled = ResearchStage.MODEL_TRAINING in run_spec.enabled_stages
        model_training_lineage: ResearchScientificLineageManifestV2 | None = None
        if model_training_enabled:
            if not isinstance(
                scientific_lineage_manifest,
                ResearchScientificLineageManifestV2,
            ):
                raise ResearchAssemblyError(
                    "NESTED_SELECTION_LINEAGE_REQUIRED",
                    "model_training requires scientific lineage v2",
                )
            model_training_lineage = scientific_lineage_manifest
            if (
                implementations.model_training_selection_spec_hash()
                != model_training_lineage.nested_selection_spec_hash
            ):
                raise ResearchAssemblyError(
                    "MODEL_TRAINING_SELECTION_SPEC_MISMATCH",
                    model_training_lineage.nested_selection_spec_hash,
                )
            implementations.validate_research_readiness_lineage(
                run_spec=run_spec,
                scientific_lineage_manifest=model_training_lineage,
            )
        if model_training_enabled and self._nested_selection_authority is None:
            raise ResearchAssemblyError(
                "NESTED_SELECTION_AUTHORITY_UNCONFIGURED",
                "model_training",
            )
        _validate_scientific_lineage_policy(
            scientific_lineage_manifest=scientific_lineage_manifest,
            agent_policy=agent_policy,
        )
        plan = self.plan(
            run_spec,
            execution_bindings,
            market_logic_hashes=agent_policy.hypothesis_hashes,
            scientific_lineage_manifest=scientific_lineage_manifest,
        )
        plan.require_ready()
        _validate_factor_policy_bindings(
            run_spec=run_spec,
            agent_policy=agent_policy,
            factor_registry=self.factor_registry,
        )
        if runtime.spec.content_hash != plan.experiment_spec.content_hash:
            raise ResearchAssemblyError(
                "RUNTIME_SPEC_MISMATCH", plan.experiment_spec.content_hash
            )
        if telemetry.experiment_spec_hash != plan.experiment_spec.content_hash:
            raise ResearchAssemblyError(
                "TELEMETRY_SPEC_MISMATCH", plan.experiment_spec.content_hash
            )
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable or None")
        checkpoint_stage: ResearchStage | None = None
        if stop_after_stage is not None:
            checkpoint_stage = ResearchStage(stop_after_stage)
            if checkpoint_stage is not ResearchStage.MODEL_TRAINING:
                raise ResearchAssemblyError(
                    "UNSUPPORTED_CHECKPOINT_BOUNDARY",
                    checkpoint_stage.value,
                )
            if not model_training_enabled:
                raise ResearchAssemblyError(
                    "MODEL_TRAINING_CHECKPOINT_UNEXPECTED",
                    "model_training",
                )
        _validate_worker_lease_inputs(
            worker_id=worker_id,
            lease_token=lease_token,
            lease_seconds=lease_seconds,
        )

        if scientific_lineage_manifest is None:  # guarded by plan.require_ready()
            raise ResearchAssemblyError(
                "SCIENTIFIC_LINEAGE_BLOCKED",
                "required_missing",
            )
        readiness_managed = _requires_resolved_research_readiness(run_spec)
        if readiness_managed and checkpoint_stage is None:
            model_success = runtime.successful_attempt(
                ResearchStage.MODEL_TRAINING.value
            )
            if model_success is None:
                raise ResearchAssemblyError(
                    "MODEL_TRAINING_CHECKPOINT_REQUIRED",
                    "first managed execution must stop after model_training",
                )
        scientific_implementations = dict(implementations.scientific_implementations)
        if model_training_enabled:
            authority = self._nested_selection_authority
            if authority is None:  # pragma: no cover - guarded before planning.
                raise RuntimeError("nested selection authority disappeared")
            if model_training_lineage is None:  # pragma: no cover - guarded above.
                raise RuntimeError("model-training lineage disappeared")
            scientific_implementations[ResearchStage.MODEL_TRAINING] = (
                implementations.bind_model_training(
                    nested_selection=NestedSelectionResearchAuthority.agent_controller(
                        authority
                    ),
                    run_spec=run_spec,
                    scientific_lineage_manifest=model_training_lineage,
                    experiment_spec=plan.experiment_spec,
                    registry=self.experiment_registry,
                    runtime=runtime,
                    workspace_root=workspace_root,
                    clock=clock,
                )
            )
            if readiness_managed:
                artifact_store = ArtifactStore(workspace_root)
                outer_audit_reader = authority.audit_reader()
                if checkpoint_stage is None:
                    score_factory = implementations.controlled_score_factory
                    if score_factory is None:  # pragma: no cover - validate_for.
                        raise RuntimeError("controlled score factory disappeared")
                    try:
                        score_factory.preflight_outer_completion(
                            experiment_spec=plan.experiment_spec,
                            registry=self.experiment_registry,
                            runtime=runtime,
                            artifact_store=artifact_store,
                            outer_audit_reader=outer_audit_reader,
                        )
                    except ResearchReadinessScoreStageError as exc:
                        raise ResearchAssemblyError(
                            "CONTROLLED_SCORE_PREFLIGHT_FAILED",
                            exc.code,
                        ) from exc
                scientific_implementations[ResearchStage.SCORE_CONSTRUCTION] = (
                    implementations.bind_controlled_score(
                        run_spec=run_spec,
                        scientific_lineage_manifest=model_training_lineage,
                        experiment_spec=plan.experiment_spec,
                        registry=self.experiment_registry,
                        runtime=runtime,
                        artifact_store=artifact_store,
                        outer_audit_reader=outer_audit_reader,
                        worker_id=worker_id,
                        lease_token=lease_token,
                        clock=clock,
                    )
                )
                scientific_implementations[ResearchStage.ROBUSTNESS] = (
                    implementations.bind_research_readiness(
                        run_spec=run_spec,
                        scientific_lineage_manifest=model_training_lineage,
                        experiment_spec=plan.experiment_spec,
                        registry=self.experiment_registry,
                        runtime=runtime,
                        artifact_store=artifact_store,
                        outer_audit_reader=outer_audit_reader,
                        worker_id=worker_id,
                        lease_token=lease_token,
                        clock=clock,
                    )
                )
        agent_context = agent_policy.bind(plan.experiment_spec)
        terminal_gateway = ProtectedTerminalEvaluationGateway(
            registry=self.experiment_registry,
            experiment_spec_hash=plan.experiment_spec.content_hash,
            evaluation_spec_hash=plan.experiment_spec.evaluation_spec_hash,
            approval_hash=protected_evaluation_approval.content_hash,
            evaluator_hash=implementations.terminal_evaluator_hash,
            evaluator=implementations.terminal_evaluator,
            clock=clock,
        )
        handlers = build_governed_stage_handlers(
            run_spec=run_spec,
            experiment_spec=plan.experiment_spec,
            agent_context=agent_context,
            role_adapters=implementations.role_adapters,
            scientific_implementations=scientific_implementations,
            terminal_evaluator=terminal_gateway,
        )
        self.experiment_registry.register_component_experiment_and_protected_approval(
            plan.experiment_spec,
            component_hash=scientific_lineage_manifest.content_hash,
            component_type="scientific_lineage",
            descriptor={
                "content_hash": scientific_lineage_manifest.content_hash,
                "schema_version": scientific_lineage_manifest.schema_version,
                "manifest": scientific_lineage_manifest.to_dict(),
            },
            approval=protected_evaluation_approval,
            registered_at=registered_at,
            approval_at=registered_at or datetime.now(timezone.utc),
        )
        return _run_governed_research(
            run_spec=run_spec,
            experiment_spec=plan.experiment_spec,
            handlers=handlers,
            registry=self.experiment_registry,
            runtime=runtime,
            telemetry=telemetry,
            workspace_root=workspace_root,
            worker_id=worker_id,
            lease_token=lease_token,
            lease_seconds=lease_seconds,
            clock=clock,
            stop_after_stage=checkpoint_stage,
        )

    def execute_research_only_g1(
        self,
        *,
        run_spec: ResearchRunSpec,
        execution_bindings: ResearchExecutionBindings,
        agent_policy: AgentContextPolicy,
        scientific_lineage_manifest: ResearchScientificLineageManifestV2,
        implementations: ResearchOnlyG1Implementations,
        semantic_authorities: ResearchOnlyG1SemanticAuthorities,
        runtime: ExperimentRuntime,
        telemetry: TelemetryStore,
        workspace_root: str | Path,
        worker_id: str,
        lease_token: str,
        lease_seconds: int = 60,
        clock: Callable[[], datetime] | None = None,
        registered_at: datetime | None = None,
        stop_after_stage: ResearchStage | str | None = None,
    ) -> ExperimentRunSummary:
        """Execute a lineage-bound G1 run without any G2 capability.

        The method has no approval or terminal-evaluator argument.  It rejects
        protected partitions and admission before any registry write, then
        atomically registers only the v3 lineage component and experiment.
        """

        if not isinstance(agent_policy, AgentContextPolicy):
            raise TypeError("agent_policy must be AgentContextPolicy")
        if type(implementations) is not ResearchOnlyG1Implementations:
            raise TypeError(
                "implementations must be an exact ResearchOnlyG1Implementations"
            )
        if not isinstance(
            scientific_lineage_manifest,
            (ResearchScientificLineageManifestV1, ResearchScientificLineageManifestV2),
        ):
            raise TypeError("research-only G1 requires a scientific lineage manifest")
        implementations.validate_for(run_spec)
        _validate_research_only_g1_semantic_authorities(
            run_spec=run_spec,
            agent_policy=agent_policy,
            configured_factor_registry=self.factor_registry,
            authorities=semantic_authorities,
        )
        model_training_enabled = ResearchStage.MODEL_TRAINING in run_spec.enabled_stages
        if model_training_enabled:
            if not isinstance(
                scientific_lineage_manifest, ResearchScientificLineageManifestV2
            ):
                raise ResearchAssemblyError(
                    "NESTED_SELECTION_LINEAGE_REQUIRED",
                    "model_training requires scientific lineage v2",
                )
            if (
                implementations.model_training_selection_spec_hash()
                != scientific_lineage_manifest.nested_selection_spec_hash
            ):
                raise ResearchAssemblyError(
                    "MODEL_TRAINING_SELECTION_SPEC_MISMATCH",
                    scientific_lineage_manifest.nested_selection_spec_hash,
                )
            implementations.validate_research_readiness_lineage(
                run_spec=run_spec,
                scientific_lineage_manifest=scientific_lineage_manifest,
            )
            if self._nested_selection_authority is None:
                raise ResearchAssemblyError(
                    "NESTED_SELECTION_AUTHORITY_UNCONFIGURED",
                    "model_training",
                )
        _validate_scientific_lineage_policy(
            scientific_lineage_manifest=scientific_lineage_manifest,
            agent_policy=agent_policy,
        )
        plan = self.plan(
            run_spec,
            execution_bindings,
            market_logic_hashes=agent_policy.hypothesis_hashes,
            scientific_lineage_manifest=scientific_lineage_manifest,
        )
        plan.require_ready()
        if plan.experiment_spec.schema_version != "experiment-spec/v3":
            raise ResearchAssemblyError(
                "RESEARCH_ONLY_LINEAGE_V3_REQUIRED",
                plan.experiment_spec.schema_version,
            )
        _validate_factor_policy_bindings(
            run_spec=run_spec,
            agent_policy=agent_policy,
            factor_registry=self.factor_registry,
        )
        if runtime.spec.content_hash != plan.experiment_spec.content_hash:
            raise ResearchAssemblyError(
                "RUNTIME_SPEC_MISMATCH", plan.experiment_spec.content_hash
            )
        if telemetry.experiment_spec_hash != plan.experiment_spec.content_hash:
            raise ResearchAssemblyError(
                "TELEMETRY_SPEC_MISMATCH", plan.experiment_spec.content_hash
            )
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable or None")
        checkpoint_stage: ResearchStage | None = None
        if stop_after_stage is not None:
            checkpoint_stage = ResearchStage(stop_after_stage)
            if checkpoint_stage is not ResearchStage.MODEL_TRAINING:
                raise ResearchAssemblyError(
                    "UNSUPPORTED_CHECKPOINT_BOUNDARY",
                    checkpoint_stage.value,
                )
            if not model_training_enabled:
                raise ResearchAssemblyError(
                    "MODEL_TRAINING_CHECKPOINT_UNEXPECTED",
                    "model_training",
                )
        _validate_worker_lease_inputs(
            worker_id=worker_id,
            lease_token=lease_token,
            lease_seconds=lease_seconds,
        )
        readiness_managed = _requires_resolved_research_readiness(run_spec)
        if readiness_managed and checkpoint_stage is None:
            model_success = runtime.successful_attempt(
                ResearchStage.MODEL_TRAINING.value
            )
            if model_success is None:
                raise ResearchAssemblyError(
                    "MODEL_TRAINING_CHECKPOINT_REQUIRED",
                    "first managed execution must stop after model_training",
                )

        scientific_implementations = dict(implementations.scientific_implementations)
        if model_training_enabled:
            authority = self._nested_selection_authority
            if authority is None:  # pragma: no cover - guarded above.
                raise RuntimeError("nested selection authority disappeared")
            scientific_implementations[ResearchStage.MODEL_TRAINING] = (
                implementations.bind_model_training(
                    nested_selection=NestedSelectionResearchAuthority.agent_controller(
                        authority
                    ),
                    run_spec=run_spec,
                    scientific_lineage_manifest=scientific_lineage_manifest,
                    experiment_spec=plan.experiment_spec,
                    registry=self.experiment_registry,
                    runtime=runtime,
                    workspace_root=workspace_root,
                    clock=clock,
                )
            )
            if readiness_managed:
                artifact_store = ArtifactStore(workspace_root)
                outer_audit_reader = authority.audit_reader()
                if checkpoint_stage is None:
                    score_factory = implementations.controlled_score_factory
                    if score_factory is None:  # pragma: no cover - validate_for.
                        raise RuntimeError("controlled score factory disappeared")
                    try:
                        score_factory.preflight_outer_completion(
                            experiment_spec=plan.experiment_spec,
                            registry=self.experiment_registry,
                            runtime=runtime,
                            artifact_store=artifact_store,
                            outer_audit_reader=outer_audit_reader,
                        )
                    except ResearchReadinessScoreStageError as exc:
                        raise ResearchAssemblyError(
                            "CONTROLLED_SCORE_PREFLIGHT_FAILED",
                            exc.code,
                        ) from exc
                scientific_implementations[ResearchStage.SCORE_CONSTRUCTION] = (
                    implementations.bind_controlled_score(
                        run_spec=run_spec,
                        scientific_lineage_manifest=scientific_lineage_manifest,
                        experiment_spec=plan.experiment_spec,
                        registry=self.experiment_registry,
                        runtime=runtime,
                        artifact_store=artifact_store,
                        outer_audit_reader=outer_audit_reader,
                        worker_id=worker_id,
                        lease_token=lease_token,
                        clock=clock,
                    )
                )
                scientific_implementations[ResearchStage.ROBUSTNESS] = (
                    implementations.bind_research_readiness(
                        run_spec=run_spec,
                        scientific_lineage_manifest=scientific_lineage_manifest,
                        experiment_spec=plan.experiment_spec,
                        registry=self.experiment_registry,
                        runtime=runtime,
                        artifact_store=artifact_store,
                        outer_audit_reader=outer_audit_reader,
                        worker_id=worker_id,
                        lease_token=lease_token,
                        clock=clock,
                    )
                )
        agent_context = agent_policy.bind(plan.experiment_spec)
        handlers = build_governed_stage_handlers(
            run_spec=run_spec,
            experiment_spec=plan.experiment_spec,
            agent_context=agent_context,
            role_adapters=implementations.role_adapters,
            scientific_implementations=scientific_implementations,
            terminal_evaluator=None,
        )
        self.experiment_registry.register_component_and_experiment(
            plan.experiment_spec,
            component_hash=scientific_lineage_manifest.content_hash,
            component_type="scientific_lineage",
            descriptor={
                "content_hash": scientific_lineage_manifest.content_hash,
                "schema_version": scientific_lineage_manifest.schema_version,
                "manifest": scientific_lineage_manifest.to_dict(),
            },
            registered_at=registered_at,
        )
        return _run_research_only_g1(
            run_spec=run_spec,
            experiment_spec=plan.experiment_spec,
            handlers=handlers,
            registry=self.experiment_registry,
            runtime=runtime,
            telemetry=telemetry,
            workspace_root=workspace_root,
            worker_id=worker_id,
            lease_token=lease_token,
            lease_seconds=lease_seconds,
            clock=clock,
            terminal_evaluator=None,
            stop_after_stage=checkpoint_stage,
        )


def plan_research_run_read_only(
    run_spec: ResearchRunSpec,
    execution_bindings: ResearchExecutionBindings,
    *,
    registry_paths: ReadOnlyRegistryPaths,
    scientific_lineage_manifest: ResearchScientificLineageManifest | None = None,
) -> ResearchAssemblyPlan:
    with ReadOnlyResearchRegistryLookup(registry_paths) as lookup:
        return _build_plan(
            run_spec,
            execution_bindings,
            lookup=lookup,
            scientific_lineage_manifest=scientific_lineage_manifest,
        )


def load_research_run_spec(path: str | Path) -> ResearchRunSpec:
    return ResearchRunSpec.from_mapping(_load_strict_json_object(path))


def load_research_scientific_lineage_manifest(
    path: str | Path,
) -> ResearchScientificLineageManifest:
    value = _load_strict_json_object(path)
    schema_version = value.get("schema_version")
    if schema_version == "research-scientific-lineage-manifest/v1":
        manifest: ResearchScientificLineageManifest = (
            ResearchScientificLineageManifestV1.from_mapping(value)
        )
    elif schema_version == "research-scientific-lineage-manifest/v2":
        manifest = ResearchScientificLineageManifestV2.from_mapping(value)
    else:
        raise ValueError("unsupported research scientific lineage schema")
    if canonical_json_bytes(value) != canonical_json_bytes(manifest.to_dict()):
        raise ValueError("research scientific lineage manifest is not canonical")
    return manifest


def load_research_execution_bindings(
    path: str | Path,
) -> ResearchExecutionBindings:
    value = _load_strict_json_object(path)
    expected = {
        "schema_version",
        "code_snapshot_hash",
        "environment_hash",
        "random_seed",
        "resource_budget",
        "retry_policy",
        "llm_policy_hash",
        "llm_transport_hash",
    }
    if set(value) != expected:
        raise ValueError("ResearchExecutionBindings wire fields differ")
    budget_value = _mapping(value["resource_budget"], name="resource_budget")
    retry_value = _mapping(value["retry_policy"], name="retry_policy")
    budget_expected = {
        "schema_version",
        "maximum_attempts",
        "maximum_wall_seconds",
        "maximum_cpu_seconds",
        "maximum_peak_memory_bytes",
        "maximum_disk_write_bytes",
        "maximum_parallel_tasks",
        "per_stage_timeout_seconds",
        "maximum_llm_calls",
        "maximum_llm_tokens",
        "maximum_llm_cost_microusd",
    }
    retry_expected = {
        "schema_version",
        "maximum_attempts_per_stage",
        "initial_backoff_seconds",
        "maximum_backoff_seconds",
        "backoff_multiplier",
        "jitter_fraction",
        "retryable_failure_codes",
    }
    if set(budget_value) != budget_expected:
        raise ValueError("ResourceBudget wire fields differ")
    if set(retry_value) != retry_expected:
        raise ValueError("RetryPolicy wire fields differ")
    retry_codes = retry_value["retryable_failure_codes"]
    if not isinstance(retry_codes, list) or not all(
        isinstance(item, str) for item in retry_codes
    ):
        raise TypeError("RetryPolicy retryable_failure_codes must be strings")
    result = ResearchExecutionBindings(
        schema_version=_string(value["schema_version"], name="schema_version"),
        code_snapshot_hash=_string(
            value["code_snapshot_hash"], name="code_snapshot_hash"
        ),
        environment_hash=_string(value["environment_hash"], name="environment_hash"),
        random_seed=_integer(value["random_seed"], name="random_seed"),
        resource_budget=ResourceBudget(
            schema_version=_string(
                budget_value["schema_version"],
                name="resource_budget.schema_version",
            ),
            maximum_attempts=_integer(
                budget_value["maximum_attempts"], name="maximum_attempts"
            ),
            maximum_wall_seconds=_number(
                budget_value["maximum_wall_seconds"], name="maximum_wall_seconds"
            ),
            maximum_cpu_seconds=_number(
                budget_value["maximum_cpu_seconds"], name="maximum_cpu_seconds"
            ),
            maximum_peak_memory_bytes=_integer(
                budget_value["maximum_peak_memory_bytes"],
                name="maximum_peak_memory_bytes",
            ),
            maximum_disk_write_bytes=_integer(
                budget_value["maximum_disk_write_bytes"],
                name="maximum_disk_write_bytes",
            ),
            maximum_parallel_tasks=_integer(
                budget_value["maximum_parallel_tasks"],
                name="maximum_parallel_tasks",
            ),
            per_stage_timeout_seconds=_number(
                budget_value["per_stage_timeout_seconds"],
                name="per_stage_timeout_seconds",
            ),
            maximum_llm_calls=_integer(
                budget_value["maximum_llm_calls"], name="maximum_llm_calls"
            ),
            maximum_llm_tokens=_integer(
                budget_value["maximum_llm_tokens"], name="maximum_llm_tokens"
            ),
            maximum_llm_cost_microusd=_integer(
                budget_value["maximum_llm_cost_microusd"],
                name="maximum_llm_cost_microusd",
            ),
        ),
        retry_policy=RetryPolicy(
            schema_version=_string(
                retry_value["schema_version"], name="retry_policy.schema_version"
            ),
            maximum_attempts_per_stage=_integer(
                retry_value["maximum_attempts_per_stage"],
                name="maximum_attempts_per_stage",
            ),
            initial_backoff_seconds=_number(
                retry_value["initial_backoff_seconds"],
                name="initial_backoff_seconds",
            ),
            maximum_backoff_seconds=_number(
                retry_value["maximum_backoff_seconds"],
                name="maximum_backoff_seconds",
            ),
            backoff_multiplier=_number(
                retry_value["backoff_multiplier"], name="backoff_multiplier"
            ),
            jitter_fraction=_number(
                retry_value["jitter_fraction"], name="jitter_fraction"
            ),
            retryable_failure_codes=tuple(retry_codes),
        ),
        llm_policy_hash=_optional_string(
            value["llm_policy_hash"], name="llm_policy_hash"
        ),
        llm_transport_hash=_optional_string(
            value["llm_transport_hash"], name="llm_transport_hash"
        ),
    )
    if canonical_json_bytes(value) != canonical_json_bytes(result.to_dict()):
        raise ValueError("ResearchExecutionBindings is not canonical")
    return result


class _StrictArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ResearchAssemblyError("INVALID_ARGUMENTS", message)


def research_cli_main(argv: Sequence[str] | None = None) -> int:
    parser = _StrictArgumentParser(
        prog="python -m alpha_research research",
        description="Read-only governed research planning.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser(
        "plan",
        help="validate immutable specs and registry bindings without execution",
    )
    plan.add_argument("--run-spec", required=True)
    plan.add_argument("--execution-bindings", required=True)
    plan.add_argument("--experiment-registry", required=True)
    plan.add_argument("--factor-registry")
    plan.add_argument("--model-registry")
    plan.add_argument("--market-logic-registry")
    plan.add_argument("--scientific-lineage-manifest")
    try:
        args = parser.parse_args(argv)
        if args.command != "plan":  # pragma: no cover - argparse closes this
            raise ResearchAssemblyError("INVALID_COMMAND", str(args.command))
        result = plan_research_run_read_only(
            load_research_run_spec(args.run_spec),
            load_research_execution_bindings(args.execution_bindings),
            scientific_lineage_manifest=(
                None
                if args.scientific_lineage_manifest is None
                else load_research_scientific_lineage_manifest(
                    args.scientific_lineage_manifest
                )
            ),
            registry_paths=ReadOnlyRegistryPaths(
                experiment=Path(args.experiment_registry),
                factor=(
                    None if args.factor_registry is None else Path(args.factor_registry)
                ),
                model=(
                    None if args.model_registry is None else Path(args.model_registry)
                ),
                market_logic=(
                    None
                    if args.market_logic_registry is None
                    else Path(args.market_logic_registry)
                ),
            ),
        )
        _write_json(result.to_dict())
        return 0 if result.ready_to_register else 1
    except (
        OSError,
        TypeError,
        ValueError,
        sqlite3.DatabaseError,
        ResearchAssemblyError,
    ) as error:
        code = (
            error.code
            if isinstance(error, ResearchAssemblyError)
            else "INVALID_PLAN_INPUT"
        )
        _write_json(
            {
                "schema_version": "research-assembly-cli-error/v1",
                "decision": "INVALID_PLAN_REQUEST",
                "error_code": code,
                "error_type": type(error).__name__,
                "assurance_boundary": {
                    "read_only": True,
                    "writes_performed": False,
                    "data_opened": False,
                    "scientific_code_executed": False,
                    "dynamic_callable_imports": False,
                    "experiment_registered": False,
                },
            }
        )
        return 2


def experiment_component_type_for_role(role: str) -> str:
    if role.startswith("data:"):
        return "data"
    if role.startswith("factor:"):
        return "factor"
    try:
        return _COMPONENT_TYPE_BY_ROLE[role]
    except KeyError:
        raise ResearchAssemblyError(
            "UNSUPPORTED_RESEARCH_COMPONENT_ROLE", role
        ) from None


def _build_plan(
    run_spec: ResearchRunSpec,
    execution_bindings: ResearchExecutionBindings,
    *,
    lookup: _RegistryLookup,
    market_logic_hashes: Sequence[str] = (),
    scientific_lineage_manifest: ResearchScientificLineageManifest | None = None,
) -> ResearchAssemblyPlan:
    if not isinstance(run_spec, ResearchRunSpec):
        raise TypeError("run_spec must be ResearchRunSpec")
    if not isinstance(execution_bindings, ResearchExecutionBindings):
        raise TypeError("execution_bindings must be ResearchExecutionBindings")
    if scientific_lineage_manifest is not None and not isinstance(
        scientific_lineage_manifest,
        (ResearchScientificLineageManifestV1, ResearchScientificLineageManifestV2),
    ):
        raise TypeError("scientific_lineage_manifest has an unsupported manifest type")
    if scientific_lineage_manifest is not None:
        scientific_lineage_manifest.validate_for(run_spec)
    experiment_spec = bridge_research_run(
        run_spec,
        execution_bindings,
        scientific_lineage_manifest_hash=(
            None
            if scientific_lineage_manifest is None
            else scientific_lineage_manifest.content_hash
        ),
    )
    checks: list[RegistryBindingCheck] = []
    if not lookup.experiment_integrity_ok():
        checks.append(
            RegistryBindingCheck(
                registry="experiment",
                role="registry:integrity",
                expected_hash=experiment_spec.content_hash,
                status="registry_integrity_failed",
                required=True,
            )
        )

    requirements = dict(run_spec.component_bindings())
    requirements.update(
        {
            "code": execution_bindings.code_snapshot_hash,
            "environment": execution_bindings.environment_hash,
        }
    )
    if execution_bindings.llm_policy_hash is not None:
        requirements["llm_policy"] = execution_bindings.llm_policy_hash
    if execution_bindings.llm_transport_hash is not None:
        requirements["llm_transport"] = execution_bindings.llm_transport_hash
    for role, digest in sorted(requirements.items()):
        expected_type = experiment_component_type_for_role(role)
        observed = lookup.experiment_component(digest)
        if observed is None:
            status = "not_registered"
            observed_type = None
        else:
            observed_type, valid_descriptor = observed
            if not valid_descriptor:
                status = "stored_payload_invalid"
            elif observed_type != expected_type:
                status = "component_type_mismatch"
            else:
                status = _PRESENT
        checks.append(
            RegistryBindingCheck(
                registry="experiment",
                role=role,
                expected_hash=digest,
                status=status,
                required=True,
                expected_component_type=expected_type,
                observed_component_type=observed_type,
            )
        )

    for offset, digest in enumerate(run_spec.factor_spec_hashes):
        role = f"factor:{offset}"
        result = lookup.factor_spec(digest)
        if result is None:
            checks.append(
                RegistryBindingCheck(
                    registry="factor",
                    role=role,
                    expected_hash=digest,
                    status="registry_not_configured",
                    required=False,
                )
            )
            continue
        if isinstance(result, str):
            status = result
        else:
            spec, lifecycle = result
            if spec.snapshot_id != run_spec.data_snapshot_hash:
                status = "cross_binding_mismatch"
            elif lifecycle in {"rejected", "deprecated"}:
                status = "factor_lifecycle_forbidden"
            else:
                status = _PRESENT
        checks.append(
            RegistryBindingCheck(
                registry="factor",
                role=role,
                expected_hash=digest,
                status=status,
                required=True,
            )
        )

    if run_spec.model_spec_hash is not None:
        model = lookup.model_spec(run_spec.model_spec_hash)
        if model is None:
            checks.append(
                RegistryBindingCheck(
                    registry="model",
                    role="model",
                    expected_hash=run_spec.model_spec_hash,
                    status="registry_not_configured",
                    required=False,
                )
            )
        else:
            if isinstance(model, str):
                status = model
            elif (
                model.label_spec_hash != run_spec.label_spec_hash
                or model.validation_spec_hash != run_spec.validation_spec_hash
            ):
                status = "cross_binding_mismatch"
            else:
                status = _PRESENT
            checks.append(
                RegistryBindingCheck(
                    registry="model",
                    role="model",
                    expected_hash=run_spec.model_spec_hash,
                    status=status,
                    required=True,
                )
            )

    lineage_status, lineage_checks = _validate_scientific_lineage_registry(
        run_spec=run_spec,
        manifest=scientific_lineage_manifest,
        lookup=lookup,
    )
    checks.extend(lineage_checks)

    explicit_logic_hashes = tuple(market_logic_hashes)
    if len(set(explicit_logic_hashes)) != len(explicit_logic_hashes):
        raise ValueError("market logic hashes must be unique")
    if (
        scientific_lineage_manifest is not None
        and explicit_logic_hashes
        and set(explicit_logic_hashes)
        != set(scientific_lineage_manifest.market_logic_hashes)
    ):
        raise ResearchAssemblyError(
            "SCIENTIFIC_LINEAGE_LOGIC_SET_MISMATCH",
            "explicit MarketLogic hashes differ from manifest",
        )
    logic_hashes = explicit_logic_hashes
    if not logic_hashes and scientific_lineage_manifest is not None:
        logic_hashes = scientific_lineage_manifest.market_logic_hashes
    for offset, digest in enumerate(logic_hashes):
        require_sha256(digest, name=f"market logic hash:{offset}")
        result = lookup.market_logic(digest)
        status = "registry_not_configured" if result is None else result
        checks.append(
            RegistryBindingCheck(
                registry="market_logic",
                role=f"hypothesis:{offset}",
                expected_hash=digest,
                status=status,
                required=(
                    bool(explicit_logic_hashes)
                    or _scientific_lineage_required(run_spec)
                    or lookup.configuration["market_logic"]
                ),
            )
        )

    return ResearchAssemblyPlan(
        research_run_spec_hash=run_spec.content_hash,
        execution_bindings_hash=execution_bindings.content_hash,
        experiment_spec=experiment_spec,
        registry_checks=tuple(checks),
        registry_configuration=lookup.configuration,
        scientific_lineage_manifest_hash=(
            None
            if scientific_lineage_manifest is None
            else scientific_lineage_manifest.content_hash
        ),
        scientific_lineage_status=lineage_status,
    )


def _validate_scientific_lineage_registry(
    *,
    run_spec: ResearchRunSpec,
    manifest: ResearchScientificLineageManifest | None,
    lookup: _RegistryLookup,
) -> tuple[str, tuple[RegistryBindingCheck, ...]]:
    required = _scientific_lineage_required(run_spec)
    if manifest is None:
        return ("required_missing" if required else "not_required"), ()
    if ResearchStage.MODEL_TRAINING in run_spec.enabled_stages and not isinstance(
        manifest, ResearchScientificLineageManifestV2
    ):
        return "model_selection_missing", ()

    deferred = False
    binding_invalid = False
    model_invalid = False
    checks: list[RegistryBindingCheck] = []
    for offset, reference in enumerate(manifest.factor_logic_bindings):
        result = lookup.market_logic_binding(reference.registry_binding_hash)
        if result is None:
            status = "registry_not_configured"
            check_required = required
            if required:
                binding_invalid = True
            else:
                deferred = True
        elif isinstance(result, str):
            status = result
            check_required = True
            binding_invalid = True
        elif (
            result.factor_hash != reference.factor_spec_hash
            or result.logic_hash != reference.market_logic_hash
            or result.content_hash != reference.registry_binding_hash
        ):
            status = "cross_binding_mismatch"
            check_required = True
            binding_invalid = True
        else:
            status = _PRESENT
            check_required = True
        checks.append(
            RegistryBindingCheck(
                registry="market_logic",
                role=f"lineage:factor:{offset}",
                expected_hash=reference.registry_binding_hash,
                status=status,
                required=check_required,
            )
        )

    if run_spec.model_spec_hash is not None:
        model = lookup.model_spec(run_spec.model_spec_hash)
        if model is None:
            if required:
                model_invalid = True
            else:
                deferred = True
        elif isinstance(model, str):
            model_invalid = True
        else:
            expected = {
                item.feature_name: item.signal_hash
                for item in manifest.model_feature_bindings
            }
            if expected != dict(model.feature_signal_hashes):
                model_invalid = True

    if binding_invalid:
        return "registry_binding_invalid", tuple(checks)
    if model_invalid:
        return "model_feature_mismatch", tuple(checks)
    if deferred:
        return "validation_deferred", tuple(checks)
    return "validated", tuple(checks)


def _scientific_lineage_required(run_spec: ResearchRunSpec) -> bool:
    return (
        ExperimentProfile(run_spec.profile) is ExperimentProfile.PRODUCTION_CANDIDATE
        or ResearchStage.ADMISSION in run_spec.enabled_stages
    )


def _validate_scientific_lineage_policy(
    *,
    scientific_lineage_manifest: ResearchScientificLineageManifest | None,
    agent_policy: AgentContextPolicy,
) -> None:
    if scientific_lineage_manifest is None:
        return
    if set(scientific_lineage_manifest.market_logic_hashes) != set(
        agent_policy.hypothesis_hashes
    ):
        raise ResearchAssemblyError(
            "SCIENTIFIC_LINEAGE_POLICY_MISMATCH",
            "AgentContextPolicy hypothesis hashes differ from manifest",
        )


def _validate_worker_lease_inputs(
    *,
    worker_id: str,
    lease_token: str,
    lease_seconds: int,
) -> None:
    if (
        not isinstance(worker_id, str)
        or not 1 <= len(worker_id) <= 128
        or not worker_id[0].isalnum()
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
            for character in worker_id
        )
    ):
        raise ValueError("worker_id is invalid")
    if not isinstance(lease_token, str) or len(lease_token) < 16:
        raise ValueError("lease_token must contain at least 16 characters")
    if (
        not isinstance(lease_seconds, int)
        or isinstance(lease_seconds, bool)
        or lease_seconds <= 0
    ):
        raise ValueError("lease_seconds must be a positive integer")


def _validate_factor_policy_bindings(
    *,
    run_spec: ResearchRunSpec,
    agent_policy: AgentContextPolicy,
    factor_registry: FactorRegistry | None,
) -> None:
    if factor_registry is None:
        return
    allowed_fields = set(agent_policy.allowed_fields)
    schemas = set(agent_policy.data_schema_hashes)
    failures: list[str] = []
    for offset, digest in enumerate(run_spec.factor_spec_hashes):
        spec = factor_registry.get(digest)
        if spec.schema_hash not in schemas:
            failures.append(f"factor:{offset}:schema")
        if not set(spec.required_fields).issubset(allowed_fields):
            failures.append(f"factor:{offset}:fields")
    if failures:
        raise ResearchAssemblyError("FACTOR_AGENT_POLICY_MISMATCH", ",".join(failures))


def _validate_research_only_g1_semantic_authorities(
    *,
    run_spec: ResearchRunSpec,
    agent_policy: AgentContextPolicy,
    configured_factor_registry: FactorRegistry | None,
    authorities: ResearchOnlyG1SemanticAuthorities,
) -> None:
    """Reload and cross-bind every G1 scientific identity before any write."""

    if type(authorities) is not ResearchOnlyG1SemanticAuthorities:
        raise TypeError(
            "semantic_authorities must be an exact ResearchOnlyG1SemanticAuthorities"
        )
    if configured_factor_registry is None:
        raise ResearchAssemblyError(
            "RESEARCH_ONLY_FACTOR_REGISTRY_REQUIRED",
            "factor_registry",
        )
    if authorities.factor_registry is not configured_factor_registry:
        raise ResearchAssemblyError(
            "RESEARCH_ONLY_FACTOR_REGISTRY_AUTHORITY_MISMATCH",
            "factor_registry",
        )
    if dict(authorities.sealed_partition_hashes) != dict(
        run_spec.data_partition_hashes
    ):
        raise ResearchAssemblyError(
            "RESEARCH_ONLY_PARTITION_AUTHORITY_MISMATCH",
            "sealed_partition_hashes",
        )

    factor_specs: list[FactorSpec] = []
    for offset, digest in enumerate(run_spec.factor_spec_hashes):
        try:
            spec = configured_factor_registry.get(digest)
            lifecycle = configured_factor_registry.current_state(digest)
        except KeyError as exc:
            raise ResearchAssemblyError(
                "RESEARCH_ONLY_FACTOR_NOT_REGISTERED",
                f"factor:{offset}",
            ) from exc
        except (TypeError, ValueError, RuntimeError, sqlite3.DatabaseError) as exc:
            raise ResearchAssemblyError(
                "RESEARCH_ONLY_FACTOR_RELOAD_INVALID",
                f"factor:{offset}",
            ) from exc
        if type(spec) is not FactorSpec or spec.content_hash != digest:
            raise ResearchAssemblyError(
                "RESEARCH_ONLY_FACTOR_RELOAD_INVALID",
                f"factor:{offset}",
            )
        if lifecycle in {"rejected", "deprecated"}:
            raise ResearchAssemblyError(
                "RESEARCH_ONLY_FACTOR_LIFECYCLE_FORBIDDEN",
                f"factor:{offset}:{lifecycle}",
            )
        factor_specs.append(spec)

    snapshot = authorities.dataset_snapshot
    request = authorities.data_request
    label = authorities.label_spec
    validation = authorities.validation_spec
    evaluation = authorities.evaluation_spec
    if snapshot.snapshot_id != run_spec.data_snapshot_hash:
        raise ResearchAssemblyError(
            "RESEARCH_ONLY_SNAPSHOT_HASH_MISMATCH",
            run_spec.data_snapshot_hash,
        )
    if hash_json(request.to_dict()) != run_spec.data_request_hash:
        raise ResearchAssemblyError(
            "RESEARCH_ONLY_DATA_REQUEST_HASH_MISMATCH",
            run_spec.data_request_hash,
        )
    if hash_json(authorities.data_quality_policy.to_dict()) != (
        run_spec.data_quality_spec_hash
    ):
        raise ResearchAssemblyError(
            "RESEARCH_ONLY_DATA_QUALITY_HASH_MISMATCH",
            run_spec.data_quality_spec_hash,
        )
    if label.content_hash != run_spec.label_spec_hash:
        raise ResearchAssemblyError(
            "RESEARCH_ONLY_LABEL_HASH_MISMATCH",
            run_spec.label_spec_hash,
        )
    if validation.content_hash != run_spec.validation_spec_hash:
        raise ResearchAssemblyError(
            "RESEARCH_ONLY_VALIDATION_HASH_MISMATCH",
            run_spec.validation_spec_hash,
        )
    if evaluation.content_hash != run_spec.evaluation_spec_hash:
        raise ResearchAssemblyError(
            "RESEARCH_ONLY_EVALUATION_HASH_MISMATCH",
            run_spec.evaluation_spec_hash,
        )

    pit_hash = security_data_contract_hash(authorities.security_contract)
    anchor = factor_specs[0]
    factor_identity = (
        anchor.dataset_id,
        anchor.snapshot_id,
        anchor.schema_hash,
        anchor.availability_hash,
        anchor.security_contract_hash,
        anchor.frequency.content_hash,
    )
    for offset, spec in enumerate(factor_specs[1:], start=1):
        observed = (
            spec.dataset_id,
            spec.snapshot_id,
            spec.schema_hash,
            spec.availability_hash,
            spec.security_contract_hash,
            spec.frequency.content_hash,
        )
        if observed != factor_identity:
            raise ResearchAssemblyError(
                "RESEARCH_ONLY_FACTOR_DATA_IDENTITY_MISMATCH",
                f"factor:{offset}",
            )
    expected_label_identity = (
        anchor.dataset_id,
        anchor.snapshot_id,
        anchor.schema_hash,
        anchor.availability_hash,
        anchor.security_contract_hash,
        anchor.frequency.content_hash,
    )
    observed_label_identity = (
        label.dataset_id,
        label.snapshot_id,
        label.schema_hash,
        label.availability_hash,
        label.security_contract_hash,
        label.frequency.content_hash,
    )
    if observed_label_identity != expected_label_identity:
        raise ResearchAssemblyError(
            "RESEARCH_ONLY_LABEL_DATA_IDENTITY_MISMATCH",
            label.label_id,
        )
    if (
        snapshot.dataset_id != anchor.dataset_id
        or snapshot.schema_hash != anchor.schema_hash
        or snapshot.snapshot_id != anchor.snapshot_id
    ):
        raise ResearchAssemblyError(
            "RESEARCH_ONLY_SNAPSHOT_DATA_IDENTITY_MISMATCH",
            snapshot.snapshot_id,
        )
    if (
        request.dataset_id != anchor.dataset_id
        or request.snapshot_id != anchor.snapshot_id
        or request.frequency.content_hash != anchor.frequency.content_hash
    ):
        raise ResearchAssemblyError(
            "RESEARCH_ONLY_REQUEST_DATA_IDENTITY_MISMATCH",
            run_spec.data_request_hash,
        )
    if pit_hash != anchor.security_contract_hash:
        raise ResearchAssemblyError(
            "RESEARCH_ONLY_PIT_AUTHORITY_MISMATCH",
            anchor.security_contract_hash,
        )

    required_fields = set().union(*(set(spec.required_fields) for spec in factor_specs))
    required_fields.add(label.price_field)
    required_fields.update(
        field
        for field in (label.adjustment_field, label.amount_field)
        if field is not None
    )
    missing_request_fields = sorted(required_fields.difference(request.fields))
    if missing_request_fields:
        raise ResearchAssemblyError(
            "RESEARCH_ONLY_REQUEST_FIELDS_MISSING",
            ",".join(missing_request_fields),
        )
    if anchor.schema_hash not in set(agent_policy.data_schema_hashes):
        raise ResearchAssemblyError(
            "RESEARCH_ONLY_AGENT_SCHEMA_MISMATCH",
            anchor.schema_hash,
        )
    missing_agent_fields = sorted(
        required_fields.difference(agent_policy.allowed_fields)
    )
    if missing_agent_fields:
        raise ResearchAssemblyError(
            "RESEARCH_ONLY_AGENT_FIELDS_MISSING",
            ",".join(missing_agent_fields),
        )

    window_starts = [
        pd.Timestamp(window[0]) for window in run_spec.data_partition_windows.values()
    ]
    window_ends = [
        pd.Timestamp(window[1]) for window in run_spec.data_partition_windows.values()
    ]
    if pd.Timestamp(request.start) > min(window_starts) or pd.Timestamp(
        request.end
    ) < max(window_ends):
        raise ResearchAssemblyError(
            "RESEARCH_ONLY_REQUEST_WINDOW_INCOMPLETE",
            run_spec.data_request_hash,
        )
    discovery_start = pd.Timestamp(
        run_spec.data_partition_windows[DataPartition.DISCOVERY][0]
    )
    validation_start, validation_end = (
        pd.Timestamp(value)
        for value in run_spec.data_partition_windows[DataPartition.VALIDATION]
    )
    for fold in validation.folds:
        if (
            pd.Timestamp(fold.train_start) < discovery_start
            or pd.Timestamp(fold.train_end) >= pd.Timestamp(fold.validation_start)
            or pd.Timestamp(fold.validation_start) < validation_start
            or pd.Timestamp(fold.validation_end) > validation_end
        ):
            raise ResearchAssemblyError(
                "RESEARCH_ONLY_VALIDATION_WINDOW_MISMATCH",
                fold.fold_id,
            )


def _component_row(row: sqlite3.Row | None, digest: str) -> tuple[str, bool] | None:
    if row is None:
        return None
    try:
        descriptor = _json_object_text(row["descriptor_json"])
        valid = descriptor.get("content_hash") == digest
    except (TypeError, ValueError, json.JSONDecodeError):
        valid = False
    return str(row["component_type"]), valid


def _open_read_only_registry(path: Path) -> sqlite3.Connection:
    absolute = path.absolute()
    metadata = os.stat(absolute, follow_symlinks=False)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ResearchAssemblyError("REGISTRY_PATH_NOT_REGULAR_FILE", absolute.name)
    uri = f"file:{quote(os.fspath(absolute), safe='/')}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _require_registry_schema(
    connection: sqlite3.Connection, expected: str, *, name: str
) -> None:
    rows = connection.execute(
        "SELECT key,value FROM registry_metadata ORDER BY key"
    ).fetchall()
    metadata = {str(row["key"]): str(row["value"]) for row in rows}
    if metadata != {"schema_version": expected}:
        raise ResearchAssemblyError("REGISTRY_SCHEMA_MISMATCH", name)


def _load_strict_json_object(path: str | Path) -> Mapping[str, object]:
    source = Path(path)
    metadata = os.stat(source, follow_symlinks=False)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ResearchAssemblyError("JSON_PATH_NOT_REGULAR_FILE", source.name)
    if metadata.st_size > _MAX_JSON_BYTES:
        raise ResearchAssemblyError("JSON_INPUT_TOO_LARGE", source.name)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source, flags)
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise ResearchAssemblyError("JSON_PATH_NOT_REGULAR_FILE", source.name)
        payload = bytearray()
        while len(payload) <= _MAX_JSON_BYTES:
            chunk = os.read(
                descriptor, min(64 * 1024, _MAX_JSON_BYTES + 1 - len(payload))
            )
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > _MAX_JSON_BYTES:
            raise ResearchAssemblyError("JSON_INPUT_TOO_LARGE", source.name)
    finally:
        os.close(descriptor)
    value = json.loads(
        bytes(payload).decode("utf-8"),
        object_pairs_hook=_unique_json_object,
        parse_constant=lambda token: (_raise_nonfinite_json(token)),
    )
    if not isinstance(value, Mapping):
        raise TypeError("JSON root must be an object")
    return dict(value)


def _unique_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key:{key}")
        result[key] = value
    return result


def _raise_nonfinite_json(token: str) -> object:
    raise ValueError(f"non-finite JSON number:{token}")


def _json_object_text(value: str) -> Mapping[str, object]:
    decoded = json.loads(value)
    if not isinstance(decoded, Mapping):
        raise TypeError("stored registry payload is not an object")
    return dict(decoded)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return dict(value)


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _optional_string(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name=name)


def _integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _number(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _write_json(value: Mapping[str, object]) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(value) + b"\n")


__all__ = [
    "GovernedResearchImplementations",
    "ModelTrainingScientificImplementationFactory",
    "ReadOnlyRegistryPaths",
    "ReadOnlyResearchRegistryLookup",
    "RegistryBindingCheck",
    "ResearchAssemblyError",
    "ResearchAssemblyPlan",
    "ResearchOnlyG1Implementations",
    "ResearchOnlyG1SemanticAuthorities",
    "ResearchRegistryAssembly",
    "experiment_component_type_for_role",
    "load_research_execution_bindings",
    "load_research_run_spec",
    "load_research_scientific_lineage_manifest",
    "plan_research_run_read_only",
    "research_cli_main",
]
