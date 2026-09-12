from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, TypeAlias

from alpha_research.agents import AgentContext, AgentResult, AgentRole, AgentTask
from alpha_research.core.hashing import (
    canonical_json_bytes,
    hash_json,
    require_sha256,
)
from alpha_research.core.immutable_json import ImmutableJsonError
from alpha_research.experiments import (
    DataPartition,
    ExperimentController,
    ExperimentProfile,
    ExperimentRegistry,
    ExperimentRunSummary,
    ExperimentSpec,
    ResourceBudget,
    RetryPolicy,
    StageContext,
    StageFailure,
    StageOutput,
)
from alpha_research.experiments.controller import StageHandler
from alpha_research.observability import TelemetryStore
from alpha_research.orchestration import ExperimentRuntime, RuntimeUsage
from alpha_research.research.artifacts import StageArtifactDescriptor, StageContract
from alpha_research.research.model_training_inputs import ModelTrainingInputError
from alpha_research.research.spec import ResearchRunSpec, ResearchStage
from factor_production.v5.llm.safe_context import assert_safe_context


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class MemoryVisibility(str, Enum):
    DEVELOPMENT = "development"
    ADAPTIVE_VALIDATION = "adaptive_validation"
    AUDIT_ONLY = "audit_only"


STAGE_AGENT_ROLES: Mapping[ResearchStage, AgentRole] = MappingProxyType(
    {
        ResearchStage.DATA_QUALITY: AgentRole.EXECUTOR,
        ResearchStage.FACTOR_GENERATION: AgentRole.RESEARCHER,
        ResearchStage.LABEL_BUILDING: AgentRole.DEVELOPER,
        ResearchStage.VALIDATION_SPLIT: AgentRole.PLANNER,
        ResearchStage.MODEL_TRAINING: AgentRole.EXECUTOR,
        ResearchStage.FACTOR_EVALUATION: AgentRole.REVIEWER,
        ResearchStage.SCORE_CONSTRUCTION: AgentRole.EXECUTOR,
        ResearchStage.PORTFOLIO_CONSTRUCTION: AgentRole.EXECUTOR,
        ResearchStage.BACKTEST: AgentRole.EXECUTOR,
        ResearchStage.ROBUSTNESS: AgentRole.REVIEWER,
        ResearchStage.REPORT: AgentRole.CONTROLLER,
    }
)

_ADAPTIVE_STAGES = frozenset(
    {
        ResearchStage.FACTOR_EVALUATION,
        ResearchStage.SCORE_CONSTRUCTION,
        ResearchStage.PORTFOLIO_CONSTRUCTION,
        ResearchStage.BACKTEST,
        ResearchStage.ROBUSTNESS,
        ResearchStage.REPORT,
    }
)
_PROTECTED_PARTITIONS = frozenset({DataPartition.TEST, DataPartition.HOLDOUT})


@dataclass(frozen=True, slots=True)
class ResearchExecutionBindings:
    """Execution identity omitted from :class:`ResearchRunSpec`.

    A bridge is deterministic only when code, environment, budget, retry, seed,
    and optional LLM transport identity are explicit inputs.
    """

    code_snapshot_hash: str
    environment_hash: str
    random_seed: int
    resource_budget: ResourceBudget
    retry_policy: RetryPolicy
    llm_policy_hash: str | None = None
    llm_transport_hash: str | None = None
    schema_version: str = "research-execution-bindings/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "research-execution-bindings/v1":
            raise ValueError("unsupported ResearchExecutionBindings schema")
        require_sha256(self.code_snapshot_hash, name="execution code_snapshot_hash")
        require_sha256(self.environment_hash, name="execution environment_hash")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise TypeError("execution random_seed must be an integer")
        if not isinstance(self.resource_budget, ResourceBudget):
            raise TypeError("execution resource_budget must be ResourceBudget")
        if not isinstance(self.retry_policy, RetryPolicy):
            raise TypeError("execution retry_policy must be RetryPolicy")
        for name in ("llm_policy_hash", "llm_transport_hash"):
            value = getattr(self, name)
            if value is not None:
                require_sha256(value, name=f"execution {name}")

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "code_snapshot_hash": self.code_snapshot_hash,
            "environment_hash": self.environment_hash,
            "random_seed": self.random_seed,
            "resource_budget": self.resource_budget.to_dict(),
            "retry_policy": self.retry_policy.to_dict(),
            "llm_policy_hash": self.llm_policy_hash,
            "llm_transport_hash": self.llm_transport_hash,
        }


@dataclass(frozen=True, slots=True)
class AgentContextPolicy:
    data_schema_hashes: tuple[str, ...]
    allowed_fields: tuple[str, ...]
    allowed_operators: tuple[str, ...]
    allowed_windows: tuple[int, ...]
    hypothesis_hashes: tuple[str, ...] = ()
    sanitized_feedback_codes: tuple[str, ...] = ()
    schema_version: str = "agent-context-policy/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "agent-context-policy/v1":
            raise ValueError("unsupported AgentContextPolicy schema")
        for name in ("data_schema_hashes", "hypothesis_hashes"):
            values = tuple(getattr(self, name))
            if len(set(values)) != len(values):
                raise ValueError(f"agent policy {name} must be unique")
            for digest in values:
                require_sha256(digest, name=f"agent policy {name}")
            object.__setattr__(self, name, values)
        for name in (
            "allowed_fields",
            "allowed_operators",
            "allowed_windows",
            "sanitized_feedback_codes",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def bind(self, spec: ExperimentSpec) -> AgentContext:
        return AgentContext.bind(
            spec,
            data_schema_hashes=self.data_schema_hashes,
            allowed_fields=self.allowed_fields,
            allowed_operators=self.allowed_operators,
            allowed_windows=self.allowed_windows,
            hypothesis_hashes=self.hypothesis_hashes,
            sanitized_feedback_codes=self.sanitized_feedback_codes,
        )


@dataclass(frozen=True, slots=True)
class AgentStageDecision:
    """Safe, content-addressed directive produced by one role adapter."""

    status: str
    decision_code: str
    directive: Mapping[str, JsonValue]
    request_hash: str | None = None
    response_hash: str | None = None
    model_id_hash: str | None = None
    usage_hash: str | None = None
    schema_version: str = "agent-stage-decision/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "agent-stage-decision/v1":
            raise ValueError("unsupported AgentStageDecision schema")
        if self.status not in {"succeeded", "rejected", "failed"}:
            raise ValueError("agent stage decision status is invalid")
        _require_safe_code(self.decision_code, name="agent decision_code")
        directive = _json_object(self.directive, name="agent directive", nonempty=True)
        assert_safe_context(directive)
        object.__setattr__(self, "directive", MappingProxyType(directive))
        lineage = (
            self.request_hash,
            self.response_hash,
            self.model_id_hash,
            self.usage_hash,
        )
        if any(item is not None for item in lineage) and not all(
            item is not None for item in lineage
        ):
            raise ValueError("agent decision LLM lineage must be complete or absent")
        for digest in lineage:
            if digest is not None:
                require_sha256(digest, name="agent decision LLM lineage")

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "decision_code": self.decision_code,
            "directive": dict(self.directive),
            "request_hash": self.request_hash,
            "response_hash": self.response_hash,
            "model_id_hash": self.model_id_hash,
            "usage_hash": self.usage_hash,
        }


@dataclass(frozen=True, slots=True)
class AgentStageInvocation:
    """The complete and deliberately redacted input visible to an Agent role."""

    stage: ResearchStage | str
    role: AgentRole | str
    context: AgentContext
    task: AgentTask
    stage_contract_hash: str
    parent_artifact_hashes: Mapping[ResearchStage | str, str]
    schema_version: str = "agent-stage-invocation/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "agent-stage-invocation/v1":
            raise ValueError("unsupported AgentStageInvocation schema")
        stage = ResearchStage(self.stage)
        role = AgentRole(self.role)
        if self.task.role is not role:
            raise ValueError("agent invocation task role differs")
        if self.task.context_hash != self.context.content_hash:
            raise ValueError("agent invocation task context differs")
        require_sha256(
            self.stage_contract_hash, name="agent invocation stage_contract_hash"
        )
        parents = {
            ResearchStage(parent): require_sha256(
                digest, name=f"agent invocation parent:{ResearchStage(parent).value}"
            )
            for parent, digest in self.parent_artifact_hashes.items()
        }
        if tuple(parents.values()) != self.task.input_artifact_hashes:
            raise ValueError("agent invocation task inputs differ from stage parents")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "role", role)
        object.__setattr__(
            self,
            "parent_artifact_hashes",
            MappingProxyType(dict(parents)),
        )


class AgentRoleAdapter(Protocol):
    def __call__(self, invocation: AgentStageInvocation) -> AgentStageDecision: ...


@dataclass(frozen=True, slots=True)
class ScientificStageRequest:
    """Trusted scientific-side request.

    Unlike the Agent invocation, this request may bind adaptive-validation
    components.  A normal stage contract can never bind test/holdout data.
    """

    research_run_spec_hash: str
    experiment_spec_hash: str
    stage: ResearchStage | str
    attempt_id: int
    attempt_number: int
    contract: StageContract
    parent_artifact_hashes: Mapping[ResearchStage | str, str]
    agent_decision: AgentStageDecision
    schema_version: str = "scientific-stage-request/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "scientific-stage-request/v1":
            raise ValueError("unsupported ScientificStageRequest schema")
        require_sha256(
            self.research_run_spec_hash,
            name="scientific request research_run_spec_hash",
        )
        require_sha256(
            self.experiment_spec_hash,
            name="scientific request experiment_spec_hash",
        )
        stage = ResearchStage(self.stage)
        if stage is ResearchStage.ADMISSION:
            raise ValueError(
                "protected admission cannot use an Agent scientific request"
            )
        if ResearchStage(self.contract.stage) is not stage:
            raise ValueError("scientific request contract stage differs")
        if self.contract.research_run_spec_hash != self.research_run_spec_hash:
            raise ValueError("scientific request contract run differs")
        if not isinstance(self.attempt_id, int) or self.attempt_id <= 0:
            raise ValueError("scientific request attempt_id must be positive")
        if not isinstance(self.attempt_number, int) or self.attempt_number <= 0:
            raise ValueError("scientific request attempt_number must be positive")
        parents = {
            ResearchStage(parent): require_sha256(
                digest, name=f"scientific request parent:{ResearchStage(parent).value}"
            )
            for parent, digest in self.parent_artifact_hashes.items()
        }
        expected = {
            ResearchStage(item) for item in self.contract.required_parent_stages
        }
        if set(parents) != expected:
            raise ValueError("scientific request parent set differs")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(
            self,
            "parent_artifact_hashes",
            MappingProxyType(dict(parents)),
        )


@dataclass(frozen=True, slots=True)
class ScientificStageResult:
    """Typed result returned by an explicitly injected scientific implementation."""

    stage: ResearchStage | str
    implementation_id: str
    implementation_hash: str
    result_payload: Mapping[str, JsonValue]
    evidence_hashes: tuple[str, ...]
    sanitized_feedback_codes: tuple[str, ...]
    usage: RuntimeUsage
    input_rows: int
    output_rows: int
    symbols: int
    disk_read_bytes: int
    cache_hits: int
    cache_misses: int
    worker_count: int
    schema_version: str = "scientific-stage-result/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "scientific-stage-result/v1":
            raise ValueError("unsupported ScientificStageResult schema")
        stage = ResearchStage(self.stage)
        if stage is ResearchStage.ADMISSION:
            raise ValueError("protected admission requires ProtectedTerminalResult")
        _require_safe_code(self.implementation_id, name="scientific implementation_id")
        if self.implementation_id.lower() in {
            "noop",
            "placeholder",
            "stub",
            "mock",
        }:
            raise ValueError("placeholder scientific implementations are forbidden")
        require_sha256(self.implementation_hash, name="scientific implementation_hash")
        payload = _json_object(
            self.result_payload, name="scientific result_payload", nonempty=True
        )
        evidence = _hash_tuple(
            self.evidence_hashes, name="scientific evidence_hashes", nonempty=True
        )
        feedback = _safe_codes(
            self.sanitized_feedback_codes, name="scientific feedback"
        )
        assert_safe_context({"sanitized_feedback_codes": list(feedback)})
        _validate_runtime_usage(self.usage)
        _validate_stage_statistics(self)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "result_payload", MappingProxyType(payload))
        object.__setattr__(self, "evidence_hashes", evidence)
        object.__setattr__(self, "sanitized_feedback_codes", feedback)

    @property
    def result_payload_hash(self) -> str:
        return hash_json(dict(self.result_payload))

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "stage": ResearchStage(self.stage).value,
            "implementation_id": self.implementation_id,
            "implementation_hash": self.implementation_hash,
            "result_payload": dict(self.result_payload),
            "result_payload_hash": self.result_payload_hash,
            "evidence_hashes": list(self.evidence_hashes),
            "sanitized_feedback_codes": list(self.sanitized_feedback_codes),
            "usage": _usage_dict(self.usage),
            "statistics": _statistics_dict(self),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ScientificStageResult":
        """Strictly reconstruct one persisted scientific result."""

        payload = dict(value)
        expected = {
            "schema_version",
            "stage",
            "implementation_id",
            "implementation_hash",
            "result_payload",
            "result_payload_hash",
            "evidence_hashes",
            "sanitized_feedback_codes",
            "usage",
            "statistics",
        }
        if set(payload) != expected:
            raise ValueError("scientific stage result fields differ")
        result_payload = _json_object(
            _mapping_object(
                payload["result_payload"], name="scientific result payload"
            ),
            name="scientific result payload",
            nonempty=True,
        )
        result_payload_hash = _string_value(
            payload["result_payload_hash"],
            name="scientific result payload hash",
        )
        require_sha256(result_payload_hash, name="scientific result payload hash")
        if hash_json(result_payload) != result_payload_hash:
            raise ValueError("scientific result payload hash differs")
        evidence = payload["evidence_hashes"]
        feedback = payload["sanitized_feedback_codes"]
        if not isinstance(evidence, list) or any(
            not isinstance(item, str) for item in evidence
        ):
            raise TypeError("scientific evidence_hashes must be a string list")
        if not isinstance(feedback, list) or any(
            not isinstance(item, str) for item in feedback
        ):
            raise TypeError("scientific feedback codes must be a string list")
        statistics = _mapping_object(
            payload["statistics"], name="scientific result statistics"
        )
        expected_statistics = {
            "input_rows",
            "output_rows",
            "symbols",
            "disk_read_bytes",
            "cache_hits",
            "cache_misses",
            "worker_count",
        }
        if set(statistics) != expected_statistics:
            raise ValueError("scientific result statistics fields differ")
        return cls(
            schema_version=_string_value(
                payload["schema_version"], name="scientific result schema_version"
            ),
            stage=_string_value(payload["stage"], name="scientific result stage"),
            implementation_id=_string_value(
                payload["implementation_id"],
                name="scientific result implementation_id",
            ),
            implementation_hash=_string_value(
                payload["implementation_hash"],
                name="scientific result implementation_hash",
            ),
            result_payload=result_payload,
            evidence_hashes=tuple(evidence),
            sanitized_feedback_codes=tuple(feedback),
            usage=_runtime_usage_from_mapping(payload["usage"]),
            input_rows=_integer_value(
                statistics["input_rows"], name="scientific result input_rows"
            ),
            output_rows=_integer_value(
                statistics["output_rows"], name="scientific result output_rows"
            ),
            symbols=_integer_value(
                statistics["symbols"], name="scientific result symbols"
            ),
            disk_read_bytes=_integer_value(
                statistics["disk_read_bytes"],
                name="scientific result disk_read_bytes",
            ),
            cache_hits=_integer_value(
                statistics["cache_hits"], name="scientific result cache_hits"
            ),
            cache_misses=_integer_value(
                statistics["cache_misses"], name="scientific result cache_misses"
            ),
            worker_count=_integer_value(
                statistics["worker_count"], name="scientific result worker_count"
            ),
        )


class ScientificStageImplementation(Protocol):
    def __call__(self, request: ScientificStageRequest) -> ScientificStageResult: ...


@dataclass(frozen=True, slots=True)
class ProtectedEvaluationRequest:
    research_run_spec_hash: str
    experiment_spec_hash: str
    report_artifact_hash: str
    protected_partition_hashes: Mapping[DataPartition | str, str]
    component_bindings: Mapping[str, str]
    schema_version: str = "protected-evaluation-request/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "protected-evaluation-request/v1":
            raise ValueError("unsupported ProtectedEvaluationRequest schema")
        require_sha256(
            self.research_run_spec_hash,
            name="protected evaluation research_run_spec_hash",
        )
        require_sha256(
            self.experiment_spec_hash,
            name="protected evaluation experiment_spec_hash",
        )
        require_sha256(
            self.report_artifact_hash,
            name="protected evaluation report_artifact_hash",
        )
        partitions: dict[DataPartition, str] = {
            DataPartition(partition): require_sha256(
                digest,
                name=f"protected evaluation partition:{DataPartition(partition).value}",
            )
            for partition, digest in self.protected_partition_hashes.items()
        }
        if not partitions or not set(partitions).issubset(_PROTECTED_PARTITIONS):
            raise ValueError("terminal evaluator requires only test/holdout partitions")
        bindings = dict(sorted(self.component_bindings.items()))
        for role, digest in bindings.items():
            if not role:
                raise ValueError("protected evaluation component role is empty")
            require_sha256(digest, name=f"protected evaluation component:{role}")
        expected_partition_bindings = {
            f"data:partition:{partition.value}": digest
            for partition, digest in partitions.items()
        }
        if any(
            bindings.get(role) != digest
            for role, digest in expected_partition_bindings.items()
        ):
            raise ValueError("terminal evaluator partition/component bindings differ")
        object.__setattr__(
            self,
            "protected_partition_hashes",
            MappingProxyType(
                dict(sorted(partitions.items(), key=lambda item: str(item[0])))
            ),
        )
        object.__setattr__(self, "component_bindings", MappingProxyType(bindings))

    @property
    def protected_partition_set_hash(self) -> str:
        return hash_json(
            {
                "schema_version": "protected-partition-set/v1",
                "partitions": {
                    DataPartition(partition).value: digest
                    for partition, digest in self.protected_partition_hashes.items()
                },
            }
        )

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "research_run_spec_hash": self.research_run_spec_hash,
            "experiment_spec_hash": self.experiment_spec_hash,
            "report_artifact_hash": self.report_artifact_hash,
            "protected_partition_hashes": {
                DataPartition(partition).value: digest
                for partition, digest in self.protected_partition_hashes.items()
            },
            "component_bindings": dict(self.component_bindings),
        }


@dataclass(frozen=True, slots=True)
class ProtectedTerminalResult:
    """Redacted terminal result; exact protected metrics remain in audit storage."""

    verdict: str
    reason_codes: tuple[str, ...]
    audit_evidence_hash: str
    evaluated_partition_hashes: Mapping[DataPartition | str, str]
    usage: RuntimeUsage
    input_rows: int
    output_rows: int
    symbols: int
    disk_read_bytes: int
    cache_hits: int
    cache_misses: int
    worker_count: int
    schema_version: str = "protected-terminal-result/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "protected-terminal-result/v1":
            raise ValueError("unsupported ProtectedTerminalResult schema")
        if self.verdict not in {"pass", "fail", "inconclusive"}:
            raise ValueError("protected terminal verdict is invalid")
        reasons = _safe_codes(
            self.reason_codes, name="protected terminal reason_codes", nonempty=True
        )
        require_sha256(
            self.audit_evidence_hash, name="protected terminal audit_evidence_hash"
        )
        partitions: dict[DataPartition, str] = {
            DataPartition(partition): require_sha256(
                digest,
                name=f"protected terminal partition:{DataPartition(partition).value}",
            )
            for partition, digest in self.evaluated_partition_hashes.items()
        }
        if not partitions or not set(partitions).issubset(_PROTECTED_PARTITIONS):
            raise ValueError(
                "protected terminal result contains a non-protected partition"
            )
        _validate_runtime_usage(self.usage)
        _validate_stage_statistics(self)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(
            self,
            "evaluated_partition_hashes",
            MappingProxyType(
                dict(sorted(partitions.items(), key=lambda item: str(item[0])))
            ),
        )

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "verdict": self.verdict,
            "reason_codes": list(self.reason_codes),
            "audit_evidence_hash": self.audit_evidence_hash,
            "evaluated_partition_hashes": {
                DataPartition(partition).value: digest
                for partition, digest in self.evaluated_partition_hashes.items()
            },
            "usage": _usage_dict(self.usage),
            "statistics": _statistics_dict(self),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ProtectedTerminalResult":
        payload = dict(value)
        expected = {
            "schema_version",
            "verdict",
            "reason_codes",
            "audit_evidence_hash",
            "evaluated_partition_hashes",
            "usage",
            "statistics",
        }
        if set(payload) != expected:
            raise ValueError("protected terminal result fields differ")
        usage = _runtime_usage_from_mapping(payload["usage"])
        statistics = _mapping_object(
            payload["statistics"], name="protected terminal statistics"
        )
        if set(statistics) != {
            "input_rows",
            "output_rows",
            "symbols",
            "disk_read_bytes",
            "cache_hits",
            "cache_misses",
            "worker_count",
        }:
            raise ValueError("protected terminal statistics fields differ")
        partition_values = _mapping_object(
            payload["evaluated_partition_hashes"],
            name="protected terminal partitions",
        )
        reasons = payload["reason_codes"]
        if not isinstance(reasons, list) or any(
            not isinstance(item, str) for item in reasons
        ):
            raise TypeError("protected terminal reason_codes must be a string list")
        return cls(
            schema_version=_string_value(
                payload["schema_version"], name="protected terminal schema_version"
            ),
            verdict=_string_value(
                payload["verdict"], name="protected terminal verdict"
            ),
            reason_codes=tuple(reasons),
            audit_evidence_hash=_string_value(
                payload["audit_evidence_hash"],
                name="protected terminal audit_evidence_hash",
            ),
            evaluated_partition_hashes={
                _string_value(
                    name, name="protected terminal partition name"
                ): _string_value(digest, name="protected terminal partition hash")
                for name, digest in partition_values.items()
            },
            usage=usage,
            input_rows=_integer_value(
                statistics["input_rows"], name="protected terminal input_rows"
            ),
            output_rows=_integer_value(
                statistics["output_rows"], name="protected terminal output_rows"
            ),
            symbols=_integer_value(
                statistics["symbols"], name="protected terminal symbols"
            ),
            disk_read_bytes=_integer_value(
                statistics["disk_read_bytes"],
                name="protected terminal disk_read_bytes",
            ),
            cache_hits=_integer_value(
                statistics["cache_hits"], name="protected terminal cache_hits"
            ),
            cache_misses=_integer_value(
                statistics["cache_misses"], name="protected terminal cache_misses"
            ),
            worker_count=_integer_value(
                statistics["worker_count"], name="protected terminal worker_count"
            ),
        )


class ProtectedTerminalEvaluator(Protocol):
    def __call__(
        self, request: ProtectedEvaluationRequest
    ) -> ProtectedTerminalResult: ...


class ProtectedTerminalEvaluationGateway:
    """Durable one-time gateway in front of the protected evaluator.

    Reservation commits before the backend can observe protected data.  A
    backend exception or invalid result deliberately leaves an indeterminate
    reservation that cannot be automatically retried.
    """

    def __init__(
        self,
        *,
        registry: ExperimentRegistry,
        experiment_spec_hash: str,
        evaluation_spec_hash: str,
        approval_hash: str,
        evaluator_hash: str,
        evaluator: ProtectedTerminalEvaluator,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(registry, ExperimentRegistry):
            raise TypeError("registry must be ExperimentRegistry")
        self.registry = registry
        self.experiment_spec_hash = require_sha256(
            experiment_spec_hash, name="gateway experiment_spec_hash"
        )
        self.evaluation_spec_hash = require_sha256(
            evaluation_spec_hash, name="gateway evaluation_spec_hash"
        )
        self.approval_hash = require_sha256(approval_hash, name="gateway approval_hash")
        self.evaluator_hash = require_sha256(
            evaluator_hash, name="gateway evaluator_hash"
        )
        if isinstance(evaluator, str) or not callable(evaluator):
            raise TypeError("gateway evaluator must be a callable object")
        if clock is not None and not callable(clock):
            raise TypeError("gateway clock must be callable or None")
        self.evaluator = evaluator
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def __call__(self, request: ProtectedEvaluationRequest) -> ProtectedTerminalResult:
        if not isinstance(request, ProtectedEvaluationRequest):
            raise TypeError("gateway request must be ProtectedEvaluationRequest")
        if request.experiment_spec_hash != self.experiment_spec_hash:
            raise ValueError("gateway request belongs to another experiment")
        if request.component_bindings.get("evaluation") != self.evaluation_spec_hash:
            raise ValueError("gateway request evaluation policy differs")
        partitions = {
            DataPartition(partition).value: digest
            for partition, digest in request.protected_partition_hashes.items()
        }
        reservation = self.registry.reserve_protected_evaluation(
            experiment_spec_hash=self.experiment_spec_hash,
            approval_hash=self.approval_hash,
            evaluation_spec_hash=self.evaluation_spec_hash,
            protected_partition_hashes=partitions,
            request_hash=request.content_hash,
            request_payload=request.to_dict(),
            evaluator_hash=self.evaluator_hash,
            at=self._now(),
        )
        if reservation.status == "completed":
            if reservation.result_payload is None or reservation.result_hash is None:
                raise RuntimeError(
                    "completed protected evaluation has no sealed result"
                )
            replay = ProtectedTerminalResult.from_mapping(reservation.result_payload)
            if replay.content_hash != reservation.result_hash:
                raise RuntimeError(
                    "sealed protected evaluation result integrity differs"
                )
            return replay

        result = self.evaluator(request)
        if not isinstance(result, ProtectedTerminalResult):
            raise TypeError("protected evaluator must return ProtectedTerminalResult")
        if dict(result.evaluated_partition_hashes) != dict(
            request.protected_partition_hashes
        ):
            raise ValueError(
                "protected evaluator returned another protected partition set"
            )
        completed = self.registry.complete_protected_evaluation(
            consumption_id=reservation.consumption_id,
            experiment_spec_hash=self.experiment_spec_hash,
            approval_hash=self.approval_hash,
            evaluation_spec_hash=self.evaluation_spec_hash,
            result_hash=result.content_hash,
            result_payload=result.to_dict(),
            at=self._now(),
        )
        if (
            completed.result_payload is None
            or completed.result_hash != result.content_hash
        ):
            raise RuntimeError("protected evaluation result was not sealed")
        return ProtectedTerminalResult.from_mapping(completed.result_payload)

    def _now(self) -> datetime:
        moment = self.clock()
        if not isinstance(moment, datetime) or moment.tzinfo is None:
            raise ValueError("gateway clock must return timezone-aware datetime")
        return moment


@dataclass(frozen=True, slots=True)
class StageMemoryRecord:
    research_run_spec_hash: str
    experiment_spec_hash: str
    stage: ResearchStage | str
    role: str
    visibility: MemoryVisibility | str
    decision_code: str
    task_hash: str | None
    agent_result_hash: str | None
    result_hash: str
    evidence_hashes: tuple[str, ...]
    sanitized_feedback_codes: tuple[str, ...]
    schema_version: str = "stage-memory-record/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "stage-memory-record/v1":
            raise ValueError("unsupported StageMemoryRecord schema")
        for name in ("research_run_spec_hash", "experiment_spec_hash", "result_hash"):
            require_sha256(str(getattr(self, name)), name=f"memory {name}")
        for name in ("task_hash", "agent_result_hash"):
            value = getattr(self, name)
            if value is not None:
                require_sha256(value, name=f"memory {name}")
        stage = ResearchStage(self.stage)
        visibility = MemoryVisibility(self.visibility)
        _require_safe_code(self.role, name="memory role")
        _require_safe_code(self.decision_code, name="memory decision_code")
        evidence = _hash_tuple(
            self.evidence_hashes, name="memory evidence_hashes", nonempty=True
        )
        feedback = _safe_codes(self.sanitized_feedback_codes, name="memory feedback")
        if visibility is MemoryVisibility.AUDIT_ONLY:
            if stage is not ResearchStage.ADMISSION:
                raise ValueError("audit-only memory is reserved for terminal admission")
            if feedback:
                raise ValueError("protected terminal memory cannot export feedback")
            if self.task_hash is not None or self.agent_result_hash is not None:
                raise ValueError(
                    "terminal evaluator cannot masquerade as an Agent task"
                )
        elif stage is ResearchStage.ADMISSION:
            raise ValueError("terminal admission memory must be audit-only")
        assert_safe_context({"sanitized_feedback_codes": list(feedback)})
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "evidence_hashes", evidence)
        object.__setattr__(self, "sanitized_feedback_codes", feedback)

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "research_run_spec_hash": self.research_run_spec_hash,
            "experiment_spec_hash": self.experiment_spec_hash,
            "stage": ResearchStage(self.stage).value,
            "role": self.role,
            "visibility": MemoryVisibility(self.visibility).value,
            "decision_code": self.decision_code,
            "task_hash": self.task_hash,
            "agent_result_hash": self.agent_result_hash,
            "result_hash": self.result_hash,
            "evidence_hashes": list(self.evidence_hashes),
            "sanitized_feedback_codes": list(self.sanitized_feedback_codes),
        }


class _GovernedAgentStageHandler:
    ROLE: AgentRole

    def __init__(
        self,
        *,
        run_spec: ResearchRunSpec,
        experiment_spec: ExperimentSpec,
        agent_context: AgentContext,
        stage: ResearchStage,
        role_adapter: AgentRoleAdapter,
        scientific_implementation: ScientificStageImplementation,
    ) -> None:
        if stage is ResearchStage.ADMISSION:
            raise ValueError("admission requires ProtectedTerminalStageHandler")
        if STAGE_AGENT_ROLES[stage] is not self.ROLE:
            raise ValueError(
                f"{type(self).__name__} cannot execute stage:{stage.value}"
            )
        if agent_context.experiment_spec_hash != experiment_spec.content_hash:
            raise ValueError("stage Agent context belongs to another experiment")
        self.run_spec = run_spec
        self.experiment_spec = experiment_spec
        self.agent_context = agent_context
        self.stage = stage
        self.contract = StageContract.for_run(run_spec, stage)
        self.role_adapter = role_adapter
        self.scientific_implementation = scientific_implementation

    def __call__(self, context: StageContext) -> StageOutput:
        self._validate_context(context)
        parents = _required_parents(self.contract, context.prior_result_hashes)
        task = AgentTask.create(
            task_id=f"{self.stage.value}-{self.ROLE.value}-{context.attempt_number}",
            role=self.ROLE,
            spec=self.experiment_spec,
            context=self.agent_context,
            input_artifact_hashes=tuple(parents.values()),
            output_schema_hash=self.contract.content_hash,
            maximum_output_artifacts=1,
        )
        invocation = AgentStageInvocation(
            stage=self.stage,
            role=self.ROLE,
            context=self.agent_context,
            task=task,
            stage_contract_hash=self.contract.content_hash,
            parent_artifact_hashes=parents,
        )
        decision = self.role_adapter(invocation)
        if not isinstance(decision, AgentStageDecision):
            raise TypeError("role adapter must return AgentStageDecision")
        if decision.status != "succeeded":
            raise StageFailure(f"{self.ROLE.value}_{decision.status}")
        agent_result = AgentResult(
            task_hash=task.content_hash,
            status=decision.status,
            output_artifact_hashes=(decision.content_hash,),
            decision_code=decision.decision_code,
            request_hash=decision.request_hash,
            response_hash=decision.response_hash,
            model_id_hash=decision.model_id_hash,
            usage_hash=decision.usage_hash,
        )
        request = ScientificStageRequest(
            research_run_spec_hash=self.run_spec.content_hash,
            experiment_spec_hash=self.experiment_spec.content_hash,
            stage=self.stage,
            attempt_id=context.attempt_id,
            attempt_number=context.attempt_number,
            contract=self.contract,
            parent_artifact_hashes=parents,
            agent_decision=decision,
        )
        try:
            scientific = self.scientific_implementation(request)
        except ModelTrainingInputError as exc:
            if self.stage is not ResearchStage.MODEL_TRAINING:
                raise
            raise StageFailure(f"model_input_{exc.code}") from exc
        except ImmutableJsonError as exc:
            if self.stage is not ResearchStage.MODEL_TRAINING:
                raise
            raise StageFailure(f"model_phase_one_{exc.code}") from exc
        except ValueError as exc:
            if self.stage is not ResearchStage.MODEL_TRAINING:
                raise
            raise StageFailure("model_training_validation_failed") from exc
        if not isinstance(scientific, ScientificStageResult):
            raise TypeError(
                "scientific implementation must return ScientificStageResult"
            )
        if ResearchStage(scientific.stage) is not self.stage:
            raise ValueError("scientific implementation returned another stage")
        visibility = (
            MemoryVisibility.ADAPTIVE_VALIDATION
            if self.stage in _ADAPTIVE_STAGES
            else MemoryVisibility.DEVELOPMENT
        )
        memory = StageMemoryRecord(
            research_run_spec_hash=self.run_spec.content_hash,
            experiment_spec_hash=self.experiment_spec.content_hash,
            stage=self.stage,
            role=self.ROLE.value,
            visibility=visibility,
            decision_code=decision.decision_code,
            task_hash=task.content_hash,
            agent_result_hash=agent_result.content_hash,
            result_hash=scientific.content_hash,
            evidence_hashes=scientific.evidence_hashes,
            sanitized_feedback_codes=scientific.sanitized_feedback_codes,
        )
        body = {
            "schema_version": "governed-agent-stage/v1",
            "bridge_hash": _bridge_hash(self.run_spec, self.experiment_spec),
            "stage_contract_hash": self.contract.content_hash,
            "role": self.ROLE.value,
            "agent_context_hash": self.agent_context.content_hash,
            "agent_task": _agent_task_dict(task),
            "agent_decision": decision.to_dict(),
            "agent_result": _agent_result_dict(agent_result),
            "scientific_result": scientific.to_dict(),
            "memory_record": memory.to_dict(),
            "memory_record_hash": memory.content_hash,
        }
        payload = self.contract.build_payload(body, parent_artifact_hashes=parents)
        StageArtifactDescriptor.from_payload(self.contract, payload)
        return StageOutput(
            logical_name=_logical_name(
                self.run_spec, self.stage, context.attempt_number
            ),
            kind="governed_research_stage",
            payload=payload,
            media_type="application/json",
            parent_hashes=tuple(parents.values()),
            usage=scientific.usage,
            input_rows=scientific.input_rows,
            output_rows=scientific.output_rows,
            symbols=scientific.symbols,
            disk_read_bytes=scientific.disk_read_bytes,
            cache_hits=scientific.cache_hits,
            cache_misses=scientific.cache_misses,
            worker_count=scientific.worker_count,
            attributes={
                "agent_role": self.ROLE.value,
                "implementation_hash": scientific.implementation_hash,
                "memory_visibility": visibility.value,
                "human_approval_required": True,
            },
        )

    def _validate_context(self, context: StageContext) -> None:
        if context.experiment_spec_hash != self.experiment_spec.content_hash:
            raise ValueError("stage context belongs to another experiment")
        if context.stage != self.stage.value:
            raise ValueError("stage context stage differs")


class ResearcherStageHandler(_GovernedAgentStageHandler):
    ROLE = AgentRole.RESEARCHER


class PlannerStageHandler(_GovernedAgentStageHandler):
    ROLE = AgentRole.PLANNER


class DeveloperStageHandler(_GovernedAgentStageHandler):
    ROLE = AgentRole.DEVELOPER


class ExecutorStageHandler(_GovernedAgentStageHandler):
    ROLE = AgentRole.EXECUTOR


class ReviewerStageHandler(_GovernedAgentStageHandler):
    ROLE = AgentRole.REVIEWER


class ControllerStageHandler(_GovernedAgentStageHandler):
    ROLE = AgentRole.CONTROLLER


_HANDLER_BY_ROLE = {
    AgentRole.RESEARCHER: ResearcherStageHandler,
    AgentRole.PLANNER: PlannerStageHandler,
    AgentRole.DEVELOPER: DeveloperStageHandler,
    AgentRole.EXECUTOR: ExecutorStageHandler,
    AgentRole.REVIEWER: ReviewerStageHandler,
    AgentRole.CONTROLLER: ControllerStageHandler,
}


class ProtectedTerminalStageHandler:
    """Admission handler with no Agent or adaptive-memory return path."""

    def __init__(
        self,
        *,
        run_spec: ResearchRunSpec,
        experiment_spec: ExperimentSpec,
        evaluator: ProtectedTerminalEvaluator,
    ) -> None:
        if ResearchStage.ADMISSION not in run_spec.enabled_stages:
            raise ValueError("ResearchRunSpec does not enable terminal admission")
        self.run_spec = run_spec
        self.experiment_spec = experiment_spec
        self.contract = StageContract.for_run(run_spec, ResearchStage.ADMISSION)
        self.evaluator = evaluator

    def __call__(self, context: StageContext) -> StageOutput:
        if context.experiment_spec_hash != self.experiment_spec.content_hash:
            raise ValueError("terminal context belongs to another experiment")
        if context.stage != ResearchStage.ADMISSION.value:
            raise ValueError("terminal context stage differs")
        parents = _required_parents(self.contract, context.prior_result_hashes)
        protected = {
            partition: self.run_spec.data_partition_hashes[partition]
            for partition in sorted(
                set(self.run_spec.data_partition_hashes).intersection(
                    _PROTECTED_PARTITIONS
                ),
                key=str,
            )
        }
        request = ProtectedEvaluationRequest(
            research_run_spec_hash=self.run_spec.content_hash,
            experiment_spec_hash=self.experiment_spec.content_hash,
            report_artifact_hash=parents[ResearchStage.REPORT],
            protected_partition_hashes=protected,
            component_bindings=self.contract.component_bindings,
        )
        result = self.evaluator(request)
        if not isinstance(result, ProtectedTerminalResult):
            raise TypeError("terminal evaluator must return ProtectedTerminalResult")
        if dict(result.evaluated_partition_hashes) != protected:
            raise ValueError(
                "terminal evaluator returned another protected partition set"
            )
        memory = StageMemoryRecord(
            research_run_spec_hash=self.run_spec.content_hash,
            experiment_spec_hash=self.experiment_spec.content_hash,
            stage=ResearchStage.ADMISSION,
            role="terminal_evaluator",
            visibility=MemoryVisibility.AUDIT_ONLY,
            decision_code=f"terminal_{result.verdict}",
            task_hash=None,
            agent_result_hash=None,
            result_hash=result.content_hash,
            evidence_hashes=(result.audit_evidence_hash,),
            sanitized_feedback_codes=(),
        )
        body = {
            "schema_version": "protected-terminal-stage/v1",
            "bridge_hash": _bridge_hash(self.run_spec, self.experiment_spec),
            "stage_contract_hash": self.contract.content_hash,
            "terminal_result": result.to_dict(),
            "memory_record": memory.to_dict(),
            "memory_record_hash": memory.content_hash,
            "feedback_exported": False,
            "human_approval_required": True,
        }
        payload = self.contract.build_payload(body, parent_artifact_hashes=parents)
        StageArtifactDescriptor.from_payload(self.contract, payload)
        return StageOutput(
            logical_name=_logical_name(
                self.run_spec, ResearchStage.ADMISSION, context.attempt_number
            ),
            kind="protected_terminal_evaluation",
            payload=payload,
            media_type="application/json",
            parent_hashes=tuple(parents.values()),
            usage=result.usage,
            input_rows=result.input_rows,
            output_rows=result.output_rows,
            symbols=result.symbols,
            disk_read_bytes=result.disk_read_bytes,
            cache_hits=result.cache_hits,
            cache_misses=result.cache_misses,
            worker_count=result.worker_count,
            attributes={
                "memory_visibility": MemoryVisibility.AUDIT_ONLY.value,
                "feedback_exported": False,
                "human_approval_required": True,
            },
        )


def bridge_research_run(
    run_spec: ResearchRunSpec,
    bindings: ResearchExecutionBindings,
    *,
    scientific_lineage_manifest_hash: str | None = None,
) -> ExperimentSpec:
    """Deterministically adapt a complete research contract to the one controller."""

    if not isinstance(run_spec, ResearchRunSpec):
        raise TypeError("run_spec must be ResearchRunSpec")
    if not isinstance(bindings, ResearchExecutionBindings):
        raise TypeError("bindings must be ResearchExecutionBindings")
    if scientific_lineage_manifest_hash is not None:
        require_sha256(
            scientific_lineage_manifest_hash,
            name="bridge scientific_lineage_manifest_hash",
        )
    visible = tuple(
        partition
        for partition in (DataPartition.DISCOVERY, DataPartition.TRAIN)
        if partition in run_spec.data_partition_hashes
    )
    suffix = f".rr-{run_spec.content_hash[:16]}"
    if scientific_lineage_manifest_hash is not None:
        suffix += f".sl-{scientific_lineage_manifest_hash[:12]}"
    version = f"{run_spec.version[: 128 - len(suffix)]}{suffix}"
    return ExperimentSpec(
        experiment_id=run_spec.run_id,
        version=version,
        profile=ExperimentProfile(run_spec.profile),
        data_partitions={
            DataPartition(partition): digest
            for partition, digest in run_spec.data_partition_hashes.items()
        },
        agent_visible_partitions=visible,
        factor_spec_hashes=run_spec.factor_spec_hashes,
        label_spec_hash=run_spec.label_spec_hash,
        validation_spec_hash=run_spec.validation_spec_hash,
        evaluation_spec_hash=run_spec.evaluation_spec_hash,
        model_spec_hash=run_spec.model_spec_hash,
        portfolio_spec_hash=run_spec.portfolio_spec_hash,
        cost_model_hash=run_spec.cost_model_hash,
        robustness_spec_hash=run_spec.robustness_spec_hash,
        code_snapshot_hash=bindings.code_snapshot_hash,
        environment_hash=bindings.environment_hash,
        llm_policy_hash=bindings.llm_policy_hash,
        llm_transport_hash=bindings.llm_transport_hash,
        stages=tuple(stage.value for stage in run_spec.enabled_stages),
        random_seed=bindings.random_seed,
        resource_budget=bindings.resource_budget,
        retry_policy=bindings.retry_policy,
        scientific_lineage_manifest_hash=scientific_lineage_manifest_hash,
        schema_version=(
            "experiment-spec/v3"
            if scientific_lineage_manifest_hash is not None
            else "experiment-spec/v2"
        ),
    )


def build_governed_stage_handlers(
    *,
    run_spec: ResearchRunSpec,
    experiment_spec: ExperimentSpec,
    agent_context: AgentContext,
    role_adapters: Mapping[AgentRole | str, AgentRoleAdapter],
    scientific_implementations: Mapping[
        ResearchStage | str, ScientificStageImplementation
    ],
    terminal_evaluator: ProtectedTerminalEvaluator | None,
) -> Mapping[str, StageHandler]:
    """Build exact handlers for the existing :class:`ExperimentController`."""

    if tuple(experiment_spec.stages) != tuple(
        stage.value for stage in run_spec.enabled_stages
    ):
        raise ValueError("ExperimentSpec stages differ from ResearchRunSpec")
    adapters = {AgentRole(role): adapter for role, adapter in role_adapters.items()}
    if set(adapters) != set(AgentRole):
        missing = sorted(role.value for role in set(AgentRole).difference(adapters))
        extra = sorted(role.value for role in set(adapters).difference(AgentRole))
        raise ValueError(
            f"role adapters must cover all Agent roles:missing={missing},extra={extra}"
        )
    implementations = {
        ResearchStage(stage): implementation
        for stage, implementation in scientific_implementations.items()
    }
    expected_scientific = set(run_spec.enabled_stages).difference(
        {ResearchStage.ADMISSION}
    )
    if set(implementations) != expected_scientific:
        missing = sorted(
            stage.value for stage in expected_scientific.difference(implementations)
        )
        extra = sorted(
            stage.value
            for stage in set(implementations).difference(expected_scientific)
        )
        raise ValueError(
            f"scientific implementations differ:missing={missing},extra={extra}"
        )
    if agent_context.experiment_spec_hash != experiment_spec.content_hash:
        raise ValueError("AgentContext belongs to another ExperimentSpec")
    if set(agent_context.visible_partition_hashes).difference(
        {DataPartition.DISCOVERY, DataPartition.TRAIN}
    ):
        raise ValueError("AgentContext exposes a protected partition")

    handlers: dict[str, StageHandler] = {}
    for stage in run_spec.enabled_stages:
        if stage is ResearchStage.ADMISSION:
            if terminal_evaluator is None:
                raise ValueError("terminal admission requires a protected evaluator")
            handlers[stage.value] = ProtectedTerminalStageHandler(
                run_spec=run_spec,
                experiment_spec=experiment_spec,
                evaluator=terminal_evaluator,
            )
            continue
        role = STAGE_AGENT_ROLES[stage]
        handler_type = _HANDLER_BY_ROLE[role]
        handlers[stage.value] = handler_type(
            run_spec=run_spec,
            experiment_spec=experiment_spec,
            agent_context=agent_context,
            stage=stage,
            role_adapter=adapters[role],
            scientific_implementation=implementations[stage],
        )
    if (
        ResearchStage.ADMISSION not in run_spec.enabled_stages
        and terminal_evaluator is not None
    ):
        raise ValueError("terminal evaluator was supplied to a run without admission")
    return MappingProxyType(handlers)


def _run_governed_research(
    *,
    run_spec: ResearchRunSpec,
    experiment_spec: ExperimentSpec,
    handlers: Mapping[str, StageHandler],
    registry: ExperimentRegistry,
    runtime: ExperimentRuntime,
    telemetry: TelemetryStore,
    workspace_root: str | Path,
    worker_id: str,
    lease_token: str,
    lease_seconds: int = 60,
    clock: Callable[[], datetime] | None = None,
    stop_after_stage: ResearchStage | str | None = None,
) -> ExperimentRunSummary:
    """Run the governed production-candidate loop up to human approval.

    This private function receives only objects that
    ``ResearchRegistryAssembly.execute`` validated before its atomic registry
    write.  It never registers an approval and never calls
    ``finalize_production``.
    """

    if (
        ExperimentProfile(run_spec.profile)
        is not ExperimentProfile.PRODUCTION_CANDIDATE
    ):
        raise ValueError("governed terminal execution requires production_candidate")
    if ResearchStage.ADMISSION not in run_spec.enabled_stages:
        raise ValueError("governed terminal execution requires admission")
    if experiment_spec.scientific_lineage_manifest_hash is None:
        raise ValueError("governed production research requires scientific lineage")
    if runtime.spec.content_hash != experiment_spec.content_hash:
        raise ValueError("runtime does not belong to the bridged ExperimentSpec")
    registry.get_experiment(experiment_spec.content_hash)
    controller = ExperimentController(
        spec=experiment_spec,
        registry=registry,
        runtime=runtime,
        telemetry=telemetry,
        workspace_root=workspace_root,
        handlers=handlers,
        worker_id=worker_id,
        lease_token=lease_token,
        lease_seconds=lease_seconds,
        clock=clock,
    )
    if stop_after_stage is None:
        summary = controller.run()
    else:
        stage = ResearchStage(stop_after_stage)
        if stage is not ResearchStage.MODEL_TRAINING:
            raise ValueError(
                "governed research may checkpoint only after model_training"
            )
        summary = controller.run_until(stage.value)
        if summary.status != "checkpointed":
            raise RuntimeError("governed research checkpoint status differs")
        return summary
    if summary.status == "completed":
        raise RuntimeError(
            "governed production research cannot self-complete before human approval"
        )
    return summary


def _run_research_only_g1(
    *,
    run_spec: ResearchRunSpec,
    experiment_spec: ExperimentSpec,
    handlers: Mapping[str, StageHandler],
    registry: ExperimentRegistry,
    runtime: ExperimentRuntime,
    telemetry: TelemetryStore,
    workspace_root: str | Path,
    worker_id: str,
    lease_token: str,
    lease_seconds: int = 60,
    clock: Callable[[], datetime] | None = None,
    terminal_evaluator: ProtectedTerminalEvaluator | None = None,
    stop_after_stage: ResearchStage | str | None = None,
) -> ExperimentRunSummary:
    """Run a lineage-bound research experiment through its G1 report only.

    This is deliberately separate from the production-candidate loop.  It has
    no protected-data approval, terminal evaluator, admission stage, or
    production finalization authority.  The ordinary controller therefore
    self-seals a non-production receipt after the final REPORT stage.
    """

    if ExperimentProfile(run_spec.profile) is ExperimentProfile.PRODUCTION_CANDIDATE:
        raise ValueError("research-only G1 forbids production_candidate profile")
    if (
        ExperimentProfile(experiment_spec.profile)
        is ExperimentProfile.PRODUCTION_CANDIDATE
    ):
        raise ValueError("research-only G1 forbids production_candidate ExperimentSpec")
    if ResearchStage.ADMISSION in run_spec.enabled_stages:
        raise ValueError("research-only G1 forbids admission")
    if "admission" in experiment_spec.stages:
        raise ValueError("research-only G1 ExperimentSpec forbids admission")
    protected = _PROTECTED_PARTITIONS.intersection(run_spec.data_partition_hashes)
    if protected:
        names = ",".join(sorted(partition.value for partition in protected))
        raise ValueError(f"research-only G1 forbids protected partitions:{names}")
    protected = _PROTECTED_PARTITIONS.intersection(experiment_spec.data_partitions)
    if protected:
        names = ",".join(sorted(partition.value for partition in protected))
        raise ValueError(
            f"research-only G1 ExperimentSpec forbids protected partitions:{names}"
        )
    if terminal_evaluator is not None:
        raise ValueError("research-only G1 terminal_evaluator must be None")
    if experiment_spec.schema_version != "experiment-spec/v3":
        raise ValueError("research-only G1 requires ExperimentSpec v3")
    if experiment_spec.scientific_lineage_manifest_hash is None:
        raise ValueError("research-only G1 requires scientific lineage")
    if tuple(experiment_spec.stages) != tuple(
        stage.value for stage in run_spec.enabled_stages
    ):
        raise ValueError("ExperimentSpec stages differ from ResearchRunSpec")
    if (
        not experiment_spec.stages
        or experiment_spec.stages[-1] != ResearchStage.REPORT.value
    ):
        raise ValueError("research-only G1 must terminate at report")
    if runtime.spec.content_hash != experiment_spec.content_hash:
        raise ValueError("runtime does not belong to the bridged ExperimentSpec")
    registry.get_experiment(experiment_spec.content_hash)
    controller = ExperimentController(
        spec=experiment_spec,
        registry=registry,
        runtime=runtime,
        telemetry=telemetry,
        workspace_root=workspace_root,
        handlers=handlers,
        worker_id=worker_id,
        lease_token=lease_token,
        lease_seconds=lease_seconds,
        clock=clock,
    )
    if stop_after_stage is None:
        summary = controller.run()
    else:
        stage = ResearchStage(stop_after_stage)
        if stage is not ResearchStage.MODEL_TRAINING:
            raise ValueError(
                "research-only G1 may checkpoint only after model_training"
            )
        summary = controller.run_until(stage.value)
        if summary.status != "checkpointed":
            raise RuntimeError("research-only G1 checkpoint status differs")
        return summary
    if summary.status == "awaiting_approval":
        raise RuntimeError("research-only G1 cannot await production approval")
    if summary.status == "completed":
        receipt = registry.load_receipt(experiment_spec.content_hash)
        if receipt.production_ready or receipt.approval_hash is not None:
            raise RuntimeError("research-only G1 receipt acquired production authority")
    return summary


def _required_parents(
    contract: StageContract, prior_results: Mapping[str, str]
) -> Mapping[ResearchStage | str, str]:
    parents: dict[ResearchStage | str, str] = {}
    for raw_stage in contract.required_parent_stages:
        stage = ResearchStage(raw_stage)
        try:
            digest = prior_results[stage.value]
        except KeyError:
            raise ValueError(
                f"required parent artifact is unavailable:{stage.value}"
            ) from None
        parents[stage] = require_sha256(
            digest, name=f"required parent artifact:{stage.value}"
        )
    return MappingProxyType(parents)


def _bridge_hash(run_spec: ResearchRunSpec, experiment_spec: ExperimentSpec) -> str:
    return hash_json(
        {
            "schema_version": "research-experiment-bridge/v1",
            "research_run_spec_hash": run_spec.content_hash,
            "experiment_spec_hash": experiment_spec.content_hash,
        }
    )


def _logical_name(
    run_spec: ResearchRunSpec, stage: ResearchStage, attempt_number: int
) -> str:
    return (
        f"research-{run_spec.content_hash[:16]}-{stage.value}-attempt-{attempt_number}"
    )


def _agent_task_dict(task: AgentTask) -> dict[str, object]:
    return {
        "schema_version": task.schema_version,
        "task_hash": task.content_hash,
        "task_id": task.task_id,
        "role": task.role.value,
        "experiment_spec_hash": task.experiment_spec_hash,
        "context_hash": task.context_hash,
        "input_artifact_hashes": list(task.input_artifact_hashes),
        "output_schema_hash": task.output_schema_hash,
        "allowed_capabilities": list(task.allowed_capabilities),
        "maximum_output_artifacts": task.maximum_output_artifacts,
    }


def _agent_result_dict(result: AgentResult) -> dict[str, object]:
    return {
        "result_hash": result.content_hash,
        "task_hash": result.task_hash,
        "status": result.status,
        "output_artifact_hashes": list(result.output_artifact_hashes),
        "decision_code": result.decision_code,
        "request_hash": result.request_hash,
        "response_hash": result.response_hash,
        "model_id_hash": result.model_id_hash,
        "usage_hash": result.usage_hash,
    }


def _json_object(
    value: Mapping[str, JsonValue] | Mapping[str, object],
    *,
    name: str,
    nonempty: bool,
) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    try:
        decoded = json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TypeError(f"{name} must contain canonical JSON values") from exc
    if not isinstance(decoded, dict) or not all(
        isinstance(key, str) for key in decoded
    ):
        raise TypeError(f"{name} must be a string-keyed object")
    if nonempty and not decoded:
        raise ValueError(f"{name} must not be empty")
    return decoded


def _mapping_object(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed object")
    return dict(value)


def _string_value(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _integer_value(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _number_value(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _runtime_usage_from_mapping(value: object) -> RuntimeUsage:
    payload = _mapping_object(value, name="protected terminal usage")
    expected = {
        "attempts_reserved",
        "attempts_active",
        "attempts_succeeded",
        "attempts_failed",
        "attempts_expired",
        "attempts_cancelled",
        "wall_seconds",
        "cpu_seconds",
        "peak_memory_bytes",
        "disk_write_bytes",
        "llm_calls",
        "llm_tokens",
        "llm_cost_microusd",
    }
    if set(payload) != expected:
        raise ValueError("protected terminal usage fields differ")
    return RuntimeUsage(
        attempts_reserved=_integer_value(
            payload["attempts_reserved"], name="usage attempts_reserved"
        ),
        attempts_active=_integer_value(
            payload["attempts_active"], name="usage attempts_active"
        ),
        attempts_succeeded=_integer_value(
            payload["attempts_succeeded"], name="usage attempts_succeeded"
        ),
        attempts_failed=_integer_value(
            payload["attempts_failed"], name="usage attempts_failed"
        ),
        attempts_expired=_integer_value(
            payload["attempts_expired"], name="usage attempts_expired"
        ),
        attempts_cancelled=_integer_value(
            payload["attempts_cancelled"], name="usage attempts_cancelled"
        ),
        wall_seconds=_number_value(payload["wall_seconds"], name="usage wall_seconds"),
        cpu_seconds=_number_value(payload["cpu_seconds"], name="usage cpu_seconds"),
        peak_memory_bytes=_integer_value(
            payload["peak_memory_bytes"], name="usage peak_memory_bytes"
        ),
        disk_write_bytes=_integer_value(
            payload["disk_write_bytes"], name="usage disk_write_bytes"
        ),
        llm_calls=_integer_value(payload["llm_calls"], name="usage llm_calls"),
        llm_tokens=_integer_value(payload["llm_tokens"], name="usage llm_tokens"),
        llm_cost_microusd=_integer_value(
            payload["llm_cost_microusd"], name="usage llm_cost_microusd"
        ),
    )


def _hash_tuple(
    values: tuple[str, ...], *, name: str, nonempty: bool
) -> tuple[str, ...]:
    normalized = tuple(values)
    if nonempty and not normalized:
        raise ValueError(f"{name} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must be unique")
    for digest in normalized:
        require_sha256(digest, name=name)
    return normalized


def _safe_codes(
    values: tuple[str, ...], *, name: str, nonempty: bool = False
) -> tuple[str, ...]:
    normalized = tuple(sorted(set(values)))
    if len(normalized) != len(values):
        raise ValueError(f"{name} must be sorted and unique")
    if nonempty and not normalized:
        raise ValueError(f"{name} must not be empty")
    for value in normalized:
        _require_safe_code(value, name=name)
    return normalized


def _require_safe_code(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        raise ValueError(f"{name} is invalid")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-")
    if not value[0].isalnum() or any(character not in allowed for character in value):
        raise ValueError(f"{name} is invalid")


def _validate_runtime_usage(usage: RuntimeUsage) -> None:
    if not isinstance(usage, RuntimeUsage):
        raise TypeError("scientific usage must be RuntimeUsage")
    for name in (
        "attempts_reserved",
        "attempts_active",
        "attempts_succeeded",
        "attempts_failed",
        "attempts_expired",
        "attempts_cancelled",
        "peak_memory_bytes",
        "disk_write_bytes",
        "llm_calls",
        "llm_tokens",
        "llm_cost_microusd",
    ):
        value = getattr(usage, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"scientific usage {name} must be non-negative")
    for name in ("wall_seconds", "cpu_seconds"):
        value = float(getattr(usage, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"scientific usage {name} must be finite and non-negative")


def _validate_stage_statistics(value: object) -> None:
    for name in (
        "input_rows",
        "output_rows",
        "symbols",
        "disk_read_bytes",
        "cache_hits",
        "cache_misses",
    ):
        observed = getattr(value, name)
        if not isinstance(observed, int) or isinstance(observed, bool) or observed < 0:
            raise ValueError(f"scientific statistic {name} must be non-negative")
    worker_count = getattr(value, "worker_count")
    if (
        not isinstance(worker_count, int)
        or isinstance(worker_count, bool)
        or worker_count <= 0
    ):
        raise ValueError("scientific statistic worker_count must be positive")


def _usage_dict(usage: RuntimeUsage) -> dict[str, int | float]:
    return {
        "attempts_reserved": usage.attempts_reserved,
        "attempts_active": usage.attempts_active,
        "attempts_succeeded": usage.attempts_succeeded,
        "attempts_failed": usage.attempts_failed,
        "attempts_expired": usage.attempts_expired,
        "attempts_cancelled": usage.attempts_cancelled,
        "wall_seconds": usage.wall_seconds,
        "cpu_seconds": usage.cpu_seconds,
        "peak_memory_bytes": usage.peak_memory_bytes,
        "disk_write_bytes": usage.disk_write_bytes,
        "llm_calls": usage.llm_calls,
        "llm_tokens": usage.llm_tokens,
        "llm_cost_microusd": usage.llm_cost_microusd,
    }


def _statistics_dict(value: object) -> dict[str, int]:
    return {
        "input_rows": int(getattr(value, "input_rows")),
        "output_rows": int(getattr(value, "output_rows")),
        "symbols": int(getattr(value, "symbols")),
        "disk_read_bytes": int(getattr(value, "disk_read_bytes")),
        "cache_hits": int(getattr(value, "cache_hits")),
        "cache_misses": int(getattr(value, "cache_misses")),
        "worker_count": int(getattr(value, "worker_count")),
    }


__all__ = [
    "AgentContextPolicy",
    "AgentRoleAdapter",
    "AgentStageDecision",
    "AgentStageInvocation",
    "ControllerStageHandler",
    "DeveloperStageHandler",
    "ExecutorStageHandler",
    "MemoryVisibility",
    "PlannerStageHandler",
    "ProtectedEvaluationRequest",
    "ProtectedTerminalEvaluationGateway",
    "ProtectedTerminalEvaluator",
    "ProtectedTerminalResult",
    "ProtectedTerminalStageHandler",
    "ResearchExecutionBindings",
    "ResearcherStageHandler",
    "ReviewerStageHandler",
    "STAGE_AGENT_ROLES",
    "ScientificStageImplementation",
    "ScientificStageRequest",
    "ScientificStageResult",
    "StageMemoryRecord",
    "bridge_research_run",
    "build_governed_stage_handlers",
]
