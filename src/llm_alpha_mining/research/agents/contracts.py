from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from llm_alpha_mining.research.core.hashing import hash_json, require_sha256
from llm_alpha_mining.research.experiments import DataPartition, ExperimentSpec
from llm_alpha_mining.mining.llm.safe_context import assert_safe_context


class AgentRole(str, Enum):
    RESEARCHER = "researcher"
    PLANNER = "planner"
    DEVELOPER = "developer"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    CONTROLLER = "controller"


ROLE_ALLOWED_CAPABILITIES: Mapping[AgentRole, tuple[str, ...]] = MappingProxyType(
    {
        AgentRole.RESEARCHER: (
            "hypothesis_catalog_read",
            "structured_llm_propose",
        ),
        AgentRole.PLANNER: (
            "contract_catalog_read",
            "experiment_plan_write",
        ),
        AgentRole.DEVELOPER: (
            "controlled_dsl_compile",
            "factor_registry_write",
        ),
        AgentRole.EXECUTOR: (
            "artifact_store_write",
            "isolated_stage_execute",
        ),
        AgentRole.REVIEWER: (
            "audit_evidence_read",
            "structured_llm_review",
        ),
        AgentRole.CONTROLLER: (
            "budget_state_read",
            "scheduler_decision_write",
        ),
    }
)

_NETWORK_CAPABILITIES = frozenset({"network", "shell", "production_publish"})


@dataclass(frozen=True, slots=True)
class AgentContext:
    experiment_spec_hash: str
    visible_partition_hashes: Mapping[DataPartition, str]
    data_schema_hashes: tuple[str, ...]
    allowed_fields: tuple[str, ...]
    allowed_operators: tuple[str, ...]
    allowed_windows: tuple[int, ...]
    hypothesis_hashes: tuple[str, ...]
    sanitized_feedback_codes: tuple[str, ...]
    schema_version: str = "agent-context/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "agent-context/v1":
            raise ValueError("unsupported AgentContext schema")
        require_sha256(self.experiment_spec_hash, name="agent context experiment hash")
        partitions = {
            DataPartition(name): require_sha256(digest, name=f"agent partition:{name}")
            for name, digest in self.visible_partition_hashes.items()
        }
        forbidden = set(partitions).difference(
            {DataPartition.DISCOVERY, DataPartition.TRAIN}
        )
        if forbidden:
            names = ",".join(sorted(item.value for item in forbidden))
            raise ValueError(f"agent context includes protected partitions:{names}")
        object.__setattr__(
            self,
            "visible_partition_hashes",
            MappingProxyType(
                dict(sorted(partitions.items(), key=lambda item: str(item[0])))
            ),
        )
        for collection_name in (
            "data_schema_hashes",
            "hypothesis_hashes",
        ):
            values = tuple(getattr(self, collection_name))
            if len(set(values)) != len(values):
                raise ValueError(f"agent {collection_name} must be unique")
            for digest in values:
                require_sha256(digest, name=f"agent {collection_name}")
            object.__setattr__(self, collection_name, values)
        for collection_name in (
            "allowed_fields",
            "allowed_operators",
            "allowed_windows",
            "sanitized_feedback_codes",
        ):
            values = tuple(getattr(self, collection_name))
            if len(set(values)) != len(values):
                raise ValueError(f"agent {collection_name} must be unique")
            object.__setattr__(self, collection_name, values)
        if not self.allowed_fields or not self.allowed_operators:
            raise ValueError("agent context requires fields and operators")
        if any(
            not isinstance(window, int) or isinstance(window, bool) or window <= 0
            for window in self.allowed_windows
        ):
            raise ValueError("agent allowed_windows must be positive integers")
        assert_safe_context(self.to_dict())

    @classmethod
    def bind(
        cls,
        spec: ExperimentSpec,
        *,
        data_schema_hashes: tuple[str, ...],
        allowed_fields: tuple[str, ...],
        allowed_operators: tuple[str, ...],
        allowed_windows: tuple[int, ...],
        hypothesis_hashes: tuple[str, ...] = (),
        sanitized_feedback_codes: tuple[str, ...] = (),
    ) -> "AgentContext":
        partitions = {
            partition: spec.data_partitions[partition]
            for partition in spec.agent_visible_partitions
        }
        return cls(
            experiment_spec_hash=spec.content_hash,
            visible_partition_hashes=partitions,
            data_schema_hashes=data_schema_hashes,
            allowed_fields=allowed_fields,
            allowed_operators=allowed_operators,
            allowed_windows=allowed_windows,
            hypothesis_hashes=hypothesis_hashes,
            sanitized_feedback_codes=sanitized_feedback_codes,
        )

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_spec_hash": self.experiment_spec_hash,
            "visible_partition_hashes": {
                partition.value: digest
                for partition, digest in self.visible_partition_hashes.items()
            },
            "data_schema_hashes": list(self.data_schema_hashes),
            "allowed_fields": list(self.allowed_fields),
            "allowed_operators": list(self.allowed_operators),
            "allowed_windows": list(self.allowed_windows),
            "hypothesis_hashes": list(self.hypothesis_hashes),
            "sanitized_feedback_codes": list(self.sanitized_feedback_codes),
        }


@dataclass(frozen=True, slots=True)
class AgentTask:
    task_id: str
    role: AgentRole
    experiment_spec_hash: str
    context_hash: str
    input_artifact_hashes: tuple[str, ...]
    output_schema_hash: str
    allowed_capabilities: tuple[str, ...]
    maximum_output_artifacts: int
    schema_version: str = "agent-task/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "agent-task/v1":
            raise ValueError("unsupported AgentTask schema")
        object.__setattr__(self, "role", AgentRole(self.role))
        if not _safe_code(self.task_id):
            raise ValueError("agent task_id is invalid")
        for name in (
            "experiment_spec_hash",
            "context_hash",
            "output_schema_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"agent task {name}")
        if len(set(self.input_artifact_hashes)) != len(self.input_artifact_hashes):
            raise ValueError("agent task input artifacts must be unique")
        for digest in self.input_artifact_hashes:
            require_sha256(digest, name="agent task input artifact")
        capabilities = tuple(sorted(set(self.allowed_capabilities)))
        expected = ROLE_ALLOWED_CAPABILITIES[self.role]
        if capabilities != tuple(sorted(expected)):
            raise ValueError("agent capabilities differ from the role allowlist")
        if _NETWORK_CAPABILITIES.intersection(capabilities):
            raise ValueError("agent task contains a forbidden broad capability")
        object.__setattr__(self, "allowed_capabilities", capabilities)
        if (
            not isinstance(self.maximum_output_artifacts, int)
            or self.maximum_output_artifacts <= 0
        ):
            raise ValueError("agent maximum_output_artifacts must be positive")

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        role: AgentRole,
        spec: ExperimentSpec,
        context: AgentContext,
        input_artifact_hashes: tuple[str, ...],
        output_schema_hash: str,
        maximum_output_artifacts: int,
    ) -> "AgentTask":
        if context.experiment_spec_hash != spec.content_hash:
            raise ValueError("agent task context belongs to another experiment")
        return cls(
            task_id=task_id,
            role=role,
            experiment_spec_hash=spec.content_hash,
            context_hash=context.content_hash,
            input_artifact_hashes=input_artifact_hashes,
            output_schema_hash=output_schema_hash,
            allowed_capabilities=ROLE_ALLOWED_CAPABILITIES[role],
            maximum_output_artifacts=maximum_output_artifacts,
        )

    @property
    def content_hash(self) -> str:
        return hash_json(
            {
                "schema_version": self.schema_version,
                "task_id": self.task_id,
                "role": self.role.value,
                "experiment_spec_hash": self.experiment_spec_hash,
                "context_hash": self.context_hash,
                "input_artifact_hashes": list(self.input_artifact_hashes),
                "output_schema_hash": self.output_schema_hash,
                "allowed_capabilities": list(self.allowed_capabilities),
                "maximum_output_artifacts": self.maximum_output_artifacts,
            }
        )


@dataclass(frozen=True, slots=True)
class AgentResult:
    task_hash: str
    status: str
    output_artifact_hashes: tuple[str, ...]
    decision_code: str
    request_hash: str | None = None
    response_hash: str | None = None
    model_id_hash: str | None = None
    usage_hash: str | None = None

    def __post_init__(self) -> None:
        require_sha256(self.task_hash, name="agent result task_hash")
        if self.status not in {"succeeded", "rejected", "failed"}:
            raise ValueError("agent result status is invalid")
        if not _safe_code(self.decision_code):
            raise ValueError("agent result decision_code is invalid")
        for digest in self.output_artifact_hashes:
            require_sha256(digest, name="agent result artifact hash")
        llm_values = (
            self.request_hash,
            self.response_hash,
            self.model_id_hash,
            self.usage_hash,
        )
        if any(value is not None for value in llm_values) and not all(
            value is not None for value in llm_values
        ):
            raise ValueError("agent LLM lineage must be complete or absent")
        for value in llm_values:
            if value is not None:
                require_sha256(value, name="agent LLM lineage hash")

    @property
    def content_hash(self) -> str:
        return hash_json(
            {
                "task_hash": self.task_hash,
                "status": self.status,
                "output_artifact_hashes": list(self.output_artifact_hashes),
                "decision_code": self.decision_code,
                "request_hash": self.request_hash,
                "response_hash": self.response_hash,
                "model_id_hash": self.model_id_hash,
                "usage_hash": self.usage_hash,
            }
        )


def _safe_code(value: str) -> bool:
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-")
    return value[0].isalnum() and all(character in allowed for character in value)


__all__ = [
    "AgentContext",
    "AgentResult",
    "AgentRole",
    "AgentTask",
    "ROLE_ALLOWED_CAPABILITIES",
]
