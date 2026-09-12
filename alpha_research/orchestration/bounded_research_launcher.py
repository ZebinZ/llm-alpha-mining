"""Fail-closed planning boundary for a future bounded research supervisor.

V1 intentionally does not spawn processes.  The existing
``IsolatedProcessExecutor`` and ``ExperimentRuntime`` do not together provide a
birth-pinned, crash-recoverable two-worker supervisor: callbacks can escape,
process-tree identity is not durable, and the runtime owns one SQLite
connection.  This module therefore produces a canonical, task-bound admission
ledger and rejects both legacy direct execution and untrusted supervisor
objects.  A later adapter may activate the ledger only after independently
providing birth identity, source-capsule verification, process-group reaping,
and immutable worker-artifact receipts.

The absence of filesystem, registry, artifact-store, subprocess, thread, clock,
or network imports is deliberate.  Default use is pure and performs zero
writes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import NoReturn, Protocol, cast

from alpha_research.core.hashing import canonical_json_bytes, hash_json
from alpha_research.orchestration.adaptive_dispatch_policy import (
    AdaptiveDispatchPolicyV1,
    AdaptiveDispatchRequestV1,
    AdaptiveDispatchStateV1,
    evaluate_adaptive_dispatch_v1,
)
from alpha_research.orchestration.process import (
    IsolatedProcessExecutor,
    ProcessTask,
)
from alpha_research.orchestration.runtime import (
    ExperimentRuntime,
    bounded_attempt_scope_hash,
)
from alpha_research.orchestration.resource_observation_provider import (
    AdaptiveResourceObservationAuthorityV1,
)


_GIB = 1024 * 1024 * 1024
WORKER_HARD_RSS_BYTES = 5 * _GIB // 2
PROJECT_HARD_RSS_BYTES = 12 * _GIB
FOREGROUND_RESERVE_BYTES = 4 * _GIB
MAXIMUM_WORKERS = 2
NUMERIC_THREAD_LIMIT = 1
PROCESS_NICE = 5
MAXIMUM_ATTEMPTS_PER_TASK = 2
DURABLE_ADMISSION_CAS_KIND = "unique-primary-key-transactional-cas/v1"


class ResearchTaskSourceV1(str, Enum):
    SEALED = "sealed"
    SYNTHETIC = "synthetic"


class BoundedLaunchReasonV1(str, Enum):
    SUPERVISOR_CAPABILITY_UNAVAILABLE = "supervisor_capability_unavailable"
    UNTRUSTED_SUPERVISOR_CAPABILITY = "untrusted_supervisor_capability"
    UNSAFE_LEGACY_BACKEND_REJECTED = "unsafe_legacy_backend_rejected"


class BirthPinnedSupervisorCapabilityV1(Protocol):
    """Interface only; V1 accepts no public implementation as execution authority.

    A future trusted adapter must be injected by a higher control plane.  Merely
    satisfying this structural protocol never activates execution in this V1.
    """

    @property
    def capability_hash(self) -> str: ...

    @property
    def source_capsule_hashes(self) -> tuple[str, ...]: ...

    @property
    def durable_admission_cas_kind(self) -> str: ...

    def consume_admission_unique_cas(
        self,
        *,
        bounded_admission_token: str,
        natural_key: str,
        task_hash: str,
        candidate_lineage_hash: str,
        source_capsule_hash: str,
        sealed_process_task_hash: str,
        queue_ordinal: int,
        attempt_number: int,
        policy_admission_token: str,
        resource_observation_authority_hash: str,
        worker_hard_rss_bytes: int,
        numeric_thread_limit: int,
        process_nice: int,
    ) -> str: ...


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a lowercase sha256 digest")
    text = value
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} must be a lowercase sha256 digest")
    return text


def _exact_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be non-empty text")
    return value


def _exact_object(
    value: object,
    *,
    expected_keys: frozenset[str],
    name: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact JSON object")
    mapping = cast(dict[object, object], value)
    if any(type(key) is not str for key in mapping):
        raise TypeError(f"{name} keys must be exact text")
    typed = cast(dict[str, object], mapping)
    actual = frozenset(typed)
    if actual != expected_keys:
        missing = sorted(expected_keys - actual)
        extra = sorted(actual - expected_keys)
        raise ValueError(f"{name} fields differ; missing={missing!r}, extra={extra!r}")
    return typed


def _exact_list(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{name} must be an exact JSON array")
    return cast(list[object], value)


def _wire_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    return value


def _wire_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact boolean")
    return value


def _wire_number(value: object, *, name: str) -> int | float:
    if type(value) is int:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError(f"{name} must be an exact finite JSON number")


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _pairs_to_exact_dict(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def _decode_canonical_wire(value: object, *, name: str) -> dict[str, object]:
    if type(value) is not bytes:
        raise TypeError(f"{name} wire must be exact bytes")
    wire = value
    if not wire.endswith(b"\n") or wire.endswith(b"\n\n"):
        raise ValueError(f"{name} wire must have exactly one trailing newline")
    try:
        decoded = cast(
            object,
            json.loads(
                wire[:-1].decode("utf-8"),
                object_pairs_hook=_pairs_to_exact_dict,
                parse_constant=_reject_json_constant,
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} wire is not valid UTF-8 JSON") from exc
    if type(decoded) is not dict:
        raise TypeError(f"{name} wire root must be an exact JSON object")
    if canonical_json_bytes(decoded) + b"\n" != wire:
        raise ValueError(f"{name} wire is not canonical")
    return cast(dict[str, object], decoded)


def _process_task_from_mapping(value: object) -> ProcessTask:
    mapping = _exact_object(
        value,
        expected_keys=frozenset(
            {
                "schema_version",
                "task_id",
                "entrypoint",
                "payload",
                "timeout_seconds",
                "cpu_limit_seconds",
                "memory_limit_bytes",
                "output_limit_bytes",
            }
        ),
        name="bounded ProcessTask",
    )
    schema_version = _exact_text(
        mapping["schema_version"], name="ProcessTask schema_version"
    )
    if schema_version != "process-task/v1":
        raise ValueError("unsupported ProcessTask schema")
    return ProcessTask(
        task_id=_exact_text(mapping["task_id"], name="ProcessTask task_id"),
        entrypoint=_exact_text(mapping["entrypoint"], name="ProcessTask entrypoint"),
        payload=_copy_plain_json(mapping["payload"]),
        timeout_seconds=_wire_number(
            mapping["timeout_seconds"], name="ProcessTask timeout_seconds"
        ),
        cpu_limit_seconds=_wire_int(
            mapping["cpu_limit_seconds"], name="ProcessTask cpu_limit_seconds"
        ),
        memory_limit_bytes=_wire_int(
            mapping["memory_limit_bytes"], name="ProcessTask memory_limit_bytes"
        ),
        output_limit_bytes=_wire_int(
            mapping["output_limit_bytes"], name="ProcessTask output_limit_bytes"
        ),
        schema_version=schema_version,
    )


def _copy_plain_json(
    value: object,
    *,
    _depth: int = 0,
    _seen: set[int] | None = None,
    _nodes: list[int] | None = None,
) -> object:
    """Copy an exact built-in JSON tree without invoking user callbacks."""

    if _depth > 64:
        raise ValueError("bounded task payload exceeds maximum JSON depth")
    seen = set() if _seen is None else _seen
    nodes = [0] if _nodes is None else _nodes
    nodes[0] += 1
    if nodes[0] > 100_000:
        raise ValueError("bounded task payload exceeds maximum JSON nodes")
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("bounded task payload contains a non-finite number")
        return value
    if type(value) is dict:
        identity = id(value)
        if identity in seen:
            raise ValueError("bounded task payload contains a cycle")
        seen.add(identity)
        mapping = cast(dict[object, object], value)
        if any(type(key) is not str for key in mapping):
            raise TypeError("bounded task payload object keys must be exact text")
        keys = cast(list[str], list(mapping))
        copied = {
            key: _copy_plain_json(
                mapping[key],
                _depth=_depth + 1,
                _seen=seen,
                _nodes=nodes,
            )
            for key in sorted(keys)
        }
        seen.remove(identity)
        return copied
    if type(value) in {list, tuple}:
        identity = id(value)
        if identity in seen:
            raise ValueError("bounded task payload contains a cycle")
        seen.add(identity)
        sequence = cast(list[object] | tuple[object, ...], value)
        copied_sequence = [
            _copy_plain_json(
                item,
                _depth=_depth + 1,
                _seen=seen,
                _nodes=nodes,
            )
            for item in sequence
        ]
        seen.remove(identity)
        return copied_sequence
    raise TypeError("bounded task payload must be an exact built-in JSON tree")


def _process_task_mapping(task: ProcessTask) -> dict[str, object]:
    if type(task) is not ProcessTask:
        raise TypeError("bounded task requires an exact ProcessTask")
    return {
        "schema_version": task.schema_version,
        "task_id": task.task_id,
        "entrypoint": task.entrypoint,
        "payload": _copy_plain_json(task.payload),
        "timeout_seconds": task.timeout_seconds,
        "cpu_limit_seconds": task.cpu_limit_seconds,
        "memory_limit_bytes": task.memory_limit_bytes,
        "output_limit_bytes": task.output_limit_bytes,
    }


def _sealed_process_task_hash(task: ProcessTask) -> str:
    return cast(str, hash_json(_process_task_mapping(task)))


@dataclass(frozen=True, slots=True)
class BoundedResearchTaskV1:
    """One immutable queue item; it is data, never execution authority."""

    queue_ordinal: int
    stage: str
    source: ResearchTaskSourceV1 | str
    candidate_lineage_hash: str
    source_capsule_hash: str
    sealed_process_task_hash: str
    process_task: ProcessTask
    maximum_attempts: int = 1
    research_only: bool = True
    production_ready: bool = False
    execution_activated: bool = False
    schema_version: str = "bounded-research-task/v1"

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != "bounded-research-task/v1"
        ):
            raise ValueError("unsupported bounded research task")
        if type(self.queue_ordinal) is not int or self.queue_ordinal < 0:
            raise ValueError(
                "bounded task queue_ordinal must be a non-negative integer"
            )
        _exact_text(self.stage, name="bounded task stage")
        if (
            len(self.stage) > 96
            or not self.stage[0].isalnum()
            or any(
                character
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
                for character in self.stage
            )
        ):
            raise ValueError("bounded task stage is not portable")
        try:
            object.__setattr__(self, "source", ResearchTaskSourceV1(self.source))
        except (TypeError, ValueError) as exc:
            raise ValueError("bounded task source is invalid") from exc
        for name in (
            "candidate_lineage_hash",
            "source_capsule_hash",
            "sealed_process_task_hash",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if type(self.process_task) is not ProcessTask:
            raise TypeError("bounded task requires an exact ProcessTask")
        if self.sealed_process_task_hash != _sealed_process_task_hash(
            self.process_task
        ):
            raise ValueError("bounded task sealed ProcessTask identity differs")
        if self.process_task.memory_limit_bytes != WORKER_HARD_RSS_BYTES:
            raise ValueError("bounded task worker hard RSS must be exactly 2.5 GiB")
        if (
            type(self.maximum_attempts) is not int
            or not 1 <= self.maximum_attempts <= MAXIMUM_ATTEMPTS_PER_TASK
        ):
            raise ValueError("bounded task maximum_attempts must be one or two")
        if (
            self.research_only is not True
            or self.production_ready is not False
            or self.execution_activated is not False
        ):
            raise ValueError(
                "bounded tasks must remain non-executing research-only data"
            )

    @property
    def natural_key(self) -> str:
        return cast(
            str,
            hash_json(
                {
                    "schema_version": "bounded-research-natural-key/v1",
                    "stage": self.stage,
                    "candidate_lineage_hash": self.candidate_lineage_hash,
                    "source_capsule_hash": self.source_capsule_hash,
                    "sealed_process_task_hash": self.sealed_process_task_hash,
                }
            ),
        )

    def runtime_attempt_scope_hash(self, experiment_spec_hash: str) -> str:
        """Bind this full immutable task identity to one runtime experiment."""

        return bounded_attempt_scope_hash(
            experiment_spec_hash=experiment_spec_hash,
            stage=self.stage,
            task_natural_key=self.natural_key,
            candidate_lineage_hash=self.candidate_lineage_hash,
            source_capsule_hash=self.source_capsule_hash,
            sealed_process_task_hash=self.sealed_process_task_hash,
        )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        process_task = _process_task_mapping(self.process_task)
        if self.sealed_process_task_hash != hash_json(process_task):
            raise ValueError("bounded task ProcessTask mutated after sealing")
        return {
            "queue_ordinal": self.queue_ordinal,
            "stage": self.stage,
            "source": cast(ResearchTaskSourceV1, self.source).value,
            "candidate_lineage_hash": self.candidate_lineage_hash,
            "source_capsule_hash": self.source_capsule_hash,
            "sealed_process_task_hash": self.sealed_process_task_hash,
            "process_task": process_task,
            "maximum_attempts": self.maximum_attempts,
            "natural_key": self.natural_key,
            "research_only": self.research_only,
            "production_ready": self.production_ready,
            "execution_activated": self.execution_activated,
            "schema_version": self.schema_version,
        }

    def to_wire_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self.to_dict())) + b"\n"

    @classmethod
    def from_mapping(cls, value: object) -> BoundedResearchTaskV1:
        mapping = _exact_object(
            value,
            expected_keys=frozenset(
                {
                    "queue_ordinal",
                    "stage",
                    "source",
                    "candidate_lineage_hash",
                    "source_capsule_hash",
                    "sealed_process_task_hash",
                    "process_task",
                    "maximum_attempts",
                    "natural_key",
                    "research_only",
                    "production_ready",
                    "execution_activated",
                    "schema_version",
                }
            ),
            name="bounded research task",
        )
        task = cls(
            queue_ordinal=_wire_int(
                mapping["queue_ordinal"], name="bounded task queue_ordinal"
            ),
            stage=_exact_text(mapping["stage"], name="bounded task stage"),
            source=_exact_text(mapping["source"], name="bounded task source"),
            candidate_lineage_hash=_digest(
                mapping["candidate_lineage_hash"],
                name="candidate_lineage_hash",
            ),
            source_capsule_hash=_digest(
                mapping["source_capsule_hash"], name="source_capsule_hash"
            ),
            sealed_process_task_hash=_digest(
                mapping["sealed_process_task_hash"],
                name="sealed_process_task_hash",
            ),
            process_task=_process_task_from_mapping(mapping["process_task"]),
            maximum_attempts=_wire_int(
                mapping["maximum_attempts"], name="bounded task maximum_attempts"
            ),
            research_only=_wire_bool(
                mapping["research_only"], name="bounded task research_only"
            ),
            production_ready=_wire_bool(
                mapping["production_ready"], name="bounded task production_ready"
            ),
            execution_activated=_wire_bool(
                mapping["execution_activated"],
                name="bounded task execution_activated",
            ),
            schema_version=_exact_text(
                mapping["schema_version"], name="bounded task schema_version"
            ),
        )
        natural_key = _digest(mapping["natural_key"], name="natural_key")
        if natural_key != task.natural_key:
            raise ValueError("bounded task natural key binding differs")
        return task

    @classmethod
    def from_wire_bytes(cls, value: object) -> BoundedResearchTaskV1:
        return cls.from_mapping(_decode_canonical_wire(value, name="bounded task"))


@dataclass(frozen=True, slots=True)
class BoundedResearchPlanV1:
    """Canonical queue and fixed resource envelope for at most two workers."""

    experiment_spec_hash: str
    tasks: tuple[BoundedResearchTaskV1, ...]
    allowed_entrypoints: tuple[str, ...]
    policy_hash: str
    maximum_workers: int = MAXIMUM_WORKERS
    numeric_thread_limit: int = NUMERIC_THREAD_LIMIT
    process_nice: int = PROCESS_NICE
    worker_hard_rss_bytes: int = WORKER_HARD_RSS_BYTES
    project_hard_rss_bytes: int = PROJECT_HARD_RSS_BYTES
    foreground_reserve_bytes: int = FOREGROUND_RESERVE_BYTES
    canonical_reducer_single_writer: bool = True
    durable_admission_cas_required: bool = True
    research_only: bool = True
    production_ready: bool = False
    execution_activated: bool = False
    schema_version: str = "bounded-research-plan/v1"

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != "bounded-research-plan/v1"
        ):
            raise ValueError("unsupported bounded research plan")
        object.__setattr__(
            self,
            "experiment_spec_hash",
            _digest(self.experiment_spec_hash, name="experiment_spec_hash"),
        )
        if (
            type(self.tasks) is not tuple
            or not self.tasks
            or any(type(item) is not BoundedResearchTaskV1 for item in self.tasks)
        ):
            raise TypeError("bounded plan tasks must be a non-empty exact tuple")
        if tuple(item.queue_ordinal for item in self.tasks) != tuple(
            range(len(self.tasks))
        ):
            raise ValueError(
                "bounded plan queue ordinals must be contiguous and canonical"
            )
        if len({item.natural_key for item in self.tasks}) != len(self.tasks):
            raise ValueError("bounded plan natural keys must be unique")
        if type(self.allowed_entrypoints) is not tuple or not self.allowed_entrypoints:
            raise TypeError("bounded plan allowlist must be an exact non-empty tuple")
        if any(type(item) is not str for item in self.allowed_entrypoints):
            raise TypeError("bounded plan allowlist must contain only text")
        if tuple(sorted(set(self.allowed_entrypoints))) != self.allowed_entrypoints:
            raise ValueError("bounded plan allowlist must be sorted and unique")
        if any(
            task.process_task.entrypoint not in self.allowed_entrypoints
            for task in self.tasks
        ):
            raise PermissionError("bounded task entrypoint is not plan-allowlisted")
        object.__setattr__(
            self, "policy_hash", _digest(self.policy_hash, name="policy_hash")
        )
        exact = {
            "maximum_workers": MAXIMUM_WORKERS,
            "numeric_thread_limit": NUMERIC_THREAD_LIMIT,
            "process_nice": PROCESS_NICE,
            "worker_hard_rss_bytes": WORKER_HARD_RSS_BYTES,
            "project_hard_rss_bytes": PROJECT_HARD_RSS_BYTES,
            "foreground_reserve_bytes": FOREGROUND_RESERVE_BYTES,
            "canonical_reducer_single_writer": True,
            "durable_admission_cas_required": True,
            "research_only": True,
            "production_ready": False,
            "execution_activated": False,
        }
        if any(
            type(getattr(self, name)) is not type(value) or getattr(self, name) != value
            for name, value in exact.items()
        ):
            raise ValueError("bounded plan fixed research envelope differs")

    @property
    def workload_hash(self) -> str:
        return cast(
            str,
            hash_json(
                {
                    "schema_version": "bounded-research-workload/v1",
                    "task_hashes": [item.content_hash for item in self.tasks],
                }
            ),
        )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_spec_hash": self.experiment_spec_hash,
            "tasks": [item.to_dict() for item in self.tasks],
            "allowed_entrypoints": list(self.allowed_entrypoints),
            "policy_hash": self.policy_hash,
            "workload_hash": self.workload_hash,
            "maximum_workers": self.maximum_workers,
            "numeric_thread_limit": self.numeric_thread_limit,
            "process_nice": self.process_nice,
            "worker_hard_rss_bytes": self.worker_hard_rss_bytes,
            "project_hard_rss_bytes": self.project_hard_rss_bytes,
            "foreground_reserve_bytes": self.foreground_reserve_bytes,
            "canonical_reducer_single_writer": self.canonical_reducer_single_writer,
            "durable_admission_cas_required": self.durable_admission_cas_required,
            "research_only": self.research_only,
            "production_ready": self.production_ready,
            "execution_activated": self.execution_activated,
            "schema_version": self.schema_version,
        }

    def to_wire_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self.to_dict())) + b"\n"

    @classmethod
    def from_mapping(cls, value: object) -> BoundedResearchPlanV1:
        mapping = _exact_object(
            value,
            expected_keys=frozenset(
                {
                    "experiment_spec_hash",
                    "tasks",
                    "allowed_entrypoints",
                    "policy_hash",
                    "workload_hash",
                    "maximum_workers",
                    "numeric_thread_limit",
                    "process_nice",
                    "worker_hard_rss_bytes",
                    "project_hard_rss_bytes",
                    "foreground_reserve_bytes",
                    "canonical_reducer_single_writer",
                    "durable_admission_cas_required",
                    "research_only",
                    "production_ready",
                    "execution_activated",
                    "schema_version",
                }
            ),
            name="bounded research plan",
        )
        raw_tasks = _exact_list(mapping["tasks"], name="bounded plan tasks")
        raw_entrypoints = _exact_list(
            mapping["allowed_entrypoints"], name="bounded plan allowed_entrypoints"
        )
        if any(type(item) is not str for item in raw_entrypoints):
            raise TypeError("bounded plan allowlist must contain exact text")
        plan = cls(
            experiment_spec_hash=_digest(
                mapping["experiment_spec_hash"], name="experiment_spec_hash"
            ),
            tasks=tuple(BoundedResearchTaskV1.from_mapping(item) for item in raw_tasks),
            allowed_entrypoints=tuple(cast(list[str], raw_entrypoints)),
            policy_hash=_digest(mapping["policy_hash"], name="policy_hash"),
            maximum_workers=_wire_int(
                mapping["maximum_workers"], name="bounded plan maximum_workers"
            ),
            numeric_thread_limit=_wire_int(
                mapping["numeric_thread_limit"],
                name="bounded plan numeric_thread_limit",
            ),
            process_nice=_wire_int(
                mapping["process_nice"], name="bounded plan process_nice"
            ),
            worker_hard_rss_bytes=_wire_int(
                mapping["worker_hard_rss_bytes"],
                name="bounded plan worker_hard_rss_bytes",
            ),
            project_hard_rss_bytes=_wire_int(
                mapping["project_hard_rss_bytes"],
                name="bounded plan project_hard_rss_bytes",
            ),
            foreground_reserve_bytes=_wire_int(
                mapping["foreground_reserve_bytes"],
                name="bounded plan foreground_reserve_bytes",
            ),
            canonical_reducer_single_writer=_wire_bool(
                mapping["canonical_reducer_single_writer"],
                name="bounded plan canonical_reducer_single_writer",
            ),
            durable_admission_cas_required=_wire_bool(
                mapping["durable_admission_cas_required"],
                name="bounded plan durable_admission_cas_required",
            ),
            research_only=_wire_bool(
                mapping["research_only"], name="bounded plan research_only"
            ),
            production_ready=_wire_bool(
                mapping["production_ready"], name="bounded plan production_ready"
            ),
            execution_activated=_wire_bool(
                mapping["execution_activated"],
                name="bounded plan execution_activated",
            ),
            schema_version=_exact_text(
                mapping["schema_version"], name="bounded plan schema_version"
            ),
        )
        workload_hash = _digest(mapping["workload_hash"], name="workload_hash")
        if workload_hash != plan.workload_hash:
            raise ValueError("bounded plan workload hash binding differs")
        return plan

    @classmethod
    def from_wire_bytes(cls, value: object) -> BoundedResearchPlanV1:
        return cls.from_mapping(_decode_canonical_wire(value, name="bounded plan"))


@dataclass(frozen=True, slots=True)
class BoundedAdmissionIntentV1:
    """Queue-head-bound preview; not a consumed token or spawn authority."""

    queue_ordinal: int
    task_hash: str
    natural_key: str
    candidate_lineage_hash: str
    source_capsule_hash: str
    sealed_process_task_hash: str
    attempt_number: int
    policy_admission_token: str
    resource_observation_authority_hash: str
    bounded_admission_token: str
    worker_hard_rss_bytes: int = WORKER_HARD_RSS_BYTES
    numeric_thread_limit: int = NUMERIC_THREAD_LIMIT
    process_nice: int = PROCESS_NICE
    durable_admission_cas_required: bool = True
    admission_token_consumed: bool = False
    research_only: bool = True
    production_ready: bool = False
    execution_activated: bool = False
    schema_version: str = "bounded-admission-intent/v1"

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != "bounded-admission-intent/v1"
        ):
            raise ValueError("unsupported bounded admission intent")
        for name in (
            "task_hash",
            "natural_key",
            "candidate_lineage_hash",
            "source_capsule_hash",
            "sealed_process_task_hash",
            "policy_admission_token",
            "resource_observation_authority_hash",
            "bounded_admission_token",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if type(self.queue_ordinal) is not int or self.queue_ordinal < 0:
            raise ValueError("bounded intent queue ordinal is invalid")
        if type(self.attempt_number) is not int or self.attempt_number != 1:
            raise ValueError("V1 planning may preview only the first attempt")
        expected = hash_json(
            {
                "schema_version": "bounded-admission-token/v1",
                "queue_ordinal": self.queue_ordinal,
                "task_hash": self.task_hash,
                "natural_key": self.natural_key,
                "candidate_lineage_hash": self.candidate_lineage_hash,
                "source_capsule_hash": self.source_capsule_hash,
                "sealed_process_task_hash": self.sealed_process_task_hash,
                "attempt_number": self.attempt_number,
                "policy_admission_token": self.policy_admission_token,
                "resource_observation_authority_hash": (
                    self.resource_observation_authority_hash
                ),
                "worker_hard_rss_bytes": self.worker_hard_rss_bytes,
                "numeric_thread_limit": self.numeric_thread_limit,
                "process_nice": self.process_nice,
            }
        )
        if self.bounded_admission_token != expected:
            raise ValueError("bounded admission token binding differs")
        fixed = {
            "worker_hard_rss_bytes": WORKER_HARD_RSS_BYTES,
            "numeric_thread_limit": NUMERIC_THREAD_LIMIT,
            "process_nice": PROCESS_NICE,
            "durable_admission_cas_required": True,
            "admission_token_consumed": False,
            "research_only": True,
            "production_ready": False,
            "execution_activated": False,
        }
        if any(
            type(getattr(self, name)) is not type(expected)
            or getattr(self, name) != expected
            for name, expected in fixed.items()
        ):
            raise ValueError("bounded intent must remain unconsumed research-only data")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def to_wire_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self.to_dict())) + b"\n"

    @classmethod
    def from_mapping(cls, value: object) -> BoundedAdmissionIntentV1:
        mapping = _exact_object(
            value,
            expected_keys=frozenset(
                {
                    "queue_ordinal",
                    "task_hash",
                    "natural_key",
                    "candidate_lineage_hash",
                    "source_capsule_hash",
                    "sealed_process_task_hash",
                    "attempt_number",
                    "policy_admission_token",
                    "resource_observation_authority_hash",
                    "bounded_admission_token",
                    "worker_hard_rss_bytes",
                    "numeric_thread_limit",
                    "process_nice",
                    "durable_admission_cas_required",
                    "admission_token_consumed",
                    "research_only",
                    "production_ready",
                    "execution_activated",
                    "schema_version",
                }
            ),
            name="bounded admission intent",
        )
        return cls(
            queue_ordinal=_wire_int(
                mapping["queue_ordinal"], name="bounded intent queue_ordinal"
            ),
            task_hash=_digest(mapping["task_hash"], name="task_hash"),
            natural_key=_digest(mapping["natural_key"], name="natural_key"),
            candidate_lineage_hash=_digest(
                mapping["candidate_lineage_hash"],
                name="candidate_lineage_hash",
            ),
            source_capsule_hash=_digest(
                mapping["source_capsule_hash"], name="source_capsule_hash"
            ),
            sealed_process_task_hash=_digest(
                mapping["sealed_process_task_hash"],
                name="sealed_process_task_hash",
            ),
            attempt_number=_wire_int(
                mapping["attempt_number"], name="bounded intent attempt_number"
            ),
            policy_admission_token=_digest(
                mapping["policy_admission_token"], name="policy_admission_token"
            ),
            resource_observation_authority_hash=_digest(
                mapping["resource_observation_authority_hash"],
                name="resource_observation_authority_hash",
            ),
            bounded_admission_token=_digest(
                mapping["bounded_admission_token"],
                name="bounded_admission_token",
            ),
            worker_hard_rss_bytes=_wire_int(
                mapping["worker_hard_rss_bytes"],
                name="bounded intent worker_hard_rss_bytes",
            ),
            numeric_thread_limit=_wire_int(
                mapping["numeric_thread_limit"],
                name="bounded intent numeric_thread_limit",
            ),
            process_nice=_wire_int(
                mapping["process_nice"], name="bounded intent process_nice"
            ),
            durable_admission_cas_required=_wire_bool(
                mapping["durable_admission_cas_required"],
                name="bounded intent durable_admission_cas_required",
            ),
            admission_token_consumed=_wire_bool(
                mapping["admission_token_consumed"],
                name="bounded intent admission_token_consumed",
            ),
            research_only=_wire_bool(
                mapping["research_only"], name="bounded intent research_only"
            ),
            production_ready=_wire_bool(
                mapping["production_ready"],
                name="bounded intent production_ready",
            ),
            execution_activated=_wire_bool(
                mapping["execution_activated"],
                name="bounded intent execution_activated",
            ),
            schema_version=_exact_text(
                mapping["schema_version"], name="bounded intent schema_version"
            ),
        )

    @classmethod
    def from_wire_bytes(cls, value: object) -> BoundedAdmissionIntentV1:
        return cls.from_mapping(_decode_canonical_wire(value, name="bounded intent"))


@dataclass(frozen=True, slots=True)
class BoundedPlanningLedgerV1:
    plan_hash: str
    workload_hash: str
    evaluation_hashes: tuple[str, ...]
    admission_intents: tuple[BoundedAdmissionIntentV1, ...]
    hold_reason: str | None
    maximum_workers: int = MAXIMUM_WORKERS
    admission_tokens_consumed: int = 0
    runtime_writes: int = 0
    ledger_writes: int = 0
    artifact_writes: int = 0
    durable_admission_cas_kind: str = DURABLE_ADMISSION_CAS_KIND
    durable_admission_claims_persisted: int = 0
    canonical_reducer_single_writer: bool = True
    durable_admission_cas_required: bool = True
    research_only: bool = True
    production_ready: bool = False
    execution_activated: bool = False
    schema_version: str = "bounded-planning-ledger/v1"

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != "bounded-planning-ledger/v1"
        ):
            raise ValueError("unsupported bounded planning ledger")
        for name in ("plan_hash", "workload_hash"):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if type(self.evaluation_hashes) is not tuple or any(
            type(item) is not str for item in self.evaluation_hashes
        ):
            raise TypeError("bounded ledger evaluation hashes must be an exact tuple")
        for digest in self.evaluation_hashes:
            _digest(digest, name="evaluation_hash")
        if type(self.admission_intents) is not tuple or any(
            type(item) is not BoundedAdmissionIntentV1
            for item in self.admission_intents
        ):
            raise TypeError("bounded ledger intents must be an exact tuple")
        if len(self.admission_intents) > MAXIMUM_WORKERS:
            raise ValueError("bounded ledger cannot preview more than two admissions")
        if len(
            {item.bounded_admission_token for item in self.admission_intents}
        ) != len(self.admission_intents):
            raise ValueError("bounded ledger admission tokens must be unique")
        if self.hold_reason is not None and type(self.hold_reason) is not str:
            raise TypeError("bounded ledger hold reason must be text or null")
        fixed = {
            "maximum_workers": MAXIMUM_WORKERS,
            "admission_tokens_consumed": 0,
            "runtime_writes": 0,
            "ledger_writes": 0,
            "artifact_writes": 0,
            "durable_admission_cas_kind": DURABLE_ADMISSION_CAS_KIND,
            "durable_admission_claims_persisted": 0,
            "canonical_reducer_single_writer": True,
            "durable_admission_cas_required": True,
            "research_only": True,
            "production_ready": False,
            "execution_activated": False,
        }
        if any(
            type(getattr(self, name)) is not type(expected)
            or getattr(self, name) != expected
            for name, expected in fixed.items()
        ):
            raise ValueError("bounded planning ledger cannot claim execution or writes")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_hash": self.plan_hash,
            "workload_hash": self.workload_hash,
            "evaluation_hashes": list(self.evaluation_hashes),
            "admission_intents": [item.to_dict() for item in self.admission_intents],
            "hold_reason": self.hold_reason,
            "maximum_workers": self.maximum_workers,
            "admission_tokens_consumed": self.admission_tokens_consumed,
            "runtime_writes": self.runtime_writes,
            "ledger_writes": self.ledger_writes,
            "artifact_writes": self.artifact_writes,
            "durable_admission_cas_kind": self.durable_admission_cas_kind,
            "durable_admission_claims_persisted": self.durable_admission_claims_persisted,
            "canonical_reducer_single_writer": self.canonical_reducer_single_writer,
            "durable_admission_cas_required": self.durable_admission_cas_required,
            "research_only": self.research_only,
            "production_ready": self.production_ready,
            "execution_activated": self.execution_activated,
            "schema_version": self.schema_version,
        }

    def to_wire_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self.to_dict())) + b"\n"

    @classmethod
    def from_mapping(cls, value: object) -> BoundedPlanningLedgerV1:
        mapping = _exact_object(
            value,
            expected_keys=frozenset(
                {
                    "plan_hash",
                    "workload_hash",
                    "evaluation_hashes",
                    "admission_intents",
                    "hold_reason",
                    "maximum_workers",
                    "admission_tokens_consumed",
                    "runtime_writes",
                    "ledger_writes",
                    "artifact_writes",
                    "durable_admission_cas_kind",
                    "durable_admission_claims_persisted",
                    "canonical_reducer_single_writer",
                    "durable_admission_cas_required",
                    "research_only",
                    "production_ready",
                    "execution_activated",
                    "schema_version",
                }
            ),
            name="bounded planning ledger",
        )
        raw_evaluations = _exact_list(
            mapping["evaluation_hashes"],
            name="bounded ledger evaluation_hashes",
        )
        evaluation_hashes = tuple(
            _digest(item, name="evaluation_hash") for item in raw_evaluations
        )
        raw_intents = _exact_list(
            mapping["admission_intents"], name="bounded ledger admission_intents"
        )
        hold_reason_value = mapping["hold_reason"]
        if hold_reason_value is not None and type(hold_reason_value) is not str:
            raise TypeError("bounded ledger hold_reason must be exact text or null")
        return cls(
            plan_hash=_digest(mapping["plan_hash"], name="plan_hash"),
            workload_hash=_digest(mapping["workload_hash"], name="workload_hash"),
            evaluation_hashes=evaluation_hashes,
            admission_intents=tuple(
                BoundedAdmissionIntentV1.from_mapping(item) for item in raw_intents
            ),
            hold_reason=hold_reason_value,
            maximum_workers=_wire_int(
                mapping["maximum_workers"], name="bounded ledger maximum_workers"
            ),
            admission_tokens_consumed=_wire_int(
                mapping["admission_tokens_consumed"],
                name="bounded ledger admission_tokens_consumed",
            ),
            runtime_writes=_wire_int(
                mapping["runtime_writes"], name="bounded ledger runtime_writes"
            ),
            ledger_writes=_wire_int(
                mapping["ledger_writes"], name="bounded ledger ledger_writes"
            ),
            artifact_writes=_wire_int(
                mapping["artifact_writes"], name="bounded ledger artifact_writes"
            ),
            durable_admission_cas_kind=_exact_text(
                mapping["durable_admission_cas_kind"],
                name="bounded ledger durable_admission_cas_kind",
            ),
            durable_admission_claims_persisted=_wire_int(
                mapping["durable_admission_claims_persisted"],
                name="bounded ledger durable_admission_claims_persisted",
            ),
            canonical_reducer_single_writer=_wire_bool(
                mapping["canonical_reducer_single_writer"],
                name="bounded ledger canonical_reducer_single_writer",
            ),
            durable_admission_cas_required=_wire_bool(
                mapping["durable_admission_cas_required"],
                name="bounded ledger durable_admission_cas_required",
            ),
            research_only=_wire_bool(
                mapping["research_only"], name="bounded ledger research_only"
            ),
            production_ready=_wire_bool(
                mapping["production_ready"], name="bounded ledger production_ready"
            ),
            execution_activated=_wire_bool(
                mapping["execution_activated"],
                name="bounded ledger execution_activated",
            ),
            schema_version=_exact_text(
                mapping["schema_version"], name="bounded ledger schema_version"
            ),
        )

    @classmethod
    def from_wire_bytes(cls, value: object) -> BoundedPlanningLedgerV1:
        return cls.from_mapping(_decode_canonical_wire(value, name="bounded ledger"))


@dataclass(frozen=True, slots=True)
class BoundedLaunchDecisionV1:
    reason: BoundedLaunchReasonV1 | str
    ledger: BoundedPlanningLedgerV1
    supervisor_capability_required: bool = True
    admission_tokens_consumed: int = 0
    runtime_writes: int = 0
    ledger_writes: int = 0
    artifact_writes: int = 0
    durable_admission_cas_required: bool = True
    durable_admission_cas_verified: bool = False
    research_only: bool = True
    production_ready: bool = False
    execution_activated: bool = False
    schema_version: str = "bounded-launch-decision/v1"

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != "bounded-launch-decision/v1"
        ):
            raise ValueError("unsupported bounded launch decision")
        object.__setattr__(self, "reason", BoundedLaunchReasonV1(self.reason))
        if type(self.ledger) is not BoundedPlanningLedgerV1:
            raise TypeError("bounded launch decision requires an exact ledger")
        fixed = {
            "supervisor_capability_required": True,
            "admission_tokens_consumed": 0,
            "runtime_writes": 0,
            "ledger_writes": 0,
            "artifact_writes": 0,
            "durable_admission_cas_required": True,
            "durable_admission_cas_verified": False,
            "research_only": True,
            "production_ready": False,
            "execution_activated": False,
        }
        if any(
            type(getattr(self, name)) is not type(expected)
            or getattr(self, name) != expected
            for name, expected in fixed.items()
        ):
            raise ValueError("bounded launch decision cannot claim execution or writes")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "reason": cast(BoundedLaunchReasonV1, self.reason).value,
            "ledger": self.ledger.to_dict(),
            "supervisor_capability_required": self.supervisor_capability_required,
            "admission_tokens_consumed": self.admission_tokens_consumed,
            "runtime_writes": self.runtime_writes,
            "ledger_writes": self.ledger_writes,
            "artifact_writes": self.artifact_writes,
            "durable_admission_cas_required": self.durable_admission_cas_required,
            "durable_admission_cas_verified": self.durable_admission_cas_verified,
            "research_only": self.research_only,
            "production_ready": self.production_ready,
            "execution_activated": self.execution_activated,
            "schema_version": self.schema_version,
        }

    def to_wire_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self.to_dict())) + b"\n"

    @classmethod
    def from_mapping(cls, value: object) -> BoundedLaunchDecisionV1:
        mapping = _exact_object(
            value,
            expected_keys=frozenset(
                {
                    "reason",
                    "ledger",
                    "supervisor_capability_required",
                    "admission_tokens_consumed",
                    "runtime_writes",
                    "ledger_writes",
                    "artifact_writes",
                    "durable_admission_cas_required",
                    "durable_admission_cas_verified",
                    "research_only",
                    "production_ready",
                    "execution_activated",
                    "schema_version",
                }
            ),
            name="bounded launch decision",
        )
        return cls(
            reason=_exact_text(
                mapping["reason"], name="bounded launch decision reason"
            ),
            ledger=BoundedPlanningLedgerV1.from_mapping(mapping["ledger"]),
            supervisor_capability_required=_wire_bool(
                mapping["supervisor_capability_required"],
                name="bounded decision supervisor_capability_required",
            ),
            admission_tokens_consumed=_wire_int(
                mapping["admission_tokens_consumed"],
                name="bounded decision admission_tokens_consumed",
            ),
            runtime_writes=_wire_int(
                mapping["runtime_writes"], name="bounded decision runtime_writes"
            ),
            ledger_writes=_wire_int(
                mapping["ledger_writes"], name="bounded decision ledger_writes"
            ),
            artifact_writes=_wire_int(
                mapping["artifact_writes"], name="bounded decision artifact_writes"
            ),
            durable_admission_cas_required=_wire_bool(
                mapping["durable_admission_cas_required"],
                name="bounded decision durable_admission_cas_required",
            ),
            durable_admission_cas_verified=_wire_bool(
                mapping["durable_admission_cas_verified"],
                name="bounded decision durable_admission_cas_verified",
            ),
            research_only=_wire_bool(
                mapping["research_only"], name="bounded decision research_only"
            ),
            production_ready=_wire_bool(
                mapping["production_ready"], name="bounded decision production_ready"
            ),
            execution_activated=_wire_bool(
                mapping["execution_activated"],
                name="bounded decision execution_activated",
            ),
            schema_version=_exact_text(
                mapping["schema_version"], name="bounded decision schema_version"
            ),
        )

    @classmethod
    def from_wire_bytes(cls, value: object) -> BoundedLaunchDecisionV1:
        return cls.from_mapping(_decode_canonical_wire(value, name="bounded decision"))


class BoundedResearchLauncherV1:
    """Pure two-worker planner which always fails closed before process launch."""

    def __init__(self, policy: AdaptiveDispatchPolicyV1 | None = None) -> None:
        selected = AdaptiveDispatchPolicyV1() if policy is None else policy
        if type(selected) is not AdaptiveDispatchPolicyV1:
            raise TypeError("bounded launcher requires an exact adaptive V1 policy")
        self.policy = selected

    def plan(
        self,
        *,
        experiment_spec_hash: str,
        tasks: tuple[BoundedResearchTaskV1, ...],
        allowed_entrypoints: tuple[str, ...],
    ) -> BoundedResearchPlanV1:
        return BoundedResearchPlanV1(
            experiment_spec_hash=experiment_spec_hash,
            tasks=tasks,
            allowed_entrypoints=allowed_entrypoints,
            policy_hash=self.policy.content_hash,
        )

    def preview(
        self,
        plan: BoundedResearchPlanV1,
        observations: tuple[AdaptiveResourceObservationAuthorityV1, ...],
        *,
        start_queue_ordinal: int = 0,
    ) -> BoundedPlanningLedgerV1:
        if type(plan) is not BoundedResearchPlanV1:
            raise TypeError("bounded launcher preview requires an exact plan")
        if plan.policy_hash != self.policy.content_hash:
            raise ValueError("bounded launcher policy binding differs")
        if type(observations) is not tuple or any(
            type(item) is not AdaptiveResourceObservationAuthorityV1
            for item in observations
        ):
            raise TypeError("bounded launcher observations must be an exact tuple")
        if type(start_queue_ordinal) is not int or not 0 <= start_queue_ordinal < len(
            plan.tasks
        ):
            raise ValueError("bounded launcher queue window start is invalid")
        window = plan.tasks[start_queue_ordinal : start_queue_ordinal + MAXIMUM_WORKERS]
        if len(observations) > len(window):
            raise ValueError("bounded launcher received unused resource observations")
        if observations:
            first_system = observations[0].system_observation
            root_identity = (
                first_system.root_pid,
                first_system.root_create_time_nanoseconds,
            )
            prior_system = first_system
            for authority in observations[1:]:
                current_system = authority.system_observation
                if (
                    current_system.root_pid,
                    current_system.root_create_time_nanoseconds,
                ) != root_identity:
                    raise ValueError(
                        "bounded launcher observation root identity differs"
                    )
                if (
                    current_system.sample_ordinal <= prior_system.sample_ordinal
                    or current_system.observed_monotonic_nanoseconds
                    <= prior_system.observed_monotonic_nanoseconds
                    or current_system.observed_at_utc <= prior_system.observed_at_utc
                ):
                    raise ValueError(
                        "bounded launcher observation sequence is not strictly ordered"
                    )
                prior_system = current_system
        state = AdaptiveDispatchStateV1()
        intents: list[BoundedAdmissionIntentV1] = []
        evaluation_hashes: list[str] = []
        hold_reason: str | None = None
        for index, task in enumerate(window):
            if index >= len(observations):
                hold_reason = "resource_observation_missing"
                break
            authority = observations[index]
            observation = authority.adaptive_observation
            starting = len(intents)
            request = AdaptiveDispatchRequestV1(
                experiment_spec_hash=plan.experiment_spec_hash,
                workload_hash=plan.workload_hash,
                active_workers=0,
                starting_workers=starting,
                pending_tasks=len(plan.tasks) - task.queue_ordinal,
                inflight_unrealized_memory_bytes=(starting * WORKER_HARD_RSS_BYTES),
                next_task_memory_limit_bytes=WORKER_HARD_RSS_BYTES,
                resource_budget_maximum_parallel_tasks=MAXIMUM_WORKERS,
                dispatch_stopped=False,
            )
            evaluation = evaluate_adaptive_dispatch_v1(
                self.policy,
                state,
                observation,
                request,
            )
            evaluation_hashes.append(evaluation.content_hash)
            state = evaluation.next_state
            if not evaluation.admit_one:
                hold_reason = evaluation.reason
                break
            token = evaluation.admission_decision_token
            if token is None:  # pragma: no cover - adaptive contract enforces this
                raise RuntimeError("adaptive admission omitted its decision token")
            bounded_token = hash_json(
                {
                    "schema_version": "bounded-admission-token/v1",
                    "queue_ordinal": task.queue_ordinal,
                    "task_hash": task.content_hash,
                    "natural_key": task.natural_key,
                    "candidate_lineage_hash": task.candidate_lineage_hash,
                    "source_capsule_hash": task.source_capsule_hash,
                    "sealed_process_task_hash": task.sealed_process_task_hash,
                    "attempt_number": 1,
                    "policy_admission_token": token,
                    "resource_observation_authority_hash": authority.content_hash,
                    "worker_hard_rss_bytes": WORKER_HARD_RSS_BYTES,
                    "numeric_thread_limit": NUMERIC_THREAD_LIMIT,
                    "process_nice": PROCESS_NICE,
                }
            )
            intents.append(
                BoundedAdmissionIntentV1(
                    queue_ordinal=task.queue_ordinal,
                    task_hash=task.content_hash,
                    natural_key=task.natural_key,
                    candidate_lineage_hash=task.candidate_lineage_hash,
                    source_capsule_hash=task.source_capsule_hash,
                    sealed_process_task_hash=task.sealed_process_task_hash,
                    attempt_number=1,
                    policy_admission_token=token,
                    resource_observation_authority_hash=authority.content_hash,
                    bounded_admission_token=bounded_token,
                )
            )
        return BoundedPlanningLedgerV1(
            plan_hash=plan.content_hash,
            workload_hash=plan.workload_hash,
            evaluation_hashes=tuple(evaluation_hashes),
            admission_intents=tuple(intents),
            hold_reason=hold_reason,
        )

    def launch(
        self,
        plan: BoundedResearchPlanV1,
        observations: tuple[AdaptiveResourceObservationAuthorityV1, ...],
        *,
        supervisor_capability: BirthPinnedSupervisorCapabilityV1 | None = None,
        legacy_executor: IsolatedProcessExecutor | None = None,
        runtime: ExperimentRuntime | None = None,
    ) -> BoundedLaunchDecisionV1:
        ledger = self.preview(plan, observations)
        if legacy_executor is not None or runtime is not None:
            reason = BoundedLaunchReasonV1.UNSAFE_LEGACY_BACKEND_REJECTED
        elif supervisor_capability is None:
            reason = BoundedLaunchReasonV1.SUPERVISOR_CAPABILITY_UNAVAILABLE
        else:
            # Structural protocol conformance is forgeable in Python.  V1 has
            # no authenticated capability issuer, so every public object is
            # rejected without calling it.
            reason = BoundedLaunchReasonV1.UNTRUSTED_SUPERVISOR_CAPABILITY
        return BoundedLaunchDecisionV1(reason=reason, ledger=ledger)


__all__ = [
    "BirthPinnedSupervisorCapabilityV1",
    "BoundedAdmissionIntentV1",
    "BoundedLaunchDecisionV1",
    "BoundedLaunchReasonV1",
    "BoundedPlanningLedgerV1",
    "BoundedResearchLauncherV1",
    "BoundedResearchPlanV1",
    "BoundedResearchTaskV1",
    "DURABLE_ADMISSION_CAS_KIND",
    "FOREGROUND_RESERVE_BYTES",
    "MAXIMUM_ATTEMPTS_PER_TASK",
    "MAXIMUM_WORKERS",
    "NUMERIC_THREAD_LIMIT",
    "PROCESS_NICE",
    "PROJECT_HARD_RSS_BYTES",
    "ResearchTaskSourceV1",
    "WORKER_HARD_RSS_BYTES",
]
