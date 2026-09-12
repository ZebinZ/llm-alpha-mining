"""No-spawn qualification for a future birth-pinned research supervisor.

The current bounded launcher has neither an authenticated supervisor issuer nor
a durable admission-CAS adapter, while its legacy executor cannot prove
PID-birth-pinned process-tree cleanup.  V1 therefore authenticates prerequisites
but never spawns and never presents its receipt as execution authority.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Final, Literal, NoReturn, cast

from alpha_research.core.hashing import hash_json
from alpha_research.core.python_runtime_capsule import (
    PythonRuntimeCapsuleError,
    PythonRuntimeCapsuleV1,
    load_python_runtime_capsule_v1,
)
from alpha_research.orchestration.bounded_research_launcher import (
    BoundedResearchPlanV1,
    BoundedResearchTaskV1,
    FOREGROUND_RESERVE_BYTES,
    MAXIMUM_WORKERS,
    NUMERIC_THREAD_LIMIT,
    PROCESS_NICE,
    PROJECT_HARD_RSS_BYTES,
    ResearchTaskSourceV1,
    WORKER_HARD_RSS_BYTES,
)
from alpha_research.orchestration.process import ProcessTask


RESEARCH_PROCESS_SUPERVISOR_V1_ACTIVATION_STATE: Final[str] = (
    "INACTIVE_QUALIFICATION_ONLY"
)
RESEARCH_PROCESS_SUPERVISOR_V1_ACTIVATION_BLOCKER: Final[str] = (
    "AUTHENTICATED_BIRTH_PINNED_SUPERVISOR_AND_DURABLE_ADMISSION_CAS_REQUIRED"
)

_ENTRYPOINT = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:"
    r"[A-Za-z_][A-Za-z0-9_]*$"
)
_MAX_JSON_DEPTH: Final[int] = 64
_MAX_JSON_NODES: Final[int] = 100_000
_LOCAL_PATH_TYPE: Final[type[Path]] = type(Path())
_RECEIPT_SCHEMA: Final[str] = "research-process-supervisor-qualification/v1"
_CONTRACT_SCHEMA: Final[str] = "research-process-supervision-contract/v1"
_ZERO_USAGE: Final[tuple[tuple[str, int], ...]] = (
    ("processes_spawned", 0),
    ("term_signals_sent", 0),
    ("kill_signals_sent", 0),
    ("raw_reads", 0),
    ("network_calls", 0),
    ("runtime_writes", 0),
    ("ledger_writes", 0),
    ("artifact_writes", 0),
)


class ResearchProcessSupervisorV1Error(RuntimeError):
    """Stable fail-closed qualification or activation error."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}:{self.detail}")


@dataclass(frozen=True, slots=True)
class ResearchProcessEntrypointPinV1:
    """Exact logical-entrypoint to capsule-script binding."""

    process_entrypoint: str
    capsule_entrypoint: str

    def __post_init__(self) -> None:
        if (
            type(self.process_entrypoint) is not str
            or _ENTRYPOINT.fullmatch(self.process_entrypoint) is None
        ):
            raise ValueError("process entrypoint pin is invalid")
        _relative_python_path(self.capsule_entrypoint)

    @property
    def content_hash(self) -> str:
        return cast(
            str,
            hash_json(
                {
                    "process_entrypoint": self.process_entrypoint,
                    "capsule_entrypoint": self.capsule_entrypoint,
                    "schema_version": "research-process-entrypoint-pin/v1",
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ResearchSourceCapsuleBindingV1:
    """Externally pinned capsule and its least-privilege entrypoint map."""

    source_capsule_hash: str
    manifest_path: Path
    manifest_document_sha256: str
    entrypoint_pins: tuple[ResearchProcessEntrypointPinV1, ...]

    def __post_init__(self) -> None:
        _sha256(self.source_capsule_hash, "source_capsule_hash")
        _sha256(self.manifest_document_sha256, "manifest_document_sha256")
        if type(self.manifest_path) is not _LOCAL_PATH_TYPE:
            raise TypeError("manifest_path must be an exact local pathlib Path")
        pins = self.entrypoint_pins
        if (
            type(pins) is not tuple
            or not pins
            or any(type(pin) is not ResearchProcessEntrypointPinV1 for pin in pins)
        ):
            raise TypeError("entrypoint pins must be a non-empty exact tuple")
        ordered = tuple(
            sorted(
                pins,
                key=lambda pin: (pin.process_entrypoint, pin.capsule_entrypoint),
            )
        )
        if pins != ordered:
            raise ValueError("entrypoint pins must be canonically sorted")
        if len({pin.process_entrypoint for pin in pins}) != len(pins):
            raise ValueError("logical entrypoint pins must be unique")
        if len({pin.capsule_entrypoint for pin in pins}) != len(pins):
            raise ValueError("capsule entrypoint pins must be unique")
        object.__setattr__(
            self,
            "manifest_path",
            Path(os.path.abspath(os.fspath(self.manifest_path.expanduser()))),
        )

    @property
    def content_hash(self) -> str:
        """Bind the local path without exposing it in the receipt."""

        return cast(
            str,
            hash_json(
                {
                    "source_capsule_hash": self.source_capsule_hash,
                    "manifest_path_sha256": hashlib.sha256(
                        os.fsencode(self.manifest_path)
                    ).hexdigest(),
                    "manifest_document_sha256": self.manifest_document_sha256,
                    "entrypoint_pin_hashes": [
                        pin.content_hash for pin in self.entrypoint_pins
                    ],
                    "research_only": True,
                    "production_ready": False,
                    "schema_version": "research-source-capsule-binding/v1",
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ResearchProcessSupervisorQualificationReceiptV1:
    """Immutable identity/status/hash/usage evidence; never spawn authority."""

    plan_hash: str
    supervision_contract_hash: str
    entrypoint_allowlist_hash: str
    capsule_binding_hashes: tuple[str, ...]
    task_hashes: tuple[str, ...]
    usage: tuple[tuple[str, int], ...]
    execution_activated: Literal[False] = False
    birth_identity_verified: Literal[False] = False
    cleanup_verified: Literal[False] = False
    human_gate_bypassed: Literal[False] = False
    raw_data_opened: Literal[False] = False
    production_ready: Literal[False] = False
    research_only: Literal[True] = True
    activation_state: str = RESEARCH_PROCESS_SUPERVISOR_V1_ACTIVATION_STATE
    activation_blocker: str = RESEARCH_PROCESS_SUPERVISOR_V1_ACTIVATION_BLOCKER
    schema_version: str = _RECEIPT_SCHEMA
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        self._validate()
        object.__setattr__(
            self,
            "receipt_hash",
            cast(str, hash_json(self._identity_payload())),
        )

    def _validate(self) -> None:
        for name in (
            "plan_hash",
            "supervision_contract_hash",
            "entrypoint_allowlist_hash",
        ):
            _sha256(getattr(self, name), name)
        for name in ("capsule_binding_hashes", "task_hashes"):
            values = getattr(self, name)
            if type(values) is not tuple or not values:
                raise TypeError(f"{name} must be a non-empty exact tuple")
            for value in values:
                _sha256(value, name)
        if tuple(sorted(self.capsule_binding_hashes)) != self.capsule_binding_hashes:
            raise ValueError("capsule binding hashes must be sorted")
        expected_usage = (
            ("tasks_inspected", len(self.task_hashes)),
            ("capsules_authenticated", len(self.capsule_binding_hashes)),
            ("entrypoints_verified", _usage_value(self.usage, 2)),
            *_ZERO_USAGE,
        )
        if type(self.usage) is not tuple or self.usage != expected_usage:
            raise ValueError("qualification usage differs from no-spawn evidence")
        if (
            self.execution_activated is not False
            or self.birth_identity_verified is not False
            or self.cleanup_verified is not False
            or self.human_gate_bypassed is not False
            or self.raw_data_opened is not False
            or self.production_ready is not False
            or self.research_only is not True
            or self.activation_state != RESEARCH_PROCESS_SUPERVISOR_V1_ACTIVATION_STATE
            or self.activation_blocker
            != RESEARCH_PROCESS_SUPERVISOR_V1_ACTIVATION_BLOCKER
            or self.schema_version != _RECEIPT_SCHEMA
            or self.supervision_contract_hash != _supervision_contract_hash()
        ):
            raise ValueError("qualification assurance boundary differs")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "plan_hash": self.plan_hash,
            "supervision_contract_hash": self.supervision_contract_hash,
            "entrypoint_allowlist_hash": self.entrypoint_allowlist_hash,
            "capsule_binding_hashes": list(self.capsule_binding_hashes),
            "task_hashes": list(self.task_hashes),
            "usage": dict(self.usage),
            "execution_activated": self.execution_activated,
            "birth_identity_verified": self.birth_identity_verified,
            "cleanup_verified": self.cleanup_verified,
            "human_gate_bypassed": self.human_gate_bypassed,
            "raw_data_opened": self.raw_data_opened,
            "production_ready": self.production_ready,
            "research_only": self.research_only,
            "activation_state": self.activation_state,
            "activation_blocker": self.activation_blocker,
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        self._validate()
        payload = self._identity_payload()
        if hash_json(payload) != self.receipt_hash:
            raise ValueError("qualification receipt mutated after construction")
        return {"receipt_hash": self.receipt_hash, **payload}


class ResearchProcessSupervisorV1:
    """Authenticate prerequisites, then fail closed at activation."""

    __slots__ = ()

    def qualify(
        self,
        plan: BoundedResearchPlanV1,
        capsule_bindings: tuple[ResearchSourceCapsuleBindingV1, ...],
    ) -> ResearchProcessSupervisorQualificationReceiptV1:
        plan_hash, task_hashes = _revalidate_plan(plan)
        bindings = _capsule_bindings(capsule_bindings)
        required_capsules = tuple(
            sorted({task.source_capsule_hash for task in plan.tasks})
        )
        if (
            tuple(binding.source_capsule_hash for binding in bindings)
            != required_capsules
        ):
            _fail("source_capsule_binding_scope_mismatch", plan_hash)
        used_entrypoints = tuple(
            sorted({task.process_task.entrypoint for task in plan.tasks})
        )
        if used_entrypoints != plan.allowed_entrypoints:
            _fail("plan_entrypoint_allowlist_overbroad", plan_hash)

        for binding in bindings:
            capsule = _authenticate_capsule(binding)
            expected = tuple(
                sorted(
                    {
                        task.process_task.entrypoint
                        for task in plan.tasks
                        if task.source_capsule_hash == binding.source_capsule_hash
                    }
                )
            )
            if (
                tuple(pin.process_entrypoint for pin in binding.entrypoint_pins)
                != expected
            ):
                _fail(
                    "capsule_logical_entrypoint_scope_mismatch",
                    binding.source_capsule_hash,
                )
            capsule_entries = tuple(
                sorted(pin.capsule_entrypoint for pin in binding.entrypoint_pins)
            )
            if capsule_entries != capsule.entrypoints:
                _fail(
                    "capsule_entrypoint_allowlist_mismatch",
                    binding.source_capsule_hash,
                )

        allowlist_hash = cast(
            str,
            hash_json(
                {
                    "schema_version": "research-process-entrypoint-allowlist/v1",
                    "capsules": [
                        {
                            "source_capsule_hash": binding.source_capsule_hash,
                            "pin_hashes": [
                                pin.content_hash for pin in binding.entrypoint_pins
                            ],
                        }
                        for binding in bindings
                    ],
                }
            ),
        )
        entrypoint_count = sum(len(binding.entrypoint_pins) for binding in bindings)
        return ResearchProcessSupervisorQualificationReceiptV1(
            plan_hash=plan_hash,
            supervision_contract_hash=_supervision_contract_hash(),
            entrypoint_allowlist_hash=allowlist_hash,
            capsule_binding_hashes=tuple(
                sorted(binding.content_hash for binding in bindings)
            ),
            task_hashes=task_hashes,
            usage=(
                ("tasks_inspected", len(plan.tasks)),
                ("capsules_authenticated", len(bindings)),
                ("entrypoints_verified", entrypoint_count),
                *_ZERO_USAGE,
            ),
        )

    def execute(
        self,
        qualification: ResearchProcessSupervisorQualificationReceiptV1,
    ) -> NoReturn:
        if type(qualification) is not ResearchProcessSupervisorQualificationReceiptV1:
            raise TypeError("exact qualification receipt required")
        qualification.to_dict()
        _fail("research_process_execution_not_activated", qualification.receipt_hash)


def qualify_research_process_supervisor_v1(
    plan: BoundedResearchPlanV1,
    capsule_bindings: tuple[ResearchSourceCapsuleBindingV1, ...],
) -> ResearchProcessSupervisorQualificationReceiptV1:
    """Convenience entrypoint for the no-spawn qualification operation."""

    return ResearchProcessSupervisorV1().qualify(plan, capsule_bindings)


def research_process_supervision_contract_v1() -> dict[str, object]:
    """Contract a future real supervisor must prove before activation."""

    return {
        "schema_version": _CONTRACT_SCHEMA,
        "resource_envelope": {
            "maximum_workers": MAXIMUM_WORKERS,
            "worker_hard_rss_bytes": WORKER_HARD_RSS_BYTES,
            "project_hard_rss_bytes": PROJECT_HARD_RSS_BYTES,
            "foreground_reserve_bytes": FOREGROUND_RESERVE_BYTES,
            "numeric_thread_limit": NUMERIC_THREAD_LIMIT,
            "process_nice": PROCESS_NICE,
        },
        "task_boundary": {
            "accepted_sources": ["sealed", "synthetic"],
            "payload": "exact_builtin_plain_json",
            "sealed_task_hash_required": True,
            "source_capsule_and_entrypoint_pins_required": True,
        },
        "process_identity": {
            "leader_and_descendants": "pid_plus_create_time",
            "start_new_session": True,
            "independent_process_group": True,
            "pid_reuse": "identity_mismatch_fail_closed",
        },
        "fail_closed_events": [
            "timeout",
            "cancellation",
            "leader_exit_with_live_descendant",
            "process_observation_access_denied",
            "descendant_residual_after_cleanup",
        ],
        "bounded_cleanup": {
            "sequence": ["SIGTERM", "SIGKILL"],
            "term_grace_seconds_maximum": 2.0,
            "kill_grace_seconds_maximum": 2.0,
            "birth_identity_recheck_before_signal": True,
            "residual_process_is_failure": True,
        },
        "authority_boundary": {
            "raw_l2_or_now_access": False,
            "manual_gate_bypass": False,
            "qualification_is_spawn_authority": False,
            "research_only": True,
            "production_ready": False,
        },
    }


def _supervision_contract_hash() -> str:
    return cast(str, hash_json(research_process_supervision_contract_v1()))


def _revalidate_plan(
    plan: BoundedResearchPlanV1,
) -> tuple[str, tuple[str, ...]]:
    if type(plan) is not BoundedResearchPlanV1:
        raise TypeError("exact BoundedResearchPlanV1 required")
    for task in plan.tasks:
        _revalidate_task(task)
    try:
        rebuilt = BoundedResearchPlanV1.from_mapping(plan.to_dict())
    except (PermissionError, TypeError, ValueError) as error:
        raise ResearchProcessSupervisorV1Error(
            "bounded_plan_validation_failed", type(error).__name__
        ) from error
    if rebuilt != plan:
        _fail("bounded_plan_identity_changed", plan.schema_version)
    return plan.content_hash, tuple(task.content_hash for task in plan.tasks)


def _revalidate_task(task: BoundedResearchTaskV1) -> None:
    if type(task) is not BoundedResearchTaskV1:
        raise TypeError("exact BoundedResearchTaskV1 required")
    if type(task.source) is not ResearchTaskSourceV1 or task.source not in {
        ResearchTaskSourceV1.SEALED,
        ResearchTaskSourceV1.SYNTHETIC,
    }:
        _fail("research_task_source_forbidden", task.stage)
    process_task = task.process_task
    if type(process_task) is not ProcessTask:
        raise TypeError("exact ProcessTask required")
    payload = _plain_json_copy(process_task.payload)
    try:
        snapshot = ProcessTask(
            task_id=process_task.task_id,
            entrypoint=process_task.entrypoint,
            payload=payload,
            timeout_seconds=process_task.timeout_seconds,
            cpu_limit_seconds=process_task.cpu_limit_seconds,
            memory_limit_bytes=process_task.memory_limit_bytes,
            output_limit_bytes=process_task.output_limit_bytes,
            schema_version=process_task.schema_version,
        )
    except (TypeError, ValueError) as error:
        raise ResearchProcessSupervisorV1Error(
            "sealed_process_task_validation_failed", type(error).__name__
        ) from error
    if snapshot != process_task:
        _fail("sealed_process_task_identity_changed", task.sealed_process_task_hash)
    if snapshot.memory_limit_bytes != WORKER_HARD_RSS_BYTES:
        _fail("worker_hard_rss_policy_mismatch", task.sealed_process_task_hash)
    if snapshot.content_hash != task.sealed_process_task_hash:
        _fail("sealed_process_task_hash_mismatch", task.sealed_process_task_hash)


def _plain_json_copy(
    value: object,
    *,
    depth: int = 0,
    seen: set[int] | None = None,
    nodes: list[int] | None = None,
) -> object:
    if depth > _MAX_JSON_DEPTH:
        raise ValueError("task payload exceeds maximum JSON depth")
    active = set() if seen is None else seen
    count = [0] if nodes is None else nodes
    count[0] += 1
    if count[0] > _MAX_JSON_NODES:
        raise ValueError("task payload exceeds maximum JSON nodes")
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("task payload contains non-finite JSON")
        return value
    if type(value) not in {dict, list}:
        raise TypeError("task payload must be exact built-in plain JSON")
    identity = id(value)
    if identity in active:
        raise ValueError("task payload contains a cycle")
    active.add(identity)
    try:
        if type(value) is list:
            return [
                _plain_json_copy(item, depth=depth + 1, seen=active, nodes=count)
                for item in cast(list[object], value)
            ]
        mapping = cast(dict[object, object], value)
        if any(type(key) is not str for key in mapping):
            raise TypeError("task payload keys must be exact text")
        keys = cast(list[str], list(mapping))
        return {
            key: _plain_json_copy(
                mapping[key], depth=depth + 1, seen=active, nodes=count
            )
            for key in sorted(keys)
        }
    finally:
        active.remove(identity)


def _capsule_bindings(
    value: tuple[ResearchSourceCapsuleBindingV1, ...],
) -> tuple[ResearchSourceCapsuleBindingV1, ...]:
    if (
        type(value) is not tuple
        or not value
        or any(type(item) is not ResearchSourceCapsuleBindingV1 for item in value)
    ):
        raise TypeError("capsule bindings must be a non-empty exact tuple")
    if value != tuple(sorted(value, key=lambda item: item.source_capsule_hash)):
        raise ValueError("capsule bindings must be canonically sorted")
    if len({item.source_capsule_hash for item in value}) != len(value):
        raise ValueError("capsule bindings must be unique")
    return value


def _authenticate_capsule(
    binding: ResearchSourceCapsuleBindingV1,
) -> PythonRuntimeCapsuleV1:
    try:
        capsule = load_python_runtime_capsule_v1(
            binding.manifest_path,
            expected_capsule_id=binding.source_capsule_hash,
            expected_document_sha256=binding.manifest_document_sha256,
        )
    except (OSError, PythonRuntimeCapsuleError, TypeError, ValueError) as error:
        detail = (
            error.code
            if isinstance(error, PythonRuntimeCapsuleError)
            else type(error).__name__
        )
        raise ResearchProcessSupervisorV1Error(
            "source_capsule_authentication_failed", detail
        ) from error
    if type(capsule) is not PythonRuntimeCapsuleV1:
        _fail("source_capsule_type_invalid", binding.source_capsule_hash)
    if (
        capsule.capsule_id != binding.source_capsule_hash
        or capsule.document_sha256 != binding.manifest_document_sha256
        or capsule.research_only is not True
        or capsule.production_ready is not False
    ):
        _fail("source_capsule_identity_mismatch", binding.source_capsule_hash)
    return capsule


def _relative_python_path(value: object) -> str:
    if type(value) is not str or not value:
        raise TypeError("capsule entrypoint must be non-empty exact text")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or path.suffix != ".py"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("capsule entrypoint must be canonical relative Python")
    return value


def _usage_value(usage: object, index: int) -> int:
    if type(usage) is not tuple or len(usage) < 3:
        raise TypeError("qualification usage must be an exact tuple")
    item = cast(tuple[object, ...], usage)[index]
    if type(item) is not tuple or len(item) != 2:
        raise TypeError("qualification usage item must be an exact pair")
    key, value = cast(tuple[object, object], item)
    if key != "entrypoints_verified" or type(value) is not int or value < 0:
        raise ValueError("qualification entrypoint usage is invalid")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _fail(code: str, detail: str) -> NoReturn:
    raise ResearchProcessSupervisorV1Error(code, detail)


__all__ = [
    "RESEARCH_PROCESS_SUPERVISOR_V1_ACTIVATION_BLOCKER",
    "RESEARCH_PROCESS_SUPERVISOR_V1_ACTIVATION_STATE",
    "ResearchProcessEntrypointPinV1",
    "ResearchProcessSupervisorQualificationReceiptV1",
    "ResearchProcessSupervisorV1",
    "ResearchProcessSupervisorV1Error",
    "ResearchSourceCapsuleBindingV1",
    "qualify_research_process_supervisor_v1",
    "research_process_supervision_contract_v1",
]
