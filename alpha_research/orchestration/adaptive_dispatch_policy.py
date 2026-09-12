"""Pure, research-only adaptive dispatch policy for bounded G1 workers.

This module deliberately performs no process launch, psutil access, clock read,
sleep, filesystem I/O, raw-data access, or runtime mutation.  A future
single-writer launcher may feed it authenticated observations and consume each
content-addressed admission token at most once.  The returned next-state marks
that token consumed before any launch; exact recomputation only reproduces the
same token.  Soft pressure only pauses new work and never authorizes terminating
an atomic task already in flight.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Final, cast

from alpha_research.core.hashing import hash_json, require_sha256


_GIB: Final[int] = 1024 * 1024 * 1024
_POLICY_SCHEMA: Final[str] = "adaptive-dispatch-policy/v1"
_OBSERVATION_SCHEMA: Final[str] = "adaptive-resource-observation/v1"
_BENCHMARK_EVIDENCE_SCHEMA: Final[str] = "adaptive-fixed-workload-benchmark-evidence/v1"
_BENCHMARK_AUTHORITY_SCHEMA: Final[str] = (
    "adaptive-fixed-workload-benchmark-authority/v1"
)
_AUTHORITY_SCHEMA: Final[str] = "adaptive-worker4-authority/v1"
_REQUEST_SCHEMA: Final[str] = "adaptive-dispatch-request/v1"
_STATE_SCHEMA: Final[str] = "adaptive-dispatch-state/v1"
_EVALUATION_SCHEMA: Final[str] = "adaptive-dispatch-evaluation/v1"
_ADMISSION_DECISION_SCHEMA: Final[str] = "adaptive-admission-decision/v1"


class MemoryPressure(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class SwapPressure(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    UNKNOWN = "unknown"


class RawReaderState(str, Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    UNKNOWN = "unknown"


class AdaptiveDispatchAction(str, Enum):
    ADMIT_ONE = "admit_one"
    HOLD = "hold"
    HARD_STOP = "hard_stop"


_REASONS: Final[frozenset[str]] = frozenset(
    {
        "admitted",
        "admission_already_consumed",
        "cpu_pressure",
        "critical_memory_pressure",
        "dispatch_stopped",
        "incomplete_observation",
        "memory_reserve_insufficient",
        "memory_pressure",
        "no_pending_tasks",
        "observation_interval_too_short",
        "observation_replay_mismatch",
        "project_capacity_insufficient",
        "project_hard_rss",
        "project_warning_rss",
        "raw_reader_not_inactive",
        "stale_observation",
        "state_binding_mismatch",
        "swap_pressure",
        "worker4_authority_invalid",
        "worker_ceiling_reached",
        "worker_hard_rss",
        "worker_warning_rss",
    }
)
_HARD_REASONS: Final[frozenset[str]] = frozenset(
    {
        "critical_memory_pressure",
        "dispatch_stopped",
        "project_hard_rss",
        "worker_hard_rss",
    }
)
_SOFT_PAUSE_REASONS: Final[frozenset[str]] = (
    _REASONS
    - _HARD_REASONS
    - {
        "admitted",
        "admission_already_consumed",
        "no_pending_tasks",
        "worker_ceiling_reached",
    }
)


def _exact_fields(
    value: Mapping[str, object], expected: frozenset[str], *, name: str
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    observed = frozenset(value)
    if observed != expected:
        raise ValueError(
            f"{name} fields differ; missing={sorted(expected - observed)!r}; "
            f"extra={sorted(observed - expected)!r}"
        )


def _exact_int(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be an integer <= {maximum}")
    return value


def _optional_exact_int(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int | None:
    if value is None:
        return None
    return _exact_int(value, name=name, minimum=minimum, maximum=maximum)


def _exact_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be a lowercase sha256 digest")
    return cast(str, require_sha256(value, name=name))


def _optional_digest(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _digest(value, name=name)


@dataclass(frozen=True, slots=True)
class AdaptiveDispatchPolicyV1:
    """Frozen V1 resource policy; changing a limit requires a new schema."""

    baseline_worker_limit: int = 2
    balanced_worker_limit: int = 3
    activated_worker_limit: int = 4
    low_pressure_windows_required: int = 3
    scale_observation_interval_seconds: int = 15
    worker_warning_rss_bytes: int = 2 * _GIB
    worker_hard_rss_bytes: int = 5 * _GIB // 2
    project_warning_rss_bytes: int = 10 * _GIB
    project_hard_rss_bytes: int = 12 * _GIB
    foreground_reserve_bytes: int = 4 * _GIB
    balanced_cpu_ceiling_basis_points: int = 7_000
    activated_cpu_ceiling_basis_points: int = 8_000
    inner_thread_limit: int = 1
    research_only: bool = True
    production_ready: bool = False
    schema_version: str = _POLICY_SCHEMA

    def __post_init__(self) -> None:
        expected = {
            "baseline_worker_limit": 2,
            "balanced_worker_limit": 3,
            "activated_worker_limit": 4,
            "low_pressure_windows_required": 3,
            "scale_observation_interval_seconds": 15,
            "worker_warning_rss_bytes": 2 * _GIB,
            "worker_hard_rss_bytes": 5 * _GIB // 2,
            "project_warning_rss_bytes": 10 * _GIB,
            "project_hard_rss_bytes": 12 * _GIB,
            "foreground_reserve_bytes": 4 * _GIB,
            "balanced_cpu_ceiling_basis_points": 7_000,
            "activated_cpu_ceiling_basis_points": 8_000,
            "inner_thread_limit": 1,
            "research_only": True,
            "production_ready": False,
            "schema_version": _POLICY_SCHEMA,
        }
        for name, required in expected.items():
            value = getattr(self, name)
            if type(value) is not type(required) or value != required:
                raise ValueError(f"adaptive dispatch V1 {name} differs")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AdaptiveDispatchPolicyV1":
        _exact_fields(value, frozenset(cls.__dataclass_fields__), name=cls.__name__)
        return cls(**dict(value))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class AdaptiveResourceObservationV1:
    """Caller-supplied resource facts; UNKNOWN is a first-class fail-closed state."""

    sample_ordinal: int
    observed_monotonic_nanoseconds: int
    metrics_complete: bool
    physical_memory_bytes: int | None
    available_memory_bytes: int | None
    project_tree_rss_bytes: int | None
    maximum_worker_tree_rss_bytes: int | None
    system_cpu_busy_basis_points: int | None
    memory_pressure: MemoryPressure | str
    swap_pressure: SwapPressure | str
    raw_reader_state: RawReaderState | str
    schema_version: str = _OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != _OBSERVATION_SCHEMA
        ):
            raise ValueError("unsupported adaptive resource observation")
        object.__setattr__(
            self,
            "sample_ordinal",
            _exact_int(self.sample_ordinal, name="sample_ordinal"),
        )
        object.__setattr__(
            self,
            "observed_monotonic_nanoseconds",
            _exact_int(
                self.observed_monotonic_nanoseconds,
                name="observed_monotonic_nanoseconds",
            ),
        )
        object.__setattr__(
            self,
            "metrics_complete",
            _exact_bool(self.metrics_complete, name="metrics_complete"),
        )
        for name in (
            "physical_memory_bytes",
            "available_memory_bytes",
            "project_tree_rss_bytes",
            "maximum_worker_tree_rss_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _optional_exact_int(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "system_cpu_busy_basis_points",
            _optional_exact_int(
                self.system_cpu_busy_basis_points,
                name="system_cpu_busy_basis_points",
                maximum=10_000,
            ),
        )
        try:
            object.__setattr__(
                self, "memory_pressure", MemoryPressure(self.memory_pressure)
            )
            object.__setattr__(self, "swap_pressure", SwapPressure(self.swap_pressure))
            object.__setattr__(
                self, "raw_reader_state", RawReaderState(self.raw_reader_state)
            )
        except ValueError as exc:
            raise ValueError("adaptive resource observation state is invalid") from exc
        if self.physical_memory_bytes == 0:
            raise ValueError("physical_memory_bytes must be positive when present")
        if (
            self.available_memory_bytes is not None
            and self.physical_memory_bytes is not None
            and self.available_memory_bytes > self.physical_memory_bytes
        ):
            raise ValueError("available memory cannot exceed physical memory")
        if self.metrics_complete and any(
            getattr(self, name) is None
            for name in (
                "physical_memory_bytes",
                "available_memory_bytes",
                "project_tree_rss_bytes",
                "maximum_worker_tree_rss_bytes",
                "system_cpu_busy_basis_points",
            )
        ):
            raise ValueError("complete resource observation has missing metrics")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_ordinal": self.sample_ordinal,
            "observed_monotonic_nanoseconds": self.observed_monotonic_nanoseconds,
            "metrics_complete": self.metrics_complete,
            "physical_memory_bytes": self.physical_memory_bytes,
            "available_memory_bytes": self.available_memory_bytes,
            "project_tree_rss_bytes": self.project_tree_rss_bytes,
            "maximum_worker_tree_rss_bytes": self.maximum_worker_tree_rss_bytes,
            "system_cpu_busy_basis_points": self.system_cpu_busy_basis_points,
            "memory_pressure": cast(MemoryPressure, self.memory_pressure).value,
            "swap_pressure": cast(SwapPressure, self.swap_pressure).value,
            "raw_reader_state": cast(RawReaderState, self.raw_reader_state).value,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "AdaptiveResourceObservationV1":
        _exact_fields(value, frozenset(cls.__dataclass_fields__), name=cls.__name__)
        return cls(**dict(value))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class AdaptiveFixedWorkloadBenchmarkEvidenceV1:
    """Canonical qualifying evidence for the frozen 1/2/4 workload matrix.

    The pure policy validates the complete canonical summary rather than treating
    an opaque receipt digest as proof.  Authentication of the enclosing authority
    remains an explicit upstream trust boundary; this object grants no execution
    capability by itself.
    """

    workload_hash: str
    policy_hash: str
    benchmark_spec_hash: str
    run_manifest_hash: str
    one_worker_result_hash: str
    two_worker_result_hash: str
    four_worker_result_hash: str
    one_worker_cold_runs: int
    one_worker_warm_runs: int
    two_worker_cold_runs: int
    two_worker_warm_runs: int
    four_worker_cold_runs: int
    four_worker_warm_runs: int
    one_to_two_speedup_basis_points: int
    two_to_four_speedup_basis_points: int
    two_worker_parallel_efficiency_basis_points: int
    four_worker_parallel_efficiency_basis_points: int
    peak_project_rss_bytes: int
    failure_count: int
    retry_count: int
    swap_pressure_observed: bool
    qualification_passed: bool
    research_only: bool = True
    production_ready: bool = False
    schema_version: str = _BENCHMARK_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != _BENCHMARK_EVIDENCE_SCHEMA
        ):
            raise ValueError("unsupported adaptive fixed-workload benchmark evidence")
        for name in (
            "workload_hash",
            "policy_hash",
            "benchmark_spec_hash",
            "run_manifest_hash",
            "one_worker_result_hash",
            "two_worker_result_hash",
            "four_worker_result_hash",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if not (
            self.one_worker_result_hash
            == self.two_worker_result_hash
            == self.four_worker_result_hash
        ):
            raise ValueError("fixed-workload benchmark result identities differ")
        for name in (
            "one_worker_cold_runs",
            "one_worker_warm_runs",
            "two_worker_cold_runs",
            "two_worker_warm_runs",
            "four_worker_cold_runs",
            "four_worker_warm_runs",
        ):
            object.__setattr__(
                self,
                name,
                _exact_int(getattr(self, name), name=name, minimum=3),
            )
        thresholds = {
            "one_to_two_speedup_basis_points": 16_000,
            "two_to_four_speedup_basis_points": 12_500,
            "two_worker_parallel_efficiency_basis_points": 6_500,
            "four_worker_parallel_efficiency_basis_points": 5_500,
        }
        for name, minimum in thresholds.items():
            object.__setattr__(
                self,
                name,
                _exact_int(getattr(self, name), name=name, minimum=minimum),
            )
        object.__setattr__(
            self,
            "peak_project_rss_bytes",
            _exact_int(
                self.peak_project_rss_bytes,
                name="peak_project_rss_bytes",
                minimum=1,
                maximum=12 * _GIB,
            ),
        )
        for name in ("failure_count", "retry_count"):
            object.__setattr__(
                self,
                name,
                _exact_int(getattr(self, name), name=name),
            )
            if getattr(self, name) != 0:
                raise ValueError(
                    f"qualifying fixed-workload benchmark {name} must be zero"
                )
        object.__setattr__(
            self,
            "swap_pressure_observed",
            _exact_bool(
                self.swap_pressure_observed,
                name="swap_pressure_observed",
            ),
        )
        object.__setattr__(
            self,
            "qualification_passed",
            _exact_bool(self.qualification_passed, name="qualification_passed"),
        )
        if self.swap_pressure_observed or not self.qualification_passed:
            raise ValueError("fixed-workload benchmark did not qualify worker four")
        if self.research_only is not True or self.production_ready is not False:
            raise ValueError("benchmark evidence must remain research-only")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "AdaptiveFixedWorkloadBenchmarkEvidenceV1":
        _exact_fields(value, frozenset(cls.__dataclass_fields__), name=cls.__name__)
        return cls(**dict(value))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class AdaptiveFixedWorkloadBenchmarkAuthorityV1:
    """Upstream-authenticated authority over one canonical benchmark evidence."""

    evidence: AdaptiveFixedWorkloadBenchmarkEvidenceV1
    evidence_hash: str
    verifier_profile_hash: str
    verification_receipt_hash: str
    authorized_worker_count: int = 4
    research_only: bool = True
    production_ready: bool = False
    schema_version: str = _BENCHMARK_AUTHORITY_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != _BENCHMARK_AUTHORITY_SCHEMA
        ):
            raise ValueError("unsupported adaptive benchmark authority")
        if type(self.evidence) is not AdaptiveFixedWorkloadBenchmarkEvidenceV1:
            raise TypeError("benchmark authority requires exact canonical evidence")
        for name in (
            "evidence_hash",
            "verifier_profile_hash",
            "verification_receipt_hash",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if self.evidence_hash != self.evidence.content_hash:
            raise ValueError("benchmark authority evidence binding differs")
        if (
            type(self.authorized_worker_count) is not int
            or self.authorized_worker_count != 4
        ):
            raise ValueError("benchmark authority worker count differs")
        if self.research_only is not True or self.production_ready is not False:
            raise ValueError("benchmark authority must remain research-only")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence": self.evidence.to_dict(),
            "evidence_hash": self.evidence_hash,
            "verifier_profile_hash": self.verifier_profile_hash,
            "verification_receipt_hash": self.verification_receipt_hash,
            "authorized_worker_count": self.authorized_worker_count,
            "research_only": self.research_only,
            "production_ready": self.production_ready,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "AdaptiveFixedWorkloadBenchmarkAuthorityV1":
        _exact_fields(value, frozenset(cls.__dataclass_fields__), name=cls.__name__)
        raw_evidence = value["evidence"]
        if not isinstance(raw_evidence, Mapping):
            raise TypeError("benchmark authority evidence must be an object")
        payload = dict(value)
        payload["evidence"] = AdaptiveFixedWorkloadBenchmarkEvidenceV1.from_mapping(
            raw_evidence
        )
        return cls(**payload)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class AdaptiveWorker4AuthorityV1:
    """Bounded activation bound to a separate canonical benchmark authority.

    This record cannot prove its own benchmark claim.  Evaluation therefore also
    requires the independently supplied, upstream-authenticated benchmark authority
    whose canonical evidence hashes are bound below.
    """

    benchmark_workload_hash: str
    activation_workload_hash: str
    policy_hash: str
    benchmark_authority_hash: str
    benchmark_evidence_hash: str
    activation_receipt_hash: str
    activation_first_sample_ordinal: int
    activation_last_sample_ordinal: int
    benchmark_worker_count: int = 4
    activated_worker_count: int = 4
    research_only: bool = True
    production_ready: bool = False
    schema_version: str = _AUTHORITY_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != _AUTHORITY_SCHEMA
        ):
            raise ValueError("unsupported adaptive worker-4 authority")
        for name in (
            "benchmark_workload_hash",
            "activation_workload_hash",
            "policy_hash",
            "benchmark_authority_hash",
            "benchmark_evidence_hash",
            "activation_receipt_hash",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if self.benchmark_workload_hash != self.activation_workload_hash:
            raise ValueError("worker-4 benchmark and activation workloads differ")
        for name in (
            "activation_first_sample_ordinal",
            "activation_last_sample_ordinal",
        ):
            object.__setattr__(
                self,
                name,
                _exact_int(getattr(self, name), name=name),
            )
        if self.activation_last_sample_ordinal < self.activation_first_sample_ordinal:
            raise ValueError("worker-4 activation ordinal window is invalid")
        if (
            type(self.benchmark_worker_count) is not int
            or self.benchmark_worker_count != 4
        ):
            raise ValueError("worker-4 benchmark count differs")
        if (
            type(self.activated_worker_count) is not int
            or self.activated_worker_count != 4
        ):
            raise ValueError("worker-4 activation count differs")
        if self.research_only is not True or self.production_ready is not False:
            raise ValueError("worker-4 authority must remain research-only")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AdaptiveWorker4AuthorityV1":
        _exact_fields(value, frozenset(cls.__dataclass_fields__), name=cls.__name__)
        return cls(**dict(value))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class AdaptiveDispatchRequestV1:
    """One serialized request to consider one additional G1 worker."""

    experiment_spec_hash: str
    workload_hash: str
    active_workers: int
    starting_workers: int
    pending_tasks: int
    inflight_unrealized_memory_bytes: int
    next_task_memory_limit_bytes: int
    resource_budget_maximum_parallel_tasks: int
    dispatch_stopped: bool
    worker4_authority: AdaptiveWorker4AuthorityV1 | None = None
    schema_version: str = _REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != _REQUEST_SCHEMA
        ):
            raise ValueError("unsupported adaptive dispatch request")
        for name in ("experiment_spec_hash", "workload_hash"):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        for name in (
            "active_workers",
            "starting_workers",
            "pending_tasks",
            "inflight_unrealized_memory_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _exact_int(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "next_task_memory_limit_bytes",
            _exact_int(
                self.next_task_memory_limit_bytes,
                name="next_task_memory_limit_bytes",
                minimum=1,
                maximum=5 * _GIB // 2,
            ),
        )
        object.__setattr__(
            self,
            "resource_budget_maximum_parallel_tasks",
            _exact_int(
                self.resource_budget_maximum_parallel_tasks,
                name="resource_budget_maximum_parallel_tasks",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "dispatch_stopped",
            _exact_bool(self.dispatch_stopped, name="dispatch_stopped"),
        )
        if self.active_workers + self.starting_workers > 4:
            raise ValueError("adaptive dispatch cannot govern more than four workers")
        maximum_unrealized = (self.active_workers + self.starting_workers) * (
            5 * _GIB // 2
        )
        if self.inflight_unrealized_memory_bytes > maximum_unrealized:
            raise ValueError("in-flight memory commitment exceeds worker hard limits")
        minimum_starting_commitment = (
            self.starting_workers * self.next_task_memory_limit_bytes
        )
        if self.inflight_unrealized_memory_bytes < minimum_starting_commitment:
            raise ValueError(
                "in-flight memory commitment does not cover starting workers"
            )
        if (
            self.worker4_authority is not None
            and type(self.worker4_authority) is not AdaptiveWorker4AuthorityV1
        ):
            raise TypeError("worker4_authority must be an exact V1 authority")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_spec_hash": self.experiment_spec_hash,
            "workload_hash": self.workload_hash,
            "active_workers": self.active_workers,
            "starting_workers": self.starting_workers,
            "pending_tasks": self.pending_tasks,
            "inflight_unrealized_memory_bytes": (self.inflight_unrealized_memory_bytes),
            "next_task_memory_limit_bytes": self.next_task_memory_limit_bytes,
            "resource_budget_maximum_parallel_tasks": (
                self.resource_budget_maximum_parallel_tasks
            ),
            "dispatch_stopped": self.dispatch_stopped,
            "worker4_authority": (
                None
                if self.worker4_authority is None
                else self.worker4_authority.to_dict()
            ),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AdaptiveDispatchRequestV1":
        _exact_fields(value, frozenset(cls.__dataclass_fields__), name=cls.__name__)
        raw_authority = value["worker4_authority"]
        if raw_authority is not None and not isinstance(raw_authority, Mapping):
            raise TypeError("worker4_authority must be an object or null")
        payload = dict(value)
        payload["worker4_authority"] = (
            None
            if raw_authority is None
            else AdaptiveWorker4AuthorityV1.from_mapping(raw_authority)
        )
        return cls(**payload)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class AdaptiveDispatchStateV1:
    """Replay-safe hysteresis state owned by one single-writer scheduler."""

    experiment_spec_hash: str | None = None
    workload_hash: str | None = None
    last_observation_ordinal: int | None = None
    last_observation_monotonic_nanoseconds: int | None = None
    last_observation_hash: str | None = None
    consumed_admission_decision_token: str | None = None
    adaptive_worker_ceiling: int = 2
    low_pressure_windows: int = 0
    schema_version: str = _STATE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != _STATE_SCHEMA:
            raise ValueError("unsupported adaptive dispatch state")
        object.__setattr__(
            self,
            "experiment_spec_hash",
            _optional_digest(self.experiment_spec_hash, name="experiment_spec_hash"),
        )
        object.__setattr__(
            self,
            "workload_hash",
            _optional_digest(self.workload_hash, name="workload_hash"),
        )
        object.__setattr__(
            self,
            "last_observation_ordinal",
            _optional_exact_int(
                self.last_observation_ordinal,
                name="last_observation_ordinal",
            ),
        )
        object.__setattr__(
            self,
            "last_observation_monotonic_nanoseconds",
            _optional_exact_int(
                self.last_observation_monotonic_nanoseconds,
                name="last_observation_monotonic_nanoseconds",
            ),
        )
        object.__setattr__(
            self,
            "last_observation_hash",
            _optional_digest(
                self.last_observation_hash,
                name="last_observation_hash",
            ),
        )
        object.__setattr__(
            self,
            "consumed_admission_decision_token",
            _optional_digest(
                self.consumed_admission_decision_token,
                name="consumed_admission_decision_token",
            ),
        )
        object.__setattr__(
            self,
            "adaptive_worker_ceiling",
            _exact_int(
                self.adaptive_worker_ceiling,
                name="adaptive_worker_ceiling",
                minimum=2,
                maximum=4,
            ),
        )
        object.__setattr__(
            self,
            "low_pressure_windows",
            _exact_int(
                self.low_pressure_windows,
                name="low_pressure_windows",
                maximum=2,
            ),
        )
        empty = self.experiment_spec_hash is None
        if (
            empty != (self.workload_hash is None)
            or empty != (self.last_observation_ordinal is None)
            or empty != (self.last_observation_monotonic_nanoseconds is None)
            or empty != (self.last_observation_hash is None)
        ):
            raise ValueError("adaptive dispatch state binding is incomplete")
        if empty and (
            self.consumed_admission_decision_token is not None
            or self.adaptive_worker_ceiling != 2
            or self.low_pressure_windows != 0
        ):
            raise ValueError("unbound adaptive dispatch state must be initial")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AdaptiveDispatchStateV1":
        _exact_fields(value, frozenset(cls.__dataclass_fields__), name=cls.__name__)
        return cls(**dict(value))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class AdaptiveDispatchEvaluationV1:
    """Content-addressed decision; it carries no authority to start a process."""

    action: AdaptiveDispatchAction | str
    reason: str
    admit_one: bool
    effective_worker_ceiling: int
    soft_pause: bool
    hard_stop_required: bool
    terminate_running_workers: bool
    projected_project_rss_bytes: int | None
    projected_available_memory_bytes: int | None
    policy_hash: str
    observation_hash: str
    request_hash: str
    prior_state_hash: str
    admission_decision_token: str | None
    next_state: AdaptiveDispatchStateV1
    research_only: bool = True
    production_ready: bool = False
    schema_version: str = _EVALUATION_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != _EVALUATION_SCHEMA
        ):
            raise ValueError("unsupported adaptive dispatch evaluation")
        try:
            object.__setattr__(self, "action", AdaptiveDispatchAction(self.action))
        except ValueError as exc:
            raise ValueError("adaptive dispatch action is invalid") from exc
        if type(self.reason) is not str or self.reason not in _REASONS:
            raise ValueError("adaptive dispatch reason is invalid")
        for name in (
            "admit_one",
            "soft_pause",
            "hard_stop_required",
            "terminate_running_workers",
        ):
            object.__setattr__(
                self,
                name,
                _exact_bool(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "effective_worker_ceiling",
            _exact_int(
                self.effective_worker_ceiling,
                name="effective_worker_ceiling",
                maximum=4,
            ),
        )
        for name in (
            "projected_project_rss_bytes",
            "projected_available_memory_bytes",
        ):
            value = getattr(self, name)
            if value is not None and type(value) is not int:
                raise ValueError(f"{name} must be an integer or null")
        for name in (
            "policy_hash",
            "observation_hash",
            "request_hash",
            "prior_state_hash",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "admission_decision_token",
            _optional_digest(
                self.admission_decision_token,
                name="admission_decision_token",
            ),
        )
        if type(self.next_state) is not AdaptiveDispatchStateV1:
            raise TypeError("next_state must be an exact V1 state")
        if self.research_only is not True or self.production_ready is not False:
            raise ValueError("adaptive dispatch evaluation must remain research-only")
        admitted = self.action is AdaptiveDispatchAction.ADMIT_ONE
        hard = self.action is AdaptiveDispatchAction.HARD_STOP
        expected_admission_token = _admission_decision_token_from_hashes(
            policy_hash=self.policy_hash,
            observation_hash=self.observation_hash,
            request_hash=self.request_hash,
            prior_state_hash=self.prior_state_hash,
        )
        if (
            self.admit_one is not admitted
            or self.hard_stop_required is not hard
            or self.terminate_running_workers is not hard
            or (hard and self.effective_worker_ceiling != 0)
            or (admitted and self.soft_pause)
        ):
            raise ValueError("adaptive dispatch decision flags differ")
        if admitted:
            if (
                self.admission_decision_token != expected_admission_token
                or self.next_state.consumed_admission_decision_token
                != expected_admission_token
            ):
                raise ValueError("adaptive admission decision token differs")
        elif self.admission_decision_token is not None:
            raise ValueError("non-admission cannot carry an admission decision token")
        if (
            (admitted and self.reason != "admitted")
            or (hard and self.reason not in _HARD_REASONS)
            or (
                self.action is AdaptiveDispatchAction.HOLD
                and self.reason in _HARD_REASONS | {"admitted"}
            )
            or (self.soft_pause != (self.reason in _SOFT_PAUSE_REASONS))
        ):
            raise ValueError("adaptive dispatch action/reason differs")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "action": cast(AdaptiveDispatchAction, self.action).value,
            "reason": self.reason,
            "admit_one": self.admit_one,
            "effective_worker_ceiling": self.effective_worker_ceiling,
            "soft_pause": self.soft_pause,
            "hard_stop_required": self.hard_stop_required,
            "terminate_running_workers": self.terminate_running_workers,
            "projected_project_rss_bytes": self.projected_project_rss_bytes,
            "projected_available_memory_bytes": (self.projected_available_memory_bytes),
            "policy_hash": self.policy_hash,
            "observation_hash": self.observation_hash,
            "request_hash": self.request_hash,
            "prior_state_hash": self.prior_state_hash,
            "admission_decision_token": self.admission_decision_token,
            "next_state": self.next_state.to_dict(),
            "research_only": self.research_only,
            "production_ready": self.production_ready,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "AdaptiveDispatchEvaluationV1":
        _exact_fields(value, frozenset(cls.__dataclass_fields__), name=cls.__name__)
        raw_state = value["next_state"]
        if not isinstance(raw_state, Mapping):
            raise TypeError("adaptive dispatch next_state must be an object")
        payload = dict(value)
        payload["next_state"] = AdaptiveDispatchStateV1.from_mapping(raw_state)
        return cls(**payload)  # type: ignore[arg-type]


def evaluate_adaptive_dispatch_v1(
    policy: AdaptiveDispatchPolicyV1,
    prior_state: AdaptiveDispatchStateV1,
    observation: AdaptiveResourceObservationV1,
    request: AdaptiveDispatchRequestV1,
    *,
    fixed_workload_benchmark_authority: (
        AdaptiveFixedWorkloadBenchmarkAuthorityV1 | None
    ) = None,
) -> AdaptiveDispatchEvaluationV1:
    """Return one deterministic admission decision and its replay state."""

    if type(policy) is not AdaptiveDispatchPolicyV1:
        raise TypeError("policy must be an exact AdaptiveDispatchPolicyV1")
    if type(prior_state) is not AdaptiveDispatchStateV1:
        raise TypeError("prior_state must be an exact AdaptiveDispatchStateV1")
    if type(observation) is not AdaptiveResourceObservationV1:
        raise TypeError("observation must be an exact AdaptiveResourceObservationV1")
    if type(request) is not AdaptiveDispatchRequestV1:
        raise TypeError("request must be an exact AdaptiveDispatchRequestV1")
    if (
        fixed_workload_benchmark_authority is not None
        and type(fixed_workload_benchmark_authority)
        is not AdaptiveFixedWorkloadBenchmarkAuthorityV1
    ):
        raise TypeError(
            "fixed_workload_benchmark_authority must be an exact V1 authority"
        )

    projected_rss, projected_available = _projected_memory(observation, request)

    if request.dispatch_stopped:
        return _evaluation(
            policy,
            prior_state,
            observation,
            request,
            action=AdaptiveDispatchAction.HARD_STOP,
            reason="dispatch_stopped",
            effective_ceiling=0,
            soft_pause=False,
            next_state=_reset_state(prior_state, observation, request),
            projected_rss=projected_rss,
            projected_available=projected_available,
        )

    replay_reason = _replay_reason(policy, prior_state, observation, request)
    if replay_reason is not None:
        return _evaluation(
            policy,
            prior_state,
            observation,
            request,
            action=AdaptiveDispatchAction.HOLD,
            reason=replay_reason,
            effective_ceiling=0,
            soft_pause=True,
            next_state=prior_state,
            projected_rss=projected_rss,
            projected_available=projected_available,
        )

    observation_replayed = (
        prior_state.last_observation_ordinal == observation.sample_ordinal
        and prior_state.last_observation_hash == observation.content_hash
    )
    current_state = _state_with_observation(prior_state, observation, request)

    if (
        observation_replayed
        and prior_state.consumed_admission_decision_token is not None
    ):
        return _evaluation(
            policy,
            prior_state,
            observation,
            request,
            action=AdaptiveDispatchAction.HOLD,
            reason="admission_already_consumed",
            effective_ceiling=0,
            soft_pause=False,
            next_state=prior_state,
            projected_rss=projected_rss,
            projected_available=projected_available,
        )

    if _known_at_least(
        observation.project_tree_rss_bytes, policy.project_hard_rss_bytes
    ):
        return _hard_stop(
            policy,
            prior_state,
            observation,
            request,
            current_state,
            "project_hard_rss",
            projected_rss,
            projected_available,
        )
    if _known_at_least(
        observation.maximum_worker_tree_rss_bytes,
        policy.worker_hard_rss_bytes,
    ):
        return _hard_stop(
            policy,
            prior_state,
            observation,
            request,
            current_state,
            "worker_hard_rss",
            projected_rss,
            projected_available,
        )
    if observation.memory_pressure is MemoryPressure.CRITICAL:
        return _hard_stop(
            policy,
            prior_state,
            observation,
            request,
            current_state,
            "critical_memory_pressure",
            projected_rss,
            projected_available,
        )

    invalid_authority = request.worker4_authority is not None and not _worker4_valid(
        policy,
        observation,
        request,
        fixed_workload_benchmark_authority,
    )
    if invalid_authority:
        return _soft_hold(
            policy,
            prior_state,
            observation,
            request,
            current_state,
            "worker4_authority_invalid",
            projected_rss,
            projected_available,
        )
    worker4_enabled = request.worker4_authority is not None
    cpu_ceiling = (
        policy.activated_cpu_ceiling_basis_points
        if worker4_enabled
        else policy.balanced_cpu_ceiling_basis_points
    )

    if (
        not observation.metrics_complete
        or observation.physical_memory_bytes is None
        or observation.available_memory_bytes is None
        or observation.project_tree_rss_bytes is None
        or observation.maximum_worker_tree_rss_bytes is None
        or observation.system_cpu_busy_basis_points is None
        or observation.memory_pressure is MemoryPressure.UNKNOWN
        or observation.swap_pressure is SwapPressure.UNKNOWN
    ):
        return _soft_hold(
            policy,
            prior_state,
            observation,
            request,
            current_state,
            "incomplete_observation",
            projected_rss,
            projected_available,
        )
    if observation.raw_reader_state is not RawReaderState.INACTIVE:
        return _soft_hold(
            policy,
            prior_state,
            observation,
            request,
            current_state,
            "raw_reader_not_inactive",
            projected_rss,
            projected_available,
        )
    if observation.memory_pressure is MemoryPressure.WARNING:
        return _soft_hold(
            policy,
            prior_state,
            observation,
            request,
            current_state,
            "memory_pressure",
            projected_rss,
            projected_available,
        )
    if observation.swap_pressure is SwapPressure.WARNING:
        return _soft_hold(
            policy,
            prior_state,
            observation,
            request,
            current_state,
            "swap_pressure",
            projected_rss,
            projected_available,
        )
    if observation.project_tree_rss_bytes >= policy.project_warning_rss_bytes:
        return _soft_hold(
            policy,
            prior_state,
            observation,
            request,
            current_state,
            "project_warning_rss",
            projected_rss,
            projected_available,
        )
    if observation.maximum_worker_tree_rss_bytes >= policy.worker_warning_rss_bytes:
        return _soft_hold(
            policy,
            prior_state,
            observation,
            request,
            current_state,
            "worker_warning_rss",
            projected_rss,
            projected_available,
        )
    if observation.system_cpu_busy_basis_points >= cpu_ceiling:
        return _soft_hold(
            policy,
            prior_state,
            observation,
            request,
            current_state,
            "cpu_pressure",
            projected_rss,
            projected_available,
        )
    if projected_rss is None or projected_rss > policy.project_hard_rss_bytes:
        return _soft_hold(
            policy,
            prior_state,
            observation,
            request,
            current_state,
            "project_capacity_insufficient",
            projected_rss,
            projected_available,
        )
    if (
        projected_available is None
        or projected_available < policy.foreground_reserve_bytes
    ):
        return _soft_hold(
            policy,
            prior_state,
            observation,
            request,
            current_state,
            "memory_reserve_insufficient",
            projected_rss,
            projected_available,
        )

    allowed_ceiling = min(
        policy.activated_worker_limit
        if worker4_enabled
        else policy.balanced_worker_limit,
        request.resource_budget_maximum_parallel_tasks,
    )
    next_state = (
        current_state
        if observation_replayed
        else _advance_low_pressure(
            policy,
            current_state,
            allowed_ceiling=max(policy.baseline_worker_limit, allowed_ceiling),
        )
    )
    effective_ceiling = min(
        next_state.adaptive_worker_ceiling,
        allowed_ceiling,
    )
    if request.pending_tasks == 0:
        return _evaluation(
            policy,
            prior_state,
            observation,
            request,
            action=AdaptiveDispatchAction.HOLD,
            reason="no_pending_tasks",
            effective_ceiling=effective_ceiling,
            soft_pause=False,
            next_state=next_state,
            projected_rss=projected_rss,
            projected_available=projected_available,
        )
    if request.active_workers + request.starting_workers >= effective_ceiling:
        return _evaluation(
            policy,
            prior_state,
            observation,
            request,
            action=AdaptiveDispatchAction.HOLD,
            reason="worker_ceiling_reached",
            effective_ceiling=effective_ceiling,
            soft_pause=False,
            next_state=next_state,
            projected_rss=projected_rss,
            projected_available=projected_available,
        )
    return _evaluation(
        policy,
        prior_state,
        observation,
        request,
        action=AdaptiveDispatchAction.ADMIT_ONE,
        reason="admitted",
        effective_ceiling=effective_ceiling,
        soft_pause=False,
        next_state=next_state,
        projected_rss=projected_rss,
        projected_available=projected_available,
    )


def _projected_memory(
    observation: AdaptiveResourceObservationV1,
    request: AdaptiveDispatchRequestV1,
) -> tuple[int | None, int | None]:
    commitment = (
        request.inflight_unrealized_memory_bytes + request.next_task_memory_limit_bytes
    )
    projected_rss = (
        None
        if observation.project_tree_rss_bytes is None
        else observation.project_tree_rss_bytes + commitment
    )
    projected_available = (
        None
        if observation.available_memory_bytes is None
        else observation.available_memory_bytes - commitment
    )
    return projected_rss, projected_available


def _replay_reason(
    policy: AdaptiveDispatchPolicyV1,
    state: AdaptiveDispatchStateV1,
    observation: AdaptiveResourceObservationV1,
    request: AdaptiveDispatchRequestV1,
) -> str | None:
    if state.experiment_spec_hash is None:
        return None
    if (
        state.experiment_spec_hash != request.experiment_spec_hash
        or state.workload_hash != request.workload_hash
    ):
        return "state_binding_mismatch"
    last_ordinal = cast(int, state.last_observation_ordinal)
    if observation.sample_ordinal < last_ordinal:
        return "stale_observation"
    if observation.sample_ordinal == last_ordinal and (
        observation.content_hash != state.last_observation_hash
    ):
        return "observation_replay_mismatch"
    if observation.sample_ordinal > last_ordinal:
        last_monotonic = cast(int, state.last_observation_monotonic_nanoseconds)
        minimum_delta = policy.scale_observation_interval_seconds * 1_000_000_000
        if observation.observed_monotonic_nanoseconds - last_monotonic < minimum_delta:
            return "observation_interval_too_short"
    return None


def _state_with_observation(
    state: AdaptiveDispatchStateV1,
    observation: AdaptiveResourceObservationV1,
    request: AdaptiveDispatchRequestV1,
) -> AdaptiveDispatchStateV1:
    if (
        state.last_observation_ordinal == observation.sample_ordinal
        and state.last_observation_hash == observation.content_hash
    ):
        return state
    return AdaptiveDispatchStateV1(
        experiment_spec_hash=request.experiment_spec_hash,
        workload_hash=request.workload_hash,
        last_observation_ordinal=observation.sample_ordinal,
        last_observation_monotonic_nanoseconds=(
            observation.observed_monotonic_nanoseconds
        ),
        last_observation_hash=observation.content_hash,
        adaptive_worker_ceiling=state.adaptive_worker_ceiling,
        low_pressure_windows=state.low_pressure_windows,
    )


def _reset_state(
    state: AdaptiveDispatchStateV1,
    observation: AdaptiveResourceObservationV1,
    request: AdaptiveDispatchRequestV1,
) -> AdaptiveDispatchStateV1:
    del state
    return AdaptiveDispatchStateV1(
        experiment_spec_hash=request.experiment_spec_hash,
        workload_hash=request.workload_hash,
        last_observation_ordinal=observation.sample_ordinal,
        last_observation_monotonic_nanoseconds=(
            observation.observed_monotonic_nanoseconds
        ),
        last_observation_hash=observation.content_hash,
    )


def _advance_low_pressure(
    policy: AdaptiveDispatchPolicyV1,
    state: AdaptiveDispatchStateV1,
    *,
    allowed_ceiling: int,
) -> AdaptiveDispatchStateV1:
    ceiling = min(state.adaptive_worker_ceiling, allowed_ceiling)
    if ceiling >= allowed_ceiling:
        windows = 0
    else:
        windows = state.low_pressure_windows + 1
        if windows >= policy.low_pressure_windows_required:
            ceiling += 1
            windows = 0
    return AdaptiveDispatchStateV1(
        experiment_spec_hash=state.experiment_spec_hash,
        workload_hash=state.workload_hash,
        last_observation_ordinal=state.last_observation_ordinal,
        last_observation_monotonic_nanoseconds=(
            state.last_observation_monotonic_nanoseconds
        ),
        last_observation_hash=state.last_observation_hash,
        adaptive_worker_ceiling=ceiling,
        low_pressure_windows=windows,
    )


def _worker4_valid(
    policy: AdaptiveDispatchPolicyV1,
    observation: AdaptiveResourceObservationV1,
    request: AdaptiveDispatchRequestV1,
    benchmark_authority: AdaptiveFixedWorkloadBenchmarkAuthorityV1 | None,
) -> bool:
    authority = request.worker4_authority
    evidence = None if benchmark_authority is None else benchmark_authority.evidence
    return bool(
        authority is not None
        and benchmark_authority is not None
        and evidence is not None
        and authority.benchmark_authority_hash == benchmark_authority.content_hash
        and authority.benchmark_evidence_hash == evidence.content_hash
        and authority.benchmark_workload_hash == request.workload_hash
        and authority.activation_workload_hash == request.workload_hash
        and authority.policy_hash == policy.content_hash
        and evidence.workload_hash == request.workload_hash
        and evidence.policy_hash == policy.content_hash
        and authority.activation_first_sample_ordinal
        <= observation.sample_ordinal
        <= authority.activation_last_sample_ordinal
    )


def _known_at_least(value: int | None, threshold: int) -> bool:
    return value is not None and value >= threshold


def _hard_stop(
    policy: AdaptiveDispatchPolicyV1,
    prior_state: AdaptiveDispatchStateV1,
    observation: AdaptiveResourceObservationV1,
    request: AdaptiveDispatchRequestV1,
    current_state: AdaptiveDispatchStateV1,
    reason: str,
    projected_rss: int | None,
    projected_available: int | None,
) -> AdaptiveDispatchEvaluationV1:
    return _evaluation(
        policy,
        prior_state,
        observation,
        request,
        action=AdaptiveDispatchAction.HARD_STOP,
        reason=reason,
        effective_ceiling=0,
        soft_pause=False,
        next_state=_reset_state(current_state, observation, request),
        projected_rss=projected_rss,
        projected_available=projected_available,
    )


def _soft_hold(
    policy: AdaptiveDispatchPolicyV1,
    prior_state: AdaptiveDispatchStateV1,
    observation: AdaptiveResourceObservationV1,
    request: AdaptiveDispatchRequestV1,
    current_state: AdaptiveDispatchStateV1,
    reason: str,
    projected_rss: int | None,
    projected_available: int | None,
) -> AdaptiveDispatchEvaluationV1:
    return _evaluation(
        policy,
        prior_state,
        observation,
        request,
        action=AdaptiveDispatchAction.HOLD,
        reason=reason,
        effective_ceiling=0,
        soft_pause=True,
        next_state=_reset_state(current_state, observation, request),
        projected_rss=projected_rss,
        projected_available=projected_available,
    )


def _evaluation(
    policy: AdaptiveDispatchPolicyV1,
    prior_state: AdaptiveDispatchStateV1,
    observation: AdaptiveResourceObservationV1,
    request: AdaptiveDispatchRequestV1,
    *,
    action: AdaptiveDispatchAction,
    reason: str,
    effective_ceiling: int,
    soft_pause: bool,
    next_state: AdaptiveDispatchStateV1,
    projected_rss: int | None,
    projected_available: int | None,
) -> AdaptiveDispatchEvaluationV1:
    hard = action is AdaptiveDispatchAction.HARD_STOP
    policy_hash = policy.content_hash
    observation_hash = observation.content_hash
    request_hash = request.content_hash
    prior_state_hash = prior_state.content_hash
    admission_decision_token = (
        _admission_decision_token_from_hashes(
            policy_hash=policy_hash,
            observation_hash=observation_hash,
            request_hash=request_hash,
            prior_state_hash=prior_state_hash,
        )
        if action is AdaptiveDispatchAction.ADMIT_ONE
        else None
    )
    if admission_decision_token is not None:
        next_state = _state_with_consumed_admission(
            next_state,
            admission_decision_token,
        )
    return AdaptiveDispatchEvaluationV1(
        action=action,
        reason=reason,
        admit_one=action is AdaptiveDispatchAction.ADMIT_ONE,
        effective_worker_ceiling=effective_ceiling,
        soft_pause=soft_pause,
        hard_stop_required=hard,
        terminate_running_workers=hard,
        projected_project_rss_bytes=projected_rss,
        projected_available_memory_bytes=projected_available,
        policy_hash=policy_hash,
        observation_hash=observation_hash,
        request_hash=request_hash,
        prior_state_hash=prior_state_hash,
        admission_decision_token=admission_decision_token,
        next_state=next_state,
    )


def _admission_decision_token_from_hashes(
    *,
    policy_hash: str,
    observation_hash: str,
    request_hash: str,
    prior_state_hash: str,
) -> str:
    return cast(
        str,
        hash_json(
            {
                "schema_version": _ADMISSION_DECISION_SCHEMA,
                "policy_hash": policy_hash,
                "observation_hash": observation_hash,
                "request_hash": request_hash,
                "prior_state_hash": prior_state_hash,
            }
        ),
    )


def _state_with_consumed_admission(
    state: AdaptiveDispatchStateV1,
    admission_decision_token: str,
) -> AdaptiveDispatchStateV1:
    if state.consumed_admission_decision_token is not None:
        raise ValueError("adaptive dispatch state already consumed an admission")
    return AdaptiveDispatchStateV1(
        experiment_spec_hash=state.experiment_spec_hash,
        workload_hash=state.workload_hash,
        last_observation_ordinal=state.last_observation_ordinal,
        last_observation_monotonic_nanoseconds=(
            state.last_observation_monotonic_nanoseconds
        ),
        last_observation_hash=state.last_observation_hash,
        consumed_admission_decision_token=admission_decision_token,
        adaptive_worker_ceiling=state.adaptive_worker_ceiling,
        low_pressure_windows=state.low_pressure_windows,
    )


__all__ = [
    "AdaptiveDispatchAction",
    "AdaptiveDispatchEvaluationV1",
    "AdaptiveFixedWorkloadBenchmarkAuthorityV1",
    "AdaptiveFixedWorkloadBenchmarkEvidenceV1",
    "AdaptiveDispatchPolicyV1",
    "AdaptiveDispatchRequestV1",
    "AdaptiveDispatchStateV1",
    "AdaptiveResourceObservationV1",
    "AdaptiveWorker4AuthorityV1",
    "MemoryPressure",
    "RawReaderState",
    "SwapPressure",
    "evaluate_adaptive_dispatch_v1",
]
