"""Read-only system resource observations for adaptive G1 dispatch.

The provider in this module only samples already-running processes and host
counters.  It never launches, signals, renices, or otherwise mutates a process.
Every partial observation is represented explicitly so the pure adaptive
dispatch policy can fail closed instead of treating missing metrics as spare
capacity.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Final, Protocol, cast

import psutil

from alpha_research.core.hashing import hash_json
from alpha_research.orchestration.adaptive_dispatch_policy import (
    AdaptiveResourceObservationV1,
    MemoryPressure,
    RawReaderState,
    SwapPressure,
)


_SCHEMA: Final[str] = "system-resource-observation/v1"
_MEMORY_PRESSURE_SEMANTICS: Final[str] = (
    "psutil_available_ratio:critical<=10%;warning<=20%;normal>20%"
)
_SWAP_PRESSURE_SEMANTICS: Final[str] = (
    "psutil_swap_io_delta:warning=sin_or_sout_increase;"
    "normal=no_increase;first_or_reset=unknown"
)
_AUTHORITY_SCHEMA: Final[str] = "adaptive-resource-observation-authority/v1"


class ResourceObservationIssue(str, Enum):
    """Closed reason vocabulary for unavailable or partial observations."""

    CPU_UNAVAILABLE = "cpu_unavailable"
    PROCESS_TREE_ACCESS_DENIED = "process_tree_access_denied"
    PROCESS_TREE_CHANGED = "process_tree_changed"
    PROCESS_TREE_DISAPPEARED = "process_tree_disappeared"
    PROCESS_TREE_INVALID = "process_tree_invalid"
    RAW_READER_STATE_UNAVAILABLE = "raw_reader_state_unavailable"
    ROOT_IDENTITY_UNAVAILABLE = "root_identity_unavailable"
    ROOT_PID_REUSED = "root_pid_reused"
    SWAP_BASELINE_UNAVAILABLE = "swap_baseline_unavailable"
    SWAP_COUNTER_RESET = "swap_counter_reset"
    SWAP_UNAVAILABLE = "swap_unavailable"
    SYSTEM_MEMORY_UNAVAILABLE = "system_memory_unavailable"


class ResourceObservationProviderError(RuntimeError):
    """The provider could not produce a correctly ordered observation."""


class _MemoryInfoLike(Protocol):
    rss: int


class _ProcessLike(Protocol):
    pid: int

    def create_time(self) -> float: ...

    def memory_info(self) -> _MemoryInfoLike: ...

    def children(self, *, recursive: bool) -> Sequence[_ProcessLike]: ...


class _VirtualMemoryLike(Protocol):
    total: int
    available: int


class _SwapMemoryLike(Protocol):
    total: int
    sin: int
    sout: int


ProcessFactory = Callable[[int], _ProcessLike]
VirtualMemoryProvider = Callable[[], _VirtualMemoryLike]
SwapMemoryProvider = Callable[[], _SwapMemoryLike]
CpuPercentProvider = Callable[[float | None], float]
RawReaderStateProvider = Callable[[], RawReaderState]


@dataclass(frozen=True, slots=True)
class SystemResourceObservationV1:
    """Immutable, exact wire observation with an adaptive-policy projection."""

    sample_ordinal: int
    observed_at_utc: str
    observed_monotonic_nanoseconds: int
    root_pid: int
    root_create_time_nanoseconds: int | None
    metrics_complete: bool
    physical_memory_bytes: int | None
    available_memory_bytes: int | None
    project_tree_rss_bytes: int | None
    maximum_worker_tree_rss_bytes: int | None
    system_cpu_busy_basis_points: int | None
    memory_pressure: MemoryPressure
    swap_pressure: SwapPressure
    raw_reader_state: RawReaderState
    swap_total_bytes: int | None
    swap_in_bytes: int | None
    swap_out_bytes: int | None
    issues: tuple[ResourceObservationIssue, ...]
    memory_pressure_semantics: str = _MEMORY_PRESSURE_SEMANTICS
    swap_pressure_semantics: str = _SWAP_PRESSURE_SEMANTICS
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        _require_exact_int(self.sample_ordinal, name="sample_ordinal", minimum=1)
        _require_utc_timestamp(self.observed_at_utc)
        _require_exact_int(
            self.observed_monotonic_nanoseconds,
            name="observed_monotonic_nanoseconds",
            minimum=1,
        )
        _require_exact_int(self.root_pid, name="root_pid", minimum=1)
        _require_optional_int(
            self.root_create_time_nanoseconds,
            name="root_create_time_nanoseconds",
            minimum=1,
        )
        if type(self.metrics_complete) is not bool:
            raise TypeError("metrics_complete must be an exact boolean")
        for name in (
            "physical_memory_bytes",
            "available_memory_bytes",
            "project_tree_rss_bytes",
            "maximum_worker_tree_rss_bytes",
            "swap_total_bytes",
            "swap_in_bytes",
            "swap_out_bytes",
        ):
            _require_optional_int(getattr(self, name), name=name)
        _require_optional_int(
            self.system_cpu_busy_basis_points,
            name="system_cpu_busy_basis_points",
            maximum=10_000,
        )
        if self.physical_memory_bytes == 0:
            raise ValueError("physical_memory_bytes must be positive when known")
        if (
            self.physical_memory_bytes is not None
            and self.available_memory_bytes is not None
            and self.available_memory_bytes > self.physical_memory_bytes
        ):
            raise ValueError("available memory cannot exceed physical memory")
        if type(self.memory_pressure) is not MemoryPressure:
            raise TypeError("memory_pressure must be an exact MemoryPressure")
        if type(self.swap_pressure) is not SwapPressure:
            raise TypeError("swap_pressure must be an exact SwapPressure")
        if type(self.raw_reader_state) is not RawReaderState:
            raise TypeError("raw_reader_state must be an exact RawReaderState")
        if type(self.issues) is not tuple or any(
            type(issue) is not ResourceObservationIssue for issue in self.issues
        ):
            raise TypeError("issues must be an exact issue tuple")
        canonical_issues = tuple(
            sorted(set(self.issues), key=lambda issue: issue.value)
        )
        if canonical_issues != self.issues:
            raise ValueError("issues must be unique and canonically ordered")
        if self.memory_pressure_semantics != _MEMORY_PRESSURE_SEMANTICS:
            raise ValueError("memory pressure semantics differ")
        if self.swap_pressure_semantics != _SWAP_PRESSURE_SEMANTICS:
            raise ValueError("swap pressure semantics differ")
        if self.schema_version != _SCHEMA:
            raise ValueError("unsupported system resource observation schema")
        required_metrics = (
            self.physical_memory_bytes,
            self.available_memory_bytes,
            self.project_tree_rss_bytes,
            self.maximum_worker_tree_rss_bytes,
            self.system_cpu_busy_basis_points,
        )
        if self.metrics_complete is not all(
            value is not None for value in required_metrics
        ):
            raise ValueError("metrics_complete differs from observed metrics")
        swap_values = (
            self.swap_total_bytes,
            self.swap_in_bytes,
            self.swap_out_bytes,
        )
        if any(value is None for value in swap_values) and not all(
            value is None for value in swap_values
        ):
            raise ValueError("swap counters must be complete or wholly unknown")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_adaptive_observation(self) -> AdaptiveResourceObservationV1:
        """Return a lossy policy input that grants no launch authority.

        The pure policy input omits the root birth identity, issue list, UTC
        timestamp, and system-observation hash.  A launcher must bind
        :meth:`to_adaptive_authority` and atomically revalidate the process-tree
        identity plus raw-reader lease immediately before spawning anything.
        """

        return AdaptiveResourceObservationV1(
            sample_ordinal=self.sample_ordinal,
            observed_monotonic_nanoseconds=self.observed_monotonic_nanoseconds,
            metrics_complete=self.metrics_complete,
            physical_memory_bytes=self.physical_memory_bytes,
            available_memory_bytes=self.available_memory_bytes,
            project_tree_rss_bytes=self.project_tree_rss_bytes,
            maximum_worker_tree_rss_bytes=self.maximum_worker_tree_rss_bytes,
            system_cpu_busy_basis_points=self.system_cpu_busy_basis_points,
            memory_pressure=self.memory_pressure,
            swap_pressure=self.swap_pressure,
            raw_reader_state=self.raw_reader_state,
        )

    def to_adaptive_authority(self) -> AdaptiveResourceObservationAuthorityV1:
        """Bind the lossy policy projection back to this complete observation."""

        return AdaptiveResourceObservationAuthorityV1.create(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_ordinal": self.sample_ordinal,
            "observed_at_utc": self.observed_at_utc,
            "observed_monotonic_nanoseconds": self.observed_monotonic_nanoseconds,
            "root_pid": self.root_pid,
            "root_create_time_nanoseconds": self.root_create_time_nanoseconds,
            "metrics_complete": self.metrics_complete,
            "physical_memory_bytes": self.physical_memory_bytes,
            "available_memory_bytes": self.available_memory_bytes,
            "project_tree_rss_bytes": self.project_tree_rss_bytes,
            "maximum_worker_tree_rss_bytes": self.maximum_worker_tree_rss_bytes,
            "system_cpu_busy_basis_points": self.system_cpu_busy_basis_points,
            "memory_pressure": self.memory_pressure.value,
            "swap_pressure": self.swap_pressure.value,
            "raw_reader_state": self.raw_reader_state.value,
            "swap_total_bytes": self.swap_total_bytes,
            "swap_in_bytes": self.swap_in_bytes,
            "swap_out_bytes": self.swap_out_bytes,
            "issues": [issue.value for issue in self.issues],
            "memory_pressure_semantics": self.memory_pressure_semantics,
            "swap_pressure_semantics": self.swap_pressure_semantics,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(cls, value: object) -> SystemResourceObservationV1:
        expected = frozenset(cls.__dataclass_fields__)
        if type(value) is not dict:
            raise TypeError("SystemResourceObservationV1 wire must be an exact object")
        mapping = cast(dict[object, object], value)
        if any(type(key) is not str for key in mapping):
            raise TypeError("SystemResourceObservationV1 wire keys must be exact text")
        typed = cast(dict[str, object], mapping)
        if frozenset(typed) != expected:
            raise ValueError("SystemResourceObservationV1 wire fields differ")
        memory_pressure = _wire_text(typed["memory_pressure"], "memory_pressure")
        swap_pressure = _wire_text(typed["swap_pressure"], "swap_pressure")
        raw_reader_state = _wire_text(
            typed["raw_reader_state"], "raw_reader_state"
        )
        raw_issues = typed["issues"]
        if type(raw_issues) is not list or any(
            type(issue) is not str for issue in raw_issues
        ):
            raise TypeError("issues must be an exact text array")
        return cls(
            sample_ordinal=_wire_int(typed["sample_ordinal"], "sample_ordinal"),
            observed_at_utc=_wire_text(typed["observed_at_utc"], "observed_at_utc"),
            observed_monotonic_nanoseconds=_wire_int(
                typed["observed_monotonic_nanoseconds"],
                "observed_monotonic_nanoseconds",
            ),
            root_pid=_wire_int(typed["root_pid"], "root_pid"),
            root_create_time_nanoseconds=_wire_optional_int(
                typed["root_create_time_nanoseconds"],
                "root_create_time_nanoseconds",
            ),
            metrics_complete=_wire_bool(
                typed["metrics_complete"], "metrics_complete"
            ),
            physical_memory_bytes=_wire_optional_int(
                typed["physical_memory_bytes"], "physical_memory_bytes"
            ),
            available_memory_bytes=_wire_optional_int(
                typed["available_memory_bytes"], "available_memory_bytes"
            ),
            project_tree_rss_bytes=_wire_optional_int(
                typed["project_tree_rss_bytes"], "project_tree_rss_bytes"
            ),
            maximum_worker_tree_rss_bytes=_wire_optional_int(
                typed["maximum_worker_tree_rss_bytes"],
                "maximum_worker_tree_rss_bytes",
            ),
            system_cpu_busy_basis_points=_wire_optional_int(
                typed["system_cpu_busy_basis_points"],
                "system_cpu_busy_basis_points",
            ),
            memory_pressure=MemoryPressure(memory_pressure),
            swap_pressure=SwapPressure(swap_pressure),
            raw_reader_state=RawReaderState(raw_reader_state),
            swap_total_bytes=_wire_optional_int(
                typed["swap_total_bytes"], "swap_total_bytes"
            ),
            swap_in_bytes=_wire_optional_int(
                typed["swap_in_bytes"], "swap_in_bytes"
            ),
            swap_out_bytes=_wire_optional_int(
                typed["swap_out_bytes"], "swap_out_bytes"
            ),
            issues=tuple(ResourceObservationIssue(issue) for issue in raw_issues),
            memory_pressure_semantics=_wire_text(
                typed["memory_pressure_semantics"], "memory_pressure_semantics"
            ),
            swap_pressure_semantics=_wire_text(
                typed["swap_pressure_semantics"], "swap_pressure_semantics"
            ),
            schema_version=_wire_text(typed["schema_version"], "schema_version"),
        )


@dataclass(frozen=True, slots=True)
class AdaptiveResourceObservationAuthorityV1:
    """Non-launching receipt that preserves facts omitted by the pure policy.

    This receipt deliberately says ``grants_spawn_authority=False``.  Even a
    valid receipt can become stale between observation and process launch; the
    launcher must atomically revalidate root identity and its raw-reader
    inactivity lease at the launch boundary.
    """

    system_observation: SystemResourceObservationV1
    system_observation_hash: str
    adaptive_observation: AdaptiveResourceObservationV1
    grants_spawn_authority: bool = False
    requires_atomic_root_and_raw_reader_revalidation: bool = True
    schema_version: str = _AUTHORITY_SCHEMA

    def __post_init__(self) -> None:
        if type(self.system_observation) is not SystemResourceObservationV1:
            raise TypeError("system_observation must be exact")
        if type(self.system_observation_hash) is not str:
            raise TypeError("system_observation_hash must be exact text")
        if self.system_observation_hash != self.system_observation.content_hash:
            raise ValueError("system observation authority hash differs")
        if type(self.adaptive_observation) is not AdaptiveResourceObservationV1:
            raise TypeError("adaptive_observation must be exact")
        expected = self.system_observation.to_adaptive_observation()
        if self.adaptive_observation.to_dict() != expected.to_dict():
            raise ValueError("adaptive observation projection differs")
        if self.grants_spawn_authority is not False:
            raise ValueError("resource observation never grants spawn authority")
        if self.requires_atomic_root_and_raw_reader_revalidation is not True:
            raise ValueError("atomic launcher revalidation is mandatory")
        if self.schema_version != _AUTHORITY_SCHEMA:
            raise ValueError("unsupported resource observation authority")

    @classmethod
    def create(
        cls, observation: SystemResourceObservationV1
    ) -> AdaptiveResourceObservationAuthorityV1:
        if type(observation) is not SystemResourceObservationV1:
            raise TypeError("observation must be exact")
        return cls(
            system_observation=observation,
            system_observation_hash=observation.content_hash,
            adaptive_observation=observation.to_adaptive_observation(),
        )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "system_observation": self.system_observation.to_dict(),
            "system_observation_hash": self.system_observation_hash,
            "adaptive_observation": self.adaptive_observation.to_dict(),
            "grants_spawn_authority": self.grants_spawn_authority,
            "requires_atomic_root_and_raw_reader_revalidation": (
                self.requires_atomic_root_and_raw_reader_revalidation
            ),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(
        cls, value: object
    ) -> AdaptiveResourceObservationAuthorityV1:
        expected = frozenset(cls.__dataclass_fields__)
        if type(value) is not dict:
            raise TypeError(
                "AdaptiveResourceObservationAuthorityV1 wire must be an exact object"
            )
        mapping = cast(dict[object, object], value)
        if any(type(key) is not str for key in mapping):
            raise TypeError("authority wire keys must be exact text")
        typed = cast(dict[str, object], mapping)
        if frozenset(typed) != expected:
            raise ValueError(
                "AdaptiveResourceObservationAuthorityV1 wire fields differ"
            )
        raw_system = typed["system_observation"]
        raw_adaptive = typed["adaptive_observation"]
        if type(raw_system) is not dict or type(raw_adaptive) is not dict:
            raise TypeError("authority observations must be exact objects")
        return cls(
            system_observation=SystemResourceObservationV1.from_mapping(raw_system),
            system_observation_hash=_wire_text(
                typed["system_observation_hash"], "system_observation_hash"
            ),
            adaptive_observation=AdaptiveResourceObservationV1.from_mapping(
                cast(dict[str, object], raw_adaptive)
            ),
            grants_spawn_authority=_wire_bool(
                typed["grants_spawn_authority"], "grants_spawn_authority"
            ),
            requires_atomic_root_and_raw_reader_revalidation=_wire_bool(
                typed["requires_atomic_root_and_raw_reader_revalidation"],
                "requires_atomic_root_and_raw_reader_revalidation",
            ),
            schema_version=_wire_text(typed["schema_version"], "schema_version"),
        )


class ResourceObservationProviderV1:
    """Thread-safe read-only sampler bound to one immutable root PID identity."""

    def __init__(
        self,
        *,
        root_pid: int | None = None,
        cpu_interval_seconds: float = 0.1,
        process_factory: ProcessFactory | None = None,
        virtual_memory_provider: VirtualMemoryProvider | None = None,
        swap_memory_provider: SwapMemoryProvider | None = None,
        cpu_percent_provider: CpuPercentProvider | None = None,
        raw_reader_state_provider: RawReaderStateProvider | None = None,
        utc_now: Callable[[], datetime] | None = None,
        monotonic_ns: Callable[[], int] | None = None,
    ) -> None:
        selected_pid = os.getpid() if root_pid is None else root_pid
        _require_exact_int(selected_pid, name="root_pid", minimum=1)
        if (
            isinstance(cpu_interval_seconds, bool)
            or not isinstance(cpu_interval_seconds, (int, float))
            or not math.isfinite(float(cpu_interval_seconds))
            or cpu_interval_seconds <= 0
        ):
            raise ValueError("cpu_interval_seconds must be a finite positive number")
        self._root_pid = selected_pid
        self._cpu_interval_seconds = float(cpu_interval_seconds)
        self._process_factory = process_factory or cast(ProcessFactory, psutil.Process)
        self._virtual_memory = virtual_memory_provider or cast(
            VirtualMemoryProvider, psutil.virtual_memory
        )
        self._swap_memory = swap_memory_provider or cast(
            SwapMemoryProvider, psutil.swap_memory
        )
        self._cpu_percent = cpu_percent_provider or cast(
            CpuPercentProvider, psutil.cpu_percent
        )
        self._raw_reader_state = raw_reader_state_provider
        self._utc_now = utc_now or (lambda: datetime.now(timezone.utc))
        self._monotonic_ns = monotonic_ns or time.monotonic_ns
        self._lock = threading.Lock()
        self._last_ordinal = 0
        self._last_monotonic_ns = 0
        self._last_swap_counters: tuple[int, int] | None = None
        self._root_create_time_ns = self._initial_root_identity()

    def observe(self) -> SystemResourceObservationV1:
        """Sample once; missing or unstable facts remain explicit and fail closed."""

        with self._lock:
            observed_at = self._canonical_utc_now()
            monotonic_ns = self._monotonic_ns()
            _require_exact_int(
                monotonic_ns,
                name="observed_monotonic_nanoseconds",
                minimum=1,
            )
            if monotonic_ns <= self._last_monotonic_ns:
                raise ResourceObservationProviderError(
                    "monotonic clock did not advance"
                )
            ordinal = self._last_ordinal + 1
            self._last_ordinal = ordinal
            self._last_monotonic_ns = monotonic_ns

            issues: list[ResourceObservationIssue] = []
            physical, available, memory_pressure, memory_issue = self._observe_memory()
            if memory_issue is not None:
                issues.append(memory_issue)
            cpu, cpu_issue = self._observe_cpu()
            if cpu_issue is not None:
                issues.append(cpu_issue)
            project_rss, worker_rss, process_issue = self._observe_process_tree()
            if process_issue is not None:
                issues.append(process_issue)
            swap_values, swap_pressure, swap_issue = self._observe_swap()
            if swap_issue is not None:
                issues.append(swap_issue)
            raw_state, raw_issue = self._observe_raw_reader_state()
            if raw_issue is not None:
                issues.append(raw_issue)

            metrics = (physical, available, project_rss, worker_rss, cpu)
            return SystemResourceObservationV1(
                sample_ordinal=ordinal,
                observed_at_utc=observed_at,
                observed_monotonic_nanoseconds=monotonic_ns,
                root_pid=self._root_pid,
                root_create_time_nanoseconds=self._root_create_time_ns,
                metrics_complete=all(value is not None for value in metrics),
                physical_memory_bytes=physical,
                available_memory_bytes=available,
                project_tree_rss_bytes=project_rss,
                maximum_worker_tree_rss_bytes=worker_rss,
                system_cpu_busy_basis_points=cpu,
                memory_pressure=memory_pressure,
                swap_pressure=swap_pressure,
                raw_reader_state=raw_state,
                swap_total_bytes=(swap_values[0] if swap_values is not None else None),
                swap_in_bytes=(swap_values[1] if swap_values is not None else None),
                swap_out_bytes=(swap_values[2] if swap_values is not None else None),
                issues=tuple(sorted(set(issues), key=lambda issue: issue.value)),
            )

    def _canonical_utc_now(self) -> str:
        observed = self._utc_now()
        if type(observed) is not datetime or observed.tzinfo is None:
            raise ResourceObservationProviderError(
                "UTC clock must return an aware datetime"
            )
        if observed.utcoffset() is None:
            raise ResourceObservationProviderError(
                "UTC clock must return an aware datetime"
            )
        return observed.astimezone(timezone.utc).isoformat(timespec="microseconds")

    def _initial_root_identity(self) -> int | None:
        try:
            process = self._process_factory(self._root_pid)
            return _process_identity(process, expected_pid=self._root_pid)[1]
        except _PROCESS_GONE_OR_DENIED:
            return None
        except (TypeError, ValueError, OverflowError, OSError):
            return None

    def _observe_memory(
        self,
    ) -> tuple[int | None, int | None, MemoryPressure, ResourceObservationIssue | None]:
        try:
            memory = self._virtual_memory()
            total = _require_exact_int(memory.total, name="memory.total", minimum=1)
            available = _require_exact_int(
                memory.available,
                name="memory.available",
            )
            if available > total:
                raise ValueError("available memory exceeds total")
        except (AttributeError, TypeError, ValueError, OSError, psutil.Error):
            return (
                None,
                None,
                MemoryPressure.UNKNOWN,
                ResourceObservationIssue.SYSTEM_MEMORY_UNAVAILABLE,
            )
        if available * 10 <= total:
            pressure = MemoryPressure.CRITICAL
        elif available * 5 <= total:
            pressure = MemoryPressure.WARNING
        else:
            pressure = MemoryPressure.NORMAL
        return total, available, pressure, None

    def _observe_cpu(
        self,
    ) -> tuple[int | None, ResourceObservationIssue | None]:
        try:
            observed = self._cpu_percent(self._cpu_interval_seconds)
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not math.isfinite(float(observed))
                or not 0 <= observed <= 100
            ):
                raise ValueError("system CPU percent is invalid")
            return int(round(float(observed) * 100)), None
        except (TypeError, ValueError, OSError, RuntimeError, psutil.Error):
            return None, ResourceObservationIssue.CPU_UNAVAILABLE

    def _observe_process_tree(
        self,
    ) -> tuple[int | None, int | None, ResourceObservationIssue | None]:
        if self._root_create_time_ns is None:
            return None, None, ResourceObservationIssue.ROOT_IDENTITY_UNAVAILABLE
        try:
            root_before = self._process_factory(self._root_pid)
            root_identity = _process_identity(
                root_before,
                expected_pid=self._root_pid,
            )
            if root_identity[1] != self._root_create_time_ns:
                return None, None, ResourceObservationIssue.ROOT_PID_REUSED

            descendants_before = _process_map(root_before.children(recursive=True))
            if root_identity in descendants_before:
                raise ValueError("root appears in its descendant closure")
            direct_before = _process_map(root_before.children(recursive=False))
            if not set(direct_before).issubset(descendants_before):
                raise ValueError("direct children differ from recursive closure")
            worker_closures_before = _worker_closures(direct_before)
            for closure in worker_closures_before.values():
                if not closure.issubset(descendants_before):
                    raise ValueError("worker subtree differs from root closure")
            if _union_worker_closures(worker_closures_before) != set(
                descendants_before
            ):
                raise ValueError("worker subtrees do not cover root closure")

            all_before = {root_identity: root_before, **descendants_before}
            rss_by_identity: dict[tuple[int, int], int] = {}
            for identity, process in all_before.items():
                sampled_identity, rss = _process_sample(
                    process,
                    process_factory=self._process_factory,
                )
                if sampled_identity != identity:
                    raise psutil.NoSuchProcess(
                        pid=identity[0],
                        msg="PID identity changed during RSS sample",
                    )
                rss_by_identity[identity] = rss

            root_after = self._process_factory(self._root_pid)
            final_root_identity = _process_identity(
                root_after,
                expected_pid=self._root_pid,
            )
            if final_root_identity != root_identity:
                return None, None, ResourceObservationIssue.ROOT_PID_REUSED
            descendants_after = _process_map(root_after.children(recursive=True))
            direct_after = _process_map(root_after.children(recursive=False))
            worker_closures_after = _worker_closures(direct_after)
            if (
                set(descendants_before) != set(descendants_after)
                or set(direct_before) != set(direct_after)
                or worker_closures_before != worker_closures_after
            ):
                return None, None, ResourceObservationIssue.PROCESS_TREE_CHANGED

            project_rss = sum(rss_by_identity.values())
            worker_totals = [
                sum(rss_by_identity[identity] for identity in closure)
                for closure in worker_closures_before.values()
            ]
            return project_rss, max(worker_totals, default=0), None
        except (psutil.AccessDenied, PermissionError):
            return None, None, ResourceObservationIssue.PROCESS_TREE_ACCESS_DENIED
        except (psutil.NoSuchProcess, psutil.ZombieProcess, ProcessLookupError):
            return None, None, ResourceObservationIssue.PROCESS_TREE_DISAPPEARED
        except (
            AttributeError,
            TypeError,
            ValueError,
            OverflowError,
            OSError,
            RuntimeError,
            psutil.Error,
        ):
            return None, None, ResourceObservationIssue.PROCESS_TREE_INVALID

    def _observe_swap(
        self,
    ) -> tuple[
        tuple[int, int, int] | None,
        SwapPressure,
        ResourceObservationIssue | None,
    ]:
        try:
            swap = self._swap_memory()
            total = _require_exact_int(swap.total, name="swap.total")
            swap_in = _require_exact_int(swap.sin, name="swap.sin")
            swap_out = _require_exact_int(swap.sout, name="swap.sout")
        except (AttributeError, TypeError, ValueError, OSError, psutil.Error):
            return None, SwapPressure.UNKNOWN, ResourceObservationIssue.SWAP_UNAVAILABLE
        current = (swap_in, swap_out)
        previous = self._last_swap_counters
        self._last_swap_counters = current
        if total == 0:
            return (total, swap_in, swap_out), SwapPressure.NORMAL, None
        if previous is None:
            return (
                (total, swap_in, swap_out),
                SwapPressure.UNKNOWN,
                ResourceObservationIssue.SWAP_BASELINE_UNAVAILABLE,
            )
        if swap_in < previous[0] or swap_out < previous[1]:
            return (
                (total, swap_in, swap_out),
                SwapPressure.UNKNOWN,
                ResourceObservationIssue.SWAP_COUNTER_RESET,
            )
        pressure = (
            SwapPressure.WARNING
            if swap_in > previous[0] or swap_out > previous[1]
            else SwapPressure.NORMAL
        )
        return (total, swap_in, swap_out), pressure, None

    def _observe_raw_reader_state(
        self,
    ) -> tuple[RawReaderState, ResourceObservationIssue | None]:
        provider = self._raw_reader_state
        if provider is None:
            return (
                RawReaderState.UNKNOWN,
                ResourceObservationIssue.RAW_READER_STATE_UNAVAILABLE,
            )
        try:
            state = provider()
            if type(state) is not RawReaderState:
                raise TypeError("raw reader state provider returned an untyped value")
            return state, None
        except (TypeError, ValueError, OSError, RuntimeError, psutil.Error):
            return (
                RawReaderState.UNKNOWN,
                ResourceObservationIssue.RAW_READER_STATE_UNAVAILABLE,
            )


_PROCESS_GONE_OR_DENIED = (
    psutil.NoSuchProcess,
    psutil.ZombieProcess,
    psutil.AccessDenied,
    ProcessLookupError,
    PermissionError,
)


def _process_identity(
    process: _ProcessLike,
    *,
    expected_pid: int | None = None,
) -> tuple[int, int]:
    pid = _require_exact_int(process.pid, name="process.pid", minimum=1)
    if expected_pid is not None and pid != expected_pid:
        raise ValueError("process factory returned a different PID")
    created = process.create_time()
    if (
        isinstance(created, bool)
        or not isinstance(created, (int, float))
        or not math.isfinite(float(created))
        or created <= 0
    ):
        raise ValueError("process create_time is invalid")
    return pid, int(round(float(created) * 1_000_000_000))


def _process_sample(
    process: _ProcessLike,
    *,
    process_factory: ProcessFactory,
    expected_pid: int | None = None,
) -> tuple[tuple[int, int], int]:
    identity = _process_identity(process, expected_pid=expected_pid)
    rss = _require_exact_int(process.memory_info().rss, name="process.rss")
    fresh = process_factory(identity[0])
    if _process_identity(fresh, expected_pid=identity[0]) != identity:
        raise psutil.NoSuchProcess(pid=identity[0], msg="PID identity changed")
    return identity, rss


def _process_map(
    processes: Sequence[_ProcessLike],
) -> dict[tuple[int, int], _ProcessLike]:
    result: dict[tuple[int, int], _ProcessLike] = {}
    pids: set[int] = set()
    for process in processes:
        identity = _process_identity(process)
        if identity in result or identity[0] in pids:
            raise ValueError("process closure contains duplicate or reused PID")
        result[identity] = process
        pids.add(identity[0])
    return result


def _worker_closures(
    direct: Mapping[tuple[int, int], _ProcessLike],
) -> dict[tuple[int, int], frozenset[tuple[int, int]]]:
    closures: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
    claimed: set[tuple[int, int]] = set()
    for identity, process in direct.items():
        descendants = _process_map(process.children(recursive=True))
        closure = frozenset((identity, *descendants))
        if claimed.intersection(closure):
            raise ValueError("worker subtree closures overlap")
        claimed.update(closure)
        closures[identity] = closure
    return closures


def _union_worker_closures(
    closures: Mapping[tuple[int, int], frozenset[tuple[int, int]]],
) -> set[tuple[int, int]]:
    covered: set[tuple[int, int]] = set()
    for closure in closures.values():
        covered.update(closure)
    return covered


def _require_exact_int(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an exact integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _require_optional_int(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int | None:
    if value is None:
        return None
    return _require_exact_int(value, name=name, minimum=minimum, maximum=maximum)


def _require_utc_timestamp(value: object) -> str:
    if type(value) is not str:
        raise TypeError("observed_at_utc must be exact text")
    try:
        observed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("observed_at_utc is invalid") from exc
    if observed.tzinfo is None or observed.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError("observed_at_utc must carry UTC offset")
    if observed.isoformat(timespec="microseconds") != value:
        raise ValueError("observed_at_utc is not canonical")
    return value


def _wire_text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be exact text")
    return value


def _wire_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    return value


def _wire_optional_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _wire_int(value, name)


def _wire_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact boolean")
    return value


__all__ = [
    "AdaptiveResourceObservationAuthorityV1",
    "ResourceObservationIssue",
    "ResourceObservationProviderError",
    "ResourceObservationProviderV1",
    "SystemResourceObservationV1",
]
