"""Pure topology policy for one raw reader and adaptive sealed-data compute.

The policy is deliberately non-authorizing: it never opens raw data, spawns a
process, consumes an admission token, or writes state.  It composes the frozen
adaptive-dispatch policy with the L2 campaign topology requested for a 16 GiB
workstation:

* at most one Samsung T5 raw reader;
* two baseline compute workers, three after sustained low pressure;
* four compute workers only while the raw reader is inactive and an independent
  fixed-workload benchmark authority is present;
* a 12 GiB project hard limit and a 4 GiB foreground reserve;
* compute may consume only a content-addressed sealed-partition catalog.

The raw reader has priority.  While a raw reader is pending, the policy either
previews that one reader or holds; it does not admit another compute worker that
could prevent the reader from starting.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Final, cast

from alpha_research.core.hashing import hash_json, require_sha256
from alpha_research.orchestration.adaptive_dispatch_policy import (
    AdaptiveDispatchAction,
    AdaptiveDispatchPolicyV1,
    AdaptiveDispatchRequestV1,
    AdaptiveDispatchStateV1,
    AdaptiveFixedWorkloadBenchmarkAuthorityV1,
    AdaptiveResourceObservationV1,
    AdaptiveWorker4AuthorityV1,
    MemoryPressure,
    RawReaderState,
    SwapPressure,
    evaluate_adaptive_dispatch_v1,
)


_GIB: Final[int] = 1024 * 1024 * 1024
_POLICY_SCHEMA: Final[str] = "l2-campaign-topology-policy/v1"
_REQUEST_SCHEMA: Final[str] = "l2-campaign-topology-request/v1"
_DECISION_SCHEMA: Final[str] = "l2-campaign-topology-decision/v1"


class L2CampaignTopologyAction(str, Enum):
    """A non-authorizing preview of the next topology transition."""

    PREVIEW_RAW_READER = "preview_raw_reader"
    PREVIEW_COMPUTE = "preview_compute"
    HOLD = "hold"
    HARD_STOP = "hard_stop"


@dataclass(frozen=True, slots=True)
class L2CampaignTopologyPolicyV1:
    maximum_raw_readers: int = 1
    baseline_compute_workers: int = 2
    balanced_compute_workers: int = 3
    activated_compute_workers: int = 4
    maximum_compute_workers_while_raw_active: int = 3
    raw_reader_memory_commitment_bytes: int = 5 * _GIB // 2
    compute_worker_memory_commitment_bytes: int = 5 * _GIB // 2
    project_hard_rss_bytes: int = 12 * _GIB
    foreground_reserve_bytes: int = 4 * _GIB
    numeric_thread_limit: int = 1
    process_nice: int = 5
    canonical_reducer_single_writer: bool = True
    research_only: bool = True
    production_ready: bool = False
    schema_version: str = _POLICY_SCHEMA

    def __post_init__(self) -> None:
        expected: dict[str, object] = {
            "maximum_raw_readers": 1,
            "baseline_compute_workers": 2,
            "balanced_compute_workers": 3,
            "activated_compute_workers": 4,
            "maximum_compute_workers_while_raw_active": 3,
            "raw_reader_memory_commitment_bytes": 5 * _GIB // 2,
            "compute_worker_memory_commitment_bytes": 5 * _GIB // 2,
            "project_hard_rss_bytes": 12 * _GIB,
            "foreground_reserve_bytes": 4 * _GIB,
            "numeric_thread_limit": 1,
            "process_nice": 5,
            "canonical_reducer_single_writer": True,
            "research_only": True,
            "production_ready": False,
            "schema_version": _POLICY_SCHEMA,
        }
        for name, required in expected.items():
            value = getattr(self, name)
            if type(value) is not type(required) or value != required:
                raise ValueError(f"L2 topology V1 {name} differs")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class L2CampaignTopologyRequestV1:
    experiment_spec_hash: str
    workload_hash: str
    sealed_partition_catalog_hash: str | None
    sealed_partition_count: int
    raw_reader_pending: bool
    active_compute_workers: int
    starting_compute_workers: int
    pending_compute_tasks: int
    inflight_unrealized_compute_memory_bytes: int
    dispatch_stopped: bool = False
    worker4_authority: AdaptiveFixedWorkloadBenchmarkAuthorityV1 | None = None
    schema_version: str = _REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _REQUEST_SCHEMA:
            raise ValueError("unsupported L2 topology request")
        for name in ("experiment_spec_hash", "workload_hash"):
            object.__setattr__(
                self,
                name,
                cast(str, require_sha256(getattr(self, name), name=name)),
            )
        if self.sealed_partition_catalog_hash is not None:
            object.__setattr__(
                self,
                "sealed_partition_catalog_hash",
                cast(
                    str,
                    require_sha256(
                        self.sealed_partition_catalog_hash,
                        name="sealed_partition_catalog_hash",
                    ),
                ),
            )
        for name, maximum in (
            ("sealed_partition_count", None),
            ("active_compute_workers", 4),
            ("starting_compute_workers", 4),
            ("pending_compute_tasks", None),
            ("inflight_unrealized_compute_memory_bytes", None),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
            if maximum is not None and value > maximum:
                raise ValueError(f"{name} exceeds {maximum}")
        if self.active_compute_workers + self.starting_compute_workers > 4:
            raise ValueError("compute worker total exceeds four")
        required_commitment = (
            self.starting_compute_workers * 5 * _GIB // 2
        )
        if self.inflight_unrealized_compute_memory_bytes < required_commitment:
            raise ValueError("compute memory commitment does not cover starters")
        if (self.sealed_partition_count == 0) != (
            self.sealed_partition_catalog_hash is None
        ):
            raise ValueError("sealed partition count and catalog binding differ")
        for name in ("raw_reader_pending", "dispatch_stopped"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be an exact boolean")
        if (
            self.worker4_authority is not None
            and type(self.worker4_authority)
            is not AdaptiveFixedWorkloadBenchmarkAuthorityV1
        ):
            raise TypeError("worker4 authority must be exact benchmark authority")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_spec_hash": self.experiment_spec_hash,
            "workload_hash": self.workload_hash,
            "sealed_partition_catalog_hash": self.sealed_partition_catalog_hash,
            "sealed_partition_count": self.sealed_partition_count,
            "raw_reader_pending": self.raw_reader_pending,
            "active_compute_workers": self.active_compute_workers,
            "starting_compute_workers": self.starting_compute_workers,
            "pending_compute_tasks": self.pending_compute_tasks,
            "inflight_unrealized_compute_memory_bytes": (
                self.inflight_unrealized_compute_memory_bytes
            ),
            "dispatch_stopped": self.dispatch_stopped,
            "worker4_authority": (
                None
                if self.worker4_authority is None
                else self.worker4_authority.to_dict()
            ),
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class L2CampaignTopologyDecisionV1:
    policy_hash: str
    request_hash: str
    observation_hash: str
    action: L2CampaignTopologyAction
    reason: str
    effective_compute_worker_ceiling: int
    raw_reader_limit: int
    admit_raw_reader_preview: bool
    admit_compute_preview: bool
    terminate_running_processes: bool
    next_adaptive_state: AdaptiveDispatchStateV1
    adaptive_evaluation_hash: str | None
    sealed_partition_catalog_hash: str | None
    single_raw_reader_enforced: bool = True
    compute_reads_raw: bool = False
    admission_token_consumed: bool = False
    grants_spawn_authority: bool = False
    canonical_reducer_single_writer: bool = True
    research_only: bool = True
    production_ready: bool = False
    schema_version: str = _DECISION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _DECISION_SCHEMA:
            raise ValueError("unsupported L2 topology decision")
        for name in ("policy_hash", "request_hash", "observation_hash"):
            require_sha256(getattr(self, name), name=name)
        if self.adaptive_evaluation_hash is not None:
            require_sha256(
                self.adaptive_evaluation_hash,
                name="adaptive_evaluation_hash",
            )
        if self.sealed_partition_catalog_hash is not None:
            require_sha256(
                self.sealed_partition_catalog_hash,
                name="sealed_partition_catalog_hash",
            )
        if type(self.action) is not L2CampaignTopologyAction:
            raise TypeError("L2 topology action must be exact enum")
        if type(self.reason) is not str or not self.reason:
            raise ValueError("L2 topology reason must be non-empty text")
        if (
            type(self.effective_compute_worker_ceiling) is not int
            or not 0 <= self.effective_compute_worker_ceiling <= 4
            or self.raw_reader_limit != 1
        ):
            raise ValueError("L2 topology ceilings differ")
        if type(self.next_adaptive_state) is not AdaptiveDispatchStateV1:
            raise TypeError("exact adaptive state required")
        preview_raw = self.action is L2CampaignTopologyAction.PREVIEW_RAW_READER
        preview_compute = self.action is L2CampaignTopologyAction.PREVIEW_COMPUTE
        hard_stop = self.action is L2CampaignTopologyAction.HARD_STOP
        if (
            self.admit_raw_reader_preview is not preview_raw
            or self.admit_compute_preview is not preview_compute
            or self.terminate_running_processes is not hard_stop
        ):
            raise ValueError("L2 topology action flags differ")
        fixed: dict[str, object] = {
            "single_raw_reader_enforced": True,
            "compute_reads_raw": False,
            "admission_token_consumed": False,
            "grants_spawn_authority": False,
            "canonical_reducer_single_writer": True,
            "research_only": True,
            "production_ready": False,
        }
        for name, required in fixed.items():
            if getattr(self, name) is not required:
                raise ValueError("L2 topology decision assurance differs")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_hash": self.policy_hash,
            "request_hash": self.request_hash,
            "observation_hash": self.observation_hash,
            "action": self.action.value,
            "reason": self.reason,
            "effective_compute_worker_ceiling": (
                self.effective_compute_worker_ceiling
            ),
            "raw_reader_limit": self.raw_reader_limit,
            "admit_raw_reader_preview": self.admit_raw_reader_preview,
            "admit_compute_preview": self.admit_compute_preview,
            "terminate_running_processes": self.terminate_running_processes,
            "next_adaptive_state": self.next_adaptive_state.to_dict(),
            "adaptive_evaluation_hash": self.adaptive_evaluation_hash,
            "sealed_partition_catalog_hash": self.sealed_partition_catalog_hash,
            "single_raw_reader_enforced": self.single_raw_reader_enforced,
            "compute_reads_raw": self.compute_reads_raw,
            "admission_token_consumed": self.admission_token_consumed,
            "grants_spawn_authority": self.grants_spawn_authority,
            "canonical_reducer_single_writer": (
                self.canonical_reducer_single_writer
            ),
            "research_only": self.research_only,
            "production_ready": self.production_ready,
            "schema_version": self.schema_version,
        }


def evaluate_l2_campaign_topology_v1(
    policy: L2CampaignTopologyPolicyV1,
    adaptive_policy: AdaptiveDispatchPolicyV1,
    prior_state: AdaptiveDispatchStateV1,
    observation: AdaptiveResourceObservationV1,
    request: L2CampaignTopologyRequestV1,
) -> L2CampaignTopologyDecisionV1:
    """Preview exactly one reader or compute transition without authorizing it."""

    if type(policy) is not L2CampaignTopologyPolicyV1:
        raise TypeError("exact L2 topology policy required")
    if type(adaptive_policy) is not AdaptiveDispatchPolicyV1:
        raise TypeError("exact adaptive policy required")
    if type(prior_state) is not AdaptiveDispatchStateV1:
        raise TypeError("exact adaptive state required")
    if type(observation) is not AdaptiveResourceObservationV1:
        raise TypeError("exact adaptive observation required")
    if type(request) is not L2CampaignTopologyRequestV1:
        raise TypeError("exact L2 topology request required")

    if request.dispatch_stopped:
        return _decision(
            policy,
            request,
            observation,
            prior_state,
            action=L2CampaignTopologyAction.HARD_STOP,
            reason="dispatch_stopped",
            ceiling=0,
        )
    if not observation.metrics_complete:
        return _hold(
            policy, request, observation, prior_state, "incomplete_observation"
        )
    if observation.raw_reader_state is RawReaderState.UNKNOWN:
        return _hold(
            policy, request, observation, prior_state, "raw_reader_state_unknown"
        )

    raw_active = observation.raw_reader_state is RawReaderState.ACTIVE
    if request.raw_reader_pending and not raw_active:
        if request.active_compute_workers + request.starting_compute_workers > 3:
            return _hold(
                policy,
                request,
                observation,
                prior_state,
                "await_compute_slot_for_reader",
            )
        raw_reason = _raw_resource_hold_reason(policy, observation)
        if raw_reason is not None:
            return _hold(policy, request, observation, prior_state, raw_reason)
        return _decision(
            policy,
            request,
            observation,
            prior_state,
            action=L2CampaignTopologyAction.PREVIEW_RAW_READER,
            reason="raw_reader_priority_admitted",
            ceiling=3,
        )

    if request.pending_compute_tasks == 0:
        return _hold(policy, request, observation, prior_state, "no_pending_compute")
    if (
        request.sealed_partition_count == 0
        or request.sealed_partition_catalog_hash is None
    ):
        return _hold(
            policy,
            request,
            observation,
            prior_state,
            "sealed_partition_unavailable",
        )

    static_ceiling = 3 if raw_active else 4
    authority = None if raw_active else request.worker4_authority
    adaptive_request = AdaptiveDispatchRequestV1(
        experiment_spec_hash=request.experiment_spec_hash,
        workload_hash=request.workload_hash,
        active_workers=request.active_compute_workers,
        starting_workers=request.starting_compute_workers,
        pending_tasks=request.pending_compute_tasks,
        inflight_unrealized_memory_bytes=(
            request.inflight_unrealized_compute_memory_bytes
        ),
        next_task_memory_limit_bytes=(
            policy.compute_worker_memory_commitment_bytes
        ),
        resource_budget_maximum_parallel_tasks=static_ceiling,
        dispatch_stopped=False,
        worker4_authority=(
            None
            if authority is None
            else _worker4_activation_from_benchmark(
                authority,
                workload_hash=request.workload_hash,
                policy_hash=adaptive_policy.content_hash,
                sample_ordinal=observation.sample_ordinal,
            )
        ),
    )
    compute_observation = replace(
        observation,
        raw_reader_state=RawReaderState.INACTIVE,
    )
    evaluation = evaluate_adaptive_dispatch_v1(
        adaptive_policy,
        prior_state,
        compute_observation,
        adaptive_request,
        fixed_workload_benchmark_authority=authority,
    )
    action = (
        L2CampaignTopologyAction.PREVIEW_COMPUTE
        if evaluation.action is AdaptiveDispatchAction.ADMIT_ONE
        else (
            L2CampaignTopologyAction.HARD_STOP
            if evaluation.action is AdaptiveDispatchAction.HARD_STOP
            else L2CampaignTopologyAction.HOLD
        )
    )
    return _decision(
        policy,
        request,
        observation,
        evaluation.next_state,
        action=action,
        reason=evaluation.reason,
        ceiling=min(static_ceiling, evaluation.effective_worker_ceiling),
        adaptive_evaluation_hash=evaluation.content_hash,
    )


def _worker4_activation_from_benchmark(
    authority: AdaptiveFixedWorkloadBenchmarkAuthorityV1,
    *,
    workload_hash: str,
    policy_hash: str,
    sample_ordinal: int,
) -> AdaptiveWorker4AuthorityV1:
    return AdaptiveWorker4AuthorityV1(
        benchmark_workload_hash=workload_hash,
        activation_workload_hash=workload_hash,
        policy_hash=policy_hash,
        benchmark_authority_hash=authority.content_hash,
        benchmark_evidence_hash=authority.evidence.content_hash,
        activation_receipt_hash=cast(
            str,
            hash_json(
                {
                    "schema_version": "l2-worker4-activation-preview/v1",
                    "benchmark_authority_hash": authority.content_hash,
                    "sample_ordinal": sample_ordinal,
                    "workload_hash": workload_hash,
                }
            ),
        ),
        activation_first_sample_ordinal=sample_ordinal,
        activation_last_sample_ordinal=sample_ordinal,
    )


def _raw_resource_hold_reason(
    policy: L2CampaignTopologyPolicyV1,
    observation: AdaptiveResourceObservationV1,
) -> str | None:
    if observation.memory_pressure is not MemoryPressure.NORMAL:
        return "raw_reader_memory_pressure"
    if observation.swap_pressure is not SwapPressure.NORMAL:
        return "raw_reader_swap_pressure"
    available = observation.available_memory_bytes
    project = observation.project_tree_rss_bytes
    cpu = observation.system_cpu_busy_basis_points
    if available is None or project is None or cpu is None:
        return "raw_reader_metrics_unavailable"
    if (
        available - policy.raw_reader_memory_commitment_bytes
        < policy.foreground_reserve_bytes
    ):
        return "raw_reader_memory_reserve_insufficient"
    if (
        project + policy.raw_reader_memory_commitment_bytes
        > policy.project_hard_rss_bytes
    ):
        return "raw_reader_project_capacity_insufficient"
    if cpu >= 7_000:
        return "raw_reader_cpu_pressure"
    return None


def _hold(
    policy: L2CampaignTopologyPolicyV1,
    request: L2CampaignTopologyRequestV1,
    observation: AdaptiveResourceObservationV1,
    state: AdaptiveDispatchStateV1,
    reason: str,
) -> L2CampaignTopologyDecisionV1:
    raw_active = observation.raw_reader_state is RawReaderState.ACTIVE
    return _decision(
        policy,
        request,
        observation,
        state,
        action=L2CampaignTopologyAction.HOLD,
        reason=reason,
        ceiling=3 if raw_active else 4,
    )


def _decision(
    policy: L2CampaignTopologyPolicyV1,
    request: L2CampaignTopologyRequestV1,
    observation: AdaptiveResourceObservationV1,
    state: AdaptiveDispatchStateV1,
    *,
    action: L2CampaignTopologyAction,
    reason: str,
    ceiling: int,
    adaptive_evaluation_hash: str | None = None,
) -> L2CampaignTopologyDecisionV1:
    return L2CampaignTopologyDecisionV1(
        policy_hash=policy.content_hash,
        request_hash=request.content_hash,
        observation_hash=observation.content_hash,
        action=action,
        reason=reason,
        effective_compute_worker_ceiling=ceiling,
        raw_reader_limit=policy.maximum_raw_readers,
        admit_raw_reader_preview=(
            action is L2CampaignTopologyAction.PREVIEW_RAW_READER
        ),
        admit_compute_preview=(
            action is L2CampaignTopologyAction.PREVIEW_COMPUTE
        ),
        terminate_running_processes=(
            action is L2CampaignTopologyAction.HARD_STOP
        ),
        next_adaptive_state=state,
        adaptive_evaluation_hash=adaptive_evaluation_hash,
        sealed_partition_catalog_hash=request.sealed_partition_catalog_hash,
    )


__all__ = [
    "L2CampaignTopologyAction",
    "L2CampaignTopologyDecisionV1",
    "L2CampaignTopologyPolicyV1",
    "L2CampaignTopologyRequestV1",
    "evaluate_l2_campaign_topology_v1",
]
