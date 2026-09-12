from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, cast

from alpha_research.core.hashing import hash_json, require_sha256
from alpha_research.experiments.receipt import ExperimentReceipt
from alpha_research.experiments.registry import (
    ExperimentRegistry,
    ExperimentRegistryConflict,
)
from alpha_research.experiments.spec import ExperimentProfile, ExperimentSpec
from alpha_research.observability import StageTelemetry, TelemetryStore
from alpha_research.orchestration.runtime import (
    AttemptLeaseConflict,
    AttemptStatus,
    ExperimentRuntime,
    RetryNotReady,
    RuntimeUsage,
    StageAttempt,
    reported_usage,
)
from factor_production.v5.artifacts import (
    ArtifactRecord as StoredArtifactRecord,
    ArtifactStore,
)


@dataclass(frozen=True, slots=True)
class StageContext:
    experiment_spec_hash: str
    stage: str
    attempt_id: int
    attempt_number: int
    input_hash: str
    prior_result_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        require_sha256(self.experiment_spec_hash, name="stage context experiment hash")
        require_sha256(self.input_hash, name="stage context input hash")
        prior = MappingProxyType(dict(self.prior_result_hashes))
        for digest in prior.values():
            require_sha256(digest, name="stage context prior result hash")
        object.__setattr__(self, "prior_result_hashes", prior)


@dataclass(frozen=True, slots=True)
class StageOutput:
    logical_name: str
    kind: str
    payload: bytes
    media_type: str
    parent_hashes: tuple[str, ...]
    usage: RuntimeUsage
    input_rows: int
    output_rows: int
    symbols: int
    disk_read_bytes: int
    cache_hits: int
    cache_misses: int
    worker_count: int
    attributes: Mapping[str, str | int | float | bool | None]

    def __post_init__(self) -> None:
        if (
            not self.logical_name.strip()
            or not self.kind.strip()
            or not self.media_type.strip()
        ):
            raise ValueError("stage output name/kind/media_type is required")
        if not isinstance(self.payload, bytes):
            raise TypeError("stage output payload must be bytes")
        if len(set(self.parent_hashes)) != len(self.parent_hashes):
            raise ValueError("stage output parents must be unique")
        for digest in self.parent_hashes:
            require_sha256(digest, name="stage output parent hash")
        for name in (
            "input_rows",
            "output_rows",
            "symbols",
            "disk_read_bytes",
            "cache_hits",
            "cache_misses",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"stage output {name} must be non-negative")
        if not isinstance(self.worker_count, int) or self.worker_count <= 0:
            raise ValueError("stage output worker_count must be positive")


class StageHandler(Protocol):
    def __call__(self, context: StageContext) -> StageOutput: ...


class StageFailure(RuntimeError):
    def __init__(self, failure_code: str, *, usage: RuntimeUsage | None = None) -> None:
        if not _safe_code(failure_code):
            raise ValueError("stage failure_code is invalid")
        super().__init__("stage execution failed")
        self.failure_code = failure_code
        self.usage = usage or reported_usage(
            wall_seconds=0.0,
            cpu_seconds=0.0,
            peak_memory_bytes=0,
            disk_write_bytes=0,
        )


@dataclass(frozen=True, slots=True)
class ExperimentRunSummary:
    experiment_spec_hash: str
    status: str
    stage_result_hashes: Mapping[str, str]
    receipt_hash: str | None
    failed_stage: str | None = None
    failure_code: str | None = None

    def __post_init__(self) -> None:
        require_sha256(self.experiment_spec_hash, name="run summary experiment hash")
        if self.status not in {
            "completed",
            "awaiting_approval",
            "cancelled",
            "checkpointed",
            "failed",
        }:
            raise ValueError("run summary status is invalid")
        results = MappingProxyType(dict(self.stage_result_hashes))
        for digest in results.values():
            require_sha256(digest, name="run summary result hash")
        if self.receipt_hash is not None:
            require_sha256(self.receipt_hash, name="run summary receipt hash")
        object.__setattr__(self, "stage_result_hashes", results)


class _AttemptLeaseHeartbeat:
    """Renew one controller-owned stage lease on an isolated SQLite connection."""

    def __init__(
        self,
        *,
        runtime_path: Path,
        spec: ExperimentSpec,
        attempt_id: int,
        worker_id: str,
        lease_token: str,
        lease_seconds: int,
        clock: Callable[[], datetime],
    ) -> None:
        self._runtime_path = runtime_path
        self._spec = spec
        self._attempt_id = attempt_id
        self._worker_id = worker_id
        self._lease_token = lease_token
        self._lease_seconds = lease_seconds
        self._clock = clock
        self._interval_seconds = min(
            30.0,
            max(0.25, float(lease_seconds) / 3.0),
        )
        self._stop = Event()
        self._failure: BaseException | None = None
        self._thread = Thread(
            target=self._run,
            name=f"experiment-attempt-heartbeat-{attempt_id}",
            daemon=True,
        )
        self._started = False

    def start(self) -> None:
        if self._started:
            raise RuntimeError("attempt heartbeat already started")
        self._started = True
        self._thread.start()

    def stop(self) -> BaseException | None:
        if not self._started:
            return None
        self._stop.set()
        self._thread.join(timeout=10.0)
        if self._thread.is_alive() and self._failure is None:
            self._failure = RuntimeError("attempt heartbeat did not stop")
        return self._failure

    @property
    def failure(self) -> BaseException | None:
        return self._failure

    def _run(self) -> None:
        try:
            with ExperimentRuntime(self._runtime_path, self._spec) as runtime:
                while not self._stop.wait(self._interval_seconds):
                    runtime.heartbeat(
                        self._attempt_id,
                        worker_id=self._worker_id,
                        lease_token=self._lease_token,
                        lease_seconds=self._lease_seconds,
                        now=self._clock(),
                    )
        except BaseException as exc:  # pragma: no cover - asserted via outcome
            self._failure = exc
            self._stop.set()


class ExperimentController:
    """One SDK entry point for deterministic local execution and recovery."""

    def __init__(
        self,
        *,
        spec: ExperimentSpec,
        registry: ExperimentRegistry,
        runtime: ExperimentRuntime,
        telemetry: TelemetryStore,
        workspace_root: str | Path,
        handlers: Mapping[str, StageHandler],
        worker_id: str,
        lease_token: str,
        lease_seconds: int = 60,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.spec = spec
        self.registry = registry
        self.runtime = runtime
        self.telemetry = telemetry
        self.artifact_store = ArtifactStore(workspace_root)
        self.handlers = dict(handlers)
        self.worker_id = worker_id
        self.lease_token = lease_token
        self.lease_seconds = lease_seconds
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        if not isinstance(lease_seconds, int) or isinstance(lease_seconds, bool):
            raise TypeError("controller lease_seconds must be an integer")
        if lease_seconds <= 0:
            raise ValueError("controller lease_seconds must be positive")
        if set(self.handlers) != set(self.spec.stages):
            raise ValueError(
                "controller handlers must exactly cover ExperimentSpec stages"
            )
        if runtime.spec.content_hash != spec.content_hash:
            raise ValueError("controller runtime belongs to another experiment")
        if telemetry.experiment_spec_hash != spec.content_hash:
            raise ValueError("controller telemetry belongs to another experiment")

    def run(self) -> ExperimentRunSummary:
        """Run every enabled stage and preserve the existing terminal semantics."""

        return self._run(stop_after_stage=None)

    def run_until(self, stop_after_stage: str) -> ExperimentRunSummary:
        """Run through one non-terminal enabled stage, then checkpoint.

        The boundary is deliberately an operator-side controller input rather
        than part of :class:`ExperimentSpec`.  A successful checkpoint keeps
        the experiment resumable and never seals a receipt.  Calling
        :meth:`run` afterwards executes only the remaining stages.
        """

        if not isinstance(stop_after_stage, str):
            raise TypeError("run_until stop_after_stage must be a string")
        if stop_after_stage not in self.spec.stages:
            raise ValueError(
                "run_until stop_after_stage is not enabled by ExperimentSpec"
            )
        if stop_after_stage == self.spec.stages[-1]:
            raise ValueError(
                "run_until stop_after_stage must precede the terminal stage"
            )
        return self._run(stop_after_stage=stop_after_stage)

    def _run(self, *, stop_after_stage: str | None) -> ExperimentRunSummary:
        experiment = self.registry.get_experiment(self.spec.content_hash)
        if experiment.status == "completed":
            receipt = self.registry.load_receipt(self.spec.content_hash)
            return ExperimentRunSummary(
                experiment_spec_hash=self.spec.content_hash,
                status="completed",
                stage_result_hashes=dict(receipt.stage_result_hashes),
                receipt_hash=receipt.content_hash,
            )
        if experiment.status in {"failed", "cancelled"}:
            raise RuntimeError("cannot resume a terminal failed/cancelled experiment")
        if stop_after_stage is not None:
            stop_index = self.spec.stages.index(stop_after_stage)
            crossed = tuple(
                stage
                for stage in self.spec.stages[stop_index + 1 :]
                if self.runtime.list_attempts(stage)
            )
            if crossed:
                raise RuntimeError(
                    "run_until boundary already crossed:" + ",".join(crossed)
                )
        if experiment.status == "registered":
            self.registry.set_status(
                self.spec.content_hash,
                "running",
                reason_code="controller_started",
                at=self.clock(),
            )
        results: dict[str, str] = {}
        for stage in self.spec.stages:
            decision_at = self.clock()
            self.runtime.expire_stale(now=decision_at)
            input_hash = stage_input_hash(
                experiment_spec_hash=self.spec.content_hash,
                stage=stage,
                prior_result_hashes=results,
            )
            succeeded = self.runtime.successful_attempt(stage)
            if succeeded is not None:
                results[stage] = self._verify_replayed_success(
                    stage=stage,
                    expected_input_hash=input_hash,
                    attempt=succeeded,
                )
                if stage == stop_after_stage:
                    return ExperimentRunSummary(
                        experiment_spec_hash=self.spec.content_hash,
                        status="checkpointed",
                        stage_result_hashes=results,
                        receipt_hash=None,
                    )
                continue
            attempts = self.runtime.list_attempts(stage)
            prior_failure = self._resume_decision(
                stage=stage,
                attempts=attempts,
                results=results,
                at=decision_at,
            )
            if prior_failure is not None:
                return prior_failure
            next_number = len(attempts) + 1
            attempt = self.runtime.reserve(
                stage,
                worker_id=self.worker_id,
                lease_token=self.lease_token,
                idempotency_key=f"{stage}-attempt-{next_number}",
                input_hash=input_hash,
                lease_seconds=self.lease_seconds,
                estimated_wall_seconds=min(
                    self.spec.resource_budget.per_stage_timeout_seconds,
                    self.spec.resource_budget.maximum_wall_seconds,
                ),
                now=self.clock(),
            )
            started_at = self.clock()
            self.runtime.start(
                attempt.attempt_id,
                worker_id=self.worker_id,
                lease_token=self.lease_token,
                lease_seconds=self.lease_seconds,
                now=started_at,
            )
            context = StageContext(
                experiment_spec_hash=self.spec.content_hash,
                stage=stage,
                attempt_id=attempt.attempt_id,
                attempt_number=attempt.attempt_number,
                input_hash=input_hash,
                prior_result_hashes=results,
            )
            heartbeat = _AttemptLeaseHeartbeat(
                runtime_path=self.runtime.path,
                spec=self.spec,
                attempt_id=attempt.attempt_id,
                worker_id=self.worker_id,
                lease_token=self.lease_token,
                lease_seconds=self.lease_seconds,
                clock=self.clock,
            )
            heartbeat.start()
            try:
                output = self.handlers[stage](context)
                current = self.runtime.get_attempt(attempt.attempt_id)
                if heartbeat.failure is not None:
                    raise StageFailure("attempt_heartbeat_failed")
                if current.status is not AttemptStatus.RUNNING:
                    raise StageFailure(
                        current.failure_code or "attempt_heartbeat_failed"
                    )
                record = self.artifact_store.put_bytes(
                    output.logical_name,
                    output.payload,
                    media_type=output.media_type,
                    role="experiment_stage_result",
                )
                descriptor = {
                    "artifact_hash": record.sha256,
                    "logical_name": output.logical_name,
                    "kind": output.kind,
                    "location": record.location,
                    "media_type": output.media_type,
                    "size_bytes": record.size_bytes,
                    "stage": stage,
                    "attempt_id": attempt.attempt_id,
                    "input_hash": input_hash,
                }
                publication_at = self.clock()
                self.registry.prepare_stage_artifact_publication(
                    self.spec.content_hash,
                    attempt_id=attempt.attempt_id,
                    stage=stage,
                    artifact_hash=record.sha256,
                    logical_name=output.logical_name,
                    kind=output.kind,
                    location=record.location,
                    media_type=output.media_type,
                    size_bytes=record.size_bytes,
                    descriptor=descriptor,
                    parent_hashes=output.parent_hashes,
                    created_at=publication_at,
                )
                self.runtime.record_checkpoint(
                    attempt.attempt_id,
                    worker_id=self.worker_id,
                    lease_token=self.lease_token,
                    checkpoint_name="stage_result",
                    artifact_hash=record.sha256,
                    location=record.location,
                    now=publication_at,
                )
            except StageFailure as exc:
                finished_at = self.clock()
                heartbeat_failure = heartbeat.stop()
                return self._terminalize_attempt_failure(
                    stage=stage,
                    attempt_id=attempt.attempt_id,
                    results=results,
                    started_at=started_at,
                    at=finished_at,
                    failure_code=(
                        "attempt_heartbeat_failed"
                        if heartbeat_failure is not None
                        else exc.failure_code
                    ),
                    usage=exc.usage,
                )
            except Exception:
                finished_at = self.clock()
                heartbeat_failure = heartbeat.stop()
                generic_usage = reported_usage(
                    wall_seconds=max(0.0, (finished_at - started_at).total_seconds()),
                    cpu_seconds=0.0,
                    peak_memory_bytes=0,
                    disk_write_bytes=0,
                )
                return self._terminalize_attempt_failure(
                    stage=stage,
                    attempt_id=attempt.attempt_id,
                    results=results,
                    started_at=started_at,
                    at=finished_at,
                    failure_code=(
                        "attempt_heartbeat_failed"
                        if heartbeat_failure is not None
                        else "stage_handler_failed"
                    ),
                    usage=generic_usage,
                )
            except BaseException:
                heartbeat.stop()
                raise
            heartbeat_failure = heartbeat.stop()
            completion_at = self.clock()
            if heartbeat_failure is not None:
                heartbeat_usage = reported_usage(
                    wall_seconds=max(0.0, (completion_at - started_at).total_seconds()),
                    cpu_seconds=0.0,
                    peak_memory_bytes=0,
                    disk_write_bytes=0,
                )
                return self._terminalize_attempt_failure(
                    stage=stage,
                    attempt_id=attempt.attempt_id,
                    results=results,
                    started_at=started_at,
                    at=completion_at,
                    failure_code="attempt_heartbeat_failed",
                    usage=heartbeat_usage,
                )
            try:
                self.runtime.heartbeat(
                    attempt.attempt_id,
                    worker_id=self.worker_id,
                    lease_token=self.lease_token,
                    lease_seconds=self.lease_seconds,
                    now=completion_at,
                )
            except AttemptLeaseConflict:
                heartbeat_usage = reported_usage(
                    wall_seconds=max(0.0, (completion_at - started_at).total_seconds()),
                    cpu_seconds=0.0,
                    peak_memory_bytes=0,
                    disk_write_bytes=0,
                )
                return self._terminalize_attempt_failure(
                    stage=stage,
                    attempt_id=attempt.attempt_id,
                    results=results,
                    started_at=started_at,
                    at=completion_at,
                    failure_code="attempt_heartbeat_failed",
                    usage=heartbeat_usage,
                )
            try:
                completed = self.runtime.succeed(
                    attempt.attempt_id,
                    worker_id=self.worker_id,
                    lease_token=self.lease_token,
                    result_hash=record.sha256,
                    usage=output.usage,
                    now=completion_at,
                )
            except AttemptLeaseConflict:
                reconcile_at = self.clock()
                heartbeat_usage = reported_usage(
                    wall_seconds=max(
                        0.0, (reconcile_at - started_at).total_seconds()
                    ),
                    cpu_seconds=0.0,
                    peak_memory_bytes=0,
                    disk_write_bytes=0,
                )
                return self._terminalize_attempt_failure(
                    stage=stage,
                    attempt_id=attempt.attempt_id,
                    results=results,
                    started_at=started_at,
                    at=reconcile_at,
                    failure_code="attempt_lease_lost",
                    usage=heartbeat_usage,
                )
            if completed.status is not AttemptStatus.SUCCEEDED:
                return self._terminalize_attempt_failure(
                    stage=stage,
                    attempt_id=attempt.attempt_id,
                    results=results,
                    started_at=started_at,
                    at=completion_at,
                    failure_code=completed.failure_code or "resource_budget_exceeded",
                    usage=output.usage,
                )
            committed_at = self.clock()
            self.registry.commit_stage_artifact_publication(
                self.spec.content_hash,
                attempt_id=attempt.attempt_id,
                stage=stage,
                artifact_hash=record.sha256,
                committed_at=committed_at,
            )
            self._record_telemetry(
                stage,
                attempt.attempt_id,
                started_at,
                committed_at,
                output,
                output.usage,
                failure_code=None,
            )
            results[stage] = record.sha256
            if stage == stop_after_stage:
                return ExperimentRunSummary(
                    experiment_spec_hash=self.spec.content_hash,
                    status="checkpointed",
                    stage_result_hashes=results,
                    receipt_hash=None,
                )
        if stop_after_stage is not None:  # pragma: no cover - validated above
            raise RuntimeError("run_until stop stage was not reached")
        if self.spec.profile is ExperimentProfile.PRODUCTION_CANDIDATE:
            return ExperimentRunSummary(
                experiment_spec_hash=self.spec.content_hash,
                status="awaiting_approval",
                stage_result_hashes=results,
                receipt_hash=None,
            )
        return self._finalize(results, approval_hash=None, production_ready=False)

    def _terminalize_attempt_failure(
        self,
        *,
        stage: str,
        attempt_id: int,
        results: Mapping[str, str],
        started_at: datetime,
        at: datetime,
        failure_code: str,
        usage: RuntimeUsage,
    ) -> ExperimentRunSummary:
        """Freeze a failed attempt without overwriting an external terminal state."""

        current = self.runtime.get_attempt(attempt_id)
        if current.status is AttemptStatus.RUNNING:
            try:
                current = self.runtime.fail(
                    attempt_id,
                    worker_id=self.worker_id,
                    lease_token=self.lease_token,
                    failure_code=failure_code,
                    usage=usage,
                    now=at,
                )
            except AttemptLeaseConflict:
                current = self.runtime.get_attempt(attempt_id)
        if current.status not in {
            AttemptStatus.FAILED,
            AttemptStatus.EXPIRED,
            AttemptStatus.CANCELLED,
        }:
            raise RuntimeError(
                "failed stage attempt did not reach a terminal failure state:"
                f"{stage}:{current.status.value}"
            )
        self.registry.abandon_stage_artifact_publication(
            self.spec.content_hash,
            attempt_id=attempt_id,
            abandoned_at=at,
        )
        self._record_telemetry(
            stage,
            attempt_id,
            started_at,
            at,
            None,
            usage,
            failure_code=current.failure_code,
        )
        return self._failure_summary(
            stage=stage,
            attempt=current,
            results=results,
            at=at,
        )

    def _resume_decision(
        self,
        *,
        stage: str,
        attempts: tuple[StageAttempt, ...],
        results: Mapping[str, str],
        at: datetime,
    ) -> ExperimentRunSummary | None:
        if not attempts:
            return None
        latest = attempts[-1]
        if latest.status in {AttemptStatus.LEASED, AttemptStatus.RUNNING}:
            raise RuntimeError(f"stage already has live work:{stage}")
        if latest.status is AttemptStatus.SUCCEEDED:
            raise RuntimeError(f"successful stage lookup differs:{stage}")
        if latest.status is AttemptStatus.CANCELLED:
            return self._failure_summary(
                stage=stage,
                attempt=latest,
                results=results,
                at=at,
            )
        if latest.status not in {AttemptStatus.FAILED, AttemptStatus.EXPIRED}:
            raise RuntimeError(f"stage attempt status is unsupported:{stage}")
        if not self._retry_allowed(latest):
            return self._failure_summary(
                stage=stage,
                attempt=latest,
                results=results,
                at=at,
            )
        retry_not_before = latest.retry_not_before
        if retry_not_before is None:
            raise RuntimeError(f"retryable stage lacks retry timestamp:{stage}")
        if _parse(at.isoformat()) < _parse(retry_not_before):
            raise RetryNotReady(f"stage retry is not ready:{retry_not_before}")
        return None

    def _retry_allowed(self, attempt: StageAttempt) -> bool:
        failure_code = attempt.failure_code
        if not isinstance(failure_code, str) or not failure_code:
            raise RuntimeError(f"terminal attempt lacks failure code:{attempt.stage}")
        return (
            attempt.status in {AttemptStatus.FAILED, AttemptStatus.EXPIRED}
            and failure_code in self.spec.retry_policy.retryable_failure_codes
            and attempt.attempt_number
            < self.spec.retry_policy.maximum_attempts_per_stage
        )

    def _failure_summary(
        self,
        *,
        stage: str,
        attempt: StageAttempt,
        results: Mapping[str, str],
        at: datetime,
    ) -> ExperimentRunSummary:
        failure_code = attempt.failure_code
        if not isinstance(failure_code, str) or not failure_code:
            raise RuntimeError(f"terminal attempt lacks failure code:{stage}")
        cancelled = attempt.status is AttemptStatus.CANCELLED
        if cancelled or not self._retry_allowed(attempt):
            self._set_terminal_status(
                target_status="cancelled" if cancelled else "failed",
                at=at,
            )
        return ExperimentRunSummary(
            experiment_spec_hash=self.spec.content_hash,
            status="cancelled" if cancelled else "failed",
            stage_result_hashes=results,
            receipt_hash=None,
            failed_stage=stage,
            failure_code=failure_code,
        )

    def _set_terminal_status(self, *, target_status: str, at: datetime) -> None:
        experiment = self.registry.get_experiment(self.spec.content_hash)
        if experiment.status == target_status:
            return
        if experiment.status not in {"registered", "running"}:
            raise RuntimeError(
                "experiment terminal status differs:"
                f"{experiment.status}->{target_status}"
            )
        try:
            self.registry.set_status(
                self.spec.content_hash,
                target_status,
                reason_code=(
                    "controller_stage_cancelled"
                    if target_status == "cancelled"
                    else "controller_stage_failed"
                ),
                at=at,
            )
        except ExperimentRegistryConflict:
            current = self.registry.get_experiment(self.spec.content_hash)
            if current.status != target_status:
                raise

    def _verify_replayed_success(
        self,
        *,
        stage: str,
        expected_input_hash: str,
        attempt: StageAttempt,
    ) -> str:
        """Re-derive every binding before trusting a persisted success."""

        if not isinstance(attempt, StageAttempt):
            raise TypeError("successful attempt record type differs")
        if (
            attempt.status is not AttemptStatus.SUCCEEDED
            or attempt.experiment_spec_hash != self.spec.content_hash
            or attempt.stage != stage
        ):
            raise RuntimeError(f"successful attempt binding differs:{stage}")
        result_hash = attempt.result_hash
        if not isinstance(result_hash, str):
            raise RuntimeError("successful attempt lacks result hash")
        require_sha256(result_hash, name="successful attempt result hash")
        if attempt.input_hash != expected_input_hash:
            raise RuntimeError(f"replayed success input hash differs:{stage}")
        attempt_id = attempt.attempt_id
        try:
            descriptor = dict(
                self.registry.load_artifact_descriptor(
                    self.spec.content_hash,
                    result_hash,
                )
            )
        except KeyError:
            self.registry.commit_stage_artifact_publication(
                self.spec.content_hash,
                attempt_id=attempt_id,
                stage=stage,
                artifact_hash=result_hash,
                committed_at=self.clock(),
            )
            descriptor = dict(
                self.registry.load_artifact_descriptor(
                    self.spec.content_hash,
                    result_hash,
                )
            )
        expected_fields = {
            "artifact_hash",
            "logical_name",
            "kind",
            "location",
            "media_type",
            "size_bytes",
            "stage",
            "attempt_id",
            "input_hash",
        }
        if set(descriptor) != expected_fields:
            raise RuntimeError(f"replayed success artifact fields differ:{stage}")
        if (
            descriptor["artifact_hash"] != result_hash
            or descriptor["stage"] != stage
            or descriptor["attempt_id"] != attempt_id
            or descriptor["input_hash"] != expected_input_hash
        ):
            raise RuntimeError(f"replayed success artifact binding differs:{stage}")
        logical_name = _descriptor_text(
            descriptor["logical_name"], name="replayed artifact logical_name"
        )
        location = _descriptor_text(
            descriptor["location"], name="replayed artifact location"
        )
        media_type = _descriptor_text(
            descriptor["media_type"], name="replayed artifact media_type"
        )
        size_bytes = _descriptor_integer(
            descriptor["size_bytes"], name="replayed artifact size_bytes"
        )
        self.artifact_store.read_bytes(
            StoredArtifactRecord(
                logical_name=logical_name,
                location=location,
                sha256=result_hash,
                size_bytes=size_bytes,
                media_type=media_type,
                role="experiment_stage_result",
            )
        )
        checkpoints = self.runtime.list_checkpoints(attempt_id)
        if (
            len(checkpoints) != 1
            or checkpoints[0].checkpoint_name != "stage_result"
            or checkpoints[0].artifact_hash != result_hash
            or checkpoints[0].location != location
        ):
            raise RuntimeError(f"replayed success checkpoint differs:{stage}")
        return result_hash

    def finalize_production(self, *, approval_hash: str) -> ExperimentRunSummary:
        if self.spec.profile is not ExperimentProfile.PRODUCTION_CANDIDATE:
            raise ValueError(
                "only production-candidate experiments use approval finalization"
            )
        require_sha256(approval_hash, name="controller approval_hash")
        experiment = self.registry.get_experiment(self.spec.content_hash)
        if experiment.status == "completed":
            receipt = self.registry.load_receipt(self.spec.content_hash)
            if not receipt.production_ready or receipt.approval_hash != approval_hash:
                raise RuntimeError("completed production receipt differs from approval")
            return ExperimentRunSummary(
                experiment_spec_hash=self.spec.content_hash,
                status="completed",
                stage_result_hashes=dict(receipt.stage_result_hashes),
                receipt_hash=receipt.content_hash,
            )
        if experiment.status in {"failed", "cancelled"}:
            raise RuntimeError("cannot finalize a terminal failed/cancelled experiment")
        results = self._successful_results()
        if set(results) != set(self.spec.stages):
            raise RuntimeError(
                "production finalization requires every successful stage"
            )
        return self._finalize(
            results, approval_hash=approval_hash, production_ready=True
        )

    def _finalize(
        self,
        results: Mapping[str, str],
        *,
        approval_hash: str | None,
        production_ready: bool,
    ) -> ExperimentRunSummary:
        attempts = self.runtime.list_attempts()
        if not attempts:
            raise RuntimeError("experiment cannot finalize without attempts")
        started_at = min(_parse(item.reserved_at) for item in attempts)
        finished_at = self.clock()
        metrics_hash = next(
            (
                results[stage]
                for stage in (
                    "admission",
                    "report",
                    "backtest",
                    "factor_evaluation",
                )
                if stage in results
            ),
            None,
        )
        receipt = ExperimentReceipt(
            experiment_spec_hash=self.spec.content_hash,
            terminal_status="completed",
            component_bindings=dict(self.spec.component_bindings()),
            stage_result_hashes=dict(results),
            artifact_hashes=self.registry.artifact_hashes(self.spec.content_hash),
            metrics_artifact_hash=metrics_hash,
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            random_seed=self.spec.random_seed,
            code_snapshot_hash=self.spec.code_snapshot_hash,
            environment_hash=self.spec.environment_hash,
            production_ready=production_ready,
            approval_hash=approval_hash,
        )
        receipt_hash = self.registry.seal_receipt(receipt, sealed_at=finished_at)
        return ExperimentRunSummary(
            experiment_spec_hash=self.spec.content_hash,
            status="completed",
            stage_result_hashes=dict(results),
            receipt_hash=receipt_hash,
        )

    def _successful_results(self) -> dict[str, str]:
        results: dict[str, str] = {}
        for stage in self.spec.stages:
            attempt = self.runtime.successful_attempt(stage)
            if attempt is not None:
                expected_input_hash = stage_input_hash(
                    experiment_spec_hash=self.spec.content_hash,
                    stage=stage,
                    prior_result_hashes=results,
                )
                results[stage] = self._verify_replayed_success(
                    stage=stage,
                    expected_input_hash=expected_input_hash,
                    attempt=attempt,
                )
        return results

    def _record_telemetry(
        self,
        stage: str,
        attempt_id: int,
        started_at: datetime,
        finished_at: datetime,
        output: StageOutput | None,
        usage: RuntimeUsage,
        *,
        failure_code: str | None,
    ) -> None:
        telemetry = StageTelemetry(
            experiment_spec_hash=self.spec.content_hash,
            trace_id=f"experiment-{self.spec.content_hash[:16]}",
            span_id=f"attempt-{attempt_id}",
            parent_span_id=None,
            attempt_id=attempt_id,
            stage=stage,
            status="failed" if failure_code is not None else "succeeded",
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            input_rows=0 if output is None else output.input_rows,
            output_rows=0 if output is None else output.output_rows,
            symbols=0 if output is None else output.symbols,
            wall_seconds=usage.wall_seconds,
            cpu_seconds=usage.cpu_seconds,
            peak_memory_bytes=usage.peak_memory_bytes,
            disk_read_bytes=0 if output is None else output.disk_read_bytes,
            disk_write_bytes=usage.disk_write_bytes,
            cache_hits=0 if output is None else output.cache_hits,
            cache_misses=0 if output is None else output.cache_misses,
            worker_count=1 if output is None else output.worker_count,
            llm_calls=usage.llm_calls,
            llm_tokens=usage.llm_tokens,
            llm_cost_microusd=usage.llm_cost_microusd,
            failure_code=failure_code,
            attributes={} if output is None else output.attributes,
        )
        self.telemetry.record_span(telemetry)


class ReceiptReporter:
    """Render only authenticated receipt fields; never recompute research results."""

    @staticmethod
    def render(receipt: ExperimentReceipt) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "experiment_spec_hash": receipt.experiment_spec_hash,
                "receipt_hash": receipt.content_hash,
                "status": receipt.terminal_status,
                "production_ready": receipt.production_ready,
                "stage_result_hashes": dict(receipt.stage_result_hashes),
                "metrics_artifact_hash": receipt.metrics_artifact_hash,
                "artifact_count": len(receipt.artifact_hashes),
                "started_at": receipt.started_at,
                "finished_at": receipt.finished_at,
            }
        )


def _parse(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("controller timestamp must be timezone-aware")
    return timestamp.astimezone(timezone.utc)


def stage_input_hash(
    *,
    experiment_spec_hash: str,
    stage: str,
    prior_result_hashes: Mapping[str, str],
) -> str:
    """Return the canonical identity of one stage's immutable inputs.

    This helper is public because trusted downstream resolvers must verify
    persisted runtime attempts with exactly the same formula as the
    controller.  Keeping one implementation prevents two subtly different
    notions of stage-input authority.
    """

    return cast(
        str,
        hash_json(
            {
                "experiment_spec_hash": experiment_spec_hash,
                "stage": stage,
                "prior_result_hashes": dict(prior_result_hashes),
            }
        ),
    )


def _descriptor_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{name} differs")
    return value


def _descriptor_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(f"{name} differs")
    return value


def _safe_code(value: str) -> bool:
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-")
    return value[0].isalnum() and all(character in allowed for character in value)


__all__ = [
    "ExperimentController",
    "ExperimentRunSummary",
    "ReceiptReporter",
    "StageContext",
    "StageFailure",
    "StageHandler",
    "StageOutput",
]
