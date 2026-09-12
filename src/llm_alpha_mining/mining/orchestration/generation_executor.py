from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence

from .evaluation_attempts import (
    CHECKPOINT_STAGES,
    AttemptLeaseConflict,
    AttemptRecord,
    AttemptStatus,
    AttemptUsage,
    CheckpointRecord,
    EvaluationAttemptStore,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class GenerationExecutionError(RuntimeError):
    pass


class CheckpointIntegrityError(GenerationExecutionError):
    pass


class SuccessfulAttemptBindingError(GenerationExecutionError):
    pass


class CandidateExecutionStatus(str, Enum):
    SUCCEEDED = "succeeded"
    SKIPPED_SUCCEEDED = "skipped_succeeded"
    FAILED = "failed"


class CandidateStateEventType(str, Enum):
    RESERVED = "reserved"
    STARTED = "started"
    RESUMED = "resumed"
    HEARTBEAT = "heartbeat"
    CHECKPOINT_VALIDATED = "checkpoint_validated"
    STAGE_STARTED = "stage_started"
    STAGE_CHECKPOINTED = "stage_checkpointed"
    SUCCEEDED = "succeeded"
    SKIPPED_SUCCEEDED = "skipped_succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class GenerationCandidate:
    candidate_id: str
    input_hash: str
    estimated_wall_seconds: float
    payload: object | None = None

    def __post_init__(self) -> None:
        if _SHA256_RE.fullmatch(self.input_hash) is None:
            raise ValueError("candidate input_hash must be a lowercase SHA-256 digest")
        if self.estimated_wall_seconds <= 0:
            raise ValueError("estimated_wall_seconds must be positive")


@dataclass(frozen=True, slots=True)
class StageArtifact:
    """A stage-produced file below the executor's artifact root."""

    location: str | Path


@dataclass(frozen=True, slots=True)
class VerifiedStageArtifact:
    stage: str
    path: Path
    checkpoint: CheckpointRecord


@dataclass(frozen=True, slots=True)
class CandidateStateEvent:
    event_type: CandidateStateEventType
    candidate_id: str
    attempt_id: int | None
    attempt_number: int | None
    worker_id: str
    timestamp: str
    stage: str | None = None
    failure_code: str | None = None


class CandidateStateBridge(Protocol):
    """Optional adapter from executor lifecycle events to CandidateState."""

    def __call__(self, event: CandidateStateEvent) -> None: ...


@dataclass(frozen=True)
class StageExecutionContext:
    candidate: GenerationCandidate
    attempt: AttemptRecord
    worker_id: str
    stage: str
    artifact_root: Path
    previous_artifacts: Mapping[str, VerifiedStageArtifact]
    _heartbeat: Callable[[], AttemptRecord]

    def heartbeat(self) -> AttemptRecord:
        """Renew the lease from inside a long-running injected stage."""

        return self._heartbeat()

    def default_output_path(self, suffix: str = ".bin") -> Path:
        if not suffix.startswith(".") or "/" in suffix or "\\" in suffix:
            raise ValueError("stage output suffix must be a simple extension")
        return (
            self.artifact_root
            / "attempts"
            / str(self.attempt.attempt_id)
            / f"{self.stage}{suffix}"
        )


StageRunner = Callable[[StageExecutionContext], StageArtifact | str | Path]


@dataclass(frozen=True, slots=True)
class CandidateExecutionSummary:
    candidate_id: str
    status: CandidateExecutionStatus
    attempt_id: int | None
    attempt_number: int | None
    resumed: bool
    executed_stages: tuple[str, ...]
    reused_stages: tuple[str, ...]
    result_hash: str | None
    failure_code: str | None
    error_type: str | None
    error_message: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status.value,
            "attempt_id": self.attempt_id,
            "attempt_number": self.attempt_number,
            "resumed": self.resumed,
            "executed_stages": list(self.executed_stages),
            "reused_stages": list(self.reused_stages),
            "result_hash": self.result_hash,
            "failure_code": self.failure_code,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass(frozen=True, slots=True)
class GenerationExecutionSummary:
    worker_id: str
    fail_fast: bool
    candidates: tuple[CandidateExecutionSummary, ...]
    usage: AttemptUsage

    @property
    def attempt_ids(self) -> tuple[int, ...]:
        return tuple(
            item.attempt_id for item in self.candidates if item.attempt_id is not None
        )

    @property
    def succeeded_count(self) -> int:
        return sum(
            item.status
            in {
                CandidateExecutionStatus.SUCCEEDED,
                CandidateExecutionStatus.SKIPPED_SUCCEEDED,
            }
            for item in self.candidates
        )

    @property
    def failed_count(self) -> int:
        return sum(
            item.status is CandidateExecutionStatus.FAILED for item in self.candidates
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "worker_id": self.worker_id,
            "fail_fast": self.fail_fast,
            "attempt_ids": list(self.attempt_ids),
            "candidate_count": len(self.candidates),
            "succeeded_count": self.succeeded_count,
            "failed_count": self.failed_count,
            "usage": {
                "attempts_reserved": self.usage.attempts_reserved,
                "attempts_active": self.usage.attempts_active,
                "attempts_succeeded": self.usage.attempts_succeeded,
                "attempts_failed": self.usage.attempts_failed,
                "attempts_expired": self.usage.attempts_expired,
                "actual_wall_seconds": self.usage.actual_wall_seconds,
            },
            "candidates": [item.to_dict() for item in self.candidates],
        }


class ResumableGenerationExecutor:
    """Run one immutable four-stage attempt per candidate with safe resume.

    Stage implementations, candidate-state transitions, time and artifact
    formats are injected.  The executor owns only leases, ordered checkpoints,
    file integrity verification and retry accounting.
    """

    def __init__(
        self,
        store: EvaluationAttemptStore,
        *,
        artifact_root: str | Path,
        worker_id: str,
        evaluator_hash: str,
        stage_runners: Mapping[str, StageRunner],
        lease_seconds: int,
        state_bridge: CandidateStateBridge | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if _SHA256_RE.fullmatch(evaluator_hash) is None:
            raise ValueError("evaluator_hash must be a lowercase SHA-256 digest")
        if not isinstance(lease_seconds, int) or isinstance(lease_seconds, bool):
            raise ValueError("lease_seconds must be a positive integer")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be a positive integer")
        missing = set(CHECKPOINT_STAGES).difference(stage_runners)
        extra = set(stage_runners).difference(CHECKPOINT_STAGES)
        if missing or extra:
            raise ValueError(
                f"stage_runners must exactly match {CHECKPOINT_STAGES}; "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
        self.store = store
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._artifact_root_resolved = self.artifact_root.resolve()
        self.worker_id = str(worker_id)
        self.evaluator_hash = evaluator_hash
        self.stage_runners = dict(stage_runners)
        self.lease_seconds = lease_seconds
        self.state_bridge = state_bridge
        self._now = now or (lambda: datetime.now(timezone.utc))

    def run(
        self,
        candidates: Sequence[GenerationCandidate],
        *,
        fail_fast: bool = True,
    ) -> GenerationExecutionSummary:
        summaries: list[CandidateExecutionSummary] = []
        for candidate in candidates:
            summary = self._execute_candidate(candidate, fail_fast=fail_fast)
            summaries.append(summary)
        return GenerationExecutionSummary(
            worker_id=self.worker_id,
            fail_fast=bool(fail_fast),
            candidates=tuple(summaries),
            usage=self.store.usage(),
        )

    def _execute_candidate(
        self,
        candidate: GenerationCandidate,
        *,
        fail_fast: bool,
    ) -> CandidateExecutionSummary:
        attempt: AttemptRecord | None = None
        current_stage: str | None = None
        resumed = False
        executed: list[str] = []
        reused: list[str] = []
        try:
            timestamp = self._timestamp()
            self.store.expire_stale(now=timestamp)
            attempts = self.store.list_attempts(candidate.candidate_id)
            successful = [
                item for item in attempts if item.status is AttemptStatus.SUCCEEDED
            ]
            if successful:
                attempt = successful[-1]
                self._require_attempt_binding(attempt, candidate, terminal=True)
                verified = self._validate_checkpoints(attempt, require_complete=True)
                reused.extend(verified)
                self._emit(
                    CandidateStateEventType.SKIPPED_SUCCEEDED,
                    candidate,
                    attempt,
                )
                return CandidateExecutionSummary(
                    candidate_id=candidate.candidate_id,
                    status=CandidateExecutionStatus.SKIPPED_SUCCEEDED,
                    attempt_id=attempt.attempt_id,
                    attempt_number=attempt.attempt_number,
                    resumed=False,
                    executed_stages=(),
                    reused_stages=tuple(reused),
                    result_hash=attempt.result_hash,
                    failure_code=None,
                    error_type=None,
                    error_message=None,
                )

            active = [
                item
                for item in attempts
                if item.status in {AttemptStatus.LEASED, AttemptStatus.RUNNING}
            ]
            if active:
                attempt = active[-1]
                if attempt.worker_id != self.worker_id:
                    raise AttemptLeaseConflict(
                        f"candidate {candidate.candidate_id} is leased by "
                        f"{attempt.worker_id} as attempt {attempt.attempt_id}"
                    )
                self._require_attempt_binding(attempt, candidate, terminal=False)
                resumed = True
                if attempt.status is AttemptStatus.LEASED:
                    attempt = self.store.start(
                        attempt.attempt_id,
                        worker_id=self.worker_id,
                        lease_seconds=self.lease_seconds,
                        now=self._timestamp(),
                    )
                    self._emit(CandidateStateEventType.STARTED, candidate, attempt)
                else:
                    attempt = self._heartbeat(candidate, attempt)
                self._emit(CandidateStateEventType.RESUMED, candidate, attempt)
            else:
                attempt = self.store.reserve(
                    candidate.candidate_id,
                    worker_id=self.worker_id,
                    input_hash=candidate.input_hash,
                    evaluator_hash=self.evaluator_hash,
                    lease_seconds=self.lease_seconds,
                    estimated_wall_seconds=candidate.estimated_wall_seconds,
                    now=timestamp,
                )
                self._emit(CandidateStateEventType.RESERVED, candidate, attempt)
                attempt = self.store.start(
                    attempt.attempt_id,
                    worker_id=self.worker_id,
                    lease_seconds=self.lease_seconds,
                    now=self._timestamp(),
                )
                self._emit(CandidateStateEventType.STARTED, candidate, attempt)

            verified = self._validate_checkpoints(attempt, require_complete=False)
            reused.extend(verified)
            verified_artifacts = dict(verified)
            for stage in CHECKPOINT_STAGES:
                current_stage = stage
                if stage in verified_artifacts:
                    continue
                attempt = self._heartbeat(candidate, attempt)
                self._emit(
                    CandidateStateEventType.STAGE_STARTED,
                    candidate,
                    attempt,
                    stage=stage,
                )
                context = StageExecutionContext(
                    candidate=candidate,
                    attempt=attempt,
                    worker_id=self.worker_id,
                    stage=stage,
                    artifact_root=self.artifact_root,
                    previous_artifacts=MappingProxyType(dict(verified_artifacts)),
                    _heartbeat=lambda c=candidate, a=attempt: self._heartbeat(c, a),
                )
                produced = self.stage_runners[stage](context)
                path, location, digest, size = self._inspect_produced_artifact(produced)
                if location in {
                    item.checkpoint.location for item in verified_artifacts.values()
                }:
                    raise CheckpointIntegrityError(
                        f"stage artifact location is already frozen:{location}"
                    )
                attempt = self._heartbeat(candidate, attempt)
                checkpoint = self.store.record_checkpoint(
                    attempt.attempt_id,
                    worker_id=self.worker_id,
                    stage=stage,
                    location=location,
                    sha256=digest,
                    size_bytes=size,
                    now=self._timestamp(),
                )
                artifact = VerifiedStageArtifact(
                    stage=stage,
                    path=path,
                    checkpoint=checkpoint,
                )
                verified_artifacts[stage] = artifact
                executed.append(stage)
                self._emit(
                    CandidateStateEventType.STAGE_CHECKPOINTED,
                    candidate,
                    attempt,
                    stage=stage,
                )

            current_stage = None
            verified_artifacts = self._validate_checkpoints(
                attempt,
                require_complete=True,
            )
            metrics = verified_artifacts[CHECKPOINT_STAGES[-1]].checkpoint
            attempt = self._heartbeat(candidate, attempt)
            finished = self.store.succeed(
                attempt.attempt_id,
                worker_id=self.worker_id,
                result_hash=metrics.sha256,
                actual_wall_seconds=self._elapsed_seconds(attempt),
                now=self._timestamp(),
            )
            self._emit(CandidateStateEventType.SUCCEEDED, candidate, finished)
            return CandidateExecutionSummary(
                candidate_id=candidate.candidate_id,
                status=CandidateExecutionStatus.SUCCEEDED,
                attempt_id=finished.attempt_id,
                attempt_number=finished.attempt_number,
                resumed=resumed,
                executed_stages=tuple(executed),
                reused_stages=tuple(reused),
                result_hash=finished.result_hash,
                failure_code=None,
                error_type=None,
                error_message=None,
            )
        except BaseException as exc:
            failure_code = self._failure_code(exc, current_stage)
            if attempt is not None:
                self._record_failure(candidate, attempt, failure_code)
            else:
                self._emit(
                    CandidateStateEventType.FAILED,
                    candidate,
                    None,
                    stage=current_stage,
                    failure_code=failure_code,
                )
            summary = CandidateExecutionSummary(
                candidate_id=candidate.candidate_id,
                status=CandidateExecutionStatus.FAILED,
                attempt_id=None if attempt is None else attempt.attempt_id,
                attempt_number=None if attempt is None else attempt.attempt_number,
                resumed=resumed,
                executed_stages=tuple(executed),
                reused_stages=tuple(reused),
                result_hash=None,
                failure_code=failure_code,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            if not isinstance(exc, Exception) or fail_fast:
                raise
            return summary

    def _validate_checkpoints(
        self,
        attempt: AttemptRecord,
        *,
        require_complete: bool,
    ) -> dict[str, VerifiedStageArtifact]:
        checkpoints = self.store.list_checkpoints(attempt.attempt_id)
        stages = tuple(item.stage for item in checkpoints)
        expected_prefix = CHECKPOINT_STAGES[: len(stages)]
        if stages != expected_prefix:
            raise CheckpointIntegrityError(
                f"attempt {attempt.attempt_id} checkpoint stages are not a prefix: "
                f"{stages}"
            )
        if require_complete and stages != CHECKPOINT_STAGES:
            raise CheckpointIntegrityError(
                f"successful attempt {attempt.attempt_id} lacks all checkpoints"
            )
        verified: dict[str, VerifiedStageArtifact] = {}
        for checkpoint in checkpoints:
            path = self._checkpoint_path(checkpoint)
            actual_size = path.stat().st_size
            if actual_size != checkpoint.size_bytes:
                raise CheckpointIntegrityError(
                    f"checkpoint size mismatch:{attempt.attempt_id}:"
                    f"{checkpoint.stage}:{actual_size}!={checkpoint.size_bytes}"
                )
            actual_digest = _hash_file(path)
            if actual_digest != checkpoint.sha256:
                raise CheckpointIntegrityError(
                    f"checkpoint hash mismatch:{attempt.attempt_id}:"
                    f"{checkpoint.stage}:{actual_digest}!={checkpoint.sha256}"
                )
            verified[checkpoint.stage] = VerifiedStageArtifact(
                stage=checkpoint.stage,
                path=path,
                checkpoint=checkpoint,
            )
            self._emit_checkpoint_validation(attempt, checkpoint.stage)
        return verified

    def _inspect_produced_artifact(
        self,
        artifact: StageArtifact | str | Path,
    ) -> tuple[Path, str, str, int]:
        raw = artifact.location if isinstance(artifact, StageArtifact) else artifact
        path = Path(raw)
        if path.is_symlink():
            raise CheckpointIntegrityError("stage artifact may not be a symbolic link")
        if path.is_absolute():
            resolved = path.resolve()
        else:
            if ".." in path.parts:
                raise CheckpointIntegrityError("stage artifact path contains '..'")
            resolved = (self.artifact_root / path).resolve()
        try:
            relative = resolved.relative_to(self._artifact_root_resolved)
        except ValueError as exc:
            raise CheckpointIntegrityError(
                "stage artifact must remain below artifact_root"
            ) from exc
        if not resolved.is_file():
            raise CheckpointIntegrityError(
                f"stage artifact is not a regular file:{relative.as_posix()}"
            )
        location = relative.as_posix()
        return resolved, location, _hash_file(resolved), resolved.stat().st_size

    def _checkpoint_path(self, checkpoint: CheckpointRecord) -> Path:
        raw = Path(checkpoint.location)
        if raw.is_absolute() or ".." in raw.parts:
            raise CheckpointIntegrityError(
                f"unsafe checkpoint location:{checkpoint.location}"
            )
        unresolved = self.artifact_root / raw
        if unresolved.is_symlink():
            raise CheckpointIntegrityError(
                f"checkpoint may not be a symbolic link:{checkpoint.location}"
            )
        path = unresolved.resolve()
        try:
            path.relative_to(self._artifact_root_resolved)
        except ValueError as exc:
            raise CheckpointIntegrityError(
                f"checkpoint escapes artifact root:{checkpoint.location}"
            ) from exc
        if not path.is_file():
            raise CheckpointIntegrityError(
                f"checkpoint file missing:{checkpoint.location}"
            )
        return path

    def _require_attempt_binding(
        self,
        attempt: AttemptRecord,
        candidate: GenerationCandidate,
        *,
        terminal: bool,
    ) -> None:
        if (
            attempt.input_hash == candidate.input_hash
            and attempt.evaluator_hash == self.evaluator_hash
        ):
            return
        message = (
            f"attempt binding mismatch for {candidate.candidate_id}:"
            f"{attempt.attempt_id}"
        )
        if terminal:
            raise SuccessfulAttemptBindingError(message)
        raise AttemptLeaseConflict(message)

    def _heartbeat(
        self,
        candidate: GenerationCandidate,
        attempt: AttemptRecord,
    ) -> AttemptRecord:
        renewed = self.store.heartbeat(
            attempt.attempt_id,
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
            now=self._timestamp(),
        )
        self._emit(CandidateStateEventType.HEARTBEAT, candidate, renewed)
        return renewed

    def _emit_checkpoint_validation(
        self,
        attempt: AttemptRecord,
        stage: str,
    ) -> None:
        if self.state_bridge is None:
            return
        self.state_bridge(
            CandidateStateEvent(
                event_type=CandidateStateEventType.CHECKPOINT_VALIDATED,
                candidate_id=attempt.candidate_id,
                attempt_id=attempt.attempt_id,
                attempt_number=attempt.attempt_number,
                worker_id=self.worker_id,
                timestamp=self._timestamp().isoformat(timespec="microseconds"),
                stage=stage,
            )
        )

    def _emit(
        self,
        event_type: CandidateStateEventType,
        candidate: GenerationCandidate,
        attempt: AttemptRecord | None,
        *,
        stage: str | None = None,
        failure_code: str | None = None,
    ) -> None:
        if self.state_bridge is None:
            return
        self.state_bridge(
            CandidateStateEvent(
                event_type=event_type,
                candidate_id=candidate.candidate_id,
                attempt_id=None if attempt is None else attempt.attempt_id,
                attempt_number=None if attempt is None else attempt.attempt_number,
                worker_id=self.worker_id,
                timestamp=self._timestamp().isoformat(timespec="microseconds"),
                stage=stage,
                failure_code=failure_code,
            )
        )

    def _record_failure(
        self,
        candidate: GenerationCandidate,
        attempt: AttemptRecord,
        failure_code: str,
    ) -> None:
        latest = self.store.get_attempt(attempt.attempt_id)
        if latest.status in {AttemptStatus.LEASED, AttemptStatus.RUNNING}:
            try:
                latest = self.store.fail(
                    attempt.attempt_id,
                    worker_id=self.worker_id,
                    failure_code=failure_code,
                    actual_wall_seconds=self._elapsed_seconds(latest),
                    now=self._timestamp(),
                )
            except BaseException:
                latest = self.store.get_attempt(attempt.attempt_id)
        try:
            self._emit(
                CandidateStateEventType.FAILED,
                candidate,
                latest,
                failure_code=failure_code,
            )
        except BaseException:
            pass

    def _elapsed_seconds(self, attempt: AttemptRecord) -> float:
        start = datetime.fromisoformat(attempt.started_at or attempt.reserved_at)
        return max(0.0, (self._timestamp() - start).total_seconds())

    def _timestamp(self) -> datetime:
        value = self._now()
        if value.tzinfo is None:
            raise ValueError("executor clock must return timezone-aware timestamps")
        return value.astimezone(timezone.utc)

    @staticmethod
    def _failure_code(exc: BaseException, stage: str | None) -> str:
        if isinstance(exc, KeyboardInterrupt):
            return "keyboard_interrupt"
        if isinstance(exc, CheckpointIntegrityError):
            return "checkpoint_integrity_error"
        if isinstance(exc, AttemptLeaseConflict):
            return "attempt_lease_conflict"
        prefix = "executor" if stage is None else f"stage_{stage}"
        error = re.sub(r"[^a-z0-9_.:-]+", "_", type(exc).__name__.lower()).strip("_")
        return f"{prefix}_{error}"[:128]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CandidateExecutionStatus",
    "CandidateExecutionSummary",
    "CandidateStateBridge",
    "CandidateStateEvent",
    "CandidateStateEventType",
    "CheckpointIntegrityError",
    "GenerationCandidate",
    "GenerationExecutionError",
    "GenerationExecutionSummary",
    "ResumableGenerationExecutor",
    "StageArtifact",
    "StageExecutionContext",
    "StageRunner",
    "SuccessfulAttemptBindingError",
    "VerifiedStageArtifact",
]
