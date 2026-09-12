from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from llm_alpha_mining.mining.orchestration.evaluation_attempts import (
    CHECKPOINT_STAGES,
    AttemptStatus,
    EvaluationAttemptStore,
)
from llm_alpha_mining.mining.orchestration.generation_executor import (
    CandidateExecutionStatus,
    CandidateStateEvent,
    CandidateStateEventType,
    GenerationCandidate,
    ResumableGenerationExecutor,
    StageExecutionContext,
)


H1 = "1" * 64
H2 = "2" * 64
H3 = "3" * 64
NOW = datetime(2026, 7, 16, 10, 0, tzinfo=timezone.utc)


def _store(tmp_path: Path, *, limit: int = 20) -> EvaluationAttemptStore:
    return EvaluationAttemptStore(
        tmp_path / "attempts.sqlite3",
        run_id="run-v5",
        protocol_hash=H1,
        workspace_descriptor_hash=H2,
        max_evaluations=limit,
    )


def _candidate(candidate_id: str = "alpha-1") -> GenerationCandidate:
    return GenerationCandidate(
        candidate_id=candidate_id,
        input_hash=H3,
        estimated_wall_seconds=30.0,
        payload={"candidate": candidate_id},
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _writers(
    calls: list[tuple[str, str, tuple[str, ...]]],
    *,
    failures: dict[tuple[str, str], BaseException] | None = None,
):
    failures = failures or {}

    def make(stage: str):
        def run(context: StageExecutionContext) -> Path:
            calls.append(
                (
                    context.candidate.candidate_id,
                    stage,
                    tuple(context.previous_artifacts),
                )
            )
            failure = failures.get((context.candidate.candidate_id, stage))
            if failure is not None:
                raise failure
            context.heartbeat()
            path = context.default_output_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                f"{context.candidate.candidate_id}:{context.attempt.attempt_id}:{stage}".encode()
            )
            return path

        return run

    return {stage: make(stage) for stage in CHECKPOINT_STAGES}


def _manual_running_attempt(
    store: EvaluationAttemptStore,
    *,
    candidate_id: str = "alpha-1",
    lease_seconds: int = 300,
):
    attempt = store.reserve(
        candidate_id,
        worker_id="worker-1",
        input_hash=H3,
        evaluator_hash=H2,
        lease_seconds=lease_seconds,
        estimated_wall_seconds=30,
        now=NOW,
    )
    return store.start(
        attempt.attempt_id,
        worker_id="worker-1",
        lease_seconds=lease_seconds,
        now=NOW,
    )


def _record_manual_checkpoint(
    store: EvaluationAttemptStore,
    root: Path,
    attempt_id: int,
    stage: str,
    payload: bytes,
    *,
    seconds: int,
) -> Path:
    path = root / "attempts" / str(attempt_id) / f"{stage}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    store.record_checkpoint(
        attempt_id,
        worker_id="worker-1",
        stage=stage,
        location=path.relative_to(root).as_posix(),
        sha256=_sha(path),
        size_bytes=path.stat().st_size,
        now=NOW + timedelta(seconds=seconds),
    )
    return path


def test_full_run_checkpoints_in_order_and_success_is_idempotent(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, tuple[str, ...]]] = []
    events: list[CandidateStateEvent] = []
    root = tmp_path / "artifacts"
    with _store(tmp_path) as store:
        executor = ResumableGenerationExecutor(
            store,
            artifact_root=root,
            worker_id="worker-1",
            evaluator_hash=H2,
            stage_runners=_writers(calls),
            lease_seconds=300,
            state_bridge=events.append,
            now=lambda: NOW,
        )

        first = executor.run([_candidate()])
        second = executor.run([_candidate()])

        assert first.attempt_ids == (1,)
        assert first.succeeded_count == 1
        assert first.failed_count == 0
        assert first.candidates[0].executed_stages == CHECKPOINT_STAGES
        assert first.candidates[0].reused_stages == ()
        assert [item[1] for item in calls] == list(CHECKPOINT_STAGES)
        assert calls[0][2] == ()
        assert calls[1][2] == ("raw_signal",)
        assert calls[2][2] == ("raw_signal", "neutral_signal")
        assert calls[3][2] == (
            "raw_signal",
            "neutral_signal",
            "diagnostics",
        )
        assert store.get_attempt(1).status is AttemptStatus.SUCCEEDED
        assert (
            tuple(item.stage for item in store.list_checkpoints(1)) == CHECKPOINT_STAGES
        )

        assert second.candidates[0].status is CandidateExecutionStatus.SKIPPED_SUCCEEDED
        assert second.candidates[0].attempt_id == 1
        assert second.candidates[0].executed_stages == ()
        assert second.candidates[0].reused_stages == CHECKPOINT_STAGES
        assert len(calls) == 4
        assert store.usage().attempts_reserved == 1
        assert second.to_dict()["attempt_ids"] == [1]
        event_types = [event.event_type for event in events]
        assert CandidateStateEventType.RESERVED in event_types
        assert CandidateStateEventType.STARTED in event_types
        assert CandidateStateEventType.HEARTBEAT in event_types
        assert CandidateStateEventType.STAGE_CHECKPOINTED in event_types
        assert CandidateStateEventType.SUCCEEDED in event_types
        assert CandidateStateEventType.SKIPPED_SUCCEEDED in event_types


def test_same_worker_resumes_only_missing_stages_after_validating_checkpoint(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, tuple[str, ...]]] = []
    root = tmp_path / "artifacts"
    with _store(tmp_path) as store:
        attempt = _manual_running_attempt(store)
        _record_manual_checkpoint(
            store,
            root,
            attempt.attempt_id,
            "raw_signal",
            b"raw",
            seconds=1,
        )
        runners = _writers(calls)
        runners["raw_signal"] = lambda context: (_ for _ in ()).throw(
            AssertionError("raw stage must not rerun")
        )
        executor = ResumableGenerationExecutor(
            store,
            artifact_root=root,
            worker_id="worker-1",
            evaluator_hash=H2,
            stage_runners=runners,
            lease_seconds=300,
            now=lambda: NOW + timedelta(seconds=2),
        )

        summary = executor.run([_candidate()])

        candidate = summary.candidates[0]
        assert candidate.status is CandidateExecutionStatus.SUCCEEDED
        assert candidate.attempt_id == attempt.attempt_id
        assert candidate.resumed
        assert candidate.reused_stages == ("raw_signal",)
        assert candidate.executed_stages == CHECKPOINT_STAGES[1:]
        assert [item[1] for item in calls] == list(CHECKPOINT_STAGES[1:])


def test_running_attempt_with_all_checkpoints_finishes_without_stage_reexecution(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    with _store(tmp_path) as store:
        attempt = _manual_running_attempt(store)
        for index, stage in enumerate(CHECKPOINT_STAGES, start=1):
            _record_manual_checkpoint(
                store,
                root,
                attempt.attempt_id,
                stage,
                stage.encode(),
                seconds=index,
            )
        runners = {
            stage: lambda context, s=stage: (_ for _ in ()).throw(
                AssertionError(f"{s} must not rerun")
            )
            for stage in CHECKPOINT_STAGES
        }
        executor = ResumableGenerationExecutor(
            store,
            artifact_root=root,
            worker_id="worker-1",
            evaluator_hash=H2,
            stage_runners=runners,
            lease_seconds=300,
            now=lambda: NOW + timedelta(seconds=5),
        )

        result = executor.run([_candidate()]).candidates[0]

        assert result.status is CandidateExecutionStatus.SUCCEEDED
        assert result.resumed
        assert result.executed_stages == ()
        assert result.reused_stages == CHECKPOINT_STAGES
        assert (
            result.result_hash
            == store.get_checkpoint(attempt.attempt_id, "metrics").sha256
        )


def test_corrupt_checkpoint_fails_attempt_and_next_run_spends_new_attempt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    calls: list[tuple[str, str, tuple[str, ...]]] = []
    with _store(tmp_path) as store:
        attempt = _manual_running_attempt(store)
        raw_path = _record_manual_checkpoint(
            store,
            root,
            attempt.attempt_id,
            "raw_signal",
            b"aaaa",
            seconds=1,
        )
        raw_path.write_bytes(b"bbbb")
        executor = ResumableGenerationExecutor(
            store,
            artifact_root=root,
            worker_id="worker-1",
            evaluator_hash=H2,
            stage_runners=_writers(calls),
            lease_seconds=300,
            now=lambda: NOW + timedelta(seconds=2),
        )

        failed = executor.run([_candidate()], fail_fast=False).candidates[0]

        assert failed.status is CandidateExecutionStatus.FAILED
        assert failed.failure_code == "checkpoint_integrity_error"
        assert calls == []
        assert store.get_attempt(attempt.attempt_id).status is AttemptStatus.FAILED

        succeeded = executor.run([_candidate()]).candidates[0]
        assert succeeded.status is CandidateExecutionStatus.SUCCEEDED
        assert succeeded.attempt_number == 2
        assert succeeded.executed_stages == CHECKPOINT_STAGES
        usage = store.usage()
        assert usage.attempts_reserved == 2
        assert usage.attempts_failed == 1
        assert usage.attempts_succeeded == 1
        assert store.get_checkpoint(attempt.attempt_id, "raw_signal").sha256 != _sha(
            raw_path
        )


def test_expired_attempt_is_retained_and_retried_as_new_attempt(tmp_path: Path) -> None:
    calls: list[tuple[str, str, tuple[str, ...]]] = []
    with _store(tmp_path) as store:
        expired = _manual_running_attempt(store, lease_seconds=1)
        executor = ResumableGenerationExecutor(
            store,
            artifact_root=tmp_path / "artifacts",
            worker_id="worker-1",
            evaluator_hash=H2,
            stage_runners=_writers(calls),
            lease_seconds=300,
            now=lambda: NOW + timedelta(seconds=2),
        )

        result = executor.run([_candidate()]).candidates[0]

        assert store.get_attempt(expired.attempt_id).status is AttemptStatus.EXPIRED
        assert result.attempt_number == 2
        assert result.status is CandidateExecutionStatus.SUCCEEDED
        assert store.usage().attempts_reserved == 2
        assert store.usage().attempts_expired == 1


def test_keyboard_interrupt_is_recorded_then_reraised(tmp_path: Path) -> None:
    calls: list[tuple[str, str, tuple[str, ...]]] = []
    failures = {("alpha-1", "diagnostics"): KeyboardInterrupt()}
    with _store(tmp_path) as store:
        executor = ResumableGenerationExecutor(
            store,
            artifact_root=tmp_path / "artifacts",
            worker_id="worker-1",
            evaluator_hash=H2,
            stage_runners=_writers(calls, failures=failures),
            lease_seconds=300,
            now=lambda: NOW,
        )

        with pytest.raises(KeyboardInterrupt):
            executor.run([_candidate()], fail_fast=False)

        attempt = store.get_attempt(1)
        assert attempt.status is AttemptStatus.FAILED
        assert attempt.failure_code == "keyboard_interrupt"
        assert tuple(item.stage for item in store.list_checkpoints(1)) == (
            "raw_signal",
            "neutral_signal",
        )


def test_continue_mode_records_failure_and_runs_next_candidate(tmp_path: Path) -> None:
    calls: list[tuple[str, str, tuple[str, ...]]] = []
    failures = {("bad", "raw_signal"): RuntimeError("synthetic failure")}
    with _store(tmp_path) as store:
        executor = ResumableGenerationExecutor(
            store,
            artifact_root=tmp_path / "artifacts",
            worker_id="worker-1",
            evaluator_hash=H2,
            stage_runners=_writers(calls, failures=failures),
            lease_seconds=300,
            now=lambda: NOW,
        )

        summary = executor.run(
            [_candidate("bad"), _candidate("good")],
            fail_fast=False,
        )

        assert summary.attempt_ids == (1, 2)
        assert summary.failed_count == 1
        assert summary.succeeded_count == 1
        assert summary.candidates[0].failure_code == "stage_raw_signal_runtimeerror"
        assert summary.candidates[1].status is CandidateExecutionStatus.SUCCEEDED
        assert store.get_attempt(1).status is AttemptStatus.FAILED
        assert store.get_attempt(2).status is AttemptStatus.SUCCEEDED


def test_fail_fast_records_failure_and_does_not_start_later_candidate(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, tuple[str, ...]]] = []
    failures = {("bad", "raw_signal"): RuntimeError("synthetic failure")}
    with _store(tmp_path) as store:
        executor = ResumableGenerationExecutor(
            store,
            artifact_root=tmp_path / "artifacts",
            worker_id="worker-1",
            evaluator_hash=H2,
            stage_runners=_writers(calls, failures=failures),
            lease_seconds=300,
            now=lambda: NOW,
        )

        with pytest.raises(RuntimeError, match="synthetic failure"):
            executor.run([_candidate("bad"), _candidate("never-started")])

        assert store.get_attempt(1).status is AttemptStatus.FAILED
        assert store.list_attempts("never-started") == ()
