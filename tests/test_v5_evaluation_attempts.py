from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from factor_production.v5.orchestration.evaluation_attempts import (
    CHECKPOINT_STAGES,
    AttemptBudgetExceeded,
    AttemptLeaseConflict,
    AttemptStatus,
    EvaluationAttemptError,
    EvaluationAttemptStore,
    completed_candidate_ids,
)


H1 = "1" * 64
H2 = "2" * 64
H3 = "3" * 64
H4 = "4" * 64
NOW = datetime(2026, 7, 16, 8, 0, tzinfo=timezone.utc)


def _store(tmp_path, *, limit: int = 3) -> EvaluationAttemptStore:
    return EvaluationAttemptStore(
        tmp_path / "attempts.sqlite3",
        run_id="run-v5",
        protocol_hash=H1,
        workspace_descriptor_hash=H2,
        max_evaluations=limit,
    )


def _reserve(store: EvaluationAttemptStore, candidate: str = "alpha-1"):
    return store.reserve(
        candidate,
        worker_id="worker-1",
        input_hash=H3,
        evaluator_hash=H4,
        lease_seconds=60,
        estimated_wall_seconds=30,
        now=NOW,
    )


def test_store_binding_is_immutable(tmp_path) -> None:
    _store(tmp_path).close()
    with pytest.raises(EvaluationAttemptError, match="binding differs"):
        EvaluationAttemptStore(
            tmp_path / "attempts.sqlite3",
            run_id="run-v5",
            protocol_hash=H1,
            workspace_descriptor_hash=H2,
            max_evaluations=4,
        )


def test_reservation_is_atomic_and_one_active_per_candidate(tmp_path) -> None:
    with _store(tmp_path) as store:
        attempt = _reserve(store)
        assert attempt.status is AttemptStatus.LEASED
        assert attempt.attempt_number == 1
        with pytest.raises(AttemptLeaseConflict):
            _reserve(store)
        assert store.usage().attempts_reserved == 1


def test_failed_and_expired_attempts_still_consume_budget(tmp_path) -> None:
    with _store(tmp_path, limit=2) as store:
        first = _reserve(store)
        store.fail(
            first.attempt_id,
            worker_id="worker-1",
            failure_code="runtime_error",
            actual_wall_seconds=2,
            now=NOW + timedelta(seconds=1),
        )
        second = store.reserve(
            "alpha-2",
            worker_id="worker-1",
            input_hash=H3,
            evaluator_hash=H4,
            lease_seconds=1,
            estimated_wall_seconds=5,
            now=NOW + timedelta(seconds=2),
        )
        assert store.expire_stale(now=NOW + timedelta(seconds=4)) == 1
        assert store.get_attempt(second.attempt_id).status is AttemptStatus.EXPIRED
        with pytest.raises(AttemptBudgetExceeded):
            store.reserve(
                "alpha-3",
                worker_id="worker-1",
                input_hash=H3,
                evaluator_hash=H4,
                lease_seconds=60,
                estimated_wall_seconds=5,
                now=NOW + timedelta(seconds=5),
            )


def test_expired_attempt_can_be_retried_with_new_number(tmp_path) -> None:
    with _store(tmp_path) as store:
        first = _reserve(store)
        store.expire_stale(now=NOW + timedelta(seconds=61))
        second = store.reserve(
            "alpha-1",
            worker_id="worker-2",
            input_hash=H3,
            evaluator_hash=H4,
            lease_seconds=60,
            estimated_wall_seconds=30,
            now=NOW + timedelta(seconds=62),
        )
        assert first.attempt_id != second.attempt_id
        assert second.attempt_number == 2


def test_heartbeat_requires_owner_and_running_state(tmp_path) -> None:
    with _store(tmp_path) as store:
        attempt = _reserve(store)
        with pytest.raises(AttemptLeaseConflict):
            store.heartbeat(
                attempt.attempt_id,
                worker_id="worker-1",
                lease_seconds=60,
                now=NOW + timedelta(seconds=1),
            )
        store.start(
            attempt.attempt_id,
            worker_id="worker-1",
            lease_seconds=60,
            now=NOW + timedelta(seconds=1),
        )
        with pytest.raises(AttemptLeaseConflict):
            store.heartbeat(
                attempt.attempt_id,
                worker_id="worker-2",
                lease_seconds=60,
                now=NOW + timedelta(seconds=2),
            )
        renewed = store.heartbeat(
            attempt.attempt_id,
            worker_id="worker-1",
            lease_seconds=120,
            now=NOW + timedelta(seconds=2),
        )
        assert renewed.status is AttemptStatus.RUNNING


def test_checkpoints_are_ordered_immutable_and_path_safe(tmp_path) -> None:
    with _store(tmp_path) as store:
        attempt = _reserve(store)
        store.start(
            attempt.attempt_id,
            worker_id="worker-1",
            lease_seconds=60,
            now=NOW + timedelta(seconds=1),
        )
        with pytest.raises(EvaluationAttemptError, match="in order"):
            store.record_checkpoint(
                attempt.attempt_id,
                worker_id="worker-1",
                stage="neutral_signal",
                location="artifacts/neutral.parquet",
                sha256=H1,
                size_bytes=1,
                now=NOW + timedelta(seconds=2),
            )
        with pytest.raises(ValueError, match="relative"):
            store.record_checkpoint(
                attempt.attempt_id,
                worker_id="worker-1",
                stage="raw_signal",
                location="../escape",
                sha256=H1,
                size_bytes=1,
                now=NOW + timedelta(seconds=2),
            )
        store.record_checkpoint(
            attempt.attempt_id,
            worker_id="worker-1",
            stage="raw_signal",
            location="artifacts/raw.parquet",
            sha256=H1,
            size_bytes=1,
            now=NOW + timedelta(seconds=2),
        )
        with pytest.raises(EvaluationAttemptError, match="already frozen"):
            store.record_checkpoint(
                attempt.attempt_id,
                worker_id="worker-1",
                stage="raw_signal",
                location="artifacts/raw2.parquet",
                sha256=H2,
                size_bytes=2,
                now=NOW + timedelta(seconds=3),
            )


def test_success_requires_all_checkpoints(tmp_path) -> None:
    with _store(tmp_path) as store:
        attempt = _reserve(store)
        store.start(
            attempt.attempt_id,
            worker_id="worker-1",
            lease_seconds=60,
            now=NOW + timedelta(seconds=1),
        )
        with pytest.raises(EvaluationAttemptError, match="four"):
            store.succeed(
                attempt.attempt_id,
                worker_id="worker-1",
                result_hash=H1,
                actual_wall_seconds=10,
                now=NOW + timedelta(seconds=2),
            )
        for index, stage in enumerate(CHECKPOINT_STAGES, start=2):
            store.record_checkpoint(
                attempt.attempt_id,
                worker_id="worker-1",
                stage=stage,
                location=f"artifacts/{stage}.bin",
                sha256=str(index) * 64,
                size_bytes=index,
                now=NOW + timedelta(seconds=index),
            )
        finished = store.succeed(
            attempt.attempt_id,
            worker_id="worker-1",
            result_hash=H1,
            actual_wall_seconds=10,
            now=NOW + timedelta(seconds=10),
        )
        assert finished.status is AttemptStatus.SUCCEEDED
        assert store.usage().actual_wall_seconds == 10
        assert completed_candidate_ids(store.list_attempts()) == frozenset({"alpha-1"})
        assert store.verify_integrity() == ()


def test_reopen_preserves_attempt_and_checkpoint_state(tmp_path) -> None:
    store = _store(tmp_path)
    attempt = _reserve(store)
    store.start(
        attempt.attempt_id,
        worker_id="worker-1",
        lease_seconds=60,
        now=NOW + timedelta(seconds=1),
    )
    store.record_checkpoint(
        attempt.attempt_id,
        worker_id="worker-1",
        stage="raw_signal",
        location="artifacts/raw.parquet",
        sha256=H1,
        size_bytes=4,
        now=NOW + timedelta(seconds=2),
    )
    store.close()
    with _store(tmp_path) as reopened:
        assert reopened.get_attempt(attempt.attempt_id).status is AttemptStatus.RUNNING
        assert reopened.list_checkpoints(attempt.attempt_id)[0].stage == "raw_signal"
