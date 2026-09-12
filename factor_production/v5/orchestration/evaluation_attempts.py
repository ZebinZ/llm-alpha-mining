from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Iterator, Sequence


ATTEMPT_STORE_SCHEMA_VERSION = "evaluation-attempt-store/v1"
CHECKPOINT_STAGES = ("raw_signal", "neutral_signal", "diagnostics", "metrics")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_CODE_RE = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,127}")


class EvaluationAttemptError(RuntimeError):
    pass


class AttemptBudgetExceeded(EvaluationAttemptError):
    pass


class AttemptLeaseConflict(EvaluationAttemptError):
    pass


class AttemptStatus(str, Enum):
    LEASED = "leased"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    attempt_id: int
    candidate_id: str
    attempt_number: int
    status: AttemptStatus
    worker_id: str
    input_hash: str
    evaluator_hash: str
    reserved_at: str
    started_at: str | None
    heartbeat_at: str
    lease_expires_at: str
    finished_at: str | None
    estimated_wall_seconds: float
    actual_wall_seconds: float | None
    result_hash: str | None
    failure_code: str | None


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    attempt_id: int
    stage: str
    location: str
    sha256: str
    size_bytes: int
    recorded_at: str


@dataclass(frozen=True, slots=True)
class AttemptUsage:
    attempts_reserved: int
    attempts_active: int
    attempts_succeeded: int
    attempts_failed: int
    attempts_expired: int
    actual_wall_seconds: float


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _require_digest(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_code(value: str, name: str) -> str:
    if not isinstance(value, str) or _SAFE_CODE_RE.fullmatch(value) is None:
        raise ValueError(f"{name} contains unsupported characters")
    return value


def _safe_relative_location(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("checkpoint location must not be empty")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError("checkpoint location must be a normalized relative path")
    return value


class EvaluationAttemptStore:
    """Persistent, fail-closed leases and per-candidate evaluation checkpoints.

    This intentionally lives in a separate SQLite file from the immutable V5
    campaign control database.  Older frozen workspaces can therefore still be
    verified byte-for-byte while new runners gain resumable evaluation attempts.
    Every reservation consumes the evaluation budget, including failed and
    expired work; a retry is a new, auditable attempt rather than an overwrite.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        protocol_hash: str,
        workspace_descriptor_hash: str,
        max_evaluations: int,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = _require_code(run_id, "run_id")
        self.protocol_hash = _require_digest(protocol_hash, "protocol_hash")
        self.workspace_descriptor_hash = _require_digest(
            workspace_descriptor_hash, "workspace_descriptor_hash"
        )
        if (
            not isinstance(max_evaluations, int)
            or isinstance(max_evaluations, bool)
            or max_evaluations <= 0
        ):
            raise ValueError("max_evaluations must be a positive integer")
        self.max_evaluations = max_evaluations
        self.connection = sqlite3.connect(self.path, timeout=5.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA busy_timeout = 5000")
        self._initialize_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "EvaluationAttemptStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        if self.connection.in_transaction:
            raise EvaluationAttemptError("nested transactions are not supported")
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            yield self.connection
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def _initialize_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS attempt_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evaluation_attempts (
                attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL CHECK(attempt_number > 0),
                status TEXT NOT NULL CHECK(status IN (
                    'leased', 'running', 'succeeded', 'failed', 'expired'
                )),
                worker_id TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                evaluator_hash TEXT NOT NULL,
                reserved_at TEXT NOT NULL,
                started_at TEXT,
                heartbeat_at TEXT NOT NULL,
                lease_expires_at TEXT NOT NULL,
                finished_at TEXT,
                estimated_wall_seconds REAL NOT NULL CHECK(estimated_wall_seconds > 0),
                actual_wall_seconds REAL,
                result_hash TEXT,
                failure_code TEXT,
                UNIQUE(candidate_id, attempt_number)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_attempt_per_candidate
            ON evaluation_attempts(candidate_id)
            WHERE status IN ('leased', 'running');
            CREATE TABLE IF NOT EXISTS evaluation_checkpoints (
                attempt_id INTEGER NOT NULL REFERENCES evaluation_attempts(attempt_id),
                stage TEXT NOT NULL CHECK(stage IN (
                    'raw_signal', 'neutral_signal', 'diagnostics', 'metrics'
                )),
                location TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
                recorded_at TEXT NOT NULL,
                PRIMARY KEY(attempt_id, stage)
            );
            """
        )
        expected = {
            "schema_version": ATTEMPT_STORE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "protocol_hash": self.protocol_hash,
            "workspace_descriptor_hash": self.workspace_descriptor_hash,
            "max_evaluations": str(self.max_evaluations),
        }
        existing = dict(
            self.connection.execute("SELECT key, value FROM attempt_metadata").fetchall()
        )
        if not existing:
            self.connection.executemany(
                "INSERT INTO attempt_metadata(key, value) VALUES (?, ?)",
                sorted(expected.items()),
            )
            self.connection.commit()
        elif existing != expected:
            raise EvaluationAttemptError(
                "attempt store binding differs from requested run/protocol/budget"
            )

    def _expire_locked(self, connection: sqlite3.Connection, now: datetime) -> int:
        now_text = _iso(now)
        cursor = connection.execute(
            """UPDATE evaluation_attempts
               SET status='expired', finished_at=?, failure_code='lease_expired'
               WHERE status IN ('leased', 'running') AND lease_expires_at <= ?""",
            (now_text, now_text),
        )
        return int(cursor.rowcount)

    def expire_stale(self, *, now: datetime | None = None) -> int:
        timestamp = now or _utc_now()
        with self._transaction() as connection:
            return self._expire_locked(connection, timestamp)

    def reserve(
        self,
        candidate_id: str,
        *,
        worker_id: str,
        input_hash: str,
        evaluator_hash: str,
        lease_seconds: int,
        estimated_wall_seconds: float,
        now: datetime | None = None,
    ) -> AttemptRecord:
        candidate = _require_code(candidate_id, "candidate_id")
        worker = _require_code(worker_id, "worker_id")
        input_digest = _require_digest(input_hash, "input_hash")
        evaluator_digest = _require_digest(evaluator_hash, "evaluator_hash")
        if (
            not isinstance(lease_seconds, int)
            or isinstance(lease_seconds, bool)
            or lease_seconds <= 0
        ):
            raise ValueError("lease_seconds must be a positive integer")
        if not isinstance(estimated_wall_seconds, (int, float)) or float(
            estimated_wall_seconds
        ) <= 0:
            raise ValueError("estimated_wall_seconds must be positive")
        timestamp = now or _utc_now()
        timestamp_text = _iso(timestamp)
        lease_expires = _iso(timestamp + timedelta(seconds=lease_seconds))
        with self._transaction() as connection:
            self._expire_locked(connection, timestamp)
            count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM evaluation_attempts"
                ).fetchone()[0]
            )
            if count >= self.max_evaluations:
                raise AttemptBudgetExceeded(
                    f"evaluation budget exhausted: {count}>={self.max_evaluations}"
                )
            active = connection.execute(
                """SELECT attempt_id FROM evaluation_attempts
                   WHERE candidate_id=? AND status IN ('leased', 'running')""",
                (candidate,),
            ).fetchone()
            if active is not None:
                raise AttemptLeaseConflict(
                    f"candidate already has active attempt {active['attempt_id']}"
                )
            next_number = int(
                connection.execute(
                    """SELECT COALESCE(MAX(attempt_number), 0) + 1
                       FROM evaluation_attempts WHERE candidate_id=?""",
                    (candidate,),
                ).fetchone()[0]
            )
            cursor = connection.execute(
                """INSERT INTO evaluation_attempts(
                    candidate_id, attempt_number, status, worker_id, input_hash,
                    evaluator_hash, reserved_at, started_at, heartbeat_at,
                    lease_expires_at, finished_at, estimated_wall_seconds,
                    actual_wall_seconds, result_hash, failure_code
                ) VALUES (?, ?, 'leased', ?, ?, ?, ?, NULL, ?, ?, NULL, ?, NULL, NULL, NULL)""",
                (
                    candidate,
                    next_number,
                    worker,
                    input_digest,
                    evaluator_digest,
                    timestamp_text,
                    timestamp_text,
                    lease_expires,
                    float(estimated_wall_seconds),
                ),
            )
            attempt_id = int(cursor.lastrowid)
        return self.get_attempt(attempt_id)

    def start(
        self,
        attempt_id: int,
        *,
        worker_id: str,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> AttemptRecord:
        worker = _require_code(worker_id, "worker_id")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        timestamp = now or _utc_now()
        with self._transaction() as connection:
            self._expire_locked(connection, timestamp)
            cursor = connection.execute(
                """UPDATE evaluation_attempts
                   SET status='running', started_at=?, heartbeat_at=?, lease_expires_at=?
                   WHERE attempt_id=? AND status='leased' AND worker_id=?""",
                (
                    _iso(timestamp),
                    _iso(timestamp),
                    _iso(timestamp + timedelta(seconds=lease_seconds)),
                    attempt_id,
                    worker,
                ),
            )
            if cursor.rowcount != 1:
                raise AttemptLeaseConflict("attempt is not leased by this worker")
        return self.get_attempt(attempt_id)

    def heartbeat(
        self,
        attempt_id: int,
        *,
        worker_id: str,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> AttemptRecord:
        worker = _require_code(worker_id, "worker_id")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        timestamp = now or _utc_now()
        with self._transaction() as connection:
            self._expire_locked(connection, timestamp)
            cursor = connection.execute(
                """UPDATE evaluation_attempts
                   SET heartbeat_at=?, lease_expires_at=?
                   WHERE attempt_id=? AND status='running' AND worker_id=?""",
                (
                    _iso(timestamp),
                    _iso(timestamp + timedelta(seconds=lease_seconds)),
                    attempt_id,
                    worker,
                ),
            )
            if cursor.rowcount != 1:
                raise AttemptLeaseConflict("running attempt is not owned by this worker")
        return self.get_attempt(attempt_id)

    def record_checkpoint(
        self,
        attempt_id: int,
        *,
        worker_id: str,
        stage: str,
        location: str,
        sha256: str,
        size_bytes: int,
        now: datetime | None = None,
    ) -> CheckpointRecord:
        worker = _require_code(worker_id, "worker_id")
        if stage not in CHECKPOINT_STAGES:
            raise ValueError(f"unknown checkpoint stage: {stage}")
        path = _safe_relative_location(location)
        digest = _require_digest(sha256, "sha256")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            raise ValueError("size_bytes must be a non-negative integer")
        timestamp = now or _utc_now()
        with self._transaction() as connection:
            self._expire_locked(connection, timestamp)
            owner = connection.execute(
                """SELECT status, worker_id FROM evaluation_attempts
                   WHERE attempt_id=?""",
                (attempt_id,),
            ).fetchone()
            if (
                owner is None
                or owner["status"] != AttemptStatus.RUNNING.value
                or owner["worker_id"] != worker
            ):
                raise AttemptLeaseConflict("checkpoint writer does not own a running attempt")
            existing = {
                row["stage"]
                for row in connection.execute(
                    "SELECT stage FROM evaluation_checkpoints WHERE attempt_id=?",
                    (attempt_id,),
                ).fetchall()
            }
            required = set(CHECKPOINT_STAGES[: CHECKPOINT_STAGES.index(stage)])
            if not required.issubset(existing):
                raise EvaluationAttemptError(
                    f"checkpoint stages must be recorded in order; missing {sorted(required-existing)}"
                )
            try:
                connection.execute(
                    """INSERT INTO evaluation_checkpoints(
                        attempt_id, stage, location, sha256, size_bytes, recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (attempt_id, stage, path, digest, size_bytes, _iso(timestamp)),
                )
            except sqlite3.IntegrityError as exc:
                raise EvaluationAttemptError(
                    f"checkpoint stage is already frozen: {attempt_id}:{stage}"
                ) from exc
        return self.get_checkpoint(attempt_id, stage)

    def succeed(
        self,
        attempt_id: int,
        *,
        worker_id: str,
        result_hash: str,
        actual_wall_seconds: float,
        now: datetime | None = None,
    ) -> AttemptRecord:
        worker = _require_code(worker_id, "worker_id")
        result_digest = _require_digest(result_hash, "result_hash")
        if actual_wall_seconds < 0:
            raise ValueError("actual_wall_seconds cannot be negative")
        timestamp = now or _utc_now()
        with self._transaction() as connection:
            self._expire_locked(connection, timestamp)
            stages = {
                row["stage"]
                for row in connection.execute(
                    "SELECT stage FROM evaluation_checkpoints WHERE attempt_id=?",
                    (attempt_id,),
                ).fetchall()
            }
            if stages != set(CHECKPOINT_STAGES):
                raise EvaluationAttemptError(
                    "successful attempt requires all four frozen checkpoints"
                )
            cursor = connection.execute(
                """UPDATE evaluation_attempts
                   SET status='succeeded', finished_at=?, actual_wall_seconds=?, result_hash=?
                   WHERE attempt_id=? AND status='running' AND worker_id=?""",
                (
                    _iso(timestamp),
                    float(actual_wall_seconds),
                    result_digest,
                    attempt_id,
                    worker,
                ),
            )
            if cursor.rowcount != 1:
                raise AttemptLeaseConflict("attempt cannot be completed by this worker")
        return self.get_attempt(attempt_id)

    def fail(
        self,
        attempt_id: int,
        *,
        worker_id: str,
        failure_code: str,
        actual_wall_seconds: float,
        now: datetime | None = None,
    ) -> AttemptRecord:
        worker = _require_code(worker_id, "worker_id")
        code = _require_code(failure_code, "failure_code")
        if actual_wall_seconds < 0:
            raise ValueError("actual_wall_seconds cannot be negative")
        timestamp = now or _utc_now()
        with self._transaction() as connection:
            self._expire_locked(connection, timestamp)
            cursor = connection.execute(
                """UPDATE evaluation_attempts
                   SET status='failed', finished_at=?, actual_wall_seconds=?, failure_code=?
                   WHERE attempt_id=? AND status IN ('leased', 'running') AND worker_id=?""",
                (
                    _iso(timestamp),
                    float(actual_wall_seconds),
                    code,
                    attempt_id,
                    worker,
                ),
            )
            if cursor.rowcount != 1:
                raise AttemptLeaseConflict("attempt cannot be failed by this worker")
        return self.get_attempt(attempt_id)

    def get_attempt(self, attempt_id: int) -> AttemptRecord:
        row = self.connection.execute(
            "SELECT * FROM evaluation_attempts WHERE attempt_id=?", (attempt_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown evaluation attempt: {attempt_id}")
        return AttemptRecord(
            attempt_id=int(row["attempt_id"]),
            candidate_id=row["candidate_id"],
            attempt_number=int(row["attempt_number"]),
            status=AttemptStatus(row["status"]),
            worker_id=row["worker_id"],
            input_hash=row["input_hash"],
            evaluator_hash=row["evaluator_hash"],
            reserved_at=row["reserved_at"],
            started_at=row["started_at"],
            heartbeat_at=row["heartbeat_at"],
            lease_expires_at=row["lease_expires_at"],
            finished_at=row["finished_at"],
            estimated_wall_seconds=float(row["estimated_wall_seconds"]),
            actual_wall_seconds=(
                None
                if row["actual_wall_seconds"] is None
                else float(row["actual_wall_seconds"])
            ),
            result_hash=row["result_hash"],
            failure_code=row["failure_code"],
        )

    def list_attempts(self, candidate_id: str | None = None) -> tuple[AttemptRecord, ...]:
        if candidate_id is None:
            rows = self.connection.execute(
                "SELECT attempt_id FROM evaluation_attempts ORDER BY attempt_id"
            ).fetchall()
        else:
            candidate = _require_code(candidate_id, "candidate_id")
            rows = self.connection.execute(
                """SELECT attempt_id FROM evaluation_attempts
                   WHERE candidate_id=? ORDER BY attempt_number""",
                (candidate,),
            ).fetchall()
        return tuple(self.get_attempt(int(row["attempt_id"])) for row in rows)

    def get_checkpoint(self, attempt_id: int, stage: str) -> CheckpointRecord:
        row = self.connection.execute(
            """SELECT * FROM evaluation_checkpoints
               WHERE attempt_id=? AND stage=?""",
            (attempt_id, stage),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown checkpoint: {attempt_id}:{stage}")
        return CheckpointRecord(
            attempt_id=int(row["attempt_id"]),
            stage=row["stage"],
            location=row["location"],
            sha256=row["sha256"],
            size_bytes=int(row["size_bytes"]),
            recorded_at=row["recorded_at"],
        )

    def list_checkpoints(self, attempt_id: int) -> tuple[CheckpointRecord, ...]:
        rows = self.connection.execute(
            """SELECT stage FROM evaluation_checkpoints WHERE attempt_id=?
               ORDER BY CASE stage
                   WHEN 'raw_signal' THEN 1 WHEN 'neutral_signal' THEN 2
                   WHEN 'diagnostics' THEN 3 WHEN 'metrics' THEN 4 END""",
            (attempt_id,),
        ).fetchall()
        return tuple(self.get_checkpoint(attempt_id, row["stage"]) for row in rows)

    def usage(self) -> AttemptUsage:
        rows = {
            row["status"]: int(row["count"])
            for row in self.connection.execute(
                "SELECT status, COUNT(*) AS count FROM evaluation_attempts GROUP BY status"
            ).fetchall()
        }
        wall = self.connection.execute(
            "SELECT COALESCE(SUM(actual_wall_seconds), 0.0) FROM evaluation_attempts"
        ).fetchone()[0]
        return AttemptUsage(
            attempts_reserved=sum(rows.values()),
            attempts_active=rows.get("leased", 0) + rows.get("running", 0),
            attempts_succeeded=rows.get("succeeded", 0),
            attempts_failed=rows.get("failed", 0),
            attempts_expired=rows.get("expired", 0),
            actual_wall_seconds=float(wall),
        )

    def verify_integrity(self) -> tuple[str, ...]:
        errors: list[str] = []
        result = self.connection.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            errors.append(f"sqlite_integrity:{result}")
        for row in self.connection.execute("SELECT * FROM evaluation_attempts"):
            for name in ("input_hash", "evaluator_hash"):
                if _SHA256_RE.fullmatch(row[name]) is None:
                    errors.append(f"invalid_{name}:{row['attempt_id']}")
            if row["result_hash"] is not None and _SHA256_RE.fullmatch(
                row["result_hash"]
            ) is None:
                errors.append(f"invalid_result_hash:{row['attempt_id']}")
            if row["status"] == "succeeded":
                stages = {
                    item["stage"]
                    for item in self.connection.execute(
                        "SELECT stage FROM evaluation_checkpoints WHERE attempt_id=?",
                        (row["attempt_id"],),
                    ).fetchall()
                }
                if stages != set(CHECKPOINT_STAGES):
                    errors.append(f"incomplete_success:{row['attempt_id']}")
        for row in self.connection.execute("SELECT * FROM evaluation_checkpoints"):
            if _SHA256_RE.fullmatch(row["sha256"]) is None:
                errors.append(
                    f"invalid_checkpoint_hash:{row['attempt_id']}:{row['stage']}"
                )
            try:
                _safe_relative_location(row["location"])
            except ValueError:
                errors.append(
                    f"unsafe_checkpoint_location:{row['attempt_id']}:{row['stage']}"
                )
        return tuple(errors)


def completed_candidate_ids(
    attempts: Sequence[AttemptRecord],
) -> frozenset[str]:
    """Return candidates with a successful immutable attempt."""

    return frozenset(
        item.candidate_id for item in attempts if item.status is AttemptStatus.SUCCEEDED
    )
