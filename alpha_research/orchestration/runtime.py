from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import TYPE_CHECKING, Iterator, cast

from alpha_research.core.hashing import hash_json, require_sha256

if TYPE_CHECKING:
    from alpha_research.experiments.spec import ExperimentSpec


EXPERIMENT_RUNTIME_SCHEMA = "experiment-runtime/v2"
_LEGACY_EXPERIMENT_RUNTIME_SCHEMA = "experiment-runtime/v1"

_STAGE_ATTEMPT_COLUMNS_V1 = (
    "attempt_id",
    "experiment_spec_hash",
    "stage",
    "attempt_number",
    "idempotency_key",
    "status",
    "worker_id",
    "lease_token_hash",
    "input_hash",
    "reserved_at",
    "started_at",
    "heartbeat_at",
    "lease_expires_at",
    "finished_at",
    "retry_not_before",
    "estimated_wall_seconds",
    "actual_wall_seconds",
    "actual_cpu_seconds",
    "peak_memory_bytes",
    "disk_write_bytes",
    "llm_calls",
    "llm_tokens",
    "llm_cost_microusd",
    "result_hash",
    "failure_code",
)
_STAGE_ATTEMPT_COLUMNS_V2 = (
    "attempt_id",
    "experiment_spec_hash",
    "stage",
    "attempt_scope_hash",
    "attempt_number",
    "idempotency_key",
    "status",
    "worker_id",
    "lease_token_hash",
    "input_hash",
    "reserved_at",
    "started_at",
    "heartbeat_at",
    "lease_expires_at",
    "reservation_request_hash",
    "finished_at",
    "retry_not_before",
    "estimated_wall_seconds",
    "actual_wall_seconds",
    "actual_cpu_seconds",
    "peak_memory_bytes",
    "disk_write_bytes",
    "llm_calls",
    "llm_tokens",
    "llm_cost_microusd",
    "result_hash",
    "failure_code",
)
_CHECKPOINT_COLUMNS = (
    "attempt_id",
    "ordinal",
    "checkpoint_name",
    "artifact_hash",
    "location",
    "recorded_at",
)

_RUNTIME_TABLES = frozenset(
    {
        "runtime_metadata",
        "runtime_control",
        "stage_attempts",
        "checkpoints",
        "circuit_breakers",
        "runtime_events",
    }
)
_COMMON_TABLE_SQL = {
    "runtime_metadata": """
        CREATE TABLE runtime_metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)
    """,
    "runtime_control": """
        CREATE TABLE runtime_control(
            singleton INTEGER PRIMARY KEY CHECK(singleton=1),
            state TEXT NOT NULL CHECK(state IN ('running','stop_requested')),
            actor TEXT NOT NULL,
            reason_code TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """,
    "checkpoints": """
        CREATE TABLE checkpoints(
            attempt_id INTEGER NOT NULL REFERENCES stage_attempts(attempt_id),
            ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
            checkpoint_name TEXT NOT NULL,
            artifact_hash TEXT NOT NULL,
            location TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            PRIMARY KEY(attempt_id,checkpoint_name),
            UNIQUE(attempt_id,ordinal)
        )
    """,
    "circuit_breakers": """
        CREATE TABLE circuit_breakers(
            service TEXT PRIMARY KEY,
            failure_count INTEGER NOT NULL CHECK(failure_count >= 0),
            state TEXT NOT NULL CHECK(state IN ('closed','open')),
            open_until TEXT,
            updated_at TEXT NOT NULL
        )
    """,
    "runtime_events": """
        CREATE TABLE runtime_events(
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            event_at TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            previous_event_hash TEXT,
            event_hash TEXT NOT NULL UNIQUE
        )
    """,
}
_STAGE_ATTEMPTS_SQL_V1 = """
    CREATE TABLE stage_attempts(
        attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_spec_hash TEXT NOT NULL,
        stage TEXT NOT NULL,
        attempt_number INTEGER NOT NULL CHECK(attempt_number > 0),
        idempotency_key TEXT NOT NULL UNIQUE,
        status TEXT NOT NULL CHECK(status IN (
            'leased','running','succeeded','failed','expired','cancelled'
        )),
        worker_id TEXT NOT NULL,
        lease_token_hash TEXT NOT NULL,
        input_hash TEXT NOT NULL,
        reserved_at TEXT NOT NULL,
        started_at TEXT,
        heartbeat_at TEXT NOT NULL,
        lease_expires_at TEXT NOT NULL,
        finished_at TEXT,
        retry_not_before TEXT,
        estimated_wall_seconds REAL NOT NULL CHECK(estimated_wall_seconds > 0),
        actual_wall_seconds REAL,
        actual_cpu_seconds REAL,
        peak_memory_bytes INTEGER,
        disk_write_bytes INTEGER,
        llm_calls INTEGER,
        llm_tokens INTEGER,
        llm_cost_microusd INTEGER,
        result_hash TEXT,
        failure_code TEXT,
        UNIQUE(stage,attempt_number)
    )
"""
_STAGE_ATTEMPTS_SQL_V2 = """
    CREATE TABLE stage_attempts(
        attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_spec_hash TEXT NOT NULL,
        stage TEXT NOT NULL,
        attempt_scope_hash TEXT NOT NULL,
        attempt_number INTEGER NOT NULL CHECK(attempt_number > 0),
        idempotency_key TEXT NOT NULL UNIQUE,
        status TEXT NOT NULL CHECK(status IN (
            'leased','running','succeeded','failed','expired','cancelled'
        )),
        worker_id TEXT NOT NULL,
        lease_token_hash TEXT NOT NULL,
        input_hash TEXT NOT NULL,
        reserved_at TEXT NOT NULL,
        started_at TEXT,
        heartbeat_at TEXT NOT NULL,
        lease_expires_at TEXT NOT NULL,
        reservation_request_hash TEXT NOT NULL
            CHECK(length(reservation_request_hash)=64
              AND reservation_request_hash NOT GLOB '*[^0-9a-f]*'),
        finished_at TEXT,
        retry_not_before TEXT,
        estimated_wall_seconds REAL NOT NULL CHECK(estimated_wall_seconds > 0),
        actual_wall_seconds REAL,
        actual_cpu_seconds REAL,
        peak_memory_bytes INTEGER,
        disk_write_bytes INTEGER,
        llm_calls INTEGER,
        llm_tokens INTEGER,
        llm_cost_microusd INTEGER,
        result_hash TEXT,
        failure_code TEXT,
        UNIQUE(attempt_scope_hash,attempt_number)
    )
"""
_ACTIVE_INDEX_SQL = {
    _LEGACY_EXPERIMENT_RUNTIME_SCHEMA: """
        CREATE UNIQUE INDEX one_active_attempt_per_stage
        ON stage_attempts(stage) WHERE status IN ('leased','running')
    """,
    EXPERIMENT_RUNTIME_SCHEMA: """
        CREATE UNIQUE INDEX one_active_attempt_per_scope
        ON stage_attempts(attempt_scope_hash)
        WHERE status IN ('leased','running')
    """,
}
_AUTO_INDEX_TABLES = {
    "sqlite_autoindex_runtime_metadata_1": "runtime_metadata",
    "sqlite_autoindex_stage_attempts_1": "stage_attempts",
    "sqlite_autoindex_stage_attempts_2": "stage_attempts",
    "sqlite_autoindex_checkpoints_1": "checkpoints",
    "sqlite_autoindex_checkpoints_2": "checkpoints",
    "sqlite_autoindex_circuit_breakers_1": "circuit_breakers",
    "sqlite_autoindex_runtime_events_1": "runtime_events",
}
_TABLE_CREATE_ORDER = (
    "runtime_metadata",
    "runtime_control",
    "stage_attempts",
    "checkpoints",
    "circuit_breakers",
    "runtime_events",
)


class RuntimeErrorBase(RuntimeError):
    pass


class DispatchStopped(RuntimeErrorBase):
    pass


class AttemptLeaseConflict(RuntimeErrorBase):
    pass


class ResourceBudgetExceeded(RuntimeErrorBase):
    pass


class RetryNotReady(RuntimeErrorBase):
    pass


class CircuitOpen(RuntimeErrorBase):
    pass


class AttemptStatus(str, Enum):
    LEASED = "leased"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class StageAttempt:
    attempt_id: int
    experiment_spec_hash: str
    stage: str
    attempt_scope_hash: str
    attempt_number: int
    idempotency_key: str
    status: AttemptStatus
    worker_id: str
    input_hash: str
    reserved_at: str
    started_at: str | None
    heartbeat_at: str
    lease_expires_at: str
    reservation_request_hash: str
    finished_at: str | None
    retry_not_before: str | None
    estimated_wall_seconds: float
    actual_wall_seconds: float | None
    actual_cpu_seconds: float | None
    peak_memory_bytes: int | None
    disk_write_bytes: int | None
    llm_calls: int | None
    llm_tokens: int | None
    llm_cost_microusd: int | None
    result_hash: str | None
    failure_code: str | None


@dataclass(frozen=True, slots=True)
class RuntimeUsage:
    attempts_reserved: int
    attempts_active: int
    attempts_succeeded: int
    attempts_failed: int
    attempts_expired: int
    attempts_cancelled: int
    wall_seconds: float
    cpu_seconds: float
    peak_memory_bytes: int
    disk_write_bytes: int
    llm_calls: int
    llm_tokens: int
    llm_cost_microusd: int


@dataclass(frozen=True, slots=True)
class Checkpoint:
    attempt_id: int
    ordinal: int
    checkpoint_name: str
    artifact_hash: str
    location: str
    recorded_at: str


class ExperimentRuntime:
    """Persistent single-node scheduler state with fail-closed worker leases."""

    def __init__(self, path: str | Path, spec: ExperimentSpec) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        database_preexisted = os.path.lexists(self.path)
        if not database_preexisted:
            try:
                descriptor = os.open(
                    self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
                )
            except FileExistsError:
                database_preexisted = True
            else:
                os.close(descriptor)
        path_stat = os.lstat(self.path)
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or path_stat.st_nlink != 1
        ):
            raise RuntimeErrorBase(
                "runtime database path must be one non-symlink regular file"
            )
        self.connection = sqlite3.connect(self.path, timeout=5.0)
        try:
            connected_path_stat = os.lstat(self.path)
            if (
                connected_path_stat.st_dev,
                connected_path_stat.st_ino,
            ) != (path_stat.st_dev, path_stat.st_ino):
                raise RuntimeErrorBase("runtime database path identity changed")
            self._database_identity = (
                connected_path_stat.st_dev,
                connected_path_stat.st_ino,
            )
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA busy_timeout = 5000")
            if int(self.connection.execute("PRAGMA foreign_keys").fetchone()[0]) != 1:
                raise RuntimeErrorBase("runtime foreign-key enforcement is disabled")
            self._initialize_schema(database_preexisted=database_preexisted)
            # Journal mode is a persistent database mutation.  Existing files are
            # therefore switched only after their full schema and event chain have
            # been authenticated; malformed databases are never repaired en route.
            self._assert_database_path_identity()
            self.connection.execute("PRAGMA journal_mode = WAL")
        except BaseException:
            self.connection.close()
            raise

    def __enter__(self) -> "ExperimentRuntime":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        if self.connection.in_transaction:
            raise RuntimeErrorBase("nested runtime transactions are forbidden")
        try:
            self._assert_database_path_identity()
            self.connection.execute("BEGIN IMMEDIATE")
            yield self.connection
            self._assert_database_path_identity()
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def _assert_database_path_identity(self) -> None:
        """Fail closed when the authoritative database pathname is rebound."""

        try:
            observed = os.lstat(self.path)
        except OSError as exc:
            raise RuntimeErrorBase(
                "runtime database path identity unavailable"
            ) from exc
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or (observed.st_dev, observed.st_ino) != self._database_identity
        ):
            raise RuntimeErrorBase("runtime database path identity changed")

    def _initialize_schema(self, *, database_preexisted: bool) -> None:
        if not database_preexisted:
            self._initialize_new_v2_database()
            return

        metadata_sql = self.connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='runtime_metadata'"
        ).fetchone()
        if metadata_sql is None:
            raise RuntimeErrorBase("runtime metadata table is missing")
        if _normalized_schema_sql(metadata_sql["sql"]) != _normalized_schema_sql(
            _COMMON_TABLE_SQL["runtime_metadata"]
        ):
            raise RuntimeErrorBase("runtime metadata table layout differs")
        try:
            existing = dict(
                self.connection.execute(
                    "SELECT key,value FROM runtime_metadata"
                ).fetchall()
            )
        except sqlite3.DatabaseError as exc:
            raise RuntimeErrorBase("runtime metadata cannot be read") from exc
        expected_keys = frozenset(self._expected_metadata(EXPERIMENT_RUNTIME_SCHEMA))
        if frozenset(existing) != expected_keys:
            raise RuntimeErrorBase("runtime database binding differs")
        for key in expected_keys - {"schema_version"}:
            if existing[key] != self._expected_metadata(EXPERIMENT_RUNTIME_SCHEMA)[key]:
                raise RuntimeErrorBase("runtime database binding differs")

        schema_version = existing["schema_version"]
        if schema_version == _LEGACY_EXPERIMENT_RUNTIME_SCHEMA:
            # This read-only preflight makes the no-repair contract explicit.
            # Hold one coherent snapshot so a concurrent authenticated opener
            # cannot append an event between the chain rows and sequence-head
            # reads.  Another opener may already have completed the migration
            # while this one waited, in which case the V2 attestation is final.
            with self._transaction() as connection:
                locked_version_row = connection.execute(
                    "SELECT value FROM runtime_metadata WHERE key='schema_version'"
                ).fetchone()
                locked_version = (
                    None
                    if locked_version_row is None
                    else str(locked_version_row["value"])
                )
                if locked_version == _LEGACY_EXPERIMENT_RUNTIME_SCHEMA:
                    self._assert_v1_schema(connection)
                elif locked_version == EXPERIMENT_RUNTIME_SCHEMA:
                    self._assert_v2_schema(connection)
                    return
                else:
                    raise RuntimeErrorBase("runtime database binding differs")
            # The migration repeats the same attestation under BEGIN IMMEDIATE
            # before executing its first DDL statement, closing the TOCTOU gap.
            self._migrate_v1_to_v2()
        elif schema_version == EXPERIMENT_RUNTIME_SCHEMA:
            # Attestation spans sqlite_master, table rows and the event-sequence
            # head.  It must therefore observe one transactionally coherent
            # snapshot rather than a sequence of independent autocommit reads.
            with self._transaction() as connection:
                self._assert_v2_schema(connection)
        else:
            raise RuntimeErrorBase("runtime database binding differs")

    def _initialize_new_v2_database(self) -> None:
        # A path that was absent before sqlite3.connect is the sole state in
        # which schema creation is authorized.  A non-empty sqlite_master here
        # indicates a creation race and is rejected rather than merged/repaired.
        if self.connection.execute(
            "SELECT 1 FROM sqlite_master LIMIT 1"
        ).fetchone() is not None:
            raise RuntimeErrorBase("new runtime database path was populated concurrently")
        expected_table_sql = dict(_COMMON_TABLE_SQL)
        expected_table_sql["stage_attempts"] = _STAGE_ATTEMPTS_SQL_V2
        with self._transaction() as connection:
            for table in _TABLE_CREATE_ORDER:
                connection.execute(expected_table_sql[table])
            connection.execute(_ACTIVE_INDEX_SQL[EXPERIMENT_RUNTIME_SCHEMA])
            connection.executemany(
                "INSERT INTO runtime_metadata(key,value) VALUES(?,?)",
                sorted(self._expected_metadata(EXPERIMENT_RUNTIME_SCHEMA).items()),
            )
            connection.execute(
                """INSERT INTO runtime_control(
                    singleton,state,actor,reason_code,updated_at
                ) VALUES(1,'running','system','initialized',?)""",
                (_iso(_utc_now()),),
            )
            self._assert_v2_schema(connection)

    def _expected_metadata(self, schema_version: str) -> dict[str, str]:
        return {
            "schema_version": schema_version,
            "experiment_spec_hash": self.spec.content_hash,
            "resource_budget_hash": self.spec.resource_budget.content_hash,
            "retry_policy_hash": self.spec.retry_policy.content_hash,
        }

    @staticmethod
    def _create_v2_indexes(connection: sqlite3.Connection) -> None:
        connection.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS one_active_attempt_per_scope
               ON stage_attempts(attempt_scope_hash)
               WHERE status IN ('leased','running')"""
        )

    @staticmethod
    def _table_columns(
        connection: sqlite3.Connection, table: str
    ) -> tuple[str, ...]:
        if table not in {"stage_attempts", "checkpoints"}:
            raise ValueError("unsupported runtime schema table")
        return tuple(
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        )

    def _migrate_v1_to_v2(self) -> None:
        with self._transaction() as connection:
            current_version_row = connection.execute(
                "SELECT value FROM runtime_metadata WHERE key='schema_version'"
            ).fetchone()
            current_version = (
                None
                if current_version_row is None
                else str(current_version_row["value"])
            )
            if current_version == EXPERIMENT_RUNTIME_SCHEMA:
                # Another authenticated opener completed the migration while
                # this connection waited for BEGIN IMMEDIATE.
                self._assert_v2_schema(connection)
                return
            if current_version != _LEGACY_EXPERIMENT_RUNTIME_SCHEMA:
                raise RuntimeErrorBase("runtime database binding differs")
            self._assert_v1_schema(connection)
            control_rows = int(
                connection.execute("SELECT COUNT(*) FROM runtime_control").fetchone()[0]
            )
            if control_rows != 1:
                raise RuntimeErrorBase("legacy runtime control row differs")
            invalid_binding = connection.execute(
                """SELECT attempt_id FROM stage_attempts
                   WHERE experiment_spec_hash<>? LIMIT 1""",
                (self.spec.content_hash,),
            ).fetchone()
            if invalid_binding is not None:
                raise RuntimeErrorBase("legacy attempt experiment binding differs")
            invalid_stage = connection.execute(
                "SELECT DISTINCT stage FROM stage_attempts"
            ).fetchall()
            if any(str(row["stage"]) not in self.spec.stages for row in invalid_stage):
                raise RuntimeErrorBase("legacy attempt stage is not enabled")
            connection.execute(
                """CREATE TABLE stage_attempts_v2_migration(
                    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_spec_hash TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    attempt_scope_hash TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL CHECK(attempt_number > 0),
                    idempotency_key TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK(status IN (
                        'leased','running','succeeded','failed','expired','cancelled'
                    )),
                    worker_id TEXT NOT NULL,
                    lease_token_hash TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    reserved_at TEXT NOT NULL,
                    started_at TEXT,
                    heartbeat_at TEXT NOT NULL,
                    lease_expires_at TEXT NOT NULL,
                    reservation_request_hash TEXT NOT NULL
                        CHECK(length(reservation_request_hash)=64
                          AND reservation_request_hash NOT GLOB '*[^0-9a-f]*'),
                    finished_at TEXT,
                    retry_not_before TEXT,
                    estimated_wall_seconds REAL NOT NULL
                        CHECK(estimated_wall_seconds > 0),
                    actual_wall_seconds REAL,
                    actual_cpu_seconds REAL,
                    peak_memory_bytes INTEGER,
                    disk_write_bytes INTEGER,
                    llm_calls INTEGER,
                    llm_tokens INTEGER,
                    llm_cost_microusd INTEGER,
                    result_hash TEXT,
                    failure_code TEXT,
                    UNIQUE(attempt_scope_hash,attempt_number)
                )"""
            )
            old_rows = connection.execute(
                "SELECT * FROM stage_attempts ORDER BY attempt_id"
            ).fetchall()
            placeholders = ",".join("?" for _ in _STAGE_ATTEMPT_COLUMNS_V2)
            for row in old_rows:
                stage = str(row["stage"])
                values: list[object] = []
                for column in _STAGE_ATTEMPT_COLUMNS_V2:
                    if column == "attempt_scope_hash":
                        values.append(
                            _legacy_attempt_scope_hash(self.spec.content_hash, stage)
                        )
                    elif column == "reservation_request_hash":
                        values.append(
                            _legacy_unavailable_reservation_request_hash(row)
                        )
                    else:
                        values.append(row[column])
                try:
                    connection.execute(
                        "INSERT INTO stage_attempts_v2_migration VALUES("
                        + placeholders
                        + ")",
                        values,
                    )
                except sqlite3.IntegrityError as exc:
                    raise RuntimeErrorBase(
                        "legacy attempt scope migration collides"
                    ) from exc
            connection.execute(
                """CREATE TABLE checkpoints_v2_migration(
                    attempt_id INTEGER NOT NULL
                        REFERENCES stage_attempts_v2_migration(attempt_id),
                    ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
                    checkpoint_name TEXT NOT NULL,
                    artifact_hash TEXT NOT NULL,
                    location TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY(attempt_id,checkpoint_name),
                    UNIQUE(attempt_id,ordinal)
                )"""
            )
            connection.execute(
                """INSERT INTO checkpoints_v2_migration(
                    attempt_id,ordinal,checkpoint_name,artifact_hash,location,recorded_at
                ) SELECT attempt_id,ordinal,checkpoint_name,artifact_hash,location,
                         recorded_at FROM checkpoints"""
            )
            connection.execute("DROP TABLE checkpoints")
            connection.execute("DROP TABLE stage_attempts")
            connection.execute(
                "ALTER TABLE stage_attempts_v2_migration RENAME TO stage_attempts"
            )
            connection.execute(
                "ALTER TABLE checkpoints_v2_migration RENAME TO checkpoints"
            )
            self._create_v2_indexes(connection)
            connection.execute(
                "UPDATE runtime_metadata SET value=? WHERE key='schema_version'",
                (EXPERIMENT_RUNTIME_SCHEMA,),
            )
            self._event_locked(
                connection,
                "runtime_schema_migrated",
                {
                    "from_schema": _LEGACY_EXPERIMENT_RUNTIME_SCHEMA,
                    "to_schema": EXPERIMENT_RUNTIME_SCHEMA,
                    "attempt_count": len(old_rows),
                },
                _iso(_utc_now()),
            )
            self._assert_v2_schema(connection)

    def _assert_v1_schema(self, connection: sqlite3.Connection) -> None:
        if self._table_columns(connection, "stage_attempts") != (
            _STAGE_ATTEMPT_COLUMNS_V1
        ):
            raise RuntimeErrorBase("legacy runtime stage_attempts layout differs")
        if self._table_columns(connection, "checkpoints") != _CHECKPOINT_COLUMNS:
            raise RuntimeErrorBase("legacy runtime checkpoints layout differs")
        errors = self._schema_attestation_errors(
            connection, _LEGACY_EXPERIMENT_RUNTIME_SCHEMA
        )
        if errors:
            raise RuntimeErrorBase(
                "legacy runtime schema attestation failed:" + ",".join(errors)
            )

    def _assert_v2_schema(self, connection: sqlite3.Connection) -> None:
        if self._table_columns(connection, "stage_attempts") != (
            _STAGE_ATTEMPT_COLUMNS_V2
        ):
            raise RuntimeErrorBase("runtime stage_attempts v2 layout differs")
        if self._table_columns(connection, "checkpoints") != _CHECKPOINT_COLUMNS:
            raise RuntimeErrorBase("runtime checkpoints v2 layout differs")
        errors = self._schema_attestation_errors(connection, EXPERIMENT_RUNTIME_SCHEMA)
        if errors:
            raise RuntimeErrorBase(
                "runtime schema attestation failed:" + ",".join(errors)
            )

    def _schema_attestation_errors(
        self, connection: sqlite3.Connection, schema_version: str
    ) -> tuple[str, ...]:
        if schema_version not in {
            _LEGACY_EXPERIMENT_RUNTIME_SCHEMA,
            EXPERIMENT_RUNTIME_SCHEMA,
        }:
            raise ValueError("unsupported runtime schema version")
        errors: list[str] = []
        try:
            objects = connection.execute(
                "SELECT type,name,tbl_name,sql FROM sqlite_master"
            ).fetchall()
        except sqlite3.DatabaseError:
            return ("sqlite_master_unreadable",)

        tables = {
            str(row["name"]): row
            for row in objects
            if row["type"] == "table" and not str(row["name"]).startswith("sqlite_")
        }
        table_names = frozenset(tables)
        for name in sorted(_RUNTIME_TABLES - table_names):
            errors.append(f"schema_table_missing:{name}")
        for name in sorted(table_names - _RUNTIME_TABLES):
            errors.append(f"schema_table_unexpected:{name}")
        internal_tables = {
            str(row["name"])
            for row in objects
            if row["type"] == "table" and str(row["name"]).startswith("sqlite_")
        }
        if "sqlite_sequence" not in internal_tables:
            errors.append("schema_table_missing:sqlite_sequence")

        expected_table_sql = dict(_COMMON_TABLE_SQL)
        expected_table_sql["stage_attempts"] = (
            _STAGE_ATTEMPTS_SQL_V1
            if schema_version == _LEGACY_EXPERIMENT_RUNTIME_SCHEMA
            else _STAGE_ATTEMPTS_SQL_V2
        )
        for name, expected_sql in expected_table_sql.items():
            row = tables.get(name)
            if row is not None and _normalized_schema_sql(
                row["sql"]
            ) != _normalized_schema_sql(expected_sql):
                errors.append(f"schema_table_sql:{name}")

        active_index_name = (
            "one_active_attempt_per_stage"
            if schema_version == _LEGACY_EXPERIMENT_RUNTIME_SCHEMA
            else "one_active_attempt_per_scope"
        )
        indexes = {
            str(row["name"]): row for row in objects if row["type"] == "index"
        }
        expected_index_names = frozenset(_AUTO_INDEX_TABLES) | {active_index_name}
        for name in sorted(expected_index_names - frozenset(indexes)):
            errors.append(f"schema_index_missing:{name}")
        for name in sorted(frozenset(indexes) - expected_index_names):
            errors.append(f"schema_index_unexpected:{name}")
        for name, table in _AUTO_INDEX_TABLES.items():
            row = indexes.get(name)
            if row is not None and (
                row["sql"] is not None or str(row["tbl_name"]) != table
            ):
                errors.append(f"schema_autoindex_layout:{name}")
        active_index = indexes.get(active_index_name)
        if active_index is not None and (
            str(active_index["tbl_name"]) != "stage_attempts"
            or _normalized_schema_sql(active_index["sql"])
            != _normalized_schema_sql(_ACTIVE_INDEX_SQL[schema_version])
        ):
            errors.append(f"schema_index_sql:{active_index_name}")

        for row in objects:
            if row["type"] in {"trigger", "view"}:
                errors.append(f"schema_{row['type']}_unexpected:{row['name']}")

        if int(connection.execute("PRAGMA foreign_keys").fetchone()[0]) != 1:
            errors.append("sqlite_foreign_keys_disabled")
        for table in sorted(_RUNTIME_TABLES & table_names):
            try:
                foreign_keys = connection.execute(
                    f'PRAGMA foreign_key_list("{table}")'
                ).fetchall()
            except sqlite3.DatabaseError:
                errors.append(f"schema_foreign_key_unreadable:{table}")
                continue
            actual_foreign_keys = tuple(
                (
                    str(row["table"]),
                    str(row["from"]),
                    str(row["to"]),
                    str(row["on_update"]),
                    str(row["on_delete"]),
                    str(row["match"]),
                )
                for row in foreign_keys
            )
            expected_foreign_keys = (
                (
                    (
                        "stage_attempts",
                        "attempt_id",
                        "attempt_id",
                        "NO ACTION",
                        "NO ACTION",
                        "NONE",
                    ),
                )
                if table == "checkpoints"
                else ()
            )
            if actual_foreign_keys != expected_foreign_keys:
                errors.append(f"schema_foreign_key:{table}")

        try:
            integrity_rows = tuple(
                str(row[0]) for row in connection.execute("PRAGMA integrity_check")
            )
            if integrity_rows != ("ok",):
                errors.append("sqlite_integrity_failed")
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                errors.append("sqlite_foreign_key_failed")
        except sqlite3.DatabaseError:
            errors.append("sqlite_integrity_unreadable")

        if "runtime_metadata" in tables:
            try:
                metadata = dict(
                    connection.execute(
                        "SELECT key,value FROM runtime_metadata"
                    ).fetchall()
                )
            except sqlite3.DatabaseError:
                errors.append("runtime_metadata_unreadable")
            else:
                if metadata != self._expected_metadata(schema_version):
                    errors.append("runtime_metadata_binding")

        if "runtime_control" in tables:
            try:
                control = connection.execute(
                    "SELECT singleton FROM runtime_control"
                ).fetchall()
            except sqlite3.DatabaseError:
                errors.append("runtime_control_unreadable")
            else:
                if len(control) != 1 or int(control[0]["singleton"]) != 1:
                    errors.append("runtime_control_row")

        if "stage_attempts" in tables:
            try:
                invalid_binding = connection.execute(
                    """SELECT attempt_id FROM stage_attempts
                       WHERE experiment_spec_hash<>? LIMIT 1""",
                    (self.spec.content_hash,),
                ).fetchone()
                invalid_stage = connection.execute(
                    "SELECT DISTINCT stage FROM stage_attempts"
                ).fetchall()
                if invalid_binding is not None:
                    errors.append("attempt_experiment_binding")
                if any(
                    str(row["stage"]) not in self.spec.stages for row in invalid_stage
                ):
                    errors.append("attempt_stage_not_enabled")
                if schema_version == EXPERIMENT_RUNTIME_SCHEMA:
                    invalid_scope = connection.execute(
                        """SELECT attempt_id FROM stage_attempts
                           WHERE length(attempt_scope_hash)<>64
                              OR attempt_scope_hash GLOB '*[^0-9a-f]*' LIMIT 1"""
                    ).fetchone()
                    invalid_request_hash = connection.execute(
                        """SELECT attempt_id FROM stage_attempts
                           WHERE length(reservation_request_hash)<>64
                              OR reservation_request_hash GLOB '*[^0-9a-f]*'
                           LIMIT 1"""
                    ).fetchone()
                    duplicate_scope_attempt = connection.execute(
                        """SELECT attempt_scope_hash,attempt_number
                           FROM stage_attempts
                           GROUP BY attempt_scope_hash,attempt_number
                           HAVING COUNT(*)>1 LIMIT 1"""
                    ).fetchone()
                    duplicate_active = connection.execute(
                        """SELECT attempt_scope_hash FROM stage_attempts
                           WHERE status IN ('leased','running')
                           GROUP BY attempt_scope_hash HAVING COUNT(*)>1 LIMIT 1"""
                    ).fetchone()
                    multiple_scope_stages = connection.execute(
                        """SELECT attempt_scope_hash FROM stage_attempts
                           GROUP BY attempt_scope_hash
                           HAVING COUNT(DISTINCT stage)>1 LIMIT 1"""
                    ).fetchone()
                    if invalid_scope is not None:
                        errors.append(f"attempt_scope_hash:{invalid_scope['attempt_id']}")
                    if invalid_request_hash is not None:
                        errors.append(
                            "reservation_request_hash:"
                            f"{invalid_request_hash['attempt_id']}"
                        )
                    if duplicate_scope_attempt is not None:
                        errors.append("attempt_scope_number_duplicate")
                    if duplicate_active is not None:
                        errors.append("attempt_scope_multiple_active")
                    if multiple_scope_stages is not None:
                        errors.append("attempt_scope_multiple_stages")
                    request_rows = connection.execute(
                        """SELECT attempt_id,stage,attempt_scope_hash,worker_id,
                                  idempotency_key,reservation_request_hash
                           FROM stage_attempts"""
                    ).fetchall()
                    for request_row in request_rows:
                        request_hash = str(request_row["reservation_request_hash"])
                        if request_hash == (
                            _legacy_unavailable_reservation_request_hash(request_row)
                        ):
                            continue
                        expected_payload_hash = hash_json(
                            {
                                "attempt_id": int(request_row["attempt_id"]),
                                "stage": str(request_row["stage"]),
                                "attempt_scope_hash": str(
                                    request_row["attempt_scope_hash"]
                                ),
                                "worker_id": str(request_row["worker_id"]),
                                "reservation_request_hash": request_hash,
                            }
                        )
                        event_count = int(
                            connection.execute(
                                """SELECT COUNT(*) FROM runtime_events
                                   WHERE event_type='attempt_reserved'
                                     AND payload_hash=?""",
                                (expected_payload_hash,),
                            ).fetchone()[0]
                        )
                        if event_count != 1:
                            errors.append(
                                "reservation_request_event:"
                                f"{request_row['attempt_id']}"
                            )
            except sqlite3.DatabaseError:
                errors.append("stage_attempts_unreadable")

        if (
            schema_version == EXPERIMENT_RUNTIME_SCHEMA
            and {"stage_attempts", "checkpoints"} <= table_names
        ):
            errors.extend(self._runtime_row_semantic_errors(connection))
        if "runtime_events" in tables:
            errors.extend(self._event_chain_errors(connection))
        return tuple(dict.fromkeys(errors))

    def _runtime_row_semantic_errors(
        self, connection: sqlite3.Connection
    ) -> tuple[str, ...]:
        """Validate durable row semantics without repairing or mutating state."""

        try:
            attempts = connection.execute(
                "SELECT * FROM stage_attempts ORDER BY attempt_scope_hash,attempt_number"
            ).fetchall()
            checkpoints = connection.execute(
                "SELECT * FROM checkpoints ORDER BY attempt_id,ordinal"
            ).fetchall()
        except sqlite3.DatabaseError:
            return ("runtime_rows_unreadable",)

        errors: list[str] = []
        attempts_by_id = {int(row["attempt_id"]): row for row in attempts}
        attempts_by_scope: dict[str, list[sqlite3.Row]] = {}
        for row in attempts:
            attempts_by_scope.setdefault(str(row["attempt_scope_hash"]), []).append(row)
        for scope, rows in attempts_by_scope.items():
            observed_numbers = tuple(row["attempt_number"] for row in rows)
            expected_numbers = tuple(range(1, len(rows) + 1))
            if (
                any(type(number) is not int for number in observed_numbers)
                or observed_numbers != expected_numbers
            ):
                errors.append(f"attempt_number_sequence:{scope}")

        attempt_times: dict[int, dict[str, datetime | None]] = {}
        usage_float_fields = ("actual_wall_seconds", "actual_cpu_seconds")
        usage_integer_fields = (
            "peak_memory_bytes",
            "disk_write_bytes",
            "llm_calls",
            "llm_tokens",
            "llm_cost_microusd",
        )
        for row in attempts:
            attempt_id = int(row["attempt_id"])
            parsed_times: dict[str, datetime | None] = {}
            for field in (
                "reserved_at",
                "started_at",
                "heartbeat_at",
                "lease_expires_at",
                "finished_at",
                "retry_not_before",
            ):
                value = row[field]
                if value is None:
                    parsed_times[field] = None
                    continue
                try:
                    parsed = _parse(value)
                except (TypeError, ValueError):
                    errors.append(f"attempt_timestamp:{attempt_id}:{field}")
                    parsed_times[field] = None
                    continue
                if type(value) is not str or _iso(parsed) != value:
                    errors.append(f"attempt_timestamp:{attempt_id}:{field}")
                parsed_times[field] = parsed
            attempt_times[attempt_id] = parsed_times

            reserved = parsed_times["reserved_at"]
            started = parsed_times["started_at"]
            heartbeat = parsed_times["heartbeat_at"]
            lease_expires = parsed_times["lease_expires_at"]
            finished = parsed_times["finished_at"]
            retry_not_before = parsed_times["retry_not_before"]
            ordered = (
                reserved is not None
                and heartbeat is not None
                and lease_expires is not None
                and reserved <= heartbeat < lease_expires
                and (started is None or reserved <= started <= heartbeat)
                and (finished is None or heartbeat <= finished)
                and (
                    retry_not_before is None
                    or (finished is not None and finished <= retry_not_before)
                )
            )
            if not ordered:
                errors.append(f"attempt_timestamp_order:{attempt_id}")

            estimate = row["estimated_wall_seconds"]
            estimate_valid = (
                type(estimate) in {int, float}
                and math.isfinite(float(estimate))
                and float(estimate) > 0
            )
            float_usage_valid = all(
                row[field] is None
                or (
                    type(row[field]) in {int, float}
                    and math.isfinite(float(row[field]))
                    and float(row[field]) >= 0
                )
                for field in usage_float_fields
            )
            integer_usage_valid = all(
                row[field] is None
                or (type(row[field]) is int and int(row[field]) >= 0)
                for field in usage_integer_fields
            )
            if not (estimate_valid and float_usage_valid and integer_usage_valid):
                errors.append(f"attempt_usage:{attempt_id}")

            usage_values = tuple(
                row[field] for field in usage_float_fields + usage_integer_fields
            )
            usage_all_none = all(value is None for value in usage_values)
            usage_all_present = all(value is not None for value in usage_values)
            status = str(row["status"])
            is_legacy = str(row["reservation_request_hash"]) == (
                _legacy_unavailable_reservation_request_hash(row)
            )
            if is_legacy:
                continue
            retry_expected = (
                type(row["attempt_number"]) is int
                and row["failure_code"]
                in self.spec.retry_policy.retryable_failure_codes
                and int(row["attempt_number"])
                < self.spec.retry_policy.maximum_attempts_per_stage
            )
            state_valid = False
            if status == "leased":
                state_valid = (
                    row["started_at"] is None
                    and row["finished_at"] is None
                    and row["retry_not_before"] is None
                    and row["result_hash"] is None
                    and row["failure_code"] is None
                    and usage_all_none
                )
            elif status == "running":
                state_valid = (
                    row["started_at"] is not None
                    and row["finished_at"] is None
                    and row["retry_not_before"] is None
                    and row["result_hash"] is None
                    and row["failure_code"] is None
                    and usage_all_none
                )
            elif status == "succeeded":
                state_valid = (
                    row["started_at"] is not None
                    and row["finished_at"] is not None
                    and row["retry_not_before"] is None
                    and _is_sha256_text(row["result_hash"])
                    and row["failure_code"] is None
                    and usage_all_present
                )
            elif status == "failed":
                state_valid = (
                    row["started_at"] is not None
                    and row["finished_at"] is not None
                    and row["result_hash"] is None
                    and row["failure_code"] is not None
                    and (row["retry_not_before"] is not None) == retry_expected
                    and usage_all_present
                )
            elif status == "expired":
                state_valid = (
                    row["finished_at"] is not None
                    and row["result_hash"] is None
                    and row["failure_code"] == "worker_lost"
                    and (row["retry_not_before"] is not None) == retry_expected
                    and usage_all_none
                )
            elif status == "cancelled":
                state_valid = (
                    row["finished_at"] is not None
                    and row["retry_not_before"] is None
                    and row["result_hash"] is None
                    and row["failure_code"] == "kill_switch"
                    and usage_all_none
                )
            if not state_valid:
                errors.append(f"attempt_state_fields:{attempt_id}")

        checkpoints_by_attempt: dict[int, list[sqlite3.Row]] = {}
        checkpoint_hashes_by_attempt: dict[int, set[str]] = {}
        for row in checkpoints:
            attempt_id = int(row["attempt_id"])
            checkpoints_by_attempt.setdefault(attempt_id, []).append(row)
            checkpoint_hashes_by_attempt.setdefault(attempt_id, set()).add(
                str(row["artifact_hash"])
            )
        for attempt_id, rows in checkpoints_by_attempt.items():
            observed_ordinals = tuple(row["ordinal"] for row in rows)
            if (
                any(type(ordinal) is not int for ordinal in observed_ordinals)
                or observed_ordinals != tuple(range(len(rows)))
            ):
                errors.append(f"checkpoint_ordinal_sequence:{attempt_id}")
            attempt_row = attempts_by_id.get(attempt_id)
            times = attempt_times.get(attempt_id, {})
            for row in rows:
                ordinal = row["ordinal"]
                try:
                    _code(row["checkpoint_name"], "checkpoint_name")
                except (TypeError, ValueError):
                    errors.append(f"checkpoint_name:{attempt_id}:{ordinal}")
                if not _is_sha256_text(row["artifact_hash"]):
                    errors.append(f"checkpoint_artifact_hash:{attempt_id}:{ordinal}")
                try:
                    _relative(row["location"])
                except (TypeError, ValueError):
                    errors.append(f"checkpoint_location:{attempt_id}:{ordinal}")
                recorded_at: datetime | None = None
                try:
                    recorded_at = _parse(row["recorded_at"])
                except (TypeError, ValueError):
                    errors.append(f"checkpoint_timestamp:{attempt_id}:{ordinal}")
                else:
                    if type(row["recorded_at"]) is not str or _iso(recorded_at) != row[
                        "recorded_at"
                    ]:
                        errors.append(f"checkpoint_timestamp:{attempt_id}:{ordinal}")
                if attempt_row is not None and recorded_at is not None:
                    lower = times.get("started_at") or times.get("reserved_at")
                    finished = times.get("finished_at")
                    lease_expires = times.get("lease_expires_at")
                    if (
                        lower is None
                        or recorded_at < lower
                        or (finished is not None and recorded_at > finished)
                        or (
                            lease_expires is not None
                            and recorded_at >= lease_expires
                        )
                    ):
                        errors.append(
                            f"checkpoint_timestamp_order:{attempt_id}:{ordinal}"
                        )

        for row in attempts:
            attempt_id = int(row["attempt_id"])
            is_legacy = str(row["reservation_request_hash"]) == (
                _legacy_unavailable_reservation_request_hash(row)
            )
            if (
                not is_legacy
                and row["status"] == "succeeded"
                and row["result_hash"]
                not in checkpoint_hashes_by_attempt.get(attempt_id, set())
            ):
                errors.append(f"attempt_success_checkpoint:{attempt_id}")
        return tuple(dict.fromkeys(errors))

    @staticmethod
    def _event_chain_errors(connection: sqlite3.Connection) -> tuple[str, ...]:
        errors: list[str] = []
        previous: str | None = None
        expected_sequence = 1
        try:
            rows = connection.execute(
                "SELECT * FROM runtime_events ORDER BY sequence"
            ).fetchall()
        except sqlite3.DatabaseError:
            return ("runtime_events_unreadable",)
        for row in rows:
            sequence = int(row["sequence"])
            if sequence != expected_sequence:
                errors.append(f"event_sequence:{sequence}")
            try:
                _code(str(row["event_type"]), "runtime event_type")
            except ValueError:
                errors.append(f"event_type:{sequence}")
            try:
                canonical_event_at = _iso(_parse(str(row["event_at"])))
            except (TypeError, ValueError):
                errors.append(f"event_at:{sequence}")
            else:
                if canonical_event_at != row["event_at"]:
                    errors.append(f"event_at_canonical:{sequence}")
            if row["previous_event_hash"] != previous:
                errors.append(f"event_chain_previous:{sequence}")
            if row["previous_event_hash"] is not None and not _is_sha256_text(
                row["previous_event_hash"]
            ):
                errors.append(f"event_previous_hash_format:{sequence}")
            if not _is_sha256_text(row["payload_hash"]):
                errors.append(f"event_payload_hash:{sequence}")
            if not _is_sha256_text(row["event_hash"]):
                errors.append(f"event_hash_format:{sequence}")
            core = {
                "event_type": row["event_type"],
                "event_at": row["event_at"],
                "payload_hash": row["payload_hash"],
                "previous_event_hash": row["previous_event_hash"],
            }
            if hash_json(core) != row["event_hash"]:
                errors.append(f"event_hash:{sequence}")
            previous = str(row["event_hash"])
            expected_sequence += 1
        try:
            sequence_row = connection.execute(
                "SELECT seq FROM sqlite_sequence WHERE name='runtime_events'"
            ).fetchone()
        except sqlite3.DatabaseError:
            errors.append("event_sequence_head_unreadable")
        else:
            durable_head = 0 if sequence_row is None else int(sequence_row["seq"])
            observed_head = 0 if not rows else int(rows[-1]["sequence"])
            if durable_head != observed_head:
                errors.append("event_sequence_head")
        return tuple(errors)

    def reserve(
        self,
        stage: str,
        *,
        worker_id: str,
        lease_token: str,
        idempotency_key: str,
        input_hash: str,
        lease_seconds: int,
        estimated_wall_seconds: float,
        now: datetime | None = None,
    ) -> StageAttempt:
        """Reserve legacy serial work in a deterministic stage-only scope.

        Bounded callers must use :meth:`reserve_scoped`; this compatibility
        entry point deliberately preserves the historical one-task-per-stage
        semantics and never accepts a caller-supplied bounded scope.
        """

        return self._reserve(
            stage,
            attempt_scope_hash=_legacy_attempt_scope_hash(
                self.spec.content_hash, stage
            ),
            worker_id=worker_id,
            lease_token=lease_token,
            idempotency_key=idempotency_key,
            input_hash=input_hash,
            lease_seconds=lease_seconds,
            estimated_wall_seconds=estimated_wall_seconds,
            now=now,
        )

    def reserve_scoped(
        self,
        stage: str,
        *,
        attempt_scope_hash: str,
        worker_id: str,
        lease_token: str,
        idempotency_key: str,
        input_hash: str,
        lease_seconds: int,
        estimated_wall_seconds: float,
        now: datetime | None = None,
    ) -> StageAttempt:
        """Reserve work in an explicit task/candidate scope.

        There is intentionally no default scope on this API.  A bounded task
        must bind its full immutable identity before entering the runtime.
        """

        return self._reserve(
            stage,
            attempt_scope_hash=require_sha256(
                attempt_scope_hash, name="attempt_scope_hash"
            ),
            worker_id=worker_id,
            lease_token=lease_token,
            idempotency_key=idempotency_key,
            input_hash=input_hash,
            lease_seconds=lease_seconds,
            estimated_wall_seconds=estimated_wall_seconds,
            now=now,
        )

    def _reserve(
        self,
        stage: str,
        *,
        attempt_scope_hash: str,
        worker_id: str,
        lease_token: str,
        idempotency_key: str,
        input_hash: str,
        lease_seconds: int,
        estimated_wall_seconds: float,
        now: datetime | None,
    ) -> StageAttempt:
        if stage not in self.spec.stages:
            raise ValueError(f"stage is not enabled by ExperimentSpec:{stage}")
        scope_digest = require_sha256(
            attempt_scope_hash, name="attempt_scope_hash"
        )
        worker = _code(worker_id, "worker_id")
        idempotency = _code(idempotency_key, "idempotency_key")
        token_hash = _secret_hash(lease_token)
        input_digest = require_sha256(input_hash, name="attempt input_hash")
        if (
            not isinstance(lease_seconds, int)
            or isinstance(lease_seconds, bool)
            or lease_seconds <= 0
        ):
            raise ValueError("lease_seconds must be a positive integer")
        if (
            not isinstance(estimated_wall_seconds, (int, float))
            or isinstance(estimated_wall_seconds, bool)
            or not math.isfinite(float(estimated_wall_seconds))
        ):
            raise ValueError("estimated_wall_seconds must be a finite positive number")
        estimate = float(estimated_wall_seconds)
        if estimate <= 0 or estimate > self.spec.resource_budget.maximum_wall_seconds:
            raise ResourceBudgetExceeded(
                "attempt wall estimate exceeds experiment budget"
            )
        timestamp = now or _utc_now()
        policy_error: RuntimeErrorBase | None = None
        reserved: StageAttempt | None = None
        with self._transaction() as connection:
            self._expire_locked(connection, timestamp)
            try:
                reserved = self._reserve_locked(
                    connection,
                    stage=stage,
                    attempt_scope_hash=scope_digest,
                    worker=worker,
                    token_hash=token_hash,
                    idempotency=idempotency,
                    input_digest=input_digest,
                    lease_seconds=lease_seconds,
                    estimate=estimate,
                    timestamp=timestamp,
                )
            except (
                AttemptLeaseConflict,
                DispatchStopped,
                ResourceBudgetExceeded,
                RetryNotReady,
            ) as exc:
                # Stale lease terminalization is a durable scheduler transition,
                # not part of a rejected reservation.  Keep the legacy behavior
                # while still performing both actions under this one lock/txn.
                policy_error = exc
        if policy_error is not None:
            raise policy_error
        if reserved is None:  # pragma: no cover - defensive control-flow guard
            raise RuntimeError("runtime reservation produced no result")
        return reserved

    def _reserve_locked(
        self,
        connection: sqlite3.Connection,
        *,
        stage: str,
        attempt_scope_hash: str,
        worker: str,
        token_hash: str,
        idempotency: str,
        input_digest: str,
        lease_seconds: int,
        estimate: float,
        timestamp: datetime,
    ) -> StageAttempt:
        if not connection.in_transaction:
            raise RuntimeErrorBase("runtime reserve requires an active transaction")
        request_hash = _reservation_request_hash(
            experiment_spec_hash=self.spec.content_hash,
            stage=stage,
            attempt_scope_hash=attempt_scope_hash,
            worker_id=worker,
            lease_token_hash=token_hash,
            idempotency_key=idempotency,
            input_hash=input_digest,
            lease_seconds=lease_seconds,
            estimated_wall_seconds=estimate,
        )
        existing = connection.execute(
            "SELECT * FROM stage_attempts WHERE idempotency_key=?",
            (idempotency,),
        ).fetchone()
        if existing is not None:
            if (
                existing["stage"] != stage
                or existing["attempt_scope_hash"] != attempt_scope_hash
                or existing["worker_id"] != worker
                or existing["lease_token_hash"] != token_hash
                or existing["input_hash"] != input_digest
                or float(existing["estimated_wall_seconds"]) != estimate
                or existing["reservation_request_hash"] != request_hash
            ):
                raise AttemptLeaseConflict("idempotency key payload differs")
            return _attempt(existing)
        self._assert_dispatch_locked(connection)
        scope_stage = connection.execute(
            """SELECT stage FROM stage_attempts
               WHERE attempt_scope_hash=? ORDER BY attempt_number LIMIT 1""",
            (attempt_scope_hash,),
        ).fetchone()
        if scope_stage is not None and str(scope_stage["stage"]) != stage:
            raise AttemptLeaseConflict(
                "attempt scope is already bound to a different stage"
            )
        active_scope = connection.execute(
            """SELECT attempt_id FROM stage_attempts
               WHERE attempt_scope_hash=? AND status IN ('leased','running')""",
            (attempt_scope_hash,),
        ).fetchone()
        if active_scope is not None:
            raise AttemptLeaseConflict(
                "stage attempt scope already has active attempt:"
                f"{active_scope['attempt_id']}"
            )
        usage = self._usage_locked(connection)
        budget = self.spec.resource_budget
        if usage.attempts_reserved >= budget.maximum_attempts:
            raise ResourceBudgetExceeded("experiment attempt budget is exhausted")
        if usage.attempts_active >= budget.maximum_parallel_tasks:
            raise ResourceBudgetExceeded("experiment parallel task budget is exhausted")
        active_wall_commitment = float(
            connection.execute(
                """SELECT COALESCE(SUM(estimated_wall_seconds), 0.0)
                   FROM stage_attempts
                   WHERE status IN ('leased','running')"""
            ).fetchone()[0]
        )
        if (
            usage.wall_seconds + active_wall_commitment + estimate
            > budget.maximum_wall_seconds
        ):
            raise ResourceBudgetExceeded("experiment projected wall budget is exhausted")
        scope_attempts = int(
            connection.execute(
                """SELECT COUNT(*) FROM stage_attempts
                   WHERE attempt_scope_hash=?""",
                (attempt_scope_hash,),
            ).fetchone()[0]
        )
        if scope_attempts >= self.spec.retry_policy.maximum_attempts_per_stage:
            raise ResourceBudgetExceeded("attempt scope retry budget is exhausted")
        previous = connection.execute(
            """SELECT status,failure_code,retry_not_before,input_hash
               FROM stage_attempts WHERE attempt_scope_hash=?
               ORDER BY attempt_number DESC LIMIT 1""",
            (attempt_scope_hash,),
        ).fetchone()
        if previous is not None:
            previous_status = str(previous["status"])
            previous_failure = previous["failure_code"]
            if previous_status == "succeeded":
                raise AttemptLeaseConflict("stage attempt scope already succeeded")
            if previous_status == "cancelled":
                raise AttemptLeaseConflict(
                    "cancelled attempt scope requires a new experiment"
                )
            if (
                previous_status in {"failed", "expired"}
                and previous_failure
                not in self.spec.retry_policy.retryable_failure_codes
            ):
                raise AttemptLeaseConflict(
                    "previous attempt scope failure is not retryable"
                )
            if previous["input_hash"] != input_digest:
                raise AttemptLeaseConflict("stage retry input_hash differs")
            retry_at = previous["retry_not_before"]
            if retry_at is not None and timestamp < _parse(retry_at):
                raise RetryNotReady(f"stage retry is not ready:{retry_at}")
        next_number = scope_attempts + 1
        now_text = _iso(timestamp)
        cursor = connection.execute(
            """INSERT INTO stage_attempts(
                experiment_spec_hash,stage,attempt_scope_hash,attempt_number,
                idempotency_key,status,worker_id,lease_token_hash,input_hash,
                reserved_at,started_at,heartbeat_at,lease_expires_at,
                reservation_request_hash,finished_at,retry_not_before,
                estimated_wall_seconds
            ) VALUES(?,?,?,?,?, 'leased', ?,?,?,?,NULL,?,?,?,NULL,NULL,?)""",
            (
                self.spec.content_hash,
                stage,
                attempt_scope_hash,
                next_number,
                idempotency,
                worker,
                token_hash,
                input_digest,
                now_text,
                now_text,
                _iso(timestamp + timedelta(seconds=lease_seconds)),
                request_hash,
                estimate,
            ),
        )
        if cursor.lastrowid is None:  # pragma: no cover
            raise RuntimeError("attempt insert did not return an identifier")
        attempt_id = int(cursor.lastrowid)
        self._event_locked(
            connection,
            "attempt_reserved",
            {
                "attempt_id": attempt_id,
                "stage": stage,
                "attempt_scope_hash": attempt_scope_hash,
                "worker_id": worker,
                "reservation_request_hash": request_hash,
            },
            now_text,
        )
        return self._get_locked(connection, attempt_id)

    def start(
        self,
        attempt_id: int,
        *,
        worker_id: str,
        lease_token: str,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> StageAttempt:
        return self._renew(
            attempt_id,
            worker_id=worker_id,
            lease_token=lease_token,
            lease_seconds=lease_seconds,
            from_status="leased",
            to_status="running",
            now=now,
        )

    def heartbeat(
        self,
        attempt_id: int,
        *,
        worker_id: str,
        lease_token: str,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> StageAttempt:
        return self._renew(
            attempt_id,
            worker_id=worker_id,
            lease_token=lease_token,
            lease_seconds=lease_seconds,
            from_status="running",
            to_status="running",
            now=now,
        )

    def _renew(
        self,
        attempt_id: int,
        *,
        worker_id: str,
        lease_token: str,
        lease_seconds: int,
        from_status: str,
        to_status: str,
        now: datetime | None,
    ) -> StageAttempt:
        worker = _code(worker_id, "worker_id")
        token_hash = _secret_hash(lease_token)
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        timestamp = now or _utc_now()
        self.expire_stale(now=timestamp)
        with self._transaction() as connection:
            self._expire_locked(connection, timestamp)
            self._assert_dispatch_locked(connection)
            started = (
                ",started_at=COALESCE(started_at,?)" if to_status == "running" else ""
            )
            parameters: list[object] = [
                to_status,
                _iso(timestamp),
                _iso(timestamp + timedelta(seconds=lease_seconds)),
            ]
            if started:
                parameters.append(_iso(timestamp))
            parameters.extend([attempt_id, from_status, worker, token_hash])
            cursor = connection.execute(
                f"""UPDATE stage_attempts SET status=?,heartbeat_at=?,lease_expires_at=?
                    {started} WHERE attempt_id=? AND status=? AND worker_id=?
                    AND lease_token_hash=?""",
                parameters,
            )
            if cursor.rowcount != 1:
                raise AttemptLeaseConflict(
                    "attempt lease is not owned by this worker/token"
                )
            return self._get_locked(connection, attempt_id)

    def record_checkpoint(
        self,
        attempt_id: int,
        *,
        worker_id: str,
        lease_token: str,
        checkpoint_name: str,
        artifact_hash: str,
        location: str,
        now: datetime | None = None,
    ) -> Checkpoint:
        worker = _code(worker_id, "worker_id")
        token_hash = _secret_hash(lease_token)
        name = _code(checkpoint_name, "checkpoint_name")
        digest = require_sha256(artifact_hash, name="checkpoint artifact_hash")
        path = _relative(location)
        timestamp = now or _utc_now()
        self.expire_stale(now=timestamp)
        with self._transaction() as connection:
            self._expire_locked(connection, timestamp)
            owner = connection.execute(
                """SELECT status,worker_id,lease_token_hash FROM stage_attempts
                   WHERE attempt_id=?""",
                (attempt_id,),
            ).fetchone()
            if (
                owner is None
                or owner["status"] != "running"
                or owner["worker_id"] != worker
                or owner["lease_token_hash"] != token_hash
            ):
                raise AttemptLeaseConflict(
                    "checkpoint writer does not own running attempt"
                )
            ordinal = int(
                connection.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE attempt_id=?", (attempt_id,)
                ).fetchone()[0]
            )
            try:
                connection.execute(
                    """INSERT INTO checkpoints(
                        attempt_id,ordinal,checkpoint_name,artifact_hash,location,recorded_at
                    ) VALUES(?,?,?,?,?,?)""",
                    (attempt_id, ordinal, name, digest, path, _iso(timestamp)),
                )
            except sqlite3.IntegrityError as exc:
                raise RuntimeErrorBase("checkpoint name is already frozen") from exc
            return Checkpoint(
                attempt_id=attempt_id,
                ordinal=ordinal,
                checkpoint_name=name,
                artifact_hash=digest,
                location=path,
                recorded_at=_iso(timestamp),
            )

    def succeed(
        self,
        attempt_id: int,
        *,
        worker_id: str,
        lease_token: str,
        result_hash: str,
        usage: RuntimeUsage,
        now: datetime | None = None,
    ) -> StageAttempt:
        result = require_sha256(result_hash, name="attempt result_hash")
        return self._finish(
            attempt_id,
            worker_id=worker_id,
            lease_token=lease_token,
            status="succeeded",
            result_hash=result,
            failure_code=None,
            usage=usage,
            now=now,
        )

    def fail(
        self,
        attempt_id: int,
        *,
        worker_id: str,
        lease_token: str,
        failure_code: str,
        usage: RuntimeUsage,
        now: datetime | None = None,
    ) -> StageAttempt:
        return self._finish(
            attempt_id,
            worker_id=worker_id,
            lease_token=lease_token,
            status="failed",
            result_hash=None,
            failure_code=_code(failure_code, "failure_code"),
            usage=usage,
            now=now,
        )

    def _finish(
        self,
        attempt_id: int,
        *,
        worker_id: str,
        lease_token: str,
        status: str,
        result_hash: str | None,
        failure_code: str | None,
        usage: RuntimeUsage,
        now: datetime | None,
    ) -> StageAttempt:
        worker = _code(worker_id, "worker_id")
        token_hash = _secret_hash(lease_token)
        _validate_reported_usage(usage)
        timestamp = now or _utc_now()
        self.expire_stale(now=timestamp)
        with self._transaction() as connection:
            self._expire_locked(connection, timestamp)
            current = self._get_locked(connection, attempt_id)
            if current.status is not AttemptStatus.RUNNING:
                raise AttemptLeaseConflict("only a running attempt can finish")
            if current.worker_id != worker:
                raise AttemptLeaseConflict("attempt worker differs")
            token = connection.execute(
                "SELECT lease_token_hash FROM stage_attempts WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()[0]
            if token != token_hash:
                raise AttemptLeaseConflict("attempt lease token differs")
            checkpoint_hashes = {
                str(row["artifact_hash"])
                for row in connection.execute(
                    "SELECT artifact_hash FROM checkpoints WHERE attempt_id=?",
                    (attempt_id,),
                ).fetchall()
            }
            if status == "succeeded" and result_hash not in checkpoint_hashes:
                raise RuntimeErrorBase(
                    "successful stage result must equal a frozen checkpoint hash"
                )
            cumulative = self._usage_locked(connection)
            budget_failure = _budget_failure(self.spec, cumulative, usage)
            terminal_status = "failed" if budget_failure is not None else status
            terminal_failure = budget_failure or failure_code
            retry_not_before = None
            if (
                terminal_status == "failed"
                and terminal_failure in self.spec.retry_policy.retryable_failure_codes
                and current.attempt_number
                < self.spec.retry_policy.maximum_attempts_per_stage
            ):
                delay = self.spec.retry_policy.delay_seconds(
                    current.attempt_number - 1, jitter_unit=0.5
                )
                retry_not_before = _iso(timestamp + timedelta(seconds=delay))
            cursor = connection.execute(
                """UPDATE stage_attempts SET status=?,finished_at=?,retry_not_before=?,
                    actual_wall_seconds=?,actual_cpu_seconds=?,peak_memory_bytes=?,
                    disk_write_bytes=?,llm_calls=?,llm_tokens=?,llm_cost_microusd=?,
                    result_hash=?,failure_code=?
                   WHERE attempt_id=? AND status='running' AND worker_id=?
                     AND lease_token_hash=?""",
                (
                    terminal_status,
                    _iso(timestamp),
                    retry_not_before,
                    usage.wall_seconds,
                    usage.cpu_seconds,
                    usage.peak_memory_bytes,
                    usage.disk_write_bytes,
                    usage.llm_calls,
                    usage.llm_tokens,
                    usage.llm_cost_microusd,
                    result_hash if terminal_status == "succeeded" else None,
                    terminal_failure,
                    attempt_id,
                    worker,
                    token_hash,
                ),
            )
            if cursor.rowcount != 1:  # pragma: no cover
                raise AttemptLeaseConflict("attempt finish lost its lease")
            self._event_locked(
                connection,
                "attempt_finished",
                {
                    "attempt_id": attempt_id,
                    "status": terminal_status,
                    "failure_code": terminal_failure,
                },
                _iso(timestamp),
            )
            return self._get_locked(connection, attempt_id)

    def request_stop(
        self,
        *,
        actor: str,
        reason_code: str,
        now: datetime | None = None,
    ) -> int:
        operator = _code(actor, "stop actor")
        reason = _code(reason_code, "stop reason_code")
        timestamp = _iso(now or _utc_now())
        with self._transaction() as connection:
            connection.execute(
                """UPDATE runtime_control SET state='stop_requested',actor=?,
                   reason_code=?,updated_at=? WHERE singleton=1""",
                (operator, reason, timestamp),
            )
            cursor = connection.execute(
                """UPDATE stage_attempts SET status='cancelled',finished_at=?,
                   failure_code='kill_switch'
                   WHERE status IN ('leased','running')""",
                (timestamp,),
            )
            count = int(cursor.rowcount)
            self._event_locked(
                connection,
                "kill_switch_requested",
                {"actor": operator, "reason_code": reason, "cancelled_attempts": count},
                timestamp,
            )
            return count

    def expire_stale(self, *, now: datetime | None = None) -> int:
        timestamp = now or _utc_now()
        with self._transaction() as connection:
            return self._expire_locked(connection, timestamp)

    def clear_stop(
        self,
        *,
        actor: str,
        reason_code: str,
        now: datetime | None = None,
    ) -> None:
        operator = _code(actor, "clear actor")
        reason = _code(reason_code, "clear reason_code")
        timestamp = _iso(now or _utc_now())
        with self._transaction() as connection:
            connection.execute(
                """UPDATE runtime_control SET state='running',actor=?,reason_code=?,
                   updated_at=? WHERE singleton=1""",
                (operator, reason, timestamp),
            )
            self._event_locked(
                connection,
                "kill_switch_cleared",
                {"actor": operator, "reason_code": reason},
                timestamp,
            )

    def dispatch_is_stopped(self) -> bool:
        row = self.connection.execute(
            "SELECT state FROM runtime_control WHERE singleton=1"
        ).fetchone()
        return row is None or row["state"] != "running"

    def assert_circuit_closed(
        self, service: str, *, now: datetime | None = None
    ) -> None:
        name = _code(service, "circuit service")
        timestamp = now or _utc_now()
        row = self.connection.execute(
            "SELECT state,open_until FROM circuit_breakers WHERE service=?", (name,)
        ).fetchone()
        if row is not None and row["state"] == "open":
            if row["open_until"] is None or timestamp < _parse(row["open_until"]):
                raise CircuitOpen(f"circuit is open:{name}")

    def record_service_success(
        self, service: str, *, now: datetime | None = None
    ) -> None:
        name = _code(service, "circuit service")
        timestamp = _iso(now or _utc_now())
        with self._transaction() as connection:
            connection.execute(
                """INSERT INTO circuit_breakers(
                    service,failure_count,state,open_until,updated_at
                ) VALUES(?,0,'closed',NULL,?)
                ON CONFLICT(service) DO UPDATE SET failure_count=0,state='closed',
                    open_until=NULL,updated_at=excluded.updated_at""",
                (name, timestamp),
            )

    def record_service_failure(
        self,
        service: str,
        *,
        failure_threshold: int,
        cooldown_seconds: int,
        retry_after_seconds: int | None = None,
        now: datetime | None = None,
    ) -> None:
        name = _code(service, "circuit service")
        if failure_threshold <= 0 or cooldown_seconds <= 0:
            raise ValueError("circuit threshold/cooldown must be positive")
        timestamp = now or _utc_now()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT failure_count FROM circuit_breakers WHERE service=?", (name,)
            ).fetchone()
            failures = (0 if row is None else int(row["failure_count"])) + 1
            open_state = (
                failures >= failure_threshold or retry_after_seconds is not None
            )
            cooldown = max(cooldown_seconds, retry_after_seconds or 0)
            connection.execute(
                """INSERT INTO circuit_breakers(
                    service,failure_count,state,open_until,updated_at
                ) VALUES(?,?,?,?,?)
                ON CONFLICT(service) DO UPDATE SET
                    failure_count=excluded.failure_count,state=excluded.state,
                    open_until=excluded.open_until,updated_at=excluded.updated_at""",
                (
                    name,
                    failures,
                    "open" if open_state else "closed",
                    (
                        _iso(timestamp + timedelta(seconds=cooldown))
                        if open_state
                        else None
                    ),
                    _iso(timestamp),
                ),
            )

    def get_attempt(self, attempt_id: int) -> StageAttempt:
        return self._get_locked(self.connection, attempt_id)

    def list_attempts(
        self,
        stage: str | None = None,
        *,
        attempt_scope_hash: str | None = None,
    ) -> tuple[StageAttempt, ...]:
        scope = (
            None
            if attempt_scope_hash is None
            else require_sha256(attempt_scope_hash, name="attempt_scope_hash")
        )
        if stage is None:
            if scope is None:
                rows = self.connection.execute(
                    "SELECT attempt_id FROM stage_attempts ORDER BY attempt_id"
                ).fetchall()
            else:
                rows = self.connection.execute(
                    """SELECT attempt_id FROM stage_attempts
                       WHERE attempt_scope_hash=? ORDER BY attempt_number""",
                    (scope,),
                ).fetchall()
        else:
            if stage not in self.spec.stages:
                raise ValueError(f"stage is not enabled by ExperimentSpec:{stage}")
            if scope is None:
                rows = self.connection.execute(
                    """SELECT attempt_id FROM stage_attempts
                       WHERE stage=? ORDER BY attempt_scope_hash,attempt_number""",
                    (stage,),
                ).fetchall()
            else:
                rows = self.connection.execute(
                    """SELECT attempt_id FROM stage_attempts
                       WHERE stage=? AND attempt_scope_hash=?
                       ORDER BY attempt_number""",
                    (stage, scope),
                ).fetchall()
        return tuple(self.get_attempt(int(row["attempt_id"])) for row in rows)

    def successful_attempt(
        self, stage: str, *, attempt_scope_hash: str | None = None
    ) -> StageAttempt | None:
        if stage not in self.spec.stages:
            raise ValueError(f"stage is not enabled by ExperimentSpec:{stage}")
        scope = (
            _legacy_attempt_scope_hash(self.spec.content_hash, stage)
            if attempt_scope_hash is None
            else require_sha256(attempt_scope_hash, name="attempt_scope_hash")
        )
        row = self.connection.execute(
            """SELECT attempt_id FROM stage_attempts
               WHERE stage=? AND attempt_scope_hash=? AND status='succeeded'
               ORDER BY attempt_number DESC LIMIT 1""",
            (stage, scope),
        ).fetchone()
        return None if row is None else self.get_attempt(int(row["attempt_id"]))

    def list_checkpoints(self, attempt_id: int) -> tuple[Checkpoint, ...]:
        rows = self.connection.execute(
            "SELECT * FROM checkpoints WHERE attempt_id=? ORDER BY ordinal",
            (attempt_id,),
        ).fetchall()
        return tuple(Checkpoint(**dict(row)) for row in rows)

    def usage(self) -> RuntimeUsage:
        return self._usage_locked(self.connection)

    def verify_integrity(self) -> tuple[str, ...]:
        try:
            self._assert_database_path_identity()
        except RuntimeErrorBase:
            return ("runtime_database_path_identity",)
        # A runtime writer can otherwise commit between the event-row scan and
        # sqlite_sequence head read, producing a false corruption verdict.  The
        # existing transaction boundary supplies a coherent snapshot and also
        # rechecks pathname identity before and after attestation.
        with self._transaction() as connection:
            return self._schema_attestation_errors(
                connection, EXPERIMENT_RUNTIME_SCHEMA
            )

    def _expire_locked(self, connection: sqlite3.Connection, now: datetime) -> int:
        rows = connection.execute(
            """SELECT attempt_id,attempt_number,stage FROM stage_attempts
               WHERE status IN ('leased','running') AND lease_expires_at <= ?""",
            (_iso(now),),
        ).fetchall()
        for row in rows:
            retry_at = None
            if (
                "worker_lost" in self.spec.retry_policy.retryable_failure_codes
                and int(row["attempt_number"])
                < self.spec.retry_policy.maximum_attempts_per_stage
            ):
                delay = self.spec.retry_policy.delay_seconds(
                    int(row["attempt_number"]) - 1, jitter_unit=0.5
                )
                retry_at = _iso(now + timedelta(seconds=delay))
            connection.execute(
                """UPDATE stage_attempts SET status='expired',finished_at=?,
                   failure_code='worker_lost',retry_not_before=? WHERE attempt_id=?""",
                (_iso(now), retry_at, int(row["attempt_id"])),
            )
        return len(rows)

    @staticmethod
    def _get_locked(connection: sqlite3.Connection, attempt_id: int) -> StageAttempt:
        row = connection.execute(
            "SELECT * FROM stage_attempts WHERE attempt_id=?", (attempt_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown stage attempt:{attempt_id}")
        return _attempt(row)

    @staticmethod
    def _usage_locked(connection: sqlite3.Connection) -> RuntimeUsage:
        counts = {
            row["status"]: int(row["count"])
            for row in connection.execute(
                "SELECT status,COUNT(*) AS count FROM stage_attempts GROUP BY status"
            ).fetchall()
        }
        sums = connection.execute(
            """SELECT
                COALESCE(SUM(actual_wall_seconds),0),
                COALESCE(SUM(actual_cpu_seconds),0),
                COALESCE(MAX(peak_memory_bytes),0),
                COALESCE(SUM(disk_write_bytes),0),
                COALESCE(SUM(llm_calls),0),
                COALESCE(SUM(llm_tokens),0),
                COALESCE(SUM(llm_cost_microusd),0)
               FROM stage_attempts"""
        ).fetchone()
        return RuntimeUsage(
            attempts_reserved=sum(counts.values()),
            attempts_active=counts.get("leased", 0) + counts.get("running", 0),
            attempts_succeeded=counts.get("succeeded", 0),
            attempts_failed=counts.get("failed", 0),
            attempts_expired=counts.get("expired", 0),
            attempts_cancelled=counts.get("cancelled", 0),
            wall_seconds=float(sums[0]),
            cpu_seconds=float(sums[1]),
            peak_memory_bytes=int(sums[2]),
            disk_write_bytes=int(sums[3]),
            llm_calls=int(sums[4]),
            llm_tokens=int(sums[5]),
            llm_cost_microusd=int(sums[6]),
        )

    @staticmethod
    def _assert_dispatch_locked(connection: sqlite3.Connection) -> None:
        row = connection.execute(
            "SELECT state,actor,reason_code FROM runtime_control WHERE singleton=1"
        ).fetchone()
        if row is None or row["state"] != "running":
            actor = "unknown" if row is None else row["actor"]
            reason = "unknown" if row is None else row["reason_code"]
            raise DispatchStopped(f"experiment dispatch stopped:{actor}:{reason}")

    @staticmethod
    def _event_locked(
        connection: sqlite3.Connection,
        event_type: str,
        payload: dict[str, object],
        event_at: str,
    ) -> None:
        previous_row = connection.execute(
            "SELECT event_hash FROM runtime_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        previous = None if previous_row is None else str(previous_row["event_hash"])
        payload_hash = hash_json(payload)
        core = {
            "event_type": event_type,
            "event_at": event_at,
            "payload_hash": payload_hash,
            "previous_event_hash": previous,
        }
        connection.execute(
            """INSERT INTO runtime_events(
                event_type,event_at,payload_hash,previous_event_hash,event_hash
            ) VALUES(?,?,?,?,?)""",
            (event_type, event_at, payload_hash, previous, hash_json(core)),
        )


def zero_usage() -> RuntimeUsage:
    return RuntimeUsage(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0)


def reported_usage(
    *,
    wall_seconds: float,
    cpu_seconds: float,
    peak_memory_bytes: int,
    disk_write_bytes: int,
    llm_calls: int = 0,
    llm_tokens: int = 0,
    llm_cost_microusd: int = 0,
) -> RuntimeUsage:
    return RuntimeUsage(
        0,
        0,
        0,
        0,
        0,
        0,
        wall_seconds,
        cpu_seconds,
        peak_memory_bytes,
        disk_write_bytes,
        llm_calls,
        llm_tokens,
        llm_cost_microusd,
    )


def _budget_failure(
    spec: ExperimentSpec, current: RuntimeUsage, extra: RuntimeUsage
) -> str | None:
    budget = spec.resource_budget
    checks = (
        (
            current.wall_seconds + extra.wall_seconds,
            budget.maximum_wall_seconds,
            "wall",
        ),
        (current.cpu_seconds + extra.cpu_seconds, budget.maximum_cpu_seconds, "cpu"),
        (extra.peak_memory_bytes, budget.maximum_peak_memory_bytes, "memory"),
        (
            current.disk_write_bytes + extra.disk_write_bytes,
            budget.maximum_disk_write_bytes,
            "disk",
        ),
        (current.llm_calls + extra.llm_calls, budget.maximum_llm_calls, "llm_calls"),
        (
            current.llm_tokens + extra.llm_tokens,
            budget.maximum_llm_tokens,
            "llm_tokens",
        ),
        (
            current.llm_cost_microusd + extra.llm_cost_microusd,
            budget.maximum_llm_cost_microusd,
            "llm_cost",
        ),
    )
    for observed, limit, name in checks:
        if observed > limit:
            return f"resource_budget_{name}_exceeded"
    return None


def _validate_reported_usage(value: RuntimeUsage) -> None:
    continuous = (value.wall_seconds, value.cpu_seconds)
    if any(
        not isinstance(item, (int, float))
        or isinstance(item, bool)
        or not math.isfinite(float(item))
        or item < 0
        for item in continuous
    ):
        raise ValueError("reported runtime duration must be finite and non-negative")
    integer_resources = (
        value.peak_memory_bytes,
        value.disk_write_bytes,
        value.llm_calls,
        value.llm_tokens,
        value.llm_cost_microusd,
    )
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in integer_resources
    ):
        raise ValueError("reported runtime counters must be non-negative integers")


def _attempt(row: sqlite3.Row) -> StageAttempt:
    return StageAttempt(
        attempt_id=int(row["attempt_id"]),
        experiment_spec_hash=row["experiment_spec_hash"],
        stage=row["stage"],
        attempt_scope_hash=row["attempt_scope_hash"],
        attempt_number=int(row["attempt_number"]),
        idempotency_key=row["idempotency_key"],
        status=AttemptStatus(row["status"]),
        worker_id=row["worker_id"],
        input_hash=row["input_hash"],
        reserved_at=row["reserved_at"],
        started_at=row["started_at"],
        heartbeat_at=row["heartbeat_at"],
        lease_expires_at=row["lease_expires_at"],
        reservation_request_hash=row["reservation_request_hash"],
        finished_at=row["finished_at"],
        retry_not_before=row["retry_not_before"],
        estimated_wall_seconds=float(row["estimated_wall_seconds"]),
        actual_wall_seconds=_float_or_none(row["actual_wall_seconds"]),
        actual_cpu_seconds=_float_or_none(row["actual_cpu_seconds"]),
        peak_memory_bytes=_int_or_none(row["peak_memory_bytes"]),
        disk_write_bytes=_int_or_none(row["disk_write_bytes"]),
        llm_calls=_int_or_none(row["llm_calls"]),
        llm_tokens=_int_or_none(row["llm_tokens"]),
        llm_cost_microusd=_int_or_none(row["llm_cost_microusd"]),
        result_hash=row["result_hash"],
        failure_code=row["failure_code"],
    )


def _float_or_none(value: object) -> float | None:
    return None if value is None else float(str(value))


def _int_or_none(value: object) -> int | None:
    return None if value is None else int(str(value))


def bounded_attempt_scope_hash(
    *,
    experiment_spec_hash: str,
    stage: str,
    task_natural_key: str,
    candidate_lineage_hash: str,
    source_capsule_hash: str,
    sealed_process_task_hash: str,
) -> str:
    """Return the canonical runtime scope for one immutable bounded task."""

    return cast(
        str,
        hash_json(
            {
                "schema_version": "bounded-runtime-attempt-scope/v1",
                "experiment_spec_hash": require_sha256(
                    experiment_spec_hash, name="experiment_spec_hash"
                ),
                "stage": _code(stage, "attempt stage"),
                "task_natural_key": require_sha256(
                    task_natural_key, name="task_natural_key"
                ),
                "candidate_lineage_hash": require_sha256(
                    candidate_lineage_hash, name="candidate_lineage_hash"
                ),
                "source_capsule_hash": require_sha256(
                    source_capsule_hash, name="source_capsule_hash"
                ),
                "sealed_process_task_hash": require_sha256(
                    sealed_process_task_hash, name="sealed_process_task_hash"
                ),
            }
        ),
    )


def _legacy_attempt_scope_hash(experiment_spec_hash: str, stage: str) -> str:
    return cast(
        str,
        hash_json(
            {
                "schema_version": "legacy-runtime-attempt-scope/v1",
                "experiment_spec_hash": require_sha256(
                    experiment_spec_hash, name="experiment_spec_hash"
                ),
                "stage": _code(stage, "attempt stage"),
            }
        ),
    )


def _reservation_request_hash(
    *,
    experiment_spec_hash: str,
    stage: str,
    attempt_scope_hash: str,
    worker_id: str,
    lease_token_hash: str,
    idempotency_key: str,
    input_hash: str,
    lease_seconds: int,
    estimated_wall_seconds: float,
) -> str:
    return cast(
        str,
        hash_json(
            {
                "schema_version": "runtime-reservation-request/v1",
                "experiment_spec_hash": experiment_spec_hash,
                "stage": stage,
                "attempt_scope_hash": attempt_scope_hash,
                "worker_id": worker_id,
                "lease_token_hash": lease_token_hash,
                "idempotency_key": idempotency_key,
                "input_hash": input_hash,
                "lease_seconds": lease_seconds,
                "estimated_wall_seconds": estimated_wall_seconds,
            }
        ),
    )


def _legacy_unavailable_reservation_request_hash(row: sqlite3.Row) -> str:
    """Return an unmatchable marker for v1 requests whose lease was mutable.

    v1 overwrote ``heartbeat_at`` and ``lease_expires_at`` on every renewal, so
    the original reservation lease cannot be reconstructed authoritatively.
    The marker keeps historical attempts readable while making idempotent
    replay of an unknowable legacy request fail closed after migration.
    """

    return cast(
        str,
        hash_json(
            {
                "schema_version": "legacy-reservation-request-unavailable/v1",
                "attempt_id": int(row["attempt_id"]),
                "idempotency_key": str(row["idempotency_key"]),
            }
        ),
    )


def _secret_hash(value: str) -> str:
    if not isinstance(value, str) or len(value) < 16:
        raise ValueError("lease_token must contain at least 16 characters")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _code(value: str, name: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        raise ValueError(f"{name} is invalid")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-")
    if not value[0].isalnum() or any(character not in allowed for character in value):
        raise ValueError(f"{name} is invalid")
    return value


def _normalized_schema_sql(value: object) -> str:
    """Normalize only known SQLite rename/format noise, not SQL literals.

    SQLite adds double quotes when tables are renamed during the authenticated
    v1-to-v2 migration.  Whitespace and those identifier quotes are the only
    tolerated differences.  In particular, case is preserved so CHECK string
    literals cannot be changed from e.g. ``'running'`` to ``'RUNNING'``.
    """

    if not isinstance(value, str):
        return ""
    normalized: list[str] = []
    in_single_quoted_literal = False
    index = 0
    while index < len(value):
        character = value[index]
        if character == "'":
            normalized.append(character)
            if (
                in_single_quoted_literal
                and index + 1 < len(value)
                and value[index + 1] == "'"
            ):
                normalized.append("'")
                index += 2
                continue
            in_single_quoted_literal = not in_single_quoted_literal
        elif not in_single_quoted_literal and character.isspace():
            index += 1
            continue
        elif not in_single_quoted_literal and character == '"':
            index += 1
            continue
        else:
            normalized.append(character)
        index += 1
    return "".join(normalized)


def _is_sha256_text(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError("checkpoint location must be a normalized relative path")
    return value


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("runtime timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _parse(value: str) -> datetime:
    timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("runtime timestamp must be timezone-aware")
    return timestamp.astimezone(timezone.utc)


__all__ = [
    "AttemptLeaseConflict",
    "AttemptStatus",
    "Checkpoint",
    "CircuitOpen",
    "DispatchStopped",
    "EXPERIMENT_RUNTIME_SCHEMA",
    "ExperimentRuntime",
    "ResourceBudgetExceeded",
    "RetryNotReady",
    "RuntimeErrorBase",
    "RuntimeUsage",
    "StageAttempt",
    "bounded_attempt_scope_hash",
    "reported_usage",
    "zero_usage",
]
