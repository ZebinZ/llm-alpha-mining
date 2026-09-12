from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_alpha_mining.mining.artifacts.hashing import canonical_json_bytes, hash_json
from llm_alpha_mining.mining.artifacts.manifest import ArtifactRecord
from llm_alpha_mining.mining.candidate_protocol_authority import (
    CandidateProtocolAuthority,
    CandidateProtocolAuthorityError,
)
from llm_alpha_mining.mining.domain.enums import CandidateState, RunState
from llm_alpha_mining.mining.domain.models import CandidateSpec
from llm_alpha_mining.mining.orchestration.state import (
    assert_candidate_transition,
    assert_run_transition,
)


REPOSITORY_SCHEMA_VERSION = "1"


class RepositoryError(RuntimeError):
    pass


class DuplicateCandidateError(RepositoryError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    protocol_hash: str
    state: RunState
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    run_id: str
    candidate_id: str
    spec_hash: str
    state: CandidateState
    spec: CandidateSpec
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class EventRecord:
    event_id: int
    run_id: str
    event_type: str
    payload: dict[str, Any]
    payload_hash: str
    occurred_at: str


class SQLiteRepository:
    """Small authoritative control-plane repository.

    External teacher/official scores are intentionally absent from this schema.
    They remain in the file-drop holdout boundary and cannot be joined into a
    provider context by repository code.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.initialize_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SQLiteRepository":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        if self.connection.in_transaction:
            raise RepositoryError("nested repository transactions are not supported")
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            yield self.connection
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def initialize_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS repository_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                protocol_hash TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS candidates (
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                candidate_id TEXT NOT NULL,
                spec_hash TEXT NOT NULL,
                spec_json TEXT NOT NULL,
                generation INTEGER NOT NULL,
                state TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, candidate_id),
                UNIQUE (run_id, spec_hash)
            );
            CREATE TABLE IF NOT EXISTS state_transitions (
                transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                entity_type TEXT NOT NULL CHECK(entity_type IN ('run', 'candidate')),
                entity_id TEXT NOT NULL,
                from_state TEXT,
                to_state TEXT NOT NULL,
                reason TEXT NOT NULL,
                occurred_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                logical_name TEXT NOT NULL,
                location TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                media_type TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (run_id, logical_name)
            );
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                occurred_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS candidate_protocol_authorities (
                run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
                control_protocol_hash TEXT NOT NULL,
                candidate_protocol_hash TEXT NOT NULL,
                authority_hash TEXT NOT NULL UNIQUE,
                authority_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                CHECK(control_protocol_hash <> candidate_protocol_hash)
            );
            CREATE TRIGGER IF NOT EXISTS candidate_protocol_authorities_immutable_update
            BEFORE UPDATE ON candidate_protocol_authorities
            BEGIN
                SELECT RAISE(ABORT, 'candidate protocol authority is immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS candidate_protocol_authorities_immutable_delete
            BEFORE DELETE ON candidate_protocol_authorities
            BEGIN
                SELECT RAISE(ABORT, 'candidate protocol authority is immutable');
            END;
            """
        )
        existing = self.connection.execute(
            "SELECT value FROM repository_metadata WHERE key='schema_version'"
        ).fetchone()
        if existing is None:
            self.connection.execute(
                "INSERT INTO repository_metadata(key, value) VALUES ('schema_version', ?)",
                (REPOSITORY_SCHEMA_VERSION,),
            )
            self.connection.commit()
        elif existing["value"] != REPOSITORY_SCHEMA_VERSION:
            raise RepositoryError(
                f"unsupported repository schema {existing['value']!r}; expected {REPOSITORY_SCHEMA_VERSION!r}"
            )

    def create_run(self, run_id: str, protocol_hash: str) -> RunRecord:
        if not run_id.strip() or not protocol_hash.strip():
            raise ValueError("run_id and protocol_hash must not be empty")
        now = _utc_now()
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO runs(run_id, protocol_hash, state, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (run_id, protocol_hash, RunState.CREATED.value, now, now),
            )
            connection.execute(
                """INSERT INTO state_transitions(
                    run_id, entity_type, entity_id, from_state, to_state, reason, occurred_at
                ) VALUES (?, 'run', ?, NULL, ?, 'run_created', ?)""",
                (run_id, run_id, RunState.CREATED.value, now),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> RunRecord:
        row = self.connection.execute(
            "SELECT * FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown run: {run_id}")
        return RunRecord(
            run_id=row["run_id"],
            protocol_hash=row["protocol_hash"],
            state=RunState(row["state"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def transition_run(
        self, run_id: str, target: RunState, *, reason: str
    ) -> RunRecord:
        current = self.get_run(run_id)
        assert_run_transition(current.state, target)
        now = _utc_now()
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE runs SET state=?, updated_at=? WHERE run_id=? AND state=?",
                (target.value, now, run_id, current.state.value),
            )
            if cursor.rowcount != 1:
                raise RepositoryError("run state changed concurrently")
            connection.execute(
                """INSERT INTO state_transitions(
                    run_id, entity_type, entity_id, from_state, to_state, reason, occurred_at
                ) VALUES (?, 'run', ?, ?, ?, ?, ?)""",
                (run_id, run_id, current.state.value, target.value, reason, now),
            )
        return self.get_run(run_id)

    def authorize_candidate_protocol(
        self,
        run_id: str,
        authority: CandidateProtocolAuthority,
    ) -> CandidateProtocolAuthority:
        """Persist the sole immutable alternate candidate-protocol authority."""

        run = self.get_run(run_id)
        if run.state is not RunState.CREATED:
            raise RepositoryError(
                "candidate protocol authority must be installed before run initialization"
            )
        if authority.run_id != run_id:
            raise RepositoryError("candidate protocol authority run_id mismatch")
        if authority.control_protocol_content_sha256 != run.protocol_hash:
            raise RepositoryError("candidate protocol authority control hash mismatch")
        payload = canonical_json_bytes(authority.to_dict()).decode("utf-8")
        try:
            with self._transaction() as connection:
                connection.execute(
                    """INSERT INTO candidate_protocol_authorities(
                        run_id,control_protocol_hash,candidate_protocol_hash,
                        authority_hash,authority_json,created_at
                    ) VALUES (?,?,?,?,?,?)""",
                    (
                        run_id,
                        authority.control_protocol_content_sha256,
                        authority.candidate_protocol_content_sha256,
                        authority.content_hash,
                        payload,
                        _utc_now(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise RepositoryError(
                "candidate protocol authority already exists or conflicts"
            ) from exc
        return self.get_candidate_protocol_authority(run_id)  # type: ignore[return-value]

    def get_candidate_protocol_authority(
        self, run_id: str
    ) -> CandidateProtocolAuthority | None:
        self.get_run(run_id)
        row = self.connection.execute(
            "SELECT * FROM candidate_protocol_authorities WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        try:
            authority = CandidateProtocolAuthority.from_dict(
                json.loads(row["authority_json"])
            )
        except (json.JSONDecodeError, CandidateProtocolAuthorityError) as exc:
            raise RepositoryError(
                "stored candidate protocol authority is invalid"
            ) from exc
        if (
            row["control_protocol_hash"] != authority.control_protocol_content_sha256
            or row["candidate_protocol_hash"]
            != authority.candidate_protocol_content_sha256
            or row["authority_hash"] != authority.content_hash
        ):
            raise RepositoryError("stored candidate protocol authority columns differ")
        return authority

    def _candidate_protocol_is_authorized(
        self, run_id: str, candidate_id: str, protocol_hash: str
    ) -> bool:
        run = self.get_run(run_id)
        authority = self.get_candidate_protocol_authority(run_id)
        if authority is not None and candidate_id not in authority.candidate_ids:
            return False
        if protocol_hash == run.protocol_hash:
            return True
        return (
            authority is not None
            and authority.candidate_protocol_content_sha256 == protocol_hash
        )

    def add_candidate(
        self,
        run_id: str,
        spec: CandidateSpec,
        *,
        initial_state: CandidateState = CandidateState.DRAFT,
    ) -> CandidateRecord:
        run = self.get_run(run_id)
        if run.state not in {RunState.INITIALIZED, RunState.RUNNING}:
            raise RepositoryError(
                f"candidates cannot be registered while run is {run.state.value}"
            )
        if not self._candidate_protocol_is_authorized(
            run_id, spec.candidate_id, spec.protocol_hash
        ):
            raise RepositoryError(
                "candidate identity or protocol hash is not authorized for run "
                f"{run.protocol_hash}:{spec.candidate_id}:{spec.protocol_hash}"
            )
        if initial_state is not CandidateState.DRAFT:
            raise RepositoryError(
                "new candidates must enter the repository in draft state"
            )
        now = _utc_now()
        spec_json = canonical_json_bytes(spec.to_dict()).decode("utf-8")
        try:
            with self._transaction() as connection:
                connection.execute(
                    """INSERT INTO candidates(
                        run_id, candidate_id, spec_hash, spec_json, generation, state, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        spec.candidate_id,
                        spec.content_hash,
                        spec_json,
                        spec.generation,
                        initial_state.value,
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """INSERT INTO state_transitions(
                        run_id, entity_type, entity_id, from_state, to_state, reason, occurred_at
                    ) VALUES (?, 'candidate', ?, NULL, ?, 'candidate_registered', ?)""",
                    (run_id, spec.candidate_id, initial_state.value, now),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateCandidateError(
                f"candidate id or content already exists in run: {spec.candidate_id}"
            ) from exc
        return self.get_candidate(run_id, spec.candidate_id)

    def _candidate_from_row(self, row: sqlite3.Row) -> CandidateRecord:
        spec = CandidateSpec.from_dict(json.loads(row["spec_json"]))
        return CandidateRecord(
            run_id=row["run_id"],
            candidate_id=row["candidate_id"],
            spec_hash=row["spec_hash"],
            state=CandidateState(row["state"]),
            spec=spec,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_candidate(self, run_id: str, candidate_id: str) -> CandidateRecord:
        row = self.connection.execute(
            "SELECT * FROM candidates WHERE run_id=? AND candidate_id=?",
            (run_id, candidate_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown candidate in run {run_id}: {candidate_id}")
        return self._candidate_from_row(row)

    def list_candidates(
        self,
        run_id: str,
        *,
        state: CandidateState | None = None,
    ) -> list[CandidateRecord]:
        if state is None:
            rows = self.connection.execute(
                "SELECT * FROM candidates WHERE run_id=? ORDER BY generation, candidate_id",
                (run_id,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM candidates WHERE run_id=? AND state=? ORDER BY generation, candidate_id",
                (run_id, state.value),
            ).fetchall()
        return [self._candidate_from_row(row) for row in rows]

    def transition_candidate(
        self,
        run_id: str,
        candidate_id: str,
        target: CandidateState,
        *,
        reason: str,
    ) -> CandidateRecord:
        current = self.get_candidate(run_id, candidate_id)
        assert_candidate_transition(current.state, target)
        now = _utc_now()
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE candidates SET state=?, updated_at=?
                   WHERE run_id=? AND candidate_id=? AND state=?""",
                (target.value, now, run_id, candidate_id, current.state.value),
            )
            if cursor.rowcount != 1:
                raise RepositoryError("candidate state changed concurrently")
            connection.execute(
                """INSERT INTO state_transitions(
                    run_id, entity_type, entity_id, from_state, to_state, reason, occurred_at
                ) VALUES (?, 'candidate', ?, ?, ?, ?, ?)""",
                (run_id, candidate_id, current.state.value, target.value, reason, now),
            )
        return self.get_candidate(run_id, candidate_id)

    def add_artifact(self, run_id: str, record: ArtifactRecord) -> None:
        self.get_run(run_id)
        with self._transaction() as connection:
            connection.execute(
                """INSERT INTO artifacts(
                    run_id, logical_name, location, sha256, size_bytes, media_type, role, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    record.logical_name,
                    record.location,
                    record.sha256,
                    record.size_bytes,
                    record.media_type,
                    record.role,
                    _utc_now(),
                ),
            )

    def record_event(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> EventRecord:
        self.get_run(run_id)
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event_type must not be empty")
        if not isinstance(payload, dict):
            raise ValueError("event payload must be a JSON object")
        payload_json = canonical_json_bytes(payload).decode("utf-8")
        with self._transaction() as connection:
            cursor = connection.execute(
                """INSERT INTO events(
                    run_id, event_type, payload_json, payload_hash, occurred_at
                ) VALUES (?, ?, ?, ?, ?)""",
                (
                    run_id,
                    event_type.strip(),
                    payload_json,
                    hash_json(payload),
                    _utc_now(),
                ),
            )
            event_id = cursor.lastrowid
        if event_id is None:  # pragma: no cover - SQLite always assigns it
            raise RepositoryError("SQLite did not assign an event id")
        return self.get_event(run_id, int(event_id))

    def _event_from_row(self, row: sqlite3.Row) -> EventRecord:
        payload = json.loads(row["payload_json"])
        if not isinstance(payload, dict):
            raise RepositoryError(
                f"event {row['event_id']} payload root is not an object"
            )
        return EventRecord(
            event_id=row["event_id"],
            run_id=row["run_id"],
            event_type=row["event_type"],
            payload=payload,
            payload_hash=row["payload_hash"],
            occurred_at=row["occurred_at"],
        )

    def get_event(self, run_id: str, event_id: int) -> EventRecord:
        row = self.connection.execute(
            "SELECT * FROM events WHERE run_id=? AND event_id=?",
            (run_id, event_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown event in run {run_id}: {event_id}")
        return self._event_from_row(row)

    def list_events(
        self,
        run_id: str,
        *,
        event_type: str | None = None,
    ) -> list[EventRecord]:
        self.get_run(run_id)
        if event_type is None:
            rows = self.connection.execute(
                "SELECT * FROM events WHERE run_id=? ORDER BY event_id",
                (run_id,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM events WHERE run_id=? AND event_type=? ORDER BY event_id",
                (run_id, event_type),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def status(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        state_rows = self.connection.execute(
            """SELECT state, COUNT(*) AS count FROM candidates
               WHERE run_id=? GROUP BY state ORDER BY state""",
            (run_id,),
        ).fetchall()
        artifact_count = self.connection.execute(
            "SELECT COUNT(*) AS count FROM artifacts WHERE run_id=?", (run_id,)
        ).fetchone()["count"]
        event_count = self.connection.execute(
            "SELECT COUNT(*) AS count FROM events WHERE run_id=?", (run_id,)
        ).fetchone()["count"]
        return {
            "run_id": run.run_id,
            "protocol_hash": run.protocol_hash,
            "run_state": run.state.value,
            "candidate_count": sum(row["count"] for row in state_rows),
            "candidate_states": {row["state"]: row["count"] for row in state_rows},
            "artifact_count": artifact_count,
            "event_count": event_count,
            "created_at": run.created_at,
            "updated_at": run.updated_at,
        }

    def verify_integrity(self, run_id: str | None = None) -> tuple[str, ...]:
        errors: list[str] = []
        result = self.connection.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            errors.append(f"sqlite_integrity:{result}")
        authority_query = "SELECT * FROM candidate_protocol_authorities"
        authority_parameters: tuple[str, ...] = ()
        if run_id is not None:
            authority_query += " WHERE run_id=?"
            authority_parameters = (run_id,)
        authorities: dict[str, CandidateProtocolAuthority] = {}
        for row in self.connection.execute(
            authority_query, authority_parameters
        ).fetchall():
            label = str(row["run_id"])
            try:
                authority = CandidateProtocolAuthority.from_dict(
                    json.loads(row["authority_json"])
                )
            except Exception as exc:
                errors.append(f"invalid_candidate_protocol_authority:{label}:{exc}")
                continue
            run = self.connection.execute(
                "SELECT protocol_hash FROM runs WHERE run_id=?", (label,)
            ).fetchone()
            if (
                run is None
                or authority.run_id != label
                or authority.control_protocol_content_sha256 != run["protocol_hash"]
                or row["control_protocol_hash"]
                != authority.control_protocol_content_sha256
                or row["candidate_protocol_hash"]
                != authority.candidate_protocol_content_sha256
                or row["authority_hash"] != authority.content_hash
            ):
                errors.append(f"candidate_protocol_authority_mismatch:{label}")
                continue
            authorities[label] = authority
        query = "SELECT * FROM candidates"
        parameters: tuple[str, ...] = ()
        if run_id is not None:
            query += " WHERE run_id=?"
            parameters = (run_id,)
        for row in self.connection.execute(query, parameters).fetchall():
            label = f"{row['run_id']}:{row['candidate_id']}"
            try:
                spec = CandidateSpec.from_dict(json.loads(row["spec_json"]))
            except Exception as exc:
                errors.append(f"invalid_candidate_spec:{label}:{exc}")
                continue
            if spec.content_hash != row["spec_hash"]:
                errors.append(
                    f"candidate_hash_mismatch:{label}:{row['spec_hash']}!={spec.content_hash}"
                )
            run = self.connection.execute(
                "SELECT protocol_hash FROM runs WHERE run_id=?", (row["run_id"],)
            ).fetchone()
            authority = authorities.get(str(row["run_id"]))
            if (
                authority is not None
                and spec.candidate_id not in authority.candidate_ids
            ):
                errors.append(f"candidate_identity_not_authorized:{label}")
            if run is None or spec.protocol_hash not in {
                run["protocol_hash"] if run is not None else "",
                (
                    authority.candidate_protocol_content_sha256
                    if authority is not None
                    else ""
                ),
            }:
                errors.append(f"candidate_protocol_mismatch:{label}")
        event_query = "SELECT * FROM events"
        if run_id is not None:
            event_query += " WHERE run_id=?"
        for row in self.connection.execute(event_query, parameters).fetchall():
            label = f"{row['run_id']}:{row['event_id']}"
            try:
                payload = json.loads(row["payload_json"])
            except Exception as exc:
                errors.append(f"invalid_event_payload:{label}:{exc}")
                continue
            if not isinstance(payload, dict):
                errors.append(f"invalid_event_payload_root:{label}")
                continue
            actual_hash = hash_json(payload)
            if actual_hash != row["payload_hash"]:
                errors.append(
                    f"event_hash_mismatch:{label}:{row['payload_hash']}!={actual_hash}"
                )
        return tuple(errors)
