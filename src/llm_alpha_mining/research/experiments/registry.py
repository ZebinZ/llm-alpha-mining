from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType, TracebackType
from typing import Iterator, Mapping

from llm_alpha_mining.research.core.hashing import (
    canonical_json_bytes,
    hash_json,
    require_sha256,
)
from llm_alpha_mining.research.experiments.approval import (
    ApprovalGrant,
    ApprovalRevocation,
)
from llm_alpha_mining.research.experiments.receipt import ExperimentReceipt
from llm_alpha_mining.research.experiments.spec import (
    DataPartition,
    ExperimentProfile,
    ExperimentSpec,
)


EXPERIMENT_REGISTRY_SCHEMA = "experiment-registry/v1"
PROTECTED_EVALUATION_ACTION = "consume_protected_data"
PROTECTED_EVALUATION_SCOPE = "locked_test_once"
PROTECTED_EVALUATION_ALLOWED_ROLES = frozenset({"research_lead", "risk_officer"})
_PROTECTED_CONSUMPTION_SCHEMA = "protected-evaluation-consumption/v1"
_PROTECTED_PARTITION_SET_SCHEMA = "protected-partition-set/v1"
_PROTECTED_PARTITION_NAMES = frozenset({"test", "holdout"})

_COMPONENT_TYPES = frozenset(
    {
        "data",
        "factor",
        "label",
        "validation",
        "evaluation",
        "model",
        "portfolio",
        "cost",
        "robustness",
        "code",
        "environment",
        "llm_policy",
        "llm_transport",
        "scientific_lineage",
    }
)
_STATUS_TRANSITIONS = {
    "registered": frozenset({"running", "failed", "cancelled"}),
    "running": frozenset({"failed", "cancelled"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}


class ExperimentRegistryError(RuntimeError):
    pass


class ExperimentRegistryConflict(ExperimentRegistryError):
    pass


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    experiment_spec_hash: str
    experiment_id: str
    version: str
    profile: str
    status: str
    registered_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    artifact_hash: str
    experiment_spec_hash: str
    logical_name: str
    kind: str
    location: str
    media_type: str
    size_bytes: int
    created_at: str


@dataclass(frozen=True, slots=True)
class RegisteredExperimentComponent:
    """One immutable component bound to a registered experiment role."""

    role: str
    component_hash: str
    component_type: str
    descriptor: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class LineageEdge:
    child_hash: str
    parent_hash: str
    relationship: str
    experiment_spec_hash: str
    recorded_at: str


@dataclass(frozen=True, slots=True)
class StageArtifactPublication:
    experiment_spec_hash: str
    attempt_id: int
    stage: str
    artifact_hash: str
    status: str
    parent_hashes: tuple[str, ...]
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ProtectedEvaluationConsumptionRecord:
    """Durable state for one fail-closed locked-test access."""

    consumption_id: str
    experiment_spec_hash: str
    approval_hash: str
    protected_partition_set_hash: str
    request_hash: str
    evaluator_hash: str
    status: str
    reserved_at: str
    completed_at: str | None
    result_hash: str | None
    result_payload: Mapping[str, object] | None


class ExperimentRegistry:
    """Append-only experiment catalog and lineage graph.

    This is deliberately a new SQLite database. Frozen V5 ledgers stay byte-for-byte
    immutable and can be linked as components or artifacts without being migrated.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=5.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA busy_timeout = 5000")
        self._initialize_schema()

    def __enter__(self) -> "ExperimentRegistry":
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
            raise ExperimentRegistryError("nested registry transactions are forbidden")
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
            CREATE TABLE IF NOT EXISTS registry_metadata(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS components(
                component_hash TEXT PRIMARY KEY,
                component_type TEXT NOT NULL,
                descriptor_json TEXT NOT NULL,
                registered_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS experiments(
                experiment_spec_hash TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                version TEXT NOT NULL,
                profile TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN (
                    'registered','running','completed','failed','cancelled'
                )),
                spec_json TEXT NOT NULL,
                registered_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(experiment_id, version)
            );
            CREATE TABLE IF NOT EXISTS experiment_components(
                experiment_spec_hash TEXT NOT NULL
                    REFERENCES experiments(experiment_spec_hash),
                role TEXT NOT NULL,
                component_hash TEXT NOT NULL REFERENCES components(component_hash),
                PRIMARY KEY(experiment_spec_hash, role)
            );
            CREATE TABLE IF NOT EXISTS artifacts(
                artifact_hash TEXT PRIMARY KEY,
                experiment_spec_hash TEXT NOT NULL
                    REFERENCES experiments(experiment_spec_hash),
                logical_name TEXT NOT NULL,
                kind TEXT NOT NULL,
                location TEXT NOT NULL,
                media_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
                descriptor_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(experiment_spec_hash, logical_name)
            );
            CREATE TABLE IF NOT EXISTS lineage(
                child_hash TEXT NOT NULL,
                parent_hash TEXT NOT NULL,
                relationship TEXT NOT NULL,
                experiment_spec_hash TEXT NOT NULL
                    REFERENCES experiments(experiment_spec_hash),
                recorded_at TEXT NOT NULL,
                PRIMARY KEY(child_hash, parent_hash, relationship)
            );
            CREATE TABLE IF NOT EXISTS stage_artifact_publications(
                experiment_spec_hash TEXT NOT NULL
                    REFERENCES experiments(experiment_spec_hash),
                attempt_id INTEGER NOT NULL,
                stage TEXT NOT NULL,
                artifact_hash TEXT NOT NULL,
                descriptor_json TEXT NOT NULL,
                parent_hashes_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN (
                    'provisional','committed','abandoned'
                )),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(experiment_spec_hash,attempt_id)
            );
            CREATE TABLE IF NOT EXISTS experiment_receipts(
                receipt_hash TEXT PRIMARY KEY,
                experiment_spec_hash TEXT NOT NULL UNIQUE
                    REFERENCES experiments(experiment_spec_hash),
                payload_json TEXT NOT NULL,
                sealed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS human_approvals(
                approval_hash TEXT PRIMARY KEY,
                experiment_spec_hash TEXT NOT NULL
                    REFERENCES experiments(experiment_spec_hash),
                action TEXT NOT NULL,
                scope TEXT NOT NULL,
                artifact_hash TEXT NOT NULL,
                actor TEXT NOT NULL,
                actor_role TEXT NOT NULL,
                signature_hash TEXT NOT NULL,
                nonce TEXT NOT NULL UNIQUE,
                issued_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS approval_revocations(
                approval_hash TEXT PRIMARY KEY
                    REFERENCES human_approvals(approval_hash),
                revoked_at TEXT NOT NULL,
                revoked_by TEXT NOT NULL,
                reason_code TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS protected_evaluation_consumptions(
                consumption_id TEXT PRIMARY KEY,
                experiment_spec_hash TEXT NOT NULL
                    REFERENCES experiments(experiment_spec_hash),
                approval_hash TEXT NOT NULL UNIQUE
                    REFERENCES human_approvals(approval_hash),
                protected_partition_set_hash TEXT NOT NULL UNIQUE,
                protected_partitions_json TEXT NOT NULL,
                request_hash TEXT NOT NULL UNIQUE,
                request_payload_json TEXT NOT NULL,
                evaluator_hash TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('reserved','completed')),
                reserved_at TEXT NOT NULL,
                completed_at TEXT,
                result_hash TEXT,
                result_payload_json TEXT,
                CHECK(
                    (status='reserved' AND completed_at IS NULL
                        AND result_hash IS NULL AND result_payload_json IS NULL)
                    OR
                    (status='completed' AND completed_at IS NOT NULL
                        AND result_hash IS NOT NULL AND result_payload_json IS NOT NULL)
                )
            );
            CREATE TABLE IF NOT EXISTS experiment_events(
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_spec_hash TEXT NOT NULL
                    REFERENCES experiments(experiment_spec_hash),
                event_type TEXT NOT NULL,
                event_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                previous_event_hash TEXT,
                event_hash TEXT NOT NULL UNIQUE
            );
            CREATE INDEX IF NOT EXISTS idx_lineage_parent ON lineage(parent_hash);
            CREATE INDEX IF NOT EXISTS idx_stage_artifact_publication_status
                ON stage_artifact_publications(experiment_spec_hash,status);
            CREATE INDEX IF NOT EXISTS idx_events_experiment
                ON experiment_events(experiment_spec_hash, sequence);
            CREATE INDEX IF NOT EXISTS idx_approval_lookup
                ON human_approvals(experiment_spec_hash,action,scope,artifact_hash);
            CREATE INDEX IF NOT EXISTS idx_protected_consumption_experiment
                ON protected_evaluation_consumptions(experiment_spec_hash,status);
            """
        )
        existing = dict(
            self.connection.execute(
                "SELECT key,value FROM registry_metadata"
            ).fetchall()
        )
        expected = {"schema_version": EXPERIMENT_REGISTRY_SCHEMA}
        if not existing:
            self.connection.execute(
                "INSERT INTO registry_metadata(key,value) VALUES('schema_version',?)",
                (EXPERIMENT_REGISTRY_SCHEMA,),
            )
            self.connection.commit()
        elif existing != expected:
            raise ExperimentRegistryConflict("experiment registry schema differs")

    def register_component(
        self,
        component_hash: str,
        *,
        component_type: str,
        descriptor: Mapping[str, object],
        registered_at: datetime | None = None,
    ) -> str:
        digest, payload_json = _component_registration_values(
            component_hash=component_hash,
            component_type=component_type,
            descriptor=descriptor,
        )
        timestamp = _iso(registered_at or _utc_now())
        with self._transaction() as connection:
            return _register_component_locked(
                connection,
                digest=digest,
                component_type=component_type,
                payload_json=payload_json,
                timestamp=timestamp,
            )

    def register_experiment(
        self, spec: ExperimentSpec, *, registered_at: datetime | None = None
    ) -> ExperimentRecord:
        if not isinstance(spec, ExperimentSpec):
            raise TypeError("spec must be ExperimentSpec")
        timestamp = _iso(registered_at or _utc_now())
        with self._transaction() as connection:
            return _register_experiment_locked(
                connection,
                spec=spec,
                timestamp=timestamp,
            )

    def register_component_and_experiment(
        self,
        spec: ExperimentSpec,
        *,
        component_hash: str,
        component_type: str,
        descriptor: Mapping[str, object],
        registered_at: datetime | None = None,
    ) -> ExperimentRecord:
        """Atomically register one new component and its bound experiment."""

        if not isinstance(spec, ExperimentSpec):
            raise TypeError("spec must be ExperimentSpec")
        digest, payload_json = _component_registration_values(
            component_hash=component_hash,
            component_type=component_type,
            descriptor=descriptor,
        )
        if (component_type, digest) not in {
            (role.split(":", 1)[0], bound_hash)
            for role, bound_hash in spec.component_bindings()
        }:
            raise ValueError("atomic component is not bound by ExperimentSpec")
        timestamp = _iso(registered_at or _utc_now())
        with self._transaction() as connection:
            _register_component_locked(
                connection,
                digest=digest,
                component_type=component_type,
                payload_json=payload_json,
                timestamp=timestamp,
            )
            return _register_experiment_locked(
                connection,
                spec=spec,
                timestamp=timestamp,
            )

    def register_components_and_experiment(
        self,
        spec: ExperimentSpec,
        *,
        components: tuple[tuple[str, str, Mapping[str, object]], ...],
        registered_at: datetime | None = None,
    ) -> ExperimentRecord:
        """Atomically register the complete component set and one experiment.

        The batch must cover every ``ExperimentSpec`` binding exactly once and
        may not contain unrelated components.  Descriptor validation happens
        before the transaction; immutable conflicts discovered while writing
        roll the entire batch back.  This is the narrow bulk primitive needed
        by result-free factor terminalization without weakening the existing
        append-only component contract.
        """

        if type(spec) is not ExperimentSpec:
            raise TypeError("spec must be an exact ExperimentSpec")
        if type(components) is not tuple or not components:
            raise TypeError("components must be a non-empty immutable tuple")
        prepared: list[tuple[str, str, str]] = []
        observed_bindings: list[tuple[str, str]] = []
        for offset, item in enumerate(components):
            if type(item) is not tuple or len(item) != 3:
                raise TypeError(
                    f"component batch item must be a three-item tuple:{offset}"
                )
            component_hash, component_type, descriptor = item
            if type(component_hash) is not str or type(component_type) is not str:
                raise TypeError(f"component batch identity must be text:{offset}")
            if not isinstance(descriptor, Mapping):
                raise TypeError(f"component descriptor must be a mapping:{offset}")
            digest, payload_json = _component_registration_values(
                component_hash=component_hash,
                component_type=component_type,
                descriptor=descriptor,
            )
            observed_bindings.append((component_type, digest))
            prepared.append((digest, component_type, payload_json))
        if len(set(observed_bindings)) != len(observed_bindings):
            raise ValueError("component batch contains duplicate bindings")
        expected_bindings = tuple(
            (role.split(":", 1)[0], digest)
            for role, digest in spec.component_bindings()
        )
        if len(set(expected_bindings)) != len(expected_bindings) or set(
            observed_bindings
        ) != set(expected_bindings):
            raise ValueError(
                "component batch does not exactly cover ExperimentSpec bindings"
            )
        timestamp = _iso(registered_at or _utc_now())
        with self._transaction() as connection:
            for digest, component_type, payload_json in sorted(prepared):
                _register_component_locked(
                    connection,
                    digest=digest,
                    component_type=component_type,
                    payload_json=payload_json,
                    timestamp=timestamp,
                )
            return _register_experiment_locked(
                connection,
                spec=spec,
                timestamp=timestamp,
            )

    def register_component_experiment_and_protected_approval(
        self,
        spec: ExperimentSpec,
        *,
        component_hash: str,
        component_type: str,
        descriptor: Mapping[str, object],
        approval: ApprovalGrant,
        registered_at: datetime | None = None,
        approval_at: datetime | None = None,
    ) -> ExperimentRecord:
        """Atomically register a governed run and its one-time test approval.

        The specialized approval targets the immutable evaluation component.
        Its exact experiment hash therefore also binds the candidate factors,
        protected partitions, code, environment, and scientific lineage.
        """

        if not isinstance(spec, ExperimentSpec):
            raise TypeError("spec must be ExperimentSpec")
        if not isinstance(approval, ApprovalGrant):
            raise TypeError("approval must be ApprovalGrant")
        digest, payload_json = _component_registration_values(
            component_hash=component_hash,
            component_type=component_type,
            descriptor=descriptor,
        )
        if (component_type, digest) not in {
            (role.split(":", 1)[0], bound_hash)
            for role, bound_hash in spec.component_bindings()
        }:
            raise ValueError("atomic component is not bound by ExperimentSpec")
        timestamp_value = registered_at or _utc_now()
        approval_moment = approval_at or timestamp_value
        timestamp = _iso(timestamp_value)
        _validate_protected_evaluation_grant(
            approval,
            spec=spec,
            at=_iso(approval_moment),
        )
        with self._transaction() as connection:
            _register_component_locked(
                connection,
                digest=digest,
                component_type=component_type,
                payload_json=payload_json,
                timestamp=timestamp,
            )
            record = _register_experiment_locked(
                connection,
                spec=spec,
                timestamp=timestamp,
            )
            _register_approval_locked(connection, approval)
            return record

    def set_status(
        self,
        experiment_spec_hash: str,
        target_status: str,
        *,
        reason_code: str,
        at: datetime | None = None,
    ) -> ExperimentRecord:
        digest = require_sha256(experiment_spec_hash, name="experiment_spec_hash")
        if target_status == "completed":
            raise ValueError("completed status is only created by sealing a receipt")
        if not _safe_code(reason_code):
            raise ValueError("experiment status reason_code is invalid")
        timestamp = _iso(at or _utc_now())
        with self._transaction() as connection:
            record = self._record_locked(connection, digest)
            if target_status not in _STATUS_TRANSITIONS[record.status]:
                raise ExperimentRegistryConflict(
                    f"illegal experiment transition:{record.status}->{target_status}"
                )
            connection.execute(
                "UPDATE experiments SET status=?,updated_at=? WHERE experiment_spec_hash=?",
                (target_status, timestamp, digest),
            )
            self._append_event_locked(
                connection,
                digest,
                "experiment_status_changed",
                {
                    "from": record.status,
                    "to": target_status,
                    "reason_code": reason_code,
                },
                timestamp,
            )
            return self._record_locked(connection, digest)

    def register_artifact(
        self,
        experiment_spec_hash: str,
        *,
        artifact_hash: str,
        logical_name: str,
        kind: str,
        location: str,
        media_type: str,
        size_bytes: int,
        descriptor: Mapping[str, object],
        created_at: datetime | None = None,
    ) -> ArtifactRecord:
        experiment_hash = require_sha256(
            experiment_spec_hash, name="experiment_spec_hash"
        )
        digest = require_sha256(artifact_hash, name="artifact_hash")
        if not _safe_code(logical_name) or not _safe_code(kind):
            raise ValueError("artifact logical_name/kind is invalid")
        normalized_location = _safe_relative_location(location)
        if not media_type.strip():
            raise ValueError("artifact media_type is empty")
        if (
            not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
        ):
            raise ValueError("artifact size_bytes must be non-negative")
        payload = dict(descriptor)
        required = {
            "artifact_hash": digest,
            "logical_name": logical_name,
            "kind": kind,
            "location": normalized_location,
            "media_type": media_type,
            "size_bytes": size_bytes,
        }
        if any(payload.get(name) != value for name, value in required.items()):
            raise ValueError("artifact descriptor does not bind immutable fields")
        payload_json = _json_text(payload)
        timestamp = _iso(created_at or _utc_now())
        with self._transaction() as connection:
            self._record_locked(connection, experiment_hash)
            return _register_artifact_locked(
                connection,
                experiment_hash=experiment_hash,
                digest=digest,
                logical_name=logical_name,
                kind=kind,
                normalized_location=normalized_location,
                media_type=media_type,
                size_bytes=size_bytes,
                payload_json=payload_json,
                timestamp=timestamp,
            )

    def register_artifact_with_approval(
        self,
        experiment_spec_hash: str,
        *,
        artifact_hash: str,
        logical_name: str,
        kind: str,
        location: str,
        media_type: str,
        size_bytes: int,
        descriptor: Mapping[str, object],
        approval_hash: str,
        approval_action: str,
        approval_scope: str,
        approval_artifact_hash: str,
        allowed_roles: frozenset[str],
    ) -> tuple[ArtifactRecord, ApprovalGrant]:
        """Atomically revalidate one approval and publish one artifact.

        This closes the approval-revocation TOCTOU at the final publication
        boundary.  It does not coordinate other SQLite registries and grants
        no authority beyond the exact artifact registration performed here.
        """

        experiment_hash = require_sha256(
            experiment_spec_hash, name="experiment_spec_hash"
        )
        digest = require_sha256(artifact_hash, name="artifact_hash")
        if not _safe_code(logical_name) or not _safe_code(kind):
            raise ValueError("artifact logical_name/kind is invalid")
        normalized_location = _safe_relative_location(location)
        if not media_type.strip():
            raise ValueError("artifact media_type is empty")
        if (
            not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
        ):
            raise ValueError("artifact size_bytes must be non-negative")
        payload = dict(descriptor)
        required = {
            "artifact_hash": digest,
            "logical_name": logical_name,
            "kind": kind,
            "location": normalized_location,
            "media_type": media_type,
            "size_bytes": size_bytes,
        }
        if any(payload.get(name) != value for name, value in required.items()):
            raise ValueError("artifact descriptor does not bind immutable fields")
        if not allowed_roles:
            raise ValueError("approval allowed_roles must not be empty")
        payload_json = _json_text(payload)
        # Approval validity must be evaluated against registry-owned wall time;
        # callers cannot backdate the final publication boundary.
        timestamp = _iso(_utc_now())
        with self._transaction() as connection:
            self._record_locked(connection, experiment_hash)
            approval = self._require_approval_locked(
                connection,
                approval_hash=approval_hash,
                experiment_spec_hash=experiment_hash,
                action=approval_action,
                scope=approval_scope,
                artifact_hash=approval_artifact_hash,
                at=timestamp,
                allowed_roles=allowed_roles,
            )
            artifact = _register_artifact_locked(
                connection,
                experiment_hash=experiment_hash,
                digest=digest,
                logical_name=logical_name,
                kind=kind,
                normalized_location=normalized_location,
                media_type=media_type,
                size_bytes=size_bytes,
                payload_json=payload_json,
                timestamp=timestamp,
            )
            return artifact, approval

    def prepare_stage_artifact_publication(
        self,
        experiment_spec_hash: str,
        *,
        attempt_id: int,
        stage: str,
        artifact_hash: str,
        logical_name: str,
        kind: str,
        location: str,
        media_type: str,
        size_bytes: int,
        descriptor: Mapping[str, object],
        parent_hashes: tuple[str, ...],
        created_at: datetime | None = None,
    ) -> StageArtifactPublication:
        """Persist a non-authoritative outbox row for one running attempt.

        Provisional rows are deliberately excluded from ``artifacts`` and all
        receipts.  Only a later successful Runtime attempt may promote the row.
        """

        experiment_hash = require_sha256(
            experiment_spec_hash, name="experiment_spec_hash"
        )
        if not isinstance(attempt_id, int) or isinstance(attempt_id, bool):
            raise ValueError("stage publication attempt_id must be an integer")
        if attempt_id <= 0:
            raise ValueError("stage publication attempt_id must be positive")
        if not _safe_code(stage):
            raise ValueError("stage publication stage is invalid")
        digest = require_sha256(artifact_hash, name="artifact_hash")
        if not _safe_code(logical_name) or not _safe_code(kind):
            raise ValueError("artifact logical_name/kind is invalid")
        normalized_location = _safe_relative_location(location)
        if not media_type.strip():
            raise ValueError("artifact media_type is empty")
        if (
            not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
        ):
            raise ValueError("artifact size_bytes must be non-negative")
        payload = dict(descriptor)
        required: dict[str, object] = {
            "artifact_hash": digest,
            "logical_name": logical_name,
            "kind": kind,
            "location": normalized_location,
            "media_type": media_type,
            "size_bytes": size_bytes,
            "stage": stage,
            "attempt_id": attempt_id,
        }
        if any(payload.get(name) != value for name, value in required.items()):
            raise ValueError("stage publication descriptor binding differs")
        payload_json = _json_text(payload)
        parents = tuple(parent_hashes)
        if len(parents) != len(set(parents)):
            raise ValueError("stage publication parents must be unique")
        for offset, parent in enumerate(parents):
            require_sha256(parent, name=f"stage publication parent:{offset}")
            if parent == digest:
                raise ValueError("stage publication self lineage is forbidden")
        parents_json = canonical_json_bytes(list(parents)).decode("utf-8")
        timestamp = _iso(created_at or _utc_now())
        with self._transaction() as connection:
            self._record_locked(connection, experiment_hash)
            for parent in parents:
                if not self._known_hash_locked(connection, parent):
                    raise ExperimentRegistryConflict(
                        f"stage publication parent is unknown:{parent}"
                    )
            existing = connection.execute(
                """SELECT * FROM stage_artifact_publications
                   WHERE experiment_spec_hash=? AND attempt_id=?""",
                (experiment_hash, attempt_id),
            ).fetchone()
            if existing is not None:
                expected = {
                    "stage": stage,
                    "artifact_hash": digest,
                    "descriptor_json": payload_json,
                    "parent_hashes_json": parents_json,
                }
                if any(existing[name] != value for name, value in expected.items()):
                    raise ExperimentRegistryConflict(
                        "immutable stage artifact publication differs"
                    )
                if existing["status"] == "abandoned":
                    raise ExperimentRegistryConflict(
                        "abandoned stage artifact publication cannot be reused"
                    )
                return _stage_artifact_publication(existing)
            connection.execute(
                """INSERT INTO stage_artifact_publications(
                    experiment_spec_hash,attempt_id,stage,artifact_hash,
                    descriptor_json,parent_hashes_json,status,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,'provisional',?,?)""",
                (
                    experiment_hash,
                    attempt_id,
                    stage,
                    digest,
                    payload_json,
                    parents_json,
                    timestamp,
                    timestamp,
                ),
            )
            self._append_event_locked(
                connection,
                experiment_hash,
                "stage_artifact_publication_prepared",
                {
                    "attempt_id": attempt_id,
                    "stage": stage,
                    "artifact_hash": digest,
                },
                timestamp,
            )
            row = connection.execute(
                """SELECT * FROM stage_artifact_publications
                   WHERE experiment_spec_hash=? AND attempt_id=?""",
                (experiment_hash, attempt_id),
            ).fetchone()
            if row is None:  # pragma: no cover
                raise RuntimeError("stage artifact publication disappeared")
            return _stage_artifact_publication(row)

    def commit_stage_artifact_publication(
        self,
        experiment_spec_hash: str,
        *,
        attempt_id: int,
        stage: str,
        artifact_hash: str,
        committed_at: datetime | None = None,
    ) -> ArtifactRecord:
        """Atomically promote one successful attempt's outbox and lineage."""

        experiment_hash = require_sha256(
            experiment_spec_hash, name="experiment_spec_hash"
        )
        digest = require_sha256(artifact_hash, name="artifact_hash")
        if not isinstance(attempt_id, int) or isinstance(attempt_id, bool):
            raise ValueError("stage publication attempt_id must be an integer")
        if attempt_id <= 0 or not _safe_code(stage):
            raise ValueError("stage publication attempt/stage is invalid")
        timestamp = _iso(committed_at or _utc_now())
        with self._transaction() as connection:
            self._record_locked(connection, experiment_hash)
            publication = connection.execute(
                """SELECT * FROM stage_artifact_publications
                   WHERE experiment_spec_hash=? AND attempt_id=?""",
                (experiment_hash, attempt_id),
            ).fetchone()
            if publication is None:
                raise ExperimentRegistryConflict(
                    "stage artifact publication is not prepared"
                )
            if publication["stage"] != stage or publication["artifact_hash"] != digest:
                raise ExperimentRegistryConflict(
                    "stage artifact publication authority differs"
                )
            if publication["status"] == "abandoned":
                raise ExperimentRegistryConflict(
                    "abandoned stage artifact publication cannot commit"
                )
            descriptor_raw = json.loads(publication["descriptor_json"])
            parents_raw = json.loads(publication["parent_hashes_json"])
            if not isinstance(descriptor_raw, Mapping) or not isinstance(
                parents_raw, list
            ):
                raise ExperimentRegistryConflict(
                    "stage artifact publication payload is invalid"
                )
            descriptor = dict(descriptor_raw)
            if _json_text(descriptor) != publication["descriptor_json"] or not all(
                isinstance(parent, str) for parent in parents_raw
            ):
                raise ExperimentRegistryConflict(
                    "stage artifact publication payload is noncanonical"
                )
            parents = tuple(str(parent) for parent in parents_raw)
            logical_name = str(descriptor.get("logical_name", ""))
            kind = str(descriptor.get("kind", ""))
            location = str(descriptor.get("location", ""))
            media_type = str(descriptor.get("media_type", ""))
            size_bytes = descriptor.get("size_bytes")
            expected: dict[str, object] = {
                "artifact_hash": digest,
                "logical_name": logical_name,
                "kind": kind,
                "location": location,
                "media_type": media_type,
                "size_bytes": size_bytes,
                "stage": stage,
                "attempt_id": attempt_id,
            }
            if any(descriptor.get(name) != value for name, value in expected.items()):
                raise ExperimentRegistryConflict(
                    "stage artifact publication descriptor differs"
                )
            if (
                not _safe_code(logical_name)
                or not _safe_code(kind)
                or _safe_relative_location(location) != location
                or not media_type.strip()
                or not isinstance(size_bytes, int)
                or isinstance(size_bytes, bool)
                or size_bytes < 0
            ):
                raise ExperimentRegistryConflict(
                    "stage artifact publication descriptor is invalid"
                )
            artifact = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_hash=?",
                (digest,),
            ).fetchone()
            if artifact is None:
                try:
                    connection.execute(
                        """INSERT INTO artifacts(
                            artifact_hash,experiment_spec_hash,logical_name,kind,
                            location,media_type,size_bytes,descriptor_json,created_at
                        ) VALUES(?,?,?,?,?,?,?,?,?)""",
                        (
                            digest,
                            experiment_hash,
                            logical_name,
                            kind,
                            location,
                            media_type,
                            size_bytes,
                            publication["descriptor_json"],
                            publication["created_at"],
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ExperimentRegistryConflict(
                        "artifact logical name is already registered"
                    ) from exc
            elif (
                artifact["experiment_spec_hash"] != experiment_hash
                or artifact["descriptor_json"] != publication["descriptor_json"]
            ):
                raise ExperimentRegistryConflict(
                    "committed stage artifact descriptor differs"
                )
            for parent in parents:
                parent_digest = require_sha256(parent, name="stage publication parent")
                if not self._known_hash_locked(connection, parent_digest):
                    raise ExperimentRegistryConflict(
                        f"stage publication parent is unknown:{parent_digest}"
                    )
                cycle = connection.execute(
                    """WITH RECURSIVE ancestors(node) AS (
                        SELECT parent_hash FROM lineage WHERE child_hash=?
                        UNION
                        SELECT lineage.parent_hash FROM lineage
                        JOIN ancestors ON lineage.child_hash=ancestors.node
                    ) SELECT 1 FROM ancestors WHERE node=? LIMIT 1""",
                    (parent_digest, digest),
                ).fetchone()
                if cycle is not None:
                    raise ExperimentRegistryConflict(
                        "stage publication lineage would create a cycle"
                    )
                existing_edge = connection.execute(
                    """SELECT experiment_spec_hash FROM lineage
                       WHERE child_hash=? AND parent_hash=?
                         AND relationship='stage_input'""",
                    (digest, parent_digest),
                ).fetchone()
                if existing_edge is None:
                    connection.execute(
                        """INSERT INTO lineage(
                            child_hash,parent_hash,relationship,
                            experiment_spec_hash,recorded_at
                        ) VALUES(?,?,'stage_input',?,?)""",
                        (digest, parent_digest, experiment_hash, timestamp),
                    )
                elif existing_edge["experiment_spec_hash"] != experiment_hash:
                    raise ExperimentRegistryConflict(
                        "stage publication lineage belongs to another experiment"
                    )
            if publication["status"] != "committed":
                connection.execute(
                    """UPDATE stage_artifact_publications
                       SET status='committed',updated_at=?
                       WHERE experiment_spec_hash=? AND attempt_id=?
                         AND status='provisional'""",
                    (timestamp, experiment_hash, attempt_id),
                )
                self._append_event_locked(
                    connection,
                    experiment_hash,
                    "stage_artifact_publication_committed",
                    {
                        "attempt_id": attempt_id,
                        "stage": stage,
                        "artifact_hash": digest,
                    },
                    timestamp,
                )
            row = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_hash=?",
                (digest,),
            ).fetchone()
            if row is None:  # pragma: no cover
                raise RuntimeError("committed stage artifact disappeared")
            return _artifact_record(row)

    def abandon_stage_artifact_publication(
        self,
        experiment_spec_hash: str,
        *,
        attempt_id: int,
        abandoned_at: datetime | None = None,
    ) -> bool:
        """Make a failed attempt's provisional outbox permanently non-authoritative."""

        experiment_hash = require_sha256(
            experiment_spec_hash, name="experiment_spec_hash"
        )
        if not isinstance(attempt_id, int) or isinstance(attempt_id, bool):
            raise ValueError("stage publication attempt_id must be an integer")
        if attempt_id <= 0:
            raise ValueError("stage publication attempt_id must be positive")
        timestamp = _iso(abandoned_at or _utc_now())
        with self._transaction() as connection:
            self._record_locked(connection, experiment_hash)
            row = connection.execute(
                """SELECT * FROM stage_artifact_publications
                   WHERE experiment_spec_hash=? AND attempt_id=?""",
                (experiment_hash, attempt_id),
            ).fetchone()
            if row is None:
                return False
            if row["status"] == "committed":
                raise ExperimentRegistryConflict(
                    "committed stage artifact publication cannot be abandoned"
                )
            if row["status"] == "abandoned":
                return False
            connection.execute(
                """UPDATE stage_artifact_publications
                   SET status='abandoned',updated_at=?
                   WHERE experiment_spec_hash=? AND attempt_id=?
                     AND status='provisional'""",
                (timestamp, experiment_hash, attempt_id),
            )
            self._append_event_locked(
                connection,
                experiment_hash,
                "stage_artifact_publication_abandoned",
                {
                    "attempt_id": attempt_id,
                    "stage": str(row["stage"]),
                    "artifact_hash": str(row["artifact_hash"]),
                },
                timestamp,
            )
            return True

    def load_stage_artifact_publication(
        self,
        experiment_spec_hash: str,
        *,
        attempt_id: int,
    ) -> StageArtifactPublication:
        experiment_hash = require_sha256(
            experiment_spec_hash, name="experiment_spec_hash"
        )
        row = self.connection.execute(
            """SELECT * FROM stage_artifact_publications
               WHERE experiment_spec_hash=? AND attempt_id=?""",
            (experiment_hash, attempt_id),
        ).fetchone()
        if row is None:
            raise KeyError(
                f"stage artifact publication is unknown:{experiment_hash}:{attempt_id}"
            )
        return _stage_artifact_publication(row)

    def add_lineage(
        self,
        experiment_spec_hash: str,
        *,
        child_hash: str,
        parent_hash: str,
        relationship: str,
        recorded_at: datetime | None = None,
    ) -> LineageEdge:
        experiment_hash = require_sha256(
            experiment_spec_hash, name="experiment_spec_hash"
        )
        child = require_sha256(child_hash, name="lineage child_hash")
        parent = require_sha256(parent_hash, name="lineage parent_hash")
        if child == parent:
            raise ValueError("lineage self-cycle is forbidden")
        if not _safe_code(relationship):
            raise ValueError("lineage relationship is invalid")
        timestamp = _iso(recorded_at or _utc_now())
        with self._transaction() as connection:
            self._record_locked(connection, experiment_hash)
            for digest in (child, parent):
                if not self._known_hash_locked(connection, digest):
                    raise ExperimentRegistryConflict(
                        f"lineage node is unknown:{digest}"
                    )
            cycle = connection.execute(
                """WITH RECURSIVE ancestors(node) AS (
                    SELECT parent_hash FROM lineage WHERE child_hash=?
                    UNION
                    SELECT lineage.parent_hash FROM lineage
                    JOIN ancestors ON lineage.child_hash=ancestors.node
                ) SELECT 1 FROM ancestors WHERE node=? LIMIT 1""",
                (parent, child),
            ).fetchone()
            if cycle is not None:
                raise ExperimentRegistryConflict("lineage edge would create a cycle")
            existing = connection.execute(
                """SELECT * FROM lineage
                   WHERE child_hash=? AND parent_hash=? AND relationship=?""",
                (child, parent, relationship),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """INSERT INTO lineage(
                        child_hash,parent_hash,relationship,experiment_spec_hash,recorded_at
                    ) VALUES(?,?,?,?,?)""",
                    (child, parent, relationship, experiment_hash, timestamp),
                )
                self._append_event_locked(
                    connection,
                    experiment_hash,
                    "lineage_recorded",
                    {
                        "child_hash": child,
                        "parent_hash": parent,
                        "relationship": relationship,
                    },
                    timestamp,
                )
            elif existing["experiment_spec_hash"] != experiment_hash:
                raise ExperimentRegistryConflict(
                    "lineage edge belongs to another experiment"
                )
            row = connection.execute(
                """SELECT * FROM lineage
                   WHERE child_hash=? AND parent_hash=? AND relationship=?""",
                (child, parent, relationship),
            ).fetchone()
            if row is None:  # pragma: no cover
                raise RuntimeError("lineage insert disappeared")
            return LineageEdge(**dict(row))

    def ancestors(self, child_hash: str) -> tuple[LineageEdge, ...]:
        child = require_sha256(child_hash, name="lineage child_hash")
        rows = self.connection.execute(
            """WITH RECURSIVE path(child_hash,parent_hash,relationship,
                    experiment_spec_hash,recorded_at) AS (
                SELECT child_hash,parent_hash,relationship,experiment_spec_hash,recorded_at
                FROM lineage WHERE child_hash=?
                UNION
                SELECT lineage.child_hash,lineage.parent_hash,lineage.relationship,
                       lineage.experiment_spec_hash,lineage.recorded_at
                FROM lineage JOIN path ON lineage.child_hash=path.parent_hash
            ) SELECT DISTINCT * FROM path ORDER BY child_hash,parent_hash,relationship""",
            (child,),
        ).fetchall()
        return tuple(LineageEdge(**dict(row)) for row in rows)

    def seal_receipt(
        self, receipt: ExperimentReceipt, *, sealed_at: datetime | None = None
    ) -> str:
        digest: str = receipt.content_hash
        timestamp = _iso(sealed_at or _utc_now())
        payload_json = _json_text(receipt.to_dict())
        with self._transaction() as connection:
            experiment = self._record_locked(connection, receipt.experiment_spec_hash)
            if experiment.status not in {"registered", "running"}:
                raise ExperimentRegistryConflict(
                    "receipt cannot seal an already terminal experiment"
                )
            spec_payload = json.loads(
                connection.execute(
                    "SELECT spec_json FROM experiments WHERE experiment_spec_hash=?",
                    (receipt.experiment_spec_hash,),
                ).fetchone()[0]
            )
            expected_components = dict(
                connection.execute(
                    """SELECT role,component_hash FROM experiment_components
                       WHERE experiment_spec_hash=? ORDER BY role""",
                    (receipt.experiment_spec_hash,),
                ).fetchall()
            )
            if dict(receipt.component_bindings) != expected_components:
                raise ExperimentRegistryConflict("receipt component bindings differ")
            if receipt.random_seed != int(spec_payload["random_seed"]):
                raise ExperimentRegistryConflict("receipt random seed differs")
            if receipt.code_snapshot_hash != spec_payload["code_snapshot_hash"]:
                raise ExperimentRegistryConflict("receipt code snapshot differs")
            if receipt.environment_hash != spec_payload["environment_hash"]:
                raise ExperimentRegistryConflict("receipt environment differs")
            known_artifacts = {
                row[0]
                for row in connection.execute(
                    "SELECT artifact_hash FROM artifacts WHERE experiment_spec_hash=?",
                    (receipt.experiment_spec_hash,),
                ).fetchall()
            }
            if not set(receipt.artifact_hashes).issubset(known_artifacts):
                raise ExperimentRegistryConflict("receipt references unknown artifacts")
            if not set(receipt.stage_result_hashes.values()).issubset(known_artifacts):
                raise ExperimentRegistryConflict(
                    "receipt stage result is not an artifact"
                )
            if receipt.production_ready:
                if spec_payload["profile"] != "production_candidate":
                    raise ExperimentRegistryConflict(
                        "research experiment cannot seal a production-ready receipt"
                    )
                if receipt.metrics_artifact_hash is None:
                    raise ExperimentRegistryConflict(
                        "production-ready receipt requires a metrics artifact"
                    )
                self._require_approval_locked(
                    connection,
                    approval_hash=receipt.approval_hash,
                    experiment_spec_hash=receipt.experiment_spec_hash,
                    action="publish",
                    scope="production_release",
                    artifact_hash=receipt.metrics_artifact_hash,
                    at=timestamp,
                    allowed_roles=frozenset(
                        {"portfolio_manager", "research_lead", "risk_officer"}
                    ),
                )
            try:
                connection.execute(
                    """INSERT INTO experiment_receipts(
                        receipt_hash,experiment_spec_hash,payload_json,sealed_at
                    ) VALUES(?,?,?,?)""",
                    (digest, receipt.experiment_spec_hash, payload_json, timestamp),
                )
            except sqlite3.IntegrityError as exc:
                raise ExperimentRegistryConflict(
                    "experiment receipt is already sealed"
                ) from exc
            connection.execute(
                "UPDATE experiments SET status=?,updated_at=? WHERE experiment_spec_hash=?",
                (receipt.terminal_status, timestamp, receipt.experiment_spec_hash),
            )
            self._append_event_locked(
                connection,
                receipt.experiment_spec_hash,
                "experiment_receipt_sealed",
                {"receipt_hash": digest, "terminal_status": receipt.terminal_status},
                timestamp,
            )
        return digest

    def register_approval(self, grant: ApprovalGrant) -> str:
        if not isinstance(grant, ApprovalGrant):
            raise TypeError("grant must be ApprovalGrant")
        with self._transaction() as connection:
            self._record_locked(connection, grant.experiment_spec_hash)
            artifact = connection.execute(
                """SELECT 1 FROM artifacts
                   WHERE artifact_hash=? AND experiment_spec_hash=?""",
                (grant.artifact_hash, grant.experiment_spec_hash),
            ).fetchone()
            if artifact is None:
                raise ExperimentRegistryConflict(
                    "approval artifact is not registered to this experiment"
                )
            return _register_approval_locked(connection, grant)

    def revoke_approval(self, revocation: ApprovalRevocation) -> None:
        with self._transaction() as connection:
            row = connection.execute(
                """SELECT experiment_spec_hash FROM human_approvals
                   WHERE approval_hash=?""",
                (revocation.approval_hash,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown approval:{revocation.approval_hash}")
            try:
                connection.execute(
                    """INSERT INTO approval_revocations(
                        approval_hash,revoked_at,revoked_by,reason_code
                    ) VALUES(?,?,?,?)""",
                    (
                        revocation.approval_hash,
                        revocation.revoked_at,
                        revocation.revoked_by,
                        revocation.reason_code,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ExperimentRegistryConflict("approval is already revoked") from exc
            self._append_event_locked(
                connection,
                row["experiment_spec_hash"],
                "human_approval_revoked",
                {
                    "approval_hash": revocation.approval_hash,
                    "revoked_by": revocation.revoked_by,
                    "reason_code": revocation.reason_code,
                },
                revocation.revoked_at,
            )

    def require_approval(
        self,
        approval_hash: str,
        *,
        experiment_spec_hash: str,
        action: str,
        scope: str,
        artifact_hash: str,
        at: datetime,
        allowed_roles: frozenset[str],
    ) -> ApprovalGrant:
        timestamp = _iso(at)
        with self._transaction() as connection:
            return self._require_approval_locked(
                connection,
                approval_hash=approval_hash,
                experiment_spec_hash=experiment_spec_hash,
                action=action,
                scope=scope,
                artifact_hash=artifact_hash,
                at=timestamp,
                allowed_roles=allowed_roles,
            )

    def require_approval_for_audit(
        self,
        approval_hash: str,
        *,
        experiment_spec_hash: str,
        action: str,
        scope: str,
        artifact_hash: str,
        at: datetime,
        allowed_roles: frozenset[str],
    ) -> ApprovalGrant:
        """Validate one approval without a write transaction or authority grant.

        This selector exists for diagnostics only.  It neither reserves work nor
        authorizes execution/publication; authority-bearing callers must use the
        transactional operation that consumes the approval at their write edge.
        """

        if self.connection.in_transaction:
            raise ExperimentRegistryError(
                "approval audit requires a durable autocommit state"
            )
        return self._require_approval_locked(
            self.connection,
            approval_hash=approval_hash,
            experiment_spec_hash=experiment_spec_hash,
            action=action,
            scope=scope,
            artifact_hash=artifact_hash,
            at=_iso(at),
            allowed_roles=allowed_roles,
        )

    def reserve_protected_evaluation(
        self,
        *,
        experiment_spec_hash: str,
        approval_hash: str,
        evaluation_spec_hash: str,
        protected_partition_hashes: Mapping[str, str],
        request_hash: str,
        request_payload: Mapping[str, object],
        evaluator_hash: str,
        at: datetime,
    ) -> ProtectedEvaluationConsumptionRecord:
        """Reserve the only permitted read of one locked partition set.

        A persisted ``reserved`` row is intentionally irreversible through this
        API.  If the worker crashes after reservation, a retry cannot prove that
        protected data was unseen and must therefore stop for human review.
        """

        experiment_digest = require_sha256(
            experiment_spec_hash, name="experiment_spec_hash"
        )
        approval_digest = require_sha256(approval_hash, name="approval_hash")
        evaluation_digest = require_sha256(
            evaluation_spec_hash, name="evaluation_spec_hash"
        )
        request_digest = require_sha256(request_hash, name="request_hash")
        evaluator_digest = require_sha256(evaluator_hash, name="evaluator_hash")
        request_payload_json = _json_text(request_payload)
        if hash_json(dict(request_payload)) != request_digest:
            raise ValueError("protected evaluation request payload hash differs")
        partition_payload = _protected_partition_payload(protected_partition_hashes)
        partition_payload_json = _json_text(partition_payload)
        partition_set_hash = hash_json(partition_payload)
        consumption_id = _protected_consumption_id(
            experiment_spec_hash=experiment_digest,
            approval_hash=approval_digest,
            protected_partition_set_hash=partition_set_hash,
            request_hash=request_digest,
            evaluator_hash=evaluator_digest,
        )
        timestamp = _iso(at)
        with self._transaction() as connection:
            self._record_locked(connection, experiment_digest)
            spec = _load_experiment_spec_locked(connection, experiment_digest)
            expected_partitions = {
                partition.value: digest
                for partition, digest in spec.data_partitions.items()
                if partition in {DataPartition.TEST, DataPartition.HOLDOUT}
            }
            if (
                spec.evaluation_spec_hash != evaluation_digest
                or partition_payload["partitions"] != expected_partitions
            ):
                raise ExperimentRegistryConflict(
                    "protected evaluation request differs from immutable experiment"
                )
            self._require_approval_locked(
                connection,
                approval_hash=approval_digest,
                experiment_spec_hash=experiment_digest,
                action=PROTECTED_EVALUATION_ACTION,
                scope=PROTECTED_EVALUATION_SCOPE,
                artifact_hash=evaluation_digest,
                at=timestamp,
                allowed_roles=PROTECTED_EVALUATION_ALLOWED_ROLES,
            )
            rows = connection.execute(
                """SELECT * FROM protected_evaluation_consumptions
                   WHERE protected_partition_set_hash=?
                      OR approval_hash=?
                      OR request_hash=?""",
                (partition_set_hash, approval_digest, request_digest),
            ).fetchall()
            if rows:
                if len(rows) != 1:
                    raise ExperimentRegistryConflict(
                        "protected evaluation identity collision"
                    )
                record = _protected_consumption_record(rows[0])
                expected = {
                    "consumption_id": consumption_id,
                    "experiment_spec_hash": experiment_digest,
                    "approval_hash": approval_digest,
                    "protected_partition_set_hash": partition_set_hash,
                    "request_hash": request_digest,
                    "evaluator_hash": evaluator_digest,
                }
                if any(
                    getattr(record, name) != value for name, value in expected.items()
                ):
                    raise ExperimentRegistryConflict(
                        "protected partition set was already consumed by another "
                        "authorization"
                    )
                if (
                    rows[0]["protected_partitions_json"] != partition_payload_json
                    or rows[0]["request_payload_json"] != request_payload_json
                ):
                    raise ExperimentRegistryConflict(
                        "immutable protected evaluation request differs"
                    )
                if record.status == "completed":
                    return record
                raise ExperimentRegistryConflict(
                    "protected evaluation consumption is indeterminate and "
                    "requires human review"
                )
            connection.execute(
                """INSERT INTO protected_evaluation_consumptions(
                    consumption_id,experiment_spec_hash,approval_hash,
                    protected_partition_set_hash,protected_partitions_json,
                    request_hash,request_payload_json,evaluator_hash,status,
                    reserved_at,completed_at,result_hash,result_payload_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    consumption_id,
                    experiment_digest,
                    approval_digest,
                    partition_set_hash,
                    partition_payload_json,
                    request_digest,
                    request_payload_json,
                    evaluator_digest,
                    "reserved",
                    timestamp,
                    None,
                    None,
                    None,
                ),
            )
            self._append_event_locked(
                connection,
                experiment_digest,
                "protected_evaluation_reserved",
                {
                    "consumption_id": consumption_id,
                    "approval_hash": approval_digest,
                    "protected_partition_set_hash": partition_set_hash,
                    "request_hash": request_digest,
                    "evaluator_hash": evaluator_digest,
                },
                timestamp,
            )
            row = connection.execute(
                """SELECT * FROM protected_evaluation_consumptions
                   WHERE consumption_id=?""",
                (consumption_id,),
            ).fetchone()
            if row is None:  # pragma: no cover - guarded by the transaction
                raise ExperimentRegistryConflict(
                    "protected evaluation reservation disappeared"
                )
            return _protected_consumption_record(row)

    def complete_protected_evaluation(
        self,
        *,
        consumption_id: str,
        experiment_spec_hash: str,
        approval_hash: str,
        evaluation_spec_hash: str,
        result_hash: str,
        result_payload: Mapping[str, object],
        at: datetime,
    ) -> ProtectedEvaluationConsumptionRecord:
        """Seal one redacted terminal result against its prior reservation."""

        consumption_digest = require_sha256(consumption_id, name="consumption_id")
        experiment_digest = require_sha256(
            experiment_spec_hash, name="experiment_spec_hash"
        )
        approval_digest = require_sha256(approval_hash, name="approval_hash")
        evaluation_digest = require_sha256(
            evaluation_spec_hash, name="evaluation_spec_hash"
        )
        result_digest = require_sha256(result_hash, name="result_hash")
        result_payload_json = _json_text(result_payload)
        if hash_json(dict(result_payload)) != result_digest:
            raise ValueError("protected evaluation result payload hash differs")
        timestamp = _iso(at)
        with self._transaction() as connection:
            self._record_locked(connection, experiment_digest)
            self._require_approval_locked(
                connection,
                approval_hash=approval_digest,
                experiment_spec_hash=experiment_digest,
                action=PROTECTED_EVALUATION_ACTION,
                scope=PROTECTED_EVALUATION_SCOPE,
                artifact_hash=evaluation_digest,
                at=timestamp,
                allowed_roles=PROTECTED_EVALUATION_ALLOWED_ROLES,
            )
            row = connection.execute(
                """SELECT * FROM protected_evaluation_consumptions
                   WHERE consumption_id=?""",
                (consumption_digest,),
            ).fetchone()
            if row is None:
                raise ExperimentRegistryConflict(
                    "protected evaluation was not reserved"
                )
            record = _protected_consumption_record(row)
            if (
                record.experiment_spec_hash != experiment_digest
                or record.approval_hash != approval_digest
            ):
                raise ExperimentRegistryConflict(
                    "protected evaluation completion identity differs"
                )
            if record.status == "completed":
                if (
                    record.result_hash != result_digest
                    or row["result_payload_json"] != result_payload_json
                ):
                    raise ExperimentRegistryConflict(
                        "immutable protected evaluation result differs"
                    )
                return record
            cursor = connection.execute(
                """UPDATE protected_evaluation_consumptions
                   SET status='completed',completed_at=?,result_hash=?,
                       result_payload_json=?
                   WHERE consumption_id=? AND status='reserved'""",
                (
                    timestamp,
                    result_digest,
                    result_payload_json,
                    consumption_digest,
                ),
            )
            if cursor.rowcount != 1:
                raise ExperimentRegistryConflict(
                    "protected evaluation reservation state changed"
                )
            self._append_event_locked(
                connection,
                experiment_digest,
                "protected_evaluation_completed",
                {
                    "consumption_id": consumption_digest,
                    "result_hash": result_digest,
                },
                timestamp,
            )
            completed = connection.execute(
                """SELECT * FROM protected_evaluation_consumptions
                   WHERE consumption_id=?""",
                (consumption_digest,),
            ).fetchone()
            if completed is None:  # pragma: no cover - guarded by the transaction
                raise ExperimentRegistryConflict(
                    "protected evaluation completion disappeared"
                )
            return _protected_consumption_record(completed)

    def load_protected_evaluation_consumption(
        self, consumption_id: str
    ) -> ProtectedEvaluationConsumptionRecord:
        digest = require_sha256(consumption_id, name="consumption_id")
        row = self.connection.execute(
            """SELECT * FROM protected_evaluation_consumptions
               WHERE consumption_id=?""",
            (digest,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown protected evaluation consumption:{digest}")
        return _protected_consumption_record(row)

    def load_receipt(self, experiment_spec_hash: str) -> ExperimentReceipt:
        digest = require_sha256(experiment_spec_hash, name="experiment_spec_hash")
        row = self.connection.execute(
            """SELECT receipt_hash,payload_json FROM experiment_receipts
               WHERE experiment_spec_hash=?""",
            (digest,),
        ).fetchone()
        if row is None:
            raise KeyError(f"experiment receipt is not sealed:{digest}")
        payload = json.loads(row["payload_json"])
        if not isinstance(payload, Mapping):
            raise ExperimentRegistryConflict("stored receipt payload is invalid")
        receipt = ExperimentReceipt.from_dict(payload)
        if receipt.content_hash != row["receipt_hash"]:
            raise ExperimentRegistryConflict("stored receipt hash differs")
        return receipt

    def get_experiment(self, experiment_spec_hash: str) -> ExperimentRecord:
        digest = require_sha256(experiment_spec_hash, name="experiment_spec_hash")
        return self._record_locked(self.connection, digest)

    def load_experiment_spec(self, experiment_spec_hash: str) -> ExperimentSpec:
        """Reload and integrity-check the canonical registered specification."""

        digest = require_sha256(experiment_spec_hash, name="experiment_spec_hash")
        return _load_experiment_spec_locked(self.connection, digest)

    def load_bound_components(
        self,
        experiment_spec_hash: str,
    ) -> tuple[RegisteredExperimentComponent, ...]:
        """Reload the exact component closure in ExperimentSpec binding order."""

        digest = require_sha256(experiment_spec_hash, name="experiment_spec_hash")
        spec = _load_experiment_spec_locked(self.connection, digest)
        expected_bindings = spec.component_bindings()
        rows = self.connection.execute(
            """SELECT binding.role,binding.component_hash,
                      component.component_type,component.descriptor_json
               FROM experiment_components AS binding
               JOIN components AS component
                 ON component.component_hash=binding.component_hash
               WHERE binding.experiment_spec_hash=?""",
            (digest,),
        ).fetchall()
        by_role = {str(row["role"]): row for row in rows}
        if len(by_role) != len(rows) or set(by_role) != {
            role for role, _ in expected_bindings
        }:
            raise ExperimentRegistryConflict("stored experiment component roles differ")
        loaded: list[RegisteredExperimentComponent] = []
        for role, expected_hash in expected_bindings:
            row = by_role[role]
            component_hash = str(row["component_hash"])
            component_type = str(row["component_type"])
            if (
                component_hash != expected_hash
                or component_type != role.split(":", 1)[0]
            ):
                raise ExperimentRegistryConflict(
                    "stored experiment component binding differs"
                )
            payload = json.loads(str(row["descriptor_json"]))
            if not isinstance(payload, Mapping):
                raise ExperimentRegistryConflict(
                    "stored component descriptor is not an object"
                )
            descriptor = dict(payload)
            try:
                observed_hash, canonical = _component_registration_values(
                    component_hash=component_hash,
                    component_type=component_type,
                    descriptor=descriptor,
                )
            except (TypeError, ValueError) as exc:
                raise ExperimentRegistryConflict(
                    "stored component descriptor integrity differs"
                ) from exc
            if observed_hash != expected_hash or canonical != row["descriptor_json"]:
                raise ExperimentRegistryConflict(
                    "stored component descriptor integrity differs"
                )
            loaded.append(
                RegisteredExperimentComponent(
                    role=role,
                    component_hash=component_hash,
                    component_type=component_type,
                    descriptor=MappingProxyType(descriptor),
                )
            )
        return tuple(loaded)

    def artifact_hashes(self, experiment_spec_hash: str) -> tuple[str, ...]:
        digest = require_sha256(experiment_spec_hash, name="experiment_spec_hash")
        self._record_locked(self.connection, digest)
        rows = self.connection.execute(
            """SELECT artifact_hash FROM artifacts
               WHERE experiment_spec_hash=? ORDER BY artifact_hash""",
            (digest,),
        ).fetchall()
        return tuple(str(row["artifact_hash"]) for row in rows)

    def load_artifact_descriptor(
        self,
        experiment_spec_hash: str,
        artifact_hash: str,
    ) -> Mapping[str, object]:
        """Load and integrity-check one immutable artifact descriptor."""

        experiment_digest = require_sha256(
            experiment_spec_hash,
            name="experiment_spec_hash",
        )
        artifact_digest = require_sha256(artifact_hash, name="artifact_hash")
        self._record_locked(self.connection, experiment_digest)
        row = self.connection.execute(
            """SELECT descriptor_json FROM artifacts
               WHERE experiment_spec_hash=? AND artifact_hash=?""",
            (experiment_digest, artifact_digest),
        ).fetchone()
        if row is None:
            raise KeyError(
                "artifact is not registered to experiment:"
                f"{experiment_digest}:{artifact_digest}"
            )
        payload = json.loads(row["descriptor_json"])
        if not isinstance(payload, Mapping):
            raise ExperimentRegistryConflict(
                "stored artifact descriptor is not an object"
            )
        descriptor = dict(payload)
        if (
            descriptor.get("artifact_hash") != artifact_digest
            or _json_text(descriptor) != row["descriptor_json"]
        ):
            raise ExperimentRegistryConflict(
                "stored artifact descriptor integrity differs"
            )
        return descriptor

    def load_artifact_descriptor_by_logical_name(
        self,
        experiment_spec_hash: str,
        logical_name: str,
    ) -> Mapping[str, object]:
        """Load one immutable descriptor through its experiment-local name."""

        experiment_digest = require_sha256(
            experiment_spec_hash,
            name="experiment_spec_hash",
        )
        if not _safe_code(logical_name):
            raise ValueError("artifact logical_name is invalid")
        self._record_locked(self.connection, experiment_digest)
        row = self.connection.execute(
            """SELECT artifact_hash FROM artifacts
               WHERE experiment_spec_hash=? AND logical_name=?""",
            (experiment_digest, logical_name),
        ).fetchone()
        if row is None:
            raise KeyError(
                "artifact logical name is not registered to experiment:"
                f"{experiment_digest}:{logical_name}"
            )
        descriptor = self.load_artifact_descriptor(
            experiment_digest,
            str(row["artifact_hash"]),
        )
        if descriptor.get("logical_name") != logical_name:
            raise ExperimentRegistryConflict(
                "stored artifact logical name integrity differs"
            )
        return descriptor

    def verify_integrity(self) -> tuple[str, ...]:
        return verify_experiment_registry_connection(self.connection)

    @staticmethod
    def _require_approval_locked(
        connection: sqlite3.Connection,
        *,
        approval_hash: str | None,
        experiment_spec_hash: str,
        action: str,
        scope: str,
        artifact_hash: str,
        at: str,
        allowed_roles: frozenset[str],
    ) -> ApprovalGrant:
        if approval_hash is None:
            raise ExperimentRegistryConflict("active human approval is required")
        digest = require_sha256(approval_hash, name="approval_hash")
        row = connection.execute(
            """SELECT approval.*,revocation.approval_hash AS revoked
               FROM human_approvals AS approval
               LEFT JOIN approval_revocations AS revocation
                 ON revocation.approval_hash=approval.approval_hash
               WHERE approval.approval_hash=?""",
            (digest,),
        ).fetchone()
        if row is None:
            raise ExperimentRegistryConflict("human approval is not registered")
        if row["revoked"] is not None:
            raise ExperimentRegistryConflict("human approval is revoked")
        expected = {
            "experiment_spec_hash": experiment_spec_hash,
            "action": action,
            "scope": scope,
            "artifact_hash": artifact_hash,
        }
        if any(row[name] != value for name, value in expected.items()):
            raise ExperimentRegistryConflict("human approval scope or artifact differs")
        if row["actor_role"] not in allowed_roles:
            raise ExperimentRegistryConflict(
                "human approval actor role is unauthorized"
            )
        moment = _parse_time(at)
        if not (
            _parse_time(row["issued_at"]) <= moment < _parse_time(row["expires_at"])
        ):
            raise ExperimentRegistryConflict(
                "human approval is not active at seal time"
            )
        payload = json.loads(row["payload_json"])
        if not isinstance(payload, Mapping):
            raise ExperimentRegistryConflict("stored approval payload is invalid")
        grant = ApprovalGrant(**dict(payload))
        if grant.content_hash != digest:
            raise ExperimentRegistryConflict("stored approval hash differs")
        grant_payload = grant.to_dict()
        if any(
            row[name] != value
            for name, value in grant_payload.items()
            if name != "schema_version"
        ):
            raise ExperimentRegistryConflict(
                "stored approval columns differ from immutable payload"
            )
        return grant

    @staticmethod
    def _record_locked(connection: sqlite3.Connection, digest: str) -> ExperimentRecord:
        row = connection.execute(
            """SELECT experiment_spec_hash,experiment_id,version,profile,status,
                      registered_at,updated_at
               FROM experiments WHERE experiment_spec_hash=?""",
            (digest,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown experiment:{digest}")
        return ExperimentRecord(**dict(row))

    @staticmethod
    def _known_hash_locked(connection: sqlite3.Connection, digest: str) -> bool:
        for table, column in (
            ("components", "component_hash"),
            ("artifacts", "artifact_hash"),
            ("experiment_receipts", "receipt_hash"),
        ):
            if (
                connection.execute(
                    f"SELECT 1 FROM {table} WHERE {column}=?", (digest,)
                ).fetchone()
                is not None
            ):
                return True
        return False

    @staticmethod
    def _append_event_locked(
        connection: sqlite3.Connection,
        experiment_spec_hash: str,
        event_type: str,
        payload: Mapping[str, object],
        event_at: str,
    ) -> str:
        previous_row = connection.execute(
            "SELECT event_hash FROM experiment_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        previous = None if previous_row is None else str(previous_row["event_hash"])
        core = {
            "experiment_spec_hash": experiment_spec_hash,
            "event_type": event_type,
            "event_at": event_at,
            "payload": dict(payload),
            "previous_event_hash": previous,
        }
        event_hash: str = hash_json(core)
        connection.execute(
            """INSERT INTO experiment_events(
                experiment_spec_hash,event_type,event_at,payload_json,
                previous_event_hash,event_hash
            ) VALUES(?,?,?,?,?,?)""",
            (
                experiment_spec_hash,
                event_type,
                event_at,
                _json_text(payload),
                previous,
                event_hash,
            ),
        )
        return event_hash


def verify_experiment_registry_connection(
    connection: sqlite3.Connection,
) -> tuple[str, ...]:
    """Recompute immutable identities for live and read-only registry users."""

    errors: list[str] = []
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        errors.append("sqlite_integrity_failed")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        errors.append("sqlite_foreign_key_failed")
    table_names = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }

    component_types: dict[str, str] = {}
    for row in connection.execute(
        """SELECT component_hash,component_type,descriptor_json
           FROM components ORDER BY component_hash"""
    ):
        digest = str(row["component_hash"])
        component_type = str(row["component_type"])
        component_types[digest] = component_type
        try:
            payload = json.loads(row["descriptor_json"])
            if not isinstance(payload, Mapping):
                raise TypeError("component descriptor is not an object")
            descriptor = dict(payload)
            if _json_text(descriptor) != row["descriptor_json"]:
                errors.append(f"component_descriptor_canonical:{digest}")
            if descriptor.get("content_hash") != digest:
                errors.append(f"component_descriptor_hash:{digest}")
            if component_type == "scientific_lineage":
                if set(descriptor) != {
                    "content_hash",
                    "schema_version",
                    "manifest",
                }:
                    errors.append(f"scientific_lineage_descriptor:{digest}")
                manifest = descriptor.get("manifest")
                if not isinstance(manifest, Mapping):
                    errors.append(f"scientific_lineage_manifest:{digest}")
                else:
                    manifest_payload = dict(manifest)
                    try:
                        # Imported lazily to keep the experiment registry below
                        # the research assembly layer at module-import time.
                        from llm_alpha_mining.research.contracts.lineage import (
                            ResearchScientificLineageManifestV1,
                            ResearchScientificLineageManifestV2,
                        )

                        lineage: (
                            ResearchScientificLineageManifestV1
                            | ResearchScientificLineageManifestV2
                        )
                        lineage_schema = manifest_payload.get("schema_version")
                        if lineage_schema == "research-scientific-lineage-manifest/v1":
                            lineage = ResearchScientificLineageManifestV1.from_mapping(
                                manifest_payload
                            )
                        elif (
                            lineage_schema == "research-scientific-lineage-manifest/v2"
                        ):
                            lineage = ResearchScientificLineageManifestV2.from_mapping(
                                manifest_payload
                            )
                        else:
                            raise ValueError("unsupported scientific lineage schema")
                    except (TypeError, ValueError):
                        errors.append(f"scientific_lineage_payload:{digest}")
                    else:
                        if (
                            descriptor.get("schema_version") != lineage.schema_version
                            or lineage.to_dict() != manifest_payload
                            or lineage.content_hash != digest
                        ):
                            errors.append(f"scientific_lineage_hash:{digest}")
                    if manifest_payload.get("schema_version") != descriptor.get(
                        "schema_version"
                    ):
                        errors.append(f"scientific_lineage_hash:{digest}")
        except (TypeError, ValueError, json.JSONDecodeError):
            errors.append(f"component_descriptor_payload:{digest}")

    experiment_specs: dict[str, ExperimentSpec] = {}
    for row in connection.execute(
        """SELECT experiment_spec_hash,experiment_id,version,profile,spec_json
           FROM experiments ORDER BY experiment_spec_hash"""
    ):
        digest = str(row["experiment_spec_hash"])
        try:
            payload = json.loads(row["spec_json"])
            if not isinstance(payload, Mapping):
                raise TypeError("experiment spec is not an object")
            spec = ExperimentSpec.from_mapping(dict(payload))
            experiment_specs[digest] = spec
            if _json_text(spec.to_dict()) != row["spec_json"]:
                errors.append(f"experiment_spec_canonical:{digest}")
            if spec.content_hash != digest:
                errors.append(f"experiment_spec_hash:{digest}")
            if (
                row["experiment_id"] != spec.experiment_id
                or row["version"] != spec.version
                or row["profile"] != spec.profile.value
            ):
                errors.append(f"experiment_spec_columns:{digest}")
            expected = dict(spec.component_bindings())
            component_rows = connection.execute(
                """SELECT role,component_hash FROM experiment_components
                   WHERE experiment_spec_hash=? ORDER BY role""",
                (digest,),
            ).fetchall()
            observed = {
                str(item["role"]): str(item["component_hash"])
                for item in component_rows
            }
            if len(observed) != len(component_rows) or observed != expected:
                errors.append(f"experiment_component_bindings:{digest}")
            for role, component_hash in observed.items():
                if component_types.get(component_hash) != role.split(":", 1)[0]:
                    errors.append(f"experiment_component_type:{digest}:{role}")
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            errors.append(f"experiment_spec_payload:{digest}")

    previous: str | None = None
    for row in connection.execute("SELECT * FROM experiment_events ORDER BY sequence"):
        sequence = row["sequence"]
        if row["previous_event_hash"] != previous:
            errors.append(f"event_chain_previous:{sequence}")
        try:
            payload = json.loads(row["payload_json"])
            core = {
                "experiment_spec_hash": row["experiment_spec_hash"],
                "event_type": row["event_type"],
                "event_at": row["event_at"],
                "payload": payload,
                "previous_event_hash": row["previous_event_hash"],
            }
            if hash_json(core) != row["event_hash"]:
                errors.append(f"event_hash:{sequence}")
        except (TypeError, ValueError, json.JSONDecodeError):
            errors.append(f"event_payload:{sequence}")
        previous = row["event_hash"]
    if "stage_artifact_publications" not in table_names:
        errors.append("stage_artifact_publications_missing")
        publication_rows: tuple[sqlite3.Row, ...] = ()
    else:
        publication_rows = tuple(
            connection.execute(
                """SELECT * FROM stage_artifact_publications
                   ORDER BY experiment_spec_hash,attempt_id"""
            )
        )
    for row in publication_rows:
        identity = f"{row['experiment_spec_hash']}:{row['attempt_id']}"
        try:
            experiment_hash = require_sha256(
                str(row["experiment_spec_hash"]),
                name="stored stage publication experiment",
            )
            artifact_hash = require_sha256(
                str(row["artifact_hash"]),
                name="stored stage publication artifact",
            )
            attempt_id = int(row["attempt_id"])
            stage = str(row["stage"])
            if attempt_id <= 0 or not _safe_code(stage):
                raise ValueError("stage publication attempt/stage differs")
            descriptor_raw = json.loads(row["descriptor_json"])
            parents_raw = json.loads(row["parent_hashes_json"])
            if not isinstance(descriptor_raw, Mapping) or not isinstance(
                parents_raw, list
            ):
                raise TypeError("stage publication payload type differs")
            descriptor = dict(descriptor_raw)
            if (
                _json_text(descriptor) != row["descriptor_json"]
                or canonical_json_bytes(parents_raw).decode("utf-8")
                != row["parent_hashes_json"]
                or not all(isinstance(parent, str) for parent in parents_raw)
                or len(parents_raw) != len(set(parents_raw))
            ):
                raise ValueError("stage publication payload is noncanonical")
            if (
                descriptor.get("artifact_hash") != artifact_hash
                or descriptor.get("stage") != stage
                or descriptor.get("attempt_id") != attempt_id
            ):
                errors.append(f"stage_publication_binding:{identity}")
            for parent in parents_raw:
                parent_hash = require_sha256(
                    str(parent), name="stored stage publication parent"
                )
                if parent_hash == artifact_hash:
                    errors.append(f"stage_publication_self_lineage:{identity}")
            status = str(row["status"])
            if status not in {"provisional", "committed", "abandoned"}:
                errors.append(f"stage_publication_status:{identity}")
            if _parse_time(str(row["updated_at"])) < _parse_time(
                str(row["created_at"])
            ):
                errors.append(f"stage_publication_time:{identity}")
            if status == "committed":
                artifact = connection.execute(
                    """SELECT descriptor_json FROM artifacts
                       WHERE experiment_spec_hash=? AND artifact_hash=?""",
                    (experiment_hash, artifact_hash),
                ).fetchone()
                if (
                    artifact is None
                    or artifact["descriptor_json"] != row["descriptor_json"]
                ):
                    errors.append(f"stage_publication_artifact:{identity}")
                for parent in parents_raw:
                    edge = connection.execute(
                        """SELECT experiment_spec_hash FROM lineage
                           WHERE child_hash=? AND parent_hash=?
                             AND relationship='stage_input'""",
                        (artifact_hash, str(parent)),
                    ).fetchone()
                    if edge is None or edge["experiment_spec_hash"] != experiment_hash:
                        errors.append(f"stage_publication_lineage:{identity}")
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            errors.append(f"stage_publication_payload:{identity}")
    for row in connection.execute(
        "SELECT receipt_hash,payload_json FROM experiment_receipts"
    ):
        try:
            payload = json.loads(row["payload_json"])
            if not isinstance(payload, Mapping):
                raise TypeError("receipt payload is not an object")
            receipt = ExperimentReceipt.from_dict(dict(payload))
            if receipt.content_hash != row["receipt_hash"]:
                errors.append(f"receipt_hash:{row['receipt_hash']}")
        except (TypeError, ValueError, json.JSONDecodeError):
            errors.append(f"receipt_payload:{row['receipt_hash']}")

    approvals: dict[str, ApprovalGrant] = {}
    for row in connection.execute(
        """SELECT * FROM human_approvals ORDER BY approval_hash"""
    ):
        digest = str(row["approval_hash"])
        try:
            payload = json.loads(row["payload_json"])
            if not isinstance(payload, Mapping):
                raise TypeError("approval payload is not an object")
            grant = ApprovalGrant(**dict(payload))
            approvals[digest] = grant
            if _json_text(grant.to_dict()) != row["payload_json"]:
                errors.append(f"approval_payload_canonical:{digest}")
            if grant.content_hash != digest:
                errors.append(f"approval_hash:{digest}")
            expected_columns = {
                "experiment_spec_hash": grant.experiment_spec_hash,
                "action": grant.action,
                "scope": grant.scope,
                "artifact_hash": grant.artifact_hash,
                "actor": grant.actor,
                "actor_role": grant.actor_role,
                "signature_hash": grant.signature_hash,
                "nonce": grant.nonce,
                "issued_at": grant.issued_at,
                "expires_at": grant.expires_at,
            }
            if any(row[name] != value for name, value in expected_columns.items()):
                errors.append(f"approval_columns:{digest}")
        except (TypeError, ValueError, json.JSONDecodeError):
            errors.append(f"approval_payload:{digest}")

    if "protected_evaluation_consumptions" not in table_names:
        errors.append("protected_evaluation_consumptions_missing")
        protected_rows: tuple[sqlite3.Row, ...] = ()
    else:
        protected_rows = tuple(
            connection.execute(
                """SELECT * FROM protected_evaluation_consumptions
                   ORDER BY consumption_id"""
            )
        )
    for row in protected_rows:
        consumption_id = str(row["consumption_id"])
        try:
            experiment_digest = require_sha256(
                str(row["experiment_spec_hash"]),
                name="stored protected experiment",
            )
            approval_digest = require_sha256(
                str(row["approval_hash"]),
                name="stored protected approval",
            )
            partition_set_digest = require_sha256(
                str(row["protected_partition_set_hash"]),
                name="stored protected partition set",
            )
            request_digest = require_sha256(
                str(row["request_hash"]),
                name="stored protected request",
            )
            evaluator_digest = require_sha256(
                str(row["evaluator_hash"]),
                name="stored protected evaluator",
            )
            partition_payload = json.loads(row["protected_partitions_json"])
            request_payload = json.loads(row["request_payload_json"])
            if not isinstance(partition_payload, Mapping) or not isinstance(
                request_payload, Mapping
            ):
                raise TypeError("protected evaluation payload is not an object")
            normalized_partitions = _protected_partition_payload(
                dict(partition_payload).get("partitions", {})
            )
            if (
                dict(partition_payload) != normalized_partitions
                or _json_text(normalized_partitions) != row["protected_partitions_json"]
                or hash_json(normalized_partitions) != partition_set_digest
            ):
                errors.append(f"protected_partition_set:{consumption_id}")
            if (
                _json_text(dict(request_payload)) != row["request_payload_json"]
                or hash_json(dict(request_payload)) != request_digest
            ):
                errors.append(f"protected_request:{consumption_id}")
            expected_consumption_id = _protected_consumption_id(
                experiment_spec_hash=experiment_digest,
                approval_hash=approval_digest,
                protected_partition_set_hash=partition_set_digest,
                request_hash=request_digest,
                evaluator_hash=evaluator_digest,
            )
            if expected_consumption_id != consumption_id:
                errors.append(f"protected_consumption_id:{consumption_id}")
            protected_grant = approvals.get(approval_digest)
            protected_spec = experiment_specs.get(experiment_digest)
            expected_protected_partitions = (
                {}
                if protected_spec is None
                else {
                    partition.value: digest
                    for partition, digest in protected_spec.data_partitions.items()
                    if partition in {DataPartition.TEST, DataPartition.HOLDOUT}
                }
            )
            if (
                protected_grant is None
                or protected_spec is None
                or protected_grant.experiment_spec_hash != experiment_digest
                or protected_grant.action != PROTECTED_EVALUATION_ACTION
                or protected_grant.scope != PROTECTED_EVALUATION_SCOPE
                or protected_grant.artifact_hash != protected_spec.evaluation_spec_hash
            ):
                errors.append(f"protected_approval_binding:{consumption_id}")
            if (
                dict(partition_payload).get("partitions")
                != expected_protected_partitions
            ):
                errors.append(f"protected_experiment_binding:{consumption_id}")
            status = str(row["status"])
            if status == "completed":
                result_payload = json.loads(row["result_payload_json"])
                if not isinstance(result_payload, Mapping):
                    raise TypeError("protected result payload is not an object")
                if (
                    _json_text(dict(result_payload)) != row["result_payload_json"]
                    or hash_json(dict(result_payload)) != row["result_hash"]
                ):
                    errors.append(f"protected_result:{consumption_id}")
            elif status != "reserved":
                errors.append(f"protected_status:{consumption_id}")
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            errors.append(f"protected_consumption_payload:{consumption_id}")
    return tuple(errors)


def _register_artifact_locked(
    connection: sqlite3.Connection,
    *,
    experiment_hash: str,
    digest: str,
    logical_name: str,
    kind: str,
    normalized_location: str,
    media_type: str,
    size_bytes: int,
    payload_json: str,
    timestamp: str,
) -> ArtifactRecord:
    existing = connection.execute(
        "SELECT * FROM artifacts WHERE artifact_hash=?", (digest,)
    ).fetchone()
    if existing is not None:
        if (
            existing["experiment_spec_hash"] != experiment_hash
            or existing["descriptor_json"] != payload_json
        ):
            raise ExperimentRegistryConflict("immutable artifact descriptor differs")
        return _artifact_record(existing)
    try:
        connection.execute(
            """INSERT INTO artifacts(
                artifact_hash,experiment_spec_hash,logical_name,kind,location,
                media_type,size_bytes,descriptor_json,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                digest,
                experiment_hash,
                logical_name,
                kind,
                normalized_location,
                media_type,
                size_bytes,
                payload_json,
                timestamp,
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise ExperimentRegistryConflict(
            "artifact logical name is already registered"
        ) from exc
    ExperimentRegistry._append_event_locked(
        connection,
        experiment_hash,
        "artifact_registered",
        {"artifact_hash": digest, "logical_name": logical_name, "kind": kind},
        timestamp,
    )
    row = connection.execute(
        "SELECT * FROM artifacts WHERE artifact_hash=?", (digest,)
    ).fetchone()
    if row is None:  # pragma: no cover
        raise RuntimeError("artifact insert disappeared")
    return _artifact_record(row)


def _component_registration_values(
    *,
    component_hash: str,
    component_type: str,
    descriptor: Mapping[str, object],
) -> tuple[str, str]:
    digest = require_sha256(component_hash, name="component_hash")
    if component_type not in _COMPONENT_TYPES:
        raise ValueError(f"unsupported experiment component type:{component_type}")
    payload = dict(descriptor)
    if payload.get("content_hash") != digest:
        raise ValueError("component descriptor must bind its content_hash")
    return digest, _json_text(payload)


def _register_component_locked(
    connection: sqlite3.Connection,
    *,
    digest: str,
    component_type: str,
    payload_json: str,
    timestamp: str,
) -> str:
    existing = connection.execute(
        """SELECT component_type,descriptor_json FROM components
           WHERE component_hash=?""",
        (digest,),
    ).fetchone()
    if existing is not None:
        if (
            existing["component_type"] != component_type
            or existing["descriptor_json"] != payload_json
        ):
            raise ExperimentRegistryConflict("immutable component descriptor differs")
        return digest
    connection.execute(
        """INSERT INTO components(
            component_hash,component_type,descriptor_json,registered_at
        ) VALUES(?,?,?,?)""",
        (digest, component_type, payload_json, timestamp),
    )
    return digest


def _register_experiment_locked(
    connection: sqlite3.Connection,
    *,
    spec: ExperimentSpec,
    timestamp: str,
) -> ExperimentRecord:
    digest = spec.content_hash
    payload_json = _json_text(spec.to_dict())
    bindings = spec.component_bindings()
    existing = connection.execute(
        """SELECT experiment_spec_hash,spec_json FROM experiments
           WHERE experiment_id=? AND version=?""",
        (spec.experiment_id, spec.version),
    ).fetchone()
    if existing is not None:
        if (
            existing["experiment_spec_hash"] != digest
            or existing["spec_json"] != payload_json
        ):
            raise ExperimentRegistryConflict(
                "experiment id/version already binds another immutable spec"
            )
        return ExperimentRegistry._record_locked(connection, digest)
    for role, component_hash in bindings:
        component = connection.execute(
            "SELECT component_type FROM components WHERE component_hash=?",
            (component_hash,),
        ).fetchone()
        expected_type = role.split(":", 1)[0]
        if component is None:
            raise ExperimentRegistryConflict(
                f"experiment component is not registered:{role}"
            )
        if component["component_type"] != expected_type:
            raise ExperimentRegistryConflict(
                f"experiment component type differs:{role}"
            )
    connection.execute(
        """INSERT INTO experiments(
            experiment_spec_hash,experiment_id,version,profile,status,
            spec_json,registered_at,updated_at
        ) VALUES(?,?,?,?,?,?,?,?)""",
        (
            digest,
            spec.experiment_id,
            spec.version,
            spec.profile.value,
            "registered",
            payload_json,
            timestamp,
            timestamp,
        ),
    )
    connection.executemany(
        """INSERT INTO experiment_components(
            experiment_spec_hash,role,component_hash
        ) VALUES(?,?,?)""",
        [(digest, role, component_hash) for role, component_hash in bindings],
    )
    ExperimentRegistry._append_event_locked(
        connection,
        digest,
        "experiment_registered",
        {"profile": spec.profile.value, "component_count": len(bindings)},
        timestamp,
    )
    return ExperimentRegistry._record_locked(connection, digest)


def _register_approval_locked(
    connection: sqlite3.Connection,
    grant: ApprovalGrant,
) -> str:
    digest: str = grant.content_hash
    payload_json = _json_text(grant.to_dict())
    existing = connection.execute(
        "SELECT payload_json FROM human_approvals WHERE approval_hash=?",
        (digest,),
    ).fetchone()
    if existing is not None:
        if existing["payload_json"] != payload_json:
            raise ExperimentRegistryConflict("immutable approval payload differs")
        return digest
    try:
        connection.execute(
            """INSERT INTO human_approvals(
                approval_hash,experiment_spec_hash,action,scope,artifact_hash,
                actor,actor_role,signature_hash,nonce,issued_at,expires_at,payload_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                digest,
                grant.experiment_spec_hash,
                grant.action,
                grant.scope,
                grant.artifact_hash,
                grant.actor,
                grant.actor_role,
                grant.signature_hash,
                grant.nonce,
                grant.issued_at,
                grant.expires_at,
                payload_json,
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise ExperimentRegistryConflict("approval nonce replay is forbidden") from exc
    ExperimentRegistry._append_event_locked(
        connection,
        grant.experiment_spec_hash,
        "human_approval_registered",
        {
            "approval_hash": digest,
            "action": grant.action,
            "scope": grant.scope,
            "artifact_hash": grant.artifact_hash,
            "actor_role": grant.actor_role,
        },
        grant.issued_at,
    )
    return digest


def _validate_protected_evaluation_grant(
    grant: ApprovalGrant,
    *,
    spec: ExperimentSpec,
    at: str,
) -> None:
    if ExperimentProfile(spec.profile) is not ExperimentProfile.PRODUCTION_CANDIDATE:
        raise ExperimentRegistryConflict(
            "protected evaluation approval requires production_candidate"
        )
    if not set(spec.data_partitions).intersection(
        {DataPartition.TEST, DataPartition.HOLDOUT}
    ):
        raise ExperimentRegistryConflict(
            "protected evaluation approval requires a test/holdout partition"
        )
    expected = {
        "experiment_spec_hash": spec.content_hash,
        "action": PROTECTED_EVALUATION_ACTION,
        "scope": PROTECTED_EVALUATION_SCOPE,
        "artifact_hash": spec.evaluation_spec_hash,
    }
    if any(getattr(grant, name) != value for name, value in expected.items()):
        raise ExperimentRegistryConflict(
            "protected evaluation approval scope or artifact differs"
        )
    if grant.actor_role not in PROTECTED_EVALUATION_ALLOWED_ROLES:
        raise ExperimentRegistryConflict(
            "protected evaluation approval actor role is unauthorized"
        )
    moment = _parse_time(at)
    if not (_parse_time(grant.issued_at) <= moment < _parse_time(grant.expires_at)):
        raise ExperimentRegistryConflict("protected evaluation approval is not active")


def _protected_partition_payload(
    protected_partition_hashes: Mapping[str, str],
) -> dict[str, object]:
    partitions = {
        str(name): require_sha256(
            digest,
            name=f"protected partition:{name}",
        )
        for name, digest in protected_partition_hashes.items()
    }
    if (
        not partitions
        or not set(partitions).issubset(_PROTECTED_PARTITION_NAMES)
        or len(partitions) != len(protected_partition_hashes)
    ):
        raise ValueError("protected partition set must contain only test/holdout")
    return {
        "schema_version": _PROTECTED_PARTITION_SET_SCHEMA,
        "partitions": dict(sorted(partitions.items())),
    }


def _protected_consumption_id(
    *,
    experiment_spec_hash: str,
    approval_hash: str,
    protected_partition_set_hash: str,
    request_hash: str,
    evaluator_hash: str,
) -> str:
    digest: str = hash_json(
        {
            "schema_version": _PROTECTED_CONSUMPTION_SCHEMA,
            "experiment_spec_hash": experiment_spec_hash,
            "approval_hash": approval_hash,
            "protected_partition_set_hash": protected_partition_set_hash,
            "request_hash": request_hash,
            "evaluator_hash": evaluator_hash,
        }
    )
    return digest


def _load_experiment_spec_locked(
    connection: sqlite3.Connection,
    experiment_spec_hash: str,
) -> ExperimentSpec:
    row = connection.execute(
        """SELECT experiment_spec_hash,experiment_id,version,profile,spec_json
           FROM experiments WHERE experiment_spec_hash=?""",
        (experiment_spec_hash,),
    ).fetchone()
    if row is None:
        raise KeyError(f"unknown experiment:{experiment_spec_hash}")
    try:
        payload = json.loads(row["spec_json"])
        if not isinstance(payload, Mapping):
            raise TypeError("experiment spec is not an object")
        spec = ExperimentSpec.from_mapping(dict(payload))
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise ExperimentRegistryConflict("stored experiment spec is invalid") from error
    if _json_text(spec.to_dict()) != row["spec_json"]:
        raise ExperimentRegistryConflict("stored experiment spec is not canonical")
    if spec.content_hash != experiment_spec_hash:
        raise ExperimentRegistryConflict("stored experiment spec hash differs")
    if (
        row["experiment_spec_hash"] != experiment_spec_hash
        or row["experiment_id"] != spec.experiment_id
        or row["version"] != spec.version
        or row["profile"] != spec.profile.value
    ):
        raise ExperimentRegistryConflict("stored experiment immutable columns differ")

    expected_bindings = tuple(sorted(spec.component_bindings()))
    component_rows = connection.execute(
        """SELECT binding.role,binding.component_hash,component.component_type
           FROM experiment_components AS binding
           LEFT JOIN components AS component
             ON component.component_hash=binding.component_hash
           WHERE binding.experiment_spec_hash=?
           ORDER BY binding.role""",
        (experiment_spec_hash,),
    ).fetchall()
    observed_bindings = tuple(
        (str(component["role"]), str(component["component_hash"]))
        for component in component_rows
    )
    if observed_bindings != expected_bindings:
        raise ExperimentRegistryConflict("stored experiment component bindings differ")
    for component in component_rows:
        role = str(component["role"])
        if component["component_type"] != role.split(":", 1)[0]:
            raise ExperimentRegistryConflict(
                f"stored experiment component type differs:{role}"
            )
    return spec


def _protected_consumption_record(
    row: sqlite3.Row,
) -> ProtectedEvaluationConsumptionRecord:
    payload: Mapping[str, object] | None = None
    raw_payload = row["result_payload_json"]
    if raw_payload is not None:
        parsed = json.loads(raw_payload)
        if not isinstance(parsed, Mapping):
            raise ExperimentRegistryConflict(
                "stored protected evaluation result is not an object"
            )
        payload = MappingProxyType(dict(parsed))
    return ProtectedEvaluationConsumptionRecord(
        consumption_id=str(row["consumption_id"]),
        experiment_spec_hash=str(row["experiment_spec_hash"]),
        approval_hash=str(row["approval_hash"]),
        protected_partition_set_hash=str(row["protected_partition_set_hash"]),
        request_hash=str(row["request_hash"]),
        evaluator_hash=str(row["evaluator_hash"]),
        status=str(row["status"]),
        reserved_at=str(row["reserved_at"]),
        completed_at=(
            None if row["completed_at"] is None else str(row["completed_at"])
        ),
        result_hash=None if row["result_hash"] is None else str(row["result_hash"]),
        result_payload=payload,
    )


def _artifact_record(row: sqlite3.Row) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_hash=row["artifact_hash"],
        experiment_spec_hash=row["experiment_spec_hash"],
        logical_name=row["logical_name"],
        kind=row["kind"],
        location=row["location"],
        media_type=row["media_type"],
        size_bytes=int(row["size_bytes"]),
        created_at=row["created_at"],
    )


def _stage_artifact_publication(row: sqlite3.Row) -> StageArtifactPublication:
    raw_parents = json.loads(row["parent_hashes_json"])
    if not isinstance(raw_parents, list) or not all(
        isinstance(parent, str) for parent in raw_parents
    ):
        raise ExperimentRegistryConflict(
            "stored stage artifact publication parents are invalid"
        )
    parents = tuple(str(parent) for parent in raw_parents)
    for offset, parent in enumerate(parents):
        require_sha256(parent, name=f"stored stage publication parent:{offset}")
    return StageArtifactPublication(
        experiment_spec_hash=str(row["experiment_spec_hash"]),
        attempt_id=int(row["attempt_id"]),
        stage=str(row["stage"]),
        artifact_hash=str(row["artifact_hash"]),
        status=str(row["status"]),
        parent_hashes=parents,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _json_text(value: Mapping[str, object]) -> str:
    encoded: bytes = canonical_json_bytes(dict(value))
    return encoded.decode("utf-8")


def _safe_relative_location(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("artifact location must be a safe relative path")
    normalized = path.as_posix()
    if normalized != value:
        raise ValueError("artifact location must be normalized")
    return normalized


def _safe_code(value: str) -> bool:
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-")
    return value[0].isalnum() and all(character in allowed for character in value)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("registry timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _parse_time(value: str) -> datetime:
    timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("registry timestamp must be timezone-aware")
    return timestamp.astimezone(timezone.utc)


__all__ = [
    "ArtifactRecord",
    "EXPERIMENT_REGISTRY_SCHEMA",
    "ExperimentRecord",
    "ExperimentRegistry",
    "ExperimentRegistryConflict",
    "ExperimentRegistryError",
    "LineageEdge",
    "PROTECTED_EVALUATION_ACTION",
    "PROTECTED_EVALUATION_ALLOWED_ROLES",
    "PROTECTED_EVALUATION_SCOPE",
    "ProtectedEvaluationConsumptionRecord",
    "RegisteredExperimentComponent",
    "StageArtifactPublication",
    "verify_experiment_registry_connection",
]
