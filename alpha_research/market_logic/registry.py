from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from types import MappingProxyType, TracebackType
from typing import Iterator, Mapping, Sequence, TypeVar

from alpha_research.core.hashing import canonical_json_bytes, hash_json, require_sha256
from alpha_research.market_logic.evidence import (
    CostResilienceCategory,
    CoverageCategory,
    EvidencePartition,
    EvidenceReasonCode,
    EvidenceVerdict,
    EvidenceVisibility,
    LogicEvidenceSummary,
    StabilityCategory,
)


MARKET_LOGIC_REGISTRY_SCHEMA = "market-logic-registry/v1"
LOGIC_FACTOR_BINDING_SCHEMA = "logic-factor-binding/v1"
MARKET_LOGIC_EVENT_SCHEMA = "market-logic-event/v1"

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
_EnumT = TypeVar("_EnumT", bound=Enum)


class MarketLogicRegistryError(RuntimeError):
    pass


class MarketLogicRegistryConflict(MarketLogicRegistryError):
    pass


class MarketLogicRegistryIntegrityError(MarketLogicRegistryError):
    pass


@dataclass(frozen=True, slots=True)
class LogicFactorBinding:
    """Content-addressed link from a logic to its implemented experiment."""

    logic_hash: str
    candidate_hash: str
    factor_hash: str
    experiment_hash: str
    schema_version: str = LOGIC_FACTOR_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != LOGIC_FACTOR_BINDING_SCHEMA:
            raise ValueError("unsupported LogicFactorBinding schema")
        for name in (
            "logic_hash",
            "candidate_hash",
            "factor_hash",
            "experiment_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"logic binding {name}")

    @property
    def content_hash(self) -> str:
        digest: str = hash_json(self.to_dict())
        return digest

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "logic_hash": self.logic_hash,
            "candidate_hash": self.candidate_hash,
            "factor_hash": self.factor_hash,
            "experiment_hash": self.experiment_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LogicFactorBinding":
        expected = {
            "schema_version",
            "logic_hash",
            "candidate_hash",
            "factor_hash",
            "experiment_hash",
        }
        if set(payload) != expected or not all(
            isinstance(payload[name], str) for name in expected
        ):
            raise ValueError("logic-factor binding payload fields differ")
        return cls(**{name: str(payload[name]) for name in expected})


@dataclass(frozen=True, slots=True)
class LogicRegistration:
    """Exact immutable descriptor accepted by atomic logic batch registration."""

    logic_hash: str
    logic_id: str
    version: str
    metadata: Mapping[str, object]
    parent_logic_hash: str | None = None

    def __post_init__(self) -> None:
        digest = require_sha256(self.logic_hash, name="logic_hash")
        _require_identifier(self.logic_id, name="logic_id")
        if not _VERSION.fullmatch(self.version):
            raise ValueError("logic version is invalid")
        parent = (
            require_sha256(self.parent_logic_hash, name="parent_logic_hash")
            if self.parent_logic_hash is not None
            else None
        )
        if parent == digest:
            raise ValueError("a logic version cannot be its own parent")
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_validated_metadata(self.metadata)),
        )


@dataclass(frozen=True, slots=True)
class LogicVersionRecord:
    logic_hash: str
    logic_id: str
    version: str
    parent_logic_hash: str | None
    metadata_hash: str
    metadata: Mapping[str, object]
    registered_at: str


@dataclass(frozen=True, slots=True)
class LogicBindingRecord:
    binding_hash: str
    binding: LogicFactorBinding
    registered_at: str


@dataclass(frozen=True, slots=True)
class LogicEvidenceRecord:
    evidence_hash: str
    summary: LogicEvidenceSummary
    recorded_at: str


@dataclass(frozen=True, slots=True)
class RetrievalLogicReference:
    """Logic identity exposed to adaptive memory; arbitrary metadata is omitted."""

    logic_hash: str
    logic_id: str
    version: str
    parent_logic_hash: str | None


@dataclass(frozen=True, slots=True)
class LogicMemoryHit:
    """A retrieval-safe logic/evidence record for the adaptive loop.

    ``LogicVersionRecord.metadata`` is deliberately not part of this type.  Logic
    metadata remains available in explicit registry/audit APIs but cannot be used
    as a covert channel for exact metrics, dates, securities, or teacher results.
    """

    logic: RetrievalLogicReference
    evidence: LogicEvidenceSummary
    bindings: tuple[LogicFactorBinding, ...]


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """Deterministic categorical lookup; embeddings are intentionally absent."""

    logic_ids: tuple[str, ...] = ()
    reason_codes: tuple[EvidenceReasonCode | str, ...] = ()
    verdicts: tuple[EvidenceVerdict | str, ...] = ()
    directional_stability: tuple[StabilityCategory | str, ...] = ()
    regime_stability: tuple[StabilityCategory | str, ...] = ()
    cost_resilience: tuple[CostResilienceCategory | str, ...] = ()
    coverage: tuple[CoverageCategory | str, ...] = ()
    limit: int = 100

    def __post_init__(self) -> None:
        logic_ids = tuple(sorted(set(self.logic_ids)))
        if any(not _IDENTIFIER.fullmatch(item) for item in logic_ids):
            raise ValueError("retrieval logic_ids contain an invalid identifier")
        reason_codes = _sorted_enums(EvidenceReasonCode, self.reason_codes)
        verdicts = _sorted_enums(EvidenceVerdict, self.verdicts)
        directional = _sorted_enums(StabilityCategory, self.directional_stability)
        regime = _sorted_enums(StabilityCategory, self.regime_stability)
        cost = _sorted_enums(CostResilienceCategory, self.cost_resilience)
        coverage = _sorted_enums(CoverageCategory, self.coverage)
        if (
            not isinstance(self.limit, int)
            or isinstance(self.limit, bool)
            or not 1 <= self.limit <= 1000
        ):
            raise ValueError("retrieval limit must be an integer in [1,1000]")
        object.__setattr__(self, "logic_ids", logic_ids)
        object.__setattr__(self, "reason_codes", reason_codes)
        object.__setattr__(self, "verdicts", verdicts)
        object.__setattr__(self, "directional_stability", directional)
        object.__setattr__(self, "regime_stability", regime)
        object.__setattr__(self, "cost_resilience", cost)
        object.__setattr__(self, "coverage", coverage)

    def matches(self, logic_id: str, summary: LogicEvidenceSummary) -> bool:
        if self.logic_ids and logic_id not in self.logic_ids:
            return False
        requested_reasons = {EvidenceReasonCode(item) for item in self.reason_codes}
        observed_reasons = {EvidenceReasonCode(item) for item in summary.reason_codes}
        if requested_reasons and not requested_reasons.issubset(observed_reasons):
            return False
        if self.verdicts and EvidenceVerdict(summary.verdict) not in self.verdicts:
            return False
        if (
            self.directional_stability
            and StabilityCategory(summary.directional_stability)
            not in self.directional_stability
        ):
            return False
        if (
            self.regime_stability
            and StabilityCategory(summary.regime_stability) not in self.regime_stability
        ):
            return False
        if (
            self.cost_resilience
            and CostResilienceCategory(summary.cost_resilience)
            not in self.cost_resilience
        ):
            return False
        if self.coverage and CoverageCategory(summary.coverage) not in self.coverage:
            return False
        return True


@dataclass(frozen=True, slots=True)
class RegistryIntegrityReport:
    sqlite_integrity: str
    logic_count: int
    binding_count: int
    evidence_count: int
    event_count: int
    last_event_hash: str | None


class MarketLogicRegistry:
    """Append-only SQLite sidecar for market logic and research memory.

    It accepts only externally content-addressed logic hashes plus compact
    metadata, so it has no dependency on a particular ``MarketLogicSpec`` class.
    All adaptive retrieval is hard-filtered to adaptive-validation evidence.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self.connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._initialize_schema()

    def __enter__(self) -> "MarketLogicRegistry":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self.connection.in_transaction:
                raise MarketLogicRegistryError(
                    "nested registry transactions are forbidden"
                )
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                yield self.connection
                self.connection.commit()
            except BaseException:
                self.connection.rollback()
                raise

    @contextmanager
    def _read_transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self.connection.in_transaction:
                raise MarketLogicRegistryError(
                    "nested registry transactions are forbidden"
                )
            try:
                self.connection.execute("BEGIN")
                yield self.connection
                self.connection.commit()
            except BaseException:
                self.connection.rollback()
                raise

    def _initialize_schema(self) -> None:
        with self._lock:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS registry_metadata(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS logic_versions(
                    logic_hash TEXT PRIMARY KEY,
                    logic_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    metadata_hash TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    registered_at TEXT NOT NULL,
                    UNIQUE(logic_id,version)
                );
                CREATE TABLE IF NOT EXISTS logic_lineage(
                    child_logic_hash TEXT PRIMARY KEY
                        REFERENCES logic_versions(logic_hash),
                    parent_logic_hash TEXT NOT NULL
                        REFERENCES logic_versions(logic_hash),
                    relationship TEXT NOT NULL CHECK(relationship='version_parent'),
                    recorded_at TEXT NOT NULL,
                    CHECK(child_logic_hash<>parent_logic_hash)
                );
                CREATE TABLE IF NOT EXISTS logic_factor_bindings(
                    binding_hash TEXT PRIMARY KEY,
                    logic_hash TEXT NOT NULL REFERENCES logic_versions(logic_hash),
                    candidate_hash TEXT NOT NULL,
                    factor_hash TEXT NOT NULL,
                    experiment_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    registered_at TEXT NOT NULL,
                    UNIQUE(logic_hash,candidate_hash,factor_hash,experiment_hash)
                );
                CREATE TABLE IF NOT EXISTS logic_evidence(
                    evidence_hash TEXT PRIMARY KEY,
                    logic_hash TEXT NOT NULL REFERENCES logic_versions(logic_hash),
                    experiment_hash TEXT NOT NULL,
                    evidence_artifact_hash TEXT NOT NULL,
                    source_partition TEXT NOT NULL CHECK(source_partition IN (
                        'adaptive_validation','test','holdout','teacher'
                    )),
                    visibility TEXT NOT NULL CHECK(visibility IN (
                        'adaptive_validation','audit_only'
                    )),
                    payload_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    UNIQUE(
                        logic_hash,experiment_hash,evidence_artifact_hash,
                        source_partition
                    ),
                    CHECK(
                        visibility='audit_only'
                        OR source_partition='adaptive_validation'
                    )
                );
                CREATE TABLE IF NOT EXISTS memory_events(
                    sequence INTEGER PRIMARY KEY,
                    event_type TEXT NOT NULL CHECK(event_type IN (
                        'logic_registered','binding_registered','evidence_registered'
                    )),
                    subject_hash TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    previous_event_hash TEXT,
                    event_hash TEXT NOT NULL UNIQUE
                );
                CREATE INDEX IF NOT EXISTS idx_logic_lineage_parent
                    ON logic_lineage(parent_logic_hash);
                CREATE INDEX IF NOT EXISTS idx_logic_binding_experiment
                    ON logic_factor_bindings(logic_hash,experiment_hash);
                CREATE INDEX IF NOT EXISTS idx_logic_evidence_retrieval
                    ON logic_evidence(visibility,source_partition,logic_hash);

                CREATE TRIGGER IF NOT EXISTS no_update_logic_versions
                BEFORE UPDATE ON logic_versions BEGIN
                    SELECT RAISE(ABORT,'market logic registry is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS no_delete_logic_versions
                BEFORE DELETE ON logic_versions BEGIN
                    SELECT RAISE(ABORT,'market logic registry is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS no_update_logic_lineage
                BEFORE UPDATE ON logic_lineage BEGIN
                    SELECT RAISE(ABORT,'market logic registry is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS no_delete_logic_lineage
                BEFORE DELETE ON logic_lineage BEGIN
                    SELECT RAISE(ABORT,'market logic registry is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS no_update_logic_bindings
                BEFORE UPDATE ON logic_factor_bindings BEGIN
                    SELECT RAISE(ABORT,'market logic registry is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS no_delete_logic_bindings
                BEFORE DELETE ON logic_factor_bindings BEGIN
                    SELECT RAISE(ABORT,'market logic registry is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS no_update_logic_evidence
                BEFORE UPDATE ON logic_evidence BEGIN
                    SELECT RAISE(ABORT,'market logic registry is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS no_delete_logic_evidence
                BEFORE DELETE ON logic_evidence BEGIN
                    SELECT RAISE(ABORT,'market logic registry is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS no_update_memory_events
                BEFORE UPDATE ON memory_events BEGIN
                    SELECT RAISE(ABORT,'market logic registry is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS no_delete_memory_events
                BEFORE DELETE ON memory_events BEGIN
                    SELECT RAISE(ABORT,'market logic registry is append-only');
                END;
                """
            )
        with self._transaction() as connection:
            existing = dict(
                connection.execute(
                    "SELECT key,value FROM registry_metadata ORDER BY key"
                ).fetchall()
            )
            expected = {"schema_version": MARKET_LOGIC_REGISTRY_SCHEMA}
            if not existing:
                connection.execute(
                    "INSERT INTO registry_metadata(key,value) VALUES(?,?)",
                    ("schema_version", MARKET_LOGIC_REGISTRY_SCHEMA),
                )
            elif existing != expected:
                raise MarketLogicRegistryConflict(
                    "market logic registry schema metadata differs"
                )

    def register_logic(
        self,
        logic_hash: str,
        *,
        logic_id: str,
        version: str,
        metadata: Mapping[str, object],
        parent_logic_hash: str | None = None,
        registered_at: datetime | None = None,
    ) -> LogicVersionRecord:
        digest = require_sha256(logic_hash, name="logic_hash")
        _require_identifier(logic_id, name="logic_id")
        if not _VERSION.fullmatch(version):
            raise ValueError("logic version is invalid")
        parent = (
            require_sha256(parent_logic_hash, name="parent_logic_hash")
            if parent_logic_hash is not None
            else None
        )
        if parent == digest:
            raise ValueError("a logic version cannot be its own parent")
        metadata_body = _validated_metadata(metadata)
        metadata_json = _json_text(metadata_body)
        metadata_hash = hash_json(metadata_body)
        timestamp = _iso(registered_at or _utc_now())

        with self._transaction() as connection:
            return self._register_logic_locked(
                connection,
                digest=digest,
                logic_id=logic_id,
                version=version,
                parent=parent,
                metadata_json=metadata_json,
                metadata_hash=metadata_hash,
                timestamp=timestamp,
            )

    def register_binding(
        self,
        binding: LogicFactorBinding,
        *,
        registered_at: datetime | None = None,
    ) -> LogicBindingRecord:
        binding_hash = binding.content_hash
        payload_json = _json_text(binding.to_dict())
        timestamp = _iso(registered_at or _utc_now())
        with self._transaction() as connection:
            return self._register_binding_locked(
                connection,
                binding=binding,
                binding_hash=binding_hash,
                payload_json=payload_json,
                timestamp=timestamp,
            )

    def register_logic_batch(
        self,
        registrations: tuple[LogicRegistration, ...],
        bindings: tuple[LogicFactorBinding, ...],
        *,
        registered_at: datetime | None = None,
    ) -> tuple[tuple[LogicVersionRecord, ...], tuple[LogicBindingRecord, ...]]:
        """Atomically register an ordered logic/binding batch in this registry.

        In-batch parents must precede children.  This transaction is local to
        the market-logic SQLite database; it does not create approval,
        publication, execution, or cross-registry authority.
        """

        if type(registrations) is not tuple or type(bindings) is not tuple:
            raise TypeError("logic registrations and bindings must be exact tuples")
        if not registrations:
            raise ValueError("logic registration batch must not be empty")
        if any(type(item) is not LogicRegistration for item in registrations):
            raise TypeError("logic batch requires exact LogicRegistration values")
        if any(type(item) is not LogicFactorBinding for item in bindings):
            raise TypeError("logic batch requires exact LogicFactorBinding values")
        logic_hashes = tuple(item.logic_hash for item in registrations)
        identities = tuple((item.logic_id, item.version) for item in registrations)
        binding_hashes = tuple(item.content_hash for item in bindings)
        if len(logic_hashes) != len(set(logic_hashes)):
            raise ValueError("logic batch contains duplicate logic hashes")
        if len(identities) != len(set(identities)):
            raise ValueError("logic batch contains duplicate logic id/version pairs")
        if len(binding_hashes) != len(set(binding_hashes)):
            raise ValueError("logic batch contains duplicate binding hashes")
        timestamp = _iso(registered_at or _utc_now())
        with self._transaction() as connection:
            logic_records = tuple(
                self._register_logic_locked(
                    connection,
                    digest=item.logic_hash,
                    logic_id=item.logic_id,
                    version=item.version,
                    parent=item.parent_logic_hash,
                    metadata_json=_json_text(item.metadata),
                    metadata_hash=hash_json(item.metadata),
                    timestamp=timestamp,
                )
                for item in registrations
            )
            binding_records = tuple(
                self._register_binding_locked(
                    connection,
                    binding=item,
                    binding_hash=item.content_hash,
                    payload_json=_json_text(item.to_dict()),
                    timestamp=timestamp,
                )
                for item in bindings
            )
            return logic_records, binding_records

    def register_evidence(
        self,
        summary: LogicEvidenceSummary,
        *,
        recorded_at: datetime | None = None,
    ) -> LogicEvidenceRecord:
        evidence_hash = summary.content_hash
        payload_json = _json_text(summary.to_dict())
        timestamp = _iso(recorded_at or _utc_now())
        with self._transaction() as connection:
            self._require_logic_locked(connection, summary.logic_hash)
            binding_exists = connection.execute(
                """SELECT 1 FROM logic_factor_bindings
                   WHERE logic_hash=? AND experiment_hash=? LIMIT 1""",
                (summary.logic_hash, summary.experiment_hash),
            ).fetchone()
            if binding_exists is None:
                raise MarketLogicRegistryConflict(
                    "logic evidence lacks a registered experiment binding"
                )
            existing = connection.execute(
                "SELECT * FROM logic_evidence WHERE evidence_hash=?", (evidence_hash,)
            ).fetchone()
            if existing is not None:
                if existing["payload_json"] != payload_json:
                    raise MarketLogicRegistryConflict(
                        "immutable evidence hash has another payload"
                    )
                return self._evidence_record(existing)
            natural = connection.execute(
                """SELECT evidence_hash FROM logic_evidence
                   WHERE logic_hash=? AND experiment_hash=?
                     AND evidence_artifact_hash=? AND source_partition=?""",
                (
                    summary.logic_hash,
                    summary.experiment_hash,
                    summary.evidence_artifact_hash,
                    EvidencePartition(summary.source_partition).value,
                ),
            ).fetchone()
            if natural is not None:
                raise MarketLogicRegistryConflict(
                    "evidence source identity already binds another immutable summary"
                )
            connection.execute(
                """INSERT INTO logic_evidence(
                    evidence_hash,logic_hash,experiment_hash,evidence_artifact_hash,
                    source_partition,visibility,payload_json,recorded_at
                ) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    evidence_hash,
                    summary.logic_hash,
                    summary.experiment_hash,
                    summary.evidence_artifact_hash,
                    EvidencePartition(summary.source_partition).value,
                    EvidenceVisibility(summary.visibility).value,
                    payload_json,
                    timestamp,
                ),
            )
            self._append_event_locked(
                connection,
                event_type="evidence_registered",
                subject_hash=evidence_hash,
                occurred_at=timestamp,
                payload={
                    "evidence_hash": evidence_hash,
                    "summary": summary.to_dict(),
                },
            )
            row = connection.execute(
                "SELECT * FROM logic_evidence WHERE evidence_hash=?", (evidence_hash,)
            ).fetchone()
            if row is None:  # pragma: no cover - guarded by the transaction
                raise MarketLogicRegistryIntegrityError("inserted evidence disappeared")
            return self._evidence_record(row)

    def _register_logic_locked(
        self,
        connection: sqlite3.Connection,
        *,
        digest: str,
        logic_id: str,
        version: str,
        parent: str | None,
        metadata_json: str,
        metadata_hash: str,
        timestamp: str,
    ) -> LogicVersionRecord:
        by_hash = connection.execute(
            "SELECT * FROM logic_versions WHERE logic_hash=?", (digest,)
        ).fetchone()
        if by_hash is not None:
            observed_parent = self._parent_locked(connection, digest)
            if (
                by_hash["logic_id"] != logic_id
                or by_hash["version"] != version
                or by_hash["metadata_hash"] != metadata_hash
                or by_hash["metadata_json"] != metadata_json
                or observed_parent != parent
            ):
                raise MarketLogicRegistryConflict(
                    "immutable logic hash already has another descriptor"
                )
            return self._logic_record_locked(connection, digest)

        by_version = connection.execute(
            "SELECT logic_hash FROM logic_versions WHERE logic_id=? AND version=?",
            (logic_id, version),
        ).fetchone()
        if by_version is not None:
            raise MarketLogicRegistryConflict(
                "logic id/version already binds another immutable hash"
            )

        prior_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM logic_versions WHERE logic_id=?", (logic_id,)
            ).fetchone()[0]
        )
        if prior_count and parent is None:
            raise MarketLogicRegistryConflict(
                "a subsequent logic version must bind a registered parent"
            )
        if parent is not None:
            parent_row = connection.execute(
                "SELECT logic_id FROM logic_versions WHERE logic_hash=?", (parent,)
            ).fetchone()
            if parent_row is None:
                raise MarketLogicRegistryConflict("logic parent is not registered")
            if parent_row["logic_id"] != logic_id:
                raise MarketLogicRegistryConflict(
                    "logic version parent belongs to another logic id"
                )

        connection.execute(
            """INSERT INTO logic_versions(
                logic_hash,logic_id,version,metadata_hash,metadata_json,registered_at
            ) VALUES(?,?,?,?,?,?)""",
            (digest, logic_id, version, metadata_hash, metadata_json, timestamp),
        )
        if parent is not None:
            connection.execute(
                """INSERT INTO logic_lineage(
                    child_logic_hash,parent_logic_hash,relationship,recorded_at
                ) VALUES(?,?,?,?)""",
                (digest, parent, "version_parent", timestamp),
            )
        self._append_event_locked(
            connection,
            event_type="logic_registered",
            subject_hash=digest,
            occurred_at=timestamp,
            payload={
                "logic_hash": digest,
                "logic_id": logic_id,
                "version": version,
                "parent_logic_hash": parent,
                "metadata_hash": metadata_hash,
            },
        )
        return self._logic_record_locked(connection, digest)

    def _register_binding_locked(
        self,
        connection: sqlite3.Connection,
        *,
        binding: LogicFactorBinding,
        binding_hash: str,
        payload_json: str,
        timestamp: str,
    ) -> LogicBindingRecord:
        self._require_logic_locked(connection, binding.logic_hash)
        existing = connection.execute(
            "SELECT * FROM logic_factor_bindings WHERE binding_hash=?",
            (binding_hash,),
        ).fetchone()
        if existing is not None:
            if existing["payload_json"] != payload_json:
                raise MarketLogicRegistryConflict(
                    "immutable binding hash has another payload"
                )
            return self._binding_record(existing)
        natural = connection.execute(
            """SELECT binding_hash FROM logic_factor_bindings
               WHERE logic_hash=? AND candidate_hash=? AND factor_hash=?
                 AND experiment_hash=?""",
            (
                binding.logic_hash,
                binding.candidate_hash,
                binding.factor_hash,
                binding.experiment_hash,
            ),
        ).fetchone()
        if natural is not None:
            raise MarketLogicRegistryConflict(
                "logic/factor/experiment binding has another immutable hash"
            )
        connection.execute(
            """INSERT INTO logic_factor_bindings(
                binding_hash,logic_hash,candidate_hash,factor_hash,
                experiment_hash,payload_json,registered_at
            ) VALUES(?,?,?,?,?,?,?)""",
            (
                binding_hash,
                binding.logic_hash,
                binding.candidate_hash,
                binding.factor_hash,
                binding.experiment_hash,
                payload_json,
                timestamp,
            ),
        )
        self._append_event_locked(
            connection,
            event_type="binding_registered",
            subject_hash=binding_hash,
            occurred_at=timestamp,
            payload=binding.to_dict(),
        )
        row = connection.execute(
            "SELECT * FROM logic_factor_bindings WHERE binding_hash=?",
            (binding_hash,),
        ).fetchone()
        if row is None:  # pragma: no cover - guarded by the transaction
            raise MarketLogicRegistryIntegrityError("inserted binding disappeared")
        return self._binding_record(row)

    def get_logic(self, logic_hash: str) -> LogicVersionRecord:
        digest = require_sha256(logic_hash, name="logic_hash")
        with self._read_transaction() as connection:
            return self._logic_record_locked(connection, digest)

    def list_logic_versions(
        self, *, logic_id: str | None = None
    ) -> tuple[LogicVersionRecord, ...]:
        if logic_id is not None:
            _require_identifier(logic_id, name="logic_id")
        with self._read_transaction() as connection:
            if logic_id is None:
                rows = connection.execute(
                    """SELECT logic_hash FROM logic_versions
                       ORDER BY logic_id,version,logic_hash"""
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT logic_hash FROM logic_versions WHERE logic_id=?
                       ORDER BY version,logic_hash""",
                    (logic_id,),
                ).fetchall()
            return tuple(
                self._logic_record_locked(connection, row["logic_hash"]) for row in rows
            )

    def lineage(self, logic_hash: str) -> tuple[str, ...]:
        """Return nearest-parent-first ancestry, failing closed on a cycle."""

        current = require_sha256(logic_hash, name="logic_hash")
        with self._read_transaction() as connection:
            self._require_logic_locked(connection, current)
            seen = {current}
            parents: list[str] = []
            while (parent := self._parent_locked(connection, current)) is not None:
                if parent in seen:
                    raise MarketLogicRegistryIntegrityError(
                        "logic lineage contains a cycle"
                    )
                seen.add(parent)
                parents.append(parent)
                current = parent
            return tuple(parents)

    def read_evidence_for_audit(self, evidence_hash: str) -> LogicEvidenceRecord:
        """Explicit audit access; this method is never used by adaptive retrieval."""

        digest = require_sha256(evidence_hash, name="evidence_hash")
        with self._read_transaction() as connection:
            row = connection.execute(
                "SELECT * FROM logic_evidence WHERE evidence_hash=?", (digest,)
            ).fetchone()
            if row is None:
                raise KeyError(f"logic evidence is not registered:{digest}")
            return self._evidence_record(row)

    def read_binding_for_audit(self, binding_hash: str) -> LogicBindingRecord:
        """Read one immutable logic/factor binding without adaptive evidence.

        Parent-lineage verification needs the exact registered binding, rather
        than a retrieval result selected through evidence.  Keeping this as an
        explicit audit API also prevents the lineage layer from inspecting
        arbitrary logic metadata or validation metrics.
        """

        digest = require_sha256(binding_hash, name="binding_hash")
        with self._read_transaction() as connection:
            row = connection.execute(
                "SELECT * FROM logic_factor_bindings WHERE binding_hash=?",
                (digest,),
            ).fetchone()
            if row is None:
                raise KeyError(f"logic-factor binding is not registered:{digest}")
            return self._binding_record(row)

    def retrieve(
        self, query: RetrievalQuery | None = None
    ) -> tuple[LogicMemoryHit, ...]:
        """Retrieve adaptive-validation memory with deterministic exact filters.

        SQL hard-codes both visibility and source partition.  Audit-only, test,
        holdout, and teacher summaries therefore cannot enter an agent prompt,
        including when the caller supplies an empty query.
        """

        normalized = query or RetrievalQuery()
        with self._read_transaction() as connection:
            rows = connection.execute(
                """SELECT e.*,l.logic_id,l.version
                   FROM logic_evidence AS e
                   JOIN logic_versions AS l ON l.logic_hash=e.logic_hash
                   WHERE e.visibility='adaptive_validation'
                     AND e.source_partition='adaptive_validation'
                   ORDER BY l.logic_id,l.version,l.logic_hash,e.evidence_hash"""
            ).fetchall()
            hits: list[LogicMemoryHit] = []
            for row in rows:
                summary = LogicEvidenceSummary.from_dict(
                    _json_object(row["payload_json"])
                )
                if not normalized.matches(row["logic_id"], summary):
                    continue
                binding_rows = connection.execute(
                    """SELECT * FROM logic_factor_bindings
                       WHERE logic_hash=? AND experiment_hash=?
                       ORDER BY binding_hash""",
                    (summary.logic_hash, summary.experiment_hash),
                ).fetchall()
                if not binding_rows:
                    raise MarketLogicRegistryIntegrityError(
                        "retrieval-visible evidence has no experiment binding"
                    )
                hits.append(
                    LogicMemoryHit(
                        logic=RetrievalLogicReference(
                            logic_hash=summary.logic_hash,
                            logic_id=row["logic_id"],
                            version=row["version"],
                            parent_logic_hash=self._parent_locked(
                                connection, summary.logic_hash
                            ),
                        ),
                        evidence=summary,
                        bindings=tuple(
                            self._binding_record(item).binding for item in binding_rows
                        ),
                    )
                )
                if len(hits) >= normalized.limit:
                    break
            return tuple(hits)

    def verify_integrity(self) -> RegistryIntegrityReport:
        with self._read_transaction() as connection:
            return self._verify_integrity_locked(connection)

    def snapshot(self) -> dict[str, object]:
        """Return a deterministic logical snapshot; no wall-clock field is added."""

        with self._read_transaction() as connection:
            report = self._verify_integrity_locked(connection)
            logic_rows = connection.execute(
                "SELECT logic_hash FROM logic_versions ORDER BY logic_hash"
            ).fetchall()
            binding_rows = connection.execute(
                "SELECT * FROM logic_factor_bindings ORDER BY binding_hash"
            ).fetchall()
            evidence_rows = connection.execute(
                "SELECT * FROM logic_evidence ORDER BY evidence_hash"
            ).fetchall()
            event_rows = connection.execute(
                "SELECT * FROM memory_events ORDER BY sequence"
            ).fetchall()
            logics = []
            for row in logic_rows:
                record = self._logic_record_locked(connection, row["logic_hash"])
                logics.append(_logic_record_dict(record))
            return {
                "schema_version": MARKET_LOGIC_REGISTRY_SCHEMA,
                "logic_versions": logics,
                "bindings": [
                    {
                        "binding_hash": row["binding_hash"],
                        "binding": self._binding_record(row).binding.to_dict(),
                        "registered_at": row["registered_at"],
                    }
                    for row in binding_rows
                ],
                "evidence": [
                    {
                        "evidence_hash": row["evidence_hash"],
                        "summary": self._evidence_record(row).summary.to_dict(),
                        "recorded_at": row["recorded_at"],
                    }
                    for row in evidence_rows
                ],
                "events": [_event_dict(row) for row in event_rows],
                "integrity": {
                    "sqlite_integrity": report.sqlite_integrity,
                    "logic_count": report.logic_count,
                    "binding_count": report.binding_count,
                    "evidence_count": report.evidence_count,
                    "event_count": report.event_count,
                    "last_event_hash": report.last_event_hash,
                },
            }

    def snapshot_hash(self) -> str:
        digest: str = hash_json(self.snapshot())
        return digest

    def _verify_integrity_locked(
        self, connection: sqlite3.Connection
    ) -> RegistryIntegrityReport:
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check != "ok":
            raise MarketLogicRegistryIntegrityError(
                f"sqlite integrity check failed:{quick_check}"
            )
        metadata = dict(
            connection.execute(
                "SELECT key,value FROM registry_metadata ORDER BY key"
            ).fetchall()
        )
        if metadata != {"schema_version": MARKET_LOGIC_REGISTRY_SCHEMA}:
            raise MarketLogicRegistryIntegrityError("registry schema metadata differs")

        logic_rows = connection.execute(
            "SELECT * FROM logic_versions ORDER BY logic_hash"
        ).fetchall()
        logic_ids: dict[str, str] = {}
        root_counts: dict[str, int] = {}
        for row in logic_rows:
            require_sha256(row["logic_hash"], name="stored logic_hash")
            body = _json_object(row["metadata_json"])
            if hash_json(body) != row["metadata_hash"]:
                raise MarketLogicRegistryIntegrityError("logic metadata hash differs")
            logic_ids[row["logic_hash"]] = row["logic_id"]
            parent = self._parent_locked(connection, row["logic_hash"])
            if parent is None:
                root_counts[row["logic_id"]] = root_counts.get(row["logic_id"], 0) + 1
            elif logic_ids.get(parent) is None:
                parent_row = connection.execute(
                    "SELECT logic_id FROM logic_versions WHERE logic_hash=?", (parent,)
                ).fetchone()
                if parent_row is None or parent_row["logic_id"] != row["logic_id"]:
                    raise MarketLogicRegistryIntegrityError(
                        "logic lineage crosses identities or has a missing parent"
                    )
        if any(count != 1 for count in root_counts.values()):
            raise MarketLogicRegistryIntegrityError(
                "each logic identity must have exactly one root version"
            )
        distinct_logic_ids = {row["logic_id"] for row in logic_rows}
        if set(root_counts) != distinct_logic_ids:
            raise MarketLogicRegistryIntegrityError(
                "logic identity lacks a root version"
            )

        binding_rows = connection.execute(
            "SELECT * FROM logic_factor_bindings ORDER BY binding_hash"
        ).fetchall()
        for row in binding_rows:
            binding_record = self._binding_record(row)
            if binding_record.binding_hash != binding_record.binding.content_hash:
                raise MarketLogicRegistryIntegrityError("binding hash differs")

        evidence_rows = connection.execute(
            "SELECT * FROM logic_evidence ORDER BY evidence_hash"
        ).fetchall()
        for row in evidence_rows:
            evidence_record = self._evidence_record(row)
            if evidence_record.evidence_hash != evidence_record.summary.content_hash:
                raise MarketLogicRegistryIntegrityError("evidence hash differs")
            if (
                row["logic_hash"] != evidence_record.summary.logic_hash
                or row["experiment_hash"] != evidence_record.summary.experiment_hash
                or row["evidence_artifact_hash"]
                != evidence_record.summary.evidence_artifact_hash
                or row["source_partition"]
                != EvidencePartition(evidence_record.summary.source_partition).value
                or row["visibility"]
                != EvidenceVisibility(evidence_record.summary.visibility).value
            ):
                raise MarketLogicRegistryIntegrityError(
                    "evidence indexed fields differ from sealed payload"
                )
            linked = connection.execute(
                """SELECT 1 FROM logic_factor_bindings
                   WHERE logic_hash=? AND experiment_hash=? LIMIT 1""",
                (
                    evidence_record.summary.logic_hash,
                    evidence_record.summary.experiment_hash,
                ),
            ).fetchone()
            if linked is None:
                raise MarketLogicRegistryIntegrityError(
                    "evidence experiment binding is missing"
                )

        event_rows = connection.execute(
            "SELECT * FROM memory_events ORDER BY sequence"
        ).fetchall()
        previous: str | None = None
        subjects: dict[str, set[str]] = {
            "logic_registered": set(),
            "binding_registered": set(),
            "evidence_registered": set(),
        }
        for expected_sequence, row in enumerate(event_rows, start=1):
            if row["sequence"] != expected_sequence:
                raise MarketLogicRegistryIntegrityError("event sequence contains a gap")
            payload = _json_object(row["payload_json"])
            if hash_json(payload) != row["payload_hash"]:
                raise MarketLogicRegistryIntegrityError("event payload hash differs")
            if row["previous_event_hash"] != previous:
                raise MarketLogicRegistryIntegrityError("event hash chain is broken")
            expected_hash = _event_hash(
                sequence=expected_sequence,
                event_type=row["event_type"],
                subject_hash=row["subject_hash"],
                occurred_at=row["occurred_at"],
                payload_hash=row["payload_hash"],
                previous_event_hash=previous,
            )
            if row["event_hash"] != expected_hash:
                raise MarketLogicRegistryIntegrityError("event hash differs")
            subjects[row["event_type"]].add(row["subject_hash"])
            previous = row["event_hash"]

        expected_subjects = {
            "logic_registered": {row["logic_hash"] for row in logic_rows},
            "binding_registered": {row["binding_hash"] for row in binding_rows},
            "evidence_registered": {row["evidence_hash"] for row in evidence_rows},
        }
        if subjects != expected_subjects or len(event_rows) != sum(
            len(items) for items in expected_subjects.values()
        ):
            raise MarketLogicRegistryIntegrityError(
                "event log and immutable records differ"
            )
        return RegistryIntegrityReport(
            sqlite_integrity=quick_check,
            logic_count=len(logic_rows),
            binding_count=len(binding_rows),
            evidence_count=len(evidence_rows),
            event_count=len(event_rows),
            last_event_hash=previous,
        )

    def _logic_record_locked(
        self, connection: sqlite3.Connection, logic_hash: str
    ) -> LogicVersionRecord:
        row = connection.execute(
            "SELECT * FROM logic_versions WHERE logic_hash=?", (logic_hash,)
        ).fetchone()
        if row is None:
            raise KeyError(f"market logic is not registered:{logic_hash}")
        metadata = MappingProxyType(_json_object(row["metadata_json"]))
        return LogicVersionRecord(
            logic_hash=row["logic_hash"],
            logic_id=row["logic_id"],
            version=row["version"],
            parent_logic_hash=self._parent_locked(connection, logic_hash),
            metadata_hash=row["metadata_hash"],
            metadata=metadata,
            registered_at=row["registered_at"],
        )

    @staticmethod
    def _parent_locked(connection: sqlite3.Connection, logic_hash: str) -> str | None:
        row = connection.execute(
            "SELECT parent_logic_hash FROM logic_lineage WHERE child_logic_hash=?",
            (logic_hash,),
        ).fetchone()
        return None if row is None else str(row["parent_logic_hash"])

    @staticmethod
    def _require_logic_locked(connection: sqlite3.Connection, logic_hash: str) -> None:
        if (
            connection.execute(
                "SELECT 1 FROM logic_versions WHERE logic_hash=?", (logic_hash,)
            ).fetchone()
            is None
        ):
            raise MarketLogicRegistryConflict(
                f"market logic is not registered:{logic_hash}"
            )

    @staticmethod
    def _binding_record(row: sqlite3.Row) -> LogicBindingRecord:
        binding = LogicFactorBinding.from_dict(_json_object(row["payload_json"]))
        return LogicBindingRecord(
            binding_hash=row["binding_hash"],
            binding=binding,
            registered_at=row["registered_at"],
        )

    @staticmethod
    def _evidence_record(row: sqlite3.Row) -> LogicEvidenceRecord:
        summary = LogicEvidenceSummary.from_dict(_json_object(row["payload_json"]))
        return LogicEvidenceRecord(
            evidence_hash=row["evidence_hash"],
            summary=summary,
            recorded_at=row["recorded_at"],
        )

    @staticmethod
    def _append_event_locked(
        connection: sqlite3.Connection,
        *,
        event_type: str,
        subject_hash: str,
        occurred_at: str,
        payload: Mapping[str, object],
    ) -> str:
        require_sha256(subject_hash, name="event subject_hash")
        last = connection.execute(
            "SELECT sequence,event_hash FROM memory_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        sequence = 1 if last is None else int(last["sequence"]) + 1
        previous = None if last is None else str(last["event_hash"])
        payload_json = _json_text(payload)
        payload_hash = hash_json(payload)
        event_hash = _event_hash(
            sequence=sequence,
            event_type=event_type,
            subject_hash=subject_hash,
            occurred_at=occurred_at,
            payload_hash=payload_hash,
            previous_event_hash=previous,
        )
        connection.execute(
            """INSERT INTO memory_events(
                sequence,event_type,subject_hash,occurred_at,payload_hash,
                payload_json,previous_event_hash,event_hash
            ) VALUES(?,?,?,?,?,?,?,?)""",
            (
                sequence,
                event_type,
                subject_hash,
                occurred_at,
                payload_hash,
                payload_json,
                previous,
                event_hash,
            ),
        )
        return event_hash


def _event_hash(
    *,
    sequence: int,
    event_type: str,
    subject_hash: str,
    occurred_at: str,
    payload_hash: str,
    previous_event_hash: str | None,
) -> str:
    digest: str = hash_json(
        {
            "schema_version": MARKET_LOGIC_EVENT_SCHEMA,
            "sequence": sequence,
            "event_type": event_type,
            "subject_hash": subject_hash,
            "occurred_at": occurred_at,
            "payload_hash": payload_hash,
            "previous_event_hash": previous_event_hash,
        }
    )
    return digest


def _logic_record_dict(record: LogicVersionRecord) -> dict[str, object]:
    return {
        "logic_hash": record.logic_hash,
        "logic_id": record.logic_id,
        "version": record.version,
        "parent_logic_hash": record.parent_logic_hash,
        "metadata_hash": record.metadata_hash,
        "metadata": dict(record.metadata),
        "registered_at": record.registered_at,
    }


def _event_dict(row: sqlite3.Row) -> dict[str, object]:
    return {
        "sequence": row["sequence"],
        "event_type": row["event_type"],
        "subject_hash": row["subject_hash"],
        "occurred_at": row["occurred_at"],
        "payload_hash": row["payload_hash"],
        "payload": _json_object(row["payload_json"]),
        "previous_event_hash": row["previous_event_hash"],
        "event_hash": row["event_hash"],
    }


def _validated_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise TypeError("logic metadata must be a mapping")
    body = dict(metadata)
    if not body:
        raise ValueError("logic metadata must not be empty")
    if any(not isinstance(key, str) or not _IDENTIFIER.fullmatch(key) for key in body):
        raise ValueError("logic metadata keys must be safe identifiers")
    encoded = canonical_json_bytes(body)
    if len(encoded) > 64 * 1024:
        raise ValueError("logic metadata exceeds the 64 KiB registry limit")
    decoded = json.loads(encoded.decode("utf-8"))
    if not isinstance(
        decoded, dict
    ):  # pragma: no cover - mapping canonicalizes to object
        raise ValueError("logic metadata must canonicalize to an object")
    return decoded


def _json_text(value: Mapping[str, object]) -> str:
    encoded: bytes = canonical_json_bytes(value)
    return encoded.decode("utf-8")


def _json_object(value: str) -> dict[str, object]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise MarketLogicRegistryIntegrityError(
            "registry JSON payload is not an object"
        )
    return parsed


def _require_identifier(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a safe lowercase identifier")
    return value


def _iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("registry timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sorted_enums(
    enum_type: type[_EnumT], values: Sequence[object]
) -> tuple[_EnumT, ...]:
    return tuple(
        sorted({enum_type(item) for item in values}, key=lambda item: item.value)
    )


__all__ = [
    "LOGIC_FACTOR_BINDING_SCHEMA",
    "MARKET_LOGIC_EVENT_SCHEMA",
    "MARKET_LOGIC_REGISTRY_SCHEMA",
    "LogicBindingRecord",
    "LogicEvidenceRecord",
    "LogicFactorBinding",
    "LogicMemoryHit",
    "LogicRegistration",
    "LogicVersionRecord",
    "MarketLogicRegistry",
    "MarketLogicRegistryConflict",
    "MarketLogicRegistryError",
    "MarketLogicRegistryIntegrityError",
    "RegistryIntegrityReport",
    "RetrievalLogicReference",
    "RetrievalQuery",
]
