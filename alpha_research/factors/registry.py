from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Iterator

import pandas as pd

from alpha_research.core.hashing import canonical_json_bytes, require_sha256
from alpha_research.factors.spec import FactorSpec


FACTOR_REGISTRY_SCHEMA = "factor-registry/v3"

_LIFECYCLE_TRANSITIONS = {
    "registered": frozenset({"validated", "rejected", "deprecated"}),
    "validated": frozenset({"deprecated"}),
    "rejected": frozenset({"deprecated"}),
    "deprecated": frozenset(),
}


class FactorRegistryConflict(RuntimeError):
    pass


class FactorRegistry:
    """Append-only factor definitions with semantic and lineage de-duplication."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self._initialize()

    def __enter__(self) -> "FactorRegistry":
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
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            yield self.connection
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS registry_metadata(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS factors(
                factor_hash TEXT PRIMARY KEY,
                factor_id TEXT NOT NULL,
                version TEXT NOT NULL,
                semantic_hash TEXT NOT NULL,
                definition_hash TEXT NOT NULL UNIQUE,
                registered_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                UNIQUE(factor_id, version)
            );
            CREATE TABLE IF NOT EXISTS factor_lineage(
                child_hash TEXT NOT NULL REFERENCES factors(factor_hash),
                parent_hash TEXT NOT NULL REFERENCES factors(factor_hash),
                PRIMARY KEY(child_hash, parent_hash)
            );
            CREATE TABLE IF NOT EXISTS factor_lifecycle_events(
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                factor_hash TEXT NOT NULL REFERENCES factors(factor_hash),
                state TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                evidence_hash TEXT,
                UNIQUE(factor_hash,state)
            );
            CREATE TABLE IF NOT EXISTS factor_correlations(
                left_hash TEXT NOT NULL REFERENCES factors(factor_hash),
                right_hash TEXT NOT NULL REFERENCES factors(factor_hash),
                evaluation_hash TEXT NOT NULL,
                correlation REAL NOT NULL,
                observation_count INTEGER NOT NULL,
                recorded_at TEXT NOT NULL,
                PRIMARY KEY(left_hash,right_hash,evaluation_hash)
            );
            """
        )
        row = self.connection.execute(
            "SELECT value FROM registry_metadata WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            self.connection.execute(
                "INSERT INTO registry_metadata(key,value) VALUES('schema_version',?)",
                (FACTOR_REGISTRY_SCHEMA,),
            )
            self.connection.commit()
        elif row["value"] != FACTOR_REGISTRY_SCHEMA:
            raise FactorRegistryConflict("factor registry schema version differs")

    def register(self, spec: FactorSpec, *, registered_at: object) -> str:
        if not isinstance(spec, FactorSpec):
            raise TypeError("factor registry requires a FactorSpec")
        timestamp = self._registered_at(registered_at)
        try:
            with self._transaction() as connection:
                return self._register_locked(connection, spec, timestamp.isoformat())
        except sqlite3.IntegrityError as exc:
            raise FactorRegistryConflict(
                "factor id/version or executable definition already exists"
            ) from exc

    def register_many(
        self,
        specs: tuple[FactorSpec, ...],
        *,
        registered_at: object,
    ) -> tuple[str, ...]:
        """Register one exact batch atomically inside this SQLite registry.

        The tuple order is authoritative.  A parent included in the same batch
        must therefore precede its child.  This method provides no cross-registry
        atomicity and does not grant execution or publication authority.
        """

        if type(specs) is not tuple:
            raise TypeError("factor batch must be an exact tuple")
        if not specs:
            raise ValueError("factor batch must not be empty")
        if any(type(spec) is not FactorSpec for spec in specs):
            raise TypeError("factor batch requires exact FactorSpec values")
        hashes = tuple(spec.content_hash for spec in specs)
        if len(hashes) != len(set(hashes)):
            raise ValueError("factor batch contains duplicate factor hashes")
        timestamp = self._registered_at(registered_at).isoformat()
        try:
            with self._transaction() as connection:
                return tuple(
                    self._register_locked(connection, spec, timestamp)
                    for spec in specs
                )
        except sqlite3.IntegrityError as exc:
            raise FactorRegistryConflict(
                "factor id/version or executable definition already exists"
            ) from exc

    @staticmethod
    def _registered_at(value: object) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            raise ValueError("factor registered_at must be timezone-aware")
        return timestamp

    @staticmethod
    def _register_locked(
        connection: sqlite3.Connection,
        spec: FactorSpec,
        registered_at: str,
    ) -> str:
        factor_hash: str = spec.content_hash
        payload = canonical_json_bytes(spec.to_dict()).decode("utf-8")
        existing = connection.execute(
            "SELECT payload_json FROM factors WHERE factor_hash=?",
            (factor_hash,),
        ).fetchone()
        if existing is not None:
            if existing["payload_json"] != payload:
                raise FactorRegistryConflict("immutable factor payload differs")
            return factor_hash
        for parent in spec.provenance.parent_factor_hashes:
            if (
                connection.execute(
                    "SELECT 1 FROM factors WHERE factor_hash=?", (parent,)
                ).fetchone()
                is None
            ):
                raise FactorRegistryConflict(
                    f"factor parent is not registered:{parent}"
                )
        connection.execute(
            """
            INSERT INTO factors(
                factor_hash,factor_id,version,semantic_hash,definition_hash,
                registered_at,payload_json
            ) VALUES(?,?,?,?,?,?,?)
            """,
            (
                factor_hash,
                spec.factor_id,
                spec.version,
                spec.semantic_hash,
                spec.definition_hash,
                registered_at,
                payload,
            ),
        )
        for parent in spec.provenance.parent_factor_hashes:
            connection.execute(
                "INSERT INTO factor_lineage(child_hash,parent_hash) VALUES(?,?)",
                (factor_hash, parent),
            )
        connection.execute(
            """
            INSERT INTO factor_lifecycle_events(
                factor_hash,state,occurred_at,evidence_hash
            ) VALUES(?,?,?,NULL)
            """,
            (factor_hash, "registered", registered_at),
        )
        return factor_hash

    def get(self, factor_hash: str) -> FactorSpec:
        require_sha256(factor_hash, name="factor hash")
        row = self.connection.execute(
            "SELECT payload_json FROM factors WHERE factor_hash=?", (factor_hash,)
        ).fetchone()
        if row is None:
            raise KeyError(factor_hash)
        spec = FactorSpec.from_mapping(json.loads(row["payload_json"]))
        if spec.content_hash != factor_hash:
            raise FactorRegistryConflict("stored factor hash differs from payload")
        return spec

    def list_hashes(self) -> tuple[str, ...]:
        return tuple(
            str(row[0])
            for row in self.connection.execute(
                "SELECT factor_hash FROM factors ORDER BY registered_at,rowid"
            ).fetchall()
        )

    def current_state(self, factor_hash: str) -> str:
        require_sha256(factor_hash, name="factor hash")
        row = self.connection.execute(
            """
            SELECT state FROM factor_lifecycle_events
            WHERE factor_hash=? ORDER BY event_id DESC LIMIT 1
            """,
            (factor_hash,),
        ).fetchone()
        if row is None:
            raise KeyError(factor_hash)
        return str(row["state"])

    def transition(
        self,
        factor_hash: str,
        *,
        to_state: str,
        occurred_at: object,
        evidence_hash: str | None = None,
    ) -> None:
        current = self.current_state(factor_hash)
        if to_state not in _LIFECYCLE_TRANSITIONS[current]:
            raise FactorRegistryConflict(
                f"invalid factor lifecycle transition:{current}->{to_state}"
            )
        if evidence_hash is not None:
            require_sha256(evidence_hash, name="factor lifecycle evidence hash")
        timestamp = pd.Timestamp(occurred_at)
        if timestamp.tzinfo is None:
            raise ValueError("factor lifecycle occurred_at must be timezone-aware")
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO factor_lifecycle_events(
                    factor_hash,state,occurred_at,evidence_hash
                ) VALUES(?,?,?,?)
                """,
                (factor_hash, to_state, timestamp.isoformat(), evidence_hash),
            )

    def record_empirical_correlation(
        self,
        left_hash: str,
        right_hash: str,
        *,
        evaluation_hash: str,
        correlation: float,
        observation_count: int,
        recorded_at: object,
    ) -> None:
        require_sha256(left_hash, name="left factor hash")
        require_sha256(right_hash, name="right factor hash")
        require_sha256(evaluation_hash, name="correlation evaluation hash")
        if left_hash == right_hash:
            raise ValueError("empirical correlation requires two different factors")
        if not math.isfinite(correlation) or not -1 <= correlation <= 1:
            raise ValueError("empirical factor correlation must lie in [-1,1]")
        if not isinstance(observation_count, int) or observation_count < 2:
            raise ValueError("empirical correlation requires at least two observations")
        timestamp = pd.Timestamp(recorded_at)
        if timestamp.tzinfo is None:
            raise ValueError("factor correlation recorded_at must be timezone-aware")
        left, right = sorted((left_hash, right_hash))
        for factor_hash in (left, right):
            if (
                self.connection.execute(
                    "SELECT 1 FROM factors WHERE factor_hash=?", (factor_hash,)
                ).fetchone()
                is None
            ):
                raise FactorRegistryConflict(
                    f"factor correlation references an unregistered factor:{factor_hash}"
                )
        try:
            with self._transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO factor_correlations(
                        left_hash,right_hash,evaluation_hash,correlation,
                        observation_count,recorded_at
                    ) VALUES(?,?,?,?,?,?)
                    """,
                    (
                        left,
                        right,
                        evaluation_hash,
                        float(correlation),
                        observation_count,
                        timestamp.isoformat(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise FactorRegistryConflict(
                "empirical correlation already recorded"
            ) from exc


__all__ = ["FACTOR_REGISTRY_SCHEMA", "FactorRegistry", "FactorRegistryConflict"]
