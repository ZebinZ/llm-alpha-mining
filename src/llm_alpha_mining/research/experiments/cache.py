from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Iterator

from llm_alpha_mining.research.core.hashing import hash_json, require_sha256
from llm_alpha_mining.mining.artifacts import ArtifactStore, hash_file


CACHE_SCHEMA = "research-content-cache/v1"


@dataclass(frozen=True, slots=True)
class SignalRealizationKey:
    factor_definition_hash: str
    data_snapshot_hash: str
    security_contract_hash: str
    availability_contract_hash: str
    frequency_spec_hash: str
    transform_spec_hash: str
    code_snapshot_hash: str
    environment_hash: str
    partition_hashes: tuple[str, ...]
    parent_cache_key_hash: str | None = None
    schema_version: str = "signal-realization-key/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "signal-realization-key/v1":
            raise ValueError("unsupported SignalRealizationKey schema")
        for name in (
            "factor_definition_hash",
            "data_snapshot_hash",
            "security_contract_hash",
            "availability_contract_hash",
            "frequency_spec_hash",
            "transform_spec_hash",
            "code_snapshot_hash",
            "environment_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"cache key {name}")
        if not self.partition_hashes or len(set(self.partition_hashes)) != len(
            self.partition_hashes
        ):
            raise ValueError("cache key partition hashes must be non-empty and unique")
        for digest in self.partition_hashes:
            require_sha256(digest, name="cache key partition hash")
        if self.parent_cache_key_hash is not None:
            require_sha256(
                self.parent_cache_key_hash, name="cache key parent_cache_key_hash"
            )

    @property
    def content_hash(self) -> str:
        digest: str = hash_json(
            {
                "schema_version": self.schema_version,
                "factor_definition_hash": self.factor_definition_hash,
                "data_snapshot_hash": self.data_snapshot_hash,
                "security_contract_hash": self.security_contract_hash,
                "availability_contract_hash": self.availability_contract_hash,
                "frequency_spec_hash": self.frequency_spec_hash,
                "transform_spec_hash": self.transform_spec_hash,
                "code_snapshot_hash": self.code_snapshot_hash,
                "environment_hash": self.environment_hash,
                "partition_hashes": list(self.partition_hashes),
                "parent_cache_key_hash": self.parent_cache_key_hash,
            }
        )
        return digest


@dataclass(frozen=True, slots=True)
class CacheEntry:
    cache_key_hash: str
    artifact_hash: str
    location: str
    size_bytes: int
    created_at: str
    hit_count: int


class ResearchContentCache:
    """Immutable content cache keyed by the complete signal realization identity."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        maximum_bytes: int,
        database_name: str = "cache/index.sqlite3",
    ) -> None:
        if not isinstance(maximum_bytes, int) or maximum_bytes <= 0:
            raise ValueError("cache maximum_bytes must be positive")
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.maximum_bytes = maximum_bytes
        self.artifacts = ArtifactStore(self.workspace_root, object_dir="cache/objects")
        self.path = self.workspace_root / database_name
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=5.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._initialize()

    def __enter__(self) -> "ResearchContentCache":
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
            CREATE TABLE IF NOT EXISTS cache_metadata(
                key TEXT PRIMARY KEY,value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cache_entries(
                cache_key_hash TEXT PRIMARY KEY,
                artifact_hash TEXT NOT NULL,
                location TEXT NOT NULL,
                size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
                created_at TEXT NOT NULL,
                last_accessed_at TEXT NOT NULL,
                hit_count INTEGER NOT NULL CHECK(hit_count >= 0)
            );
            CREATE TABLE IF NOT EXISTS cache_misses(
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key_hash TEXT NOT NULL,
                observed_at TEXT NOT NULL
            );
            """
        )
        expected = {
            "schema_version": CACHE_SCHEMA,
            "maximum_bytes": str(self.maximum_bytes),
        }
        existing = dict(
            self.connection.execute("SELECT key,value FROM cache_metadata").fetchall()
        )
        if not existing:
            self.connection.executemany(
                "INSERT INTO cache_metadata(key,value) VALUES(?,?)",
                sorted(expected.items()),
            )
            self.connection.commit()
        elif existing != expected:
            raise ValueError("content cache binding differs")

    def get(
        self, key: SignalRealizationKey, *, accessed_at: datetime | None = None
    ) -> bytes | None:
        timestamp = _iso(accessed_at or _utc_now())
        digest = key.content_hash
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM cache_entries WHERE cache_key_hash=?", (digest,)
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO cache_misses(cache_key_hash,observed_at) VALUES(?,?)",
                    (digest, timestamp),
                )
                return None
            path: Path = (self.workspace_root / row["location"]).resolve()
            try:
                path.relative_to(self.workspace_root)
            except ValueError:
                raise RuntimeError("cache entry resolves outside workspace") from None
            if not path.is_file() or path.stat().st_size != int(row["size_bytes"]):
                raise RuntimeError("cache artifact is missing or has a different size")
            if hash_file(path) != row["artifact_hash"]:
                raise RuntimeError("cache artifact hash differs")
            connection.execute(
                """UPDATE cache_entries SET hit_count=hit_count+1,last_accessed_at=?
                   WHERE cache_key_hash=?""",
                (timestamp, digest),
            )
            return path.read_bytes()

    def put(
        self,
        key: SignalRealizationKey,
        payload: bytes,
        *,
        created_at: datetime | None = None,
    ) -> CacheEntry:
        if not isinstance(payload, bytes):
            raise TypeError("cache payload must be bytes")
        timestamp = _iso(created_at or _utc_now())
        key_hash = key.content_hash
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM cache_entries WHERE cache_key_hash=?", (key_hash,)
            ).fetchone()
            if existing is not None:
                path = self.workspace_root / existing["location"]
                if (
                    hash_file(path) != existing["artifact_hash"]
                    or path.read_bytes() != payload
                ):
                    raise ValueError(
                        "immutable cache key already binds different content"
                    )
                return _entry(existing)
            used = int(
                connection.execute(
                    "SELECT COALESCE(SUM(size_bytes),0) FROM cache_entries"
                ).fetchone()[0]
            )
            if used + len(payload) > self.maximum_bytes:
                raise ValueError("content cache byte budget is exhausted")
            record = self.artifacts.put_bytes(
                f"cache-{key_hash}", payload, role="intermediate_cache"
            )
            connection.execute(
                """INSERT INTO cache_entries(
                    cache_key_hash,artifact_hash,location,size_bytes,created_at,
                    last_accessed_at,hit_count
                ) VALUES(?,?,?,?,?,?,0)""",
                (
                    key_hash,
                    record.sha256,
                    record.location,
                    record.size_bytes,
                    timestamp,
                    timestamp,
                ),
            )
            row = connection.execute(
                "SELECT * FROM cache_entries WHERE cache_key_hash=?", (key_hash,)
            ).fetchone()
            if row is None:  # pragma: no cover
                raise RuntimeError("cache insert disappeared")
            return _entry(row)

    def stats(self) -> dict[str, float | int | None]:
        entries, bytes_used, hits = self.connection.execute(
            """SELECT COUNT(*),COALESCE(SUM(size_bytes),0),COALESCE(SUM(hit_count),0)
               FROM cache_entries"""
        ).fetchone()
        misses = int(
            self.connection.execute("SELECT COUNT(*) FROM cache_misses").fetchone()[0]
        )
        total = int(hits) + misses
        return {
            "entry_count": int(entries),
            "bytes_used": int(bytes_used),
            "hit_count": int(hits),
            "miss_count": misses,
            "hit_rate": None if total == 0 else int(hits) / total,
        }

    def verify_integrity(self) -> tuple[str, ...]:
        errors: list[str] = []
        if self.connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            errors.append("sqlite_integrity_failed")
        for row in self.connection.execute("SELECT * FROM cache_entries"):
            path = self.workspace_root / row["location"]
            if not path.is_file():
                errors.append(f"missing:{row['cache_key_hash']}")
            elif hash_file(path) != row["artifact_hash"]:
                errors.append(f"hash:{row['cache_key_hash']}")
        return tuple(errors)


def _entry(row: sqlite3.Row) -> CacheEntry:
    return CacheEntry(
        cache_key_hash=row["cache_key_hash"],
        artifact_hash=row["artifact_hash"],
        location=row["location"],
        size_bytes=int(row["size_bytes"]),
        created_at=row["created_at"],
        hit_count=int(row["hit_count"]),
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("cache timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


__all__ = [
    "CACHE_SCHEMA",
    "CacheEntry",
    "ResearchContentCache",
    "SignalRealizationKey",
]
