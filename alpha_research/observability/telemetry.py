from __future__ import annotations

import json
import math
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Iterator, Mapping

import pandas as pd

from alpha_research.core.hashing import canonical_json_bytes, hash_json, require_sha256


TELEMETRY_SCHEMA = "research-telemetry/v1"

_ATTRIBUTE_FORBIDDEN = (
    "raw_data",
    "security_identifier",
    "credential",
    "secret",
    "holdout_value",
    "test_value",
)

_SENSITIVE_VALUE = re.compile(
    r"(?:bearer\s+[A-Za-z0-9._-]{6,}|"
    r"(?:api[_-]?key|password|client[_-]?secret|access[_-]?token)\s*[:=]|"
    r"(?:sk|ghp|gho|xox[abprs])[-_][A-Za-z0-9_-]{6,}|"
    r"AKIA[0-9A-Z]{12,})",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class StageTelemetry:
    experiment_spec_hash: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    attempt_id: int
    stage: str
    status: str
    started_at: str
    finished_at: str
    input_rows: int
    output_rows: int
    symbols: int
    wall_seconds: float
    cpu_seconds: float
    peak_memory_bytes: int
    disk_read_bytes: int
    disk_write_bytes: int
    cache_hits: int
    cache_misses: int
    worker_count: int
    llm_calls: int
    llm_tokens: int
    llm_cost_microusd: int
    failure_code: str | None
    attributes: Mapping[str, str | int | float | bool | None]
    schema_version: str = "stage-telemetry/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "stage-telemetry/v1":
            raise ValueError("unsupported StageTelemetry schema")
        require_sha256(self.experiment_spec_hash, name="telemetry experiment hash")
        for name in ("trace_id", "span_id", "stage"):
            if not _safe_code(str(getattr(self, name))):
                raise ValueError(f"telemetry {name} is invalid")
        if self.parent_span_id is not None and not _safe_code(self.parent_span_id):
            raise ValueError("telemetry parent_span_id is invalid")
        if not isinstance(self.attempt_id, int) or self.attempt_id <= 0:
            raise ValueError("telemetry attempt_id must be positive")
        if self.status not in {"succeeded", "failed", "cancelled"}:
            raise ValueError("telemetry status is not terminal")
        started = pd.Timestamp(self.started_at)
        finished = pd.Timestamp(self.finished_at)
        if started.tzinfo is None or finished.tzinfo is None or finished < started:
            raise ValueError("telemetry timestamps are invalid")
        object.__setattr__(self, "started_at", started.isoformat())
        object.__setattr__(self, "finished_at", finished.isoformat())
        integers = (
            "input_rows",
            "output_rows",
            "symbols",
            "peak_memory_bytes",
            "disk_read_bytes",
            "disk_write_bytes",
            "cache_hits",
            "cache_misses",
            "llm_calls",
            "llm_tokens",
            "llm_cost_microusd",
        )
        for name in integers:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"telemetry {name} must be non-negative")
        if not isinstance(self.worker_count, int) or self.worker_count <= 0:
            raise ValueError("telemetry worker_count must be positive")
        for name in ("wall_seconds", "cpu_seconds"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"telemetry {name} must be finite and non-negative")
        if self.status == "failed" and not self.failure_code:
            raise ValueError("failed telemetry requires failure_code")
        if self.status != "failed" and self.failure_code is not None:
            raise ValueError("non-failed telemetry cannot carry failure_code")
        attributes = dict(sorted(self.attributes.items()))
        for key, value in attributes.items():
            lower = key.lower()
            if not _safe_code(key) or any(
                token in lower for token in _ATTRIBUTE_FORBIDDEN
            ):
                raise ValueError(f"telemetry attribute key is forbidden:{key}")
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise TypeError(f"telemetry attribute is not scalar:{key}")
            if isinstance(value, str) and len(value) > 256:
                raise ValueError(f"telemetry attribute string is too long:{key}")
            if isinstance(value, str) and _SENSITIVE_VALUE.search(value):
                raise ValueError(f"telemetry attribute value is sensitive:{key}")
        object.__setattr__(self, "attributes", MappingProxyType(attributes))

    @property
    def rows_per_second(self) -> float | None:
        return None if self.wall_seconds <= 0 else self.output_rows / self.wall_seconds

    @property
    def symbols_per_second(self) -> float | None:
        return None if self.wall_seconds <= 0 else self.symbols / self.wall_seconds

    @property
    def cache_hit_rate(self) -> float | None:
        total = self.cache_hits + self.cache_misses
        return None if total == 0 else self.cache_hits / total

    @property
    def parallel_efficiency(self) -> float | None:
        capacity = self.wall_seconds * self.worker_count
        return None if capacity <= 0 else min(1.0, self.cpu_seconds / capacity)

    @property
    def content_hash(self) -> str:
        digest: str = hash_json(self.to_dict())
        return digest

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_spec_hash": self.experiment_spec_hash,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "attempt_id": self.attempt_id,
            "stage": self.stage,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "symbols": self.symbols,
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "disk_read_bytes": self.disk_read_bytes,
            "disk_write_bytes": self.disk_write_bytes,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "worker_count": self.worker_count,
            "llm_calls": self.llm_calls,
            "llm_tokens": self.llm_tokens,
            "llm_cost_microusd": self.llm_cost_microusd,
            "failure_code": self.failure_code,
            "attributes": dict(self.attributes),
            "derived": {
                "rows_per_second": self.rows_per_second,
                "symbols_per_second": self.symbols_per_second,
                "cache_hit_rate": self.cache_hit_rate,
                "parallel_efficiency": self.parallel_efficiency,
            },
        }


@dataclass(frozen=True, slots=True)
class OperationalSignal:
    experiment_spec_hash: str
    signal_name: str
    severity: str
    value: float
    threshold: float
    observed_at: str
    evidence_hash: str

    def __post_init__(self) -> None:
        require_sha256(self.experiment_spec_hash, name="signal experiment hash")
        require_sha256(self.evidence_hash, name="signal evidence hash")
        if not _safe_code(self.signal_name):
            raise ValueError("operational signal name is invalid")
        if self.severity not in {"info", "warning", "critical"}:
            raise ValueError("operational signal severity is invalid")
        if not math.isfinite(self.value) or not math.isfinite(self.threshold):
            raise ValueError("operational signal values must be finite")
        timestamp = pd.Timestamp(self.observed_at)
        if timestamp.tzinfo is None:
            raise ValueError("operational signal timestamp must be timezone-aware")
        object.__setattr__(self, "observed_at", timestamp.isoformat())

    @property
    def content_hash(self) -> str:
        digest: str = hash_json(
            {
                "experiment_spec_hash": self.experiment_spec_hash,
                "signal_name": self.signal_name,
                "severity": self.severity,
                "value": self.value,
                "threshold": self.threshold,
                "observed_at": self.observed_at,
                "evidence_hash": self.evidence_hash,
            }
        )
        return digest


class TelemetryStore:
    def __init__(self, path: str | Path, *, experiment_spec_hash: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.experiment_spec_hash = require_sha256(
            experiment_spec_hash, name="telemetry experiment_spec_hash"
        )
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._initialize()

    def __enter__(self) -> "TelemetryStore":
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
            CREATE TABLE IF NOT EXISTS telemetry_metadata(
                key TEXT PRIMARY KEY,value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS stage_spans(
                span_hash TEXT PRIMARY KEY,
                span_id TEXT NOT NULL UNIQUE,
                trace_id TEXT NOT NULL,
                attempt_id INTEGER NOT NULL,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS operational_signals(
                signal_hash TEXT PRIMARY KEY,
                signal_name TEXT NOT NULL,
                severity TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        expected = {
            "schema_version": TELEMETRY_SCHEMA,
            "experiment_spec_hash": self.experiment_spec_hash,
        }
        existing = dict(
            self.connection.execute(
                "SELECT key,value FROM telemetry_metadata"
            ).fetchall()
        )
        if not existing:
            self.connection.executemany(
                "INSERT INTO telemetry_metadata(key,value) VALUES(?,?)",
                sorted(expected.items()),
            )
            self.connection.commit()
        elif existing != expected:
            raise ValueError("telemetry store binding differs")

    def record_span(self, telemetry: StageTelemetry) -> str:
        if telemetry.experiment_spec_hash != self.experiment_spec_hash:
            raise ValueError("telemetry span experiment binding differs")
        digest = telemetry.content_hash
        payload = canonical_json_bytes(telemetry.to_dict()).decode("utf-8")
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT span_hash,payload_json FROM stage_spans WHERE span_id=?",
                (telemetry.span_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["span_hash"] != digest
                    or existing["payload_json"] != payload
                ):
                    raise ValueError("immutable telemetry span differs")
                return digest
            connection.execute(
                """INSERT INTO stage_spans(
                    span_hash,span_id,trace_id,attempt_id,stage,status,payload_json,recorded_at
                ) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    digest,
                    telemetry.span_id,
                    telemetry.trace_id,
                    telemetry.attempt_id,
                    telemetry.stage,
                    telemetry.status,
                    payload,
                    telemetry.finished_at,
                ),
            )
        return digest

    def record_signal(self, signal: OperationalSignal) -> str:
        if signal.experiment_spec_hash != self.experiment_spec_hash:
            raise ValueError("operational signal experiment binding differs")
        digest = signal.content_hash
        payload = canonical_json_bytes(
            {
                "experiment_spec_hash": signal.experiment_spec_hash,
                "signal_name": signal.signal_name,
                "severity": signal.severity,
                "value": signal.value,
                "threshold": signal.threshold,
                "observed_at": signal.observed_at,
                "evidence_hash": signal.evidence_hash,
            }
        ).decode("utf-8")
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT payload_json FROM operational_signals WHERE signal_hash=?",
                (digest,),
            ).fetchone()
            if existing is not None:
                if existing["payload_json"] != payload:
                    raise ValueError("immutable operational signal differs")
                return digest
            connection.execute(
                """INSERT INTO operational_signals(
                    signal_hash,signal_name,severity,observed_at,payload_json
                ) VALUES(?,?,?,?,?)""",
                (
                    digest,
                    signal.signal_name,
                    signal.severity,
                    signal.observed_at,
                    payload,
                ),
            )
        return digest

    def summary(self) -> Mapping[str, float | int | None]:
        rows = self.connection.execute(
            "SELECT payload_json FROM stage_spans"
        ).fetchall()
        spans = [json.loads(row[0]) for row in rows]
        failure_count = sum(item["status"] == "failed" for item in spans)
        wall = [float(item["wall_seconds"]) for item in spans]
        rows_per_second = [
            item["derived"]["rows_per_second"]
            for item in spans
            if item["derived"]["rows_per_second"] is not None
        ]
        critical = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM operational_signals WHERE severity='critical'"
            ).fetchone()[0]
        )
        return MappingProxyType(
            {
                "span_count": len(spans),
                "failure_count": failure_count,
                "failure_rate": None if not spans else failure_count / len(spans),
                "total_wall_seconds": sum(wall),
                "mean_rows_per_second": (
                    None
                    if not rows_per_second
                    else sum(rows_per_second) / len(rows_per_second)
                ),
                "critical_signal_count": critical,
            }
        )

    def verify_integrity(self) -> tuple[str, ...]:
        errors: list[str] = []
        if self.connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            errors.append("sqlite_integrity_failed")
        for row in self.connection.execute("SELECT * FROM stage_spans"):
            payload = json.loads(row["payload_json"])
            if hash_json(payload) != row["span_hash"]:
                errors.append(f"span_hash:{row['span_id']}")
        for row in self.connection.execute("SELECT * FROM operational_signals"):
            payload = json.loads(row["payload_json"])
            if hash_json(payload) != row["signal_hash"]:
                errors.append(f"signal_hash:{row['signal_hash']}")
        return tuple(errors)


def _safe_code(value: str) -> bool:
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-")
    return value[0].isalnum() and all(character in allowed for character in value)


__all__ = [
    "OperationalSignal",
    "StageTelemetry",
    "TELEMETRY_SCHEMA",
    "TelemetryStore",
]
