from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Iterator, Mapping

import pandas as pd

from alpha_research.core.hashing import canonical_json_bytes, require_sha256
from alpha_research.models.runner import ModelResult
from alpha_research.models.spec import ModelSpec


MODEL_REGISTRY_SCHEMA = "model-registry/v1"


class ModelRegistryConflict(RuntimeError):
    pass


class ModelRegistry:
    """Append-only model definitions and immutable run receipts."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self._initialize()

    def __enter__(self) -> "ModelRegistry":
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
            CREATE TABLE IF NOT EXISTS models(
                model_hash TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                version TEXT NOT NULL,
                registered_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                UNIQUE(model_id, version)
            );
            CREATE TABLE IF NOT EXISTS model_results(
                result_hash TEXT PRIMARY KEY,
                model_hash TEXT NOT NULL REFERENCES models(model_hash),
                registered_at TEXT NOT NULL,
                descriptor_json TEXT NOT NULL
            );
            """
        )
        row = self.connection.execute(
            "SELECT value FROM registry_metadata WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            self.connection.execute(
                "INSERT INTO registry_metadata(key,value) VALUES('schema_version',?)",
                (MODEL_REGISTRY_SCHEMA,),
            )
            self.connection.commit()
        elif row["value"] != MODEL_REGISTRY_SCHEMA:
            raise ModelRegistryConflict("model registry schema version differs")

    def register_spec(self, spec: ModelSpec, *, registered_at: object) -> str:
        timestamp = _timestamp(registered_at)
        spec_hash: str = spec.content_hash
        payload = canonical_json_bytes(spec.to_dict()).decode("utf-8")
        existing = self.connection.execute(
            "SELECT payload_json FROM models WHERE model_hash=?", (spec.content_hash,)
        ).fetchone()
        if existing is not None:
            if existing["payload_json"] != payload:
                raise ModelRegistryConflict("immutable model payload differs")
            return spec_hash
        try:
            with self._transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO models(model_hash,model_id,version,registered_at,payload_json)
                    VALUES(?,?,?,?,?)
                    """,
                    (
                        spec.content_hash,
                        spec.model_id,
                        spec.version,
                        timestamp.isoformat(),
                        payload,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ModelRegistryConflict("model id/version already exists") from exc
        return spec_hash

    def register_result(self, result: ModelResult, *, registered_at: object) -> str:
        timestamp = _timestamp(registered_at)
        result_hash: str = result.content_hash
        descriptor = result.descriptor()
        payload = canonical_json_bytes(descriptor).decode("utf-8")
        model = self.connection.execute(
            "SELECT 1 FROM models WHERE model_hash=?", (result.model_spec_hash,)
        ).fetchone()
        if model is None:
            raise ModelRegistryConflict("model result references an unregistered spec")
        existing = self.connection.execute(
            "SELECT descriptor_json FROM model_results WHERE result_hash=?",
            (result.content_hash,),
        ).fetchone()
        if existing is not None:
            if existing["descriptor_json"] != payload:
                raise ModelRegistryConflict("immutable model result descriptor differs")
            return result_hash
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO model_results(result_hash,model_hash,registered_at,descriptor_json)
                VALUES(?,?,?,?)
                """,
                (
                    result.content_hash,
                    result.model_spec_hash,
                    timestamp.isoformat(),
                    payload,
                ),
            )
        return result_hash

    def get_spec(self, model_hash: str) -> ModelSpec:
        require_sha256(model_hash, name="model hash")
        row = self.connection.execute(
            "SELECT payload_json FROM models WHERE model_hash=?", (model_hash,)
        ).fetchone()
        if row is None:
            raise KeyError(model_hash)
        spec = ModelSpec.from_mapping(json.loads(row["payload_json"]))
        if spec.content_hash != model_hash:
            raise ModelRegistryConflict("stored model hash differs from payload")
        return spec

    def get_result_descriptor(self, result_hash: str) -> Mapping[str, object]:
        require_sha256(result_hash, name="model result hash")
        row = self.connection.execute(
            "SELECT descriptor_json FROM model_results WHERE result_hash=?",
            (result_hash,),
        ).fetchone()
        if row is None:
            raise KeyError(result_hash)
        value = json.loads(row["descriptor_json"])
        if not isinstance(value, dict):  # pragma: no cover
            raise ModelRegistryConflict("model result descriptor is not an object")
        return value


def _timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("model registry timestamp must be timezone-aware")
    return timestamp


__all__ = ["MODEL_REGISTRY_SCHEMA", "ModelRegistry", "ModelRegistryConflict"]
