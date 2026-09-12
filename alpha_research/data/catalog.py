from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Iterator, Mapping

from alpha_research.core.data import (
    DataSchema,
    DatasetSnapshot,
    DatasetSpec,
    FieldRole,
    FieldSpec,
    SourceAsset,
)
from alpha_research.core.frequency import (
    AvailabilityLag,
    AvailabilitySpec,
    FrequencyMode,
    FrequencySpec,
    RevisionPolicy,
    TimestampPolicy,
)
from alpha_research.core.hashing import canonical_json_bytes, hash_file


CATALOG_SCHEMA_VERSION = "alpha-research-data-catalog/v2"


class CatalogConflictError(RuntimeError):
    pass


class DataCatalog:
    """Append-only SQLite registry for schemas, datasets, and snapshots."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self._initialize()

    def __enter__(self) -> "DataCatalog":
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
            CREATE TABLE IF NOT EXISTS catalog_metadata(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS schemas(
                schema_hash TEXT PRIMARY KEY,
                schema_id TEXT NOT NULL,
                version TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                UNIQUE(schema_id, version)
            );
            CREATE TABLE IF NOT EXISTS datasets(
                dataset_hash TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                version TEXT NOT NULL,
                schema_hash TEXT NOT NULL REFERENCES schemas(schema_hash),
                payload_json TEXT NOT NULL,
                UNIQUE(dataset_id, version)
            );
            CREATE TABLE IF NOT EXISTS snapshots(
                snapshot_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                dataset_hash TEXT NOT NULL REFERENCES datasets(dataset_hash),
                schema_hash TEXT NOT NULL REFERENCES schemas(schema_hash),
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS snapshot_materializations(
                snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
                materialization_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY(snapshot_id, materialization_hash)
            );
            """
        )
        row = self.connection.execute(
            "SELECT value FROM catalog_metadata WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            self.connection.execute(
                "INSERT INTO catalog_metadata(key,value) VALUES('schema_version',?)",
                (CATALOG_SCHEMA_VERSION,),
            )
            self.connection.commit()
        elif row["value"] != CATALOG_SCHEMA_VERSION:
            raise CatalogConflictError("data catalog schema version differs")

    def register_schema(self, schema: DataSchema) -> None:
        payload = _json(schema.to_dict())
        self._insert_immutable(
            table="schemas",
            identity_column="schema_hash",
            identity=schema.content_hash,
            columns=("schema_hash", "schema_id", "version", "payload_json"),
            values=(schema.content_hash, schema.schema_id, schema.version, payload),
            payload=payload,
        )

    def register_dataset(self, dataset: DatasetSpec) -> None:
        if (
            self.connection.execute(
                "SELECT 1 FROM schemas WHERE schema_hash=?", (dataset.schema_hash,)
            ).fetchone()
            is None
        ):
            raise CatalogConflictError("dataset schema is not registered")
        payload = _json(dataset.to_dict())
        self._insert_immutable(
            table="datasets",
            identity_column="dataset_hash",
            identity=dataset.content_hash,
            columns=(
                "dataset_hash",
                "dataset_id",
                "version",
                "schema_hash",
                "payload_json",
            ),
            values=(
                dataset.content_hash,
                dataset.dataset_id,
                dataset.version,
                dataset.schema_hash,
                payload,
            ),
            payload=payload,
        )

    def register_snapshot(self, snapshot: DatasetSnapshot) -> None:
        dataset = self.connection.execute(
            "SELECT dataset_hash,schema_hash FROM datasets WHERE dataset_id=? AND version=?",
            (snapshot.dataset_id, snapshot.dataset_version),
        ).fetchone()
        if dataset is None:
            raise CatalogConflictError("snapshot dataset is not registered")
        if (
            dataset["dataset_hash"] != snapshot.dataset_spec_hash
            or dataset["schema_hash"] != snapshot.schema_hash
        ):
            raise CatalogConflictError("snapshot dataset/schema binding differs")
        payload = _json(snapshot.manifest_dict())
        self._insert_immutable(
            table="snapshots",
            identity_column="snapshot_id",
            identity=snapshot.snapshot_id,
            columns=(
                "snapshot_id",
                "dataset_id",
                "dataset_version",
                "dataset_hash",
                "schema_hash",
                "payload_json",
            ),
            values=(
                snapshot.snapshot_id,
                snapshot.dataset_id,
                snapshot.dataset_version,
                snapshot.dataset_spec_hash,
                snapshot.schema_hash,
                payload,
            ),
            payload=payload,
        )
        materialization = _json(snapshot.materialization_payload())
        self._insert_immutable(
            table="snapshot_materializations",
            identity_column="materialization_hash",
            identity=snapshot.materialization_hash,
            columns=("snapshot_id", "materialization_hash", "payload_json"),
            values=(
                snapshot.snapshot_id,
                snapshot.materialization_hash,
                materialization,
            ),
            payload=materialization,
            extra_identity=("snapshot_id", snapshot.snapshot_id),
        )

    def _insert_immutable(
        self,
        *,
        table: str,
        identity_column: str,
        identity: str,
        columns: tuple[str, ...],
        values: tuple[object, ...],
        payload: str,
        extra_identity: tuple[str, object] | None = None,
    ) -> None:
        where = f"{identity_column}=?"
        parameters: tuple[object, ...] = (identity,)
        if extra_identity is not None:
            where += f" AND {extra_identity[0]}=?"
            parameters += (extra_identity[1],)
        row = self.connection.execute(
            f"SELECT payload_json FROM {table} WHERE {where}", parameters
        ).fetchone()
        if row is not None:
            if row["payload_json"] != payload:
                raise CatalogConflictError(f"immutable {table} payload differs")
            return
        placeholders = ",".join("?" for _ in values)
        try:
            with self._transaction() as connection:
                connection.execute(
                    f"INSERT INTO {table}({','.join(columns)}) VALUES({placeholders})",
                    values,
                )
        except sqlite3.IntegrityError as exc:
            raise CatalogConflictError(
                f"immutable {table} identity/version conflict"
            ) from exc

    def get_schema(self, schema_hash: str) -> DataSchema:
        row = self.connection.execute(
            "SELECT payload_json FROM schemas WHERE schema_hash=?", (schema_hash,)
        ).fetchone()
        if row is None:
            raise KeyError(schema_hash)
        return _schema_from_dict(json.loads(row["payload_json"]))

    def get_dataset(self, dataset_id: str, version: str) -> DatasetSpec:
        row = self.connection.execute(
            "SELECT payload_json FROM datasets WHERE dataset_id=? AND version=?",
            (dataset_id, version),
        ).fetchone()
        if row is None:
            raise KeyError((dataset_id, version))
        return _dataset_from_dict(json.loads(row["payload_json"]))

    def get_snapshot(self, snapshot_id: str) -> DatasetSnapshot:
        row = self.connection.execute(
            "SELECT payload_json FROM snapshots WHERE snapshot_id=?", (snapshot_id,)
        ).fetchone()
        if row is None:
            raise KeyError(snapshot_id)
        materialization = self.connection.execute(
            """
            SELECT payload_json FROM snapshot_materializations
            WHERE snapshot_id=? ORDER BY rowid LIMIT 1
            """,
            (snapshot_id,),
        ).fetchone()
        if materialization is None:
            raise CatalogConflictError("snapshot has no registered materialization")
        return _snapshot_from_dict(
            {
                **json.loads(row["payload_json"]),
                **json.loads(materialization["payload_json"]),
            }
        )


class SnapshotManifestStore:
    """Write-once content-addressed JSON manifests outside raw data storage."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def publish(self, snapshot: DatasetSnapshot) -> Path:
        target = self.root / f"{snapshot.snapshot_id}.json"
        payload = canonical_json_bytes(snapshot.manifest_dict()) + b"\n"
        if target.exists():
            if target.read_bytes() != payload:
                raise CatalogConflictError("snapshot manifest content conflict")
            return target
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return target

    def verify(self, snapshot: DatasetSnapshot) -> None:
        target = self.root / f"{snapshot.snapshot_id}.json"
        expected = canonical_json_bytes(snapshot.manifest_dict()) + b"\n"
        if not target.is_file() or target.read_bytes() != expected:
            raise CatalogConflictError("snapshot manifest missing or changed")
        for asset in snapshot.source_assets:
            path = Path(asset.uri)
            if not path.is_file() or hash_file(path) != asset.sha256:
                raise CatalogConflictError(
                    f"snapshot source asset changed:{asset.logical_name}"
                )


def _json(value: object) -> str:
    encoded: bytes = canonical_json_bytes(value)
    return encoded.decode("utf-8")


def _schema_from_dict(value: Mapping[str, object]) -> DataSchema:
    raw_fields = value["fields"]
    if not isinstance(raw_fields, list) or not all(
        isinstance(item, Mapping) for item in raw_fields
    ):
        raise TypeError("catalog schema fields must be a list of objects")
    fields = tuple(
        FieldSpec(
            name=str(item["name"]),
            dtype=str(item["dtype"]),
            role=FieldRole(str(item["role"])),
            nullable=bool(item["nullable"]),
            unit=None if item["unit"] is None else str(item["unit"]),
            minimum=item["minimum"],
            maximum=item["maximum"],
            applicability_field=item.get("applicability_field"),
        )
        for item in raw_fields
    )
    raw_key_fields = value.get("key_fields")
    if raw_key_fields is not None and not isinstance(raw_key_fields, list):
        raise TypeError("catalog schema key_fields must be a list")
    key_fields = (
        tuple(str(item) for item in raw_key_fields)
        if isinstance(raw_key_fields, list) and raw_key_fields
        else None
    )
    return DataSchema(
        schema_id=str(value["schema_id"]),
        version=str(value["version"]),
        fields=fields,
        timestamp_field=str(value["timestamp_field"]),
        security_field=str(value["security_field"]),
        key_fields=key_fields,
    )


def _frequency_from_dict(value: Mapping[str, object]) -> FrequencySpec:
    return FrequencySpec(
        mode=FrequencyMode(str(value["mode"])),
        interval=None if value["interval"] is None else str(value["interval"]),
        calendar_id=str(value["calendar_id"]),
        timezone=str(value["timezone"]),
        session_id=str(value["session_id"]),
        timestamp_policy=TimestampPolicy(str(value["timestamp_policy"])),
        sampling_policy=str(value["sampling_policy"]),
    )


def _availability_from_dict(value: Mapping[str, object]) -> AvailabilitySpec:
    lag_value = value["publication_lag"]
    lag = (
        AvailabilityLag(
            duration=str(lag_value["duration"]),
            applies_from=str(lag_value["applies_from"]),
        )
        if isinstance(lag_value, dict)
        else str(lag_value)
    )
    return AvailabilitySpec(
        effective_at_field=str(value["effective_at_field"]),
        known_at_field=str(value["known_at_field"]),
        publication_lag=lag,
        revision_policy=RevisionPolicy(str(value["revision_policy"])),
    )


def _dataset_from_dict(value: Mapping[str, object]) -> DatasetSpec:
    frequency = value["frequency"]
    availability = value["availability"]
    if not isinstance(frequency, Mapping) or not isinstance(availability, Mapping):
        raise TypeError("catalog dataset time contracts must be objects")
    return DatasetSpec(
        dataset_id=str(value["dataset_id"]),
        version=str(value["version"]),
        owner=str(value["owner"]),
        description=str(value["description"]),
        storage_format=str(value["storage_format"]),
        schema_hash=str(value["schema_hash"]),
        frequency=_frequency_from_dict(frequency),
        availability=_availability_from_dict(availability),
    )


def _snapshot_from_dict(value: Mapping[str, object]) -> DatasetSnapshot:
    raw_assets = value["source_assets"]
    if not isinstance(raw_assets, list) or not all(
        isinstance(item, Mapping) for item in raw_assets
    ):
        raise TypeError("catalog snapshot source assets must be a list of objects")
    assets = tuple(
        SourceAsset(
            logical_name=str(item["logical_name"]),
            uri=str(item["uri"]),
            sha256=str(item["sha256"]),
            size_bytes=int(item["size_bytes"]),
            mtime_ns=int(item["mtime_ns"]),
            mode=int(item["mode"]),
        )
        for item in raw_assets
    )
    return DatasetSnapshot(
        snapshot_id=str(value["snapshot_id"]),
        dataset_id=str(value["dataset_id"]),
        dataset_version=str(value["dataset_version"]),
        dataset_spec_hash=str(value["dataset_spec_hash"]),
        schema_hash=str(value["schema_hash"]),
        source_assets=assets,
        source_vintage_at=str(value["source_vintage_at"]),
        created_at=str(value["created_at"]),
    )


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "CatalogConflictError",
    "DataCatalog",
    "SnapshotManifestStore",
]
