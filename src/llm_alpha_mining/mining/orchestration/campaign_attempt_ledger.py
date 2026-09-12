from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from llm_alpha_mining.mining.artifacts import ArtifactManifest
from llm_alpha_mining.mining.artifacts.hashing import (
    canonical_json_bytes,
    hash_bytes,
    hash_file,
    hash_json,
)


CAMPAIGN_ATTEMPT_LEDGER_SCHEMA = "campaign-attempt-ledger/v1"
CHILD_ATTEMPT_SNAPSHOT_SCHEMA = "campaign-child-attempt-snapshot/v1"
CHILD_ATTEMPT_LOGICAL_STATE_SCHEMA = "evaluation-attempt-logical-state/v1"
CAMPAIGN_ATTEMPT_CHECKPOINT_SCHEMA = "campaign-attempt-checkpoint/v1"
CAMPAIGN_LEDGER_SEAL_RECEIPT_SCHEMA = "campaign-attempt-ledger-seal-receipt/v1"
DEFAULT_CAMPAIGN_ATTEMPT_BUDGET = 120

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CODE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}")
_CHILD_STATUSES = frozenset({"leased", "running", "succeeded", "failed", "expired"})
_TERMINAL_CHILD_STATUSES = frozenset({"succeeded", "failed", "expired"})
_CHECKPOINT_STAGES = ("raw_signal", "neutral_signal", "diagnostics", "metrics")
_ATTEMPT_IMMUTABLE_FIELDS = (
    "attempt_id",
    "candidate_id",
    "attempt_number",
    "worker_id",
    "input_hash",
    "evaluator_hash",
    "reserved_at",
    "estimated_wall_seconds",
)


class CampaignAttemptLedgerError(RuntimeError):
    pass


class CampaignAttemptBudgetExceeded(CampaignAttemptLedgerError):
    pass


class CampaignAttemptConflict(CampaignAttemptLedgerError):
    pass


class CampaignAttemptLedgerSealed(CampaignAttemptLedgerError):
    """A mutation was attempted after the campaign ledger was sealed."""


class ChildSnapshotIntegrityError(CampaignAttemptLedgerError):
    pass


class CampaignAttemptStatus(str, Enum):
    RESERVED = "reserved"
    LEASED = "leased"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"
    INTERRUPTED = "interrupted"
    ORPHANED = "orphaned"


_TERMINAL_LEDGER_STATUSES = frozenset(
    {
        CampaignAttemptStatus.SUCCEEDED.value,
        CampaignAttemptStatus.FAILED.value,
        CampaignAttemptStatus.EXPIRED.value,
        CampaignAttemptStatus.INTERRUPTED.value,
        CampaignAttemptStatus.ORPHANED.value,
    }
)


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
    if not isinstance(value, str) or _CODE_RE.fullmatch(value) is None:
        raise ValueError(f"{name} contains unsupported characters")
    return value


def _positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_hashed_payload(path: Path, payload: bytes) -> str:
    """Atomically replace both a payload and its checksum sidecar.

    There is no portable two-file atomic rename.  The payload is therefore
    committed first and the sidecar second; readers fail closed unless both
    exist and agree.  A crash can leave an unreadable receipt, never a receipt
    that silently authenticates different bytes.
    """

    digest = hash_bytes(payload)
    _atomic_write(path, payload)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    _atomic_write(sidecar, f"{digest}  {path.name}\n".encode("utf-8"))
    return digest


def _load_workspace_descriptor(root: Path) -> tuple[dict[str, Any], str]:
    path = root / "workspace.json"
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ChildSnapshotIntegrityError(
            f"cannot load workspace descriptor: {exc}"
        ) from exc
    expected = {
        "schema_version",
        "run_id",
        "protocol_hash",
        "database",
        "protocol",
        "manifest",
        "descriptor_hash",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ChildSnapshotIntegrityError("workspace descriptor schema is invalid")
    if value["schema_version"] != "alpha-mining-workspace/v5":
        raise ChildSnapshotIntegrityError("workspace descriptor version is unsupported")
    core = {key: value[key] for key in expected - {"descriptor_hash"}}
    if hash_json(core) != value["descriptor_hash"]:
        raise ChildSnapshotIntegrityError("workspace descriptor hash mismatch")
    _require_code(value["run_id"], "workspace run_id")
    _require_digest(value["protocol_hash"], "workspace protocol_hash")
    _require_digest(value["descriptor_hash"], "workspace descriptor_hash")
    for name in ("database", "protocol", "manifest"):
        location = Path(value[name])
        if location.is_absolute() or ".." in location.parts:
            raise ChildSnapshotIntegrityError(f"unsafe workspace {name} path")
    return value, hash_bytes(raw)


def _read_attempt_logical_payload(
    path: Path, *, immutable: bool = False
) -> tuple[dict[str, Any], str]:
    """Read SQLite logical rows in a read transaction; never read checkpoint payloads."""

    if not path.is_file():
        raise ChildSnapshotIntegrityError(f"child attempt store is missing: {path}")
    options = "mode=ro&immutable=1" if immutable else "mode=ro"
    connection = sqlite3.connect(
        f"{path.resolve().as_uri()}?{options}", uri=True, timeout=5.0
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only = ON")
        connection.execute("BEGIN")
        payload = {
            "schema_version": CHILD_ATTEMPT_LOGICAL_STATE_SCHEMA,
            "attempt_metadata": [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM attempt_metadata ORDER BY key"
                ).fetchall()
            ],
            "evaluation_attempts": [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM evaluation_attempts ORDER BY attempt_id"
                ).fetchall()
            ],
            "evaluation_checkpoints": [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM evaluation_checkpoints ORDER BY attempt_id, stage"
                ).fetchall()
            ],
        }
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        connection.rollback()
    except sqlite3.Error as exc:
        raise ChildSnapshotIntegrityError(
            f"cannot read child attempt store: {exc}"
        ) from exc
    finally:
        connection.close()
    if integrity != "ok":
        raise ChildSnapshotIntegrityError(
            f"child attempt SQLite integrity failed: {integrity}"
        )
    return payload, hash_json(payload)


def _validate_child_attempt_rows(payload: Mapping[str, Any]) -> None:
    metadata_rows = payload.get("attempt_metadata")
    attempts = payload.get("evaluation_attempts")
    checkpoints = payload.get("evaluation_checkpoints")
    if (
        not isinstance(metadata_rows, list)
        or not isinstance(attempts, list)
        or not isinstance(checkpoints, list)
    ):
        raise ChildSnapshotIntegrityError("attempt logical payload lists are invalid")
    metadata = {
        row.get("key"): row.get("value")
        for row in metadata_rows
        if isinstance(row, dict)
    }
    if len(metadata) != len(metadata_rows):
        raise ChildSnapshotIntegrityError(
            "attempt metadata rows are malformed or duplicated"
        )
    if metadata.get("schema_version") != "evaluation-attempt-store/v1":
        raise ChildSnapshotIntegrityError("child attempt-store schema mismatch")
    attempt_ids: set[int] = set()
    attempt_status: dict[int, str] = {}
    checkpoint_by_attempt: dict[int, list[dict[str, Any]]] = {}
    for row in attempts:
        if not isinstance(row, dict):
            raise ChildSnapshotIntegrityError("child attempt row is not an object")
        attempt_id = row.get("attempt_id")
        if (
            not isinstance(attempt_id, int)
            or isinstance(attempt_id, bool)
            or attempt_id <= 0
        ):
            raise ChildSnapshotIntegrityError("child attempt_id is invalid")
        if attempt_id in attempt_ids:
            raise ChildSnapshotIntegrityError("duplicate child attempt_id")
        attempt_ids.add(attempt_id)
        status = row.get("status")
        if status not in _CHILD_STATUSES:
            raise ChildSnapshotIntegrityError(f"unsupported child status: {status!r}")
        attempt_status[attempt_id] = status
        _require_code(row.get("candidate_id"), "child candidate_id")
        _require_code(row.get("worker_id"), "child worker_id")
        _require_digest(row.get("input_hash"), "child input_hash")
        _require_digest(row.get("evaluator_hash"), "child evaluator_hash")
        result_hash = row.get("result_hash")
        if result_hash is not None:
            _require_digest(result_hash, "child result_hash")
        if status == "succeeded" and result_hash is None:
            raise ChildSnapshotIntegrityError(
                "successful child attempt lacks result hash"
            )
    for row in checkpoints:
        if not isinstance(row, dict):
            raise ChildSnapshotIntegrityError("child checkpoint row is not an object")
        attempt_id = row.get("attempt_id")
        if attempt_id not in attempt_ids:
            raise ChildSnapshotIntegrityError(
                "checkpoint references unknown child attempt"
            )
        stage = row.get("stage")
        if stage not in _CHECKPOINT_STAGES:
            raise ChildSnapshotIntegrityError(
                f"unknown child checkpoint stage: {stage!r}"
            )
        _require_digest(row.get("sha256"), "child checkpoint sha256")
        size = row.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ChildSnapshotIntegrityError("child checkpoint size is invalid")
        checkpoint_by_attempt.setdefault(attempt_id, []).append(row)
    for attempt_id, rows in checkpoint_by_attempt.items():
        positions = sorted(_CHECKPOINT_STAGES.index(row["stage"]) for row in rows)
        if len(positions) != len(set(positions)) or positions != list(
            range(len(positions))
        ):
            raise ChildSnapshotIntegrityError(
                f"child checkpoint sequence is not a unique prefix: {attempt_id}"
            )
    for attempt_id, status in attempt_status.items():
        if (
            status == "succeeded"
            and len(checkpoint_by_attempt.get(attempt_id, ())) != 4
        ):
            raise ChildSnapshotIntegrityError(
                f"successful child attempt lacks four checkpoints: {attempt_id}"
            )


@dataclass(frozen=True, slots=True)
class ChildAttemptSnapshot:
    run_id: str
    protocol_hash: str
    workspace_descriptor_hash: str
    workspace_descriptor_file_sha256: str
    manifest_file_sha256: str
    manifest_content_sha256: str
    attempt_store_logical_state_sha256: str
    attempt_metadata: tuple[dict[str, Any], ...]
    attempts: tuple[dict[str, Any], ...]
    checkpoints: tuple[dict[str, Any], ...]
    snapshot_sha256: str

    def _core_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CHILD_ATTEMPT_SNAPSHOT_SCHEMA,
            "run_id": self.run_id,
            "protocol_hash": self.protocol_hash,
            "workspace_descriptor_hash": self.workspace_descriptor_hash,
            "workspace_descriptor_file_sha256": self.workspace_descriptor_file_sha256,
            "manifest_file_sha256": self.manifest_file_sha256,
            "manifest_content_sha256": self.manifest_content_sha256,
            "attempt_store": {
                "schema_version": CHILD_ATTEMPT_LOGICAL_STATE_SCHEMA,
                "logical_state_sha256": self.attempt_store_logical_state_sha256,
                "attempt_metadata": list(self.attempt_metadata),
                "evaluation_attempts": list(self.attempts),
                "evaluation_checkpoints": list(self.checkpoints),
                "attempt_count": len(self.attempts),
                "checkpoint_count": len(self.checkpoints),
                "status_counts": dict(
                    sorted(Counter(row["status"] for row in self.attempts).items())
                ),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        value = self._core_dict()
        value["snapshot_sha256"] = self.snapshot_sha256
        return value

    def __post_init__(self) -> None:
        _require_code(self.run_id, "snapshot run_id")
        for name in (
            "protocol_hash",
            "workspace_descriptor_hash",
            "workspace_descriptor_file_sha256",
            "manifest_file_sha256",
            "manifest_content_sha256",
            "attempt_store_logical_state_sha256",
            "snapshot_sha256",
        ):
            _require_digest(getattr(self, name), name)
        logical_payload = {
            "schema_version": CHILD_ATTEMPT_LOGICAL_STATE_SCHEMA,
            "attempt_metadata": list(self.attempt_metadata),
            "evaluation_attempts": list(self.attempts),
            "evaluation_checkpoints": list(self.checkpoints),
        }
        _validate_child_attempt_rows(logical_payload)
        if hash_json(logical_payload) != self.attempt_store_logical_state_sha256:
            raise ChildSnapshotIntegrityError(
                "child attempt logical-state hash mismatch"
            )
        if hash_json(self._core_dict()) != self.snapshot_sha256:
            raise ChildSnapshotIntegrityError("child snapshot hash mismatch")
        metadata = {row["key"]: row["value"] for row in self.attempt_metadata}
        if metadata.get("run_id") != self.run_id:
            raise ChildSnapshotIntegrityError("attempt metadata run_id mismatch")
        if metadata.get("protocol_hash") != self.protocol_hash:
            raise ChildSnapshotIntegrityError("attempt metadata protocol_hash mismatch")
        if metadata.get("workspace_descriptor_hash") != self.workspace_descriptor_hash:
            raise ChildSnapshotIntegrityError(
                "attempt metadata descriptor hash mismatch"
            )

    @classmethod
    def from_core_dict(cls, value: Mapping[str, Any]) -> "ChildAttemptSnapshot":
        if not isinstance(value, Mapping):
            raise ChildSnapshotIntegrityError("child snapshot must be an object")
        expected = {
            "schema_version",
            "run_id",
            "protocol_hash",
            "workspace_descriptor_hash",
            "workspace_descriptor_file_sha256",
            "manifest_file_sha256",
            "manifest_content_sha256",
            "attempt_store",
        }
        if (
            set(value) != expected
            or value.get("schema_version") != CHILD_ATTEMPT_SNAPSHOT_SCHEMA
        ):
            raise ChildSnapshotIntegrityError("child snapshot core schema is invalid")
        attempt_store = value["attempt_store"]
        if not isinstance(attempt_store, Mapping):
            raise ChildSnapshotIntegrityError("child attempt_store is invalid")
        required_store = {
            "schema_version",
            "logical_state_sha256",
            "attempt_metadata",
            "evaluation_attempts",
            "evaluation_checkpoints",
            "attempt_count",
            "checkpoint_count",
            "status_counts",
        }
        if set(attempt_store) != required_store:
            raise ChildSnapshotIntegrityError("child attempt_store schema is invalid")
        if attempt_store.get("schema_version") != CHILD_ATTEMPT_LOGICAL_STATE_SCHEMA:
            raise ChildSnapshotIntegrityError("child logical-state schema mismatch")
        attempts = tuple(dict(row) for row in attempt_store["evaluation_attempts"])
        checkpoints = tuple(
            dict(row) for row in attempt_store["evaluation_checkpoints"]
        )
        metadata = tuple(dict(row) for row in attempt_store["attempt_metadata"])
        if attempt_store["attempt_count"] != len(attempts):
            raise ChildSnapshotIntegrityError("child attempt count mismatch")
        if attempt_store["checkpoint_count"] != len(checkpoints):
            raise ChildSnapshotIntegrityError("child checkpoint count mismatch")
        status_counts = dict(sorted(Counter(row["status"] for row in attempts).items()))
        if attempt_store["status_counts"] != status_counts:
            raise ChildSnapshotIntegrityError("child status counts mismatch")
        core = dict(value)
        return cls(
            run_id=value["run_id"],
            protocol_hash=value["protocol_hash"],
            workspace_descriptor_hash=value["workspace_descriptor_hash"],
            workspace_descriptor_file_sha256=value["workspace_descriptor_file_sha256"],
            manifest_file_sha256=value["manifest_file_sha256"],
            manifest_content_sha256=value["manifest_content_sha256"],
            attempt_store_logical_state_sha256=attempt_store["logical_state_sha256"],
            attempt_metadata=metadata,
            attempts=attempts,
            checkpoints=checkpoints,
            snapshot_sha256=hash_json(core),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChildAttemptSnapshot":
        if not isinstance(value, Mapping) or "snapshot_sha256" not in value:
            raise ChildSnapshotIntegrityError("child snapshot hash is missing")
        expected_hash = value["snapshot_sha256"]
        core = {key: item for key, item in value.items() if key != "snapshot_sha256"}
        snapshot = cls.from_core_dict(core)
        if snapshot.snapshot_sha256 != expected_hash:
            raise ChildSnapshotIntegrityError("serialized child snapshot hash mismatch")
        return snapshot


def import_child_workspace_snapshot(
    workspace_root: str | Path,
    *,
    immutable_attempt_store: str | Path | None = None,
    require_manifest_binding: bool = False,
) -> ChildAttemptSnapshot:
    """Create a metadata-only snapshot of a child workspace.

    The function hashes and parses ``workspace.json`` and ``manifest.json`` and
    reads logical rows from the attempt SQLite database.  It deliberately never
    opens any path named by an evaluation checkpoint or artifact manifest.
    SQLite ``-wal``/``-shm`` files are never copied or treated as authority.
    """

    root = Path(workspace_root).resolve()
    descriptor, descriptor_file_sha256 = _load_workspace_descriptor(root)
    manifest_path = root / descriptor["manifest"]
    try:
        manifest = ArtifactManifest.load(manifest_path, verify_sidecar=True)
    except Exception as exc:
        raise ChildSnapshotIntegrityError(
            f"cannot verify child manifest: {exc}"
        ) from exc
    manifest_file_sha256 = hash_file(manifest_path)
    if manifest.metadata.get("run_id") != descriptor["run_id"]:
        raise ChildSnapshotIntegrityError("child manifest run_id mismatch")
    if manifest.metadata.get("protocol_hash") != descriptor["protocol_hash"]:
        raise ChildSnapshotIntegrityError("child manifest protocol_hash mismatch")
    attempt_store_path = (
        root / "evaluation_attempts.sqlite3"
        if immutable_attempt_store is None
        else Path(immutable_attempt_store).resolve()
    )
    if require_manifest_binding:
        if immutable_attempt_store is None:
            raise ChildSnapshotIntegrityError(
                "immutable attempt store must be supplied for manifest-bound import"
            )
        matches = [
            record
            for record in manifest.artifacts
            if (root / record.location).resolve() == attempt_store_path
        ]
        if len(matches) != 1:
            raise ChildSnapshotIntegrityError(
                "immutable attempt store is not uniquely bound by child manifest"
            )
        record = matches[0]
        if (
            not attempt_store_path.is_file()
            or attempt_store_path.stat().st_size != record.size_bytes
            or hash_file(attempt_store_path) != record.sha256
        ):
            raise ChildSnapshotIntegrityError(
                "immutable attempt-store artifact hash binding changed"
            )
    logical, logical_hash = _read_attempt_logical_payload(
        attempt_store_path, immutable=immutable_attempt_store is not None
    )
    _validate_child_attempt_rows(logical)
    metadata = {row["key"]: row["value"] for row in logical["attempt_metadata"]}
    expected_binding = {
        "run_id": descriptor["run_id"],
        "protocol_hash": descriptor["protocol_hash"],
        "workspace_descriptor_hash": descriptor["descriptor_hash"],
    }
    for name, expected in expected_binding.items():
        if metadata.get(name) != expected:
            raise ChildSnapshotIntegrityError(f"child attempt metadata {name} mismatch")
    core = {
        "schema_version": CHILD_ATTEMPT_SNAPSHOT_SCHEMA,
        "run_id": descriptor["run_id"],
        "protocol_hash": descriptor["protocol_hash"],
        "workspace_descriptor_hash": descriptor["descriptor_hash"],
        "workspace_descriptor_file_sha256": descriptor_file_sha256,
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_content_sha256": manifest.content_hash,
        "attempt_store": {
            "schema_version": CHILD_ATTEMPT_LOGICAL_STATE_SCHEMA,
            "logical_state_sha256": logical_hash,
            "attempt_metadata": logical["attempt_metadata"],
            "evaluation_attempts": logical["evaluation_attempts"],
            "evaluation_checkpoints": logical["evaluation_checkpoints"],
            "attempt_count": len(logical["evaluation_attempts"]),
            "checkpoint_count": len(logical["evaluation_checkpoints"]),
            "status_counts": dict(
                sorted(
                    Counter(
                        row["status"] for row in logical["evaluation_attempts"]
                    ).items()
                )
            ),
        },
    }
    return ChildAttemptSnapshot.from_core_dict(core)


@dataclass(frozen=True, slots=True)
class CampaignReservation:
    global_attempt_id: int
    idempotency_key: str
    candidate_id: str
    generation: int
    child_run_id: str
    worker_id: str
    input_hash: str
    evaluator_hash: str
    status: CampaignAttemptStatus
    reserved_at: str
    updated_at: str
    child_workspace_descriptor_hash: str | None
    child_attempt_id: int | None
    failure_code: str | None


@dataclass(frozen=True, slots=True)
class CampaignAttemptUsage:
    max_attempts: int
    consumed_attempts: int
    remaining_attempts: int
    global_reservations: int
    unbound_imported_attempts: int
    imported_snapshots: int
    status_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "consumed_attempts": self.consumed_attempts,
            "remaining_attempts": self.remaining_attempts,
            "global_reservations": self.global_reservations,
            "unbound_imported_attempts": self.unbound_imported_attempts,
            "imported_snapshots": self.imported_snapshots,
            "status_counts": dict(sorted(self.status_counts.items())),
        }


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    snapshot_sha256: str
    run_id: str
    attempt_count: int
    inserted_attempts: int
    updated_attempts: int
    attached_attempts: int
    idempotent_snapshot: bool
    usage: CampaignAttemptUsage


@dataclass(frozen=True, slots=True)
class CampaignLedgerSealReceipt:
    """Externally pinned identity of a closed, immutable campaign ledger."""

    campaign_id: str
    protocol_hash: str
    max_attempts: int
    sealed_at: str
    logical_head_sha256: str
    database_file_sha256: str
    usage_sha256: str
    content_sha256: str
    file_sha256: str | None = None
    schema_version: str = CAMPAIGN_LEDGER_SEAL_RECEIPT_SCHEMA

    def _core_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "protocol_hash": self.protocol_hash,
            "max_attempts": self.max_attempts,
            "sealed_at": self.sealed_at,
            "logical_head_sha256": self.logical_head_sha256,
            "database_file_sha256": self.database_file_sha256,
            "usage_sha256": self.usage_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._core_dict(), "content_sha256": self.content_sha256}

    def __post_init__(self) -> None:
        if self.schema_version != CAMPAIGN_LEDGER_SEAL_RECEIPT_SCHEMA:
            raise CampaignAttemptConflict("unsupported campaign ledger seal receipt")
        _require_code(self.campaign_id, "seal campaign_id")
        _require_digest(self.protocol_hash, "seal protocol_hash")
        _positive_int(self.max_attempts, "seal max_attempts")
        try:
            timestamp = datetime.fromisoformat(self.sealed_at)
        except (TypeError, ValueError) as exc:
            raise CampaignAttemptConflict("seal timestamp is invalid") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise CampaignAttemptConflict("seal timestamp must be timezone-aware")
        for name in (
            "logical_head_sha256",
            "database_file_sha256",
            "usage_sha256",
            "content_sha256",
        ):
            _require_digest(getattr(self, name), name)
        if self.file_sha256 is not None:
            _require_digest(self.file_sha256, "seal file_sha256")
        if hash_json(self._core_dict()) != self.content_sha256:
            raise CampaignAttemptConflict("campaign ledger seal content hash mismatch")

    @classmethod
    def build(
        cls,
        *,
        campaign_id: str,
        protocol_hash: str,
        max_attempts: int,
        sealed_at: str,
        logical_head_sha256: str,
        database_file_sha256: str,
        usage_sha256: str,
    ) -> "CampaignLedgerSealReceipt":
        core = {
            "schema_version": CAMPAIGN_LEDGER_SEAL_RECEIPT_SCHEMA,
            "campaign_id": campaign_id,
            "protocol_hash": protocol_hash,
            "max_attempts": max_attempts,
            "sealed_at": sealed_at,
            "logical_head_sha256": logical_head_sha256,
            "database_file_sha256": database_file_sha256,
            "usage_sha256": usage_sha256,
        }
        return cls(
            **{key: value for key, value in core.items() if key != "schema_version"},
            content_sha256=hash_json(core),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CampaignLedgerSealReceipt":
        expected = {
            "schema_version",
            "campaign_id",
            "protocol_hash",
            "max_attempts",
            "sealed_at",
            "logical_head_sha256",
            "database_file_sha256",
            "usage_sha256",
            "content_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise CampaignAttemptConflict(
                "campaign ledger seal receipt schema is invalid"
            )
        return cls(**dict(value))


def _child_status(row: Mapping[str, Any]) -> CampaignAttemptStatus:
    if row["status"] == "failed" and row.get("failure_code") == "keyboard_interrupt":
        return CampaignAttemptStatus.INTERRUPTED
    return CampaignAttemptStatus(row["status"])


def _checkpoint_rows_for(
    snapshot: ChildAttemptSnapshot, attempt_id: int
) -> list[dict[str, Any]]:
    return [
        dict(row) for row in snapshot.checkpoints if row["attempt_id"] == attempt_id
    ]


def _validate_attempt_evolution(
    old_attempt: Mapping[str, Any],
    old_checkpoints: Sequence[Mapping[str, Any]],
    new_attempt: Mapping[str, Any],
    new_checkpoints: Sequence[Mapping[str, Any]],
) -> None:
    for name in _ATTEMPT_IMMUTABLE_FIELDS:
        if old_attempt.get(name) != new_attempt.get(name):
            raise ChildSnapshotIntegrityError(
                f"child attempt immutable field changed: {name}"
            )
    old_status = old_attempt["status"]
    new_status = new_attempt["status"]
    if old_status in _TERMINAL_CHILD_STATUSES:
        if dict(old_attempt) != dict(new_attempt) or [
            dict(row) for row in old_checkpoints
        ] != [dict(row) for row in new_checkpoints]:
            raise ChildSnapshotIntegrityError("terminal child attempt was mutated")
        return
    allowed = {
        # A campaign snapshot need not observe every intermediate local-store
        # state.  Therefore a later snapshot may validly prove the transitive
        # leased -> running -> succeeded transition in one reconciliation.
        "leased": {"leased", "running", "succeeded", "failed", "expired"},
        "running": {"running", "succeeded", "failed", "expired"},
    }
    if new_status not in allowed[old_status]:
        raise ChildSnapshotIntegrityError(
            f"invalid child attempt transition: {old_status}->{new_status}"
        )
    if len(new_checkpoints) < len(old_checkpoints):
        raise ChildSnapshotIntegrityError("child checkpoint history was truncated")
    if [dict(row) for row in new_checkpoints[: len(old_checkpoints)]] != [
        dict(row) for row in old_checkpoints
    ]:
        raise ChildSnapshotIntegrityError("frozen child checkpoint was mutated")


class CampaignAttemptLedger:
    """Campaign-wide, crash-conservative attempt budget ledger.

    A global reservation consumes budget before child execution starts.  Child
    snapshots may later be attached to that reservation; unbound imported child
    attempts consume independently.  Thus a process crash can over-count until a
    proven attachment is reconciled, but it can never allow work past the cap.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        campaign_id: str,
        protocol_hash: str,
        max_attempts: int = DEFAULT_CAMPAIGN_ATTEMPT_BUDGET,
        parent_checkpoint_hash: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.campaign_id = _require_code(campaign_id, "campaign_id")
        self.protocol_hash = _require_digest(protocol_hash, "protocol_hash")
        self.max_attempts = _positive_int(max_attempts, "max_attempts")
        if parent_checkpoint_hash is not None:
            _require_digest(parent_checkpoint_hash, "parent_checkpoint_hash")
        self.connection = sqlite3.connect(self.path, timeout=10.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA busy_timeout = 10000")
        self._closed = False
        self._initialize(parent_checkpoint_hash)

    def close(self) -> None:
        if not self._closed:
            self.connection.close()
            self._closed = True

    def __enter__(self) -> "CampaignAttemptLedger":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        if self.connection.in_transaction:
            raise CampaignAttemptLedgerError(
                "nested ledger transactions are not supported"
            )
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            yield self.connection
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def _initialize(self, requested_parent_hash: str | None) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS ledger_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS campaign_reservations (
                global_attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                idempotency_key TEXT NOT NULL UNIQUE,
                candidate_id TEXT NOT NULL,
                generation INTEGER NOT NULL CHECK(generation >= 0),
                child_run_id TEXT NOT NULL,
                worker_id TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                evaluator_hash TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN (
                    'reserved','leased','running','succeeded','failed','expired','interrupted','orphaned'
                )),
                reserved_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                child_workspace_descriptor_hash TEXT,
                child_attempt_id INTEGER,
                failure_code TEXT,
                UNIQUE(child_workspace_descriptor_hash, child_attempt_id)
            );
            CREATE TABLE IF NOT EXISTS child_snapshots (
                snapshot_sha256 TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                protocol_hash TEXT NOT NULL,
                workspace_descriptor_hash TEXT NOT NULL,
                workspace_descriptor_file_sha256 TEXT NOT NULL,
                manifest_file_sha256 TEXT NOT NULL,
                manifest_content_sha256 TEXT NOT NULL,
                attempt_store_logical_state_sha256 TEXT NOT NULL,
                attempt_count INTEGER NOT NULL CHECK(attempt_count >= 0),
                checkpoint_count INTEGER NOT NULL CHECK(checkpoint_count >= 0),
                imported_at TEXT NOT NULL,
                snapshot_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS child_attempts (
                workspace_descriptor_hash TEXT NOT NULL,
                child_attempt_id INTEGER NOT NULL,
                run_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                evaluator_hash TEXT NOT NULL,
                child_status TEXT NOT NULL,
                failure_code TEXT,
                reservation_id INTEGER UNIQUE REFERENCES campaign_reservations(global_attempt_id),
                first_snapshot_sha256 TEXT NOT NULL REFERENCES child_snapshots(snapshot_sha256),
                latest_snapshot_sha256 TEXT NOT NULL REFERENCES child_snapshots(snapshot_sha256),
                attempt_json TEXT NOT NULL,
                checkpoints_json TEXT NOT NULL,
                PRIMARY KEY(workspace_descriptor_hash, child_attempt_id)
            );
            CREATE TABLE IF NOT EXISTS child_snapshot_membership (
                snapshot_sha256 TEXT NOT NULL REFERENCES child_snapshots(snapshot_sha256),
                workspace_descriptor_hash TEXT NOT NULL,
                child_attempt_id INTEGER NOT NULL,
                attempt_sha256 TEXT NOT NULL,
                checkpoints_sha256 TEXT NOT NULL,
                PRIMARY KEY(snapshot_sha256, workspace_descriptor_hash, child_attempt_id)
            );
            """
        )
        existing = dict(
            self.connection.execute("SELECT key,value FROM ledger_metadata")
        )
        if not existing:
            values = {
                "schema_version": CAMPAIGN_ATTEMPT_LEDGER_SCHEMA,
                "campaign_id": self.campaign_id,
                "protocol_hash": self.protocol_hash,
                "max_attempts": str(self.max_attempts),
                "parent_checkpoint_hash": requested_parent_hash or "",
                "lifecycle_state": "open",
                "sealed_at": "",
            }
            self.connection.executemany(
                "INSERT INTO ledger_metadata(key,value) VALUES (?,?)",
                sorted(values.items()),
            )
            self.connection.commit()
            self.parent_checkpoint_hash = requested_parent_hash
            self._sealed_local = False
            return
        required = {
            "schema_version": CAMPAIGN_ATTEMPT_LEDGER_SCHEMA,
            "campaign_id": self.campaign_id,
            "protocol_hash": self.protocol_hash,
            "max_attempts": str(self.max_attempts),
        }
        if any(existing.get(name) != value for name, value in required.items()):
            raise CampaignAttemptLedgerError(
                "ledger binding differs from requested campaign"
            )
        stored_parent = existing.get("parent_checkpoint_hash", "") or None
        lifecycle = existing.get("lifecycle_state", "open")
        if lifecycle == "sealed":
            if (
                requested_parent_hash is not None
                and requested_parent_hash != stored_parent
            ):
                raise CampaignAttemptLedgerSealed(
                    "sealed campaign ledger parent binding is immutable"
                )
            self.parent_checkpoint_hash = stored_parent
            self._sealed_local = True
            return
        if lifecycle != "open":
            raise CampaignAttemptLedgerError("ledger lifecycle metadata is invalid")
        if requested_parent_hash is not None and stored_parent not in (
            None,
            requested_parent_hash,
        ):
            raise CampaignAttemptLedgerError("ledger parent checkpoint binding differs")
        if requested_parent_hash is not None and stored_parent is None:
            self.connection.execute(
                "UPDATE ledger_metadata SET value=? WHERE key='parent_checkpoint_hash'",
                (requested_parent_hash,),
            )
            self.connection.commit()
            stored_parent = requested_parent_hash
        self.parent_checkpoint_hash = stored_parent
        # Backward-compatible, monotonic schema migration for ledgers created
        # before transactional sealing was introduced.
        additions = []
        if "lifecycle_state" not in existing:
            additions.append(("lifecycle_state", "open"))
        if "sealed_at" not in existing:
            additions.append(("sealed_at", ""))
        if additions:
            self.connection.executemany(
                "INSERT INTO ledger_metadata(key,value) VALUES (?,?)", additions
            )
            self.connection.commit()
        self._sealed_local = (
            dict(self.connection.execute("SELECT key,value FROM ledger_metadata")).get(
                "lifecycle_state"
            )
            == "sealed"
        )

    def _assert_mutable_handle(self) -> None:
        if getattr(self, "_sealed_local", False) or self._closed:
            raise CampaignAttemptLedgerSealed("campaign ledger is sealed and immutable")

    def _lifecycle_locked(self, connection: sqlite3.Connection) -> str:
        row = connection.execute(
            "SELECT value FROM ledger_metadata WHERE key='lifecycle_state'"
        ).fetchone()
        if row is None or row[0] not in {"open", "sealed"}:
            raise CampaignAttemptLedgerError(
                "campaign ledger lifecycle metadata is invalid"
            )
        return str(row[0])

    def _assert_open_locked(self, connection: sqlite3.Connection) -> None:
        if self._lifecycle_locked(connection) != "open":
            raise CampaignAttemptLedgerSealed("campaign ledger is sealed and immutable")

    def _consumed_locked(self, connection: sqlite3.Connection) -> int:
        reservations = int(
            connection.execute("SELECT COUNT(*) FROM campaign_reservations").fetchone()[
                0
            ]
        )
        unbound = int(
            connection.execute(
                "SELECT COUNT(*) FROM child_attempts WHERE reservation_id IS NULL"
            ).fetchone()[0]
        )
        return reservations + unbound

    def _assert_capacity_locked(
        self, connection: sqlite3.Connection, increment: int = 1
    ) -> None:
        consumed = self._consumed_locked(connection)
        if consumed + increment > self.max_attempts:
            raise CampaignAttemptBudgetExceeded(
                f"campaign attempt budget exhausted: {consumed}+{increment}>{self.max_attempts}"
            )

    def _reservation_from_row(self, row: sqlite3.Row) -> CampaignReservation:
        return CampaignReservation(
            global_attempt_id=int(row["global_attempt_id"]),
            idempotency_key=row["idempotency_key"],
            candidate_id=row["candidate_id"],
            generation=int(row["generation"]),
            child_run_id=row["child_run_id"],
            worker_id=row["worker_id"],
            input_hash=row["input_hash"],
            evaluator_hash=row["evaluator_hash"],
            status=CampaignAttemptStatus(row["status"]),
            reserved_at=row["reserved_at"],
            updated_at=row["updated_at"],
            child_workspace_descriptor_hash=row["child_workspace_descriptor_hash"],
            child_attempt_id=(
                None
                if row["child_attempt_id"] is None
                else int(row["child_attempt_id"])
            ),
            failure_code=row["failure_code"],
        )

    def get_reservation(self, idempotency_key: str) -> CampaignReservation:
        key = _require_code(idempotency_key, "idempotency_key")
        row = self.connection.execute(
            "SELECT * FROM campaign_reservations WHERE idempotency_key=?", (key,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown campaign reservation: {key}")
        return self._reservation_from_row(row)

    def list_reservations(self) -> tuple[CampaignReservation, ...]:
        rows = self.connection.execute(
            "SELECT * FROM campaign_reservations ORDER BY global_attempt_id"
        ).fetchall()
        return tuple(self._reservation_from_row(row) for row in rows)

    def reserve(
        self,
        idempotency_key: str,
        *,
        candidate_id: str,
        generation: int,
        child_run_id: str,
        input_hash: str,
        evaluator_hash: str,
        worker_id: str,
        now: datetime | None = None,
    ) -> CampaignReservation:
        self._assert_mutable_handle()
        key = _require_code(idempotency_key, "idempotency_key")
        candidate = _require_code(candidate_id, "candidate_id")
        run_id = _require_code(child_run_id, "child_run_id")
        worker = _require_code(worker_id, "worker_id")
        _nonnegative_int(generation, "generation")
        input_digest = _require_digest(input_hash, "input_hash")
        evaluator_digest = _require_digest(evaluator_hash, "evaluator_hash")
        timestamp = _iso(now or _utc_now())
        binding = (
            candidate,
            generation,
            run_id,
            worker,
            input_digest,
            evaluator_digest,
        )
        with self._transaction() as connection:
            self._assert_open_locked(connection)
            existing = connection.execute(
                "SELECT * FROM campaign_reservations WHERE idempotency_key=?", (key,)
            ).fetchone()
            if existing is not None:
                actual = tuple(
                    existing[name]
                    for name in (
                        "candidate_id",
                        "generation",
                        "child_run_id",
                        "worker_id",
                        "input_hash",
                        "evaluator_hash",
                    )
                )
                if actual != binding:
                    raise CampaignAttemptConflict("idempotency key binding was changed")
                attempt_id = int(existing["global_attempt_id"])
            else:
                self._assert_capacity_locked(connection)
                cursor = connection.execute(
                    """INSERT INTO campaign_reservations(
                           idempotency_key,candidate_id,generation,child_run_id,worker_id,
                           input_hash,evaluator_hash,status,reserved_at,updated_at,
                           child_workspace_descriptor_hash,child_attempt_id,failure_code
                       ) VALUES (?,?,?,?,?,?,?,'reserved',?,?,NULL,NULL,NULL)""",
                    (
                        key,
                        candidate,
                        generation,
                        run_id,
                        worker,
                        input_digest,
                        evaluator_digest,
                        timestamp,
                        timestamp,
                    ),
                )
                attempt_id = int(cursor.lastrowid)
        row = self.connection.execute(
            "SELECT * FROM campaign_reservations WHERE global_attempt_id=?",
            (attempt_id,),
        ).fetchone()
        return self._reservation_from_row(row)

    def mark_unattached_terminal(
        self,
        idempotency_key: str,
        *,
        status: CampaignAttemptStatus,
        failure_code: str,
        now: datetime | None = None,
    ) -> CampaignReservation:
        self._assert_mutable_handle()
        if status.value not in {
            CampaignAttemptStatus.FAILED.value,
            CampaignAttemptStatus.EXPIRED.value,
            CampaignAttemptStatus.INTERRUPTED.value,
            CampaignAttemptStatus.ORPHANED.value,
        }:
            raise ValueError(
                "unattached terminal status must consume a failed-like outcome"
            )
        key = _require_code(idempotency_key, "idempotency_key")
        code = _require_code(failure_code, "failure_code")
        with self._transaction() as connection:
            self._assert_open_locked(connection)
            row = connection.execute(
                "SELECT * FROM campaign_reservations WHERE idempotency_key=?", (key,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown campaign reservation: {key}")
            if row["child_attempt_id"] is not None:
                raise CampaignAttemptConflict(
                    "attached reservation status comes from child snapshot"
                )
            if row["status"] in _TERMINAL_LEDGER_STATUSES:
                if row["status"] != status.value or row["failure_code"] != code:
                    raise CampaignAttemptConflict(
                        "terminal campaign reservation is immutable"
                    )
            else:
                connection.execute(
                    """UPDATE campaign_reservations SET status=?,failure_code=?,updated_at=?
                       WHERE global_attempt_id=?""",
                    (
                        status.value,
                        code,
                        _iso(now or _utc_now()),
                        row["global_attempt_id"],
                    ),
                )
        return self.get_reservation(key)

    def _bind_parent_checkpoint_locked(
        self, connection: sqlite3.Connection, digest: str
    ) -> None:
        stored = connection.execute(
            "SELECT value FROM ledger_metadata WHERE key='parent_checkpoint_hash'"
        ).fetchone()[0]
        if stored and stored != digest:
            raise CampaignAttemptConflict(
                "campaign ledger is bound to another parent checkpoint"
            )
        if not stored:
            connection.execute(
                "UPDATE ledger_metadata SET value=? WHERE key='parent_checkpoint_hash'",
                (digest,),
            )
            self.parent_checkpoint_hash = digest

    def _attach_locked(
        self,
        connection: sqlite3.Connection,
        *,
        idempotency_key: str,
        workspace_descriptor_hash: str,
        child_attempt_id: int,
        now: datetime,
    ) -> bool:
        reservation = connection.execute(
            "SELECT * FROM campaign_reservations WHERE idempotency_key=?",
            (idempotency_key,),
        ).fetchone()
        if reservation is None:
            raise KeyError(f"unknown campaign reservation: {idempotency_key}")
        child = connection.execute(
            """SELECT * FROM child_attempts
               WHERE workspace_descriptor_hash=? AND child_attempt_id=?""",
            (workspace_descriptor_hash, child_attempt_id),
        ).fetchone()
        if child is None:
            raise CampaignAttemptConflict(
                "child attempt is not present in imported snapshot"
            )
        expected = (
            reservation["child_run_id"],
            reservation["candidate_id"],
            reservation["input_hash"],
            reservation["evaluator_hash"],
        )
        actual = (
            child["run_id"],
            child["candidate_id"],
            child["input_hash"],
            child["evaluator_hash"],
        )
        if actual != expected:
            raise CampaignAttemptConflict(
                "child attempt does not match global reservation binding"
            )
        existing_reservation_id = child["reservation_id"]
        if existing_reservation_id is not None and int(existing_reservation_id) != int(
            reservation["global_attempt_id"]
        ):
            raise CampaignAttemptConflict(
                "child attempt is attached to another reservation"
            )
        old_child_key = (
            reservation["child_workspace_descriptor_hash"],
            reservation["child_attempt_id"],
        )
        new_child_key = (workspace_descriptor_hash, child_attempt_id)
        if old_child_key != (None, None) and old_child_key != new_child_key:
            raise CampaignAttemptConflict(
                "global reservation is attached to another child attempt"
            )
        changed = existing_reservation_id is None or old_child_key == (None, None)
        connection.execute(
            """UPDATE child_attempts SET reservation_id=?
               WHERE workspace_descriptor_hash=? AND child_attempt_id=?""",
            (
                reservation["global_attempt_id"],
                workspace_descriptor_hash,
                child_attempt_id,
            ),
        )
        attempt_payload = json.loads(child["attempt_json"])
        child_status = _child_status(attempt_payload)
        connection.execute(
            """UPDATE campaign_reservations
               SET child_workspace_descriptor_hash=?,child_attempt_id=?,status=?,failure_code=?,updated_at=?
               WHERE global_attempt_id=?""",
            (
                workspace_descriptor_hash,
                child_attempt_id,
                child_status.value,
                child["failure_code"],
                _iso(now),
                reservation["global_attempt_id"],
            ),
        )
        return changed

    def attach_child_attempt(
        self,
        idempotency_key: str,
        *,
        child_snapshot_hash: str,
        child_attempt_id: int,
        now: datetime | None = None,
    ) -> CampaignReservation:
        self._assert_mutable_handle()
        key = _require_code(idempotency_key, "idempotency_key")
        snapshot_hash = _require_digest(child_snapshot_hash, "child_snapshot_hash")
        _positive_int(child_attempt_id, "child_attempt_id")
        with self._transaction() as connection:
            self._assert_open_locked(connection)
            snapshot = connection.execute(
                "SELECT workspace_descriptor_hash FROM child_snapshots WHERE snapshot_sha256=?",
                (snapshot_hash,),
            ).fetchone()
            if snapshot is None:
                raise CampaignAttemptConflict("child snapshot has not been imported")
            member = connection.execute(
                """SELECT 1 FROM child_snapshot_membership
                   WHERE snapshot_sha256=? AND workspace_descriptor_hash=? AND child_attempt_id=?""",
                (
                    snapshot_hash,
                    snapshot["workspace_descriptor_hash"],
                    child_attempt_id,
                ),
            ).fetchone()
            if member is None:
                raise CampaignAttemptConflict(
                    "child attempt is not a member of the named snapshot"
                )
            self._attach_locked(
                connection,
                idempotency_key=key,
                workspace_descriptor_hash=snapshot["workspace_descriptor_hash"],
                child_attempt_id=child_attempt_id,
                now=now or _utc_now(),
            )
        return self.get_reservation(key)

    def _reconcile_locked(
        self,
        connection: sqlite3.Connection,
        snapshot: ChildAttemptSnapshot,
        bindings: Mapping[int, str],
        *,
        now: datetime,
    ) -> tuple[int, int, int, bool]:
        self._assert_open_locked(connection)
        if snapshot.protocol_hash != self.protocol_hash:
            raise ChildSnapshotIntegrityError(
                "child snapshot protocol differs from campaign"
            )
        snapshot_json = canonical_json_bytes(snapshot.to_dict()).decode("utf-8")
        existing_snapshot = connection.execute(
            "SELECT snapshot_json FROM child_snapshots WHERE snapshot_sha256=?",
            (snapshot.snapshot_sha256,),
        ).fetchone()
        idempotent = existing_snapshot is not None
        if (
            existing_snapshot is not None
            and existing_snapshot["snapshot_json"] != snapshot_json
        ):
            raise ChildSnapshotIntegrityError(
                "snapshot hash collision or stored snapshot tampering"
            )
        if existing_snapshot is None:
            connection.execute(
                """INSERT INTO child_snapshots(
                    snapshot_sha256,run_id,protocol_hash,workspace_descriptor_hash,
                    workspace_descriptor_file_sha256,manifest_file_sha256,
                    manifest_content_sha256,attempt_store_logical_state_sha256,
                    attempt_count,checkpoint_count,imported_at,snapshot_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    snapshot.snapshot_sha256,
                    snapshot.run_id,
                    snapshot.protocol_hash,
                    snapshot.workspace_descriptor_hash,
                    snapshot.workspace_descriptor_file_sha256,
                    snapshot.manifest_file_sha256,
                    snapshot.manifest_content_sha256,
                    snapshot.attempt_store_logical_state_sha256,
                    len(snapshot.attempts),
                    len(snapshot.checkpoints),
                    _iso(now),
                    snapshot_json,
                ),
            )
        snapshot_ids = {int(row["attempt_id"]) for row in snapshot.attempts}
        existing_ids = {
            int(row[0])
            for row in connection.execute(
                "SELECT child_attempt_id FROM child_attempts WHERE workspace_descriptor_hash=?",
                (snapshot.workspace_descriptor_hash,),
            ).fetchall()
        }
        if not existing_ids.issubset(snapshot_ids):
            raise ChildSnapshotIntegrityError(
                "new child snapshot truncates prior attempt history"
            )
        unknown_bindings = set(bindings) - snapshot_ids
        if unknown_bindings:
            raise CampaignAttemptConflict(
                f"bindings reference attempts absent from child snapshot: {sorted(unknown_bindings)}"
            )
        inserted = updated = attached = 0
        for attempt in snapshot.attempts:
            attempt_id = int(attempt["attempt_id"])
            checkpoint_rows = _checkpoint_rows_for(snapshot, attempt_id)
            attempt_json = canonical_json_bytes(attempt).decode("utf-8")
            checkpoints_json = canonical_json_bytes(checkpoint_rows).decode("utf-8")
            existing = connection.execute(
                """SELECT * FROM child_attempts
                   WHERE workspace_descriptor_hash=? AND child_attempt_id=?""",
                (snapshot.workspace_descriptor_hash, attempt_id),
            ).fetchone()
            if existing is None:
                if attempt_id not in bindings:
                    self._assert_capacity_locked(connection)
                connection.execute(
                    """INSERT INTO child_attempts(
                        workspace_descriptor_hash,child_attempt_id,run_id,candidate_id,
                        input_hash,evaluator_hash,child_status,failure_code,reservation_id,
                        first_snapshot_sha256,latest_snapshot_sha256,attempt_json,checkpoints_json
                    ) VALUES (?,?,?,?,?,?,?,?,NULL,?,?,?,?)""",
                    (
                        snapshot.workspace_descriptor_hash,
                        attempt_id,
                        snapshot.run_id,
                        attempt["candidate_id"],
                        attempt["input_hash"],
                        attempt["evaluator_hash"],
                        attempt["status"],
                        attempt.get("failure_code"),
                        snapshot.snapshot_sha256,
                        snapshot.snapshot_sha256,
                        attempt_json,
                        checkpoints_json,
                    ),
                )
                inserted += 1
            else:
                old_attempt = json.loads(existing["attempt_json"])
                old_checkpoints = json.loads(existing["checkpoints_json"])
                _validate_attempt_evolution(
                    old_attempt, old_checkpoints, attempt, checkpoint_rows
                )
                if old_attempt != attempt or old_checkpoints != checkpoint_rows:
                    connection.execute(
                        """UPDATE child_attempts
                           SET child_status=?,failure_code=?,latest_snapshot_sha256=?,
                               attempt_json=?,checkpoints_json=?
                           WHERE workspace_descriptor_hash=? AND child_attempt_id=?""",
                        (
                            attempt["status"],
                            attempt.get("failure_code"),
                            snapshot.snapshot_sha256,
                            attempt_json,
                            checkpoints_json,
                            snapshot.workspace_descriptor_hash,
                            attempt_id,
                        ),
                    )
                    if existing["reservation_id"] is not None:
                        mapped = _child_status(attempt)
                        connection.execute(
                            """UPDATE campaign_reservations SET status=?,failure_code=?,updated_at=?
                               WHERE global_attempt_id=?""",
                            (
                                mapped.value,
                                attempt.get("failure_code"),
                                _iso(now),
                                existing["reservation_id"],
                            ),
                        )
                    updated += 1
                elif existing["latest_snapshot_sha256"] != snapshot.snapshot_sha256:
                    # Even an identical logical attempt must point at the most
                    # recently reconciled immutable child snapshot.  Campaign
                    # close can then prove exact bidirectional final-state
                    # membership instead of merely finding an old compatible
                    # row somewhere in history.
                    connection.execute(
                        """UPDATE child_attempts SET latest_snapshot_sha256=?
                           WHERE workspace_descriptor_hash=? AND child_attempt_id=?""",
                        (
                            snapshot.snapshot_sha256,
                            snapshot.workspace_descriptor_hash,
                            attempt_id,
                        ),
                    )
            connection.execute(
                """INSERT OR IGNORE INTO child_snapshot_membership(
                    snapshot_sha256,workspace_descriptor_hash,child_attempt_id,
                    attempt_sha256,checkpoints_sha256
                ) VALUES (?,?,?,?,?)""",
                (
                    snapshot.snapshot_sha256,
                    snapshot.workspace_descriptor_hash,
                    attempt_id,
                    hash_json(attempt),
                    hash_json(checkpoint_rows),
                ),
            )
            if attempt_id in bindings:
                key = _require_code(bindings[attempt_id], "binding idempotency_key")
                if self._attach_locked(
                    connection,
                    idempotency_key=key,
                    workspace_descriptor_hash=snapshot.workspace_descriptor_hash,
                    child_attempt_id=attempt_id,
                    now=now,
                ):
                    attached += 1
        if self._consumed_locked(connection) > self.max_attempts:
            raise CampaignAttemptBudgetExceeded(
                "child reconciliation would exceed campaign budget"
            )
        return inserted, updated, attached, idempotent

    def reconcile_child_snapshot(
        self,
        snapshot: ChildAttemptSnapshot,
        bindings: Mapping[int, str] | None = None,
        *,
        now: datetime | None = None,
    ) -> ReconcileResult:
        self._assert_mutable_handle()
        if not isinstance(snapshot, ChildAttemptSnapshot):
            raise TypeError("snapshot must be a ChildAttemptSnapshot")
        binding_map = dict(bindings or {})
        timestamp = now or _utc_now()
        with self._transaction() as connection:
            inserted, updated, attached, idempotent = self._reconcile_locked(
                connection, snapshot, binding_map, now=timestamp
            )
        return ReconcileResult(
            snapshot_sha256=snapshot.snapshot_sha256,
            run_id=snapshot.run_id,
            attempt_count=len(snapshot.attempts),
            inserted_attempts=inserted,
            updated_attempts=updated,
            attached_attempts=attached,
            idempotent_snapshot=idempotent,
            usage=self.usage(),
        )

    def import_checkpoint(
        self,
        checkpoint: "CampaignAttemptCheckpoint",
        *,
        now: datetime | None = None,
    ) -> tuple[ReconcileResult, ...]:
        self._assert_mutable_handle()
        if checkpoint.campaign_id != self.campaign_id:
            raise CampaignAttemptConflict("parent checkpoint campaign_id mismatch")
        if checkpoint.protocol_hash != self.protocol_hash:
            raise CampaignAttemptConflict("parent checkpoint protocol_hash mismatch")
        if checkpoint.max_attempts != self.max_attempts:
            raise CampaignAttemptConflict("parent checkpoint max_attempts mismatch")
        binding_hash = checkpoint.file_sha256 or checkpoint.content_sha256
        timestamp = now or _utc_now()
        intermediate: list[tuple[ChildAttemptSnapshot, tuple[int, int, int, bool]]] = []
        with self._transaction() as connection:
            self._assert_open_locked(connection)
            self._bind_parent_checkpoint_locked(connection, binding_hash)
            for snapshot in checkpoint.child_snapshots:
                intermediate.append(
                    (
                        snapshot,
                        self._reconcile_locked(connection, snapshot, {}, now=timestamp),
                    )
                )
            imported_keys = {
                (snapshot.workspace_descriptor_hash, int(attempt["attempt_id"]))
                for snapshot in checkpoint.child_snapshots
                for attempt in snapshot.attempts
            }
            if len(imported_keys) != checkpoint.consumed_attempts:
                raise CampaignAttemptConflict(
                    "parent checkpoint unique attempt count mismatch"
                )
        current_usage = self.usage()
        return tuple(
            ReconcileResult(
                snapshot_sha256=snapshot.snapshot_sha256,
                run_id=snapshot.run_id,
                attempt_count=len(snapshot.attempts),
                inserted_attempts=counts[0],
                updated_attempts=counts[1],
                attached_attempts=counts[2],
                idempotent_snapshot=counts[3],
                usage=current_usage,
            )
            for snapshot, counts in intermediate
        )

    def usage(self) -> CampaignAttemptUsage:
        global_rows = self.connection.execute(
            "SELECT status,COUNT(*) AS count FROM campaign_reservations GROUP BY status"
        ).fetchall()
        child_rows = self.connection.execute(
            """SELECT CASE WHEN child_status='failed' AND failure_code='keyboard_interrupt'
                            THEN 'interrupted' ELSE child_status END AS status,
                      COUNT(*) AS count
               FROM child_attempts WHERE reservation_id IS NULL GROUP BY status"""
        ).fetchall()
        status_counts: Counter[str] = Counter()
        for row in (*global_rows, *child_rows):
            status_counts[row["status"]] += int(row["count"])
        global_count = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM campaign_reservations"
            ).fetchone()[0]
        )
        unbound = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM child_attempts WHERE reservation_id IS NULL"
            ).fetchone()[0]
        )
        snapshots = int(
            self.connection.execute("SELECT COUNT(*) FROM child_snapshots").fetchone()[
                0
            ]
        )
        consumed = global_count + unbound
        return CampaignAttemptUsage(
            max_attempts=self.max_attempts,
            consumed_attempts=consumed,
            remaining_attempts=self.max_attempts - consumed,
            global_reservations=global_count,
            unbound_imported_attempts=unbound,
            imported_snapshots=snapshots,
            status_counts=dict(sorted(status_counts.items())),
        )

    def logical_snapshot(self) -> dict[str, Any]:
        metadata = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM ledger_metadata ORDER BY key"
            ).fetchall()
        ]
        reservations = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM campaign_reservations ORDER BY global_attempt_id"
            ).fetchall()
        ]
        snapshots = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM child_snapshots ORDER BY snapshot_sha256"
            ).fetchall()
        ]
        child_attempts = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM child_attempts ORDER BY workspace_descriptor_hash,child_attempt_id"
            ).fetchall()
        ]
        membership = [
            dict(row)
            for row in self.connection.execute(
                """SELECT * FROM child_snapshot_membership
                   ORDER BY snapshot_sha256,workspace_descriptor_hash,child_attempt_id"""
            ).fetchall()
        ]
        core = {
            "schema_version": CAMPAIGN_ATTEMPT_LEDGER_SCHEMA,
            "ledger_metadata": metadata,
            "campaign_reservations": reservations,
            "child_snapshots": snapshots,
            "child_attempts": child_attempts,
            "child_snapshot_membership": membership,
        }
        return {
            "schema_version": CAMPAIGN_ATTEMPT_LEDGER_SCHEMA,
            "sha256": hash_json(core),
            "usage": self.usage().to_dict(),
            "logical_state": core,
        }

    def logical_snapshot_hash(self) -> str:
        return self.logical_snapshot()["sha256"]

    def _assert_sealable_locked(self, connection: sqlite3.Connection) -> None:
        self._assert_open_locked(connection)
        active_reservations = connection.execute(
            """SELECT global_attempt_id,status FROM campaign_reservations
               WHERE status NOT IN ('succeeded','failed','expired','interrupted','orphaned')"""
        ).fetchall()
        active_children = connection.execute(
            """SELECT workspace_descriptor_hash,child_attempt_id,child_status
               FROM child_attempts WHERE child_status NOT IN ('succeeded','failed','expired')"""
        ).fetchall()
        if active_reservations or active_children:
            raise CampaignAttemptConflict(
                "campaign ledger has active or nonterminal attempts"
            )

        reservations = connection.execute(
            "SELECT * FROM campaign_reservations ORDER BY global_attempt_id"
        ).fetchall()
        children = connection.execute(
            "SELECT * FROM child_attempts ORDER BY workspace_descriptor_hash,child_attempt_id"
        ).fetchall()
        child_by_key = {
            (row["workspace_descriptor_hash"], int(row["child_attempt_id"])): row
            for row in children
        }
        for row in reservations:
            descriptor = row["child_workspace_descriptor_hash"]
            child_attempt_id = row["child_attempt_id"]
            if descriptor is None and child_attempt_id is None:
                if not row["failure_code"]:
                    raise CampaignAttemptConflict(
                        "unattached terminal reservation lacks a failure code"
                    )
                continue
            if descriptor is None or child_attempt_id is None:
                raise CampaignAttemptConflict("reservation has a partial child binding")
            child = child_by_key.get((descriptor, int(child_attempt_id)))
            if child is None or child["reservation_id"] != row["global_attempt_id"]:
                raise CampaignAttemptConflict(
                    "reservation/child binding is not symmetric"
                )
            attempt = json.loads(child["attempt_json"])
            expected_status = _child_status(attempt).value
            if row["status"] != expected_status:
                raise CampaignAttemptConflict(
                    "reservation status differs from child attempt"
                )

        latest_by_descriptor: dict[str, str] = {}
        for row in children:
            descriptor = str(row["workspace_descriptor_hash"])
            latest = str(row["latest_snapshot_sha256"])
            previous = latest_by_descriptor.setdefault(descriptor, latest)
            if previous != latest:
                raise CampaignAttemptConflict(
                    "child attempts do not share one final reconciled snapshot"
                )
            attempt = json.loads(row["attempt_json"])
            checkpoints = json.loads(row["checkpoints_json"])
            if row["child_status"] != attempt.get("status"):
                raise CampaignAttemptConflict(
                    "child_status differs from attempt_json.status"
                )
            member = connection.execute(
                """SELECT attempt_sha256,checkpoints_sha256
                   FROM child_snapshot_membership
                   WHERE snapshot_sha256=? AND workspace_descriptor_hash=?
                     AND child_attempt_id=?""",
                (latest, descriptor, row["child_attempt_id"]),
            ).fetchone()
            if member is None:
                raise CampaignAttemptConflict(
                    "final child snapshot membership is missing"
                )
            if member["attempt_sha256"] != hash_json(attempt) or member[
                "checkpoints_sha256"
            ] != hash_json(checkpoints):
                raise CampaignAttemptConflict(
                    "final child snapshot membership hash mismatch"
                )
        for descriptor, latest in latest_by_descriptor.items():
            member_ids = {
                int(row[0])
                for row in connection.execute(
                    """SELECT child_attempt_id FROM child_snapshot_membership
                       WHERE snapshot_sha256=? AND workspace_descriptor_hash=?""",
                    (latest, descriptor),
                ).fetchall()
            }
            child_ids = {
                int(row["child_attempt_id"])
                for row in children
                if row["workspace_descriptor_hash"] == descriptor
            }
            if member_ids != child_ids:
                raise CampaignAttemptConflict(
                    "final child snapshot membership differs from child attempts"
                )
        if self._consumed_locked(connection) != self.usage().consumed_attempts:
            raise CampaignAttemptConflict("campaign attempt usage is inconsistent")

    def seal_and_close(
        self,
        receipt_path: str | Path,
        *,
        now: datetime | None = None,
    ) -> CampaignLedgerSealReceipt:
        """Transactionally seal the ledger, close it, and pin its final file.

        A crash after the database commit but before receipt replacement leaves
        a sealed database without a valid external receipt.  Preflight rejects
        that state, which is deliberately safer than silently reopening it.
        """

        self._assert_mutable_handle()
        target = Path(receipt_path).resolve()
        receipt_sidecar = target.with_suffix(target.suffix + ".sha256")
        ledger = self.path.resolve()
        protected = {
            ledger,
            Path(str(ledger) + "-wal"),
            Path(str(ledger) + "-shm"),
        }
        if target in protected or receipt_sidecar in protected:
            raise CampaignAttemptConflict(
                "campaign ledger seal receipt collides with ledger storage"
            )
        sealed_at = _iso(now or _utc_now())
        with self._transaction() as connection:
            self._assert_sealable_locked(connection)
            connection.execute(
                "UPDATE ledger_metadata SET value='sealed' WHERE key='lifecycle_state'"
            )
            connection.execute(
                "UPDATE ledger_metadata SET value=? WHERE key='sealed_at'", (sealed_at,)
            )
        self._sealed_local = True
        logical = self.logical_snapshot()
        usage = logical["usage"]
        checkpoint = self.connection.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        ).fetchone()
        if checkpoint is None or int(checkpoint[0]) != 0:
            raise CampaignAttemptLedgerError(
                "cannot checkpoint campaign ledger before seal"
            )
        self.close()
        database_digest = hash_file(self.path)
        receipt = CampaignLedgerSealReceipt.build(
            campaign_id=self.campaign_id,
            protocol_hash=self.protocol_hash,
            max_attempts=self.max_attempts,
            sealed_at=sealed_at,
            logical_head_sha256=logical["sha256"],
            database_file_sha256=database_digest,
            usage_sha256=hash_json(usage),
        )
        encoded = canonical_json_bytes(receipt.to_dict()) + b"\n"
        file_digest = _write_hashed_payload(target, encoded)
        return CampaignLedgerSealReceipt(
            campaign_id=receipt.campaign_id,
            protocol_hash=receipt.protocol_hash,
            max_attempts=receipt.max_attempts,
            sealed_at=receipt.sealed_at,
            logical_head_sha256=receipt.logical_head_sha256,
            database_file_sha256=receipt.database_file_sha256,
            usage_sha256=receipt.usage_sha256,
            content_sha256=receipt.content_sha256,
            file_sha256=file_digest,
        )

    def verify_integrity(self) -> tuple[str, ...]:
        errors: list[str] = []
        result = self.connection.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            errors.append(f"sqlite_integrity:{result}")
        if self.usage().consumed_attempts > self.max_attempts:
            errors.append("campaign_budget_exceeded")
        for row in self.connection.execute("SELECT * FROM campaign_reservations"):
            for name in ("input_hash", "evaluator_hash"):
                if _SHA256_RE.fullmatch(row[name]) is None:
                    errors.append(f"invalid_{name}:{row['global_attempt_id']}")
        return tuple(errors)


def load_campaign_ledger_seal_receipt(
    path: str | Path,
) -> CampaignLedgerSealReceipt:
    target = Path(path)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    try:
        fields = sidecar.read_text(encoding="utf-8").split()
        expected_file_hash = fields[0]
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, IndexError, json.JSONDecodeError) as exc:
        raise CampaignAttemptConflict(
            "cannot load campaign ledger seal receipt"
        ) from exc
    _require_digest(expected_file_hash, "campaign ledger seal sidecar")
    if hash_file(target) != expected_file_hash:
        raise CampaignAttemptConflict("campaign ledger seal receipt sidecar mismatch")
    receipt = CampaignLedgerSealReceipt.from_dict(value)
    return CampaignLedgerSealReceipt(
        campaign_id=receipt.campaign_id,
        protocol_hash=receipt.protocol_hash,
        max_attempts=receipt.max_attempts,
        sealed_at=receipt.sealed_at,
        logical_head_sha256=receipt.logical_head_sha256,
        database_file_sha256=receipt.database_file_sha256,
        usage_sha256=receipt.usage_sha256,
        content_sha256=receipt.content_sha256,
        file_sha256=expected_file_hash,
    )


def _readonly_ledger_snapshot(connection: sqlite3.Connection) -> dict[str, Any]:
    connection.row_factory = sqlite3.Row
    metadata = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM ledger_metadata ORDER BY key"
        ).fetchall()
    ]
    reservations = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM campaign_reservations ORDER BY global_attempt_id"
        ).fetchall()
    ]
    snapshots = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM child_snapshots ORDER BY snapshot_sha256"
        ).fetchall()
    ]
    children = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM child_attempts ORDER BY workspace_descriptor_hash,child_attempt_id"
        ).fetchall()
    ]
    membership = [
        dict(row)
        for row in connection.execute(
            """SELECT * FROM child_snapshot_membership
               ORDER BY snapshot_sha256,workspace_descriptor_hash,child_attempt_id"""
        ).fetchall()
    ]
    metadata_map = {row["key"]: row["value"] for row in metadata}
    try:
        maximum = int(metadata_map["max_attempts"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CampaignAttemptConflict(
            "sealed ledger maximum attempt metadata is invalid"
        ) from exc
    status_counts: Counter[str] = Counter(str(row["status"]) for row in reservations)
    for row in children:
        if row["reservation_id"] is not None:
            continue
        status = str(row["child_status"])
        if status == "failed" and row["failure_code"] == "keyboard_interrupt":
            status = "interrupted"
        status_counts[status] += 1
    unbound = sum(row["reservation_id"] is None for row in children)
    consumed = len(reservations) + unbound
    usage = CampaignAttemptUsage(
        max_attempts=maximum,
        consumed_attempts=consumed,
        remaining_attempts=maximum - consumed,
        global_reservations=len(reservations),
        unbound_imported_attempts=unbound,
        imported_snapshots=len(snapshots),
        status_counts=dict(sorted(status_counts.items())),
    ).to_dict()
    core = {
        "schema_version": CAMPAIGN_ATTEMPT_LEDGER_SCHEMA,
        "ledger_metadata": metadata,
        "campaign_reservations": reservations,
        "child_snapshots": snapshots,
        "child_attempts": children,
        "child_snapshot_membership": membership,
    }
    return {
        "schema_version": CAMPAIGN_ATTEMPT_LEDGER_SCHEMA,
        "sha256": hash_json(core),
        "usage": usage,
        "logical_state": core,
    }


def load_sealed_campaign_ledger_snapshot(
    ledger_path: str | Path,
    receipt_path: str | Path,
) -> tuple[dict[str, Any], CampaignLedgerSealReceipt]:
    """Read a sealed live ledger without WAL/shm side effects.

    The externally pinned receipt authenticates both the logical head and the
    exact closed SQLite file.  ``immutable=1`` ensures this verification path
    cannot create or update ``-wal``/``-shm`` files.
    """

    ledger = Path(ledger_path).resolve()
    receipt = load_campaign_ledger_seal_receipt(receipt_path)
    if not ledger.is_file() or hash_file(ledger) != receipt.database_file_sha256:
        raise CampaignAttemptConflict("sealed campaign ledger file hash mismatch")
    uri = f"{ledger.as_uri()}?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        snapshot = _readonly_ledger_snapshot(connection)
    except sqlite3.Error as exc:
        raise CampaignAttemptConflict("cannot read sealed campaign ledger") from exc
    finally:
        if "connection" in locals():
            connection.close()
    if integrity != "ok":
        raise CampaignAttemptConflict(
            f"sealed campaign ledger integrity failed: {integrity}"
        )
    metadata = {
        row["key"]: row["value"] for row in snapshot["logical_state"]["ledger_metadata"]
    }
    required = {
        "schema_version": CAMPAIGN_ATTEMPT_LEDGER_SCHEMA,
        "campaign_id": receipt.campaign_id,
        "protocol_hash": receipt.protocol_hash,
        "max_attempts": str(receipt.max_attempts),
        "lifecycle_state": "sealed",
        "sealed_at": receipt.sealed_at,
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        raise CampaignAttemptConflict(
            "sealed campaign ledger metadata differs from receipt"
        )
    if snapshot["sha256"] != receipt.logical_head_sha256:
        raise CampaignAttemptConflict("sealed campaign ledger logical head mismatch")
    if hash_json(snapshot["usage"]) != receipt.usage_sha256:
        raise CampaignAttemptConflict("sealed campaign ledger usage hash mismatch")
    return snapshot, receipt


@dataclass(frozen=True, slots=True)
class CampaignAttemptCheckpoint:
    checkpoint_id: str
    campaign_id: str
    protocol_hash: str
    max_attempts: int
    consumed_attempts: int
    remaining_attempts: int
    parent_registry: dict[str, Any]
    child_snapshots: tuple[ChildAttemptSnapshot, ...]
    content_sha256: str
    file_sha256: str | None = None

    def _core_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CAMPAIGN_ATTEMPT_CHECKPOINT_SCHEMA,
            "checkpoint_id": self.checkpoint_id,
            "campaign_id": self.campaign_id,
            "protocol_hash": self.protocol_hash,
            "max_attempts": self.max_attempts,
            "consumed_attempts": self.consumed_attempts,
            "remaining_attempts": self.remaining_attempts,
            "parent_registry": self.parent_registry,
            "child_snapshots": [
                snapshot.to_dict() for snapshot in self.child_snapshots
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        value = self._core_dict()
        value["content_sha256"] = self.content_sha256
        return value

    def __post_init__(self) -> None:
        _require_code(self.checkpoint_id, "checkpoint_id")
        _require_code(self.campaign_id, "campaign_id")
        _require_digest(self.protocol_hash, "protocol_hash")
        _positive_int(self.max_attempts, "max_attempts")
        _nonnegative_int(self.consumed_attempts, "consumed_attempts")
        _nonnegative_int(self.remaining_attempts, "remaining_attempts")
        if self.consumed_attempts + self.remaining_attempts != self.max_attempts:
            raise CampaignAttemptConflict(
                "checkpoint budget arithmetic is inconsistent"
            )
        identities = {
            (snapshot.workspace_descriptor_hash, int(row["attempt_id"]))
            for snapshot in self.child_snapshots
            for row in snapshot.attempts
        }
        if len(identities) != self.consumed_attempts:
            raise CampaignAttemptConflict(
                "checkpoint consumed count differs from unique attempts"
            )
        required_registry = {
            "registry_sha256",
            "source_registry_sha256",
            "wire_registry_sha256",
            "candidate_set_sha256",
            "candidate_count",
            "candidate_ids",
        }
        if set(self.parent_registry) != required_registry:
            raise CampaignAttemptConflict(
                "parent registry checkpoint schema is invalid"
            )
        ids = self.parent_registry["candidate_ids"]
        if not isinstance(ids, list) or ids != sorted(set(ids)):
            raise CampaignAttemptConflict(
                "parent candidate ids must be sorted and unique"
            )
        if self.parent_registry["candidate_count"] != len(ids):
            raise CampaignAttemptConflict("parent candidate count mismatch")
        for name in (
            "registry_sha256",
            "source_registry_sha256",
            "wire_registry_sha256",
            "candidate_set_sha256",
        ):
            _require_digest(self.parent_registry[name], f"parent {name}")
        if hash_json(ids) != self.parent_registry["candidate_set_sha256"]:
            raise CampaignAttemptConflict("parent candidate set hash mismatch")
        if hash_json(self._core_dict()) != self.content_sha256:
            raise CampaignAttemptConflict(
                "campaign attempt checkpoint content hash mismatch"
            )
        if self.file_sha256 is not None:
            _require_digest(self.file_sha256, "file_sha256")

    @classmethod
    def build(
        cls,
        *,
        checkpoint_id: str,
        campaign_id: str,
        protocol_hash: str,
        max_attempts: int,
        parent_registry: Mapping[str, Any],
        child_snapshots: Sequence[ChildAttemptSnapshot],
    ) -> "CampaignAttemptCheckpoint":
        snapshots = tuple(child_snapshots)
        identities = {
            (snapshot.workspace_descriptor_hash, int(row["attempt_id"]))
            for snapshot in snapshots
            for row in snapshot.attempts
        }
        core = {
            "schema_version": CAMPAIGN_ATTEMPT_CHECKPOINT_SCHEMA,
            "checkpoint_id": checkpoint_id,
            "campaign_id": campaign_id,
            "protocol_hash": protocol_hash,
            "max_attempts": max_attempts,
            "consumed_attempts": len(identities),
            "remaining_attempts": max_attempts - len(identities),
            "parent_registry": dict(parent_registry),
            "child_snapshots": [snapshot.to_dict() for snapshot in snapshots],
        }
        return cls(
            checkpoint_id=checkpoint_id,
            campaign_id=campaign_id,
            protocol_hash=protocol_hash,
            max_attempts=max_attempts,
            consumed_attempts=len(identities),
            remaining_attempts=max_attempts - len(identities),
            parent_registry=dict(parent_registry),
            child_snapshots=snapshots,
            content_sha256=hash_json(core),
        )

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, file_sha256: str | None = None
    ) -> "CampaignAttemptCheckpoint":
        expected = {
            "schema_version",
            "checkpoint_id",
            "campaign_id",
            "protocol_hash",
            "max_attempts",
            "consumed_attempts",
            "remaining_attempts",
            "parent_registry",
            "child_snapshots",
            "content_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise CampaignAttemptConflict(
                "campaign attempt checkpoint schema is invalid"
            )
        if value["schema_version"] != CAMPAIGN_ATTEMPT_CHECKPOINT_SCHEMA:
            raise CampaignAttemptConflict(
                "campaign attempt checkpoint version is unsupported"
            )
        return cls(
            checkpoint_id=value["checkpoint_id"],
            campaign_id=value["campaign_id"],
            protocol_hash=value["protocol_hash"],
            max_attempts=value["max_attempts"],
            consumed_attempts=value["consumed_attempts"],
            remaining_attempts=value["remaining_attempts"],
            parent_registry=dict(value["parent_registry"]),
            child_snapshots=tuple(
                ChildAttemptSnapshot.from_dict(item)
                for item in value["child_snapshots"]
            ),
            content_sha256=value["content_sha256"],
            file_sha256=file_sha256,
        )


def write_campaign_attempt_checkpoint(
    checkpoint: CampaignAttemptCheckpoint, path: str | Path
) -> tuple[Path, str]:
    destination = Path(path)
    payload = canonical_json_bytes(checkpoint.to_dict()) + b"\n"
    digest = hash_bytes(payload)
    _atomic_write(destination, payload)
    _atomic_write(
        destination.with_suffix(".sha256"),
        f"{digest}  {destination.name}\n".encode("utf-8"),
    )
    return destination, digest


def load_campaign_attempt_checkpoint(path: str | Path) -> CampaignAttemptCheckpoint:
    source = Path(path)
    sidecar = source.with_suffix(".sha256")
    try:
        expected = sidecar.read_text(encoding="utf-8").split()[0]
        actual = hash_file(source)
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, IndexError, json.JSONDecodeError) as exc:
        raise CampaignAttemptConflict(
            f"cannot load campaign attempt checkpoint: {exc}"
        ) from exc
    if expected != actual:
        raise CampaignAttemptConflict("campaign attempt checkpoint sidecar mismatch")
    return CampaignAttemptCheckpoint.from_dict(value, file_sha256=actual)


__all__ = [
    "CAMPAIGN_ATTEMPT_CHECKPOINT_SCHEMA",
    "CAMPAIGN_ATTEMPT_LEDGER_SCHEMA",
    "CHILD_ATTEMPT_LOGICAL_STATE_SCHEMA",
    "CHILD_ATTEMPT_SNAPSHOT_SCHEMA",
    "DEFAULT_CAMPAIGN_ATTEMPT_BUDGET",
    "CampaignAttemptBudgetExceeded",
    "CampaignAttemptCheckpoint",
    "CampaignAttemptConflict",
    "CampaignAttemptLedger",
    "CampaignAttemptLedgerError",
    "CampaignAttemptStatus",
    "CampaignAttemptUsage",
    "CampaignReservation",
    "ChildAttemptSnapshot",
    "ChildSnapshotIntegrityError",
    "ReconcileResult",
    "import_child_workspace_snapshot",
    "load_campaign_attempt_checkpoint",
    "write_campaign_attempt_checkpoint",
]
