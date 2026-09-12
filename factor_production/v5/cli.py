from __future__ import annotations

import argparse
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Sequence

from factor_production.v5.artifacts import (
    ArtifactManifest,
    ArtifactStore,
    canonical_json_bytes,
    hash_json,
)
from factor_production.v5.artifacts.hashing import hash_bytes, hash_file
from factor_production.v5.candidate_protocol_authority import (
    CandidateProtocolAuthority,
)
from factor_production.v5.domain.enums import RunState
from factor_production.v5.orchestration.repository import SQLiteRepository
from factor_production.v5.protocol import (
    default_protocol,
    load_protocol,
    write_protocol,
)


WORKSPACE_SCHEMA = "alpha-mining-workspace/v5"
DESCRIPTOR_NAME = "workspace.json"
CANDIDATE_PROTOCOL_AUTHORITY_NAME = "candidate_protocol_authority.json"


class WorkspaceError(RuntimeError):
    pass


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _descriptor_core(
    *,
    run_id: str,
    protocol_hash: str,
    database: str = "control.sqlite3",
    protocol: str = "protocol.json",
    manifest: str = "manifest.json",
    candidate_protocol_authority: str | None = None,
    candidate_protocol_authority_file_sha256: str | None = None,
    candidate_protocol_authority_content_sha256: str | None = None,
) -> dict[str, str]:
    core = {
        "schema_version": WORKSPACE_SCHEMA,
        "run_id": run_id,
        "protocol_hash": protocol_hash,
        "database": database,
        "protocol": protocol,
        "manifest": manifest,
    }
    authority_values = (
        candidate_protocol_authority,
        candidate_protocol_authority_file_sha256,
        candidate_protocol_authority_content_sha256,
    )
    if any(value is not None for value in authority_values):
        if not all(isinstance(value, str) and value for value in authority_values):
            raise WorkspaceError("candidate protocol descriptor binding is incomplete")
        core.update(
            {
                "candidate_protocol_authority": candidate_protocol_authority,
                "candidate_protocol_authority_file_sha256": (
                    candidate_protocol_authority_file_sha256
                ),
                "candidate_protocol_authority_content_sha256": (
                    candidate_protocol_authority_content_sha256
                ),
            }
        )
    return core


def _write_descriptor(workspace: Path, core: dict[str, str]) -> Path:
    payload = {**core, "descriptor_hash": hash_json(core)}
    path = workspace / DESCRIPTOR_NAME
    _atomic_write(path, canonical_json_bytes(payload) + b"\n")
    return path


def _load_descriptor(workspace: Path) -> dict[str, str]:
    path = workspace / DESCRIPTOR_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkspaceError(
            f"cannot load V5 workspace descriptor {path}: {exc}"
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
    dual_fields = {
        "candidate_protocol_authority",
        "candidate_protocol_authority_file_sha256",
        "candidate_protocol_authority_content_sha256",
    }
    if not isinstance(payload, dict) or frozenset(payload) not in {
        frozenset(expected),
        frozenset(expected | dual_fields),
    }:
        raise WorkspaceError("workspace descriptor has an invalid schema")
    if payload["schema_version"] != WORKSPACE_SCHEMA:
        raise WorkspaceError(
            f"unsupported workspace schema: {payload['schema_version']!r}"
        )
    core = {key: payload[key] for key in set(payload) - {"descriptor_hash"}}
    if payload["descriptor_hash"] != hash_json(core):
        raise WorkspaceError("workspace descriptor content hash mismatch")
    for key in (
        "database",
        "protocol",
        "manifest",
        *(("candidate_protocol_authority",) if dual_fields.issubset(payload) else ()),
    ):
        location = Path(payload[key])
        if location.is_absolute() or ".." in location.parts:
            raise WorkspaceError(f"unsafe workspace {key} path: {payload[key]!r}")
    return payload


def initialize_workspace(
    workspace: str | Path,
    *,
    protocol_path: str | Path | None = None,
    run_id: str | None = None,
    candidate_protocol_authority: CandidateProtocolAuthority | None = None,
) -> dict[str, Any]:
    root = Path(workspace).resolve()
    descriptor_path = root / DESCRIPTOR_NAME
    if descriptor_path.exists():
        raise WorkspaceError(f"V5 workspace is already initialized: {root}")
    root.mkdir(parents=True, exist_ok=True)
    actual_run_id = run_id or f"v5-{uuid.uuid4().hex[:16]}"
    if not actual_run_id.strip() or any(char in actual_run_id for char in "/\\"):
        raise WorkspaceError("run_id must be a non-empty path-safe string")
    protocol = (
        load_protocol(protocol_path)
        if protocol_path is not None
        else default_protocol(actual_run_id)
    )
    if candidate_protocol_authority is not None and (
        candidate_protocol_authority.run_id != actual_run_id
        or candidate_protocol_authority.control_protocol_content_sha256
        != protocol.content_hash
    ):
        raise WorkspaceError("candidate protocol authority does not bind this run")
    protocol_destination = write_protocol(protocol, root / "protocol.json")

    store = ArtifactStore(root)
    protocol_artifact = store.put_file(
        "frozen_protocol",
        protocol_destination,
        media_type="application/json",
        role="protocol",
    )
    authority_artifact = None
    authority_file_sha256 = None
    if candidate_protocol_authority is not None:
        authority_payload = (
            canonical_json_bytes(candidate_protocol_authority.to_dict()) + b"\n"
        )
        authority_destination = root / CANDIDATE_PROTOCOL_AUTHORITY_NAME
        _atomic_write(authority_destination, authority_payload)
        authority_file_sha256 = hash_bytes(authority_payload)
        authority_artifact = store.put_file(
            "frozen_candidate_protocol_authority",
            authority_destination,
            media_type="application/json",
            role="candidate_protocol_authority",
        )
    manifest = ArtifactManifest(
        manifest_id=f"{actual_run_id}-manifest",
        metadata={
            "run_id": actual_run_id,
            "protocol_hash": protocol.content_hash,
            "teacher_feedback_policy": "holdout_only",
            **(
                {}
                if candidate_protocol_authority is None
                else {
                    "candidate_protocol_authority_file_sha256": authority_file_sha256,
                    "candidate_protocol_authority_content_sha256": (
                        candidate_protocol_authority.content_hash
                    ),
                    "candidate_protocol_content_sha256": (
                        candidate_protocol_authority.candidate_protocol_content_sha256
                    ),
                }
            ),
        },
    )
    manifest.add(protocol_artifact)
    if authority_artifact is not None:
        manifest.add(authority_artifact)
    manifest.write(root / "manifest.json")

    with SQLiteRepository(root / "control.sqlite3") as repository:
        repository.create_run(actual_run_id, protocol.content_hash)
        if candidate_protocol_authority is not None:
            repository.authorize_candidate_protocol(
                actual_run_id, candidate_protocol_authority
            )
        repository.add_artifact(actual_run_id, protocol_artifact)
        if authority_artifact is not None:
            repository.add_artifact(actual_run_id, authority_artifact)
        repository.transition_run(
            actual_run_id,
            RunState.INITIALIZED,
            reason="workspace_initialized_and_protocol_frozen",
        )
        status = repository.status(actual_run_id)
    _write_descriptor(
        root,
        _descriptor_core(
            run_id=actual_run_id,
            protocol_hash=protocol.content_hash,
            **(
                {}
                if candidate_protocol_authority is None
                else {
                    "candidate_protocol_authority": CANDIDATE_PROTOCOL_AUTHORITY_NAME,
                    "candidate_protocol_authority_file_sha256": authority_file_sha256,
                    "candidate_protocol_authority_content_sha256": (
                        candidate_protocol_authority.content_hash
                    ),
                }
            ),
        ),
    )
    return {
        "ok": True,
        "workspace": str(root),
        "protocol_hash": protocol.content_hash,
        **(
            {}
            if candidate_protocol_authority is None
            else {
                "candidate_protocol_hash": (
                    candidate_protocol_authority.candidate_protocol_content_sha256
                ),
                "candidate_protocol_authority_content_sha256": (
                    candidate_protocol_authority.content_hash
                ),
            }
        ),
        **status,
    }


def workspace_status(workspace: str | Path) -> dict[str, Any]:
    root = Path(workspace).resolve()
    descriptor = _load_descriptor(root)
    with SQLiteRepository(root / descriptor["database"]) as repository:
        status = repository.status(descriptor["run_id"])
    return {"ok": True, "workspace": str(root), **status}


def verify_workspace(workspace: str | Path) -> dict[str, Any]:
    root = Path(workspace).resolve()
    errors: list[str] = []
    try:
        descriptor = _load_descriptor(root)
    except Exception as exc:
        return {"ok": False, "workspace": str(root), "errors": [f"descriptor:{exc}"]}
    try:
        protocol = load_protocol(root / descriptor["protocol"])
        if protocol.content_hash != descriptor["protocol_hash"]:
            errors.append(
                f"protocol_hash:{descriptor['protocol_hash']}!={protocol.content_hash}"
            )
    except Exception as exc:
        errors.append(f"protocol:{exc}")
    manifest_checked = 0
    manifest = None
    try:
        manifest = ArtifactManifest.load(root / descriptor["manifest"])
        verification = manifest.verify(root)
        manifest_checked = verification.checked
        errors.extend(verification.errors)
        if manifest.metadata.get("protocol_hash") != descriptor["protocol_hash"]:
            errors.append("manifest_protocol_hash_mismatch")
    except Exception as exc:
        errors.append(f"manifest:{exc}")
    try:
        with SQLiteRepository(root / descriptor["database"]) as repository:
            run = repository.get_run(descriptor["run_id"])
            if run.protocol_hash != descriptor["protocol_hash"]:
                errors.append("repository_protocol_hash_mismatch")
            authority = repository.get_candidate_protocol_authority(
                descriptor["run_id"]
            )
            descriptor_has_authority = "candidate_protocol_authority" in descriptor
            if descriptor_has_authority != (authority is not None):
                errors.append("candidate_protocol_authority_presence_mismatch")
            if authority is not None:
                try:
                    authority_path = root / descriptor["candidate_protocol_authority"]
                    authority_file_hash = hash_file(authority_path)
                    authority_value = CandidateProtocolAuthority.from_dict(
                        json.loads(authority_path.read_text(encoding="utf-8"))
                    )
                    if (
                        authority_file_hash
                        != descriptor["candidate_protocol_authority_file_sha256"]
                    ):
                        errors.append("candidate_protocol_authority_file_hash_mismatch")
                    if (
                        authority_value.content_hash
                        != descriptor["candidate_protocol_authority_content_sha256"]
                        or authority_value != authority
                    ):
                        errors.append("candidate_protocol_authority_content_mismatch")
                    if manifest is None or (
                        manifest.metadata.get(
                            "candidate_protocol_authority_file_sha256"
                        )
                        != authority_file_hash
                        or manifest.metadata.get(
                            "candidate_protocol_authority_content_sha256"
                        )
                        != authority.content_hash
                        or manifest.metadata.get("candidate_protocol_content_sha256")
                        != authority.candidate_protocol_content_sha256
                    ):
                        errors.append("manifest_candidate_protocol_authority_mismatch")
                except Exception as exc:
                    errors.append(f"candidate_protocol_authority:{exc}")
            errors.extend(repository.verify_integrity(descriptor["run_id"]))
    except Exception as exc:
        errors.append(f"repository:{exc}")
    return {
        "ok": not errors,
        "workspace": str(root),
        "run_id": descriptor["run_id"],
        "protocol_hash": descriptor["protocol_hash"],
        **(
            {}
            if "candidate_protocol_authority_content_sha256" not in descriptor
            else {
                "candidate_protocol_authority_content_sha256": descriptor[
                    "candidate_protocol_authority_content_sha256"
                ]
            }
        ),
        "manifest_artifacts_checked": manifest_checked,
        "errors": errors,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m factor_production.v5",
        description="V5 alpha-mining research control plane",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser(
        "init", help="initialize and freeze a V5 run workspace"
    )
    init.add_argument("--workspace", required=True)
    init.add_argument("--protocol")
    init.add_argument("--run-id")
    for name in ("status", "verify"):
        command = subparsers.add_parser(name, help=f"{name} a V5 run workspace")
        command.add_argument("--workspace", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            result = initialize_workspace(
                args.workspace,
                protocol_path=args.protocol,
                run_id=args.run_id,
            )
        elif args.command == "status":
            result = workspace_status(args.workspace)
        elif args.command == "verify":
            result = verify_workspace(args.workspace)
        else:  # pragma: no cover - argparse enforces commands
            raise AssertionError(args.command)
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
