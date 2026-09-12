"""Immutable source capsules for isolated Python research execution."""

from __future__ import annotations

import __main__
import ctypes
import errno
import hashlib
import json
import os
import runpy
import shutil
import stat
import sys
import sysconfig
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Final, NoReturn, SupportsIndex, cast


def _early_canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _early_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label}_invalid")
    return value


def _early_read_regular(path: Path, *, mode: int, maximum_bytes: int) -> bytes:
    absolute = Path(os.path.abspath(path))
    real = Path(os.path.realpath(path))
    if absolute != real:
        raise ValueError(f"symlink_path_forbidden:{path}")
    before = real.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != mode
        or before.st_size > maximum_bytes
    ):
        raise ValueError(f"file_invalid:{path}")
    descriptor = os.open(
        real,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError(f"file_too_large:{path}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = real.lstat()
    identities = {
        (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        for item in (before, opened, after, final)
    }
    if len(identities) != 1:
        raise ValueError(f"file_changed_during_read:{path}")
    return b"".join(chunks)


def _early_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ValueError(f"{label}_invalid")
    return cast(dict[str, object], value)


def _early_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label}_invalid")
    return cast(list[object], value)


def _early_relative_python_path(value: object) -> str:
    if type(value) is not str:
        raise ValueError("source_path_invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or path.suffix != ".py"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"source_path_invalid:{value}")
    return value


def _early_verify_bootstrap_capsule(source_root: Path) -> None:
    if len(sys.argv) < 7 or sys.argv[1] != "--python-runtime-capsule-bootstrap-v1":
        raise ValueError("bootstrap_arguments_invalid")
    if not sys.dont_write_bytecode:
        raise ValueError("bytecode_writes_must_be_disabled")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("bytecode_environment_guard_missing")
    capsule_id = _early_sha256(sys.argv[3], "capsule_id")
    expected_document_sha = _early_sha256(sys.argv[4], "document_sha")
    candidate = _early_relative_python_path(sys.argv[6])
    root = source_root.parent
    for root_directory, expected_mode in ((root, 0o550), (source_root, 0o550)):
        absolute = Path(os.path.abspath(root_directory))
        real = Path(os.path.realpath(root_directory))
        metadata = real.lstat()
        if (
            absolute != real
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != expected_mode
        ):
            raise ValueError(f"directory_invalid:{root_directory}")
    manifest = Path(os.path.abspath(sys.argv[2]))
    if (
        manifest.parent != root
        or root.name != capsule_id
        or manifest.name != f"{capsule_id}.python-runtime-capsule.json"
    ):
        raise ValueError("manifest_location_invalid")
    payload = _early_read_regular(
        manifest,
        mode=0o440,
        maximum_bytes=16 * 1024 * 1024,
    )
    if hashlib.sha256(payload).hexdigest() != expected_document_sha:
        raise ValueError("document_sha_mismatch")
    document = _early_mapping(json.loads(payload.decode("utf-8")), "document")
    if _early_canonical_json_bytes(document) + b"\n" != payload:
        raise ValueError("document_not_canonical")
    if set(document) != {
        "capsule_id",
        "schema_version",
        "source_closure",
        "entrypoints",
        "environment_authority",
        "research_only",
        "production_ready",
    }:
        raise ValueError("capsule_fields_differ")
    identity = dict(document)
    if (
        identity.pop("capsule_id") != capsule_id
        or hashlib.sha256(_early_canonical_json_bytes(identity)).hexdigest()
        != capsule_id
        or document["schema_version"] != "python-runtime-capsule/v1"
        or document["research_only"] is not True
        or document["production_ready"] is not False
    ):
        raise ValueError("capsule_identity_invalid")
    authority = _early_mapping(
        document["environment_authority"],
        "environment_authority",
    )
    if (
        set(authority) != {"path", "authority_id", "document_sha256"}
        or type(authority["path"]) is not str
    ):
        raise ValueError("environment_authority_reference_invalid")
    _early_sha256(authority["authority_id"], "environment_authority_id")
    _early_sha256(authority["document_sha256"], "environment_document_sha")
    closure = _early_mapping(document["source_closure"], "source_closure")
    if set(closure) != {
        "closure_hash",
        "schema_version",
        "root_paths",
        "dynamic_import_bindings",
        "files",
    }:
        raise ValueError("source_closure_fields_differ")
    closure_identity = dict(closure)
    closure_hash = _early_sha256(
        closure_identity.pop("closure_hash"),
        "closure_hash",
    )
    if (
        hashlib.sha256(_early_canonical_json_bytes(closure_identity)).hexdigest()
        != closure_hash
        or closure["schema_version"] != "python-source-closure/v1"
    ):
        raise ValueError("source_closure_identity_invalid")
    roots = _early_list(closure["root_paths"], "root_paths")
    if any(type(item) is not str for item in roots):
        raise ValueError("source_closure_roots_invalid")
    root_values = cast(list[str], roots)
    if (
        root_values != sorted(root_values)
        or len(root_values) != len(set(root_values))
        or "alpha_research/core/python_runtime_capsule.py" not in root_values
        or candidate not in root_values
    ):
        raise ValueError("source_closure_roots_invalid")
    entrypoints = _early_list(document["entrypoints"], "entrypoints")
    if any(type(item) is not str for item in entrypoints):
        raise ValueError("candidate_entrypoint_not_allowed")
    entrypoint_values = cast(list[str], entrypoints)
    if (
        entrypoint_values != sorted(entrypoint_values)
        or len(entrypoint_values) != len(set(entrypoint_values))
        or candidate not in entrypoint_values
    ):
        raise ValueError("candidate_entrypoint_not_allowed")
    evidence_by_path: dict[str, tuple[str, int]] = {}
    for raw in _early_list(closure["files"], "source_files"):
        evidence = _early_mapping(raw, "source_evidence")
        if set(evidence) != {
            "schema_version",
            "project_relative_path",
            "source_sha256",
            "size_bytes",
        }:
            raise ValueError("source_evidence_fields_differ")
        relative = _early_relative_python_path(evidence["project_relative_path"])
        digest = _early_sha256(evidence["source_sha256"], "source_sha")
        size = evidence["size_bytes"]
        if (
            evidence["schema_version"] != "python-source-file-evidence/v1"
            or type(size) is not int
            or size < 0
            or relative in evidence_by_path
        ):
            raise ValueError(f"source_evidence_invalid:{relative}")
        evidence_by_path[relative] = (digest, size)
    if list(evidence_by_path) != sorted(evidence_by_path):
        raise ValueError("source_evidence_order_invalid")
    observed: set[str] = set()
    for directory_name, names, files in os.walk(source_root, followlinks=False):
        current = Path(directory_name)
        current_absolute = Path(os.path.abspath(current))
        current_real = Path(os.path.realpath(current))
        current_metadata = current_real.lstat()
        if (
            current_absolute != current_real
            or not stat.S_ISDIR(current_metadata.st_mode)
            or stat.S_IMODE(current_metadata.st_mode) != 0o550
        ):
            raise ValueError(f"source_directory_invalid:{current}")
        for name in names:
            child = current / name
            child_metadata = child.lstat()
            if stat.S_ISLNK(child_metadata.st_mode):
                raise ValueError(f"source_symlink_forbidden:{child}")
        for name in files:
            path = current / name
            relative = path.relative_to(source_root).as_posix()
            expected = evidence_by_path.get(relative)
            if expected is None:
                raise ValueError(f"unexpected_source_file:{relative}")
            content = _early_read_regular(
                path,
                mode=0o440,
                maximum_bytes=16 * 1024 * 1024,
            )
            if (
                len(content) != expected[1]
                or hashlib.sha256(content).hexdigest() != expected[0]
            ):
                raise ValueError(f"source_content_mismatch:{relative}")
            observed.add(relative)
    if observed != set(evidence_by_path):
        raise ValueError("source_scope_mismatch")
    if {item.name for item in root.iterdir()} != {"src", manifest.name}:
        raise ValueError("capsule_root_scope_mismatch")


if __name__ == "__main__":
    if not (
        sys.flags.dont_write_bytecode
        and sys.flags.isolated
        and sys.flags.ignore_environment
        and sys.flags.no_site
        and sys.flags.no_user_site
        and sys.flags.safe_path
    ):
        raise SystemExit("isolated_python_required: launch with -B -I -S")
    _bootstrap_absolute = Path(os.path.abspath(__file__))
    _bootstrap_real = Path(os.path.realpath(__file__))
    if _bootstrap_absolute != _bootstrap_real:
        raise SystemExit("trusted_bootstrap_symlink_forbidden")
    _bootstrap_source_root = _bootstrap_real.parents[2]
    _bootstrap_paths = sysconfig.get_paths()
    _bootstrap_trusted_roots = {
        Path(os.path.realpath(value))
        for name, value in _bootstrap_paths.items()
        if name in {"stdlib", "platstdlib", "purelib", "platlib"} and value
    }
    _bootstrap_zip = (
        Path(sys.base_prefix)
        / "lib"
        / (f"python{sys.version_info.major}{sys.version_info.minor}.zip")
    )
    _bootstrap_trusted_roots.add(Path(os.path.realpath(_bootstrap_zip)))
    _bootstrap_runtime_paths: list[str] = []
    for _bootstrap_item in sys.path:
        if not _bootstrap_item:
            raise SystemExit("empty_python_path_forbidden")
        _bootstrap_item_absolute = Path(os.path.abspath(_bootstrap_item))
        _bootstrap_item_real = Path(os.path.realpath(_bootstrap_item))
        if _bootstrap_item_absolute != _bootstrap_item_real:
            raise SystemExit(f"runtime_path_symlink_forbidden:{_bootstrap_item}")
        if not any(
            _bootstrap_item_real == root or root in _bootstrap_item_real.parents
            for root in _bootstrap_trusted_roots
        ):
            raise SystemExit(f"untrusted_python_path:{_bootstrap_item}")
        _bootstrap_runtime_paths.append(os.fspath(_bootstrap_item_real))
    for _bootstrap_name in ("purelib", "platlib"):
        _bootstrap_value = _bootstrap_paths.get(_bootstrap_name)
        if not _bootstrap_value:
            continue
        _bootstrap_library_absolute = Path(os.path.abspath(_bootstrap_value))
        _bootstrap_library_real = Path(os.path.realpath(_bootstrap_value))
        if _bootstrap_library_absolute != _bootstrap_library_real:
            raise SystemExit(
                f"runtime_path_symlink_forbidden:{_bootstrap_library_absolute}"
            )
        if os.fspath(_bootstrap_library_real) not in _bootstrap_runtime_paths:
            _bootstrap_runtime_paths.append(os.fspath(_bootstrap_library_real))
    sys.path[:] = [
        os.fspath(_bootstrap_source_root),
        *_bootstrap_runtime_paths,
    ]
    try:
        _early_verify_bootstrap_capsule(_bootstrap_source_root)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"bootstrap_preflight_failed:{error}") from error

from alpha_research.core.hashing import (  # noqa: E402
    canonical_json_bytes,
    hash_json,
    require_sha256,
)
from alpha_research.core.immutable_json import (  # noqa: E402
    ImmutableJsonError,
    load_immutable_json_document,
    write_immutable_json_document,
)
from alpha_research.core.python_environment_authority import (  # noqa: E402
    PythonEnvironmentAuthorityError,
    load_python_environment_authority_v2,
    verify_current_python_environment_authority_v2,
)
from alpha_research.core.python_runtime_capsule_execution_protocol import (  # noqa: E402
    PYTHON_RUNTIME_CAPSULE_EXECUTION_MIRROR_MARKER_PATH,
)
from alpha_research.core.source_closure import (  # noqa: E402
    PythonDynamicImportBindingV1,
    PythonSourceClosureV1,
    PythonSourceFileEvidenceV1,
    build_python_source_closure_v1,
)


_SCHEMA: Final[str] = "python-runtime-capsule/v1"
_SUFFIX: Final[str] = ".python-runtime-capsule.json"
_SOURCE_MODE: Final[int] = 0o440
_DIRECTORY_MODE: Final[int] = 0o550
_MAX_SOURCE_BYTES: Final[int] = 16 * 1024 * 1024
_BOOTSTRAP_RELATIVE_PATH: Final[str] = "alpha_research/core/python_runtime_capsule.py"
_BOOTSTRAP_MARKER: Final[str] = "--python-runtime-capsule-bootstrap-v1"
_DARWIN_RENAME_EXCL: Final[int] = 0x00000004
_LINUX_RENAME_NOREPLACE: Final[int] = 1
_AT_FDCWD: Final[int] = -100
_PUBLISHED_CAPSULE_LOAD_ATTEMPTS: Final[int] = 32
_DIRECTORY_OPEN_FLAGS: Final[int] = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


class PythonRuntimeCapsuleError(ValueError):
    """Stable fail-closed error at the runtime-capsule boundary."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}:{self.detail}")


@dataclass(frozen=True, slots=True)
class PythonRuntimeCapsuleV1:
    """Content identity for one exact source closure and runtime authority."""

    source_closure: PythonSourceClosureV1
    entrypoints: tuple[str, ...]
    environment_authority_path: str
    environment_authority_id: str
    environment_authority_document_sha256: str
    research_only: bool = True
    production_ready: bool = False
    capsule_id: str = field(init=False)
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        if type(self.source_closure) is not PythonSourceClosureV1:
            raise TypeError("source_closure must be exact PythonSourceClosureV1")
        entries = tuple(
            sorted(_relative_python_path(item) for item in self.entrypoints)
        )
        if (
            not entries
            or entries != self.entrypoints
            or len(entries) != len(set(entries))
            or not set(entries).issubset(self.source_closure.root_paths)
        ):
            raise ValueError("capsule entrypoints must be sorted unique closure roots")
        authority_path = _absolute(self.environment_authority_path)
        require_sha256(
            self.environment_authority_id,
            name="environment authority id",
        )
        require_sha256(
            self.environment_authority_document_sha256,
            name="environment authority document sha256",
        )
        if (
            self.schema_version != _SCHEMA
            or self.research_only is not True
            or self.production_ready is not False
        ):
            raise ValueError("runtime capsule assurance boundary differs")
        object.__setattr__(self, "entrypoints", entries)
        object.__setattr__(
            self, "environment_authority_path", os.fspath(authority_path)
        )
        object.__setattr__(
            self, "capsule_id", cast(str, hash_json(self.identity_payload()))
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_closure": self.source_closure.to_dict(),
            "entrypoints": list(self.entrypoints),
            "environment_authority": {
                "path": self.environment_authority_path,
                "authority_id": self.environment_authority_id,
                "document_sha256": self.environment_authority_document_sha256,
            },
            "research_only": self.research_only,
            "production_ready": self.production_ready,
        }

    def to_dict(self) -> dict[str, object]:
        return {"capsule_id": self.capsule_id, **self.identity_payload()}

    @property
    def document_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict()) + b"\n").hexdigest()

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PythonRuntimeCapsuleV1":
        expected = {
            "capsule_id",
            "schema_version",
            "source_closure",
            "entrypoints",
            "environment_authority",
            "research_only",
            "production_ready",
        }
        if set(value) != expected:
            raise ValueError("runtime capsule fields differ")
        authority = _mapping(value["environment_authority"], "environment_authority")
        if set(authority) != {"path", "authority_id", "document_sha256"}:
            raise ValueError("environment authority reference fields differ")
        result = cls(
            source_closure=_source_closure(value["source_closure"]),
            entrypoints=_strings(value["entrypoints"], "entrypoints"),
            environment_authority_path=_text(authority["path"], "authority path"),
            environment_authority_id=_text(authority["authority_id"], "authority id"),
            environment_authority_document_sha256=_text(
                authority["document_sha256"], "authority document sha256"
            ),
            research_only=_boolean(value["research_only"], "research_only"),
            production_ready=_boolean(value["production_ready"], "production_ready"),
            schema_version=_text(value["schema_version"], "schema_version"),
        )
        if value["capsule_id"] != result.capsule_id:
            raise ValueError("runtime capsule self hash differs")
        return result


_CAPABILITY_CONSTRUCTION_KEY: Final[object] = object()


class PythonRuntimeCapsuleCandidateCapabilityV1:
    """Opaque, process-local proof of one active capsule candidate run.

    This is a research-process misuse barrier, not a security boundary against
    malicious code running as the same operating-system user.
    """

    __slots__ = ()

    def __new__(
        cls,
        construction_key: object = None,
    ) -> "PythonRuntimeCapsuleCandidateCapabilityV1":
        if construction_key is not _CAPABILITY_CONSTRUCTION_KEY:
            raise PythonRuntimeCapsuleError(
                "candidate_capability_construction_forbidden",
                cls.__name__,
            )
        return super().__new__(cls)

    def __repr__(self) -> str:
        return "<PythonRuntimeCapsuleCandidateCapabilityV1 opaque>"

    def __reduce__(self) -> NoReturn:
        raise TypeError("candidate runtime capability cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("candidate runtime capability cannot be serialized")

    def __copy__(self) -> NoReturn:
        raise TypeError("candidate runtime capability cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("candidate runtime capability cannot be copied")


@dataclass(frozen=True, slots=True)
class _CandidateExecutionContextV1:
    manifest_path: str
    capsule_id: str
    manifest_document_sha256: str
    environment_project_root: str
    candidate_entrypoint: str
    process_id: int
    nonce: str


_ACTIVE_CANDIDATE_CONTEXT: _CandidateExecutionContextV1 | None = None
_CONSUMED_CANDIDATE_NONCE: str | None = None
_CONSUMED_CANDIDATE_CONTEXT: _CandidateExecutionContextV1 | None = None
_ACTIVE_CANDIDATE_CAPABILITY: PythonRuntimeCapsuleCandidateCapabilityV1 | None = None


def build_python_runtime_capsule_v1(
    project_root: str | Path,
    output_root: str | Path,
    *,
    source_closure: PythonSourceClosureV1,
    entrypoints: Sequence[str],
    environment_authority_path: str | Path,
    environment_project_root: str | Path,
    expected_environment_authority_id: str,
    expected_environment_authority_document_sha256: str,
) -> tuple[PythonRuntimeCapsuleV1, Path]:
    """Copy one exact closure into a no-clobber, read-only capsule."""

    if type(source_closure) is not PythonSourceClosureV1:
        raise TypeError("source_closure must be exact PythonSourceClosureV1")
    source_root = _directory(project_root, mode=None)
    if _BOOTSTRAP_RELATIVE_PATH not in source_closure.root_paths:
        raise PythonRuntimeCapsuleError(
            "trusted_bootstrap_not_closure_root",
            _BOOTSTRAP_RELATIVE_PATH,
        )
    source_evidence = {
        item.project_relative_path: item for item in source_closure.files
    }
    local_project_root = _directory(Path(__file__).resolve().parents[2], mode=None)
    trusted_bootstrap_closure = build_python_source_closure_v1(
        local_project_root,
        root_paths=(_BOOTSTRAP_RELATIVE_PATH,),
    )
    for trusted in trusted_bootstrap_closure.files:
        if source_evidence.get(trusted.project_relative_path) != trusted:
            raise PythonRuntimeCapsuleError(
                "trusted_bootstrap_source_mismatch",
                trusted.project_relative_path,
            )
    authority_path = _file(environment_authority_path, mode=_SOURCE_MODE)
    authority = verify_current_python_environment_authority_v2(
        authority_path,
        project_root=environment_project_root,
        expected_authority_id=expected_environment_authority_id,
        expected_authority_document_sha256=(
            expected_environment_authority_document_sha256
        ),
    )
    capsule = PythonRuntimeCapsuleV1(
        source_closure=source_closure,
        entrypoints=tuple(entrypoints),
        environment_authority_path=os.fspath(authority_path),
        environment_authority_id=authority.authority_id,
        environment_authority_document_sha256=(
            expected_environment_authority_document_sha256
        ),
    )
    return _publish_python_runtime_capsule_copy_v1(
        capsule,
        source_root=source_root,
        output_root=output_root,
    )


def load_python_runtime_capsule_provenance_v1(
    manifest_path: str | Path,
    *,
    expected_capsule_id: str,
    expected_document_sha256: str,
) -> PythonRuntimeCapsuleV1:
    """Authenticate archived capsule bytes without treating them as executable.

    A desktop host may restore owner-only write permission on an archive
    directory.  Provenance loading therefore permits owner-controlled,
    non-group/world-writable directories, but still requires an externally
    pinned canonical manifest, exact source hashes, regular 0440 files, no
    symlinks, and an authenticated environment authority.  Callers must never
    execute from this storage tree; use an execution mirror instead.
    """

    capsule, manifest = _load_python_runtime_capsule_manifest_v1(
        manifest_path,
        expected_capsule_id=expected_capsule_id,
        expected_document_sha256=expected_document_sha256,
    )
    root = _private_capsule_storage_directory(manifest.parent)
    if root.name != capsule.capsule_id:
        raise PythonRuntimeCapsuleError("capsule_root_name_mismatch", root.name)
    _load_capsule_environment_authority_v1(capsule)
    _verify_capsule_tree(
        root,
        capsule,
        directory_mode=None,
        require_private_owner=True,
    )
    return capsule


def materialize_python_runtime_capsule_execution_mirror_v1(
    manifest_path: str | Path,
    *,
    expected_capsule_id: str,
    expected_document_sha256: str,
    execution_root: str | Path | None = None,
) -> tuple[PythonRuntimeCapsuleV1, Path]:
    """Publish and strictly reverify an immutable execution-only capsule copy."""

    capsule = load_python_runtime_capsule_provenance_v1(
        manifest_path,
        expected_capsule_id=expected_capsule_id,
        expected_document_sha256=expected_document_sha256,
    )
    manifest = _file(manifest_path, mode=_SOURCE_MODE)
    target_root = _absolute(
        default_python_runtime_capsule_execution_root_v1()
        if execution_root is None
        else execution_root
    )
    storage_root = manifest.parent
    if _paths_physically_overlap_before_create(target_root, storage_root):
        raise PythonRuntimeCapsuleError(
            "execution_root_overlaps_capsule_storage",
            f"{target_root}:{storage_root}",
        )
    execution_parent = _private_execution_output_directory(target_root)
    mirrored, mirrored_manifest = _publish_python_runtime_capsule_copy_v1(
        capsule,
        source_root=manifest.parent / "src",
        output_root=execution_parent,
    )
    if mirrored != capsule:
        raise PythonRuntimeCapsuleError(
            "execution_mirror_identity_mismatch",
            capsule.capsule_id,
        )
    return mirrored, mirrored_manifest


def default_python_runtime_capsule_execution_root_v1() -> Path:
    """Return the local, user-private cache used only for capsule execution."""

    temporary_root = Path(os.path.realpath(tempfile.gettempdir()))
    return temporary_root / (
        f"alpha-research-python-runtime-capsules-v1-{os.geteuid()}"
    )


def require_python_runtime_capsule_execution_mirror_support_v1(
    capsule: PythonRuntimeCapsuleV1,
) -> None:
    """Reject a legacy capsule before a nested worker attempt can be consumed."""

    if type(capsule) is not PythonRuntimeCapsuleV1:
        raise TypeError("capsule must be exact PythonRuntimeCapsuleV1")
    paths = {
        evidence.project_relative_path for evidence in capsule.source_closure.files
    }
    if PYTHON_RUNTIME_CAPSULE_EXECUTION_MIRROR_MARKER_PATH not in paths:
        raise PythonRuntimeCapsuleError(
            "execution_mirror_protocol_unsupported",
            capsule.capsule_id,
        )
    marker_evidence = next(
        evidence
        for evidence in capsule.source_closure.files
        if evidence.project_relative_path
        == PYTHON_RUNTIME_CAPSULE_EXECUTION_MIRROR_MARKER_PATH
    )
    local_root = _directory(Path(__file__).resolve().parents[2], mode=None)
    marker_path = _file(
        local_root / PYTHON_RUNTIME_CAPSULE_EXECUTION_MIRROR_MARKER_PATH,
        mode=None,
    )
    marker_payload = _read_stable(marker_path, maximum_bytes=_MAX_SOURCE_BYTES)
    if (
        len(marker_payload) != marker_evidence.size_bytes
        or hashlib.sha256(marker_payload).hexdigest() != marker_evidence.source_sha256
    ):
        raise PythonRuntimeCapsuleError(
            "execution_mirror_protocol_marker_mismatch",
            capsule.capsule_id,
        )


def _publish_python_runtime_capsule_copy_v1(
    capsule: PythonRuntimeCapsuleV1,
    *,
    source_root: str | Path,
    output_root: str | Path,
) -> tuple[PythonRuntimeCapsuleV1, Path]:
    """Copy authenticated closure bytes into one atomic content address."""

    source = _directory(source_root, mode=None)
    parent = _output_directory(output_root)
    root = parent / capsule.capsule_id
    manifest = root / f"{capsule.capsule_id}{_SUFFIX}"
    if root.exists() or root.is_symlink():
        loaded = _load_concurrently_published_capsule(root, capsule)
        if loaded != capsule:
            raise PythonRuntimeCapsuleError("capsule_content_conflict", os.fspath(root))
        return loaded, manifest
    staging = Path(tempfile.mkdtemp(prefix=f".{capsule.capsule_id}.", dir=parent))
    staging_published = False
    staging_descriptor: int | None = None
    try:
        source_output = staging / "src"
        source_output.mkdir(mode=0o750)
        for evidence in capsule.source_closure.files:
            payload = _read_exact_source(source, evidence)
            target = source_output / evidence.project_relative_path
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
            _write_exclusive(target, payload)
        write_immutable_json_document(
            staging,
            filename=manifest.name,
            value=capsule.to_dict(),
        )
        for directory, _, _ in os.walk(source_output, topdown=False):
            os.chmod(directory, _DIRECTORY_MODE)
        # macOS renamex_np rejects moving a non-owner-writable directory.
        # The unpublished root stays 0750 only until the no-replace rename;
        # every source directory and file is already final-mode and the
        # winner is synchronously finalized and strictly reloaded below.
        os.chmod(staging, 0o750)
        _verify_capsule_tree(staging, capsule)
        staging_descriptor = _open_owned_unfinalized_capsule_directory(staging)
        try:
            _rename_directory_no_replace(staging, root)
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            loaded = _load_concurrently_published_capsule(root, capsule)
            if loaded != capsule:
                raise PythonRuntimeCapsuleError(
                    "capsule_content_conflict", os.fspath(root)
                ) from None
            return loaded, manifest
        staging_published = True
        loaded = _finalize_owned_published_capsule(
            root,
            capsule,
            descriptor=staging_descriptor,
        )
        if loaded != capsule:
            raise PythonRuntimeCapsuleError("capsule_content_conflict", os.fspath(root))
    finally:
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        if (
            not staging_published
            and staging.exists()
            and not staging.is_symlink()
        ):
            os.chmod(staging, 0o750)
            for directory, _, _ in os.walk(staging):
                os.chmod(directory, 0o750)
            shutil.rmtree(staging)
    return capsule, manifest


def _open_owned_unfinalized_capsule_directory(path: Path) -> int:
    """Open the publisher-owned staging inode before it enters the namespace."""

    try:
        descriptor = os.open(path, _DIRECTORY_OPEN_FLAGS)
    except OSError as error:
        raise PythonRuntimeCapsuleError(
            "capsule_staging_directory_invalid",
            os.fspath(path),
        ) from error
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o750
    ):
        os.close(descriptor)
        raise PythonRuntimeCapsuleError(
            "capsule_staging_directory_invalid",
            os.fspath(path),
        )
    return descriptor


def _finalize_owned_published_capsule(
    root: Path,
    expected: PythonRuntimeCapsuleV1,
    *,
    descriptor: int,
) -> PythonRuntimeCapsuleV1:
    """Finalize only the exact staging inode published by this rename winner."""

    path = _absolute(root)
    opened = os.fstat(descriptor)
    linked = path.lstat()
    if (
        not stat.S_ISDIR(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o750
        or stat.S_ISLNK(linked.st_mode)
        or not stat.S_ISDIR(linked.st_mode)
        or (linked.st_dev, linked.st_ino) != (opened.st_dev, opened.st_ino)
        or linked.st_uid != opened.st_uid
        or stat.S_IMODE(linked.st_mode) != 0o750
    ):
        raise PythonRuntimeCapsuleError(
            "capsule_published_identity_mismatch",
            os.fspath(path),
        )
    os.fchmod(descriptor, _DIRECTORY_MODE)
    finalized = os.fstat(descriptor)
    linked_final = path.lstat()
    if (
        not stat.S_ISDIR(finalized.st_mode)
        or (finalized.st_dev, finalized.st_ino) != (opened.st_dev, opened.st_ino)
        or finalized.st_uid != opened.st_uid
        or stat.S_IMODE(finalized.st_mode) != _DIRECTORY_MODE
        or stat.S_ISLNK(linked_final.st_mode)
        or not stat.S_ISDIR(linked_final.st_mode)
        or (linked_final.st_dev, linked_final.st_ino)
        != (opened.st_dev, opened.st_ino)
        or linked_final.st_uid != opened.st_uid
        or stat.S_IMODE(linked_final.st_mode) != _DIRECTORY_MODE
    ):
        raise PythonRuntimeCapsuleError(
            "capsule_published_identity_mismatch",
            os.fspath(path),
        )
    manifest = path / f"{expected.capsule_id}{_SUFFIX}"
    return load_python_runtime_capsule_v1(
        manifest,
        expected_capsule_id=expected.capsule_id,
        expected_document_sha256=expected.document_sha256,
    )


def _load_concurrently_published_capsule(
    root: Path,
    expected: PythonRuntimeCapsuleV1,
) -> PythonRuntimeCapsuleV1:
    """Load a winner's final root without ever mutating a loser-observed path."""

    manifest = root / f"{expected.capsule_id}{_SUFFIX}"
    last_error: PythonRuntimeCapsuleError | None = None
    for attempt in range(_PUBLISHED_CAPSULE_LOAD_ATTEMPTS):
        try:
            return load_python_runtime_capsule_v1(
                manifest,
                expected_capsule_id=expected.capsule_id,
                expected_document_sha256=expected.document_sha256,
            )
        except PythonRuntimeCapsuleError as error:
            last_error = error
            if not _capsule_publication_transition_may_be_in_progress(root, error):
                raise
            if attempt + 1 < _PUBLISHED_CAPSULE_LOAD_ATTEMPTS:
                os.sched_yield()
    if last_error is None:
        raise AssertionError("published capsule load attempted zero times")
    raise last_error


def _capsule_publication_transition_may_be_in_progress(
    root: Path,
    error: PythonRuntimeCapsuleError,
) -> bool:
    try:
        metadata = _absolute(root).lstat()
    except (OSError, PythonRuntimeCapsuleError, ValueError):
        return False
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
    ):
        return False
    mode = stat.S_IMODE(metadata.st_mode)
    if mode == 0o750:
        return error.code in {"capsule_manifest_invalid", "directory_invalid"}
    return mode == _DIRECTORY_MODE and (
        error.code == "directory_invalid"
        or (
            error.code == "capsule_manifest_invalid"
            and any(
                code in error.detail
                for code in {
                    "document_changed_while_reading",
                    "document_invalid",
                }
            )
        )
    )


def load_python_runtime_capsule_v1(
    manifest_path: str | Path,
    *,
    expected_capsule_id: str,
    expected_document_sha256: str,
) -> PythonRuntimeCapsuleV1:
    """Authenticate a capsule manifest, authority reference, and exact tree."""

    capsule, manifest = _load_python_runtime_capsule_manifest_v1(
        manifest_path,
        expected_capsule_id=expected_capsule_id,
        expected_document_sha256=expected_document_sha256,
    )
    root = _directory(manifest.parent, mode=_DIRECTORY_MODE)
    if root.name != capsule.capsule_id:
        raise PythonRuntimeCapsuleError("capsule_root_name_mismatch", root.name)
    _load_capsule_environment_authority_v1(capsule)
    _verify_capsule_tree(root, capsule)
    return capsule


def _load_python_runtime_capsule_manifest_v1(
    manifest_path: str | Path,
    *,
    expected_capsule_id: str,
    expected_document_sha256: str,
) -> tuple[PythonRuntimeCapsuleV1, Path]:
    capsule_id = require_sha256(expected_capsule_id, name="expected capsule id")
    try:
        value = load_immutable_json_document(
            manifest_path,
            expected_filename=f"{capsule_id}{_SUFFIX}",
            expected_document_sha256=expected_document_sha256,
            maximum_bytes=16 * 1024 * 1024,
        )
        capsule = PythonRuntimeCapsuleV1.from_mapping(value)
    except (ImmutableJsonError, TypeError, ValueError) as error:
        raise PythonRuntimeCapsuleError(
            "capsule_manifest_invalid",
            f"{type(error).__name__}:{error}",
        ) from error
    if capsule.capsule_id != capsule_id:
        raise PythonRuntimeCapsuleError("capsule_id_mismatch", capsule.capsule_id)
    manifest = _file(manifest_path, mode=_SOURCE_MODE)
    return capsule, manifest


def _load_capsule_environment_authority_v1(
    capsule: PythonRuntimeCapsuleV1,
) -> None:
    try:
        load_python_environment_authority_v2(
            capsule.environment_authority_path,
            expected_authority_id=capsule.environment_authority_id,
            expected_document_sha256=(capsule.environment_authority_document_sha256),
        )
    except PythonEnvironmentAuthorityError as error:
        raise PythonRuntimeCapsuleError(
            "capsule_environment_authority_invalid", error.code
        ) from error


def verify_current_python_runtime_capsule_v1(
    manifest_path: str | Path,
    *,
    expected_capsule_id: str,
    expected_document_sha256: str,
    environment_project_root: str | Path,
) -> PythonRuntimeCapsuleV1:
    """Verify the trusted bootstrap before any candidate entrypoint executes."""

    capsule = load_python_runtime_capsule_v1(
        manifest_path,
        expected_capsule_id=expected_capsule_id,
        expected_document_sha256=expected_document_sha256,
    )
    manifest = _file(manifest_path, mode=_SOURCE_MODE)
    root = _directory(manifest.parent, mode=_DIRECTORY_MODE)
    source_root = root / "src"
    main_file = getattr(__main__, "__file__", None)
    if type(main_file) is not str or not main_file:
        raise PythonRuntimeCapsuleError("bootstrap_path_unavailable", repr(main_file))
    bootstrap = _file(main_file, mode=_SOURCE_MODE)
    argv_bootstrap = _file(sys.argv[0], mode=_SOURCE_MODE)
    if not os.path.samefile(bootstrap, argv_bootstrap):
        raise PythonRuntimeCapsuleError(
            "bootstrap_argv_path_mismatch", os.fspath(bootstrap)
        )
    try:
        relative = bootstrap.relative_to(source_root).as_posix()
    except ValueError as error:
        raise PythonRuntimeCapsuleError(
            "bootstrap_outside_capsule", os.fspath(bootstrap)
        ) from error
    if (
        relative != _BOOTSTRAP_RELATIVE_PATH
        or relative not in capsule.source_closure.root_paths
    ):
        raise PythonRuntimeCapsuleError("trusted_bootstrap_not_allowed", relative)
    if not (
        sys.flags.dont_write_bytecode
        and sys.flags.isolated
        and sys.flags.ignore_environment
        and sys.flags.no_site
        and sys.flags.no_user_site
        and sys.flags.safe_path
    ):
        raise PythonRuntimeCapsuleError("isolated_python_required", repr(sys.flags))
    _require_python_runtime_capsule_bytecode_environment_v1()
    if "sitecustomize" in sys.modules or "usercustomize" in sys.modules:
        raise PythonRuntimeCapsuleError(
            "customized_site_forbidden", "sitecustomize_or_usercustomize"
        )
    entries = tuple(_absolute_path_entry(item) for item in sys.path)
    if not entries or entries[0] != source_root or len(entries) != len(set(entries)):
        raise PythonRuntimeCapsuleError(
            "capsule_python_path_head_invalid",
            repr(tuple(os.fspath(item) for item in entries)),
        )
    trusted = _interpreter_library_roots()
    for item in entries[1:]:
        if not _within(item, trusted):
            raise PythonRuntimeCapsuleError("untrusted_python_path", os.fspath(item))
    for name, module in tuple(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        if type(origin) is not str or not origin:
            continue
        path = _absolute_runtime_origin(origin)
        if not (path == bootstrap or _within(path, (source_root, *trusted))):
            raise PythonRuntimeCapsuleError("untrusted_loaded_module", f"{name}:{path}")
    try:
        verify_current_python_environment_authority_v2(
            capsule.environment_authority_path,
            project_root=environment_project_root,
            expected_authority_id=capsule.environment_authority_id,
            expected_authority_document_sha256=(
                capsule.environment_authority_document_sha256
            ),
        )
    except PythonEnvironmentAuthorityError as error:
        raise PythonRuntimeCapsuleError(
            "current_python_environment_invalid", error.code
        ) from error
    return capsule


def run_python_runtime_capsule_v1(
    manifest_path: str | Path,
    *,
    expected_capsule_id: str,
    expected_document_sha256: str,
    environment_project_root: str | Path,
    candidate_entrypoint: str,
    candidate_arguments: Sequence[str] = (),
) -> None:
    """Verify first, then execute exactly one allowlisted candidate script."""

    capsule = verify_current_python_runtime_capsule_v1(
        manifest_path,
        expected_capsule_id=expected_capsule_id,
        expected_document_sha256=expected_document_sha256,
        environment_project_root=environment_project_root,
    )
    relative = _relative_python_path(candidate_entrypoint)
    if relative not in capsule.entrypoints:
        raise PythonRuntimeCapsuleError("candidate_entrypoint_not_allowed", relative)
    manifest = _file(manifest_path, mode=_SOURCE_MODE)
    source_root = _directory(manifest.parent / "src", mode=_DIRECTORY_MODE)
    candidate = _file(source_root / relative, mode=_SOURCE_MODE)
    if candidate.relative_to(source_root).as_posix() != relative:
        raise PythonRuntimeCapsuleError("candidate_entrypoint_path_mismatch", relative)
    arguments = _strings(candidate_arguments, "candidate arguments")
    current_module = sys.modules.get(__name__)
    if current_module is None:
        raise PythonRuntimeCapsuleError("bootstrap_module_unavailable", __name__)
    canonical_module_name = "alpha_research.core.python_runtime_capsule"
    previous_module = sys.modules.get(canonical_module_name)
    if previous_module is not None and previous_module is not current_module:
        raise PythonRuntimeCapsuleError(
            "runtime_capsule_module_alias_conflict",
            canonical_module_name,
        )
    global _ACTIVE_CANDIDATE_CAPABILITY
    global _ACTIVE_CANDIDATE_CONTEXT
    global _CONSUMED_CANDIDATE_CONTEXT
    global _CONSUMED_CANDIDATE_NONCE
    if (
        _ACTIVE_CANDIDATE_CONTEXT is not None
        or _CONSUMED_CANDIDATE_NONCE is not None
        or _CONSUMED_CANDIDATE_CONTEXT is not None
        or _ACTIVE_CANDIDATE_CAPABILITY is not None
    ):
        raise PythonRuntimeCapsuleError(
            "candidate_execution_context_already_active",
            relative,
        )
    context = _CandidateExecutionContextV1(
        manifest_path=os.fspath(manifest),
        capsule_id=capsule.capsule_id,
        manifest_document_sha256=expected_document_sha256,
        environment_project_root=os.fspath(
            _directory(environment_project_root, mode=None)
        ),
        candidate_entrypoint=relative,
        process_id=os.getpid(),
        nonce=os.urandom(32).hex(),
    )
    _ACTIVE_CANDIDATE_CONTEXT = context
    sys.modules[canonical_module_name] = current_module
    sys.argv = [os.fspath(candidate), *arguments]
    try:
        runpy.run_path(os.fspath(candidate), run_name="__main__")
    finally:
        consumed = (
            _ACTIVE_CANDIDATE_CONTEXT is None
            and _CONSUMED_CANDIDATE_NONCE == context.nonce
            and _CONSUMED_CANDIDATE_CONTEXT is context
            and _ACTIVE_CANDIDATE_CAPABILITY is not None
        )
        _ACTIVE_CANDIDATE_CONTEXT = None
        _CONSUMED_CANDIDATE_NONCE = None
        _CONSUMED_CANDIDATE_CONTEXT = None
        _ACTIVE_CANDIDATE_CAPABILITY = None
        if previous_module is None:
            sys.modules.pop(canonical_module_name, None)
        else:
            sys.modules[canonical_module_name] = previous_module
        if not consumed:
            raise PythonRuntimeCapsuleError(
                "candidate_execution_context_not_consumed",
                relative,
            )


def consume_python_runtime_capsule_candidate_context_v1(
    candidate_entrypoint_path: str | Path,
) -> PythonRuntimeCapsuleV1:
    """Consume the bootstrap-installed context before candidate-side input opens."""

    global _ACTIVE_CANDIDATE_CAPABILITY
    global _ACTIVE_CANDIDATE_CONTEXT
    global _CONSUMED_CANDIDATE_CONTEXT
    global _CONSUMED_CANDIDATE_NONCE
    context = _ACTIVE_CANDIDATE_CONTEXT
    if (
        context is None
        or _CONSUMED_CANDIDATE_NONCE is not None
        or _CONSUMED_CANDIDATE_CONTEXT is not None
        or _ACTIVE_CANDIDATE_CAPABILITY is not None
    ):
        raise PythonRuntimeCapsuleError(
            "candidate_execution_context_missing",
            os.fspath(candidate_entrypoint_path),
        )
    if context.process_id != os.getpid():
        raise PythonRuntimeCapsuleError(
            "candidate_execution_process_mismatch",
            str(context.process_id),
        )
    capsule = load_python_runtime_capsule_v1(
        context.manifest_path,
        expected_capsule_id=context.capsule_id,
        expected_document_sha256=context.manifest_document_sha256,
    )
    relative = _relative_python_path(context.candidate_entrypoint)
    if relative not in capsule.entrypoints:
        raise PythonRuntimeCapsuleError(
            "candidate_entrypoint_not_allowed",
            relative,
        )
    manifest = _file(context.manifest_path, mode=_SOURCE_MODE)
    source_root = _directory(manifest.parent / "src", mode=_DIRECTORY_MODE)
    expected = _file(source_root / relative, mode=_SOURCE_MODE)
    current = _file(candidate_entrypoint_path, mode=_SOURCE_MODE)
    argv_entrypoint = _file(sys.argv[0], mode=_SOURCE_MODE)
    if not (
        os.path.samefile(current, expected)
        and os.path.samefile(argv_entrypoint, expected)
    ):
        raise PythonRuntimeCapsuleError(
            "candidate_entrypoint_origin_mismatch",
            os.fspath(current),
        )
    if not (
        sys.flags.dont_write_bytecode
        and sys.flags.isolated
        and sys.flags.ignore_environment
        and sys.flags.no_site
        and sys.flags.no_user_site
        and sys.flags.safe_path
    ):
        raise PythonRuntimeCapsuleError("isolated_python_required", repr(sys.flags))
    _require_python_runtime_capsule_bytecode_environment_v1()
    entries = tuple(_absolute_path_entry(item) for item in sys.path)
    if not entries or entries[0] != source_root or len(entries) != len(set(entries)):
        raise PythonRuntimeCapsuleError(
            "capsule_python_path_head_invalid",
            repr(tuple(os.fspath(item) for item in entries)),
        )
    trusted = _interpreter_library_roots()
    for item in entries[1:]:
        if not _within(item, trusted):
            raise PythonRuntimeCapsuleError("untrusted_python_path", os.fspath(item))
    for name, module in tuple(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        if type(origin) is not str or not origin:
            continue
        path = _absolute_runtime_origin(origin)
        if not _within(path, (source_root, *trusted)):
            raise PythonRuntimeCapsuleError("untrusted_loaded_module", f"{name}:{path}")
    try:
        verify_current_python_environment_authority_v2(
            capsule.environment_authority_path,
            project_root=context.environment_project_root,
            expected_authority_id=capsule.environment_authority_id,
            expected_authority_document_sha256=(
                capsule.environment_authority_document_sha256
            ),
        )
    except PythonEnvironmentAuthorityError as error:
        raise PythonRuntimeCapsuleError(
            "current_python_environment_invalid",
            error.code,
        ) from error
    capability = PythonRuntimeCapsuleCandidateCapabilityV1(_CAPABILITY_CONSTRUCTION_KEY)
    _ACTIVE_CANDIDATE_CONTEXT = None
    _CONSUMED_CANDIDATE_NONCE = context.nonce
    _CONSUMED_CANDIDATE_CONTEXT = context
    _ACTIVE_CANDIDATE_CAPABILITY = capability
    return capsule


def get_python_runtime_capsule_candidate_capability_v1(
    candidate_entrypoint_path: str | Path,
) -> PythonRuntimeCapsuleCandidateCapabilityV1:
    """Return the opaque capability for the currently consumed candidate run."""

    context = _CONSUMED_CANDIDATE_CONTEXT
    capability = _ACTIVE_CANDIDATE_CAPABILITY
    if (
        context is None
        or capability is None
        or _ACTIVE_CANDIDATE_CONTEXT is not None
        or _CONSUMED_CANDIDATE_NONCE != context.nonce
    ):
        raise PythonRuntimeCapsuleError(
            "candidate_capability_unavailable",
            os.fspath(candidate_entrypoint_path),
        )
    _verify_candidate_capability_live_origin(
        context,
        candidate_entrypoint_path=candidate_entrypoint_path,
    )
    return capability


def verify_python_runtime_capsule_candidate_capability_v1(
    capability: object,
    *,
    expected_capsule_id: str,
    expected_candidate_entrypoint: str,
) -> PythonRuntimeCapsuleV1:
    """Verify one capability against this live run and its fixed identity."""

    context = _CONSUMED_CANDIDATE_CONTEXT
    active = _ACTIVE_CANDIDATE_CAPABILITY
    if (
        type(capability) is not PythonRuntimeCapsuleCandidateCapabilityV1
        or capability is not active
        or context is None
        or _ACTIVE_CANDIDATE_CONTEXT is not None
        or _CONSUMED_CANDIDATE_NONCE != context.nonce
    ):
        raise PythonRuntimeCapsuleError(
            "candidate_capability_invalid_or_expired",
            type(capability).__name__,
        )
    if context.process_id != os.getpid():
        raise PythonRuntimeCapsuleError(
            "candidate_capability_process_mismatch",
            str(context.process_id),
        )
    capsule_id = require_sha256(
        expected_capsule_id,
        name="expected candidate capability capsule id",
    )
    if capsule_id != context.capsule_id:
        raise PythonRuntimeCapsuleError(
            "candidate_capability_capsule_mismatch",
            capsule_id,
        )
    relative = _relative_python_path(expected_candidate_entrypoint)
    if relative != context.candidate_entrypoint:
        raise PythonRuntimeCapsuleError(
            "candidate_capability_entrypoint_mismatch",
            relative,
        )
    _verify_candidate_capability_live_origin(
        context,
        candidate_entrypoint_path=None,
    )
    return load_python_runtime_capsule_v1(
        context.manifest_path,
        expected_capsule_id=context.capsule_id,
        expected_document_sha256=context.manifest_document_sha256,
    )


def verify_python_runtime_capsule_candidate_execution_v1(
    capability: object,
    *,
    expected_capsule_id: str,
    expected_candidate_entrypoint: str,
) -> tuple[PythonRuntimeCapsuleV1, Path]:
    """Verify a live capability and return its strict execution manifest."""

    capsule = verify_python_runtime_capsule_candidate_capability_v1(
        capability,
        expected_capsule_id=expected_capsule_id,
        expected_candidate_entrypoint=expected_candidate_entrypoint,
    )
    context = _CONSUMED_CANDIDATE_CONTEXT
    if context is None:
        raise PythonRuntimeCapsuleError(
            "candidate_capability_invalid_or_expired",
            type(capability).__name__,
        )
    manifest = _file(context.manifest_path, mode=_SOURCE_MODE)
    if (
        manifest.parent.name != capsule.capsule_id
        or manifest.name != f"{capsule.capsule_id}{_SUFFIX}"
    ):
        raise PythonRuntimeCapsuleError(
            "candidate_execution_manifest_mismatch",
            os.fspath(manifest),
        )
    return capsule, manifest


def _verify_candidate_capability_live_origin(
    context: _CandidateExecutionContextV1,
    *,
    candidate_entrypoint_path: str | Path | None,
) -> None:
    _require_python_runtime_capsule_bytecode_environment_v1()
    if context.process_id != os.getpid():
        raise PythonRuntimeCapsuleError(
            "candidate_capability_process_mismatch",
            str(context.process_id),
        )
    manifest = _file(context.manifest_path, mode=_SOURCE_MODE)
    source_root = _directory(manifest.parent / "src", mode=_DIRECTORY_MODE)
    relative = _relative_python_path(context.candidate_entrypoint)
    expected = _file(source_root / relative, mode=_SOURCE_MODE)
    argv_entrypoint = _file(sys.argv[0], mode=_SOURCE_MODE)
    paths_match = os.path.samefile(argv_entrypoint, expected)
    if candidate_entrypoint_path is not None:
        current = _file(candidate_entrypoint_path, mode=_SOURCE_MODE)
        paths_match = paths_match and os.path.samefile(current, expected)
    if not paths_match:
        raise PythonRuntimeCapsuleError(
            "candidate_capability_entrypoint_origin_mismatch",
            relative,
        )


def python_runtime_capsule_command_v1(
    manifest_path: str | Path,
    *,
    expected_capsule_id: str,
    expected_document_sha256: str,
    environment_project_root: str | Path,
    candidate_entrypoint: str,
    candidate_arguments: Sequence[str] = (),
) -> tuple[str, ...]:
    """Build a shell-free, pinned research-only bootstrap command."""

    capsule = load_python_runtime_capsule_v1(
        manifest_path,
        expected_capsule_id=expected_capsule_id,
        expected_document_sha256=expected_document_sha256,
    )
    relative = _relative_python_path(candidate_entrypoint)
    if relative not in capsule.entrypoints:
        raise PythonRuntimeCapsuleError("candidate_entrypoint_not_allowed", relative)
    manifest = _file(manifest_path, mode=_SOURCE_MODE)
    source_root = _directory(manifest.parent / "src", mode=_DIRECTORY_MODE)
    bootstrap = _file(source_root / _BOOTSTRAP_RELATIVE_PATH, mode=_SOURCE_MODE)
    _file(source_root / relative, mode=_SOURCE_MODE)
    environment_root = _directory(environment_project_root, mode=None)
    try:
        verify_current_python_environment_authority_v2(
            capsule.environment_authority_path,
            project_root=environment_root,
            expected_authority_id=capsule.environment_authority_id,
            expected_authority_document_sha256=(
                capsule.environment_authority_document_sha256
            ),
        )
    except PythonEnvironmentAuthorityError as error:
        raise PythonRuntimeCapsuleError(
            "current_python_environment_invalid",
            error.code,
        ) from error
    executable = _file(Path(os.path.realpath(sys.executable)), mode=None)
    arguments = _strings(candidate_arguments, "candidate arguments")
    return (
        os.fspath(executable),
        "-B",
        "-I",
        "-S",
        os.fspath(bootstrap),
        _BOOTSTRAP_MARKER,
        os.fspath(manifest),
        capsule.capsule_id,
        expected_document_sha256,
        os.fspath(environment_root),
        relative,
        *arguments,
    )


def python_runtime_capsule_environment_v1(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the mandatory environment for a capsule subprocess.

    ``-B`` remains the interpreter-level authority because isolated mode ignores
    Python configuration variables.  Keeping the environment guard as well
    prevents a candidate-spawned descendant from silently re-enabling bytecode
    writes and makes omission at a launcher boundary fail closed.
    """

    environment_source = os.environ if source is None else source
    if any(
        type(key) is not str or type(value) is not str
        for key, value in environment_source.items()
    ):
        raise TypeError("capsule environment must contain exact text keys and values")
    environment = dict(environment_source)
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONPYCACHEPREFIX"):
        environment.pop(name, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _require_python_runtime_capsule_bytecode_environment_v1() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise PythonRuntimeCapsuleError(
            "bytecode_environment_guard_missing",
            "PYTHONDONTWRITEBYTECODE must equal 1",
        )


def _bootstrap_main(arguments: Sequence[str]) -> int:
    if len(arguments) < 7 or arguments[1] != _BOOTSTRAP_MARKER:
        raise PythonRuntimeCapsuleError(
            "bootstrap_arguments_invalid",
            "expected marker, manifest, capsule id, manifest sha, "
            "environment root, and candidate entrypoint",
        )
    run_python_runtime_capsule_v1(
        arguments[2],
        expected_capsule_id=arguments[3],
        expected_document_sha256=arguments[4],
        environment_project_root=arguments[5],
        candidate_entrypoint=arguments[6],
        candidate_arguments=arguments[7:],
    )
    return 0


def _verify_capsule_tree(
    root: Path,
    capsule: PythonRuntimeCapsuleV1,
    *,
    directory_mode: int | None = _DIRECTORY_MODE,
    require_private_owner: bool = False,
) -> None:
    source_root = _capsule_tree_directory(
        root / "src",
        mode=directory_mode,
        require_private_owner=require_private_owner,
    )
    expected = {
        item.project_relative_path: item for item in capsule.source_closure.files
    }
    observed: set[str] = set()
    for directory, names, files in os.walk(source_root, followlinks=False):
        current = _capsule_tree_directory(
            directory,
            mode=directory_mode,
            require_private_owner=require_private_owner,
        )
        for name in names:
            _capsule_tree_directory(
                current / name,
                mode=directory_mode,
                require_private_owner=require_private_owner,
            )
        for name in files:
            path = _file(current / name, mode=_SOURCE_MODE)
            relative = path.relative_to(source_root).as_posix()
            evidence = expected.get(relative)
            if evidence is None:
                raise PythonRuntimeCapsuleError("unexpected_capsule_file", relative)
            payload = _read_stable(path, maximum_bytes=_MAX_SOURCE_BYTES)
            if (
                len(payload) != evidence.size_bytes
                or hashlib.sha256(payload).hexdigest() != evidence.source_sha256
            ):
                raise PythonRuntimeCapsuleError("capsule_source_mismatch", relative)
            observed.add(relative)
    if observed != set(expected):
        raise PythonRuntimeCapsuleError(
            "capsule_source_scope_mismatch",
            ",".join(sorted(set(expected) - observed)),
        )
    allowed_root = {"src", f"{capsule.capsule_id}{_SUFFIX}"}
    if {item.name for item in root.iterdir()} != allowed_root:
        raise PythonRuntimeCapsuleError("capsule_root_scope_mismatch", os.fspath(root))


def _capsule_tree_directory(
    value: str | Path,
    *,
    mode: int | None,
    require_private_owner: bool,
) -> Path:
    path = _directory(value, mode=mode)
    if require_private_owner:
        metadata = path.lstat()
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o022:
            raise PythonRuntimeCapsuleError(
                "capsule_storage_directory_not_private",
                os.fspath(path),
            )
    return path


def _source_closure(value: object) -> PythonSourceClosureV1:
    raw = _mapping(value, "source closure")
    if set(raw) != {
        "closure_hash",
        "schema_version",
        "root_paths",
        "dynamic_import_bindings",
        "files",
    }:
        raise ValueError("source closure fields differ")
    bindings: list[PythonDynamicImportBindingV1] = []
    for item in _mappings(raw["dynamic_import_bindings"], "dynamic bindings"):
        if set(item) != {"import_site_path", "target_paths"}:
            raise ValueError("dynamic import binding fields differ")
        bindings.append(
            PythonDynamicImportBindingV1(
                import_site_path=_text(item["import_site_path"], "import site"),
                target_paths=_strings(item["target_paths"], "target paths"),
            )
        )
    files: list[PythonSourceFileEvidenceV1] = []
    for item in _mappings(raw["files"], "source files"):
        if set(item) != {
            "schema_version",
            "project_relative_path",
            "source_sha256",
            "size_bytes",
        }:
            raise ValueError("source evidence fields differ")
        files.append(
            PythonSourceFileEvidenceV1(
                project_relative_path=_text(
                    item["project_relative_path"], "source path"
                ),
                source_sha256=_text(item["source_sha256"], "source sha256"),
                size_bytes=_integer(item["size_bytes"], "source size"),
                schema_version=_text(item["schema_version"], "source schema"),
            )
        )
    closure = PythonSourceClosureV1(
        root_paths=_strings(raw["root_paths"], "root paths"),
        dynamic_import_bindings=tuple(bindings),
        files=tuple(files),
        schema_version=_text(raw["schema_version"], "closure schema"),
    )
    if raw["closure_hash"] != closure.closure_hash:
        raise ValueError("source closure self hash differs")
    return closure


def _read_exact_source(root: Path, evidence: PythonSourceFileEvidenceV1) -> bytes:
    path = _file(root / evidence.project_relative_path, mode=None)
    if path.resolve() != path or root not in path.parents:
        raise PythonRuntimeCapsuleError("source_path_escape", os.fspath(path))
    payload = _read_stable(path, maximum_bytes=_MAX_SOURCE_BYTES)
    if (
        len(payload) != evidence.size_bytes
        or hashlib.sha256(payload).hexdigest() != evidence.source_sha256
    ):
        raise PythonRuntimeCapsuleError("source_closure_mismatch", os.fspath(path))
    return payload


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        _SOURCE_MODE,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short capsule source write")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, _SOURCE_MODE)
    finally:
        os.close(descriptor)


def _rename_directory_no_replace(source: Path, target: Path) -> None:
    """Atomically publish a complete directory without replacing any target."""

    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    if sys.platform == "darwin":
        library = ctypes.CDLL(None, use_errno=True)
        try:
            rename_exclusive = library.renamex_np
        except AttributeError as error:
            raise PythonRuntimeCapsuleError(
                "atomic_no_replace_unavailable",
                "renamex_np",
            ) from error
        rename_exclusive.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_exclusive.restype = ctypes.c_int
        result = int(
            rename_exclusive(
                source_bytes,
                target_bytes,
                _DARWIN_RENAME_EXCL,
            )
        )
    elif sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        try:
            rename_no_replace = library.renameat2
        except AttributeError as error:
            raise PythonRuntimeCapsuleError(
                "atomic_no_replace_unavailable",
                "renameat2",
            ) from error
        rename_no_replace.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_no_replace.restype = ctypes.c_int
        result = int(
            rename_no_replace(
                _AT_FDCWD,
                source_bytes,
                _AT_FDCWD,
                target_bytes,
                _LINUX_RENAME_NOREPLACE,
            )
        )
    elif os.name == "nt":
        os.rename(source, target)
        return
    else:
        raise PythonRuntimeCapsuleError(
            "atomic_no_replace_unavailable",
            sys.platform,
        )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            os.fspath(target),
        )


def _absolute_path_entry(value: object) -> Path:
    if type(value) is not str or not value:
        raise PythonRuntimeCapsuleError("empty_python_path_forbidden", repr(value))
    expanded = Path(value).expanduser()
    if not expanded.is_absolute():
        raise PythonRuntimeCapsuleError("relative_python_path_forbidden", value)
    absolute = Path(os.path.abspath(expanded))
    real = Path(os.path.realpath(expanded))
    if absolute != real:
        raise PythonRuntimeCapsuleError("symlink_python_path_forbidden", value)
    return real


def _absolute_runtime_origin(value: str) -> Path:
    expanded = Path(value).expanduser()
    if not expanded.is_absolute():
        raise PythonRuntimeCapsuleError("relative_module_origin_forbidden", value)
    absolute = Path(os.path.abspath(expanded))
    real = Path(os.path.realpath(expanded))
    if absolute != real:
        raise PythonRuntimeCapsuleError("symlink_module_origin_forbidden", value)
    if real.exists():
        metadata = real.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise PythonRuntimeCapsuleError("module_origin_invalid", value)
    return real


def _interpreter_library_roots() -> tuple[Path, ...]:
    paths = sysconfig.get_paths()
    standard_library_zip = Path(
        os.path.realpath(
            Path(sys.base_prefix)
            / "lib"
            / f"python{sys.version_info.major}{sys.version_info.minor}.zip"
        )
    )
    candidates = {
        _absolute_path_entry(value)
        for name, value in paths.items()
        if name in {"stdlib", "platstdlib", "purelib", "platlib"} and value
    }
    candidates.add(standard_library_zip)
    roots = tuple(sorted(candidates, key=os.fspath))
    if not roots:
        raise PythonRuntimeCapsuleError("interpreter_library_roots_missing", "")
    return roots


def _within(path: Path, roots: Sequence[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _absolute(value: str | Path) -> Path:
    path = Path(os.path.abspath(os.fspath(Path(value).expanduser())))
    if not path.is_absolute():
        raise ValueError("path must be absolute")
    _reject_symlink_ancestors(path)
    return path


def _reject_symlink_ancestors(path: Path) -> None:
    current = path
    while True:
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            metadata = None
        if metadata is not None and stat.S_ISLNK(metadata.st_mode):
            raise PythonRuntimeCapsuleError(
                "symlink_path_forbidden", os.fspath(current)
            )
        if current.parent == current:
            return
        current = current.parent


def _paths_physically_overlap_before_create(
    candidate: Path,
    existing_root: Path,
) -> bool:
    """Detect lexical and filesystem-alias overlap without creating candidate."""

    if (
        candidate == existing_root
        or candidate in existing_root.parents
        or existing_root in candidate.parents
    ):
        return True
    root = _directory(existing_root, mode=None)
    root_identity = _path_identity(root)
    candidate_ancestors = _existing_ancestor_identities(candidate)
    if root_identity in candidate_ancestors:
        return True
    try:
        candidate_metadata = candidate.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(candidate_metadata.st_mode):
        raise PythonRuntimeCapsuleError(
            "symlink_path_forbidden",
            os.fspath(candidate),
        )
    candidate_identity = (
        candidate_metadata.st_dev,
        candidate_metadata.st_ino,
    )
    return candidate_identity in _existing_ancestor_identities(root)


def _existing_ancestor_identities(path: Path) -> set[tuple[int, int]]:
    identities: set[tuple[int, int]] = set()
    current = path
    while True:
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            metadata = None
        if metadata is not None:
            if stat.S_ISLNK(metadata.st_mode):
                raise PythonRuntimeCapsuleError(
                    "symlink_path_forbidden",
                    os.fspath(current),
                )
            identities.add((metadata.st_dev, metadata.st_ino))
        if current.parent == current:
            return identities
        current = current.parent


def _path_identity(path: Path) -> tuple[int, int]:
    metadata = path.lstat()
    return metadata.st_dev, metadata.st_ino


def _read_stable(path: Path, *, maximum_bytes: int) -> bytes:
    before = path.lstat()
    if before.st_size > maximum_bytes:
        raise PythonRuntimeCapsuleError("source_too_large", os.fspath(path))
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise PythonRuntimeCapsuleError("source_too_large", os.fspath(path))
        after_read = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    identities = {
        (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        for item in (before, opened, after_read, after)
    }
    if len(identities) != 1:
        raise PythonRuntimeCapsuleError("source_changed_during_read", os.fspath(path))
    return b"".join(chunks)


def _directory(value: str | Path, *, mode: int | None) -> Path:
    path = _absolute(value)
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or (mode is not None and stat.S_IMODE(metadata.st_mode) != mode)
    ):
        raise PythonRuntimeCapsuleError("directory_invalid", os.fspath(path))
    return path


def _private_capsule_storage_directory(value: str | Path) -> Path:
    path = _directory(value, mode=None)
    metadata = path.lstat()
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o022:
        raise PythonRuntimeCapsuleError(
            "capsule_storage_directory_not_private",
            os.fspath(path),
        )
    return path


def _output_directory(value: str | Path) -> Path:
    path = _absolute(value)
    path.mkdir(parents=True, exist_ok=True, mode=0o750)
    result = _directory(path, mode=None)
    metadata = result.stat()
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o022:
        raise PythonRuntimeCapsuleError(
            "output_directory_not_private",
            os.fspath(result),
        )
    return result


def _private_execution_output_directory(value: str | Path) -> Path:
    path = _absolute(value)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    result = _directory(path, mode=None)
    metadata = result.lstat()
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise PythonRuntimeCapsuleError(
            "execution_output_directory_not_private",
            os.fspath(result),
        )
    return result


def _file(value: str | Path, *, mode: int | None) -> Path:
    path = _absolute(value)
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or (mode is not None and stat.S_IMODE(metadata.st_mode) != mode)
    ):
        raise PythonRuntimeCapsuleError("file_invalid", os.fspath(path))
    return path


def _relative_python_path(value: object) -> str:
    text = _text(value, "Python source path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or path.suffix != ".py"
        or any(item in {"", ".", ".."} for item in path.parts)
    ):
        raise ValueError("Python source path must be canonical project-relative")
    return text


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return cast(Mapping[str, object], value)


def _mappings(value: object, label: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, Mapping) for item in value
    ):
        raise TypeError(f"{label} must be a list of mappings")
    return tuple(cast(Mapping[str, object], item) for item in value)


def _strings(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(
        type(item) is not str for item in value
    ):
        raise TypeError(f"{label} must be strings")
    return tuple(cast(str, item) for item in value)


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{label} must be nonempty text")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{label} must be a nonnegative integer")
    return value


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a boolean")
    return value


__all__ = [
    "PythonRuntimeCapsuleCandidateCapabilityV1",
    "PythonRuntimeCapsuleError",
    "PythonRuntimeCapsuleV1",
    "build_python_runtime_capsule_v1",
    "consume_python_runtime_capsule_candidate_context_v1",
    "default_python_runtime_capsule_execution_root_v1",
    "get_python_runtime_capsule_candidate_capability_v1",
    "load_python_runtime_capsule_v1",
    "load_python_runtime_capsule_provenance_v1",
    "materialize_python_runtime_capsule_execution_mirror_v1",
    "python_runtime_capsule_command_v1",
    "python_runtime_capsule_environment_v1",
    "require_python_runtime_capsule_execution_mirror_support_v1",
    "run_python_runtime_capsule_v1",
    "verify_python_runtime_capsule_candidate_capability_v1",
    "verify_python_runtime_capsule_candidate_execution_v1",
    "verify_current_python_runtime_capsule_v1",
]


if __name__ == "__main__":
    raise SystemExit(_bootstrap_main(sys.argv))
