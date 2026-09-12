from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, cast

from llm_alpha_mining.mining.artifacts.hashing import (
    canonical_json_bytes,
    hash_bytes,
    hash_file,
    hash_json,
)


MANIFEST_SCHEMA = "artifact-manifest/v5"


class ArtifactError(ValueError):
    pass


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ArtifactError(
            f"artifact location must be a safe relative path: {value!r}"
        )
    return path.as_posix()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    logical_name: str
    location: str
    sha256: str
    size_bytes: int
    media_type: str = "application/octet-stream"
    role: str = "research"

    def __post_init__(self) -> None:
        if not self.logical_name.strip():
            raise ArtifactError("logical_name must not be empty")
        object.__setattr__(self, "location", _safe_relative_path(self.location))
        if len(self.sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.sha256
        ):
            raise ArtifactError("sha256 must be a lowercase 64-character digest")
        if self.size_bytes < 0:
            raise ArtifactError("size_bytes must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_name": self.logical_name,
            "location": self.location,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRecord":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class ManifestVerification:
    ok: bool
    checked: int
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checked": self.checked, "errors": list(self.errors)}


@dataclass(slots=True)
class ArtifactManifest:
    manifest_id: str
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if not self.manifest_id.strip():
            raise ArtifactError("manifest_id must not be empty")
        if self.schema_version != MANIFEST_SCHEMA:
            raise ArtifactError(f"unsupported manifest schema: {self.schema_version!r}")
        self._validate_uniqueness()

    def _validate_uniqueness(self) -> None:
        names = [record.logical_name for record in self.artifacts]
        if len(names) != len(set(names)):
            raise ArtifactError("manifest logical names must be unique")

    def add(self, record: ArtifactRecord) -> None:
        self.artifacts.append(record)
        try:
            self._validate_uniqueness()
        except Exception:
            self.artifacts.pop()
            raise

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "artifacts": [
                record.to_dict()
                for record in sorted(self.artifacts, key=lambda item: item.logical_name)
            ],
            "metadata": self.metadata,
        }

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def write(self, path: str | Path, *, write_sidecar: bool = True) -> Path:
        destination = Path(path)
        payload = canonical_json_bytes(self.to_dict()) + b"\n"
        _atomic_write(destination, payload)
        if write_sidecar:
            digest = hash_bytes(payload)
            _atomic_write(
                destination.with_suffix(destination.suffix + ".sha256"),
                f"{digest}  {destination.name}\n".encode(),
            )
        return destination

    @classmethod
    def load(
        cls, path: str | Path, *, verify_sidecar: bool = True
    ) -> "ArtifactManifest":
        source = Path(path)
        if verify_sidecar:
            sidecar = source.with_suffix(source.suffix + ".sha256")
            if not sidecar.is_file():
                raise ArtifactError(f"manifest hash sidecar is missing: {sidecar}")
            expected = sidecar.read_text(encoding="utf-8").split()[0]
            actual = hash_file(source)
            if actual != expected:
                raise ArtifactError(
                    f"manifest hash mismatch: expected {expected}, got {actual}"
                )
        payload = json.loads(source.read_text(encoding="utf-8"))
        allowed = {"schema_version", "manifest_id", "artifacts", "metadata"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ArtifactError(f"unknown manifest fields: {unknown}")
        return cls(
            schema_version=payload["schema_version"],
            manifest_id=payload["manifest_id"],
            artifacts=[
                ArtifactRecord.from_dict(item) for item in payload.get("artifacts", [])
            ],
            metadata=dict(payload.get("metadata", {})),
        )

    def verify(self, root: str | Path) -> ManifestVerification:
        base = Path(root).resolve()
        errors: list[str] = []
        for record in self.artifacts:
            path = (base / record.location).resolve()
            try:
                path.relative_to(base)
            except ValueError:
                errors.append(f"outside_root:{record.logical_name}:{record.location}")
                continue
            if not path.is_file():
                errors.append(f"missing:{record.logical_name}:{record.location}")
                continue
            size = path.stat().st_size
            if size != record.size_bytes:
                errors.append(
                    f"size_mismatch:{record.logical_name}:{record.size_bytes}!={size}"
                )
            digest = hash_file(path)
            if digest != record.sha256:
                errors.append(
                    f"hash_mismatch:{record.logical_name}:{record.sha256}!={digest}"
                )
        return ManifestVerification(
            ok=not errors, checked=len(self.artifacts), errors=tuple(errors)
        )


class ArtifactStore:
    """Immutable content-addressed object store rooted inside a run workspace."""

    def __init__(
        self, workspace_root: str | Path, *, object_dir: str = "artifacts/objects"
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        normalized_object_dir = _safe_relative_path(object_dir)
        self.object_root = self.workspace_root.joinpath(
            *PurePosixPath(normalized_object_dir).parts
        )
        _reject_symlink_components(self.object_root, root=self.workspace_root)
        self.object_root.mkdir(parents=True, exist_ok=True)
        observed = self.object_root.lstat()
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise ArtifactError(
                f"artifact object root must be a regular directory: {self.object_root}"
            )

    def _destination(self, digest: str) -> Path:
        return self.object_root / digest[:2] / digest

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.workspace_root).as_posix()

    def put_bytes(
        self,
        logical_name: str,
        data: bytes,
        *,
        media_type: str = "application/octet-stream",
        role: str = "research",
    ) -> ArtifactRecord:
        if not isinstance(data, bytes):
            raise TypeError("artifact payload must be bytes")
        digest = hash_bytes(data)
        destination = self._destination(digest)
        _publish_content_addressed_bytes(
            destination,
            data,
            digest=digest,
            root=self.workspace_root,
        )
        return ArtifactRecord(
            logical_name=logical_name,
            location=self._relative(destination),
            sha256=digest,
            size_bytes=len(data),
            media_type=media_type,
            role=role,
        )

    def read_bytes(self, record: ArtifactRecord) -> bytes:
        """Reload and verify one object without following caller-controlled paths."""

        if type(record) is not ArtifactRecord:
            raise TypeError("artifact reload requires an exact ArtifactRecord")
        expected_location = self._relative(self._destination(record.sha256))
        if record.location != expected_location:
            raise ArtifactError(
                "artifact location does not match its content-addressed digest"
            )
        path = self.workspace_root.joinpath(*PurePosixPath(record.location).parts)
        payload = _read_regular_bytes(
            path,
            expected_size=record.size_bytes,
            root=self.workspace_root,
        )
        digest = hash_bytes(payload)
        if digest != record.sha256:
            raise ArtifactError(
                f"artifact hash mismatch: expected {record.sha256}, got {digest}"
            )
        return payload

    def put_json(
        self, logical_name: str, value: Any, *, role: str = "research"
    ) -> ArtifactRecord:
        return self.put_bytes(
            logical_name,
            canonical_json_bytes(value) + b"\n",
            media_type="application/json",
            role=role,
        )

    def put_file(
        self,
        logical_name: str,
        source: str | Path,
        *,
        media_type: str = "application/octet-stream",
        role: str = "research",
    ) -> ArtifactRecord:
        staging, digest, size_bytes = _stage_regular_source(
            source,
            staging_root=self.object_root,
            root=self.workspace_root,
        )
        destination = self._destination(digest)
        try:
            _publish_content_addressed_staging(
                destination,
                staging,
                digest=digest,
                expected_size=size_bytes,
                root=self.workspace_root,
            )
        finally:
            staging.unlink(missing_ok=True)
        return ArtifactRecord(
            logical_name=logical_name,
            location=self._relative(destination),
            sha256=digest,
            size_bytes=size_bytes,
            media_type=media_type,
            role=role,
        )


def _publish_content_addressed_bytes(
    destination: Path,
    data: bytes,
    *,
    digest: str,
    root: Path,
) -> None:
    """Atomically link a complete staging inode without replacing a target."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination.parent, root=root)
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".staging",
        dir=destination.parent,
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            offset = 0
            while offset < len(data):
                written = handle.write(data[offset:])
                if written is None or written <= 0:
                    raise OSError("artifact staging write made no progress")
                offset += written
            handle.flush()
            os.fsync(handle.fileno())
            # Preserve the store's historical owner-writable mode. Immutability
            # is enforced by exclusive publication and content verification,
            # not by preventing integrity tests or operators from detecting a
            # deliberate post-publication mutation.
            os.fchmod(handle.fileno(), 0o600)
        try:
            os.link(staging, destination, follow_symlinks=False)
        except FileExistsError:
            existing = _read_regular_bytes(
                destination,
                expected_size=len(data),
                root=root,
            )
            if existing != data or hash_bytes(existing) != digest:
                raise ArtifactError(
                    f"content-address collision or corruption at {destination}"
                ) from None
        staging.unlink()
        _fsync_directory(destination.parent)
    finally:
        staging.unlink(missing_ok=True)


def _stage_regular_source(
    source: str | Path,
    *,
    staging_root: Path,
    root: Path,
) -> tuple[Path, str, int]:
    """Copy one stable regular source into a complete, fsynced staging inode."""

    source_descriptor, source_state, source_path = _open_regular_source(source)
    try:
        staging_root.mkdir(parents=True, exist_ok=True)
        _reject_symlink_components(staging_root, root=root)
        descriptor, staging_name = tempfile.mkstemp(
            prefix=".put-file.",
            suffix=".staging",
            dir=staging_root,
        )
    except BaseException:
        os.close(source_descriptor)
        raise
    staging = Path(staging_name)
    try:
        digest = hashlib.sha256()
        remaining = source_state.st_size
        while remaining:
            try:
                chunk = os.read(source_descriptor, min(remaining, 1024 * 1024))
            except OSError as error:
                raise ArtifactError(
                    f"artifact source cannot be read safely: {source_path}"
                ) from error
            if not chunk:
                raise ArtifactError(
                    f"artifact source changed while reading: {source_path}"
                )
            _write_all(descriptor, chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        try:
            exceeds_bound = bool(os.read(source_descriptor, 1))
        except OSError as error:
            raise ArtifactError(
                f"artifact source cannot be read safely: {source_path}"
            ) from error
        if exceeds_bound:
            raise ArtifactError(f"artifact source changed while reading: {source_path}")
        after = os.fstat(source_descriptor)
        if not _same_file_state(source_state, after):
            raise ArtifactError(f"artifact source changed while reading: {source_path}")
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o600)
        staged_state = os.fstat(descriptor)
        if (
            not stat.S_ISREG(staged_state.st_mode)
            or staged_state.st_size != source_state.st_size
        ):
            raise ArtifactError(f"artifact staging object is incomplete: {staging}")
        return staging, digest.hexdigest(), source_state.st_size
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
        os.close(source_descriptor)


def _open_regular_source(
    source: str | Path,
) -> tuple[int, os.stat_result, Path]:
    """Open a regular source without following a final-component symlink."""

    source_path = Path(source)
    try:
        original = source_path.lstat()
    except OSError as error:
        raise ArtifactError(f"artifact source is unavailable: {source_path}") from error
    if stat.S_ISLNK(original.st_mode):
        raise ArtifactError(f"artifact source must not be a symlink: {source_path}")
    if not stat.S_ISREG(original.st_mode):
        raise ArtifactError(f"artifact source must be a regular file: {source_path}")
    try:
        resolved = source_path.resolve(strict=True)
        before = resolved.lstat()
    except OSError as error:
        raise ArtifactError(f"artifact source is unavailable: {source_path}") from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or not _same_file_state(original, before)
    ):
        raise ArtifactError(f"artifact source changed before opening: {source_path}")
    try:
        descriptor = os.open(
            resolved,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise ArtifactError(
            f"artifact source cannot be opened safely: {source_path}"
        ) from error
    try:
        opened = os.fstat(descriptor)
    except OSError as error:
        os.close(descriptor)
        raise ArtifactError(
            f"artifact source cannot be inspected safely: {source_path}"
        ) from error
    if not _same_file_state(before, opened):
        os.close(descriptor)
        raise ArtifactError(f"artifact source changed before opening: {source_path}")
    return descriptor, opened, resolved


def _same_file_state(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
        and left.st_nlink == right.st_nlink
    )


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            raise OSError("artifact staging write made no progress")
        offset += written


def _publish_content_addressed_staging(
    destination: Path,
    staging: Path,
    *,
    digest: str,
    expected_size: int,
    root: Path,
) -> None:
    """Publish a complete staging inode without ever replacing an object."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination.parent, root=root)
    try:
        os.link(staging, destination, follow_symlinks=False)
    except FileExistsError:
        _verify_regular_file(
            destination,
            expected_size=expected_size,
            expected_digest=digest,
            root=root,
        )
    else:
        staged_state = staging.lstat()
        linked_state = destination.lstat()
        if (
            staged_state.st_dev != linked_state.st_dev
            or staged_state.st_ino != linked_state.st_ino
            or not stat.S_ISREG(linked_state.st_mode)
            or linked_state.st_size != expected_size
        ):
            raise ArtifactError(
                f"artifact object changed during publication: {destination}"
            )
        _fsync_directory(destination.parent)
        try:
            current_state = destination.lstat()
        except OSError as error:
            raise ArtifactError(
                f"artifact object changed during publication: {destination}"
            ) from error
        if (
            staged_state.st_dev != current_state.st_dev
            or staged_state.st_ino != current_state.st_ino
            or not stat.S_ISREG(current_state.st_mode)
            or current_state.st_size != expected_size
        ):
            raise ArtifactError(
                f"artifact object changed during publication: {destination}"
            )


def _verify_regular_file(
    path: Path,
    *,
    expected_size: int,
    expected_digest: str,
    root: Path,
) -> None:
    """Stream-verify an existing object and reject path or inode replacement."""

    _reject_symlink_components(path, root=root)
    try:
        before = path.lstat()
    except OSError as error:
        raise ArtifactError(f"artifact object is unavailable: {path}") from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_size != expected_size
    ):
        raise ArtifactError(f"artifact object type or size differs: {path}")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise ArtifactError(
            f"artifact object cannot be opened safely: {path}"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if not _same_content_state(before, opened):
            raise ArtifactError(f"artifact object changed while reading: {path}")
        digest = hashlib.sha256()
        remaining = expected_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ArtifactError(f"artifact object ended early: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ArtifactError(f"artifact object exceeds its bound size: {path}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = path.lstat()
    except OSError as error:
        raise ArtifactError(f"artifact object changed while reading: {path}") from error
    if not _same_content_state(opened, after) or not _same_content_state(
        after,
        current,
    ):
        raise ArtifactError(f"artifact object changed while reading: {path}")
    actual = digest.hexdigest()
    if actual != expected_digest:
        raise ArtifactError(f"content-address collision or corruption at {path}")


def _same_content_state(left: os.stat_result, right: os.stat_result) -> bool:
    """Compare content identity while tolerating a publisher unlinking its staging name."""

    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
    )


def _read_regular_bytes(
    path: Path,
    *,
    expected_size: int,
    root: Path,
) -> bytes:
    _reject_symlink_components(path, root=root)
    try:
        before = path.lstat()
    except OSError as error:
        raise ArtifactError(f"artifact object is unavailable: {path}") from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_size != expected_size
    ):
        raise ArtifactError(f"artifact object type or size differs: {path}")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise ArtifactError(
            f"artifact object cannot be opened safely: {path}"
        ) from error
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = expected_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ArtifactError(f"artifact object ended early: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ArtifactError(f"artifact object exceeds its bound size: {path}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        opened.st_dev != before.st_dev
        or opened.st_ino != before.st_ino
        or opened.st_size != before.st_size
        or after.st_dev != opened.st_dev
        or after.st_ino != opened.st_ino
        or after.st_size != opened.st_size
        or after.st_mtime_ns != opened.st_mtime_ns
    ):
        raise ArtifactError(f"artifact object changed while reading: {path}")
    return b"".join(chunks)


def _reject_symlink_components(path: Path, *, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ArtifactError(
            f"artifact path resolves outside workspace: {path}"
        ) from error
    current = path
    while True:
        try:
            observed = current.lstat()
        except FileNotFoundError:
            observed = None
        except OSError as error:
            raise ArtifactError(
                f"artifact path cannot be inspected: {current}"
            ) from error
        if observed is not None and stat.S_ISLNK(observed.st_mode):
            raise ArtifactError(f"artifact path contains a symlink: {current}")
        if current == root:
            return
        current = current.parent


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
