"""Small shared primitive for immutable canonical JSON artifacts.

Every pathname component is traversed from the filesystem anchor with retained
directory descriptors.  Leaf operations are descriptor-relative, and the
retained chain is reauthenticated before returning.  This prevents a concurrent
rename/replacement of any parent directory from silently redirecting a load or
publication to another namespace.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import secrets
import stat
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from alpha_research.core.hashing import canonical_json_bytes, require_sha256


class ImmutableJsonError(ValueError):
    """Stable filesystem-boundary error for immutable JSON documents."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}:{self.detail}")


_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_READ_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
_STAGING_NAME_PREFIX = ".immutable-json-stage-"
_STAGING_NAME_ATTEMPTS = 8
_LINUX_RENAME_NOREPLACE = 1
_DARWIN_RENAME_EXCL = 0x00000004


@dataclass(slots=True)
class _DirectoryChain:
    path: Path
    descriptors: list[int]
    component_names: tuple[str, ...]
    initial_stats: tuple[os.stat_result, ...]

    @property
    def leaf_descriptor(self) -> int:
        return self.descriptors[-1]

    def close(self) -> None:
        while self.descriptors:
            os.close(self.descriptors.pop())


def write_immutable_json_document(
    output_root: str | Path,
    *,
    filename: str,
    value: Mapping[str, object],
    mode: int = 0o440,
) -> Path:
    """Publish canonical JSON once without exposing a partial final document.

    Bytes and metadata are made durable under a random same-directory staging
    name before an atomic no-replace rename.  A killed writer can therefore
    leave only an unreferenced staging artifact; the requested final name stays
    absent and an idempotent retry remains possible.
    """

    return _write_immutable_bytes_document(
        output_root,
        filename=filename,
        payload=canonical_json_bytes(dict(value)) + b"\n",
        mode=mode,
    )


def write_immutable_bytes_document(
    output_root: str | Path,
    *,
    filename: str,
    payload: bytes,
    mode: int = 0o440,
) -> Path:
    """Publish one nonempty immutable binary document through the same dirfd chain."""

    if not isinstance(payload, bytes) or not payload:
        raise TypeError("immutable binary document payload must be nonempty bytes")
    return _write_immutable_bytes_document(
        output_root,
        filename=filename,
        payload=payload,
        mode=mode,
    )


def _write_immutable_bytes_document(
    output_root: str | Path,
    *,
    filename: str,
    payload: bytes,
    mode: int,
) -> Path:
    if not filename or Path(filename).name != filename:
        raise ValueError("immutable document filename must be one path component")
    root = _absolute_path(output_root)
    target = root / filename
    chain = _open_directory_chain(
        root,
        create=True,
        invalid_code="output_root_invalid",
    )
    try:
        _validate_directory_chain(chain, "document_changed_while_writing")
        stage_name, descriptor = _create_staging_document_at(
            chain.leaf_descriptor,
            filename=filename,
        )
        stage_linked = True

        try:
            created = os.fstat(descriptor)
            if not stat.S_ISREG(created.st_mode) or created.st_nlink != 1:
                raise ImmutableJsonError(
                    "document_changed_while_writing",
                    filename,
                )
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("short immutable document write")
                offset += written
            os.fchmod(descriptor, mode)
            os.fsync(descriptor)
            staged = os.fstat(descriptor)
            if (
                not _regular_document_valid(
                    staged,
                    mode=mode,
                    maximum_bytes=len(payload),
                )
                or staged.st_size != len(payload)
            ):
                raise ImmutableJsonError(
                    "document_changed_while_writing",
                    filename,
                )
            _validate_directory_chain(
                chain,
                "document_changed_while_writing",
            )
            try:
                _rename_noreplace_at(
                    chain.leaf_descriptor,
                    source_name=stage_name,
                    target_name=filename,
                )
            except FileExistsError:
                _unlink_owned_staging_document_at(
                    chain.leaf_descriptor,
                    stage_name=stage_name,
                    descriptor=descriptor,
                    detail=filename,
                )
                stage_linked = False
                existing = _read_regular_file_at(
                    chain,
                    filename,
                    display_path=target,
                    mode=mode,
                    maximum_bytes=len(payload),
                )
                if existing != payload:
                    raise ImmutableJsonError(
                        "content_address_conflict",
                        filename,
                    ) from None
                os.fsync(chain.leaf_descriptor)
                _validate_directory_chain(
                    chain,
                    "document_changed_while_writing",
                )
                return target
            except OSError as error:
                raise ImmutableJsonError(
                    "document_changed_while_writing",
                    filename,
                ) from error
            stage_linked = False
            published = os.fstat(descriptor)
            linked = _stat_entry(
                chain.leaf_descriptor,
                filename,
                code="document_changed_while_writing",
                detail=filename,
            )
            if (
                not _regular_document_valid(
                    published,
                    mode=mode,
                    maximum_bytes=len(payload),
                )
                or published.st_size != len(payload)
                or not _same_file_state(linked, published)
            ):
                raise ImmutableJsonError(
                    "document_changed_while_writing",
                    filename,
                )
            os.fsync(chain.leaf_descriptor)
            _validate_directory_chain(
                chain,
                "document_changed_while_writing",
            )
        finally:
            try:
                if stage_linked:
                    _unlink_owned_staging_document_at(
                        chain.leaf_descriptor,
                        stage_name=stage_name,
                        descriptor=descriptor,
                        detail=filename,
                    )
            finally:
                os.close(descriptor)

        if (
            _read_regular_file_at(
                chain,
                filename,
                display_path=target,
                mode=mode,
                maximum_bytes=len(payload),
            )
            != payload
        ):
            raise ImmutableJsonError(
                "document_changed_while_writing",
                filename,
            )
        _validate_directory_chain(chain, "document_changed_while_writing")
        return target
    finally:
        chain.close()


def _create_staging_document_at(
    parent_descriptor: int,
    *,
    filename: str,
) -> tuple[str, int]:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    staging_mode = 0o600
    filename_hash = hashlib.sha256(os.fsencode(filename)).hexdigest()[:16]
    for _ in range(_STAGING_NAME_ATTEMPTS):
        stage_name = (
            f"{_STAGING_NAME_PREFIX}{filename_hash}-{secrets.token_hex(16)}"
        )
        try:
            descriptor = os.open(
                stage_name,
                flags,
                staging_mode,
                dir_fd=parent_descriptor,
            )
        except FileExistsError:
            continue
        except OSError as error:
            raise ImmutableJsonError(
                "document_changed_while_writing",
                filename,
            ) from error
        return stage_name, descriptor
    raise ImmutableJsonError("staging_name_exhausted", filename)


def _rename_noreplace_at(
    parent_descriptor: int,
    *,
    source_name: str,
    target_name: str,
) -> None:
    """Atomically publish one staged entry without replacing any target."""

    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function_name = "renameatx_np"
        flag = _DARWIN_RENAME_EXCL
    elif sys.platform.startswith("linux"):
        function_name = "renameat2"
        flag = _LINUX_RENAME_NOREPLACE
    else:
        raise ImmutableJsonError(
            "atomic_noreplace_unsupported",
            target_name,
        )
    try:
        function = getattr(libc, function_name)
    except AttributeError as error:
        raise ImmutableJsonError(
            "atomic_noreplace_unsupported",
            target_name,
        ) from error
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = function(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(target_name),
        flag,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), target_name)
    raise OSError(error_number, os.strerror(error_number), target_name)


def _unlink_owned_staging_document_at(
    parent_descriptor: int,
    *,
    stage_name: str,
    descriptor: int,
    detail: str,
) -> None:
    """Remove only the still-linked, single-name staging inode we opened."""

    try:
        opened = os.fstat(descriptor)
        linked = os.stat(
            stage_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or not _same_file_identity(opened, linked)
        ):
            raise ImmutableJsonError(
                "document_changed_while_writing",
                detail,
            )
        os.unlink(stage_name, dir_fd=parent_descriptor)
    except FileNotFoundError:
        raise ImmutableJsonError(
            "document_changed_while_writing",
            detail,
        ) from None
    except ImmutableJsonError:
        raise
    except OSError as error:
        raise ImmutableJsonError(
            "document_changed_while_writing",
            detail,
        ) from error


def load_immutable_json_document(
    path: str | Path,
    *,
    expected_filename: str,
    expected_document_sha256: str,
    mode: int = 0o440,
    maximum_bytes: int = 16 * 1024 * 1024,
) -> Mapping[str, object]:
    """Strictly load one externally SHA-pinned canonical JSON document."""

    document_sha = require_sha256(
        expected_document_sha256,
        name="expected immutable JSON document sha256",
    )
    target = _absolute_path(path)
    if target.name != expected_filename:
        raise ImmutableJsonError("filename_mismatch", target.name)
    chain = _open_directory_chain(
        target.parent,
        create=False,
        invalid_code="document_invalid",
    )
    try:
        payload = _read_regular_file_at(
            chain,
            target.name,
            display_path=target,
            mode=mode,
            maximum_bytes=maximum_bytes,
        )
    finally:
        chain.close()
    if hashlib.sha256(payload).hexdigest() != document_sha:
        raise ImmutableJsonError("document_sha_mismatch", target.name)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ImmutableJsonError("document_json_invalid", target.name) from error
    if not isinstance(value, Mapping):
        raise ImmutableJsonError("document_root_invalid", target.name)
    if canonical_json_bytes(dict(value)) + b"\n" != payload:
        raise ImmutableJsonError("document_not_canonical", target.name)
    return value


def _absolute_path(value: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(value).expanduser())))


def _open_directory_chain(
    path: Path,
    *,
    create: bool,
    invalid_code: str,
) -> _DirectoryChain:
    if not path.is_absolute() or not path.anchor:
        raise ImmutableJsonError(invalid_code, os.fspath(path))
    descriptors: list[int] = []
    names: list[str] = []
    initial_stats: list[os.stat_result] = []
    try:
        try:
            root_descriptor = os.open(path.anchor, _DIRECTORY_OPEN_FLAGS)
        except OSError as error:
            raise ImmutableJsonError(invalid_code, path.anchor) from error
        descriptors.append(root_descriptor)
        root_stat = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise ImmutableJsonError(invalid_code, path.anchor)
        initial_stats.append(root_stat)

        current = Path(path.anchor)
        for name in path.parts[1:]:
            current /= name
            parent_descriptor = descriptors[-1]
            try:
                child_descriptor = os.open(
                    name,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=parent_descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise ImmutableJsonError(
                        invalid_code,
                        os.fspath(path),
                    ) from None
                try:
                    os.mkdir(name, mode=0o750, dir_fd=parent_descriptor)
                except FileExistsError:
                    pass
                except OSError as error:
                    _raise_directory_component_error(
                        parent_descriptor,
                        name,
                        current,
                        invalid_code,
                        error,
                    )
                try:
                    child_descriptor = os.open(
                        name,
                        _DIRECTORY_OPEN_FLAGS,
                        dir_fd=parent_descriptor,
                    )
                except OSError as error:
                    _raise_directory_component_error(
                        parent_descriptor,
                        name,
                        current,
                        invalid_code,
                        error,
                    )
            except OSError as error:
                _raise_directory_component_error(
                    parent_descriptor,
                    name,
                    current,
                    invalid_code,
                    error,
                )
            child_stat = os.fstat(child_descriptor)
            if not stat.S_ISDIR(child_stat.st_mode):
                os.close(child_descriptor)
                raise ImmutableJsonError(invalid_code, os.fspath(current))
            descriptors.append(child_descriptor)
            names.append(name)
            initial_stats.append(child_stat)

        result = _DirectoryChain(
            path=path,
            descriptors=descriptors,
            component_names=tuple(names),
            initial_stats=tuple(initial_stats),
        )
        _validate_directory_chain(result, invalid_code)
        return result
    except BaseException:
        while descriptors:
            os.close(descriptors.pop())
        raise


def _raise_directory_component_error(
    parent_descriptor: int,
    name: str,
    display_path: Path,
    invalid_code: str,
    error: OSError,
) -> NoReturn:
    try:
        observed = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError:
        observed = None
    if observed is not None and stat.S_ISLNK(observed.st_mode):
        raise ImmutableJsonError(
            "path_symlink_forbidden",
            os.fspath(display_path),
        ) from error
    raise ImmutableJsonError(invalid_code, os.fspath(display_path)) from error


def _validate_directory_chain(chain: _DirectoryChain, code: str) -> None:
    try:
        for index, descriptor in enumerate(chain.descriptors):
            observed = os.fstat(descriptor)
            initial = chain.initial_stats[index]
            if (
                not stat.S_ISDIR(observed.st_mode)
                or observed.st_dev != initial.st_dev
                or observed.st_ino != initial.st_ino
                or observed.st_mode != initial.st_mode
            ):
                raise ImmutableJsonError(code, os.fspath(chain.path))
            if index == 0:
                linked = os.stat(chain.path.anchor, follow_symlinks=False)
            else:
                linked = os.stat(
                    chain.component_names[index - 1],
                    dir_fd=chain.descriptors[index - 1],
                    follow_symlinks=False,
                )
            if (
                not stat.S_ISDIR(linked.st_mode)
                or linked.st_dev != observed.st_dev
                or linked.st_ino != observed.st_ino
                or linked.st_mode != observed.st_mode
            ):
                raise ImmutableJsonError(code, os.fspath(chain.path))
    except ImmutableJsonError:
        raise
    except OSError as error:
        raise ImmutableJsonError(code, os.fspath(chain.path)) from error


def _read_regular_file_at(
    chain: _DirectoryChain,
    name: str,
    *,
    display_path: Path,
    mode: int,
    maximum_bytes: int,
) -> bytes:
    _validate_directory_chain(chain, "document_changed_while_reading")
    parent_descriptor = chain.leaf_descriptor
    before = _stat_entry(
        parent_descriptor,
        name,
        code="document_invalid",
        detail=os.fspath(display_path),
    )
    if stat.S_ISLNK(before.st_mode):
        raise ImmutableJsonError(
            "path_symlink_forbidden",
            os.fspath(display_path),
        )
    if not _regular_document_valid(
        before,
        mode=mode,
        maximum_bytes=maximum_bytes,
    ):
        raise ImmutableJsonError("document_invalid", os.fspath(display_path))
    try:
        descriptor = os.open(
            name,
            _READ_OPEN_FLAGS,
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        raise ImmutableJsonError(
            "document_changed_while_reading",
            os.fspath(display_path),
        ) from error
    try:
        opened = os.fstat(descriptor)
        if (
            not _regular_document_valid(
                opened,
                mode=mode,
                maximum_bytes=maximum_bytes,
            )
            or not _same_file_state(before, opened)
        ):
            raise ImmutableJsonError(
                "document_changed_while_reading",
                os.fspath(display_path),
            )
        chunks: list[bytes] = []
        observed_size = 0
        while True:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, maximum_bytes - observed_size + 1),
            )
            if not chunk:
                break
            chunks.append(chunk)
            observed_size += len(chunk)
            if observed_size > maximum_bytes:
                raise ImmutableJsonError(
                    "document_invalid",
                    os.fspath(display_path),
                )
        after = os.fstat(descriptor)
        linked_after = _stat_entry(
            parent_descriptor,
            name,
            code="document_changed_while_reading",
            detail=os.fspath(display_path),
        )
    finally:
        os.close(descriptor)
    _validate_directory_chain(chain, "document_changed_while_reading")
    if (
        not _same_file_state(opened, after)
        or not _same_file_state(after, linked_after)
        or observed_size != after.st_size
    ):
        raise ImmutableJsonError(
            "document_changed_while_reading",
            os.fspath(display_path),
        )
    return b"".join(chunks)


def _stat_entry(
    parent_descriptor: int,
    name: str,
    *,
    code: str,
    detail: str,
) -> os.stat_result:
    try:
        return os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise ImmutableJsonError(code, detail) from error


def _regular_document_valid(
    observed: os.stat_result,
    *,
    mode: int,
    maximum_bytes: int,
) -> bool:
    return (
        stat.S_ISREG(observed.st_mode)
        and stat.S_IMODE(observed.st_mode) == mode
        and observed.st_nlink == 1
        and observed.st_size > 0
        and observed.st_size <= maximum_bytes
    )


def _same_file_state(
    left: os.stat_result,
    right: os.stat_result,
) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_mode == right.st_mode
        and left.st_nlink == right.st_nlink
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _same_file_identity(
    left: os.stat_result,
    right: os.stat_result,
) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


__all__ = [
    "ImmutableJsonError",
    "load_immutable_json_document",
    "write_immutable_bytes_document",
    "write_immutable_json_document",
]
