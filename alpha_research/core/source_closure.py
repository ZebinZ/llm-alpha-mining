"""Stable, import-aware source closure construction.

The closure is intentionally derived without importing project modules.  Each
source file is read through a pinned descriptor, and the exact bytes used for
AST import discovery are also the bytes that are hashed.  A second complete
read verifies that no file changed while the transitive closure was built.

Non-literal dynamic imports cannot be inferred safely.  Callers must bind every
such import site to all permitted local target source paths.  An encountered
unbound site, or a declared binding that is not encountered, fails closed.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Final, cast

from alpha_research.core.hashing import hash_json, require_sha256


_SCHEMA: Final[str] = "python-source-closure/v1"
_FILE_SCHEMA: Final[str] = "python-source-file-evidence/v1"
_MAX_SOURCE_BYTES: Final[int] = 16 * 1024 * 1024


class PythonSourceClosureError(ValueError):
    """Stable fail-closed error while constructing a local source closure."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}:{self.detail}")


@dataclass(frozen=True, slots=True)
class PythonSourceFileEvidenceV1:
    """Content identity for one project-relative regular Python source."""

    project_relative_path: str
    source_sha256: str
    size_bytes: int
    schema_version: str = _FILE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _FILE_SCHEMA:
            raise ValueError("unsupported Python source-file evidence schema")
        object.__setattr__(
            self,
            "project_relative_path",
            _relative_python_path(self.project_relative_path),
        )
        require_sha256(self.source_sha256, name="source file sha256")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("source file size must be a nonnegative integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "project_relative_path": self.project_relative_path,
            "source_sha256": self.source_sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class PythonDynamicImportBindingV1:
    """Explicit local targets for one source containing a non-literal import."""

    import_site_path: str
    target_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "import_site_path",
            _relative_python_path(self.import_site_path),
        )
        targets = tuple(
            sorted({_relative_python_path(item) for item in self.target_paths})
        )
        if not targets:
            raise ValueError("dynamic import binding requires at least one target")
        object.__setattr__(self, "target_paths", targets)

    def to_dict(self) -> dict[str, object]:
        return {
            "import_site_path": self.import_site_path,
            "target_paths": list(self.target_paths),
        }


@dataclass(frozen=True, slots=True)
class PythonSourceClosureV1:
    """Canonical transitive closure for explicit and dynamic source roots."""

    root_paths: tuple[str, ...]
    dynamic_import_bindings: tuple[PythonDynamicImportBindingV1, ...]
    files: tuple[PythonSourceFileEvidenceV1, ...]
    closure_hash: str = field(init=False)
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA:
            raise ValueError("unsupported Python source closure schema")
        roots = tuple(sorted({_relative_python_path(item) for item in self.root_paths}))
        bindings = tuple(
            sorted(self.dynamic_import_bindings, key=lambda item: item.import_site_path)
        )
        files = tuple(sorted(self.files, key=lambda item: item.project_relative_path))
        if (
            not roots
            or len(roots) != len(self.root_paths)
            or any(type(item) is not PythonDynamicImportBindingV1 for item in bindings)
            or len({item.import_site_path for item in bindings}) != len(bindings)
            or not files
            or any(type(item) is not PythonSourceFileEvidenceV1 for item in files)
            or len({item.project_relative_path for item in files}) != len(files)
        ):
            raise ValueError("Python source closure members are not canonical")
        file_paths = {item.project_relative_path for item in files}
        if not set(roots).issubset(file_paths):
            raise ValueError("Python source closure does not contain every root")
        for binding in bindings:
            if binding.import_site_path not in file_paths or not set(
                binding.target_paths
            ).issubset(file_paths):
                raise ValueError("dynamic import binding is outside source closure")
        object.__setattr__(self, "root_paths", roots)
        object.__setattr__(self, "dynamic_import_bindings", bindings)
        object.__setattr__(self, "files", files)
        object.__setattr__(
            self,
            "closure_hash",
            cast(str, hash_json(self.identity_payload())),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "root_paths": list(self.root_paths),
            "dynamic_import_bindings": [
                item.to_dict() for item in self.dynamic_import_bindings
            ],
            "files": [item.to_dict() for item in self.files],
        }

    def to_dict(self) -> dict[str, object]:
        return {"closure_hash": self.closure_hash, **self.identity_payload()}


@dataclass(frozen=True, slots=True)
class _StableSource:
    relative_path: str
    payload: bytes
    source_sha256: str
    identity: tuple[int, int, int, int, int, int]


def build_python_source_closure_v1(
    project_root: str | Path,
    *,
    root_paths: Sequence[str],
    dynamic_import_bindings: Mapping[str, Sequence[str]] | None = None,
) -> PythonSourceClosureV1:
    """Build and then independently revalidate one transitive local closure."""

    root = _regular_directory(project_root)
    supplied_roots = tuple(root_paths)
    roots = tuple(sorted({_relative_python_path(item) for item in supplied_roots}))
    if not roots or len(roots) != len(supplied_roots):
        raise ValueError("source closure roots must be nonempty and unique")
    bindings = _normalize_dynamic_bindings(dynamic_import_bindings or {})
    binding_map = {item.import_site_path: item.target_paths for item in bindings}
    pending = list(reversed(roots))
    sources: dict[str, _StableSource] = {}
    dynamic_sites_seen: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in sources:
            continue
        source = _read_stable_source(root, relative)
        sources[relative] = source
        dependencies, has_nonliteral_dynamic_import = _source_dependencies(
            root,
            relative,
            source.payload,
        )
        if has_nonliteral_dynamic_import:
            declared = binding_map.get(relative)
            if declared is None:
                raise PythonSourceClosureError(
                    "unbound_dynamic_import_site",
                    relative,
                )
            dynamic_sites_seen.add(relative)
            dependencies.update(declared)
        dependencies.update(_parent_package_initializers(root, relative))
        for dependency in sorted(dependencies, reverse=True):
            if dependency not in sources:
                pending.append(dependency)
    unused = set(binding_map) - dynamic_sites_seen
    if unused:
        raise PythonSourceClosureError(
            "unused_dynamic_import_binding",
            ",".join(sorted(unused)),
        )
    for relative, first in sorted(sources.items()):
        second = _read_stable_source(root, relative)
        if (
            second.payload != first.payload
            or second.source_sha256 != first.source_sha256
            or second.identity != first.identity
        ):
            raise PythonSourceClosureError(
                "source_closure_changed_during_build",
                relative,
            )
    evidence = tuple(
        PythonSourceFileEvidenceV1(
            project_relative_path=relative,
            source_sha256=source.source_sha256,
            size_bytes=len(source.payload),
        )
        for relative, source in sorted(sources.items())
    )
    return PythonSourceClosureV1(
        root_paths=roots,
        dynamic_import_bindings=bindings,
        files=evidence,
    )


def stable_regular_file_sha256_v1(
    path: str | Path,
    *,
    maximum_bytes: int = 64 * 1024 * 1024,
) -> str:
    """Hash one regular file twice and reject identity or content drift."""

    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be a positive integer")
    target = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    first = _read_stable_regular_file(target, maximum_bytes=maximum_bytes)
    second = _read_stable_regular_file(target, maximum_bytes=maximum_bytes)
    if first[0] != second[0] or first[1] != second[1]:
        raise PythonSourceClosureError(
            "regular_file_changed_during_hash",
            os.fspath(target),
        )
    return hashlib.sha256(first[0]).hexdigest()


def _normalize_dynamic_bindings(
    value: Mapping[str, Sequence[str]],
) -> tuple[PythonDynamicImportBindingV1, ...]:
    if not isinstance(value, Mapping):
        raise TypeError("dynamic import bindings must be a mapping")
    result = tuple(
        PythonDynamicImportBindingV1(
            import_site_path=_relative_python_path(site),
            target_paths=tuple(targets),
        )
        for site, targets in value.items()
    )
    if len({item.import_site_path for item in result}) != len(result):
        raise ValueError("dynamic import binding sites must be unique")
    return tuple(sorted(result, key=lambda item: item.import_site_path))


def _source_dependencies(
    root: Path,
    relative: str,
    payload: bytes,
) -> tuple[set[str], bool]:
    try:
        tree = ast.parse(payload, filename=relative)
    except (SyntaxError, ValueError) as error:
        raise PythonSourceClosureError(
            "source_ast_invalid",
            f"{relative}:{type(error).__name__}",
        ) from error
    dependencies: set[str] = set()
    module_name, package_name = _module_and_package(relative)
    del module_name
    has_nonliteral_dynamic_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = _local_module_path(root, alias.name)
                if target is not None:
                    dependencies.add(target)
        elif isinstance(node, ast.ImportFrom):
            base_name = _resolved_import_from(node, package_name, relative)
            if base_name:
                base_path = _local_module_path(root, base_name)
                if base_path is not None:
                    dependencies.add(base_path)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    target = _local_module_path(root, f"{base_name}.{alias.name}")
                    if target is not None:
                        dependencies.add(target)
        elif isinstance(node, ast.Call) and _is_dynamic_import_call(node.func):
            target_name = _literal_dynamic_module_name(node)
            if target_name is None:
                has_nonliteral_dynamic_import = True
            else:
                target = _local_module_path(root, target_name)
                if target is not None:
                    dependencies.add(target)
    return dependencies, has_nonliteral_dynamic_import


def _resolved_import_from(
    node: ast.ImportFrom,
    package_name: str,
    relative: str,
) -> str | None:
    if node.level == 0:
        return node.module
    reference = "." * node.level + (node.module or "")
    try:
        return importlib.util.resolve_name(reference, package_name)
    except (ImportError, ValueError) as error:
        raise PythonSourceClosureError(
            "relative_import_invalid",
            f"{relative}:{reference}",
        ) from error


def _is_dynamic_import_call(function: ast.expr) -> bool:
    if isinstance(function, ast.Name):
        return function.id in {"__import__", "import_module"}
    return (
        isinstance(function, ast.Attribute)
        and function.attr == "import_module"
        and isinstance(function.value, ast.Name)
        and function.value.id == "importlib"
    )


def _literal_dynamic_module_name(node: ast.Call) -> str | None:
    if not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and type(first.value) is str and first.value:
        return cast(str, first.value)
    return None


def _module_and_package(relative: str) -> tuple[str, str]:
    path = PurePosixPath(relative)
    parts = list(path.parts)
    stem = parts[-1].removesuffix(".py")
    if stem == "__init__":
        module = ".".join(parts[:-1])
        return module, module
    module = ".".join((*parts[:-1], stem))
    return module, module.rpartition(".")[0]


def _local_module_path(root: Path, module_name: str) -> str | None:
    if (
        type(module_name) is not str
        or not module_name
        or any(not item.isidentifier() for item in module_name.split("."))
    ):
        return None
    relative = module_name.replace(".", "/")
    candidates = (f"{relative}.py", f"{relative}/__init__.py")
    existing: list[str] = []
    for candidate in candidates:
        path = root / candidate
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise PythonSourceClosureError(
                "local_module_unavailable",
                candidate,
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise PythonSourceClosureError(
                "local_module_not_regular",
                candidate,
            )
        existing.append(candidate)
    if len(existing) > 1:
        raise PythonSourceClosureError(
            "local_module_ambiguous",
            module_name,
        )
    return None if not existing else existing[0]


def _parent_package_initializers(root: Path, relative: str) -> set[str]:
    parts = PurePosixPath(relative).parts
    result: set[str] = set()
    for length in range(1, len(parts)):
        candidate = "/".join((*parts[:length], "__init__.py"))
        path = root / candidate
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise PythonSourceClosureError(
                "package_initializer_unavailable",
                candidate,
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise PythonSourceClosureError(
                "package_initializer_not_regular",
                candidate,
            )
        result.add(candidate)
    return result


def _read_stable_source(root: Path, relative: str) -> _StableSource:
    safe = _relative_python_path(relative)
    target = root / safe
    payload, identity = _read_stable_regular_file(
        target,
        maximum_bytes=_MAX_SOURCE_BYTES,
    )
    return _StableSource(
        relative_path=safe,
        payload=payload,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        identity=identity,
    )


def _read_stable_regular_file(
    target: Path,
    *,
    maximum_bytes: int,
) -> tuple[bytes, tuple[int, int, int, int, int, int]]:
    lexical = Path(os.path.abspath(os.fspath(target)))
    try:
        before = os.lstat(lexical)
    except OSError as error:
        raise PythonSourceClosureError(
            "regular_file_unavailable",
            os.fspath(lexical),
        ) from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise PythonSourceClosureError(
            "regular_file_not_regular",
            os.fspath(lexical),
        )
    if before.st_size > maximum_bytes:
        raise PythonSourceClosureError(
            "regular_file_too_large",
            os.fspath(lexical),
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lexical, flags)
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
                raise PythonSourceClosureError(
                    "regular_file_too_large",
                    os.fspath(lexical),
                )
        after_read = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.lstat(lexical)
    identities = tuple(
        _stat_identity(item) for item in (before, opened, after_read, after)
    )
    if len(set(identities)) != 1:
        raise PythonSourceClosureError(
            "regular_file_changed_during_read",
            os.fspath(lexical),
        )
    return b"".join(chunks), identities[0]


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _regular_directory(value: str | Path) -> Path:
    path = Path(os.path.abspath(os.fspath(Path(value).expanduser())))
    try:
        metadata = path.lstat()
    except OSError as error:
        raise PythonSourceClosureError(
            "project_root_unavailable",
            os.fspath(path),
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise PythonSourceClosureError(
            "project_root_not_directory",
            os.fspath(path),
        )
    return path


def _relative_python_path(value: object) -> str:
    if type(value) is not str or not value:
        raise TypeError("source path must be nonempty text")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.suffix != ".py"
    ):
        raise ValueError("source path must be canonical project-relative Python")
    return value


__all__ = [
    "PythonDynamicImportBindingV1",
    "PythonSourceClosureError",
    "PythonSourceClosureV1",
    "PythonSourceFileEvidenceV1",
    "build_python_source_closure_v1",
    "stable_regular_file_sha256_v1",
]
