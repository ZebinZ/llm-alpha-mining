"""Content-addressed Python environment declarations for research execution.

This module is deliberately independent from the historical
``requirements-core.lock`` and every V1 runner.  It provides a strict V2
authority that can be adopted by new execution surfaces without changing the
meaning or live verification behaviour of any sealed V1 evidence.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Final, cast

from alpha_research.core.hashing import (
    canonical_json_bytes,
    hash_json,
    require_sha256,
)
from alpha_research.core.immutable_json import (
    ImmutableJsonError,
    load_immutable_json_document,
    write_immutable_json_document,
)


_LOCK_SCHEMA: Final[str] = "python-package-lock/v2"
_AUTHORITY_SCHEMA: Final[str] = "python-environment-authority/v2"
_LOCK_SUFFIX: Final[str] = ".python-package-lock.json"
_AUTHORITY_SUFFIX: Final[str] = ".python-environment-authority.json"
_PROFILE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[a-z0-9][a-z0-9._-]{0,127}"
)
_PYTHON_VERSION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[0-9]+\.[0-9]+\.[0-9]+"
)
_PACKAGE_SEPARATOR_PATTERN: Final[re.Pattern[str]] = re.compile(r"[-_.]+")
_FORBIDDEN_VERSION_CHARACTERS: Final[frozenset[str]] = frozenset(
    " \t\r\n*<>=!~;@"
)

L2_CORE_PACKAGE_NAMES: Final[tuple[str, ...]] = (
    "mypy",
    "numpy",
    "pandas",
    "psutil",
    "pyarrow",
    "pytest",
    "scikit-learn",
    "scipy",
)

VersionProvider = Callable[[str], str]


class PythonEnvironmentAuthorityError(ValueError):
    """Stable fail-closed error at the V2 environment authority boundary."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}:{self.detail}")


@dataclass(frozen=True, slots=True)
class PythonPackageLockV2:
    """An exact, content-addressed set of direct Python package pins."""

    profile: str
    packages: tuple[tuple[str, str], ...]
    transitive_artifact_hashes_present: bool = False
    production_ready: bool = False
    lock_id: str = field(init=False)
    schema_version: str = _LOCK_SCHEMA

    def __post_init__(self) -> None:
        _require_profile(self.profile)
        packages = _canonical_packages(self.packages)
        if packages != self.packages:
            raise ValueError("package lock pins must be sorted and canonical")
        if self.schema_version != _LOCK_SCHEMA:
            raise ValueError("unsupported Python package lock schema")
        if (
            type(self.transitive_artifact_hashes_present) is not bool
            or self.transitive_artifact_hashes_present
            or type(self.production_ready) is not bool
            or self.production_ready
        ):
            raise ValueError("package lock assurance boundary differs")
        object.__setattr__(
            self,
            "lock_id",
            cast(str, hash_json(self.identity_payload())),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "packages": dict(self.packages),
            "transitive_artifact_hashes_present": (
                self.transitive_artifact_hashes_present
            ),
            "production_ready": self.production_ready,
        }

    def to_dict(self) -> dict[str, object]:
        return {"lock_id": self.lock_id, **self.identity_payload()}

    @property
    def document_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self.to_dict()) + b"\n"
        ).hexdigest()

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PythonPackageLockV2":
        expected = {
            "lock_id",
            "schema_version",
            "profile",
            "packages",
            "transitive_artifact_hashes_present",
            "production_ready",
        }
        if set(value) != expected:
            raise ValueError("Python package lock fields differ")
        result = cls(
            profile=_text(value["profile"], "profile"),
            packages=_packages_from_wire(value["packages"], "packages"),
            transitive_artifact_hashes_present=_boolean(
                value["transitive_artifact_hashes_present"],
                "transitive_artifact_hashes_present",
            ),
            production_ready=_boolean(
                value["production_ready"],
                "production_ready",
            ),
            schema_version=_text(value["schema_version"], "schema_version"),
        )
        if (
            require_sha256(
                _text(value["lock_id"], "lock_id"),
                name="Python package lock id",
            )
            != result.lock_id
        ):
            raise ValueError("Python package lock self hash differs")
        return result


@dataclass(frozen=True, slots=True)
class PythonEnvironmentAuthorityV2:
    """A fail-closed binding between one lock and one live Python runtime."""

    profile: str
    lock_path: str
    lock_id: str
    lock_document_sha256: str
    locked_packages: tuple[tuple[str, str], ...]
    observed_packages: tuple[tuple[str, str], ...]
    python_implementation: str
    python_version: str
    python_build: str
    platform: str
    machine: str
    environment_lock_match: bool = True
    transitive_artifact_hashes_present: bool = False
    research_only: bool = True
    production_ready: bool = False
    authority_id: str = field(init=False)
    schema_version: str = _AUTHORITY_SCHEMA

    def __post_init__(self) -> None:
        _require_profile(self.profile)
        _require_project_relative_path(self.lock_path)
        require_sha256(self.lock_id, name="Python package lock id")
        require_sha256(
            self.lock_document_sha256,
            name="Python package lock document sha256",
        )
        locked = _canonical_packages(self.locked_packages)
        observed = _canonical_packages(self.observed_packages)
        if (
            locked != self.locked_packages
            or observed != self.observed_packages
        ):
            raise ValueError("environment package pins must be sorted and canonical")
        if locked != observed:
            raise ValueError("locked and observed package versions differ")
        if self.python_implementation != "CPython":
            raise ValueError("Python implementation must be CPython")
        if not _PYTHON_VERSION_PATTERN.fullmatch(self.python_version):
            raise ValueError("Python version must be canonical major.minor.patch")
        for name in ("python_build", "platform", "machine"):
            value = getattr(self, name)
            if type(value) is not str or not value or value.strip() != value:
                raise ValueError(f"{name} is invalid")
        if self.schema_version != _AUTHORITY_SCHEMA:
            raise ValueError("unsupported Python environment authority schema")
        if (
            type(self.environment_lock_match) is not bool
            or not self.environment_lock_match
            or type(self.transitive_artifact_hashes_present) is not bool
            or self.transitive_artifact_hashes_present
            or type(self.research_only) is not bool
            or not self.research_only
            or type(self.production_ready) is not bool
            or self.production_ready
        ):
            raise ValueError("Python environment assurance boundary differs")
        object.__setattr__(
            self,
            "authority_id",
            cast(str, hash_json(self.identity_payload())),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "lock_path": self.lock_path,
            "lock_id": self.lock_id,
            "lock_document_sha256": self.lock_document_sha256,
            "locked_packages": dict(self.locked_packages),
            "observed_packages": dict(self.observed_packages),
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
            "python_build": self.python_build,
            "platform": self.platform,
            "machine": self.machine,
            "environment_lock_match": self.environment_lock_match,
            "transitive_artifact_hashes_present": (
                self.transitive_artifact_hashes_present
            ),
            "research_only": self.research_only,
            "production_ready": self.production_ready,
        }

    def to_dict(self) -> dict[str, object]:
        return {"authority_id": self.authority_id, **self.identity_payload()}

    @property
    def document_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self.to_dict()) + b"\n"
        ).hexdigest()

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> "PythonEnvironmentAuthorityV2":
        expected = {
            "authority_id",
            "schema_version",
            "profile",
            "lock_path",
            "lock_id",
            "lock_document_sha256",
            "locked_packages",
            "observed_packages",
            "python_implementation",
            "python_version",
            "python_build",
            "platform",
            "machine",
            "environment_lock_match",
            "transitive_artifact_hashes_present",
            "research_only",
            "production_ready",
        }
        if set(value) != expected:
            raise ValueError("Python environment authority fields differ")
        result = cls(
            profile=_text(value["profile"], "profile"),
            lock_path=_text(value["lock_path"], "lock_path"),
            lock_id=_text(value["lock_id"], "lock_id"),
            lock_document_sha256=_text(
                value["lock_document_sha256"],
                "lock_document_sha256",
            ),
            locked_packages=_packages_from_wire(
                value["locked_packages"],
                "locked_packages",
            ),
            observed_packages=_packages_from_wire(
                value["observed_packages"],
                "observed_packages",
            ),
            python_implementation=_text(
                value["python_implementation"],
                "python_implementation",
            ),
            python_version=_text(value["python_version"], "python_version"),
            python_build=_text(value["python_build"], "python_build"),
            platform=_text(value["platform"], "platform"),
            machine=_text(value["machine"], "machine"),
            environment_lock_match=_boolean(
                value["environment_lock_match"],
                "environment_lock_match",
            ),
            transitive_artifact_hashes_present=_boolean(
                value["transitive_artifact_hashes_present"],
                "transitive_artifact_hashes_present",
            ),
            research_only=_boolean(value["research_only"], "research_only"),
            production_ready=_boolean(
                value["production_ready"],
                "production_ready",
            ),
            schema_version=_text(value["schema_version"], "schema_version"),
        )
        if (
            require_sha256(
                _text(value["authority_id"], "authority_id"),
                name="Python environment authority id",
            )
            != result.authority_id
        ):
            raise ValueError("Python environment authority self hash differs")
        return result


def observe_installed_python_packages_v2(
    package_names: Sequence[str],
    *,
    version_provider: VersionProvider = importlib.metadata.version,
) -> tuple[tuple[str, str], ...]:
    """Read exact installed versions and reject every missing distribution."""

    names = tuple(_canonical_package_name(item) for item in package_names)
    if not names or names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ValueError("package names must be a nonempty sorted unique sequence")
    output: list[tuple[str, str]] = []
    for name in names:
        try:
            version = version_provider(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise PythonEnvironmentAuthorityError("package_missing", name) from error
        output.append((name, _exact_version(version, name=name)))
    return tuple(output)


def build_current_python_environment_authority_v2(
    project_root: str | Path,
    *,
    profile: str,
    lock_path: str | Path,
    expected_lock_id: str,
    expected_lock_document_sha256: str,
    version_provider: VersionProvider = importlib.metadata.version,
) -> PythonEnvironmentAuthorityV2:
    """Bind a strict package lock to the currently executing CPython runtime."""

    root = _regular_project_root(project_root)
    lock_file = _regular_file_inside_project(root, lock_path, "lock path")
    relative_lock = lock_file.relative_to(root).as_posix()
    package_lock = load_python_package_lock_v2(
        lock_file,
        expected_lock_id=expected_lock_id,
        expected_document_sha256=expected_lock_document_sha256,
    )
    if package_lock.profile != profile:
        raise PythonEnvironmentAuthorityError(
            "lock_profile_mismatch",
            f"{package_lock.profile}:{profile}",
        )
    observed = observe_installed_python_packages_v2(
        tuple(name for name, _ in package_lock.packages),
        version_provider=version_provider,
    )
    if observed != package_lock.packages:
        raise PythonEnvironmentAuthorityError(
            "package_version_mismatch",
            _package_diff(package_lock.packages, observed),
        )
    return PythonEnvironmentAuthorityV2(
        profile=profile,
        lock_path=relative_lock,
        lock_id=package_lock.lock_id,
        lock_document_sha256=package_lock.document_sha256,
        locked_packages=package_lock.packages,
        observed_packages=observed,
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        python_build=sys.version,
        platform=platform.platform(),
        machine=platform.machine(),
    )


def verify_current_python_environment_authority_v2(
    authority_path: str | Path,
    *,
    project_root: str | Path,
    expected_authority_id: str,
    expected_authority_document_sha256: str,
    version_provider: VersionProvider = importlib.metadata.version,
) -> PythonEnvironmentAuthorityV2:
    """Reopen an authority and fail closed on lock, package or runtime drift."""

    root = _regular_project_root(project_root)
    authority = load_python_environment_authority_v2(
        authority_path,
        expected_authority_id=expected_authority_id,
        expected_document_sha256=expected_authority_document_sha256,
    )
    lock_path = _regular_file_inside_project(
        root,
        root / authority.lock_path,
        "authority lock path",
    )
    package_lock = load_python_package_lock_v2(
        lock_path,
        expected_lock_id=authority.lock_id,
        expected_document_sha256=authority.lock_document_sha256,
    )
    if (
        package_lock.profile != authority.profile
        or package_lock.packages != authority.locked_packages
        or package_lock.transitive_artifact_hashes_present
        != authority.transitive_artifact_hashes_present
    ):
        raise PythonEnvironmentAuthorityError(
            "authority_lock_binding_mismatch",
            authority.authority_id,
        )
    observed = observe_installed_python_packages_v2(
        tuple(name for name, _ in package_lock.packages),
        version_provider=version_provider,
    )
    if observed != authority.observed_packages:
        raise PythonEnvironmentAuthorityError(
            "package_version_mismatch",
            _package_diff(authority.observed_packages, observed),
        )
    current = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_build": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    expected = {
        name: cast(str, getattr(authority, name))
        for name in current
    }
    if current != expected:
        detail = ",".join(
            f"{name}={expected[name]}->{current[name]}"
            for name in sorted(current)
            if current[name] != expected[name]
        )
        raise PythonEnvironmentAuthorityError(
            "python_runtime_mismatch",
            detail,
        )
    return authority


def write_python_package_lock_v2(
    package_lock: PythonPackageLockV2,
    output_root: str | Path,
) -> Path:
    if type(package_lock) is not PythonPackageLockV2:
        raise TypeError("package_lock must be exact PythonPackageLockV2")
    return cast(
        Path,
        write_immutable_json_document(
            output_root,
            filename=f"{package_lock.lock_id}{_LOCK_SUFFIX}",
            value=package_lock.to_dict(),
        ),
    )


def load_python_package_lock_v2(
    path: str | Path,
    *,
    expected_lock_id: str,
    expected_document_sha256: str,
) -> PythonPackageLockV2:
    lock_id = require_sha256(expected_lock_id, name="expected package lock id")
    try:
        value = load_immutable_json_document(
            path,
            expected_filename=f"{lock_id}{_LOCK_SUFFIX}",
            expected_document_sha256=expected_document_sha256,
            maximum_bytes=256 * 1024,
        )
    except ImmutableJsonError as error:
        raise PythonEnvironmentAuthorityError(
            f"package_lock_{error.code}",
            error.detail,
        ) from error
    try:
        result = PythonPackageLockV2.from_mapping(value)
    except (TypeError, ValueError) as error:
        raise PythonEnvironmentAuthorityError(
            "package_lock_schema_invalid",
            type(error).__name__,
        ) from error
    if result.lock_id != lock_id:
        raise PythonEnvironmentAuthorityError(
            "package_lock_id_mismatch",
            result.lock_id,
        )
    return result


def write_python_environment_authority_v2(
    authority: PythonEnvironmentAuthorityV2,
    output_root: str | Path,
) -> Path:
    if type(authority) is not PythonEnvironmentAuthorityV2:
        raise TypeError(
            "authority must be exact PythonEnvironmentAuthorityV2"
        )
    return cast(
        Path,
        write_immutable_json_document(
            output_root,
            filename=f"{authority.authority_id}{_AUTHORITY_SUFFIX}",
            value=authority.to_dict(),
        ),
    )


def load_python_environment_authority_v2(
    path: str | Path,
    *,
    expected_authority_id: str,
    expected_document_sha256: str,
) -> PythonEnvironmentAuthorityV2:
    authority_id = require_sha256(
        expected_authority_id,
        name="expected Python environment authority id",
    )
    try:
        value = load_immutable_json_document(
            path,
            expected_filename=f"{authority_id}{_AUTHORITY_SUFFIX}",
            expected_document_sha256=expected_document_sha256,
            maximum_bytes=512 * 1024,
        )
    except ImmutableJsonError as error:
        raise PythonEnvironmentAuthorityError(
            f"environment_authority_{error.code}",
            error.detail,
        ) from error
    try:
        result = PythonEnvironmentAuthorityV2.from_mapping(value)
    except (TypeError, ValueError) as error:
        raise PythonEnvironmentAuthorityError(
            "environment_authority_schema_invalid",
            type(error).__name__,
        ) from error
    if result.authority_id != authority_id:
        raise PythonEnvironmentAuthorityError(
            "environment_authority_id_mismatch",
            result.authority_id,
        )
    return result


def _canonical_packages(
    packages: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    if not packages:
        raise ValueError("package pins must not be empty")
    output: list[tuple[str, str]] = []
    for item in packages:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError("each package pin must be an exact pair")
        name, version = item
        canonical_name = _canonical_package_name(name)
        if canonical_name != name:
            raise ValueError("package name must already be canonical")
        output.append((canonical_name, _exact_version(version, name=name)))
    ordered = tuple(sorted(output))
    if len({name for name, _ in ordered}) != len(ordered):
        raise ValueError("package pins contain duplicate names")
    return ordered


def _packages_from_wire(value: object, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{name} must be a nonempty object")
    pairs: list[tuple[str, str]] = []
    for raw_name, raw_version in value.items():
        if type(raw_name) is not str or type(raw_version) is not str:
            raise TypeError(f"{name} keys and values must be strings")
        pairs.append((raw_name, raw_version))
    return _canonical_packages(tuple(sorted(pairs)))


def _canonical_package_name(value: object) -> str:
    if type(value) is not str:
        raise TypeError("package name must be a string")
    text = _PACKAGE_SEPARATOR_PATTERN.sub("-", value).lower()
    if (
        not text
        or text != value
        or text[0] == "-"
        or text[-1] == "-"
        or any(not (char.isascii() and (char.isalnum() or char == "-")) for char in text)
    ):
        raise ValueError("package name must use canonical PEP 503 form")
    return text


def _exact_version(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} version must be a string")
    if (
        not value
        or value.strip() != value
        or any(char in _FORBIDDEN_VERSION_CHARACTERS for char in value)
    ):
        raise ValueError(f"{name} must have one exact installed version")
    return value


def _require_profile(value: object) -> str:
    if type(value) is not str or not _PROFILE_PATTERN.fullmatch(value):
        raise ValueError("profile must be a canonical lowercase identifier")
    return value


def _require_project_relative_path(value: object) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError("lock path must be canonical project-relative POSIX")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("lock path must be canonical project-relative POSIX")
    return value


def _regular_project_root(value: str | Path) -> Path:
    root = Path(os.path.abspath(os.fspath(Path(value).expanduser())))
    if not root.is_dir() or root.is_symlink():
        raise PythonEnvironmentAuthorityError(
            "project_root_invalid",
            os.fspath(root),
        )
    return root


def _regular_file_inside_project(
    project_root: Path,
    value: str | Path,
    name: str,
) -> Path:
    candidate = Path(value).expanduser()
    path = (
        candidate
        if candidate.is_absolute()
        else project_root / candidate
    )
    path = Path(os.path.abspath(os.fspath(path)))
    try:
        path.relative_to(project_root)
    except ValueError as error:
        raise PythonEnvironmentAuthorityError(
            "path_outside_project",
            f"{name}:{path}",
        ) from error
    if not path.is_file() or path.is_symlink():
        raise PythonEnvironmentAuthorityError(
            "file_invalid",
            f"{name}:{path}",
        )
    return path


def _package_diff(
    expected: Sequence[tuple[str, str]],
    observed: Sequence[tuple[str, str]],
) -> str:
    expected_map = dict(expected)
    observed_map = dict(observed)
    names = sorted(set(expected_map) | set(observed_map))
    return ",".join(
        f"{name}={expected_map.get(name, '<missing>')}"
        f"->{observed_map.get(name, '<missing>')}"
        for name in names
        if expected_map.get(name) != observed_map.get(name)
    )


def _text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    return value


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


__all__ = [
    "L2_CORE_PACKAGE_NAMES",
    "PythonEnvironmentAuthorityError",
    "PythonEnvironmentAuthorityV2",
    "PythonPackageLockV2",
    "build_current_python_environment_authority_v2",
    "load_python_environment_authority_v2",
    "load_python_package_lock_v2",
    "observe_installed_python_packages_v2",
    "verify_current_python_environment_authority_v2",
    "write_python_environment_authority_v2",
    "write_python_package_lock_v2",
]
