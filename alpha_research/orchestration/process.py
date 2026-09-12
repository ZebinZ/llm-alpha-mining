from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping, cast

import psutil

from alpha_research.core.hashing import hash_json
from alpha_research.orchestration.runtime import RuntimeUsage, reported_usage


_TASK_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")
_ENTRYPOINT = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$"
)


class ProcessExecutionStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ProcessTask:
    task_id: str
    entrypoint: str
    payload: object
    timeout_seconds: float
    cpu_limit_seconds: int
    memory_limit_bytes: int
    output_limit_bytes: int = 1_000_000
    schema_version: str = "process-task/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "process-task/v1":
            raise ValueError("unsupported ProcessTask schema")
        if not _TASK_ID.fullmatch(self.task_id):
            raise ValueError("process task_id is invalid")
        if not _ENTRYPOINT.fullmatch(self.entrypoint):
            raise ValueError("process task entrypoint is invalid")
        hash_json(self.payload)
        if (
            not isinstance(self.timeout_seconds, (int, float))
            or isinstance(self.timeout_seconds, bool)
            or not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("process timeout_seconds must be a finite positive number")
        for name in ("cpu_limit_seconds", "memory_limit_bytes", "output_limit_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"process {name} must be a positive integer")

    @property
    def content_hash(self) -> str:
        return cast(
            str,
            hash_json(
                {
                    "schema_version": self.schema_version,
                    "task_id": self.task_id,
                    "entrypoint": self.entrypoint,
                    "payload": self.payload,
                    "timeout_seconds": self.timeout_seconds,
                    "cpu_limit_seconds": self.cpu_limit_seconds,
                    "memory_limit_bytes": self.memory_limit_bytes,
                    "output_limit_bytes": self.output_limit_bytes,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ProcessExecutionResult:
    task_hash: str
    status: ProcessExecutionStatus
    output: object | None
    output_hash: str | None
    failure_code: str | None
    diagnostic_hash: str
    return_code: int
    usage: RuntimeUsage
    worker_error_type: str | None = None
    worker_error_detail_sha256: str | None = None


class IsolatedProcessExecutor:
    """Runs exact allowlisted Python entrypoints in a constrained child process.

    This provides process and resource isolation. It deliberately does not claim to
    be an operating-system filesystem sandbox; generated factor logic must still
    enter through the controlled DSL instead of becoming an allowlisted entrypoint.
    """

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        allowed_entrypoints: tuple[str, ...],
        poll_interval_seconds: float = 0.05,
    ) -> None:
        root = Path(workspace_root).resolve()
        if not root.is_dir():
            raise ValueError("process workspace_root must be an existing directory")
        allowed = tuple(sorted(set(allowed_entrypoints)))
        if not allowed or len(allowed) != len(allowed_entrypoints):
            raise ValueError(
                "process entrypoint allowlist must be non-empty and unique"
            )
        if any(not _ENTRYPOINT.fullmatch(value) for value in allowed):
            raise ValueError("process entrypoint allowlist contains an invalid value")
        if (
            not isinstance(poll_interval_seconds, (int, float))
            or isinstance(poll_interval_seconds, bool)
            or not math.isfinite(float(poll_interval_seconds))
            or poll_interval_seconds <= 0
            or poll_interval_seconds > 1
        ):
            raise ValueError("process poll interval must be finite and in (0,1]")
        self.workspace_root = root
        self.allowed_entrypoints = allowed
        self.poll_interval_seconds = float(poll_interval_seconds)
        self._bootstrap = Path(__file__).with_name("worker_bootstrap.py").resolve()

    def execute(
        self,
        task: ProcessTask,
        *,
        should_cancel: Callable[[], bool] | None = None,
        heartbeat: Callable[[], None] | None = None,
    ) -> ProcessExecutionResult:
        if task.entrypoint not in self.allowed_entrypoints:
            raise PermissionError("process task entrypoint is not allowlisted")
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="alpha-worker-") as temporary:
            root = Path(temporary)
            input_path = root / "input.json"
            output_path = root / "output.json"
            stdout_path = root / "stdout.log"
            stderr_path = root / "stderr.log"
            envelope = {
                "schema_version": "process-worker-envelope/v1",
                "task_hash": task.content_hash,
                "entrypoint": task.entrypoint,
                "payload": task.payload,
                "output_limit_bytes": task.output_limit_bytes,
            }
            input_path.write_text(
                json.dumps(envelope, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            environment = _minimal_environment(root)
            command = (
                sys.executable,
                "-I",
                str(self._bootstrap),
                str(input_path),
                str(output_path),
            )
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    command,
                    cwd=root,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                    close_fds=True,
                    preexec_fn=_limits(
                        task.cpu_limit_seconds,
                        task.output_limit_bytes,
                    ),
                )
                process_monitor = psutil.Process(process.pid)
                peak_rss = 0
                state = ProcessExecutionStatus.FAILED
                failure: str | None = "worker_process_exit"
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    peak_rss = max(peak_rss, _resident_bytes(process_monitor))
                    if peak_rss > task.memory_limit_bytes:
                        state = ProcessExecutionStatus.FAILED
                        failure = "memory_limit"
                        _terminate_group(process)
                        break
                    if should_cancel is not None and should_cancel():
                        state = ProcessExecutionStatus.CANCELLED
                        failure = "kill_switch"
                        _terminate_group(process)
                        break
                    if elapsed >= task.timeout_seconds:
                        state = ProcessExecutionStatus.TIMED_OUT
                        failure = "stage_timeout"
                        _terminate_group(process)
                        break
                    if heartbeat is not None:
                        heartbeat()
                    time.sleep(self.poll_interval_seconds)
                return_code = process.wait()
            wall = time.monotonic() - started
            response = _load_response(output_path, task.output_limit_bytes)
            cpu_seconds = _non_negative_float(response, "cpu_seconds")
            peak_memory = max(
                peak_rss, _non_negative_int(response, "peak_memory_bytes")
            )
            if failure not in {"kill_switch", "stage_timeout", "memory_limit"}:
                if (
                    return_code == 0
                    and response is not None
                    and response.get("ok") is True
                ):
                    state = ProcessExecutionStatus.SUCCEEDED
                    failure = None
                else:
                    state = ProcessExecutionStatus.FAILED
                    if return_code == -signal.SIGXCPU:
                        failure = "cpu_limit"
                    else:
                        failure = (
                            str(response.get("failure_code"))
                            if response is not None and response.get("failure_code")
                            else "worker_process_exit"
                        )
            # A child can allocate and exit between two parent polling
            # samples.  The bootstrap's ru_maxrss closes that observation gap;
            # a late receipt must never turn a hard resource breach into
            # success.
            if (
                state is not ProcessExecutionStatus.CANCELLED
                and peak_memory > task.memory_limit_bytes
            ):
                state = ProcessExecutionStatus.FAILED
                failure = "memory_limit"
            elif (
                state is not ProcessExecutionStatus.CANCELLED
                and cpu_seconds > task.cpu_limit_seconds
            ):
                state = ProcessExecutionStatus.FAILED
                failure = "cpu_limit"
            elif (
                state is ProcessExecutionStatus.SUCCEEDED
                and wall >= task.timeout_seconds
            ):
                state = ProcessExecutionStatus.TIMED_OUT
                failure = "stage_timeout"
            output = None if response is None else response.get("result")
            output_hash = (
                hash_json(output) if state is ProcessExecutionStatus.SUCCEEDED else None
            )
            disk_write = sum(
                path.stat().st_size
                for path in (input_path, output_path, stdout_path, stderr_path)
                if path.exists()
            )
            usage = reported_usage(
                wall_seconds=wall,
                cpu_seconds=cpu_seconds,
                peak_memory_bytes=peak_memory,
                disk_write_bytes=disk_write,
            )
            diagnostic_hash = hash_json(
                {
                    "task_hash": task.content_hash,
                    "status": state.value,
                    "failure_code": failure,
                    "return_code": return_code,
                    "stdout_hash": _file_hash(stdout_path),
                    "stderr_hash": _file_hash(stderr_path),
                    "response_hash": None if response is None else hash_json(response),
                }
            )
            worker_error_type = _worker_error_type(response, state)
            worker_error_detail_sha256 = _worker_error_detail_sha256(
                response,
                state,
            )
            return ProcessExecutionResult(
                task_hash=task.content_hash,
                status=state,
                output=output if state is ProcessExecutionStatus.SUCCEEDED else None,
                output_hash=output_hash,
                failure_code=failure,
                diagnostic_hash=diagnostic_hash,
                return_code=return_code,
                usage=usage,
                worker_error_type=worker_error_type,
                worker_error_detail_sha256=worker_error_detail_sha256,
            )


def _minimal_environment(temporary: Path) -> Mapping[str, str]:
    return {
        "HOME": str(temporary),
        "TMPDIR": str(temporary),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def _limits(
    cpu_seconds: int,
    output_bytes: int,
) -> Callable[[], None] | None:
    if os.name != "posix":  # pragma: no cover - the research runtime targets POSIX
        return None

    def apply() -> None:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (output_bytes, output_bytes))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

    return apply


def _resident_bytes(process: psutil.Process) -> int:
    """Return the observable process-tree RSS without requiring global PID access.

    Some managed macOS runtimes permit reading the launched child's own RSS but
    deny ``sysctl(KERN_PROC_ALL)``, which psutil uses while discovering
    descendants.  Treat that discovery denial like psutil's ``AccessDenied``
    and retain the directly observed child RSS.  The worker bootstrap's
    post-exit ``ru_maxrss`` receipt remains the second, independent hard-limit
    check for allocations between polling samples.
    """

    try:
        resident = int(process.memory_info().rss)
    except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
        return 0
    try:
        children = process.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
        return resident
    for child in children:
        try:
            if child.is_running():
                resident += int(child.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
            continue
    return resident


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=1.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)


def _load_response(path: Path, maximum_bytes: int) -> dict[str, object] | None:
    if not path.is_file() or path.stat().st_size > maximum_bytes:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _non_negative_float(response: dict[str, object] | None, key: str) -> float:
    value = None if response is None else response.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return 0.0


def _non_negative_int(response: dict[str, object] | None, key: str) -> int:
    value = None if response is None else response.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _worker_error_type(
    response: dict[str, object] | None,
    state: ProcessExecutionStatus,
) -> str | None:
    """Retain only the child's exception class, never its message or traceback."""

    if state is ProcessExecutionStatus.SUCCEEDED or response is None:
        return None
    value = response.get("error_type")
    if (
        type(value) is str
        and 1 <= len(value) <= 128
        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", value) is not None
    ):
        return value
    return None


def _worker_error_detail_sha256(
    response: dict[str, object] | None,
    state: ProcessExecutionStatus,
) -> str | None:
    """Retain an already-sanitized domain-detail digest when one exists."""

    if state is ProcessExecutionStatus.SUCCEEDED or response is None:
        return None
    value = response.get("error_detail_sha256")
    if type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None:
        return value
    return None


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "IsolatedProcessExecutor",
    "ProcessExecutionResult",
    "ProcessExecutionStatus",
    "ProcessTask",
]
