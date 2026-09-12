"""Capability marker for archive-separated Python capsule execution."""

from __future__ import annotations

from typing import Final


PYTHON_RUNTIME_CAPSULE_EXECUTION_MIRROR_PROTOCOL_V1: Final[str] = (
    "python-runtime-capsule-execution-mirror/v1"
)
PYTHON_RUNTIME_CAPSULE_EXECUTION_MIRROR_MARKER_PATH: Final[str] = (
    "alpha_research/core/python_runtime_capsule_execution_protocol.py"
)


__all__ = [
    "PYTHON_RUNTIME_CAPSULE_EXECUTION_MIRROR_MARKER_PATH",
    "PYTHON_RUNTIME_CAPSULE_EXECUTION_MIRROR_PROTOCOL_V1",
]
