from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import pandas as pd

from alpha_research.core.hashing import hash_json, require_sha256


@dataclass(frozen=True, slots=True)
class ExperimentReceipt:
    experiment_spec_hash: str
    terminal_status: str
    component_bindings: Mapping[str, str]
    stage_result_hashes: Mapping[str, str]
    artifact_hashes: tuple[str, ...]
    metrics_artifact_hash: str | None
    started_at: str
    finished_at: str
    random_seed: int
    code_snapshot_hash: str
    environment_hash: str
    production_ready: bool
    approval_hash: str | None = None
    schema_version: str = "experiment-receipt/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "experiment-receipt/v1":
            raise ValueError("unsupported ExperimentReceipt schema")
        require_sha256(self.experiment_spec_hash, name="receipt experiment_spec_hash")
        if self.terminal_status not in {"completed", "failed", "cancelled"}:
            raise ValueError("receipt status must be terminal")
        components = dict(sorted(self.component_bindings.items()))
        stages = dict(self.stage_result_hashes)
        if not components:
            raise ValueError("receipt must bind experiment components")
        for role, digest in components.items():
            if not role.strip():
                raise ValueError("receipt component role is empty")
            require_sha256(digest, name=f"receipt component:{role}")
        for stage, digest in stages.items():
            if not stage.strip():
                raise ValueError("receipt stage name is empty")
            require_sha256(digest, name=f"receipt stage:{stage}")
        if len(set(self.artifact_hashes)) != len(self.artifact_hashes):
            raise ValueError("receipt artifact hashes must be unique")
        if tuple(sorted(self.artifact_hashes)) != self.artifact_hashes:
            raise ValueError("receipt artifact hashes must be sorted")
        for digest in self.artifact_hashes:
            require_sha256(digest, name="receipt artifact hash")
        if self.metrics_artifact_hash is not None:
            require_sha256(
                self.metrics_artifact_hash, name="receipt metrics_artifact_hash"
            )
            if self.metrics_artifact_hash not in self.artifact_hashes:
                raise ValueError("receipt metrics artifact is absent from artifact set")
        started = pd.Timestamp(self.started_at)
        finished = pd.Timestamp(self.finished_at)
        if started.tzinfo is None or finished.tzinfo is None:
            raise ValueError("receipt timestamps must be timezone-aware")
        if finished < started:
            raise ValueError("receipt finished_at precedes started_at")
        object.__setattr__(self, "started_at", started.isoformat())
        object.__setattr__(self, "finished_at", finished.isoformat())
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise TypeError("receipt random_seed must be an integer")
        require_sha256(self.code_snapshot_hash, name="receipt code_snapshot_hash")
        require_sha256(self.environment_hash, name="receipt environment_hash")
        if not isinstance(self.production_ready, bool):
            raise TypeError("receipt production_ready must be boolean")
        if self.approval_hash is not None:
            require_sha256(self.approval_hash, name="receipt approval_hash")
        if self.production_ready:
            if self.terminal_status != "completed":
                raise ValueError("only a completed receipt may be production-ready")
            if self.approval_hash is None:
                raise ValueError("production-ready receipt requires human approval")
        object.__setattr__(self, "component_bindings", MappingProxyType(components))
        object.__setattr__(self, "stage_result_hashes", MappingProxyType(stages))

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_spec_hash": self.experiment_spec_hash,
            "terminal_status": self.terminal_status,
            "component_bindings": dict(self.component_bindings),
            "stage_result_hashes": dict(self.stage_result_hashes),
            "artifact_hashes": list(self.artifact_hashes),
            "metrics_artifact_hash": self.metrics_artifact_hash,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "random_seed": self.random_seed,
            "code_snapshot_hash": self.code_snapshot_hash,
            "environment_hash": self.environment_hash,
            "production_ready": self.production_ready,
            "approval_hash": self.approval_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ExperimentReceipt":
        expected = {
            "schema_version",
            "experiment_spec_hash",
            "terminal_status",
            "component_bindings",
            "stage_result_hashes",
            "artifact_hashes",
            "metrics_artifact_hash",
            "started_at",
            "finished_at",
            "random_seed",
            "code_snapshot_hash",
            "environment_hash",
            "production_ready",
            "approval_hash",
        }
        if set(value) != expected:
            raise ValueError("experiment receipt schema differs")
        components = value["component_bindings"]
        stages = value["stage_result_hashes"]
        artifacts = value["artifact_hashes"]
        if not isinstance(components, Mapping) or not isinstance(stages, Mapping):
            raise TypeError("receipt bindings must be objects")
        if not isinstance(artifacts, list):
            raise TypeError("receipt artifact_hashes must be a list")
        if not isinstance(value["production_ready"], bool):
            raise TypeError("receipt production_ready must be boolean")
        return cls(
            schema_version=str(value["schema_version"]),
            experiment_spec_hash=str(value["experiment_spec_hash"]),
            terminal_status=str(value["terminal_status"]),
            component_bindings={str(k): str(v) for k, v in components.items()},
            stage_result_hashes={str(k): str(v) for k, v in stages.items()},
            artifact_hashes=tuple(str(item) for item in artifacts),
            metrics_artifact_hash=(
                None
                if value["metrics_artifact_hash"] is None
                else str(value["metrics_artifact_hash"])
            ),
            started_at=str(value["started_at"]),
            finished_at=str(value["finished_at"]),
            random_seed=int(str(value["random_seed"])),
            code_snapshot_hash=str(value["code_snapshot_hash"]),
            environment_hash=str(value["environment_hash"]),
            production_ready=value["production_ready"],
            approval_hash=(
                None if value["approval_hash"] is None else str(value["approval_hash"])
            ),
        )


__all__ = ["ExperimentReceipt"]
