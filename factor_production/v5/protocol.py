from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from factor_production.v5.artifacts.hashing import canonical_json_bytes, hash_file, hash_json
from factor_production.v5.domain.models import CANDIDATE_SPEC_SCHEMA
from factor_production.v5.orchestration.budget import BudgetLimits
from factor_production.v5.orchestration.stopping import StoppingConfig


PROTOCOL_SCHEMA = "alpha-mining-protocol/v5"


class ProtocolError(ValueError):
    pass


def _freeze_json(value: Any, *, path: str = "$.metadata") -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolError(f"non-finite metadata number at {path}")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProtocolError(f"metadata key at {path} must be a string")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, path=f"{path}[]") for item in value)
    raise ProtocolError(f"metadata value at {path} is not JSON-compatible")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ExternalEvaluationPolicy:
    mode: str = "file_drop"
    feedback_policy: str = "holdout_only"
    may_influence_search: bool = False

    def __post_init__(self) -> None:
        if self.mode != "file_drop":
            raise ProtocolError("V5 external evaluation mode must be 'file_drop'")
        if self.feedback_policy != "holdout_only":
            raise ProtocolError("external evaluation feedback_policy must be 'holdout_only'")
        if self.may_influence_search is not False:
            raise ProtocolError(
                "external teacher/official scores must not influence proposal, selection, or stopping"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "feedback_policy": self.feedback_policy,
            "may_influence_search": self.may_influence_search,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExternalEvaluationPolicy":
        expected = {"mode", "feedback_policy", "may_influence_search"}
        if set(value) != expected:
            raise ProtocolError(
                f"external_evaluation fields must be exactly {sorted(expected)}; got {sorted(value)}"
            )
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class V5Protocol:
    protocol_id: str
    random_seed: int
    allowed_fields: tuple[str, ...]
    allowed_operators: tuple[str, ...]
    allowed_windows: tuple[int, ...]
    max_expression_depth: int
    budget: BudgetLimits
    stopping: StoppingConfig
    external_evaluation: ExternalEvaluationPolicy
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = PROTOCOL_SCHEMA
    candidate_schema_version: str = CANDIDATE_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PROTOCOL_SCHEMA:
            raise ProtocolError(f"unsupported protocol schema: {self.schema_version!r}")
        if self.candidate_schema_version != CANDIDATE_SPEC_SCHEMA:
            raise ProtocolError(
                f"candidate schema must be pinned to {CANDIDATE_SPEC_SCHEMA!r}"
            )
        if not self.protocol_id.strip():
            raise ProtocolError("protocol_id must not be empty")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise ProtocolError("random_seed must be an integer")
        if not isinstance(self.max_expression_depth, int) or self.max_expression_depth <= 0:
            raise ProtocolError("max_expression_depth must be a positive integer")
        for name in ("allowed_fields", "allowed_operators", "allowed_windows"):
            values = tuple(getattr(self, name))
            if not values or len(values) != len(set(values)):
                raise ProtocolError(f"{name} must be non-empty and unique")
            object.__setattr__(self, name, values)
        if any(not isinstance(item, str) or not item.strip() for item in self.allowed_fields):
            raise ProtocolError("allowed_fields entries must be non-empty strings")
        if any(not isinstance(item, str) or not item.strip() for item in self.allowed_operators):
            raise ProtocolError("allowed_operators entries must be non-empty strings")
        if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in self.allowed_windows):
            raise ProtocolError("allowed_windows entries must be positive integers")
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_schema_version": self.candidate_schema_version,
            "protocol_id": self.protocol_id,
            "random_seed": self.random_seed,
            "allowed_fields": list(self.allowed_fields),
            "allowed_operators": list(self.allowed_operators),
            "allowed_windows": list(self.allowed_windows),
            "max_expression_depth": self.max_expression_depth,
            "budget": self.budget.to_dict(),
            "stopping": self.stopping.to_dict(),
            "external_evaluation": self.external_evaluation.to_dict(),
            "metadata": _thaw_json(self.metadata),
        }

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V5Protocol":
        expected = {
            "schema_version",
            "candidate_schema_version",
            "protocol_id",
            "random_seed",
            "allowed_fields",
            "allowed_operators",
            "allowed_windows",
            "max_expression_depth",
            "budget",
            "stopping",
            "external_evaluation",
            "metadata",
        }
        optional = {"metadata"}
        unknown = sorted(set(value) - expected)
        missing = sorted((expected - optional) - set(value))
        if unknown or missing:
            raise ProtocolError(f"protocol fields invalid; unknown={unknown}, missing={missing}")
        stopping = value["stopping"]
        if set(stopping) != {"patience_generations", "min_improvement"}:
            raise ProtocolError("stopping fields must be patience_generations and min_improvement")
        return cls(
            schema_version=value["schema_version"],
            candidate_schema_version=value["candidate_schema_version"],
            protocol_id=value["protocol_id"],
            random_seed=value["random_seed"],
            allowed_fields=tuple(value["allowed_fields"]),
            allowed_operators=tuple(value["allowed_operators"]),
            allowed_windows=tuple(value["allowed_windows"]),
            max_expression_depth=value["max_expression_depth"],
            budget=BudgetLimits.from_dict(value["budget"]),
            stopping=StoppingConfig(**dict(stopping)),
            external_evaluation=ExternalEvaluationPolicy.from_dict(value["external_evaluation"]),
            metadata=value.get("metadata", {}),
        )


def load_protocol(path: str | Path, *, expected_file_hash: str | None = None) -> V5Protocol:
    source = Path(path)
    if expected_file_hash is not None:
        actual = hash_file(source)
        if actual != expected_file_hash:
            raise ProtocolError(f"protocol file hash mismatch: expected {expected_file_hash}, got {actual}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot load protocol {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("protocol root must be a JSON object")
    return V5Protocol.from_dict(payload)


def write_protocol(protocol: V5Protocol, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(protocol.to_dict()) + b"\n")
    return destination


def default_protocol(protocol_id: str) -> V5Protocol:
    return V5Protocol(
        protocol_id=protocol_id,
        random_seed=20260716,
        allowed_fields=("Returns", "Amount", "Volume", "Close", "VWAP"),
        allowed_operators=(
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Neg",
            "CsRank",
            "TsMean",
            "TsStd",
            "TsDelta",
            "TsCorr",
            "TsSkew",
            "TsKurt",
        ),
        allowed_windows=(3, 5, 10, 20, 30, 60),
        max_expression_depth=5,
        budget=BudgetLimits(
            max_candidates=500,
            max_generations=6,
            max_provider_calls=60,
            max_evaluations=500,
            max_wall_seconds=86_400,
        ),
        stopping=StoppingConfig(patience_generations=2, min_improvement=0.0005),
        external_evaluation=ExternalEvaluationPolicy(),
        metadata={"purpose": "production research; external scores are holdout-only"},
    )
