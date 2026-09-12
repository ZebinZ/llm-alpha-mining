from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import cast

from llm_alpha_mining.research.core.frequency import FrequencySpec
from llm_alpha_mining.research.core.hashing import hash_json, require_sha256
from llm_alpha_mining.mining.llm.schemas import canonical_expression_ast


class FactorDirection(IntEnum):
    NEGATIVE = -1
    POSITIVE = 1


class PreprocessKind(str, Enum):
    WINSORIZE_QUANTILE = "winsorize_quantile"
    ZSCORE = "zscore"
    NEUTRALIZE_OLS = "neutralize_ols"


@dataclass(frozen=True, slots=True)
class AggregationSpec:
    method: str = "none"
    window: str = "native"
    smoothing_span: int = 0
    reducer: str = "last"

    def __post_init__(self) -> None:
        _text(self.method, name="aggregation.method")
        _text(self.window, name="aggregation.window")
        _text(self.reducer, name="aggregation.reducer")
        if self.method not in {"none", "calendar_daily"}:
            raise ValueError(f"unsupported factor aggregation:{self.method}")
        if not self.window:
            raise ValueError("factor aggregation window must not be empty")
        if not isinstance(self.smoothing_span, int) or isinstance(
            self.smoothing_span, bool
        ):
            raise TypeError("FactorSpec aggregation.smoothing_span must be an integer")
        if self.smoothing_span < 0:
            raise ValueError("factor smoothing span must be non-negative")
        if self.reducer not in {"last", "mean", "sum", "std", "skew", "kurt"}:
            raise ValueError(f"unsupported factor aggregation reducer:{self.reducer}")
        if self.method == "none" and (
            self.window != "native" or self.smoothing_span or self.reducer != "last"
        ):
            raise ValueError("native aggregation cannot declare transforms")
        if self.method == "calendar_daily" and self.window not in {
            "full_day",
            "open30",
            "midday30",
            "postlunch30",
            "close30",
            "close60",
        }:
            raise ValueError("calendar-daily aggregation window is unsupported")

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "window": self.window,
            "smoothing_span": self.smoothing_span,
            "reducer": self.reducer,
        }


@dataclass(frozen=True, slots=True)
class PreprocessStep:
    kind: PreprocessKind | str
    lower: float | None = None
    upper: float | None = None
    exposure_fields: tuple[str, ...] = ()
    ridge: float = 0.0

    def __post_init__(self) -> None:
        kind = self.kind
        if not isinstance(kind, (PreprocessKind, str)):
            raise TypeError("FactorSpec preprocessing.kind must be text")
        object.__setattr__(self, "kind", PreprocessKind(kind))
        if type(self.exposure_fields) is not tuple or not all(
            isinstance(item, str) for item in self.exposure_fields
        ):
            raise TypeError(
                "FactorSpec preprocessing.exposure_fields must be a tuple of strings"
            )
        exposures = tuple(sorted(item.strip() for item in self.exposure_fields))
        if any(not item for item in exposures) or len(exposures) != len(set(exposures)):
            raise ValueError(
                "neutralization exposure fields must be unique and non-empty"
            )
        object.__setattr__(self, "exposure_fields", exposures)
        lower = _optional_number(self.lower, name="preprocessing.lower")
        upper = _optional_number(self.upper, name="preprocessing.upper")
        ridge = _number(self.ridge, name="preprocessing.ridge")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "ridge", ridge)
        if ridge < 0:
            raise ValueError("preprocessing ridge must be non-negative")
        if self.kind is PreprocessKind.WINSORIZE_QUANTILE:
            if lower is None or upper is None:
                raise ValueError("winsorization requires lower and upper quantiles")
            if not 0 <= lower < upper <= 1:
                raise ValueError("winsorization quantiles are invalid")
            if exposures or ridge:
                raise ValueError("winsorization does not accept exposure parameters")
        elif self.kind is PreprocessKind.ZSCORE:
            if lower is not None or upper is not None or exposures or ridge:
                raise ValueError("zscore does not accept parameters")
        else:
            if lower is not None or upper is not None:
                raise ValueError("neutralization does not accept quantile parameters")
            if not exposures:
                raise ValueError("neutralization requires exposure fields")

    def to_dict(self) -> dict[str, object]:
        kind = self.kind
        if not isinstance(kind, PreprocessKind):  # pragma: no cover
            raise RuntimeError("preprocessing kind was not normalized")
        return {
            "kind": kind.value,
            "lower": self.lower,
            "upper": self.upper,
            "exposure_fields": list(self.exposure_fields),
            "ridge": self.ridge,
        }


@dataclass(frozen=True, slots=True)
class FactorProvenance:
    origin: str
    actor_id: str
    model_id: str | None = None
    prompt_hash: str | None = None
    parent_factor_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.origin, name="provenance.origin")
        _text(self.actor_id, name="provenance.actor_id")
        _optional_text(self.model_id, name="provenance.model_id")
        _optional_text(self.prompt_hash, name="provenance.prompt_hash")
        if type(self.parent_factor_hashes) is not tuple or not all(
            isinstance(item, str) for item in self.parent_factor_hashes
        ):
            raise TypeError(
                "FactorSpec provenance.parent_factor_hashes must be a tuple of strings"
            )
        if self.origin not in {"human", "llm", "mutation", "crossover", "migration"}:
            raise ValueError(f"unknown factor origin:{self.origin}")
        if not self.actor_id.strip():
            raise ValueError("factor provenance actor_id must not be empty")
        if self.origin == "llm" and (not self.model_id or not self.prompt_hash):
            raise ValueError("LLM factors require model_id and prompt_hash")
        if self.prompt_hash is not None:
            require_sha256(self.prompt_hash, name="factor prompt_hash")
        for digest in self.parent_factor_hashes:
            require_sha256(digest, name="parent factor hash")
        if len(self.parent_factor_hashes) != len(set(self.parent_factor_hashes)):
            raise ValueError("parent factor hashes must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "actor_id": self.actor_id,
            "model_id": self.model_id,
            "prompt_hash": self.prompt_hash,
            "parent_factor_hashes": list(self.parent_factor_hashes),
        }


@dataclass(frozen=True, slots=True)
class ComplexityBudget:
    maximum_ast_nodes: int = 64
    maximum_call_depth: int = 6
    maximum_window: int = 252
    maximum_fields: int = 8
    maximum_relative_ops_per_cell: int = 2048
    maximum_estimated_state_bytes_per_security: int = 16 * 1024 * 1024
    maximum_estimated_flops_per_cell: int = 4096

    def __post_init__(self) -> None:
        for name in (
            "maximum_ast_nodes",
            "maximum_call_depth",
            "maximum_window",
            "maximum_fields",
            "maximum_relative_ops_per_cell",
            "maximum_estimated_state_bytes_per_security",
            "maximum_estimated_flops_per_cell",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"factor complexity {name} must be positive")

    def to_dict(self) -> dict[str, int]:
        return {
            "maximum_ast_nodes": self.maximum_ast_nodes,
            "maximum_call_depth": self.maximum_call_depth,
            "maximum_window": self.maximum_window,
            "maximum_fields": self.maximum_fields,
            "maximum_relative_ops_per_cell": self.maximum_relative_ops_per_cell,
            "maximum_estimated_state_bytes_per_security": (
                self.maximum_estimated_state_bytes_per_security
            ),
            "maximum_estimated_flops_per_cell": self.maximum_estimated_flops_per_cell,
        }


@dataclass(frozen=True, slots=True)
class FactorSpec:
    factor_id: str
    version: str
    family: str
    hypothesis: str
    economic_rationale: str
    falsification_criterion: str
    expression: str
    direction: FactorDirection | int
    required_fields: tuple[str, ...]
    dataset_id: str
    snapshot_id: str
    schema_hash: str
    availability_hash: str
    security_contract_hash: str
    frequency: FrequencySpec
    aggregation: AggregationSpec
    operator_registry_version: str
    operator_registry_digest: str
    preprocessing: tuple[PreprocessStep, ...]
    complexity_budget: ComplexityBudget
    provenance: FactorProvenance
    schema_version: str = "factor-spec/v1"

    def __post_init__(self) -> None:
        _text(self.schema_version, name="schema_version")
        if self.schema_version != "factor-spec/v1":
            raise ValueError("unsupported FactorSpec schema")
        for name in (
            "factor_id",
            "version",
            "family",
            "hypothesis",
            "economic_rationale",
            "falsification_criterion",
            "expression",
            "dataset_id",
            "snapshot_id",
            "schema_hash",
            "availability_hash",
            "security_contract_hash",
            "operator_registry_version",
            "operator_registry_digest",
        ):
            value = _text(getattr(self, name), name=name)
            if not value.strip():
                raise ValueError(f"factor {name} must not be empty")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.factor_id):
            raise ValueError("factor_id has an invalid portable form")
        direction = self.direction
        if isinstance(direction, bool) or not isinstance(direction, int):
            raise TypeError("FactorSpec direction must be an integer")
        object.__setattr__(self, "direction", FactorDirection(direction))
        if type(self.required_fields) is not tuple or not all(
            isinstance(item, str) for item in self.required_fields
        ):
            raise TypeError("FactorSpec required_fields must be a tuple of strings")
        fields = tuple(sorted(item.strip() for item in self.required_fields))
        if (
            not fields
            or any(not item for item in fields)
            or len(fields) != len(set(fields))
        ):
            raise ValueError("factor required_fields must be unique and non-empty")
        future_tokens = (
            "future",
            "forward",
            "nextreturn",
            "next_return",
            "label",
            "target",
        )
        for field in fields:
            normalized = field.casefold().replace("-", "").replace(" ", "")
            if any(token in normalized for token in future_tokens):
                raise ValueError(f"future_or_label_field_forbidden:{field}")
        object.__setattr__(self, "required_fields", fields)
        require_sha256(self.snapshot_id, name="factor snapshot_id")
        require_sha256(self.schema_hash, name="factor schema_hash")
        require_sha256(self.availability_hash, name="factor availability_hash")
        require_sha256(
            self.security_contract_hash, name="factor security_contract_hash"
        )
        require_sha256(
            self.operator_registry_digest, name="factor operator_registry_digest"
        )
        if not isinstance(self.frequency, FrequencySpec):
            raise TypeError("FactorSpec frequency must be a FrequencySpec")
        if not isinstance(self.aggregation, AggregationSpec):
            raise TypeError("FactorSpec aggregation must be an AggregationSpec")
        if type(self.preprocessing) is not tuple or not all(
            isinstance(item, PreprocessStep) for item in self.preprocessing
        ):
            raise TypeError(
                "FactorSpec preprocessing must be a tuple of PreprocessStep values"
            )
        if not isinstance(self.complexity_budget, ComplexityBudget):
            raise TypeError("FactorSpec complexity_budget must be a ComplexityBudget")
        if not isinstance(self.provenance, FactorProvenance):
            raise TypeError("FactorSpec provenance must be a FactorProvenance")
        if len(self.preprocessing) != len(
            {
                (
                    item.kind,
                    item.lower,
                    item.upper,
                    item.exposure_fields,
                    item.ridge,
                )
                for item in self.preprocessing
            }
        ):
            raise ValueError("factor preprocessing steps contain duplicates")
        canonical_expression_ast(self.expression)

    @property
    def semantic_hash(self) -> str:
        return cast(
            str,
            hash_json(
                {
                    "expression_ast": canonical_expression_ast(self.expression),
                    "frequency": self.frequency.to_dict(),
                    "aggregation": self.aggregation.to_dict(),
                    "preprocessing": [item.to_dict() for item in self.preprocessing],
                    "operator_registry_digest": self.operator_registry_digest,
                },
            ),
        )

    @property
    def definition_hash(self) -> str:
        """Execution semantics, data timing, universe and transforms.

        Human-readable identity, rationale, direction and lineage are excluded;
        changing any executable data contract produces a new definition hash.
        """
        return cast(
            str,
            hash_json(
                {
                    "expression_ast": canonical_expression_ast(self.expression),
                    "required_fields": list(self.required_fields),
                    "dataset_id": self.dataset_id,
                    "snapshot_id": self.snapshot_id,
                    "schema_hash": self.schema_hash,
                    "availability_hash": self.availability_hash,
                    "security_contract_hash": self.security_contract_hash,
                    "frequency": self.frequency.to_dict(),
                    "aggregation": self.aggregation.to_dict(),
                    "operator_registry_version": self.operator_registry_version,
                    "operator_registry_digest": self.operator_registry_digest,
                    "preprocessing": [item.to_dict() for item in self.preprocessing],
                },
            ),
        )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "factor_id": self.factor_id,
            "version": self.version,
            "family": self.family,
            "hypothesis": self.hypothesis,
            "economic_rationale": self.economic_rationale,
            "falsification_criterion": self.falsification_criterion,
            "expression": self.expression,
            "direction": int(self.direction),
            "required_fields": list(self.required_fields),
            "dataset_id": self.dataset_id,
            "snapshot_id": self.snapshot_id,
            "schema_hash": self.schema_hash,
            "availability_hash": self.availability_hash,
            "security_contract_hash": self.security_contract_hash,
            "frequency": self.frequency.to_dict(),
            "aggregation": self.aggregation.to_dict(),
            "operator_registry_version": self.operator_registry_version,
            "operator_registry_digest": self.operator_registry_digest,
            "preprocessing": [item.to_dict() for item in self.preprocessing],
            "complexity_budget": self.complexity_budget.to_dict(),
            "provenance": self.provenance.to_dict(),
            "semantic_hash": self.semantic_hash,
            "definition_hash": self.definition_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FactorSpec":
        expected = {
            "schema_version",
            "factor_id",
            "version",
            "family",
            "hypothesis",
            "economic_rationale",
            "falsification_criterion",
            "expression",
            "direction",
            "required_fields",
            "dataset_id",
            "snapshot_id",
            "schema_hash",
            "availability_hash",
            "security_contract_hash",
            "frequency",
            "aggregation",
            "operator_registry_version",
            "operator_registry_digest",
            "preprocessing",
            "complexity_budget",
            "provenance",
            "semantic_hash",
            "definition_hash",
        }
        if set(value) != expected:
            raise ValueError("FactorSpec wire fields differ")
        frequency = _mapping(value["frequency"], name="frequency")
        aggregation = _mapping(value["aggregation"], name="aggregation")
        budget = _mapping(value["complexity_budget"], name="complexity_budget")
        provenance = _mapping(value["provenance"], name="provenance")
        _require_fields(
            frequency,
            {
                "mode",
                "interval",
                "calendar_id",
                "timezone",
                "session_id",
                "timestamp_policy",
                "sampling_policy",
            },
            name="frequency",
        )
        _require_fields(
            aggregation,
            {"method", "window", "smoothing_span", "reducer"},
            name="aggregation",
        )
        _require_fields(
            budget,
            {
                "maximum_ast_nodes",
                "maximum_call_depth",
                "maximum_window",
                "maximum_fields",
                "maximum_relative_ops_per_cell",
                "maximum_estimated_state_bytes_per_security",
                "maximum_estimated_flops_per_cell",
            },
            name="complexity_budget",
        )
        _require_fields(
            provenance,
            {
                "origin",
                "actor_id",
                "model_id",
                "prompt_hash",
                "parent_factor_hashes",
            },
            name="provenance",
        )
        raw_preprocessing = value["preprocessing"]
        if not isinstance(raw_preprocessing, list) or not all(
            isinstance(item, Mapping) for item in raw_preprocessing
        ):
            raise TypeError("FactorSpec preprocessing must be a list of objects")
        for item in raw_preprocessing:
            _require_fields(
                item,
                {"kind", "lower", "upper", "exposure_fields", "ridge"},
                name="preprocessing",
            )
        required_fields = _string_tuple(
            value["required_fields"], name="required_fields"
        )
        parent_hashes = _string_tuple(
            provenance["parent_factor_hashes"], name="parent_factor_hashes"
        )
        spec = cls(
            schema_version=_text(value["schema_version"], name="schema_version"),
            factor_id=_text(value["factor_id"], name="factor_id"),
            version=_text(value["version"], name="version"),
            family=_text(value["family"], name="family"),
            hypothesis=_text(value["hypothesis"], name="hypothesis"),
            economic_rationale=_text(
                value["economic_rationale"], name="economic_rationale"
            ),
            falsification_criterion=_text(
                value["falsification_criterion"], name="falsification_criterion"
            ),
            expression=_text(value["expression"], name="expression"),
            direction=_integer(value["direction"], name="direction"),
            required_fields=required_fields,
            dataset_id=_text(value["dataset_id"], name="dataset_id"),
            snapshot_id=_text(value["snapshot_id"], name="snapshot_id"),
            schema_hash=_text(value["schema_hash"], name="schema_hash"),
            availability_hash=_text(
                value["availability_hash"], name="availability_hash"
            ),
            security_contract_hash=_text(
                value["security_contract_hash"], name="security_contract_hash"
            ),
            frequency=FrequencySpec.from_mapping(frequency),
            aggregation=AggregationSpec(
                method=_text(aggregation["method"], name="aggregation.method"),
                window=_text(aggregation["window"], name="aggregation.window"),
                smoothing_span=_integer(
                    aggregation["smoothing_span"], name="aggregation.smoothing_span"
                ),
                reducer=_text(aggregation["reducer"], name="aggregation.reducer"),
            ),
            operator_registry_version=_text(
                value["operator_registry_version"], name="operator_registry_version"
            ),
            operator_registry_digest=_text(
                value["operator_registry_digest"], name="operator_registry_digest"
            ),
            preprocessing=tuple(
                PreprocessStep(
                    kind=_text(item["kind"], name="preprocessing.kind"),
                    lower=_optional_number(item["lower"], name="preprocessing.lower"),
                    upper=_optional_number(item["upper"], name="preprocessing.upper"),
                    exposure_fields=_string_tuple(
                        item["exposure_fields"], name="preprocessing.exposure_fields"
                    ),
                    ridge=_number(item["ridge"], name="preprocessing.ridge"),
                )
                for item in raw_preprocessing
            ),
            complexity_budget=ComplexityBudget(
                maximum_ast_nodes=_integer(
                    budget["maximum_ast_nodes"], name="maximum_ast_nodes"
                ),
                maximum_call_depth=_integer(
                    budget["maximum_call_depth"], name="maximum_call_depth"
                ),
                maximum_window=_integer(
                    budget["maximum_window"], name="maximum_window"
                ),
                maximum_fields=_integer(
                    budget["maximum_fields"], name="maximum_fields"
                ),
                maximum_relative_ops_per_cell=_integer(
                    budget["maximum_relative_ops_per_cell"],
                    name="maximum_relative_ops_per_cell",
                ),
                maximum_estimated_state_bytes_per_security=_integer(
                    budget["maximum_estimated_state_bytes_per_security"],
                    name="maximum_estimated_state_bytes_per_security",
                ),
                maximum_estimated_flops_per_cell=_integer(
                    budget["maximum_estimated_flops_per_cell"],
                    name="maximum_estimated_flops_per_cell",
                ),
            ),
            provenance=FactorProvenance(
                origin=_text(provenance["origin"], name="provenance.origin"),
                actor_id=_text(provenance["actor_id"], name="provenance.actor_id"),
                model_id=_optional_text(
                    provenance["model_id"], name="provenance.model_id"
                ),
                prompt_hash=_optional_text(
                    provenance["prompt_hash"], name="provenance.prompt_hash"
                ),
                parent_factor_hashes=parent_hashes,
            ),
        )
        if spec.semantic_hash != _text(value["semantic_hash"], name="semantic_hash"):
            raise ValueError("FactorSpec semantic hash differs")
        if spec.definition_hash != _text(
            value["definition_hash"], name="definition_hash"
        ):
            raise ValueError("FactorSpec definition hash differs")
        return spec


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"FactorSpec {name} must be an object")
    return value


def _require_fields(
    value: Mapping[str, object], expected: set[str], *, name: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"FactorSpec {name} fields differ")


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"FactorSpec {name} must be text")
    return value


def _optional_text(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name=name)


def _number(value: object, *, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise TypeError(f"FactorSpec {name} must be a finite number")
    return float(value)


def _optional_number(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _number(value, name=name)


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"FactorSpec {name} must be a list of strings")
    return tuple(value)


def _integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"FactorSpec {name} must be an integer")
    return value


__all__ = [
    "AggregationSpec",
    "ComplexityBudget",
    "FactorDirection",
    "FactorProvenance",
    "FactorSpec",
    "PreprocessKind",
    "PreprocessStep",
]
