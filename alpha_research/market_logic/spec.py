from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import cast

from alpha_research.core.hashing import hash_json, require_sha256
from alpha_research.high_frequency.provider_semantics import (
    DataCapability,
    SourceKind,
)
from factor_production.v5.llm.schemas import MECHANISM_CODES


_SAFE_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


class BooleanOperator(str, Enum):
    AND = "and"
    OR = "or"


class ConditionRelation(str, Enum):
    ABOVE_BASELINE = "above_baseline"
    BELOW_BASELINE = "below_baseline"
    INCREASING = "increasing"
    DECREASING = "decreasing"
    WIDENING = "widening"
    NARROWING = "narrowing"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    DIVERGING_FROM = "diverging_from"
    CONVERGING_TO = "converging_to"


class WindowUnit(str, Enum):
    EVENTS = "events"
    BARS = "bars"
    SESSIONS = "sessions"
    DAYS = "days"


class PredictionDirection(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    HIGHER = "higher"
    LOWER = "lower"


class HorizonUnit(str, Enum):
    BARS = "bars"
    SESSIONS = "sessions"
    DAYS = "days"
    WEEKS = "weeks"


class DataDomain(str, Enum):
    MARKET_DATA = "market_data"
    FUNDAMENTAL = "fundamental"
    ALTERNATIVE = "alternative"


class LogicOrigin(str, Enum):
    HUMAN = "human"
    LLM = "llm"
    MUTATION = "mutation"
    CROSSOVER = "crossover"
    MIGRATION = "migration"


@dataclass(frozen=True, slots=True)
class WindowBandSpec:
    minimum: int
    maximum: int
    unit: WindowUnit | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit", WindowUnit(self.unit))
        for name in ("minimum", "maximum"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"condition window {name} must be a positive integer")
        if self.minimum > self.maximum:
            raise ValueError("condition window band is inverted")

    def to_dict(self) -> dict[str, object]:
        unit = self.unit
        if not isinstance(unit, WindowUnit):  # pragma: no cover
            raise RuntimeError("window unit was not normalized")
        return {"minimum": self.minimum, "maximum": self.maximum, "unit": unit.value}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "WindowBandSpec":
        _require_exact_keys(value, {"minimum", "maximum", "unit"}, path="window")
        return cls(
            minimum=_integer(value["minimum"], path="window.minimum"),
            maximum=_integer(value["maximum"], path="window.maximum"),
            unit=_string(value["unit"], path="window.unit"),
        )


@dataclass(frozen=True, slots=True)
class ConditionPredicateSpec:
    """One observable market condition, without an executable expression."""

    observable: str
    relation: ConditionRelation | str
    window: WindowBandSpec
    reference_observable: str | None = None

    def __post_init__(self) -> None:
        _require_code(self.observable, name="condition observable")
        if not isinstance(self.window, WindowBandSpec):
            raise TypeError("condition window must be a WindowBandSpec")
        object.__setattr__(self, "relation", ConditionRelation(self.relation))
        reference = self.reference_observable
        if reference is not None:
            _require_code(reference, name="condition reference_observable")
            if reference == self.observable:
                raise ValueError("condition observable cannot compare with itself")
        comparative = {
            ConditionRelation.GREATER_THAN,
            ConditionRelation.LESS_THAN,
            ConditionRelation.DIVERGING_FROM,
            ConditionRelation.CONVERGING_TO,
        }
        relation = ConditionRelation(self.relation)
        if (relation in comparative) != (reference is not None):
            raise ValueError(
                "comparative condition relations require exactly one reference observable"
            )

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        relation = self.relation
        if not isinstance(relation, ConditionRelation):  # pragma: no cover
            raise RuntimeError("condition relation was not normalized")
        return {
            "kind": "predicate",
            "observable": self.observable,
            "relation": relation.value,
            "window": self.window.to_dict(),
            "reference_observable": self.reference_observable,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ConditionPredicateSpec":
        _require_exact_keys(
            value,
            {"kind", "observable", "relation", "window", "reference_observable"},
            path="condition.predicate",
        )
        if value["kind"] != "predicate":
            raise ValueError("condition predicate kind differs")
        reference = value["reference_observable"]
        if reference is not None and not isinstance(reference, str):
            raise TypeError("condition reference_observable must be text or null")
        return cls(
            observable=_string(value["observable"], path="condition.observable"),
            relation=_string(value["relation"], path="condition.relation"),
            window=WindowBandSpec.from_mapping(
                _mapping(value["window"], path="condition.window")
            ),
            reference_observable=reference,
        )


@dataclass(frozen=True, slots=True)
class ConditionGroupSpec:
    """A canonical, immutable AND/OR tree of market conditions."""

    operator: BooleanOperator | str
    clauses: tuple[ConditionPredicateSpec | ConditionGroupSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "operator", BooleanOperator(self.operator))
        clauses = tuple(self.clauses)
        if len(clauses) < 2:
            raise ValueError("condition group requires at least two clauses")
        if any(
            not isinstance(item, (ConditionPredicateSpec, ConditionGroupSpec))
            for item in clauses
        ):
            raise TypeError("condition group clauses must be condition specifications")
        keyed = [(hash_json(item.to_dict()), item) for item in clauses]
        if len({digest for digest, _ in keyed}) != len(keyed):
            raise ValueError("condition group clauses must be unique")
        object.__setattr__(
            self,
            "clauses",
            tuple(item for _, item in sorted(keyed, key=lambda pair: pair[0])),
        )

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        operator = self.operator
        if not isinstance(operator, BooleanOperator):  # pragma: no cover
            raise RuntimeError("boolean operator was not normalized")
        return {
            "kind": "group",
            "operator": operator.value,
            "clauses": [item.to_dict() for item in self.clauses],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ConditionGroupSpec":
        _require_exact_keys(
            value, {"kind", "operator", "clauses"}, path="condition.group"
        )
        if value["kind"] != "group":
            raise ValueError("condition group kind differs")
        return cls(
            operator=_string(value["operator"], path="condition.operator"),
            clauses=tuple(
                condition_from_mapping(item)
                for item in _mapping_sequence(
                    value["clauses"], path="condition.clauses"
                )
            ),
        )


ConditionSpec = ConditionPredicateSpec | ConditionGroupSpec


def condition_from_mapping(value: Mapping[str, object]) -> ConditionSpec:
    kind = value.get("kind")
    if kind == "predicate":
        return ConditionPredicateSpec.from_mapping(value)
    if kind == "group":
        return ConditionGroupSpec.from_mapping(value)
    raise ValueError("condition kind must be predicate or group")


@dataclass(frozen=True, slots=True)
class PredictionHorizonSpec:
    value: int
    unit: HorizonUnit | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit", HorizonUnit(self.unit))
        if (
            not isinstance(self.value, int)
            or isinstance(self.value, bool)
            or self.value <= 0
        ):
            raise ValueError("prediction horizon must be a positive integer")

    def to_dict(self) -> dict[str, object]:
        unit = self.unit
        if not isinstance(unit, HorizonUnit):  # pragma: no cover
            raise RuntimeError("horizon unit was not normalized")
        return {"value": self.value, "unit": unit.value}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PredictionHorizonSpec":
        _require_exact_keys(value, {"value", "unit"}, path="prediction.horizon")
        return cls(
            value=_integer(value["value"], path="prediction.horizon.value"),
            unit=_string(value["unit"], path="prediction.horizon.unit"),
        )


@dataclass(frozen=True, slots=True)
class PredictionBeliefSpec:
    target: str
    direction: PredictionDirection | str
    horizon: PredictionHorizonSpec

    def __post_init__(self) -> None:
        _require_code(self.target, name="prediction target")
        object.__setattr__(self, "direction", PredictionDirection(self.direction))

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        direction = self.direction
        if not isinstance(direction, PredictionDirection):  # pragma: no cover
            raise RuntimeError("prediction direction was not normalized")
        return {
            "target": self.target,
            "direction": direction.value,
            "horizon": self.horizon.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PredictionBeliefSpec":
        _require_exact_keys(
            value, {"target", "direction", "horizon"}, path="prediction"
        )
        return cls(
            target=_string(value["target"], path="prediction.target"),
            direction=_string(value["direction"], path="prediction.direction"),
            horizon=PredictionHorizonSpec.from_mapping(
                _mapping(value["horizon"], path="prediction.horizon")
            ),
        )


@dataclass(frozen=True, slots=True)
class MarketHypothesis:
    """First-class hypothesis ``H = <C, B>`` from AlphaLogics."""

    condition: ConditionSpec
    belief: PredictionBeliefSpec

    def __post_init__(self) -> None:
        if not isinstance(self.condition, (ConditionPredicateSpec, ConditionGroupSpec)):
            raise TypeError("market hypothesis condition has an invalid type")
        if not isinstance(self.belief, PredictionBeliefSpec):
            raise TypeError("market hypothesis belief must be a PredictionBeliefSpec")
        depth, predicates = _condition_shape(self.condition)
        if depth > 8:
            raise ValueError("market hypothesis condition depth exceeds eight")
        if predicates > 32:
            raise ValueError("market hypothesis has more than 32 predicates")

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "condition": self.condition.to_dict(),
            "belief": self.belief.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "MarketHypothesis":
        _require_exact_keys(value, {"condition", "belief"}, path="hypothesis")
        return cls(
            condition=condition_from_mapping(
                _mapping(value["condition"], path="hypothesis.condition")
            ),
            belief=PredictionBeliefSpec.from_mapping(
                _mapping(value["belief"], path="hypothesis.belief")
            ),
        )


@dataclass(frozen=True, slots=True)
class FalsificationSpec:
    """Pre-result failure statement; it deliberately has no metric/result fields."""

    criterion_code: str
    failure_description: str

    def __post_init__(self) -> None:
        _require_code(self.criterion_code, name="falsification criterion_code")
        _require_text(
            self.failure_description, name="falsification failure_description"
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "criterion_code": self.criterion_code,
            "failure_description": self.failure_description,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FalsificationSpec":
        _require_exact_keys(
            value,
            {"criterion_code", "failure_description"},
            path="falsification",
        )
        return cls(
            criterion_code=_string(
                value["criterion_code"], path="falsification.criterion_code"
            ),
            failure_description=_string(
                value["failure_description"], path="falsification.failure_description"
            ),
        )


@dataclass(frozen=True, slots=True)
class CapabilityRequirement:
    data_domain: DataDomain | str
    source_kind: SourceKind | str
    capability: DataCapability | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_domain", DataDomain(self.data_domain))
        object.__setattr__(self, "source_kind", SourceKind(self.source_kind))
        object.__setattr__(self, "capability", DataCapability(self.capability))

    @property
    def key(self) -> tuple[str, str, str]:
        domain = DataDomain(self.data_domain)
        source = SourceKind(self.source_kind)
        capability = DataCapability(self.capability)
        return domain.value, source.value, capability.value

    def to_dict(self) -> dict[str, str]:
        domain, source, capability = self.key
        return {
            "data_domain": domain,
            "source_kind": source,
            "capability": capability,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CapabilityRequirement":
        _require_exact_keys(
            value,
            {"data_domain", "source_kind", "capability"},
            path="capability_requirement",
        )
        return cls(
            data_domain=_string(
                value["data_domain"], path="capability_requirement.data_domain"
            ),
            source_kind=_string(
                value["source_kind"], path="capability_requirement.source_kind"
            ),
            capability=_string(
                value["capability"], path="capability_requirement.capability"
            ),
        )


@dataclass(frozen=True, slots=True)
class MarketLogicProvenance:
    origin: LogicOrigin | str
    actor_id: str
    model_id: str | None = None
    prompt_hash: str | None = None
    parent_logic_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", LogicOrigin(self.origin))
        _require_text(self.actor_id, name="logic provenance actor_id")
        parents = tuple(sorted(self.parent_logic_hashes))
        if len(parents) != len(set(parents)):
            raise ValueError("parent logic hashes must be unique")
        for digest in parents:
            require_sha256(digest, name="parent logic hash")
        object.__setattr__(self, "parent_logic_hashes", parents)
        origin = LogicOrigin(self.origin)
        if origin is LogicOrigin.LLM:
            if self.model_id is None or self.prompt_hash is None:
                raise ValueError("LLM market logic requires model_id and prompt_hash")
        if self.model_id is not None:
            _require_text(self.model_id, name="logic provenance model_id")
        if self.prompt_hash is not None:
            require_sha256(self.prompt_hash, name="logic provenance prompt_hash")
        if origin in {LogicOrigin.MUTATION, LogicOrigin.CROSSOVER} and not parents:
            raise ValueError("evolutionary market logic requires parent logic hashes")

    def to_dict(self) -> dict[str, object]:
        origin = self.origin
        if not isinstance(origin, LogicOrigin):  # pragma: no cover
            raise RuntimeError("logic provenance origin was not normalized")
        return {
            "origin": origin.value,
            "actor_id": self.actor_id,
            "model_id": self.model_id,
            "prompt_hash": self.prompt_hash,
            "parent_logic_hashes": list(self.parent_logic_hashes),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "MarketLogicProvenance":
        _require_exact_keys(
            value,
            {"origin", "actor_id", "model_id", "prompt_hash", "parent_logic_hashes"},
            path="provenance",
        )
        model_id = value["model_id"]
        prompt_hash = value["prompt_hash"]
        if model_id is not None and not isinstance(model_id, str):
            raise TypeError("provenance model_id must be text or null")
        if prompt_hash is not None and not isinstance(prompt_hash, str):
            raise TypeError("provenance prompt_hash must be text or null")
        return cls(
            origin=_string(value["origin"], path="provenance.origin"),
            actor_id=_string(value["actor_id"], path="provenance.actor_id"),
            model_id=model_id,
            prompt_hash=prompt_hash,
            parent_logic_hashes=_string_sequence(
                value["parent_logic_hashes"], path="provenance.parent_logic_hashes"
            ),
        )


@dataclass(frozen=True, slots=True)
class MarketLogicSpec:
    """Immutable, result-free market logic admitted before factor generation."""

    logic_id: str
    version: str
    hypothesis: MarketHypothesis
    mechanism_code: str
    economic_rationale: str
    falsification: FalsificationSpec
    capability_requirements: tuple[CapabilityRequirement, ...]
    eligible_provider_regime_hashes: tuple[str, ...]
    semantic_catalog_hash: str
    availability_hash: str
    data_profile_hash: str
    provenance: MarketLogicProvenance
    schema_version: str = "market-logic-spec/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "market-logic-spec/v1":
            raise ValueError("unsupported MarketLogicSpec schema")
        if not isinstance(self.hypothesis, MarketHypothesis):
            raise TypeError("market logic hypothesis must be a MarketHypothesis")
        if not isinstance(self.falsification, FalsificationSpec):
            raise TypeError("market logic falsification must be a FalsificationSpec")
        if not isinstance(self.provenance, MarketLogicProvenance):
            raise TypeError("market logic provenance must be MarketLogicProvenance")
        _require_code(self.logic_id, name="market logic_id")
        _require_code(self.version, name="market logic version")
        if self.mechanism_code not in set(MECHANISM_CODES):
            raise ValueError(
                f"unknown market logic mechanism_code:{self.mechanism_code}"
            )
        _require_text(self.economic_rationale, name="market logic economic_rationale")
        if any(
            not isinstance(item, CapabilityRequirement)
            for item in self.capability_requirements
        ):
            raise TypeError(
                "market logic capability requirements must be CapabilityRequirement objects"
            )
        requirements = tuple(
            sorted(self.capability_requirements, key=lambda item: item.key)
        )
        if not requirements:
            raise ValueError("market logic requires at least one data capability")
        if len({item.key for item in requirements}) != len(requirements):
            raise ValueError("market logic capability requirements must be unique")
        object.__setattr__(self, "capability_requirements", requirements)
        regimes = tuple(sorted(self.eligible_provider_regime_hashes))
        if not regimes or len(regimes) != len(set(regimes)):
            raise ValueError(
                "eligible provider regime hashes must be non-empty and unique"
            )
        for digest in regimes:
            require_sha256(digest, name="eligible provider regime hash")
        object.__setattr__(self, "eligible_provider_regime_hashes", regimes)
        for name in ("semantic_catalog_hash", "availability_hash", "data_profile_hash"):
            require_sha256(str(getattr(self, name)), name=f"market logic {name}")

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "logic_id": self.logic_id,
            "version": self.version,
            "hypothesis": self.hypothesis.to_dict(),
            "mechanism_code": self.mechanism_code,
            "economic_rationale": self.economic_rationale,
            "falsification": self.falsification.to_dict(),
            "capability_requirements": [
                item.to_dict() for item in self.capability_requirements
            ],
            "eligible_provider_regime_hashes": list(
                self.eligible_provider_regime_hashes
            ),
            "semantic_catalog_hash": self.semantic_catalog_hash,
            "availability_hash": self.availability_hash,
            "data_profile_hash": self.data_profile_hash,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "MarketLogicSpec":
        _require_exact_keys(
            value,
            {
                "schema_version",
                "logic_id",
                "version",
                "hypothesis",
                "mechanism_code",
                "economic_rationale",
                "falsification",
                "capability_requirements",
                "eligible_provider_regime_hashes",
                "semantic_catalog_hash",
                "availability_hash",
                "data_profile_hash",
                "provenance",
            },
            path="market_logic",
        )
        return cls(
            schema_version=_string(value["schema_version"], path="schema_version"),
            logic_id=_string(value["logic_id"], path="logic_id"),
            version=_string(value["version"], path="version"),
            hypothesis=MarketHypothesis.from_mapping(
                _mapping(value["hypothesis"], path="hypothesis")
            ),
            mechanism_code=_string(value["mechanism_code"], path="mechanism_code"),
            economic_rationale=_string(
                value["economic_rationale"], path="economic_rationale"
            ),
            falsification=FalsificationSpec.from_mapping(
                _mapping(value["falsification"], path="falsification")
            ),
            capability_requirements=tuple(
                CapabilityRequirement.from_mapping(item)
                for item in _mapping_sequence(
                    value["capability_requirements"], path="capability_requirements"
                )
            ),
            eligible_provider_regime_hashes=_string_sequence(
                value["eligible_provider_regime_hashes"],
                path="eligible_provider_regime_hashes",
            ),
            semantic_catalog_hash=_string(
                value["semantic_catalog_hash"], path="semantic_catalog_hash"
            ),
            availability_hash=_string(
                value["availability_hash"], path="availability_hash"
            ),
            data_profile_hash=_string(
                value["data_profile_hash"], path="data_profile_hash"
            ),
            provenance=MarketLogicProvenance.from_mapping(
                _mapping(value["provenance"], path="provenance")
            ),
        )


def _condition_shape(condition: ConditionSpec) -> tuple[int, int]:
    if isinstance(condition, ConditionPredicateSpec):
        return 1, 1
    children = [_condition_shape(item) for item in condition.clauses]
    return 1 + max(depth for depth, _ in children), sum(count for _, count in children)


def iter_predicates(condition: ConditionSpec) -> tuple[ConditionPredicateSpec, ...]:
    if isinstance(condition, ConditionPredicateSpec):
        return (condition,)
    return tuple(
        predicate
        for clause in condition.clauses
        for predicate in iter_predicates(clause)
    )


def _require_code(value: str, *, name: str) -> None:
    if not isinstance(value, str) or _SAFE_CODE.fullmatch(value) is None:
        raise ValueError(f"{name} contains unsupported characters")


def _require_text(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError(f"{name} must be non-empty text of at most 4096 characters")


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], *, path: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{path} wire fields differ:missing={sorted(expected - actual)},"
            f"unknown={sorted(actual - expected)}"
        )


def _mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{path} must be an object with text keys")
    return cast(Mapping[str, object], value)


def _mapping_sequence(value: object, *, path: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{path} must be a list of objects")
    return tuple(_mapping(item, path=f"{path}[]") for item in value)


def _string(value: object, *, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be text")
    return value


def _integer(value: object, *, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{path} must be an integer")
    return value


def _string_sequence(value: object, *, path: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{path} must be a list of strings")
    if not all(isinstance(item, str) for item in value):
        raise TypeError(f"{path} must contain only strings")
    return tuple(cast(str, item) for item in value)


__all__ = [
    "BooleanOperator",
    "CapabilityRequirement",
    "ConditionGroupSpec",
    "ConditionPredicateSpec",
    "ConditionRelation",
    "ConditionSpec",
    "DataDomain",
    "FalsificationSpec",
    "HorizonUnit",
    "LogicOrigin",
    "MarketHypothesis",
    "MarketLogicProvenance",
    "MarketLogicSpec",
    "PredictionBeliefSpec",
    "PredictionDirection",
    "PredictionHorizonSpec",
    "WindowBandSpec",
    "WindowUnit",
    "condition_from_mapping",
    "iter_predicates",
]
