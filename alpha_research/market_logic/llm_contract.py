from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from alpha_research.core.hashing import hash_json, require_sha256
from alpha_research.high_frequency.provider_semantics import (
    DataCapability,
    SourceKind,
)
from alpha_research.market_logic.spec import (
    BooleanOperator,
    CapabilityRequirement,
    ConditionGroupSpec,
    ConditionPredicateSpec,
    ConditionRelation,
    DataDomain,
    FalsificationSpec,
    HorizonUnit,
    MarketHypothesis,
    MarketLogicProvenance,
    MarketLogicSpec,
    PredictionBeliefSpec,
    PredictionDirection,
    PredictionHorizonSpec,
    WindowBandSpec,
    WindowUnit,
)
from factor_production.v5.llm.domain import validate_safe_model_id
from factor_production.v5.llm.safe_context import (
    UnsafeLLMContext,
    assert_safe_context,
)
from factor_production.v5.llm.schemas import (
    MECHANISM_CODES,
    StructuredOutputError,
    reject_chain_of_thought,
    validate_strict_json,
)


_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_LOGIC_ID = re.compile(r"[a-z][a-z0-9_.:-]{2,127}")
_FREE_EXECUTION_TEXT = re.compile(
    r"(?:__import__|os[.]system|subprocess|eval\s*[(]|exec\s*[(]|"
    r"(?:execute|run|invoke)\s+(?:python|shell|bash|command|code))",
    flags=re.IGNORECASE,
)
_BLIND_RESULT_TEXT = re.compile(
    r"(?:test|validation|holdout|teacher|official)\s*[-_/ ]*"
    r"(?:set|period|window|score|result|metric|feedback|review)",
    flags=re.IGNORECASE,
)
_FORBIDDEN_FIELD_TOKENS = frozenset(
    {
        "command",
        "date",
        "eval",
        "exec",
        "expression",
        "holdout",
        "ic",
        "metric",
        "python",
        "rankic",
        "score",
        "security",
        "sharpe",
        "shell",
        "teacher",
        "test",
        "validation",
    }
)


class MarketLogicAdmissionError(ValueError):
    """The result-free LLM draft exceeded its trusted admission authority."""


def _normalized_tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = "".join(
        character for character in normalized if unicodedata.category(character) != "Cf"
    )
    return tuple(token for token in re.split(r"[^a-z0-9]+", normalized) if token)


def _assert_no_blind_results_or_free_execution(value: Any, *, path: str = "$") -> None:
    """Reject result channels and executable instructions before type conversion.

    The closed JSON schema is the primary boundary.  This recursive guard is
    defense in depth for provider-controlled prose and unknown keys.  Error
    messages deliberately never echo the untrusted value or field name.
    """

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise UnsafeLLMContext(f"non-string field forbidden at {path}")
            if set(_normalized_tokens(key)).intersection(_FORBIDDEN_FIELD_TOKENS):
                raise UnsafeLLMContext(f"result or execution field forbidden at {path}")
            _assert_no_blind_results_or_free_execution(item, path=f"{path}.<field>")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_blind_results_or_free_execution(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFKC", value)
        if _FREE_EXECUTION_TEXT.search(normalized):
            raise UnsafeLLMContext(f"free-execution instruction forbidden at {path}")
        if _BLIND_RESULT_TEXT.search(normalized):
            raise UnsafeLLMContext(f"blind-result reference forbidden at {path}")


def _require_code(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _CODE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a bounded code")
    if set(_normalized_tokens(value)).intersection(_FORBIDDEN_FIELD_TOKENS):
        raise ValueError(f"{name} uses a reserved research-boundary token")
    return value


def _normalize_codes(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    normalized = tuple(sorted(values))
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must be non-empty and unique")
    for value in normalized:
        _require_code(value, name=name)
    return normalized


def _normalize_hashes(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    normalized = tuple(sorted(values))
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must be non-empty and unique")
    for digest in normalized:
        require_sha256(digest, name=name)
    return normalized


@dataclass(frozen=True, slots=True)
class MarketLogicAdmissionAuthority:
    """Trusted, result-free values that an LLM is never allowed to choose.

    The caller binds provider regimes, data availability, profile identity,
    provenance, and closed vocabularies.  The structured LLM response contains
    none of those authoritative hashes.
    """

    version: str
    actor_id: str
    model_id: str
    prompt_hash: str
    semantic_catalog_hash: str
    availability_hash: str
    data_profile_hash: str
    eligible_provider_regime_hashes: tuple[str, ...]
    allowed_observables: tuple[str, ...]
    allowed_prediction_targets: tuple[str, ...]
    allowed_falsification_codes: tuple[str, ...]
    allowed_capability_requirements: tuple[CapabilityRequirement, ...]
    parent_logic_hashes: tuple[str, ...] = ()
    maximum_proposals: int = 20
    maximum_conditions: int = 8
    maximum_window: int = 252
    maximum_horizon: int = 20
    schema_version: str = "market-logic-llm-authority/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "market-logic-llm-authority/v1":
            raise ValueError("unsupported MarketLogicAdmissionAuthority schema")
        _require_code(self.version, name="authority version")
        _require_code(self.actor_id, name="authority actor_id")
        validate_safe_model_id(self.model_id)
        for name in (
            "prompt_hash",
            "semantic_catalog_hash",
            "availability_hash",
            "data_profile_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"authority {name}")
        object.__setattr__(
            self,
            "eligible_provider_regime_hashes",
            _normalize_hashes(
                self.eligible_provider_regime_hashes,
                name="eligible provider regime hashes",
            ),
        )
        object.__setattr__(
            self,
            "parent_logic_hashes",
            (
                tuple(sorted(self.parent_logic_hashes))
                if not self.parent_logic_hashes
                else _normalize_hashes(
                    self.parent_logic_hashes, name="parent logic hashes"
                )
            ),
        )
        for name in (
            "allowed_observables",
            "allowed_prediction_targets",
            "allowed_falsification_codes",
        ):
            object.__setattr__(
                self,
                name,
                _normalize_codes(getattr(self, name), name=name),
            )
        requirements = tuple(
            sorted(self.allowed_capability_requirements, key=lambda item: item.key)
        )
        if (
            not requirements
            or any(not isinstance(item, CapabilityRequirement) for item in requirements)
            or len({item.key for item in requirements}) != len(requirements)
        ):
            raise ValueError(
                "allowed capability requirements must be non-empty, typed, and unique"
            )
        object.__setattr__(self, "allowed_capability_requirements", requirements)
        for name in (
            "maximum_proposals",
            "maximum_conditions",
            "maximum_window",
            "maximum_horizon",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"authority {name} must be positive")
        if self.maximum_proposals > 50 or self.maximum_conditions > 32:
            raise ValueError(
                "authority exceeds the bounded proposal or condition ceiling"
            )
        assert_safe_context(self.to_dict())

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    @property
    def output_schema(self) -> dict[str, Any]:
        return market_logic_output_schema(self)

    @property
    def output_schema_hash(self) -> str:
        return hash_json(self.output_schema)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "actor_id": self.actor_id,
            "model_id": self.model_id,
            "prompt_hash": self.prompt_hash,
            "semantic_catalog_hash": self.semantic_catalog_hash,
            "availability_hash": self.availability_hash,
            "data_profile_hash": self.data_profile_hash,
            "eligible_provider_regime_hashes": list(
                self.eligible_provider_regime_hashes
            ),
            "allowed_observables": list(self.allowed_observables),
            "allowed_prediction_targets": list(self.allowed_prediction_targets),
            "allowed_falsification_codes": list(self.allowed_falsification_codes),
            "allowed_capability_requirements": [
                item.to_dict() for item in self.allowed_capability_requirements
            ],
            "parent_logic_hashes": list(self.parent_logic_hashes),
            "maximum_proposals": self.maximum_proposals,
            "maximum_conditions": self.maximum_conditions,
            "maximum_window": self.maximum_window,
            "maximum_horizon": self.maximum_horizon,
        }


def _string_schema(
    *, enum: Sequence[str] | None = None, max_length: int = 128
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "string",
        "minLength": 1,
        "maxLength": max_length,
    }
    if enum is not None:
        schema["enum"] = list(enum)
    return schema


def market_logic_output_schema(
    authority: MarketLogicAdmissionAuthority,
) -> dict[str, Any]:
    """Build the exact JSON schema supplied to a structured-output provider."""

    capabilities = authority.allowed_capability_requirements
    domains = sorted({DataDomain(item.data_domain).value for item in capabilities})
    sources = sorted({SourceKind(item.source_kind).value for item in capabilities})
    capability_codes = sorted(
        {DataCapability(item.capability).value for item in capabilities}
    )
    predicate = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "observable",
            "relation",
            "window_minimum",
            "window_maximum",
            "window_unit",
            "reference_observable",
        ],
        "properties": {
            "observable": _string_schema(enum=authority.allowed_observables),
            "relation": _string_schema(
                enum=tuple(item.value for item in ConditionRelation)
            ),
            "window_minimum": {
                "type": "integer",
                "minimum": 1,
                "maximum": authority.maximum_window,
            },
            "window_maximum": {
                "type": "integer",
                "minimum": 1,
                "maximum": authority.maximum_window,
            },
            "window_unit": _string_schema(
                enum=tuple(item.value for item in WindowUnit)
            ),
            # The strict V5 schema validator intentionally supports one JSON
            # type per field, so "none" is the closed null sentinel.
            "reference_observable": _string_schema(
                enum=("none", *authority.allowed_observables)
            ),
        },
    }
    capability = {
        "type": "object",
        "additionalProperties": False,
        "required": ["data_domain", "source_kind", "capability"],
        "properties": {
            "data_domain": _string_schema(enum=domains),
            "source_kind": _string_schema(enum=sources),
            "capability": _string_schema(enum=capability_codes),
        },
    }
    proposal = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "logic_id",
            "condition_operator",
            "conditions",
            "prediction",
            "mechanism_code",
            "economic_rationale",
            "falsification",
            "capability_requirements",
        ],
        "properties": {
            "logic_id": {
                "type": "string",
                "minLength": 3,
                "maxLength": 128,
                "pattern": r"[a-z][a-z0-9_.:-]{2,127}",
            },
            "condition_operator": _string_schema(
                enum=tuple(item.value for item in BooleanOperator)
            ),
            "conditions": {
                "type": "array",
                "items": predicate,
                "minItems": 1,
                "maxItems": authority.maximum_conditions,
                "uniqueItems": True,
            },
            "prediction": {
                "type": "object",
                "additionalProperties": False,
                "required": ["target", "direction", "horizon_value", "horizon_unit"],
                "properties": {
                    "target": _string_schema(enum=authority.allowed_prediction_targets),
                    "direction": _string_schema(
                        enum=tuple(item.value for item in PredictionDirection)
                    ),
                    "horizon_value": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": authority.maximum_horizon,
                    },
                    "horizon_unit": _string_schema(
                        enum=tuple(item.value for item in HorizonUnit)
                    ),
                },
            },
            "mechanism_code": _string_schema(enum=MECHANISM_CODES),
            "economic_rationale": _string_schema(max_length=1024),
            "falsification": {
                "type": "object",
                "additionalProperties": False,
                "required": ["criterion_code", "failure_description"],
                "properties": {
                    "criterion_code": _string_schema(
                        enum=authority.allowed_falsification_codes
                    ),
                    "failure_description": _string_schema(max_length=512),
                },
            },
            "capability_requirements": {
                "type": "array",
                "items": capability,
                "minItems": 1,
                "maxItems": len(capabilities),
                "uniqueItems": True,
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "proposals"],
        "properties": {
            "schema_version": {
                "type": "string",
                "const": "market-logic-llm-output/v1",
            },
            "proposals": {
                "type": "array",
                "items": proposal,
                "minItems": 0,
                "maxItems": authority.maximum_proposals,
            },
        },
    }


@dataclass(frozen=True, slots=True)
class MarketLogicDraft:
    """Typed, result-free and non-executable provider proposal."""

    logic_id: str
    condition_operator: BooleanOperator
    conditions: tuple[ConditionPredicateSpec, ...]
    belief: PredictionBeliefSpec
    mechanism_code: str
    economic_rationale: str
    falsification: FalsificationSpec
    capability_requirements: tuple[CapabilityRequirement, ...]

    def __post_init__(self) -> None:
        if _LOGIC_ID.fullmatch(self.logic_id) is None:
            raise MarketLogicAdmissionError("invalid market logic draft ID")
        operator = BooleanOperator(self.condition_operator)
        object.__setattr__(self, "condition_operator", operator)
        conditions = tuple(self.conditions)
        if not conditions or any(
            not isinstance(item, ConditionPredicateSpec) for item in conditions
        ):
            raise MarketLogicAdmissionError("draft conditions must be typed predicates")
        if len({item.content_hash for item in conditions}) != len(conditions):
            raise MarketLogicAdmissionError("draft conditions must be unique")
        if len(conditions) == 1 and operator is not BooleanOperator.AND:
            raise MarketLogicAdmissionError(
                "a single-condition draft must use the canonical and operator"
            )
        object.__setattr__(
            self,
            "conditions",
            tuple(sorted(conditions, key=lambda item: item.content_hash)),
        )
        if not isinstance(self.belief, PredictionBeliefSpec):
            raise TypeError("draft belief must be a PredictionBeliefSpec")
        if self.mechanism_code not in set(MECHANISM_CODES):
            raise MarketLogicAdmissionError("unknown draft mechanism code")
        if (
            not isinstance(self.economic_rationale, str)
            or not self.economic_rationale.strip()
            or len(self.economic_rationale) > 1024
        ):
            raise MarketLogicAdmissionError("invalid draft economic rationale")
        if not isinstance(self.falsification, FalsificationSpec):
            raise TypeError("draft falsification must be a FalsificationSpec")
        requirements = tuple(
            sorted(self.capability_requirements, key=lambda item: item.key)
        )
        if (
            not requirements
            or any(not isinstance(item, CapabilityRequirement) for item in requirements)
            or len({item.key for item in requirements}) != len(requirements)
        ):
            raise MarketLogicAdmissionError(
                "draft capability requirements must be typed and unique"
            )
        object.__setattr__(self, "capability_requirements", requirements)
        assert_safe_context(self.to_dict())
        _assert_no_blind_results_or_free_execution(self.to_dict())

    @property
    def hypothesis(self) -> MarketHypothesis:
        condition: ConditionPredicateSpec | ConditionGroupSpec
        if len(self.conditions) == 1:
            condition = self.conditions[0]
        else:
            condition = ConditionGroupSpec(
                operator=self.condition_operator,
                clauses=self.conditions,
            )
        return MarketHypothesis(condition=condition, belief=self.belief)

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "logic_id": self.logic_id,
            "condition_operator": self.condition_operator.value,
            "conditions": [
                {
                    "observable": item.observable,
                    "relation": ConditionRelation(item.relation).value,
                    "window_minimum": item.window.minimum,
                    "window_maximum": item.window.maximum,
                    "window_unit": WindowUnit(item.window.unit).value,
                    "reference_observable": item.reference_observable or "none",
                }
                for item in self.conditions
            ],
            "prediction": {
                "target": self.belief.target,
                "direction": PredictionDirection(self.belief.direction).value,
                "horizon_value": self.belief.horizon.value,
                "horizon_unit": HorizonUnit(self.belief.horizon.unit).value,
            },
            "mechanism_code": self.mechanism_code,
            "economic_rationale": self.economic_rationale,
            "falsification": self.falsification.to_dict(),
            "capability_requirements": [
                item.to_dict() for item in self.capability_requirements
            ],
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        authority: MarketLogicAdmissionAuthority,
    ) -> "MarketLogicDraft":
        proposal_schema = cast(
            Mapping[str, Any],
            authority.output_schema["properties"]["proposals"]["items"],
        )
        reject_chain_of_thought(value)
        _assert_no_blind_results_or_free_execution(value)
        assert_safe_context(value)
        validate_strict_json(proposal_schema, value)
        conditions = tuple(
            ConditionPredicateSpec(
                observable=item["observable"],
                relation=item["relation"],
                window=WindowBandSpec(
                    minimum=item["window_minimum"],
                    maximum=item["window_maximum"],
                    unit=item["window_unit"],
                ),
                reference_observable=(
                    None
                    if item["reference_observable"] == "none"
                    else item["reference_observable"]
                ),
            )
            for item in value["conditions"]
        )
        requirements = tuple(
            CapabilityRequirement.from_mapping(item)
            for item in value["capability_requirements"]
        )
        allowed_requirements = {
            item.key for item in authority.allowed_capability_requirements
        }
        if any(item.key not in allowed_requirements for item in requirements):
            raise MarketLogicAdmissionError(
                "draft capability tuple is outside the trusted authority"
            )
        draft = cls(
            logic_id=value["logic_id"],
            condition_operator=BooleanOperator(value["condition_operator"]),
            conditions=conditions,
            belief=PredictionBeliefSpec(
                target=value["prediction"]["target"],
                direction=value["prediction"]["direction"],
                horizon=PredictionHorizonSpec(
                    value=value["prediction"]["horizon_value"],
                    unit=value["prediction"]["horizon_unit"],
                ),
            ),
            mechanism_code=value["mechanism_code"],
            economic_rationale=value["economic_rationale"],
            falsification=FalsificationSpec.from_mapping(value["falsification"]),
            capability_requirements=requirements,
        )
        if draft.belief.target not in set(authority.allowed_prediction_targets):
            raise MarketLogicAdmissionError("draft target is outside trusted authority")
        if draft.falsification.criterion_code not in set(
            authority.allowed_falsification_codes
        ):
            raise MarketLogicAdmissionError(
                "draft falsification is outside trusted authority"
            )
        return draft

    def admit(self, authority: MarketLogicAdmissionAuthority) -> MarketLogicSpec:
        """Bind trusted hashes/provenance and return the validated domain spec."""

        if len(self.conditions) > authority.maximum_conditions or any(
            item.window.maximum > authority.maximum_window for item in self.conditions
        ):
            raise MarketLogicAdmissionError(
                "draft condition shape exceeds the trusted authority"
            )
        if any(
            item.observable not in set(authority.allowed_observables)
            or (
                item.reference_observable is not None
                and item.reference_observable not in set(authority.allowed_observables)
            )
            for item in self.conditions
        ):
            raise MarketLogicAdmissionError(
                "draft observable is outside the trusted authority"
            )
        if any(
            item.key
            not in {
                allowed.key for allowed in authority.allowed_capability_requirements
            }
            for item in self.capability_requirements
        ):
            raise MarketLogicAdmissionError(
                "draft capability is outside the trusted authority"
            )
        if (
            self.belief.target not in set(authority.allowed_prediction_targets)
            or self.belief.horizon.value > authority.maximum_horizon
        ):
            raise MarketLogicAdmissionError(
                "draft prediction is outside the trusted authority"
            )
        if self.falsification.criterion_code not in set(
            authority.allowed_falsification_codes
        ):
            raise MarketLogicAdmissionError(
                "draft falsification is outside the trusted authority"
            )
        return MarketLogicSpec(
            logic_id=self.logic_id,
            version=authority.version,
            hypothesis=self.hypothesis,
            mechanism_code=self.mechanism_code,
            economic_rationale=self.economic_rationale,
            falsification=self.falsification,
            capability_requirements=self.capability_requirements,
            eligible_provider_regime_hashes=(authority.eligible_provider_regime_hashes),
            semantic_catalog_hash=authority.semantic_catalog_hash,
            availability_hash=authority.availability_hash,
            data_profile_hash=authority.data_profile_hash,
            provenance=MarketLogicProvenance(
                origin="llm",
                actor_id=authority.actor_id,
                model_id=authority.model_id,
                prompt_hash=authority.prompt_hash,
                parent_logic_hashes=authority.parent_logic_hashes,
            ),
        )


def parse_market_logic_output(
    payload: Mapping[str, Any],
    *,
    authority: MarketLogicAdmissionAuthority,
) -> tuple[MarketLogicDraft, ...]:
    """Parse an exact structured response without exposing evaluation results."""

    reject_chain_of_thought(payload)
    _assert_no_blind_results_or_free_execution(payload)
    assert_safe_context(payload)
    validate_strict_json(authority.output_schema, payload)
    drafts = tuple(
        MarketLogicDraft.from_mapping(item, authority=authority)
        for item in payload["proposals"]
    )
    ids = tuple(item.logic_id for item in drafts)
    if len(ids) != len(set(ids)):
        raise StructuredOutputError("market logic proposal IDs must be unique")
    identities = tuple(item.content_hash for item in drafts)
    if len(identities) != len(set(identities)):
        raise StructuredOutputError(
            "market logic proposals must be semantically unique"
        )
    return drafts


def admit_market_logic_output(
    payload: Mapping[str, Any],
    *,
    authority: MarketLogicAdmissionAuthority,
) -> tuple[MarketLogicSpec, ...]:
    return tuple(
        draft.admit(authority)
        for draft in parse_market_logic_output(payload, authority=authority)
    )


__all__ = [
    "MarketLogicAdmissionAuthority",
    "MarketLogicAdmissionError",
    "MarketLogicDraft",
    "admit_market_logic_output",
    "market_logic_output_schema",
    "parse_market_logic_output",
]
