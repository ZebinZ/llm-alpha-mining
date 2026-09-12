from __future__ import annotations

import math
import ast
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from factor_production.v5.artifacts.hashing import hash_json
from factor_production.v5.llm.domain import LLMRole, thaw_json


class StructuredOutputError(ValueError):
    pass


_REASONING_KEYS = frozenset(
    {
        "analysis",
        "chain_of_thought",
        "chain-of-thought",
        "cot",
        "reasoning",
        "rationale",
        "scratchpad",
        "thinking",
    }
)

_REASONING_TEXT = re.compile(
    r"(?:chain[\s_-]*of[\s_-]*thought|scratch[\s_-]*pad|hidden[\s_-]*reasoning|"
    r"private[\s_-]*reasoning|system[\s_-]*prompt|developer[\s_-]*message)",
    flags=re.IGNORECASE,
)


def _security_normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(character for character in normalized if unicodedata.category(character) != "Cf")


def reject_chain_of_thought(value: Any, *, path: str = "$") -> None:
    """Fail closed if a provider attempts to persist hidden reasoning.

    The approved outputs use closed reason-code enums.  A short factor
    hypothesis is a deliverable specification, not model scratch work.
    """

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = _security_normalize(str(key)).strip().casefold().replace(" ", "_")
            if normalized in _REASONING_KEYS:
                raise StructuredOutputError(f"chain-of-thought field forbidden at {path}.{key}")
            reject_chain_of_thought(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_chain_of_thought(item, path=f"{path}[{index}]")
    elif isinstance(value, str) and _REASONING_TEXT.search(_security_normalize(value)):
        raise StructuredOutputError(f"chain-of-thought marker forbidden at {path}")


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, (list, tuple))
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise StructuredOutputError(f"validator does not support JSON type {expected!r}")


def validate_strict_json(schema: Mapping[str, Any], value: Any, *, path: str = "$") -> None:
    """Validate the deliberately small, vendor-neutral schema subset we emit."""

    expected_type = schema.get("type")
    if not isinstance(expected_type, str) or not _type_matches(value, expected_type):
        raise StructuredOutputError(
            f"{path} must be {expected_type}; got {type(value).__name__}"
        )
    if "const" in schema and value != schema["const"]:
        raise StructuredOutputError(f"{path} must equal {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise StructuredOutputError(f"{path} is outside the closed enum")

    if expected_type == "object":
        properties = schema.get("properties")
        if not isinstance(properties, Mapping):
            raise StructuredOutputError(f"invalid schema: {path} object has no properties")
        if schema.get("additionalProperties") is not False:
            raise StructuredOutputError(
                f"invalid schema: {path} must set additionalProperties=false"
            )
        required = set(schema.get("required", ()))
        unknown = sorted(set(value) - set(properties))
        missing = sorted(required - set(value))
        if unknown or missing:
            # Unknown provider keys are untrusted text and must not be echoed.
            raise StructuredOutputError(f"{path} object fields are outside the closed schema")
        for key, item in value.items():
            validate_strict_json(properties[key], item, path=f"{path}.{key}")
        return
    if expected_type == "array":
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if minimum is not None and len(value) < minimum:
            raise StructuredOutputError(f"{path} has fewer than {minimum} items")
        if maximum is not None and len(value) > maximum:
            raise StructuredOutputError(f"{path} has more than {maximum} items")
        if schema.get("uniqueItems"):
            normalized = [repr(thaw_json(item)) for item in value]
            if len(normalized) != len(set(normalized)):
                raise StructuredOutputError(f"{path} items must be unique")
        item_schema = schema.get("items")
        if not isinstance(item_schema, Mapping):
            raise StructuredOutputError(f"invalid schema: {path} array has no item schema")
        for index, item in enumerate(value):
            validate_strict_json(item_schema, item, path=f"{path}[{index}]")
        return
    if expected_type == "string":
        if len(value) < schema.get("minLength", 0):
            raise StructuredOutputError(f"{path} is too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise StructuredOutputError(f"{path} is too long")
        pattern = schema.get("pattern")
        if pattern is not None and re.fullmatch(pattern, value) is None:
            raise StructuredOutputError(f"{path} does not match the frozen pattern")
        return
    if expected_type in {"integer", "number"}:
        if "minimum" in schema and value < schema["minimum"]:
            raise StructuredOutputError(f"{path} is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise StructuredOutputError(f"{path} is above maximum")


def _string(*, enum: list[str] | None = None, max_length: int = 128) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "string", "minLength": 1, "maxLength": max_length}
    if enum is not None:
        result["enum"] = enum
    return result


def _string_array(
    *,
    enum: list[str] | None = None,
    minimum: int = 0,
    maximum: int = 16,
) -> dict[str, Any]:
    return {
        "type": "array",
        "items": _string(enum=enum),
        "minItems": minimum,
        "maxItems": maximum,
        "uniqueItems": True,
    }


def _identifier_array(*, minimum: int = 0, maximum: int = 16) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "type": "string",
            "minLength": 1,
            "maxLength": 64,
            "pattern": r"[A-Za-z][A-Za-z0-9_]*",
        },
        "minItems": minimum,
        "maxItems": maximum,
        "uniqueItems": True,
    }


MECHANISM_CODES = [
    "behavioral_reversal",
    "liquidity",
    "volatility",
    "price_volume",
    "microstructure",
    "seasonality",
    "distribution_shape",
    "path_dependence",
    "cross_sectional",
]

PROPOSAL_TAG_CODES = [
    "mechanism_first",
    "novelty",
    "repair",
    "diversification",
    "low_turnover",
    "daily_signal",
    "intraday_signal",
]


AGGREGATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["method", "window", "smoothing_span"],
    "properties": {
        "method": _string(enum=["none", "last", "mean", "sum", "std", "skew", "kurt"]),
        "window": _string(
            enum=["full_day", "open30", "midday30", "postlunch30", "close30", "close60"]
        ),
        "smoothing_span": {"type": "integer", "enum": [0, 5, 10, 20, 30]},
    },
}


CANDIDATE_DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "candidate_id",
        "hypothesis",
        "expression",
        "direction",
        "frequency",
        "family",
        "required_fields",
        "parent_ids",
        "aggregation",
        "tags",
    ],
    "properties": {
        "candidate_id": {
            "type": "string",
            "minLength": 3,
            "maxLength": 64,
            "pattern": r"[a-z][a-z0-9_]*",
        },
        "hypothesis": _string(max_length=320),
        "expression": _string(max_length=1024),
        "direction": {"type": "integer", "enum": [-1, 1]},
        "frequency": _string(enum=["daily", "minute"]),
        "family": _string(enum=MECHANISM_CODES),
        "required_fields": _identifier_array(minimum=1, maximum=16),
        "parent_ids": _string_array(maximum=8),
        "aggregation": AGGREGATION_SCHEMA,
        "tags": _string_array(enum=PROPOSAL_TAG_CODES, maximum=16),
    },
}


CRITIC_CODES = [
    "mechanism_supported",
    "mechanism_unclear",
    "likely_redundant",
    "fragile_construction",
    "unnecessary_complexity",
    "repairable",
]
RISK_CODES = [
    "no_blocker",
    "data_leakage",
    "unavailable_field",
    "unsafe_semantics",
    "excessive_complexity",
    "lineage_violation",
]
ARBITER_CODES = [
    "balanced_mechanism",
    "diversifies_batch",
    "critic_reject",
    "risk_block",
    "not_selected",
]


PROPOSER_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "proposals"],
    "properties": {
        "schema_version": {"type": "string", "const": "llm-proposer-output/v1"},
        "proposals": {
            "type": "array",
            "items": CANDIDATE_DRAFT_SCHEMA,
            "minItems": 0,
            "maxItems": 50,
        },
    },
}


def _decision_item(*, decision_name: str, decisions: list[str], codes: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["candidate_id", decision_name, "reason_codes"],
        "properties": {
            "candidate_id": CANDIDATE_DRAFT_SCHEMA["properties"]["candidate_id"],
            decision_name: _string(enum=decisions),
            "reason_codes": _string_array(enum=codes, minimum=1, maximum=len(codes)),
        },
    }


CRITIC_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "reviews"],
    "properties": {
        "schema_version": {"type": "string", "const": "llm-critic-output/v1"},
        "reviews": {
            "type": "array",
            "items": _decision_item(
                decision_name="verdict", decisions=["approve", "reject"], codes=CRITIC_CODES
            ),
            "minItems": 0,
            "maxItems": 50,
        },
    },
}


RISK_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "assessments"],
    "properties": {
        "schema_version": {"type": "string", "const": "llm-risk-output/v1"},
        "assessments": {
            "type": "array",
            "items": _decision_item(
                decision_name="decision", decisions=["allow", "block"], codes=RISK_CODES
            ),
            "minItems": 0,
            "maxItems": 50,
        },
    },
}


_ARBITER_ITEM = _decision_item(
    decision_name="decision", decisions=["select", "reject"], codes=ARBITER_CODES
)
_ARBITER_ITEM["required"].append("priority")
_ARBITER_ITEM["properties"]["priority"] = {
    "type": "integer",
    "minimum": 0,
    "maximum": 100,
}

ARBITER_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "decisions"],
    "properties": {
        "schema_version": {"type": "string", "const": "llm-arbiter-output/v1"},
        "decisions": {
            "type": "array",
            "items": _ARBITER_ITEM,
            "minItems": 0,
            "maxItems": 50,
        },
    },
}


ROLE_SCHEMAS: Mapping[LLMRole, Mapping[str, Any]] = MappingProxyType(
    {
        LLMRole.PROPOSER: PROPOSER_OUTPUT_SCHEMA,
        LLMRole.CRITIC: CRITIC_OUTPUT_SCHEMA,
        LLMRole.RISK: RISK_OUTPUT_SCHEMA,
        LLMRole.ARBITER: ARBITER_OUTPUT_SCHEMA,
    }
)


ROLE_SCHEMA_IDS: Mapping[LLMRole, str] = MappingProxyType(
    {role: f"{role.value}-structured-output/v1" for role in LLMRole}
)


def schema_hash(role: LLMRole | str) -> str:
    return hash_json(ROLE_SCHEMAS[LLMRole(role)])


@dataclass(frozen=True, slots=True)
class CandidateDraft:
    candidate_id: str
    hypothesis: str
    expression: str
    direction: int
    frequency: str
    family: str
    required_fields: tuple[str, ...]
    parent_ids: tuple[str, ...]
    aggregation: Mapping[str, Any]
    tags: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateDraft":
        validate_strict_json(CANDIDATE_DRAFT_SCHEMA, value)
        hypothesis = _security_normalize(value["hypothesis"])
        if "\n" in hypothesis or "\r" in hypothesis:
            raise StructuredOutputError("hypothesis must be a compact deliverable, not process notes")
        if re.search(
            r"\b(?:i|we)\s+(?:think|thought|considered|reasoned|analysed|analyzed|decided|tried)\b",
            hypothesis,
            flags=re.IGNORECASE,
        ):
            raise StructuredOutputError("hypothesis contains reasoning-process narration")
        return cls(
            candidate_id=value["candidate_id"],
            hypothesis=value["hypothesis"],
            expression=value["expression"],
            direction=value["direction"],
            frequency=value["frequency"],
            family=value["family"],
            required_fields=tuple(value["required_fields"]),
            parent_ids=tuple(value["parent_ids"]),
            aggregation=MappingProxyType(dict(value["aggregation"])),
            tags=tuple(value["tags"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "hypothesis": self.hypothesis,
            "expression": self.expression,
            "direction": self.direction,
            "frequency": self.frequency,
            "family": self.family,
            "required_fields": list(self.required_fields),
            "parent_ids": list(self.parent_ids),
            "aggregation": dict(self.aggregation),
            "tags": list(self.tags),
        }

    @property
    def canonical_expression_ast(self) -> str:
        return canonical_expression_ast(self.expression)

    @property
    def semantic_hash(self) -> str:
        """Name/prose/sign-independent semantic identity for panel de-duplication."""

        return hash_json(
            {
                "expression_ast": self.canonical_expression_ast,
                "frequency": self.frequency,
                "aggregation": canonical_aggregation(self.frequency, self.aggregation),
            }
        )


_COMMUTATIVE_OPERATORS = frozenset({"Add", "Mul", "Multiply"})


def canonical_aggregation(frequency: str, aggregation: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize panel and CandidateSpec aggregation wires to one identity."""

    values = dict(aggregation)
    if frequency == "daily" and not values:
        return {"method": "none", "window": "full_day", "smoothing_span": 0}
    if set(values) == {"method", "window", "smoothing_span"}:
        return {
            "method": values["method"],
            "window": values["window"],
            "smoothing_span": values["smoothing_span"],
        }
    # Historical V5 minute candidates use a richer intraday/interday wire.
    # Preserve every legacy field so changes such as EMA20 -> EMA30 remain
    # distinct while the expression canonicalizer still handles reordering.
    return values


def _canonicalize_expression_node(node: ast.AST) -> ast.AST:
    """Canonicalize the closed DSL, including associative commutative calls.

    ``Add(a, b)`` and ``Add(b, a)`` (and nested equivalents) must not consume
    separate research attempts merely because an LLM reordered operands.
    Unknown calls are retained in positional order and are rejected later by
    the frozen DSL allow-list.
    """

    if isinstance(node, ast.Call):
        function = _canonicalize_expression_node(node.func)
        args = [_canonicalize_expression_node(item) for item in node.args]
        keywords = [
            ast.keyword(arg=item.arg, value=_canonicalize_expression_node(item.value))
            for item in node.keywords
        ]
        name = function.id if isinstance(function, ast.Name) else None
        if name in _COMMUTATIVE_OPERATORS:
            flattened: list[ast.AST] = []

            def collect(item: ast.AST) -> None:
                if (
                    isinstance(item, ast.Call)
                    and isinstance(item.func, ast.Name)
                    and item.func.id == name
                    and not item.keywords
                ):
                    for child in item.args:
                        collect(child)
                else:
                    flattened.append(item)

            for item in args:
                collect(item)
            flattened.sort(
                key=lambda item: ast.dump(item, annotate_fields=True, include_attributes=False)
            )
            if len(flattened) >= 2:
                rebuilt = ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=flattened[:2], keywords=[])
                for item in flattened[2:]:
                    rebuilt = ast.Call(
                        func=ast.Name(id=name, ctx=ast.Load()),
                        args=[rebuilt, item],
                        keywords=[],
                    )
                return rebuilt
        return ast.Call(func=function, args=args, keywords=keywords)
    if isinstance(node, ast.Name):
        return ast.Name(id=node.id, ctx=ast.Load())
    if isinstance(node, ast.Constant):
        return ast.Constant(value=node.value)
    if isinstance(node, ast.UnaryOp):
        return ast.UnaryOp(op=node.op, operand=_canonicalize_expression_node(node.operand))
    # Preserve a deterministic representation for unsupported syntax.  The
    # safe interpreter remains the authority that rejects it before review.
    return node


def canonical_expression_ast(expression: str) -> str:
    try:
        tree = ast.parse(str(expression), mode="eval")
    except SyntaxError:
        raise StructuredOutputError("invalid expression syntax") from None
    canonical = ast.Expression(body=_canonicalize_expression_node(tree.body))
    ast.fix_missing_locations(canonical)
    return ast.dump(canonical, annotate_fields=True, include_attributes=False)


class CriticVerdict(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"


class RiskDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"


class ArbiterDecision(str, Enum):
    SELECT = "select"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class Review:
    candidate_id: str
    decision: str
    reason_codes: tuple[str, ...]
    priority: int | None = None


def parse_role_output(role: LLMRole | str, payload: Mapping[str, Any]) -> Any:
    """Second, role-specific type pass after structural validation."""

    normalized = LLMRole(role)
    reject_chain_of_thought(payload)
    validate_strict_json(ROLE_SCHEMAS[normalized], payload)
    if normalized is LLMRole.PROPOSER:
        drafts = tuple(CandidateDraft.from_dict(item) for item in payload["proposals"])
        ids = [item.candidate_id for item in drafts]
        if len(ids) != len(set(ids)):
            raise StructuredOutputError("proposer emitted duplicate candidate IDs")
        return drafts
    if normalized is LLMRole.CRITIC:
        items = tuple(
            Review(item["candidate_id"], CriticVerdict(item["verdict"]).value, tuple(item["reason_codes"]))
            for item in payload["reviews"]
        )
    elif normalized is LLMRole.RISK:
        items = tuple(
            Review(item["candidate_id"], RiskDecision(item["decision"]).value, tuple(item["reason_codes"]))
            for item in payload["assessments"]
        )
    else:
        items = tuple(
            Review(
                item["candidate_id"],
                ArbiterDecision(item["decision"]).value,
                tuple(item["reason_codes"]),
                item["priority"],
            )
            for item in payload["decisions"]
        )
    ids = [item.candidate_id for item in items]
    if len(ids) != len(set(ids)):
        raise StructuredOutputError(f"{normalized.value} emitted duplicate candidate IDs")
    if normalized is LLMRole.RISK:
        for item in items:
            codes = set(item.reason_codes)
            if item.decision == RiskDecision.ALLOW.value and codes != {"no_blocker"}:
                raise StructuredOutputError("risk allow requires exactly the no_blocker code")
            if item.decision == RiskDecision.BLOCK.value and "no_blocker" in codes:
                raise StructuredOutputError("risk block cannot carry the no_blocker code")
    return items
