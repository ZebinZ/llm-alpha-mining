from __future__ import annotations

import ast
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from factor_production.v5.artifacts.hashing import hash_json
from factor_production.v5.llm.domain import (
    LLMContractError,
    validate_provider_request_digest,
    validate_safe_model_id,
)
from factor_production.v5.llm.schemas import (
    CandidateDraft,
    CRITIC_CODES,
    MECHANISM_CODES,
    RISK_CODES,
    Review,
    StructuredOutputError,
    canonical_expression_ast,
    reject_chain_of_thought,
)
from factor_production.v5.providers.base import SanitizedFeedback


class UnsafeLLMContext(ValueError):
    pass


_DIGEST = re.compile(r"[0-9a-f]{64}")
_PERSISTED_PROVIDER_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_CANDIDATE_ID = re.compile(r"[a-z][a-z0-9_]{2,63}")
_DATE = re.compile(r"(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])")
_COMPACT_DATE = re.compile(r"(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])")
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_SECURITY = re.compile(
    r"(?:(?:SH|SZ|BJ)\s*\d{6}|\d{6}\s*[.]?\s*(?:SH|SZ|BJ|XSHG|XSHE)|(?<!\d)\d{6}(?!\d))",
    flags=re.IGNORECASE,
)
_SECURITY_COMPACT = re.compile(
    r"(?:(?:sh|sz|bj)\d{6}|\d{6}(?:sh|sz|bj|xshg|xshe)|(?<!\d)\d{6}(?!\d))",
    flags=re.IGNORECASE,
)
_EXACT_METRIC_TEXT = re.compile(
    r"(?:(?:neutral(?:ized)?|market\s*[-_/]?\s*neutral|中性化?)\s*[-_/]?\s*)?"
    r"(?:rank\s*[-_/]?\s*ic|(?<![A-Za-z0-9])ic|information\s+coefficient|"
    r"秩\s*[-_/]?\s*ic|信息系数|秩相关系数|sharpe|spread|turnover|"
    r"score|p\s*[-_]?\s*value|t\s*[-_]?\s*stat(?:istic)?)"
    r"\s*(?:value|\u503c)?\s*(?::|=|~=|≈|is|was|of|\u4e3a|\u662f)?\s*"
    r"[+-]?(?:\d+(?:[.]\d*)?|[.]\d+)%?",
    flags=re.IGNORECASE,
)
_COT_OR_INJECTION = re.compile(
    r"(?:chain\s*[-_]?\s*of\s*[-_]?\s*thought|scratch\s*[-_]?\s*pad|"
    r"hidden\s*[-_]?\s*reasoning|private\s*[-_]?\s*reasoning|"
    r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?|"
    r"system\s*[-_]?\s*prompt|developer\s*[-_]?\s*message|jail\s*break|prompt\s*injection)",
    flags=re.IGNORECASE,
)
_COT_COMPACT_TOKENS = (
    "chainofthought",
    "scratchpad",
    "hiddenreasoning",
    "privatereasoning",
    "ignorepreviousinstruction",
    "ignorepriorinstruction",
    "systemprompt",
    "developermessage",
    "jailbreak",
    "promptinjection",
)
_SECRET_COMPACT_TOKENS = (
    "apikey",
    "bearer",
    "accesstoken",
    "refreshtoken",
    "clientsecret",
    "privatesecret",
    "privatekey",
    "credential",
    "password",
    "secret",
    "token",
)
_CREDENTIAL_PREFIX = re.compile(
    r"(?<![A-Za-z0-9])(?:sk|ghp|gho|github_pat|xox[abprs])[-_][A-Za-z0-9_-]{6,}|"
    r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{12,}|"
    r"(?<![A-Za-z0-9])eyJ[A-Za-z0-9_-]{6,}[.][A-Za-z0-9_-]{6,}[.][A-Za-z0-9_-]{6,}",
    flags=re.IGNORECASE,
)
_METRIC_COMPACT = re.compile(
    r"(?<![a-z0-9])"
    r"(?:(?:neutral(?:ized)?|marketneutral|中性化?)?"
    r"(?:rankic|ic|informationcoefficient|秩ic|信息系数|秩相关系数)"
    r"(?:value|is|was|of|值为?|为|是)?[+-]?\d+%?|"
    r"(?:sharpe|spread|turnover|score|pvalue|tstat(?:istic)?)"
    r"(?:is|was|of|\u4e3a|\u662f)?[+-]?\d+%?)",
    flags=re.IGNORECASE,
)
_TEACHER_TOKENS = (
    "teacher",
    "official",
    "holdout",
    "hidden_score",
    "admission_score",
    "external_score",
    "\u8001\u5e08\u8bc4\u5206",
    "\u9690\u85cf\u8bc4\u5206",
    "\u5b98\u65b9\u8bc4\u5206",
    "\u6837\u672c\u5916\u8bc4\u5206",
)
_FORBIDDEN_KEYS = (
    "raw_data",
    "market_data",
    "price_matrix",
    "return_matrix",
    "security_id",
    "security_code",
    "stock_code",
    "trade_date",
    "dates",
    "symbols",
    "tickers",
    "features",
    "observations",
    "timeseries",
    "time_series",
    "rank_ic",
    "ic_value",
    "sharpe",
    "p_value",
    "t_stat",
    "local_score",
    "exact_metric",
)


def security_normalize(value: str) -> str:
    """NFKC/casefold and remove format controls used for DLP evasion."""

    normalized = unicodedata.normalize("NFKC", str(value))
    normalized = "".join(
        character for character in normalized if unicodedata.category(character) != "Cf"
    )
    return normalized.casefold()


def _collapsed(value: str) -> str:
    # ``\w`` retains underscores, which allowed strings such as ``tea_cher``
    # and ``ignore_previous_instructions`` to split policy markers.  Alnum-only
    # normalization removes every separator class after NFKC/Cf stripping.
    return "".join(character for character in security_normalize(value) if character.isalnum())


def _is_digest(value: str) -> bool:
    return (
        _DIGEST.fullmatch(value) is not None
        or _PERSISTED_PROVIDER_DIGEST.fullmatch(value) is not None
    )


def assert_safe_context(value: Any, *, path: str = "$") -> None:
    """DLP guard applied to request, response, and provider metadata strings."""

    if isinstance(value, Mapping):
        if len(value) > 128:
            raise UnsafeLLMContext(f"oversized context object at {path}")
        for key, item in value.items():
            if not isinstance(key, str):
                raise UnsafeLLMContext(f"non-string context key at {path}")
            normalized = security_normalize(key).strip()
            compact = _collapsed(normalized)
            if any(_collapsed(token) in compact for token in _TEACHER_TOKENS):
                raise UnsafeLLMContext(f"teacher/official field forbidden at {path}.<field>")
            if any(
                compact == _collapsed(token) or compact.endswith(_collapsed(token))
                for token in _FORBIDDEN_KEYS
            ):
                raise UnsafeLLMContext(f"raw identifier or exact metric forbidden at {path}.<field>")
            if any(token in compact for token in _SECRET_COMPACT_TOKENS):
                raise UnsafeLLMContext(f"credential metadata forbidden at {path}.<field>")
            # Never reflect an untrusted key through a later exception path.
            assert_safe_context(item, path=f"{path}.<field>")
        return
    if isinstance(value, (list, tuple)):
        if len(value) > 256:
            raise UnsafeLLMContext(f"oversized context collection at {path}")
        for index, item in enumerate(value):
            assert_safe_context(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        normalized = security_normalize(value)
        compact = _collapsed(normalized)
        if any(_collapsed(token) in compact for token in _TEACHER_TOKENS):
            raise UnsafeLLMContext(f"teacher/official text forbidden at {path}")
        if any(token in compact for token in _SECRET_COMPACT_TOKENS) or _CREDENTIAL_PREFIX.search(value):
            raise UnsafeLLMContext(f"credential-like text forbidden at {path}")
        if _COT_OR_INJECTION.search(normalized) or any(
            token in compact for token in _COT_COMPACT_TOKENS
        ):
            raise UnsafeLLMContext(f"reasoning/prompt-injection text forbidden at {path}")
        if not _is_digest(value):
            if (
                _DATE.search(normalized)
                or _COMPACT_DATE.search(normalized)
                or _COMPACT_DATE.search(compact)
                or _YEAR.search(normalized)
            ):
                raise UnsafeLLMContext(f"exact date forbidden at {path}")
            if _SECURITY.search(normalized) or _SECURITY_COMPACT.search(compact):
                raise UnsafeLLMContext(f"security identifier forbidden at {path}")
            if _EXACT_METRIC_TEXT.search(normalized) or _METRIC_COMPACT.search(compact):
                raise UnsafeLLMContext(f"exact local metric text forbidden at {path}")
        if len(value) > 4096:
            raise UnsafeLLMContext(f"oversized context string at {path}")
        return
    if value is None or isinstance(value, (bool, int, float)):
        return
    raise UnsafeLLMContext(f"non-JSON context value at {path}: {type(value).__name__}")


def assert_safe_response_metadata(*, model_id: str, provider_request_id: str) -> None:
    # Metadata is provider-controlled and therefore crosses the same boundary
    # as output text.  It is never exempt merely because it is not a prompt.
    try:
        validate_safe_model_id(model_id)
    except ValueError:
        raise UnsafeLLMContext("unsafe model identifier") from None
    try:
        validate_provider_request_digest(provider_request_id)
    except LLMContractError:
        raise UnsafeLLMContext("unsafe provider request identifier") from None
    assert_safe_context({"model_identifier": model_id, "provider_identifier": provider_request_id})


def _sha(value: str, name: str) -> str:
    if not isinstance(value, str) or not _is_digest(value):
        raise UnsafeLLMContext(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class ParentCatalogEntry:
    """Safe lineage binding.  Deliberately contains no result or score field."""

    candidate_id: str
    spec_hash: str
    semantic_hash: str
    mechanism_code: str
    direction: int
    frequency: str
    lineage_depth: int
    expression: str
    canonical_expression_ast: str
    required_fields: tuple[str, ...]
    aggregation_method: str
    aggregation_window: str
    smoothing_span: int

    def __post_init__(self) -> None:
        if _CANDIDATE_ID.fullmatch(self.candidate_id) is None:
            raise UnsafeLLMContext("invalid parent candidate_id")
        _sha(self.spec_hash, "parent spec_hash")
        _sha(self.semantic_hash, "parent semantic_hash")
        if self.mechanism_code not in set(MECHANISM_CODES):
            raise UnsafeLLMContext("invalid parent mechanism_code")
        if isinstance(self.direction, bool) or self.direction not in {-1, 1}:
            raise UnsafeLLMContext("invalid parent direction")
        if self.frequency not in {"daily", "minute"}:
            raise UnsafeLLMContext("invalid parent frequency")
        if not isinstance(self.lineage_depth, int) or isinstance(self.lineage_depth, bool) or self.lineage_depth < 0:
            raise UnsafeLLMContext("invalid parent lineage_depth")
        if not isinstance(self.expression, str) or not self.expression or len(self.expression) > 1024:
            raise UnsafeLLMContext("invalid parent expression")
        try:
            expected_ast = canonical_expression_ast(self.expression)
        except StructuredOutputError:
            raise UnsafeLLMContext("invalid parent expression syntax") from None
        if self.canonical_expression_ast != expected_ast:
            raise UnsafeLLMContext("parent canonical_expression_ast does not bind expression")
        if not self.canonical_expression_ast.startswith("Expression(body="):
            raise UnsafeLLMContext("parent safe mechanism fields must not be empty")
        if len(self.canonical_expression_ast) > 4096 or "\n" in self.canonical_expression_ast:
            raise UnsafeLLMContext("invalid parent canonical_expression_ast")
        fields = tuple(self.required_fields)
        if (
            not fields
            or len(fields) != len(set(fields))
            or any(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", item) is None for item in fields)
        ):
            raise UnsafeLLMContext("parent required_fields must be non-empty and unique")
        object.__setattr__(self, "required_fields", fields)
        tree = ast.parse(self.expression, mode="eval")
        call_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        expression_fields = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        } - call_names
        if expression_fields != set(fields):
            raise UnsafeLLMContext("parent required_fields do not bind expression")
        if self.aggregation_method not in {"none", "last", "mean", "sum", "std", "skew", "kurt"}:
            raise UnsafeLLMContext("invalid parent aggregation_method")
        if self.aggregation_window not in {
            "full_day", "open30", "midday30", "postlunch30", "close30", "close60"
        }:
            raise UnsafeLLMContext("invalid parent aggregation_window")
        if self.smoothing_span not in {0, 5, 10, 20, 30}:
            raise UnsafeLLMContext("invalid parent smoothing_span")
        if self.frequency == "daily" and self.aggregation_method != "none":
            raise UnsafeLLMContext("daily parent aggregation must be none")
        if self.frequency == "minute" and self.aggregation_method == "none":
            raise UnsafeLLMContext("minute parent aggregation must be explicit")
        expected_semantic = hash_json(
            {
                "expression_ast": self.canonical_expression_ast,
                "frequency": self.frequency,
                "aggregation": {
                    "method": self.aggregation_method,
                    "window": self.aggregation_window,
                    "smoothing_span": self.smoothing_span,
                },
            }
        )
        if self.semantic_hash != expected_semantic:
            raise UnsafeLLMContext("parent semantic_hash does not bind safe executable fields")
        if self.spec_hash != hash_json(self._bound_spec_payload()):
            raise UnsafeLLMContext("parent spec_hash does not bind catalog fields")
        assert_safe_context(self.to_dict())

    def _bound_spec_payload(self) -> dict[str, Any]:
        """The complete score-free parent specification covered by spec_hash."""

        return {
            "schema_version": "safe-parent-catalog-spec/v1",
            "candidate_id": self.candidate_id,
            "semantic_hash": self.semantic_hash,
            "mechanism_code": self.mechanism_code,
            "direction": self.direction,
            "frequency": self.frequency,
            "lineage_depth": self.lineage_depth,
            "expression": self.expression,
            "canonical_expression_ast": self.canonical_expression_ast,
            "required_fields": list(self.required_fields),
            "aggregation_method": self.aggregation_method,
            "aggregation_window": self.aggregation_window,
            "smoothing_span": self.smoothing_span,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "spec_hash": self.spec_hash,
            "semantic_hash": self.semantic_hash,
            "mechanism_code": self.mechanism_code,
            "direction": self.direction,
            "frequency": self.frequency,
            "lineage_depth": self.lineage_depth,
            "expression": self.expression,
            "canonical_expression_ast": self.canonical_expression_ast,
            "required_fields": list(self.required_fields),
            "aggregation_method": self.aggregation_method,
            "aggregation_window": self.aggregation_window,
            "smoothing_span": self.smoothing_span,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParentCatalogEntry":
        expected = {
            "candidate_id",
            "spec_hash",
            "semantic_hash",
            "mechanism_code",
            "direction",
            "frequency",
            "lineage_depth",
            "expression",
            "canonical_expression_ast",
            "required_fields",
            "aggregation_method",
            "aggregation_window",
            "smoothing_span",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise UnsafeLLMContext("parent catalog fields must be exact")
        return cls(
            candidate_id=value["candidate_id"],
            spec_hash=value["spec_hash"],
            semantic_hash=value["semantic_hash"],
            mechanism_code=value["mechanism_code"],
            direction=value["direction"],
            frequency=value["frequency"],
            lineage_depth=value["lineage_depth"],
            expression=value["expression"],
            canonical_expression_ast=value["canonical_expression_ast"],
            required_fields=tuple(value["required_fields"]),
            aggregation_method=value["aggregation_method"],
            aggregation_window=value["aggregation_window"],
            smoothing_span=value["smoothing_span"],
        )

    @classmethod
    def from_draft(
        cls,
        draft: CandidateDraft,
        *,
        lineage_depth: int,
        spec_hash: str | None = None,
    ) -> "ParentCatalogEntry":
        aggregation = dict(draft.aggregation)
        values = {
            "candidate_id": draft.candidate_id,
            "semantic_hash": draft.semantic_hash,
            "mechanism_code": draft.family,
            "direction": draft.direction,
            "frequency": draft.frequency,
            "lineage_depth": lineage_depth,
            "expression": draft.expression,
            "canonical_expression_ast": draft.canonical_expression_ast,
            "required_fields": draft.required_fields,
            "aggregation_method": aggregation["method"],
            "aggregation_window": aggregation["window"],
            "smoothing_span": aggregation["smoothing_span"],
        }
        expected = hash_json(
            {
                "schema_version": "safe-parent-catalog-spec/v1",
                "candidate_id": values["candidate_id"],
                "semantic_hash": values["semantic_hash"],
                "mechanism_code": values["mechanism_code"],
                "direction": values["direction"],
                "frequency": values["frequency"],
                "lineage_depth": values["lineage_depth"],
                "expression": values["expression"],
                "canonical_expression_ast": values["canonical_expression_ast"],
                "required_fields": list(values["required_fields"]),
                "aggregation_method": values["aggregation_method"],
                "aggregation_window": values["aggregation_window"],
                "smoothing_span": values["smoothing_span"],
            }
        )
        if spec_hash is not None and spec_hash != expected:
            raise UnsafeLLMContext("supplied parent spec_hash does not bind draft fields")
        return cls(spec_hash=expected, **values)


@dataclass(frozen=True, slots=True)
class CandidateDescriptor:
    """Deterministic reviewer input; proposer prose and tags are excluded."""

    candidate_id: str
    descriptor_hash: str
    semantic_hash: str
    expression: str
    canonical_expression_ast: str
    mechanism_code: str
    direction: int
    frequency: str
    required_fields: tuple[str, ...]
    parent_ids: tuple[str, ...]
    aggregation_method: str
    aggregation_window: str
    smoothing_span: int

    def __post_init__(self) -> None:
        if _CANDIDATE_ID.fullmatch(self.candidate_id) is None:
            raise UnsafeLLMContext("invalid candidate descriptor ID")
        _sha(self.descriptor_hash, "descriptor_hash")
        _sha(self.semantic_hash, "descriptor semantic_hash")
        if not isinstance(self.expression, str) or not self.expression or len(self.expression) > 1024:
            raise UnsafeLLMContext("invalid descriptor expression")
        try:
            expected_ast = canonical_expression_ast(self.expression)
        except StructuredOutputError:
            raise UnsafeLLMContext("invalid descriptor expression syntax") from None
        if self.canonical_expression_ast != expected_ast:
            raise UnsafeLLMContext("descriptor AST does not bind authoritative expression")
        if self.mechanism_code not in set(MECHANISM_CODES):
            raise UnsafeLLMContext("invalid descriptor mechanism")
        if isinstance(self.direction, bool) or self.direction not in {-1, 1}:
            raise UnsafeLLMContext("invalid descriptor direction")
        if self.frequency not in {"daily", "minute"}:
            raise UnsafeLLMContext("invalid descriptor frequency")
        fields = tuple(self.required_fields)
        parents = tuple(self.parent_ids)
        if (
            not fields
            or len(fields) != len(set(fields))
            or any(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", item) is None for item in fields)
        ):
            raise UnsafeLLMContext("invalid descriptor required_fields")
        if (
            len(parents) != len(set(parents))
            or any(_CANDIDATE_ID.fullmatch(item) is None for item in parents)
        ):
            raise UnsafeLLMContext("invalid descriptor parent_ids")
        object.__setattr__(self, "required_fields", fields)
        object.__setattr__(self, "parent_ids", parents)
        if self.aggregation_method not in {"none", "last", "mean", "sum", "std", "skew", "kurt"}:
            raise UnsafeLLMContext("invalid descriptor aggregation_method")
        if self.aggregation_window not in {
            "full_day", "open30", "midday30", "postlunch30", "close30", "close60"
        }:
            raise UnsafeLLMContext("invalid descriptor aggregation_window")
        if self.smoothing_span not in {0, 5, 10, 20, 30}:
            raise UnsafeLLMContext("invalid descriptor smoothing_span")
        if self.frequency == "daily" and self.aggregation_method != "none":
            raise UnsafeLLMContext("daily descriptor aggregation must be none")
        if self.frequency == "minute" and self.aggregation_method == "none":
            raise UnsafeLLMContext("minute descriptor aggregation must be explicit")

        tree = ast.parse(self.expression, mode="eval")
        call_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        expression_fields = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        } - call_names
        if expression_fields != set(fields):
            raise UnsafeLLMContext("descriptor required_fields do not bind expression")
        expected_semantic = hash_json(
            {
                "expression_ast": expected_ast,
                "frequency": self.frequency,
                "aggregation": {
                    "method": self.aggregation_method,
                    "window": self.aggregation_window,
                    "smoothing_span": self.smoothing_span,
                },
            }
        )
        if self.semantic_hash != expected_semantic:
            raise UnsafeLLMContext("descriptor semantic_hash does not bind authoritative fields")
        if self.descriptor_hash != hash_json(self._bound_descriptor_payload()):
            raise UnsafeLLMContext("descriptor_hash does not bind authoritative fields")
        assert_safe_context(self.to_dict())

    def _bound_descriptor_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "safe-candidate-descriptor/v1",
            "candidate_id": self.candidate_id,
            "semantic_hash": self.semantic_hash,
            "expression": self.expression,
            "canonical_expression_ast": self.canonical_expression_ast,
            "mechanism_code": self.mechanism_code,
            "direction": self.direction,
            "frequency": self.frequency,
            "required_fields": list(self.required_fields),
            "parent_ids": list(self.parent_ids),
            "aggregation_method": self.aggregation_method,
            "aggregation_window": self.aggregation_window,
            "smoothing_span": self.smoothing_span,
        }

    @classmethod
    def from_draft(cls, draft: CandidateDraft) -> "CandidateDescriptor":
        aggregation = dict(draft.aggregation)
        values = {
            "candidate_id": draft.candidate_id,
            "semantic_hash": draft.semantic_hash,
            "expression": draft.expression,
            "canonical_expression_ast": draft.canonical_expression_ast,
            "mechanism_code": draft.family,
            "direction": draft.direction,
            "frequency": draft.frequency,
            "required_fields": draft.required_fields,
            "parent_ids": draft.parent_ids,
            "aggregation_method": aggregation["method"],
            "aggregation_window": aggregation["window"],
            "smoothing_span": aggregation["smoothing_span"],
        }
        descriptor_hash = hash_json(
            {
                "schema_version": "safe-candidate-descriptor/v1",
                **{
                    key: list(item) if isinstance(item, tuple) else item
                    for key, item in values.items()
                },
            }
        )
        return cls(descriptor_hash=descriptor_hash, **values)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateDescriptor":
        expected = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != expected:
            raise UnsafeLLMContext("candidate descriptor fields must be exact")
        return cls(
            candidate_id=value["candidate_id"],
            descriptor_hash=value["descriptor_hash"],
            semantic_hash=value["semantic_hash"],
            expression=value["expression"],
            canonical_expression_ast=value["canonical_expression_ast"],
            mechanism_code=value["mechanism_code"],
            direction=value["direction"],
            frequency=value["frequency"],
            required_fields=tuple(value["required_fields"]),
            parent_ids=tuple(value["parent_ids"]),
            aggregation_method=value["aggregation_method"],
            aggregation_window=value["aggregation_window"],
            smoothing_span=value["smoothing_span"],
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "candidate_id": self.candidate_id,
            "descriptor_hash": self.descriptor_hash,
            "semantic_hash": self.semantic_hash,
            "expression": self.expression,
            "canonical_expression_ast": self.canonical_expression_ast,
            "mechanism_code": self.mechanism_code,
            "direction": self.direction,
            "frequency": self.frequency,
            "required_fields": list(self.required_fields),
            "parent_ids": list(self.parent_ids),
            "aggregation_method": self.aggregation_method,
            "aggregation_window": self.aggregation_window,
            "smoothing_span": self.smoothing_span,
        }
        assert_safe_context(payload)
        return payload


_SAFE_ROOT_FIELDS = {
    "context_version",
    "campaign_hash",
    "protocol_hash",
    "generation",
    "requested_count",
    "allowed_fields",
    "allowed_operators",
    "allowed_windows",
    "parent_catalog",
    "feedback",
    "candidate_descriptors",
    "critic_reviews",
    "risk_assessments",
}


def _validate_review_items(
    name: str,
    values: Any,
    decisions: set[str],
    allowed_reason_codes: set[str],
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise UnsafeLLMContext(f"{name} must be a list")
    candidate_ids: list[str] = []
    for item in values:
        expected = {"candidate_id", "decision", "reason_codes"}
        if not isinstance(item, Mapping) or set(item) != expected:
            raise UnsafeLLMContext(f"invalid {name} item fields")
        if _CANDIDATE_ID.fullmatch(item["candidate_id"]) is None or item["decision"] not in decisions:
            raise UnsafeLLMContext(f"invalid {name} decision")
        if not isinstance(item["reason_codes"], (list, tuple)) or not all(
            isinstance(code, str) for code in item["reason_codes"]
        ):
            raise UnsafeLLMContext(f"invalid {name} reason codes")
        reason_codes = tuple(item["reason_codes"])
        if (
            not reason_codes
            or len(reason_codes) != len(set(reason_codes))
            or not set(reason_codes) <= allowed_reason_codes
        ):
            raise UnsafeLLMContext(f"invalid {name} reason codes")
        candidate_ids.append(item["candidate_id"])
    if len(candidate_ids) != len(set(candidate_ids)):
        raise UnsafeLLMContext(f"duplicate {name} candidate IDs")
    return tuple(candidate_ids)


def validate_safe_context_payload(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != _SAFE_ROOT_FIELDS:
        raise UnsafeLLMContext("safe context fields must be exact")
    if value["context_version"] != "safe-alpha-research-context/v2":
        raise UnsafeLLMContext("unsupported safe context version")
    _sha(value["campaign_hash"], "campaign_hash")
    _sha(value["protocol_hash"], "protocol_hash")
    for name in ("generation", "requested_count"):
        item = value[name]
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise UnsafeLLMContext(f"{name} must be a non-negative integer")
    if not 1 <= value["requested_count"] <= 50:
        raise UnsafeLLMContext("requested_count must be in [1, 50]")
    for name in ("allowed_fields", "allowed_operators"):
        items = value[name]
        if not isinstance(items, (list, tuple)) or not items or not all(
            isinstance(item, str) and item for item in items
        ):
            raise UnsafeLLMContext(f"{name} must be a non-empty string list")
    windows = value["allowed_windows"]
    if not isinstance(windows, (list, tuple)) or not windows or not all(
        isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in windows
    ):
        raise UnsafeLLMContext("allowed_windows must be positive integers")
    catalog = value["parent_catalog"]
    if not isinstance(catalog, (list, tuple)):
        raise UnsafeLLMContext("parent_catalog must be a list")
    entries = tuple(ParentCatalogEntry.from_dict(item) for item in catalog)
    if len({item.candidate_id for item in entries}) != len(entries):
        raise UnsafeLLMContext("parent_catalog candidate IDs must be unique")
    if len({item.spec_hash for item in entries}) != len(entries):
        raise UnsafeLLMContext("parent_catalog spec hashes must be unique")
    if len({item.semantic_hash for item in entries}) != len(entries):
        raise UnsafeLLMContext("parent_catalog semantic hashes must be unique")
    feedback = value["feedback"]
    if not isinstance(feedback, (list, tuple)):
        raise UnsafeLLMContext("feedback must be a list")
    for item in feedback:
        try:
            SanitizedFeedback.from_dict(dict(item))
        except Exception:
            raise UnsafeLLMContext("invalid sanitized feedback") from None
    descriptors = value["candidate_descriptors"]
    if not isinstance(descriptors, (list, tuple)):
        raise UnsafeLLMContext("candidate_descriptors must be a list")
    descriptor_entries = tuple(CandidateDescriptor.from_dict(item) for item in descriptors)
    descriptor_ids = tuple(item.candidate_id for item in descriptor_entries)
    if len(descriptor_ids) != len(set(descriptor_ids)):
        raise UnsafeLLMContext("candidate descriptor IDs must be unique")
    if len({item.semantic_hash for item in descriptor_entries}) != len(descriptor_entries):
        raise UnsafeLLMContext("candidate descriptor semantic hashes must be unique")
    critic_ids = _validate_review_items(
        "critic_reviews",
        value["critic_reviews"],
        {"approve", "reject"},
        set(CRITIC_CODES),
    )
    risk_ids = _validate_review_items(
        "risk_assessments",
        value["risk_assessments"],
        {"allow", "block"},
        set(RISK_CODES),
    )
    expected_ids = set(descriptor_ids)
    if critic_ids and set(critic_ids) != expected_ids:
        raise UnsafeLLMContext("critic review coverage does not bind descriptors")
    if risk_ids and set(risk_ids) != expected_ids:
        raise UnsafeLLMContext("risk assessment coverage does not bind descriptors")
    assert_safe_context(value)
    try:
        reject_chain_of_thought(value)
    except StructuredOutputError:
        raise UnsafeLLMContext("reasoning marker forbidden") from None


@dataclass(frozen=True, slots=True)
class SafeResearchContext:
    """Closed research context with hash-bound lineage and no observations."""

    campaign_hash: str
    protocol_hash: str
    generation: int
    requested_count: int
    allowed_fields: tuple[str, ...]
    allowed_operators: tuple[str, ...]
    allowed_windows: tuple[int, ...]
    parent_catalog: tuple[ParentCatalogEntry, ...] = ()
    feedback: tuple[SanitizedFeedback, ...] = ()

    def __post_init__(self) -> None:
        _sha(self.campaign_hash, "campaign_hash")
        _sha(self.protocol_hash, "protocol_hash")
        if not isinstance(self.generation, int) or isinstance(self.generation, bool) or self.generation < 0:
            raise UnsafeLLMContext("generation must be a non-negative integer")
        if not isinstance(self.requested_count, int) or isinstance(self.requested_count, bool):
            raise UnsafeLLMContext("requested_count must be an integer")
        if not 1 <= self.requested_count <= 50:
            raise UnsafeLLMContext("requested_count must be in [1, 50]")
        for name in ("allowed_fields", "allowed_operators", "allowed_windows"):
            values = tuple(getattr(self, name))
            if not values or len(values) != len(set(values)):
                raise UnsafeLLMContext(f"{name} must be non-empty and unique")
            object.__setattr__(self, name, values)
        catalog = tuple(self.parent_catalog)
        if any(not isinstance(item, ParentCatalogEntry) for item in catalog):
            raise UnsafeLLMContext("parent_catalog accepts ParentCatalogEntry only")
        if self.generation == 0 and catalog:
            raise UnsafeLLMContext("generation zero cannot expose a parent catalog")
        if self.generation > 0 and not catalog:
            raise UnsafeLLMContext("derived generations require a closed parent catalog")
        if len({item.candidate_id for item in catalog}) != len(catalog):
            raise UnsafeLLMContext("parent_catalog candidate IDs must be unique")
        if len({item.spec_hash for item in catalog}) != len(catalog):
            raise UnsafeLLMContext("parent_catalog spec hashes must be unique")
        if len({item.semantic_hash for item in catalog}) != len(catalog):
            raise UnsafeLLMContext("parent_catalog semantic hashes must be unique")
        object.__setattr__(self, "parent_catalog", catalog)
        if any(not isinstance(item, SanitizedFeedback) for item in self.feedback):
            raise UnsafeLLMContext("only SanitizedFeedback may enter the LLM context")
        by_id = {item.candidate_id: item for item in catalog}
        for item in self.feedback:
            parent = by_id.get(item.candidate_id)
            if parent is None or parent.spec_hash != item.candidate_hash:
                raise UnsafeLLMContext("feedback must bind to a parent catalog entry")
        self.to_dict()

    def to_dict(
        self,
        *,
        candidate_drafts: Iterable[CandidateDraft] = (),
        critic_reviews: Iterable[Review] = (),
        risk_assessments: Iterable[Review] = (),
    ) -> dict[str, Any]:
        # The proposer may emit free-form hypothesis prose, but no later role
        # receives it.  Review roles see only deterministic AST/enum records.
        descriptors = [CandidateDescriptor.from_draft(item).to_dict() for item in candidate_drafts]
        payload = {
            "context_version": "safe-alpha-research-context/v2",
            "campaign_hash": self.campaign_hash,
            "protocol_hash": self.protocol_hash,
            "generation": self.generation,
            "requested_count": self.requested_count,
            "allowed_fields": list(self.allowed_fields),
            "allowed_operators": list(self.allowed_operators),
            "allowed_windows": list(self.allowed_windows),
            "parent_catalog": [item.to_dict() for item in self.parent_catalog],
            "feedback": [item.to_dict() for item in self.feedback],
            "candidate_descriptors": descriptors,
            "critic_reviews": [
                {
                    "candidate_id": item.candidate_id,
                    "decision": item.decision,
                    "reason_codes": list(item.reason_codes),
                }
                for item in critic_reviews
            ],
            "risk_assessments": [
                {
                    "candidate_id": item.candidate_id,
                    "decision": item.decision,
                    "reason_codes": list(item.reason_codes),
                }
                for item in risk_assessments
            ],
        }
        validate_safe_context_payload(payload)
        return payload
