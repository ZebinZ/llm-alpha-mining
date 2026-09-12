from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from llm_alpha_mining.mining.domain.enums import FactorFrequency, ProposalKind


CANDIDATE_SPEC_SCHEMA = "candidate-spec/v5"

# Official/teacher holdout outputs are deliberately not part of a candidate
# specification.  This guard also prevents a provider from smuggling them into
# the free-form parameters that are replayed in a later generation.
_HOLDOUT_KEY_TOKENS = frozenset(
    {
        "teacher",
        "official",
        "hidden_score",
        "holdout_score",
        "admission_score",
        "external_score",
    }
)


JsonScalar = None | bool | int | float | str
JsonValue = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]


class CandidateSpecError(ValueError):
    pass


def _deep_freeze(value: Any, *, path: str = "$") -> JsonValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CandidateSpecError(f"non-finite number at {path}")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CandidateSpecError(f"non-string object key at {path}")
            normalized = key.strip().lower()
            if any(token in normalized for token in _HOLDOUT_KEY_TOKENS):
                raise CandidateSpecError(
                    f"external holdout field is forbidden in candidate spec: {path}.{key}"
                )
            frozen[key] = _deep_freeze(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item, path=f"{path}[]") for item in value)
    raise CandidateSpecError(
        f"value at {path} is not JSON-compatible: {type(value).__name__}"
    )


def _thaw(value: JsonValue) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise CandidateSpecError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise CandidateSpecError(f"{name} must not be empty")
    return normalized


def _strict_string_tuple(
    value: Any, name: str, *, required: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise CandidateSpecError(f"{name} must be a list or tuple of strings")
    normalized = tuple(_nonempty(item, f"{name} item") for item in value)
    if required and not normalized:
        raise CandidateSpecError(f"{name} must not be empty")
    return normalized


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """The one authoritative candidate wire model used throughout V5.

    The object is immutable, strictly deserialized, and contains only proposal
    provenance and formula semantics.  Local evaluation results and external
    teacher/official results live in score observations, never in this model.
    """

    candidate_id: str
    hypothesis: str
    expression: str
    direction: int
    frequency: FactorFrequency | str
    generation: int
    family: str
    required_fields: tuple[str, ...] | list[str]
    protocol_hash: str
    provider: str
    parent_ids: tuple[str, ...] | list[str] = ()
    aggregation: Mapping[str, JsonValue] = field(default_factory=dict)
    parameters: Mapping[str, JsonValue] = field(default_factory=dict)
    tags: tuple[str, ...] | list[str] = ()
    campaign_round: int | None = None
    lineage_depth: int | None = None
    proposal_kind: ProposalKind | str | None = None
    schema_version: str = CANDIDATE_SPEC_SCHEMA
    _explicit_multi_generation: bool = field(
        init=False,
        default=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_id", _nonempty(self.candidate_id, "candidate_id")
        )
        object.__setattr__(self, "hypothesis", _nonempty(self.hypothesis, "hypothesis"))
        object.__setattr__(self, "expression", _nonempty(self.expression, "expression"))
        object.__setattr__(self, "family", _nonempty(self.family, "family"))
        object.__setattr__(
            self, "protocol_hash", _nonempty(self.protocol_hash, "protocol_hash")
        )
        object.__setattr__(self, "provider", _nonempty(self.provider, "provider"))
        try:
            frequency = FactorFrequency(self.frequency)
        except ValueError as exc:
            raise CandidateSpecError(
                f"unsupported frequency: {self.frequency!r}"
            ) from exc
        object.__setattr__(self, "frequency", frequency)
        if self.schema_version != CANDIDATE_SPEC_SCHEMA:
            raise CandidateSpecError(
                f"unsupported candidate schema {self.schema_version!r}; expected {CANDIDATE_SPEC_SCHEMA!r}"
            )
        if isinstance(self.direction, bool) or self.direction not in {-1, 1}:
            raise CandidateSpecError("direction must be exactly -1 or 1")
        if (
            not isinstance(self.generation, int)
            or isinstance(self.generation, bool)
            or self.generation < 0
        ):
            raise CandidateSpecError("generation must be a non-negative integer")

        required_fields = _strict_string_tuple(
            self.required_fields, "required_fields", required=True
        )
        parent_ids = _strict_string_tuple(self.parent_ids, "parent_ids")
        tags = _strict_string_tuple(self.tags, "tags")
        for name, values in (
            ("required_fields", required_fields),
            ("parent_ids", parent_ids),
            ("tags", tags),
        ):
            if len(values) != len(set(values)):
                raise CandidateSpecError(f"{name} must not contain duplicates")
        if self.candidate_id in parent_ids:
            raise CandidateSpecError("candidate cannot be its own parent")

        lineage_values = (self.campaign_round, self.lineage_depth, self.proposal_kind)
        explicit_multi_generation = any(value is not None for value in lineage_values)
        if explicit_multi_generation and any(value is None for value in lineage_values):
            raise CandidateSpecError(
                "campaign_round, lineage_depth and proposal_kind must be supplied together"
            )
        if explicit_multi_generation:
            campaign_round = self.campaign_round
            lineage_depth = self.lineage_depth
            if (
                not isinstance(campaign_round, int)
                or isinstance(campaign_round, bool)
                or campaign_round < 0
            ):
                raise CandidateSpecError(
                    "campaign_round must be a non-negative integer"
                )
            if (
                not isinstance(lineage_depth, int)
                or isinstance(lineage_depth, bool)
                or lineage_depth < 0
            ):
                raise CandidateSpecError("lineage_depth must be a non-negative integer")
            try:
                proposal_kind = ProposalKind(self.proposal_kind)
            except ValueError as exc:
                raise CandidateSpecError(
                    f"unsupported proposal_kind: {self.proposal_kind!r}"
                ) from exc
            if campaign_round != self.generation:
                raise CandidateSpecError(
                    "generation is the legacy campaign-round alias and must equal campaign_round"
                )
            self._validate_lineage_shape(
                proposal_kind=proposal_kind,
                lineage_depth=lineage_depth,
                parent_ids=parent_ids,
            )
        else:
            # Legacy V5 records predate explicit campaign-round semantics.  We
            # infer useful runtime values but deliberately omit them from the
            # wire form so historical content hashes remain byte-compatible.
            if self.generation == 0 and parent_ids:
                raise CandidateSpecError(
                    "generation zero candidate cannot have parents"
                )
            if self.generation > 0 and not parent_ids:
                raise CandidateSpecError(
                    "non-zero generation candidate must name at least one parent"
                )
            campaign_round = self.generation
            lineage_depth = self.generation
            proposal_kind = (
                ProposalKind.ROOT
                if not parent_ids
                else ProposalKind.MUTATION
                if len(parent_ids) == 1
                else ProposalKind.CROSSOVER
            )
        object.__setattr__(self, "campaign_round", campaign_round)
        object.__setattr__(self, "lineage_depth", lineage_depth)
        object.__setattr__(self, "proposal_kind", proposal_kind)
        object.__setattr__(
            self, "_explicit_multi_generation", explicit_multi_generation
        )
        object.__setattr__(self, "required_fields", required_fields)
        object.__setattr__(self, "parent_ids", parent_ids)
        object.__setattr__(self, "tags", tags)
        if not isinstance(self.aggregation, Mapping):
            raise CandidateSpecError("aggregation must be a JSON object")
        if not isinstance(self.parameters, Mapping):
            raise CandidateSpecError("parameters must be a JSON object")
        if not re.fullmatch(r"[0-9a-f]{64}", self.protocol_hash):
            raise CandidateSpecError("protocol_hash must be a lowercase SHA-256 digest")
        object.__setattr__(
            self, "aggregation", _deep_freeze(self.aggregation, path="$.aggregation")
        )
        object.__setattr__(
            self, "parameters", _deep_freeze(self.parameters, path="$.parameters")
        )

    @staticmethod
    def _validate_lineage_shape(
        *,
        proposal_kind: ProposalKind,
        lineage_depth: int,
        parent_ids: tuple[str, ...],
    ) -> None:
        parent_count = len(parent_ids)
        if proposal_kind is ProposalKind.ROOT:
            if parent_count or lineage_depth != 0:
                raise CandidateSpecError(
                    "root proposals require no parents and lineage_depth zero"
                )
            return
        if lineage_depth <= 0:
            raise CandidateSpecError("derived proposals require positive lineage_depth")
        if proposal_kind in {ProposalKind.MUTATION, ProposalKind.REPAIR}:
            if parent_count != 1:
                raise CandidateSpecError(
                    f"{proposal_kind.value} proposals require exactly one parent"
                )
            return
        if (
            proposal_kind in {ProposalKind.CROSSOVER, ProposalKind.COMPOSITE}
            and parent_count < 2
        ):
            raise CandidateSpecError(
                f"{proposal_kind.value} proposals require at least two parents"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "hypothesis": self.hypothesis,
            "expression": self.expression,
            "direction": self.direction,
            "frequency": self.frequency.value,
            "generation": self.generation,
            "family": self.family,
            "required_fields": list(self.required_fields),
            "protocol_hash": self.protocol_hash,
            "provider": self.provider,
            "parent_ids": list(self.parent_ids),
            "aggregation": _thaw(self.aggregation),
            "parameters": _thaw(self.parameters),
            "tags": list(self.tags),
        }
        if self._explicit_multi_generation:
            payload.update(
                {
                    "campaign_round": self.campaign_round,
                    "lineage_depth": self.lineage_depth,
                    "proposal_kind": self.proposal_kind.value,
                }
            )
        return payload

    @property
    def has_explicit_lineage(self) -> bool:
        """Whether the three multi-generation fields were present on the wire."""

        return self._explicit_multi_generation

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateSpec":
        allowed = {
            "schema_version",
            "candidate_id",
            "hypothesis",
            "expression",
            "direction",
            "frequency",
            "generation",
            "family",
            "required_fields",
            "protocol_hash",
            "provider",
            "parent_ids",
            "aggregation",
            "parameters",
            "tags",
            "campaign_round",
            "lineage_depth",
            "proposal_kind",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise CandidateSpecError(f"unknown CandidateSpec fields: {unknown}")
        required = allowed - {
            "schema_version",
            "parent_ids",
            "aggregation",
            "parameters",
            "tags",
            "campaign_round",
            "lineage_depth",
            "proposal_kind",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise CandidateSpecError(f"missing CandidateSpec fields: {missing}")
        return cls(
            schema_version=payload.get("schema_version", CANDIDATE_SPEC_SCHEMA),
            candidate_id=payload["candidate_id"],
            hypothesis=payload["hypothesis"],
            expression=payload["expression"],
            direction=payload["direction"],
            frequency=payload["frequency"],
            generation=payload["generation"],
            family=payload["family"],
            required_fields=payload["required_fields"],
            protocol_hash=payload["protocol_hash"],
            provider=payload["provider"],
            parent_ids=payload.get("parent_ids", ()),
            aggregation=payload.get("aggregation", {}),
            parameters=payload.get("parameters", {}),
            tags=payload.get("tags", ()),
            campaign_round=payload.get("campaign_round"),
            lineage_depth=payload.get("lineage_depth"),
            proposal_kind=payload.get("proposal_kind"),
        )

    @property
    def content_hash(self) -> str:
        from llm_alpha_mining.mining.artifacts.hashing import hash_json

        return hash_json(self.to_dict())
