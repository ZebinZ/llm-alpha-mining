"""Immutable preregistration contract for the R0 three-source campaign.

The protocol freezes the scientific search boundary before any candidate is
evaluated.  It intentionally contains no controller, registry, or execution
logic: those concerns belong to the existing governed research pipeline.
"""

from __future__ import annotations

import hmac
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, cast

from alpha_research.core.hashing import (
    canonical_json_bytes,
    hash_json,
    require_sha256,
)


R0_DATA_SOURCES = ("snapshot", "order", "trade")
R0_ABLATION_ARMS = (
    ("snapshot",),
    ("order",),
    ("trade",),
    ("snapshot", "order"),
    ("snapshot", "trade"),
    ("order", "trade"),
    ("snapshot", "order", "trade"),
)
R0_MODEL_LADDER = ("ridge", "elastic_net", "lightgbm", "mlp", "tcn_gru")

_LEGACY_SCHEMA_VERSION = "research-campaign-protocol/r0-v1"
_SCHEMA_VERSION = "research-campaign-protocol/r0-v2"
_CUTOVER_DATE = "2021-06-07"
_EXTERNAL_SCORING_POLICY = "audit_only_after_search_freeze"
_MULTIPLE_TESTING_METHOD = "benjamini_hochberg"
_MULTIPLE_TESTING_SCOPE = "full_tested_family"
_RESEARCH_BOUNDARY = "research_only_human_approval_required"
_ALLOWED_P_VALUE_METHODS = frozenset({"hac", "block_bootstrap"})
_PRIMARY_METRIC = "daily_rank_ic_mean"
_PRIMARY_TEST_DIRECTION = "hypothesis_declared_one_sided"


@dataclass(frozen=True, slots=True)
class ResearchCampaignProtocol:
    """Fail-closed preregistration for one R0 Alpha research campaign.

    ``final_submission_target`` is a planning target, not a pass quota or a
    minimum.  The protocol deliberately has no final-minimum field: a campaign
    may submit fewer candidates when fewer satisfy the frozen admission rules.
    """

    search_design_authority_hash: str | None
    data_identity_hash: str
    source_data_identity_hashes: Mapping[str, str]
    joint_daily_spec_hash: str
    cutover_policy_hash: str
    candidate_family_hash: str
    primary_label_spec_hash: str
    evaluation_calendar_hash: str
    validation_spec_hash: str
    evaluation_spec_hash: str
    statistical_test_spec_hash: str
    robustness_spec_hash: str
    cost_model_hash: str
    raw_idea_budget: int
    frozen_factor_spec_cap: int
    adaptive_validation_cap: int
    locked_test_cap: int
    final_submission_target: int
    final_submission_maximum: int
    p_value_method: str
    primary_metric: str = _PRIMARY_METRIC
    primary_test_direction: str = _PRIMARY_TEST_DIRECTION
    protocol_id: str = "r0-three-source-alpha-research"
    protocol_version: str = "2"
    data_sources: tuple[str, ...] = R0_DATA_SOURCES
    ablation_arms: tuple[tuple[str, ...], ...] = R0_ABLATION_ARMS
    cutover_date: str = _CUTOVER_DATE
    maximum_generations: int = 6
    pareto_no_improvement_patience: int = 2
    external_scores_may_influence_search: bool = False
    external_scoring_policy: str = _EXTERNAL_SCORING_POLICY
    failed_or_missing_p_value: float = 1.0
    multiple_testing_method: str = _MULTIPLE_TESTING_METHOD
    multiple_testing_scope: str = _MULTIPLE_TESTING_SCOPE
    model_ladder: tuple[str, ...] = R0_MODEL_LADDER
    deployment_boundary: str = _RESEARCH_BOUNDARY
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in {_LEGACY_SCHEMA_VERSION, _SCHEMA_VERSION}:
            raise ValueError("unsupported ResearchCampaignProtocol schema")
        if self.protocol_id != "r0-three-source-alpha-research":
            raise ValueError("unsupported R0 campaign protocol_id")
        if self.schema_version == _SCHEMA_VERSION:
            if self.protocol_version != "2":
                raise ValueError("unsupported R0 campaign protocol_version")
            if self.search_design_authority_hash is None:
                raise ValueError(
                    "R0 v2 protocol requires search design authority hash"
                )
            require_sha256(
                _require_exact_string(
                    self.search_design_authority_hash,
                    name="search_design_authority_hash",
                ),
                name="research campaign search design authority hash",
            )
        elif (
            self.protocol_version != "1"
            or self.search_design_authority_hash is not None
        ):
            raise ValueError(
                "legacy R0 v1 protocol cannot bind a search design authority"
            )

        for name in (
            "data_identity_hash",
            "joint_daily_spec_hash",
            "cutover_policy_hash",
            "candidate_family_hash",
            "primary_label_spec_hash",
            "evaluation_calendar_hash",
            "validation_spec_hash",
            "evaluation_spec_hash",
            "statistical_test_spec_hash",
            "robustness_spec_hash",
            "cost_model_hash",
        ):
            require_sha256(
                _require_exact_string(getattr(self, name), name=name),
                name=f"research campaign {name}",
            )

        provided_source_hashes = {
            _require_exact_string(
                source, name="source data identity key"
            ): require_sha256(
                _require_exact_string(digest, name=f"{source} data identity hash"),
                name=f"research campaign source data identity:{source}",
            )
            for source, digest in self.source_data_identity_hashes.items()
        }
        if set(provided_source_hashes) != set(R0_DATA_SOURCES):
            raise ValueError(
                "R0 source data identity hashes must cover snapshot/order/trade"
            )
        source_hashes = {
            source: provided_source_hashes[source] for source in R0_DATA_SOURCES
        }
        expected_data_identity = hash_json(source_hashes)
        if not hmac.compare_digest(self.data_identity_hash, expected_data_identity):
            raise ValueError(
                "R0 composite data identity does not match its three source hashes"
            )
        object.__setattr__(
            self,
            "source_data_identity_hashes",
            MappingProxyType(source_hashes),
        )

        if self.data_sources != R0_DATA_SOURCES:
            raise ValueError("R0 data sources must be snapshot/order/trade in order")
        if self.ablation_arms != R0_ABLATION_ARMS:
            raise ValueError("R0 campaign requires all seven canonical ablation arms")
        if self.cutover_date != _CUTOVER_DATE:
            raise ValueError("R0 campaign cutover date must be 2021-06-07")

        funnel_names = (
            "raw_idea_budget",
            "frozen_factor_spec_cap",
            "adaptive_validation_cap",
            "locked_test_cap",
            "final_submission_maximum",
            "final_submission_target",
        )
        funnel = tuple(
            _require_positive_integer(getattr(self, name), name=name)
            for name in funnel_names
        )
        if not 500 <= self.raw_idea_budget <= 1_000:
            raise ValueError("R0 raw idea budget must lie in [500,1000]")
        if not 150 <= self.frozen_factor_spec_cap <= 300:
            raise ValueError("R0 frozen FactorSpec cap must lie in [150,300]")
        if self.adaptive_validation_cap > 80:
            raise ValueError("R0 adaptive validation cap must not exceed 80")
        if self.locked_test_cap > 30:
            raise ValueError("R0 locked test cap must not exceed 30")
        if not 12 <= self.final_submission_target <= 24:
            raise ValueError("R0 final submission target must lie in [12,24]")
        if self.final_submission_maximum > 30:
            raise ValueError("R0 final submission maximum must not exceed 30")
        if any(left < right for left, right in zip(funnel, funnel[1:], strict=False)):
            raise ValueError(
                "candidate funnel must be non-increasing from ideas to final target"
            )

        maximum_generations = _require_positive_integer(
            self.maximum_generations, name="maximum_generations"
        )
        if maximum_generations > 6:
            raise ValueError("R0 campaign may run at most six generations")
        if (
            _require_positive_integer(
                self.pareto_no_improvement_patience,
                name="pareto_no_improvement_patience",
            )
            != 2
        ):
            raise ValueError("R0 Pareto early-stop patience must be two generations")

        if type(self.external_scores_may_influence_search) is not bool:
            raise TypeError("external_scores_may_influence_search must be boolean")
        if self.external_scores_may_influence_search:
            raise ValueError("external scores must not influence candidate search")
        if self.external_scoring_policy != _EXTERNAL_SCORING_POLICY:
            raise ValueError("external scoring is permitted only after search freeze")

        if self.p_value_method not in _ALLOWED_P_VALUE_METHODS:
            raise ValueError("p_value_method must be hac or block_bootstrap")
        if self.primary_metric != _PRIMARY_METRIC:
            raise ValueError("R0 primary metric must be daily_rank_ic_mean")
        if self.primary_test_direction != _PRIMARY_TEST_DIRECTION:
            raise ValueError(
                "R0 primary test direction must be hypothesis-declared and one-sided"
            )
        if (
            type(self.failed_or_missing_p_value) not in (int, float)
            or isinstance(self.failed_or_missing_p_value, bool)
            or float(self.failed_or_missing_p_value) != 1.0
        ):
            raise ValueError("failed or missing p-values must fail closed to 1.0")
        object.__setattr__(
            self, "failed_or_missing_p_value", float(self.failed_or_missing_p_value)
        )
        if self.multiple_testing_method != _MULTIPLE_TESTING_METHOD:
            raise ValueError("R0 campaign requires Benjamini-Hochberg correction")
        if self.multiple_testing_scope != _MULTIPLE_TESTING_SCOPE:
            raise ValueError("Benjamini-Hochberg must cover the full tested family")

        if self.model_ladder != R0_MODEL_LADDER:
            raise ValueError("R0 model ladder order is frozen")
        if self.deployment_boundary != _RESEARCH_BOUNDARY:
            raise ValueError("R0 campaign is research-only and requires human approval")

    @property
    def content_hash(self) -> str:
        """SHA-256 identity of the canonical protocol payload."""

        return cast(str, hash_json(self._payload_dict()))

    def _payload_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "protocol_version": self.protocol_version,
            "data_sources": list(self.data_sources),
            "ablation_arms": [list(arm) for arm in self.ablation_arms],
            "data_identity_hash": self.data_identity_hash,
            "source_data_identity_hashes": dict(self.source_data_identity_hashes),
            "joint_daily_spec_hash": self.joint_daily_spec_hash,
            "cutover_date": self.cutover_date,
            "cutover_policy_hash": self.cutover_policy_hash,
            "candidate_family_hash": self.candidate_family_hash,
            "primary_label_spec_hash": self.primary_label_spec_hash,
            "evaluation_calendar_hash": self.evaluation_calendar_hash,
            "validation_spec_hash": self.validation_spec_hash,
            "evaluation_spec_hash": self.evaluation_spec_hash,
            "statistical_test_spec_hash": self.statistical_test_spec_hash,
            "robustness_spec_hash": self.robustness_spec_hash,
            "cost_model_hash": self.cost_model_hash,
            "raw_idea_budget": self.raw_idea_budget,
            "frozen_factor_spec_cap": self.frozen_factor_spec_cap,
            "adaptive_validation_cap": self.adaptive_validation_cap,
            "locked_test_cap": self.locked_test_cap,
            "final_submission_target": self.final_submission_target,
            "final_submission_maximum": self.final_submission_maximum,
            "maximum_generations": self.maximum_generations,
            "pareto_no_improvement_patience": self.pareto_no_improvement_patience,
            "external_scores_may_influence_search": (
                self.external_scores_may_influence_search
            ),
            "external_scoring_policy": self.external_scoring_policy,
            "primary_metric": self.primary_metric,
            "primary_test_direction": self.primary_test_direction,
            "p_value_method": self.p_value_method,
            "failed_or_missing_p_value": self.failed_or_missing_p_value,
            "multiple_testing_method": self.multiple_testing_method,
            "multiple_testing_scope": self.multiple_testing_scope,
            "model_ladder": list(self.model_ladder),
            "deployment_boundary": self.deployment_boundary,
        }
        if self.schema_version == _SCHEMA_VERSION:
            payload["search_design_authority_hash"] = (
                self.search_design_authority_hash
            )
        return payload

    def to_dict(self) -> dict[str, object]:
        """Return the strict, self-authenticating wire mapping."""

        payload = self._payload_dict()
        return {**payload, "content_hash": self.content_hash}

    def to_wire_bytes(self) -> bytes:
        """Return canonical UTF-8 JSON with one required trailing newline."""

        return cast(bytes, canonical_json_bytes(self.to_dict()) + b"\n")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ResearchCampaignProtocol":
        schema_version = _wire_string(
            value.get("schema_version"),
            name="schema_version",
        )
        if schema_version not in {_LEGACY_SCHEMA_VERSION, _SCHEMA_VERSION}:
            raise ValueError("unsupported ResearchCampaignProtocol schema")
        expected = {
            "schema_version",
            "protocol_id",
            "protocol_version",
            "data_sources",
            "ablation_arms",
            "data_identity_hash",
            "source_data_identity_hashes",
            "joint_daily_spec_hash",
            "cutover_date",
            "cutover_policy_hash",
            "candidate_family_hash",
            "primary_label_spec_hash",
            "evaluation_calendar_hash",
            "validation_spec_hash",
            "evaluation_spec_hash",
            "statistical_test_spec_hash",
            "robustness_spec_hash",
            "cost_model_hash",
            "raw_idea_budget",
            "frozen_factor_spec_cap",
            "adaptive_validation_cap",
            "locked_test_cap",
            "final_submission_target",
            "final_submission_maximum",
            "maximum_generations",
            "pareto_no_improvement_patience",
            "external_scores_may_influence_search",
            "external_scoring_policy",
            "primary_metric",
            "primary_test_direction",
            "p_value_method",
            "failed_or_missing_p_value",
            "multiple_testing_method",
            "multiple_testing_scope",
            "model_ladder",
            "deployment_boundary",
            "content_hash",
        }
        if schema_version == _SCHEMA_VERSION:
            expected.add("search_design_authority_hash")
        if set(value) != expected:
            raise ValueError("ResearchCampaignProtocol wire fields differ")

        content_hash = _wire_string(value["content_hash"], name="content_hash")
        require_sha256(content_hash, name="research campaign wire content_hash")
        protocol = cls(
            search_design_authority_hash=(
                _wire_string(
                    value["search_design_authority_hash"],
                    name="search_design_authority_hash",
                )
                if schema_version == _SCHEMA_VERSION
                else None
            ),
            schema_version=schema_version,
            protocol_id=_wire_string(value["protocol_id"], name="protocol_id"),
            protocol_version=_wire_string(
                value["protocol_version"], name="protocol_version"
            ),
            data_sources=_wire_string_tuple(value["data_sources"], name="data_sources"),
            ablation_arms=_wire_ablation_arms(value["ablation_arms"]),
            data_identity_hash=_wire_string(
                value["data_identity_hash"], name="data_identity_hash"
            ),
            source_data_identity_hashes=_wire_source_hash_mapping(
                value["source_data_identity_hashes"]
            ),
            joint_daily_spec_hash=_wire_string(
                value["joint_daily_spec_hash"], name="joint_daily_spec_hash"
            ),
            cutover_date=_wire_string(value["cutover_date"], name="cutover_date"),
            cutover_policy_hash=_wire_string(
                value["cutover_policy_hash"], name="cutover_policy_hash"
            ),
            candidate_family_hash=_wire_string(
                value["candidate_family_hash"], name="candidate_family_hash"
            ),
            primary_label_spec_hash=_wire_string(
                value["primary_label_spec_hash"], name="primary_label_spec_hash"
            ),
            evaluation_calendar_hash=_wire_string(
                value["evaluation_calendar_hash"], name="evaluation_calendar_hash"
            ),
            validation_spec_hash=_wire_string(
                value["validation_spec_hash"], name="validation_spec_hash"
            ),
            evaluation_spec_hash=_wire_string(
                value["evaluation_spec_hash"], name="evaluation_spec_hash"
            ),
            statistical_test_spec_hash=_wire_string(
                value["statistical_test_spec_hash"],
                name="statistical_test_spec_hash",
            ),
            robustness_spec_hash=_wire_string(
                value["robustness_spec_hash"], name="robustness_spec_hash"
            ),
            cost_model_hash=_wire_string(
                value["cost_model_hash"], name="cost_model_hash"
            ),
            raw_idea_budget=_wire_integer(
                value["raw_idea_budget"], name="raw_idea_budget"
            ),
            frozen_factor_spec_cap=_wire_integer(
                value["frozen_factor_spec_cap"], name="frozen_factor_spec_cap"
            ),
            adaptive_validation_cap=_wire_integer(
                value["adaptive_validation_cap"], name="adaptive_validation_cap"
            ),
            locked_test_cap=_wire_integer(
                value["locked_test_cap"], name="locked_test_cap"
            ),
            final_submission_target=_wire_integer(
                value["final_submission_target"], name="final_submission_target"
            ),
            final_submission_maximum=_wire_integer(
                value["final_submission_maximum"], name="final_submission_maximum"
            ),
            maximum_generations=_wire_integer(
                value["maximum_generations"], name="maximum_generations"
            ),
            pareto_no_improvement_patience=_wire_integer(
                value["pareto_no_improvement_patience"],
                name="pareto_no_improvement_patience",
            ),
            external_scores_may_influence_search=_wire_boolean(
                value["external_scores_may_influence_search"],
                name="external_scores_may_influence_search",
            ),
            external_scoring_policy=_wire_string(
                value["external_scoring_policy"], name="external_scoring_policy"
            ),
            primary_metric=_wire_string(value["primary_metric"], name="primary_metric"),
            primary_test_direction=_wire_string(
                value["primary_test_direction"], name="primary_test_direction"
            ),
            p_value_method=_wire_string(value["p_value_method"], name="p_value_method"),
            failed_or_missing_p_value=_wire_number(
                value["failed_or_missing_p_value"],
                name="failed_or_missing_p_value",
            ),
            multiple_testing_method=_wire_string(
                value["multiple_testing_method"], name="multiple_testing_method"
            ),
            multiple_testing_scope=_wire_string(
                value["multiple_testing_scope"], name="multiple_testing_scope"
            ),
            model_ladder=_wire_string_tuple(value["model_ladder"], name="model_ladder"),
            deployment_boundary=_wire_string(
                value["deployment_boundary"], name="deployment_boundary"
            ),
        )
        if not hmac.compare_digest(protocol.content_hash, content_hash):
            raise ValueError("ResearchCampaignProtocol content_hash mismatch")
        return protocol

    @classmethod
    def from_wire_bytes(cls, payload: bytes) -> "ResearchCampaignProtocol":
        """Decode only the exact canonical wire representation."""

        if type(payload) is not bytes:
            raise TypeError("ResearchCampaignProtocol wire payload must be bytes")
        try:
            decoded = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("ResearchCampaignProtocol wire is invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise TypeError("ResearchCampaignProtocol wire root must be an object")
        if payload != canonical_json_bytes(decoded) + b"\n":
            raise ValueError("ResearchCampaignProtocol wire is not canonical")
        return cls.from_mapping(decoded)


def _require_exact_string(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    return value


def _require_positive_integer(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _wire_string(value: object, *, name: str) -> str:
    return _require_exact_string(value, name=f"wire {name}")


def _wire_integer(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"wire {name} must be an integer")
    return value


def _wire_number(value: object, *, name: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"wire {name} must be a number")
    assert isinstance(value, (int, float))
    return float(value)


def _wire_boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"wire {name} must be boolean")
    return value


def _wire_string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(type(item) is str for item in value):
        raise TypeError(f"wire {name} must be a string array")
    return tuple(value)


def _wire_ablation_arms(value: object) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list):
        raise TypeError("wire ablation_arms must be an array")
    arms: list[tuple[str, ...]] = []
    for arm in value:
        if not isinstance(arm, list) or not all(type(item) is str for item in arm):
            raise TypeError("wire ablation arm must be a string array")
        arms.append(tuple(arm))
    return tuple(arms)


def _wire_source_hash_mapping(value: object) -> Mapping[str, str]:
    if not isinstance(value, dict) or not all(
        type(source) is str and type(digest) is str for source, digest in value.items()
    ):
        raise TypeError("wire source_data_identity_hashes must be a string object")
    return {str(source): str(digest) for source, digest in value.items()}


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key:{key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant:{value}")


__all__ = [
    "R0_ABLATION_ARMS",
    "R0_DATA_SOURCES",
    "R0_MODEL_LADDER",
    "ResearchCampaignProtocol",
]
