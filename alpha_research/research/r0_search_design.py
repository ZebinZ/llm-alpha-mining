"""Immutable search-boundary authority for the R0 three-source campaign.

This contract is deliberately earlier than
:class:`~alpha_research.research.campaign_protocol.ResearchCampaignProtocol`.
It freezes what may be searched, which labels and feedback may be used, and
when the search must stop.  It intentionally does *not* contain a candidate
family hash or implement evaluation: the terminal campaign protocol and
existing protected evaluator remain the sole locked-test boundary.
"""

from __future__ import annotations

import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, cast

from alpha_research.core.hashing import (
    canonical_json_bytes,
    hash_json,
    require_sha256,
)
from alpha_research.research.campaign_protocol import (
    R0_ABLATION_ARMS,
    R0_DATA_SOURCES,
    R0_MODEL_LADDER,
)


R0_202106_PILOT_TRADING_DATES = (
    "20210601",
    "20210602",
    "20210603",
    "20210604",
    "20210607",
    "20210608",
    "20210609",
    "20210610",
    "20210611",
    "20210615",
    "20210616",
    "20210617",
    "20210618",
    "20210621",
    "20210622",
    "20210623",
    "20210624",
    "20210625",
    "20210628",
    "20210629",
    "20210630",
)

_SCHEMA_VERSION = "r0-search-design-authority/v1"
_AUTHORITY_ID = "r0-three-source-alpha-search-design"
_AUTHORITY_VERSION = "1"
_CUTOVER_DATE = "2021-06-07"
_PILOT_MONTH = "202106"
_FINAL_MONTH_START = "202001"
_FINAL_MONTH_END = "202112"

_PRIMARY_HORIZON = 1
_SECONDARY_HORIZONS = (5, 20)
_SIGNIFICANCE_HORIZON_KEYS = ("1", "5", "20")
_SECONDARY_HORIZON_ROLE = "diagnostic_ic_decay_only_cannot_replace_primary_or_promote"

_RAW_IDEA_BUDGET = 800
_FROZEN_FACTOR_SPEC_CAP = 240
_ADAPTIVE_VALIDATION_CAP = 80
_LOCKED_TEST_CAP = 30
_FINAL_SUBMISSION_TARGET = 18
_FINAL_SUBMISSION_MAXIMUM = 30
_MAXIMUM_GENERATIONS = 6
_PARETO_PATIENCE = 2

_DIRECTION_POLICY = "hypothesis_declared_before_any_label_result"
_VALIDATION_FEEDBACK_POLICY = "coarse_metrics_only_no_row_level_labels"
_LOCKED_TEST_ACCESS_POLICY = (
    "protected_terminal_evaluator_after_candidate_family_freeze"
)
_EXTERNAL_HOLDOUT_POLICY = "external_only_never_agent_visible"
_LOCAL_LOCKED_TEST_CLAIM = "historically_exposed_local_replay_not_oos"
_MULTIPLE_TESTING_METHOD = "benjamini_hochberg"
_MULTIPLE_TESTING_SCOPE = "full_tested_family"

_AGENT_VISIBLE_PARTITIONS = ("discovery", "train")
_PILOT_ASSURANCE_BOUNDARY = MappingProxyType(
    {
        "research_only": True,
        "provider_semantics_signed": False,
        "point_in_time_security_master_bound": False,
        "research_eligible": False,
        "ic_ready": False,
        "production_ready": False,
    }
)


@dataclass(frozen=True, slots=True)
class R0SearchDesignAuthority:
    """Fail-closed preregistration before candidate feedback is observed.

    The 2021-06 pilot proves the current data path and is bound exactly, but it
    is not promoted to a continuous IC-ready dataset.  A full 2020--2021
    catalog and the final candidate family remain mandatory terminal bindings.
    """

    pilot_joint_catalog_id: str
    pilot_joint_catalog_document_sha256: str
    pilot_joint_shard_id: str
    pilot_bundle_id: str
    pilot_bundle_manifest_sha256: str
    pilot_joint_spec_hash: str
    pilot_joint_spec_document_sha256: str
    activation_policy_hash: str
    activation_schedule_hash: str
    pilot_source_frame_hashes: Mapping[str, str]
    pilot_data_identity_hash: str
    primary_label_spec_hash: str
    secondary_label_spec_hashes: Mapping[str, str]
    validation_spec_hash: str
    evaluation_spec_hash: str
    significance_spec_hashes_by_horizon: Mapping[str, str]
    robustness_spec_hash: str
    cost_model_hash: str
    pareto_policy_hash: str
    resource_budget_hash: str
    partition_policy_hash: str
    data_sources: tuple[str, ...] = R0_DATA_SOURCES
    ablation_arms: tuple[tuple[str, ...], ...] = R0_ABLATION_ARMS
    cutover_date: str = _CUTOVER_DATE
    pilot_month: str = _PILOT_MONTH
    pilot_trading_dates: tuple[str, ...] = R0_202106_PILOT_TRADING_DATES
    pilot_assurance_boundary: Mapping[str, bool] = _PILOT_ASSURANCE_BOUNDARY
    required_final_month_start: str = _FINAL_MONTH_START
    required_final_month_end: str = _FINAL_MONTH_END
    terminal_full_catalog_binding_required: bool = True
    terminal_candidate_family_binding_required: bool = True
    primary_horizon_sessions: int = _PRIMARY_HORIZON
    secondary_horizon_sessions: tuple[int, ...] = _SECONDARY_HORIZONS
    secondary_horizon_role: str = _SECONDARY_HORIZON_ROLE
    raw_idea_budget: int = _RAW_IDEA_BUDGET
    frozen_factor_spec_cap: int = _FROZEN_FACTOR_SPEC_CAP
    adaptive_validation_cap: int = _ADAPTIVE_VALIDATION_CAP
    locked_test_cap: int = _LOCKED_TEST_CAP
    final_submission_target: int = _FINAL_SUBMISSION_TARGET
    final_submission_maximum: int = _FINAL_SUBMISSION_MAXIMUM
    maximum_generations: int = _MAXIMUM_GENERATIONS
    pareto_no_improvement_patience: int = _PARETO_PATIENCE
    model_ladder: tuple[str, ...] = R0_MODEL_LADDER
    direction_policy: str = _DIRECTION_POLICY
    failed_or_missing_p_value: float = 1.0
    multiple_testing_method: str = _MULTIPLE_TESTING_METHOD
    multiple_testing_scope: str = _MULTIPLE_TESTING_SCOPE
    external_scores_may_influence_search: bool = False
    agent_visible_partitions: tuple[str, ...] = _AGENT_VISIBLE_PARTITIONS
    validation_feedback_policy: str = _VALIDATION_FEEDBACK_POLICY
    locked_test_access_policy: str = _LOCKED_TEST_ACCESS_POLICY
    external_holdout_policy: str = _EXTERNAL_HOLDOUT_POLICY
    local_locked_test_claim: str = _LOCAL_LOCKED_TEST_CLAIM
    research_only: bool = True
    human_approval_required: bool = True
    authority_id: str = _AUTHORITY_ID
    authority_version: str = _AUTHORITY_VERSION
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_fixed_search_boundary()
        self._normalize_and_validate_hash_bindings()

    def _validate_fixed_search_boundary(self) -> None:
        fixed_values: tuple[tuple[str, object, object], ...] = (
            ("schema_version", self.schema_version, _SCHEMA_VERSION),
            ("authority_id", self.authority_id, _AUTHORITY_ID),
            ("authority_version", self.authority_version, _AUTHORITY_VERSION),
            ("data_sources", self.data_sources, R0_DATA_SOURCES),
            ("ablation_arms", self.ablation_arms, R0_ABLATION_ARMS),
            ("cutover_date", self.cutover_date, _CUTOVER_DATE),
            ("pilot_month", self.pilot_month, _PILOT_MONTH),
            (
                "pilot_trading_dates",
                self.pilot_trading_dates,
                R0_202106_PILOT_TRADING_DATES,
            ),
            (
                "required_final_month_start",
                self.required_final_month_start,
                _FINAL_MONTH_START,
            ),
            (
                "required_final_month_end",
                self.required_final_month_end,
                _FINAL_MONTH_END,
            ),
            (
                "primary_horizon_sessions",
                self.primary_horizon_sessions,
                _PRIMARY_HORIZON,
            ),
            (
                "secondary_horizon_sessions",
                self.secondary_horizon_sessions,
                _SECONDARY_HORIZONS,
            ),
            (
                "secondary_horizon_role",
                self.secondary_horizon_role,
                _SECONDARY_HORIZON_ROLE,
            ),
            ("raw_idea_budget", self.raw_idea_budget, _RAW_IDEA_BUDGET),
            (
                "frozen_factor_spec_cap",
                self.frozen_factor_spec_cap,
                _FROZEN_FACTOR_SPEC_CAP,
            ),
            (
                "adaptive_validation_cap",
                self.adaptive_validation_cap,
                _ADAPTIVE_VALIDATION_CAP,
            ),
            ("locked_test_cap", self.locked_test_cap, _LOCKED_TEST_CAP),
            (
                "final_submission_target",
                self.final_submission_target,
                _FINAL_SUBMISSION_TARGET,
            ),
            (
                "final_submission_maximum",
                self.final_submission_maximum,
                _FINAL_SUBMISSION_MAXIMUM,
            ),
            (
                "maximum_generations",
                self.maximum_generations,
                _MAXIMUM_GENERATIONS,
            ),
            (
                "pareto_no_improvement_patience",
                self.pareto_no_improvement_patience,
                _PARETO_PATIENCE,
            ),
            ("model_ladder", self.model_ladder, R0_MODEL_LADDER),
            ("direction_policy", self.direction_policy, _DIRECTION_POLICY),
            (
                "multiple_testing_method",
                self.multiple_testing_method,
                _MULTIPLE_TESTING_METHOD,
            ),
            (
                "multiple_testing_scope",
                self.multiple_testing_scope,
                _MULTIPLE_TESTING_SCOPE,
            ),
            (
                "agent_visible_partitions",
                self.agent_visible_partitions,
                _AGENT_VISIBLE_PARTITIONS,
            ),
            (
                "validation_feedback_policy",
                self.validation_feedback_policy,
                _VALIDATION_FEEDBACK_POLICY,
            ),
            (
                "locked_test_access_policy",
                self.locked_test_access_policy,
                _LOCKED_TEST_ACCESS_POLICY,
            ),
            (
                "external_holdout_policy",
                self.external_holdout_policy,
                _EXTERNAL_HOLDOUT_POLICY,
            ),
            (
                "local_locked_test_claim",
                self.local_locked_test_claim,
                _LOCAL_LOCKED_TEST_CLAIM,
            ),
        )
        for name, observed, expected in fixed_values:
            if not _exactly_equal(observed, expected):
                raise ValueError(f"R0 search design {name} differs")

        required_true = (
            "terminal_full_catalog_binding_required",
            "terminal_candidate_family_binding_required",
            "research_only",
            "human_approval_required",
        )
        for name in required_true:
            if getattr(self, name) is not True:
                raise ValueError(f"R0 search design {name} must be true")
        if type(self.external_scores_may_influence_search) is not bool:
            raise TypeError("external_scores_may_influence_search must be boolean")
        if self.external_scores_may_influence_search:
            raise ValueError("external scores must not influence R0 search")
        if (
            type(self.failed_or_missing_p_value) not in (int, float)
            or isinstance(self.failed_or_missing_p_value, bool)
            or float(self.failed_or_missing_p_value) != 1.0
        ):
            raise ValueError("failed or missing R0 search p-values must equal 1.0")
        object.__setattr__(self, "failed_or_missing_p_value", 1.0)

        assurance = _normalize_assurance(self.pilot_assurance_boundary)
        if assurance != dict(_PILOT_ASSURANCE_BOUNDARY):
            raise ValueError("R0 pilot assurance boundary may not be promoted")
        object.__setattr__(
            self,
            "pilot_assurance_boundary",
            MappingProxyType(assurance),
        )

    def _normalize_and_validate_hash_bindings(self) -> None:
        for name in (
            "pilot_joint_catalog_id",
            "pilot_joint_catalog_document_sha256",
            "pilot_joint_shard_id",
            "pilot_bundle_id",
            "pilot_bundle_manifest_sha256",
            "pilot_joint_spec_hash",
            "pilot_joint_spec_document_sha256",
            "activation_policy_hash",
            "activation_schedule_hash",
            "pilot_data_identity_hash",
            "primary_label_spec_hash",
            "validation_spec_hash",
            "evaluation_spec_hash",
            "robustness_spec_hash",
            "cost_model_hash",
            "pareto_policy_hash",
            "resource_budget_hash",
            "partition_policy_hash",
        ):
            require_sha256(
                _exact_string(getattr(self, name), name=name),
                name=f"R0 search design {name}",
            )

        source_hashes = _normalize_hash_mapping(
            self.pilot_source_frame_hashes,
            name="pilot_source_frame_hashes",
            expected_keys=R0_DATA_SOURCES,
        )
        expected_data_identity = hash_json(source_hashes)
        if not hmac.compare_digest(
            self.pilot_data_identity_hash,
            expected_data_identity,
        ):
            raise ValueError(
                "R0 pilot composite data identity differs from source frame hashes"
            )
        object.__setattr__(
            self,
            "pilot_source_frame_hashes",
            MappingProxyType(source_hashes),
        )

        secondary_labels = _normalize_hash_mapping(
            self.secondary_label_spec_hashes,
            name="secondary_label_spec_hashes",
            expected_keys=("5", "20"),
        )
        label_hashes = (self.primary_label_spec_hash, *secondary_labels.values())
        if len(set(label_hashes)) != len(label_hashes):
            raise ValueError("R0 label spec hashes must be unique by horizon")
        object.__setattr__(
            self,
            "secondary_label_spec_hashes",
            MappingProxyType(secondary_labels),
        )

        significance = _normalize_hash_mapping(
            self.significance_spec_hashes_by_horizon,
            name="significance_spec_hashes_by_horizon",
            expected_keys=_SIGNIFICANCE_HORIZON_KEYS,
        )
        if len(set(significance.values())) != len(significance):
            raise ValueError("R0 significance specs must be unique by horizon")
        object.__setattr__(
            self,
            "significance_spec_hashes_by_horizon",
            MappingProxyType(significance),
        )

    @property
    def content_hash(self) -> str:
        """SHA-256 identity of the canonical search-design payload."""

        return cast(str, hash_json(self._payload_dict()))

    def _payload_dict(self) -> dict[str, object]:
        return {
            field.name: _to_wire_value(getattr(self, field.name))
            for field in fields(self)
        }

    def to_dict(self) -> dict[str, object]:
        payload = self._payload_dict()
        return {**payload, "content_hash": self.content_hash}

    def to_wire_bytes(self) -> bytes:
        """Return canonical UTF-8 JSON with one trailing newline."""

        return cast(bytes, canonical_json_bytes(self.to_dict())) + b"\n"

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "R0SearchDesignAuthority":
        expected = {field.name for field in fields(cls)} | {"content_hash"}
        if set(value) != expected:
            raise ValueError("R0SearchDesignAuthority wire fields differ")

        content_hash = _wire_string(value["content_hash"], name="content_hash")
        require_sha256(content_hash, name="R0 search design wire content_hash")
        payload = dict(value)
        del payload["content_hash"]
        payload["data_sources"] = _wire_string_tuple(
            payload["data_sources"], name="data_sources"
        )
        payload["ablation_arms"] = _wire_ablation_arms(payload["ablation_arms"])
        payload["pilot_trading_dates"] = _wire_string_tuple(
            payload["pilot_trading_dates"], name="pilot_trading_dates"
        )
        payload["pilot_source_frame_hashes"] = _wire_hash_mapping(
            payload["pilot_source_frame_hashes"],
            name="pilot_source_frame_hashes",
        )
        payload["pilot_assurance_boundary"] = _wire_bool_mapping(
            payload["pilot_assurance_boundary"],
            name="pilot_assurance_boundary",
        )
        payload["secondary_label_spec_hashes"] = _wire_hash_mapping(
            payload["secondary_label_spec_hashes"],
            name="secondary_label_spec_hashes",
        )
        payload["secondary_horizon_sessions"] = _wire_integer_tuple(
            payload["secondary_horizon_sessions"],
            name="secondary_horizon_sessions",
        )
        payload["significance_spec_hashes_by_horizon"] = _wire_hash_mapping(
            payload["significance_spec_hashes_by_horizon"],
            name="significance_spec_hashes_by_horizon",
        )
        payload["model_ladder"] = _wire_string_tuple(
            payload["model_ladder"], name="model_ladder"
        )
        payload["agent_visible_partitions"] = _wire_string_tuple(
            payload["agent_visible_partitions"],
            name="agent_visible_partitions",
        )
        for name in (
            "primary_horizon_sessions",
            "raw_idea_budget",
            "frozen_factor_spec_cap",
            "adaptive_validation_cap",
            "locked_test_cap",
            "final_submission_target",
            "final_submission_maximum",
            "maximum_generations",
            "pareto_no_improvement_patience",
        ):
            payload[name] = _wire_integer(payload[name], name=name)
        for name in (
            "terminal_full_catalog_binding_required",
            "terminal_candidate_family_binding_required",
            "external_scores_may_influence_search",
            "research_only",
            "human_approval_required",
        ):
            payload[name] = _wire_boolean(payload[name], name=name)
        payload["failed_or_missing_p_value"] = _wire_number(
            payload["failed_or_missing_p_value"],
            name="failed_or_missing_p_value",
        )
        result = cls(**cast(Any, payload))
        if not hmac.compare_digest(result.content_hash, content_hash):
            raise ValueError("R0SearchDesignAuthority content_hash mismatch")
        return result

    @classmethod
    def from_wire_bytes(cls, payload: bytes) -> "R0SearchDesignAuthority":
        if type(payload) is not bytes:
            raise TypeError("R0SearchDesignAuthority wire payload must be bytes")
        try:
            decoded = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("R0SearchDesignAuthority wire is invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise TypeError("R0SearchDesignAuthority wire root must be an object")
        if payload != canonical_json_bytes(decoded) + b"\n":
            raise ValueError("R0SearchDesignAuthority wire is not canonical")
        return cls.from_mapping(decoded)


def _exactly_equal(observed: object, expected: object) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, tuple):
        assert isinstance(observed, tuple)
        return len(observed) == len(expected) and all(
            _exactly_equal(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def _to_wire_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _to_wire_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_wire_value(item) for item in value]
    if type(value) in (str, int, float, bool) or value is None:
        return value
    raise TypeError(f"unsupported R0 search design wire value:{type(value).__name__}")


def _exact_string(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be canonical and non-empty")
    return value


def _normalize_hash_mapping(
    value: Mapping[str, str],
    *,
    name: str,
    expected_keys: tuple[str, ...],
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if set(value) != set(expected_keys):
        raise ValueError(f"{name} keys differ")
    result: dict[str, str] = {}
    for key in expected_keys:
        digest = _exact_string(value[key], name=f"{name}:{key}")
        result[key] = require_sha256(digest, name=f"R0 search design {name}:{key}")
    return result


def _normalize_assurance(value: Mapping[str, bool]) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise TypeError("pilot_assurance_boundary must be a mapping")
    if set(value) != set(_PILOT_ASSURANCE_BOUNDARY):
        raise ValueError("pilot_assurance_boundary keys differ")
    result: dict[str, bool] = {}
    for key in _PILOT_ASSURANCE_BOUNDARY:
        observed = value[key]
        if type(observed) is not bool:
            raise TypeError(f"pilot assurance {key} must be boolean")
        result[key] = observed
    return result


def _wire_string(value: object, *, name: str) -> str:
    return _exact_string(value, name=f"wire {name}")


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


def _wire_integer_tuple(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(type(item) is int for item in value):
        raise TypeError(f"wire {name} must be an integer array")
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


def _wire_hash_mapping(value: object, *, name: str) -> Mapping[str, str]:
    if not isinstance(value, dict) or not all(
        type(key) is str and type(digest) is str for key, digest in value.items()
    ):
        raise TypeError(f"wire {name} must be a string object")
    return {str(key): str(digest) for key, digest in value.items()}


def _wire_bool_mapping(value: object, *, name: str) -> Mapping[str, bool]:
    if not isinstance(value, dict) or not all(
        type(key) is str and type(flag) is bool for key, flag in value.items()
    ):
        raise TypeError(f"wire {name} must be a boolean object")
    return {str(key): bool(flag) for key, flag in value.items()}


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
    "R0_202106_PILOT_TRADING_DATES",
    "R0SearchDesignAuthority",
]
