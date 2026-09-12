from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, cast

import math

from alpha_research.core.hashing import hash_json, require_sha256


class ExperimentProfile(str, Enum):
    RESEARCH = "research"
    PRODUCTION_CANDIDATE = "production_candidate"


class DataPartition(str, Enum):
    DISCOVERY = "discovery"
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    HOLDOUT = "holdout"


CANONICAL_STAGES = (
    "data_quality",
    "factor_generation",
    "label_building",
    "validation_split",
    "model_training",
    "factor_evaluation",
    "score_construction",
    "portfolio_construction",
    "backtest",
    "robustness",
    "report",
    "admission",
)

_AGENT_VISIBLE_PARTITIONS = frozenset({DataPartition.DISCOVERY, DataPartition.TRAIN})


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    maximum_attempts: int
    maximum_wall_seconds: float
    maximum_cpu_seconds: float
    maximum_peak_memory_bytes: int
    maximum_disk_write_bytes: int
    maximum_parallel_tasks: int
    per_stage_timeout_seconds: float
    maximum_llm_calls: int = 0
    maximum_llm_tokens: int = 0
    maximum_llm_cost_microusd: int = 0
    schema_version: str = "resource-budget/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "resource-budget/v1":
            raise ValueError("unsupported ResourceBudget schema")
        for name in (
            "maximum_attempts",
            "maximum_peak_memory_bytes",
            "maximum_disk_write_bytes",
            "maximum_parallel_tasks",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"resource budget {name} must be a positive integer")
        for name in (
            "maximum_llm_calls",
            "maximum_llm_tokens",
            "maximum_llm_cost_microusd",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"resource budget {name} must be non-negative")
        for name in (
            "maximum_wall_seconds",
            "maximum_cpu_seconds",
            "per_stage_timeout_seconds",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"resource budget {name} must be positive")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "maximum_attempts": self.maximum_attempts,
            "maximum_wall_seconds": self.maximum_wall_seconds,
            "maximum_cpu_seconds": self.maximum_cpu_seconds,
            "maximum_peak_memory_bytes": self.maximum_peak_memory_bytes,
            "maximum_disk_write_bytes": self.maximum_disk_write_bytes,
            "maximum_parallel_tasks": self.maximum_parallel_tasks,
            "per_stage_timeout_seconds": self.per_stage_timeout_seconds,
            "maximum_llm_calls": self.maximum_llm_calls,
            "maximum_llm_tokens": self.maximum_llm_tokens,
            "maximum_llm_cost_microusd": self.maximum_llm_cost_microusd,
        }


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    maximum_attempts_per_stage: int
    initial_backoff_seconds: float
    maximum_backoff_seconds: float
    backoff_multiplier: float
    jitter_fraction: float
    retryable_failure_codes: tuple[str, ...]
    schema_version: str = "retry-policy/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "retry-policy/v1":
            raise ValueError("unsupported RetryPolicy schema")
        if (
            not isinstance(self.maximum_attempts_per_stage, int)
            or isinstance(self.maximum_attempts_per_stage, bool)
            or self.maximum_attempts_per_stage <= 0
        ):
            raise ValueError("retry maximum_attempts_per_stage must be positive")
        for name in (
            "initial_backoff_seconds",
            "maximum_backoff_seconds",
            "backoff_multiplier",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"retry {name} must be positive")
        if self.maximum_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("retry maximum backoff is below initial backoff")
        if self.backoff_multiplier < 1:
            raise ValueError("retry backoff_multiplier must be at least one")
        if (
            not math.isfinite(self.jitter_fraction)
            or not 0 <= self.jitter_fraction <= 1
        ):
            raise ValueError("retry jitter_fraction must lie in [0,1]")
        codes = tuple(sorted(set(self.retryable_failure_codes)))
        if codes != self.retryable_failure_codes:
            raise ValueError("retryable failure codes must be sorted and unique")
        if any(not _safe_code(code) for code in codes):
            raise ValueError("retryable failure code contains unsupported characters")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def delay_seconds(self, retry_number: int, *, jitter_unit: float = 0.5) -> float:
        if not isinstance(retry_number, int) or retry_number < 0:
            raise ValueError("retry_number must be a non-negative integer")
        if not math.isfinite(jitter_unit) or not 0 <= jitter_unit <= 1:
            raise ValueError("jitter_unit must lie in [0,1]")
        base = min(
            self.maximum_backoff_seconds,
            self.initial_backoff_seconds * self.backoff_multiplier**retry_number,
        )
        jitter = (2 * jitter_unit - 1) * self.jitter_fraction
        return float(max(0.0, base * (1 + jitter)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "maximum_attempts_per_stage": self.maximum_attempts_per_stage,
            "initial_backoff_seconds": self.initial_backoff_seconds,
            "maximum_backoff_seconds": self.maximum_backoff_seconds,
            "backoff_multiplier": self.backoff_multiplier,
            "jitter_fraction": self.jitter_fraction,
            "retryable_failure_codes": list(self.retryable_failure_codes),
        }


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    experiment_id: str
    version: str
    profile: ExperimentProfile
    data_partitions: Mapping[DataPartition, str]
    agent_visible_partitions: tuple[DataPartition, ...]
    factor_spec_hashes: tuple[str, ...]
    label_spec_hash: str
    validation_spec_hash: str
    evaluation_spec_hash: str
    model_spec_hash: str | None
    portfolio_spec_hash: str | None
    cost_model_hash: str | None
    robustness_spec_hash: str | None
    code_snapshot_hash: str
    environment_hash: str
    llm_policy_hash: str | None
    llm_transport_hash: str | None
    stages: tuple[str, ...]
    random_seed: int
    resource_budget: ResourceBudget
    retry_policy: RetryPolicy
    scientific_lineage_manifest_hash: str | None = None
    schema_version: str = "experiment-spec/v2"

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile", ExperimentProfile(self.profile))
        if self.schema_version not in {
            "experiment-spec/v2",
            "experiment-spec/v3",
        }:
            raise ValueError("unsupported ExperimentSpec schema")
        if self.schema_version == "experiment-spec/v2":
            if self.scientific_lineage_manifest_hash is not None:
                raise ValueError("ExperimentSpec v2 cannot bind scientific lineage")
        elif self.scientific_lineage_manifest_hash is None:
            raise ValueError("ExperimentSpec v3 requires scientific lineage")
        if not _safe_code(self.experiment_id) or not _safe_code(self.version):
            raise ValueError("experiment id/version contains unsupported characters")
        partitions = {
            DataPartition(name): require_sha256(digest, name=f"data partition:{name}")
            for name, digest in self.data_partitions.items()
        }
        if not partitions or DataPartition.DISCOVERY not in partitions:
            raise ValueError("experiment requires a discovery data partition")
        if len(set(partitions.values())) != len(partitions):
            repeated = sorted(
                digest
                for digest in set(partitions.values())
                if tuple(partitions.values()).count(digest) > 1
            )
            aliases = [
                "/".join(
                    sorted(
                        partition.value
                        for partition, value in partitions.items()
                        if value == digest
                    )
                )
                for digest in repeated
            ]
            raise ValueError(
                "experiment data partitions must bind distinct content hashes:"
                + ",".join(aliases)
            )
        object.__setattr__(
            self,
            "data_partitions",
            MappingProxyType(
                dict(sorted(partitions.items(), key=lambda item: str(item[0])))
            ),
        )
        visible = tuple(DataPartition(value) for value in self.agent_visible_partitions)
        if len(set(visible)) != len(visible):
            raise ValueError("agent-visible partitions must be unique")
        if not set(visible).issubset(partitions):
            raise ValueError("agent-visible partition is not bound to this experiment")
        forbidden = set(visible).difference(_AGENT_VISIBLE_PARTITIONS)
        if forbidden:
            names = ",".join(sorted(item.value for item in forbidden))
            raise ValueError(
                f"agent context cannot access protected partitions:{names}"
            )
        object.__setattr__(self, "agent_visible_partitions", visible)
        if not self.factor_spec_hashes or len(set(self.factor_spec_hashes)) != len(
            self.factor_spec_hashes
        ):
            raise ValueError("experiment factor hashes must be non-empty and unique")
        for offset, digest in enumerate(self.factor_spec_hashes):
            require_sha256(digest, name=f"factor spec hash:{offset}")
        for name in (
            "label_spec_hash",
            "validation_spec_hash",
            "evaluation_spec_hash",
            "code_snapshot_hash",
            "environment_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"experiment {name}")
        for name in (
            "model_spec_hash",
            "portfolio_spec_hash",
            "cost_model_hash",
            "robustness_spec_hash",
            "llm_policy_hash",
            "llm_transport_hash",
            "scientific_lineage_manifest_hash",
        ):
            value = getattr(self, name)
            if value is not None:
                require_sha256(value, name=f"experiment {name}")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise TypeError("experiment random_seed must be an integer")
        if not self.stages or len(set(self.stages)) != len(self.stages):
            raise ValueError("experiment stages must be non-empty and unique")
        positions = []
        for stage in self.stages:
            if stage not in CANONICAL_STAGES:
                raise ValueError(f"unknown experiment stage:{stage}")
            positions.append(CANONICAL_STAGES.index(stage))
        if positions != sorted(positions):
            raise ValueError("experiment stages must follow canonical order")
        if self.llm_policy_hash is None:
            if any(
                (
                    self.resource_budget.maximum_llm_calls,
                    self.resource_budget.maximum_llm_tokens,
                    self.resource_budget.maximum_llm_cost_microusd,
                )
            ):
                raise ValueError("LLM budget requires a bound LLM policy")
            if self.llm_transport_hash is not None:
                raise ValueError("LLM transport requires a bound LLM policy")
        elif self.resource_budget.maximum_llm_calls <= 0:
            raise ValueError("bound LLM policy requires a positive LLM call budget")
        elif self.llm_transport_hash is None:
            raise ValueError("bound LLM policy requires a bound LLM transport")
        elif not visible:
            raise ValueError("bound LLM policy requires an explicit safe data context")
        stage_contracts = {
            "model_training": self.model_spec_hash,
            "portfolio_construction": self.portfolio_spec_hash,
            "backtest": self.cost_model_hash,
            "robustness": self.robustness_spec_hash,
        }
        missing_stage_contracts = sorted(
            stage
            for stage, contract in stage_contracts.items()
            if stage in self.stages and contract is None
        )
        if missing_stage_contracts:
            raise ValueError(
                "experiment stage lacks its contract:"
                + ",".join(missing_stage_contracts)
            )
        profile = self.profile
        if not isinstance(profile, ExperimentProfile):  # pragma: no cover
            raise RuntimeError("experiment profile was not normalized")
        if profile is ExperimentProfile.PRODUCTION_CANDIDATE:
            required = {
                "portfolio_spec_hash": self.portfolio_spec_hash,
                "cost_model_hash": self.cost_model_hash,
                "robustness_spec_hash": self.robustness_spec_hash,
            }
            missing = sorted(name for name, value in required.items() if value is None)
            if missing:
                raise ValueError(
                    "production-candidate experiment lacks required contracts:"
                    + ",".join(missing)
                )
            if not ({DataPartition.TEST, DataPartition.HOLDOUT} & set(partitions)):
                raise ValueError(
                    "production-candidate experiment requires a protected test/holdout partition"
                )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def component_bindings(self) -> tuple[tuple[str, str], ...]:
        bindings: list[tuple[str, str]] = [
            (f"data:{partition.value}", digest)
            for partition, digest in self.data_partitions.items()
        ]
        bindings.extend(
            (f"factor:{offset}", digest)
            for offset, digest in enumerate(self.factor_spec_hashes)
        )
        bindings.extend(
            [
                ("label", self.label_spec_hash),
                ("validation", self.validation_spec_hash),
                ("evaluation", self.evaluation_spec_hash),
                ("code", self.code_snapshot_hash),
                ("environment", self.environment_hash),
            ]
        )
        optional = (
            ("model", self.model_spec_hash),
            ("portfolio", self.portfolio_spec_hash),
            ("cost", self.cost_model_hash),
            ("robustness", self.robustness_spec_hash),
            ("llm_policy", self.llm_policy_hash),
            ("llm_transport", self.llm_transport_hash),
            (
                "scientific_lineage",
                self.scientific_lineage_manifest_hash,
            ),
        )
        bindings.extend((role, value) for role, value in optional if value is not None)
        return tuple(bindings)

    def to_dict(self) -> dict[str, object]:
        profile = self.profile
        if not isinstance(profile, ExperimentProfile):  # pragma: no cover
            raise RuntimeError("experiment profile was not normalized")
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "version": self.version,
            "profile": profile.value,
            "data_partitions": {
                partition.value: digest
                for partition, digest in self.data_partitions.items()
            },
            "agent_visible_partitions": [
                partition.value for partition in self.agent_visible_partitions
            ],
            "factor_spec_hashes": list(self.factor_spec_hashes),
            "label_spec_hash": self.label_spec_hash,
            "validation_spec_hash": self.validation_spec_hash,
            "evaluation_spec_hash": self.evaluation_spec_hash,
            "model_spec_hash": self.model_spec_hash,
            "portfolio_spec_hash": self.portfolio_spec_hash,
            "cost_model_hash": self.cost_model_hash,
            "robustness_spec_hash": self.robustness_spec_hash,
            "code_snapshot_hash": self.code_snapshot_hash,
            "environment_hash": self.environment_hash,
            "llm_policy_hash": self.llm_policy_hash,
            "llm_transport_hash": self.llm_transport_hash,
            "stages": list(self.stages),
            "random_seed": self.random_seed,
            "resource_budget": self.resource_budget.to_dict(),
            "retry_policy": self.retry_policy.to_dict(),
        }
        if self.schema_version == "experiment-spec/v3":
            payload["scientific_lineage_manifest_hash"] = (
                self.scientific_lineage_manifest_hash
            )
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ExperimentSpec":
        schema = value.get("schema_version")
        if schema not in {"experiment-spec/v2", "experiment-spec/v3"}:
            raise ValueError("unsupported ExperimentSpec schema")
        expected = {
            "schema_version",
            "experiment_id",
            "version",
            "profile",
            "data_partitions",
            "agent_visible_partitions",
            "factor_spec_hashes",
            "label_spec_hash",
            "validation_spec_hash",
            "evaluation_spec_hash",
            "model_spec_hash",
            "portfolio_spec_hash",
            "cost_model_hash",
            "robustness_spec_hash",
            "code_snapshot_hash",
            "environment_hash",
            "llm_policy_hash",
            "llm_transport_hash",
            "stages",
            "random_seed",
            "resource_budget",
            "retry_policy",
        }
        if schema == "experiment-spec/v3":
            expected.add("scientific_lineage_manifest_hash")
        if set(value) != expected:
            raise ValueError("ExperimentSpec wire fields differ")
        partitions = _string_mapping(value["data_partitions"], name="data_partitions")
        visible = _string_list(
            value["agent_visible_partitions"],
            name="agent_visible_partitions",
        )
        factors = _string_list(value["factor_spec_hashes"], name="factor_spec_hashes")
        stages = _string_list(value["stages"], name="stages")
        budget = _object(value["resource_budget"], name="resource_budget")
        retry = _object(value["retry_policy"], name="retry_policy")
        budget_expected = {
            "schema_version",
            "maximum_attempts",
            "maximum_wall_seconds",
            "maximum_cpu_seconds",
            "maximum_peak_memory_bytes",
            "maximum_disk_write_bytes",
            "maximum_parallel_tasks",
            "per_stage_timeout_seconds",
            "maximum_llm_calls",
            "maximum_llm_tokens",
            "maximum_llm_cost_microusd",
        }
        retry_expected = {
            "schema_version",
            "maximum_attempts_per_stage",
            "initial_backoff_seconds",
            "maximum_backoff_seconds",
            "backoff_multiplier",
            "jitter_fraction",
            "retryable_failure_codes",
        }
        if set(budget) != budget_expected:
            raise ValueError("ResourceBudget wire fields differ")
        if set(retry) != retry_expected:
            raise ValueError("RetryPolicy wire fields differ")
        retry_codes = _string_list(
            retry["retryable_failure_codes"],
            name="retryable_failure_codes",
        )
        return cls(
            schema_version=str(schema),
            experiment_id=_string(value["experiment_id"], name="experiment_id"),
            version=_string(value["version"], name="version"),
            profile=ExperimentProfile(_string(value["profile"], name="profile")),
            data_partitions={
                DataPartition(name): digest for name, digest in partitions.items()
            },
            agent_visible_partitions=tuple(DataPartition(item) for item in visible),
            factor_spec_hashes=tuple(factors),
            label_spec_hash=_string(value["label_spec_hash"], name="label_spec_hash"),
            validation_spec_hash=_string(
                value["validation_spec_hash"], name="validation_spec_hash"
            ),
            evaluation_spec_hash=_string(
                value["evaluation_spec_hash"], name="evaluation_spec_hash"
            ),
            model_spec_hash=_optional_string(
                value["model_spec_hash"], name="model_spec_hash"
            ),
            portfolio_spec_hash=_optional_string(
                value["portfolio_spec_hash"], name="portfolio_spec_hash"
            ),
            cost_model_hash=_optional_string(
                value["cost_model_hash"], name="cost_model_hash"
            ),
            robustness_spec_hash=_optional_string(
                value["robustness_spec_hash"], name="robustness_spec_hash"
            ),
            code_snapshot_hash=_string(
                value["code_snapshot_hash"], name="code_snapshot_hash"
            ),
            environment_hash=_string(
                value["environment_hash"], name="environment_hash"
            ),
            llm_policy_hash=_optional_string(
                value["llm_policy_hash"], name="llm_policy_hash"
            ),
            llm_transport_hash=_optional_string(
                value["llm_transport_hash"], name="llm_transport_hash"
            ),
            stages=tuple(stages),
            random_seed=_integer(value["random_seed"], name="random_seed"),
            resource_budget=ResourceBudget(
                schema_version=_string(
                    budget["schema_version"],
                    name="resource_budget.schema_version",
                ),
                maximum_attempts=_integer(
                    budget["maximum_attempts"], name="maximum_attempts"
                ),
                maximum_wall_seconds=_number(
                    budget["maximum_wall_seconds"],
                    name="maximum_wall_seconds",
                ),
                maximum_cpu_seconds=_number(
                    budget["maximum_cpu_seconds"],
                    name="maximum_cpu_seconds",
                ),
                maximum_peak_memory_bytes=_integer(
                    budget["maximum_peak_memory_bytes"],
                    name="maximum_peak_memory_bytes",
                ),
                maximum_disk_write_bytes=_integer(
                    budget["maximum_disk_write_bytes"],
                    name="maximum_disk_write_bytes",
                ),
                maximum_parallel_tasks=_integer(
                    budget["maximum_parallel_tasks"],
                    name="maximum_parallel_tasks",
                ),
                per_stage_timeout_seconds=_number(
                    budget["per_stage_timeout_seconds"],
                    name="per_stage_timeout_seconds",
                ),
                maximum_llm_calls=_integer(
                    budget["maximum_llm_calls"], name="maximum_llm_calls"
                ),
                maximum_llm_tokens=_integer(
                    budget["maximum_llm_tokens"], name="maximum_llm_tokens"
                ),
                maximum_llm_cost_microusd=_integer(
                    budget["maximum_llm_cost_microusd"],
                    name="maximum_llm_cost_microusd",
                ),
            ),
            retry_policy=RetryPolicy(
                schema_version=_string(
                    retry["schema_version"],
                    name="retry_policy.schema_version",
                ),
                maximum_attempts_per_stage=_integer(
                    retry["maximum_attempts_per_stage"],
                    name="maximum_attempts_per_stage",
                ),
                initial_backoff_seconds=_number(
                    retry["initial_backoff_seconds"],
                    name="initial_backoff_seconds",
                ),
                maximum_backoff_seconds=_number(
                    retry["maximum_backoff_seconds"],
                    name="maximum_backoff_seconds",
                ),
                backoff_multiplier=_number(
                    retry["backoff_multiplier"], name="backoff_multiplier"
                ),
                jitter_fraction=_number(
                    retry["jitter_fraction"], name="jitter_fraction"
                ),
                retryable_failure_codes=tuple(retry_codes),
            ),
            scientific_lineage_manifest_hash=(
                None
                if schema == "experiment-spec/v2"
                else _string(
                    value["scientific_lineage_manifest_hash"],
                    name="scientific_lineage_manifest_hash",
                )
            ),
        )


def _safe_code(value: str) -> bool:
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-")
    return value[0].isalnum() and all(character in allowed for character in value)


def _object(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return dict(value)


def _string_mapping(value: object, *, name: str) -> dict[str, str]:
    mapping = _object(value, name=name)
    if not all(
        isinstance(key, str) and isinstance(item, str) for key, item in mapping.items()
    ):
        raise TypeError(f"{name} must contain strings")
    return {str(key): str(item) for key, item in mapping.items()}


def _string_list(value: object, *, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{name} must be a string list")
    return list(value)


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _optional_string(value: object, *, name: str) -> str | None:
    return None if value is None else _string(value, name=name)


def _integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _number(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    return float(value)


__all__ = [
    "CANONICAL_STAGES",
    "DataPartition",
    "ExperimentProfile",
    "ExperimentSpec",
    "ResourceBudget",
    "RetryPolicy",
]
