from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, cast

import re
import pandas as pd

from alpha_research.core.data import DataRole
from alpha_research.core.hashing import hash_json, require_sha256
from alpha_research.experiments.spec import DataPartition, ExperimentProfile


class ResearchStage(str, Enum):
    DATA_QUALITY = "data_quality"
    FACTOR_GENERATION = "factor_generation"
    LABEL_BUILDING = "label_building"
    VALIDATION_SPLIT = "validation_split"
    MODEL_TRAINING = "model_training"
    FACTOR_EVALUATION = "factor_evaluation"
    SCORE_CONSTRUCTION = "score_construction"
    PORTFOLIO_CONSTRUCTION = "portfolio_construction"
    BACKTEST = "backtest"
    ROBUSTNESS = "robustness"
    REPORT = "report"
    ADMISSION = "admission"


RESEARCH_STAGE_ORDER = tuple(ResearchStage)

_CANONICAL_DATA_ROLES: Mapping[DataPartition, DataRole] = MappingProxyType(
    {
        DataPartition.DISCOVERY: DataRole.RESEARCH,
        DataPartition.TRAIN: DataRole.TRAIN,
        DataPartition.VALIDATION: DataRole.VALIDATION,
        DataPartition.TEST: DataRole.LOCAL_TEST,
        DataPartition.HOLDOUT: DataRole.EXTERNAL_HOLDOUT,
    }
)


def canonical_data_role(partition: DataPartition | str) -> DataRole:
    """Return the one permitted core role for an experiment partition."""

    return _CANONICAL_DATA_ROLES[DataPartition(partition)]


@dataclass(frozen=True, slots=True)
class ResearchRunSpec:
    """Complete identity of one governed statistical-arbitrage research run.

    This contract deliberately binds *specification* hashes, not mutable paths.
    Runtime artifacts are bound separately by :class:`StageArtifactDescriptor`.
    Optional downstream blocks are all-or-none where partial identity would make
    a result ambiguous (for example cost + execution + backtest).
    """

    run_id: str
    version: str
    profile: ExperimentProfile | str
    data_partition_hashes: Mapping[DataPartition | str, str]
    data_partition_roles: Mapping[DataPartition | str, DataRole | str]
    data_partition_windows: Mapping[DataPartition | str, tuple[str, str]]
    data_snapshot_hash: str
    data_request_hash: str
    data_quality_spec_hash: str
    factor_spec_hashes: tuple[str, ...]
    label_spec_hash: str
    validation_spec_hash: str
    evaluation_spec_hash: str
    report_spec_hash: str
    model_spec_hash: str | None = None
    score_spec_hash: str | None = None
    portfolio_spec_hash: str | None = None
    risk_spec_hash: str | None = None
    cost_model_hash: str | None = None
    execution_spec_hash: str | None = None
    backtest_spec_hash: str | None = None
    robustness_spec_hash: str | None = None
    admission_spec_hash: str | None = None
    schema_version: str = "research-run-spec/v2"

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str):
            raise TypeError("research schema_version must be text")
        if self.schema_version != "research-run-spec/v2":
            raise ValueError("unsupported ResearchRunSpec schema")
        if not _safe_code(self.run_id) or not _safe_code(self.version):
            raise ValueError("research run id/version contains unsupported characters")
        if not isinstance(self.profile, (ExperimentProfile, str)):
            raise TypeError("research profile must be text")
        object.__setattr__(self, "profile", ExperimentProfile(self.profile))

        partitions: dict[DataPartition, str] = {}
        for key, digest in self.data_partition_hashes.items():
            partition = DataPartition(key)
            if not isinstance(digest, str):
                raise TypeError(
                    f"research data partition hash must be text:{partition.value}"
                )
            partitions[partition] = require_sha256(
                digest, name=f"research data partition:{partition.value}"
            )
        roles = {
            DataPartition(key): DataRole(role)
            for key, role in self.data_partition_roles.items()
        }
        windows: dict[DataPartition, tuple[str, str]] = {}
        for key, raw_window in self.data_partition_windows.items():
            partition = DataPartition(key)
            if not isinstance(raw_window, (tuple, list)) or len(raw_window) != 2:
                raise TypeError("research partition window must be a start/end pair")
            if not all(isinstance(item, str) for item in raw_window):
                raise TypeError("research partition window values must be text")
            start = pd.Timestamp(raw_window[0])
            end = pd.Timestamp(raw_window[1])
            if start.tzinfo is None or end.tzinfo is None or start >= end:
                raise ValueError(
                    f"research partition window is invalid:{partition.value}"
                )
            windows[partition] = (start.isoformat(), end.isoformat())
        if set(partitions) != set(roles) or set(partitions) != set(windows):
            raise ValueError(
                "research partition hashes, roles and windows must have identical keys"
            )
        required_partitions = {
            DataPartition.DISCOVERY,
            DataPartition.TRAIN,
            DataPartition.VALIDATION,
        }
        missing_partitions = sorted(
            (item.value for item in required_partitions.difference(partitions))
        )
        if missing_partitions:
            raise ValueError(
                "research run lacks required data partitions:"
                + ",".join(missing_partitions)
            )
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
                "research data partitions must bind distinct content hashes:"
                + ",".join(aliases)
            )
        ordered_windows_by_time = sorted(
            (
                pd.Timestamp(start),
                pd.Timestamp(end),
                partition,
            )
            for partition, (start, end) in windows.items()
        )
        for previous, current in zip(
            ordered_windows_by_time, ordered_windows_by_time[1:], strict=False
        ):
            if current[0] < previous[1]:
                raise ValueError(
                    "research data partition windows overlap:"
                    f"{previous[2].value}/{current[2].value}"
                )
        mismatches = sorted(
            partition.value
            for partition, role in roles.items()
            if role is not canonical_data_role(partition)
        )
        if mismatches:
            raise ValueError(
                "research partition/core role mapping differs:" + ",".join(mismatches)
            )
        ordered_partitions = dict(
            sorted(partitions.items(), key=lambda item: item[0].value)
        )
        ordered_roles = dict(sorted(roles.items(), key=lambda item: item[0].value))
        ordered_windows = dict(sorted(windows.items(), key=lambda item: item[0].value))
        object.__setattr__(
            self, "data_partition_hashes", MappingProxyType(ordered_partitions)
        )
        object.__setattr__(
            self, "data_partition_roles", MappingProxyType(ordered_roles)
        )
        object.__setattr__(
            self, "data_partition_windows", MappingProxyType(ordered_windows)
        )

        for name in (
            "data_snapshot_hash",
            "data_request_hash",
            "data_quality_spec_hash",
            "label_spec_hash",
            "validation_spec_hash",
            "evaluation_spec_hash",
            "report_spec_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"research {name} must be text")
            require_sha256(value, name=f"research {name}")
        if (
            not isinstance(self.factor_spec_hashes, tuple)
            or not self.factor_spec_hashes
            or len(set(self.factor_spec_hashes)) != len(self.factor_spec_hashes)
        ):
            raise ValueError("research factor hashes must be non-empty and unique")
        for offset, digest in enumerate(self.factor_spec_hashes):
            if not isinstance(digest, str):
                raise TypeError("research factor spec hashes must be text")
            require_sha256(digest, name=f"research factor_spec_hash:{offset}")
        for name in (
            "model_spec_hash",
            "score_spec_hash",
            "portfolio_spec_hash",
            "risk_spec_hash",
            "cost_model_hash",
            "execution_spec_hash",
            "backtest_spec_hash",
            "robustness_spec_hash",
            "admission_spec_hash",
        ):
            value = getattr(self, name)
            if value is not None:
                require_sha256(value, name=f"research {name}")
        self._validate_optional_chain()

    def _validate_optional_chain(self) -> None:
        if self.score_spec_hash is None and self.portfolio_spec_hash is not None:
            raise ValueError("portfolio specification requires a score specification")
        if self.portfolio_spec_hash is None and self.risk_spec_hash is not None:
            raise ValueError("risk specification requires a portfolio specification")

        execution_group = (
            self.cost_model_hash,
            self.execution_spec_hash,
            self.backtest_spec_hash,
        )
        if any(item is not None for item in execution_group) and not all(
            item is not None for item in execution_group
        ):
            raise ValueError(
                "cost, execution and backtest specification hashes are all-or-none"
            )
        if self.backtest_spec_hash is not None and self.portfolio_spec_hash is None:
            raise ValueError(
                "backtest specification requires a portfolio specification"
            )
        if self.robustness_spec_hash is not None and self.backtest_spec_hash is None:
            raise ValueError(
                "robustness specification requires a backtest specification"
            )

        profile = self.profile
        if not isinstance(profile, ExperimentProfile):  # pragma: no cover
            raise RuntimeError("research profile was not normalized")
        if profile is ExperimentProfile.PRODUCTION_CANDIDATE:
            if not (
                {DataPartition.TEST, DataPartition.HOLDOUT}
                & set(self.data_partition_hashes)
            ):
                raise ValueError(
                    "production research requires a protected test or holdout partition"
                )
            required = {
                "score_spec_hash": self.score_spec_hash,
                "portfolio_spec_hash": self.portfolio_spec_hash,
                "risk_spec_hash": self.risk_spec_hash,
                "cost_model_hash": self.cost_model_hash,
                "execution_spec_hash": self.execution_spec_hash,
                "backtest_spec_hash": self.backtest_spec_hash,
                "robustness_spec_hash": self.robustness_spec_hash,
                "admission_spec_hash": self.admission_spec_hash,
            }
            missing = sorted(name for name, value in required.items() if value is None)
            if missing:
                raise ValueError(
                    "production research lacks required contracts:" + ",".join(missing)
                )

    @property
    def enabled_stages(self) -> tuple[ResearchStage, ...]:
        stages = [
            ResearchStage.DATA_QUALITY,
            ResearchStage.FACTOR_GENERATION,
            ResearchStage.LABEL_BUILDING,
            ResearchStage.VALIDATION_SPLIT,
        ]
        if self.model_spec_hash is not None:
            stages.append(ResearchStage.MODEL_TRAINING)
        stages.append(ResearchStage.FACTOR_EVALUATION)
        if self.score_spec_hash is not None:
            stages.append(ResearchStage.SCORE_CONSTRUCTION)
        if self.portfolio_spec_hash is not None:
            stages.append(ResearchStage.PORTFOLIO_CONSTRUCTION)
        if self.backtest_spec_hash is not None:
            stages.append(ResearchStage.BACKTEST)
        if self.robustness_spec_hash is not None:
            stages.append(ResearchStage.ROBUSTNESS)
        stages.append(ResearchStage.REPORT)
        if self.admission_spec_hash is not None:
            stages.append(ResearchStage.ADMISSION)
        return tuple(stages)

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def component_bindings(self) -> Mapping[str, str]:
        bindings: dict[str, str] = {
            f"data:partition:{DataPartition(partition).value}": digest
            for partition, digest in self.data_partition_hashes.items()
        }
        bindings.update(
            {
                "data:snapshot": self.data_snapshot_hash,
                "data:request": self.data_request_hash,
                "data:quality": self.data_quality_spec_hash,
                "label": self.label_spec_hash,
                "validation": self.validation_spec_hash,
                "evaluation": self.evaluation_spec_hash,
                "report": self.report_spec_hash,
            }
        )
        bindings.update(
            {
                f"factor:{offset}": digest
                for offset, digest in enumerate(self.factor_spec_hashes)
            }
        )
        optional = (
            ("model", self.model_spec_hash),
            ("score", self.score_spec_hash),
            ("portfolio", self.portfolio_spec_hash),
            ("risk", self.risk_spec_hash),
            ("cost", self.cost_model_hash),
            ("execution", self.execution_spec_hash),
            ("backtest", self.backtest_spec_hash),
            ("robustness", self.robustness_spec_hash),
            ("admission", self.admission_spec_hash),
        )
        bindings.update(
            (role, digest) for role, digest in optional if digest is not None
        )
        return MappingProxyType(dict(sorted(bindings.items())))

    def to_dict(self) -> dict[str, object]:
        profile = self.profile
        if not isinstance(profile, ExperimentProfile):  # pragma: no cover
            raise RuntimeError("research profile was not normalized")
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "version": self.version,
            "profile": profile.value,
            "data_partition_hashes": {
                DataPartition(partition).value: digest
                for partition, digest in self.data_partition_hashes.items()
            },
            "data_partition_roles": {
                DataPartition(partition).value: DataRole(role).value
                for partition, role in self.data_partition_roles.items()
            },
            "data_partition_windows": {
                DataPartition(partition).value: list(window)
                for partition, window in self.data_partition_windows.items()
            },
            "data_snapshot_hash": self.data_snapshot_hash,
            "data_request_hash": self.data_request_hash,
            "data_quality_spec_hash": self.data_quality_spec_hash,
            "factor_spec_hashes": list(self.factor_spec_hashes),
            "label_spec_hash": self.label_spec_hash,
            "validation_spec_hash": self.validation_spec_hash,
            "evaluation_spec_hash": self.evaluation_spec_hash,
            "report_spec_hash": self.report_spec_hash,
            "model_spec_hash": self.model_spec_hash,
            "score_spec_hash": self.score_spec_hash,
            "portfolio_spec_hash": self.portfolio_spec_hash,
            "risk_spec_hash": self.risk_spec_hash,
            "cost_model_hash": self.cost_model_hash,
            "execution_spec_hash": self.execution_spec_hash,
            "backtest_spec_hash": self.backtest_spec_hash,
            "robustness_spec_hash": self.robustness_spec_hash,
            "admission_spec_hash": self.admission_spec_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ResearchRunSpec":
        expected = {
            "schema_version",
            "run_id",
            "version",
            "profile",
            "data_partition_hashes",
            "data_partition_roles",
            "data_partition_windows",
            "data_snapshot_hash",
            "data_request_hash",
            "data_quality_spec_hash",
            "factor_spec_hashes",
            "label_spec_hash",
            "validation_spec_hash",
            "evaluation_spec_hash",
            "report_spec_hash",
            "model_spec_hash",
            "score_spec_hash",
            "portfolio_spec_hash",
            "risk_spec_hash",
            "cost_model_hash",
            "execution_spec_hash",
            "backtest_spec_hash",
            "robustness_spec_hash",
            "admission_spec_hash",
        }
        if set(value) != expected:
            raise ValueError("ResearchRunSpec wire fields differ")
        partition_hashes = _string_mapping(
            value["data_partition_hashes"], name="data_partition_hashes"
        )
        partition_roles = _string_mapping(
            value["data_partition_roles"], name="data_partition_roles"
        )
        partition_windows = _window_mapping(value["data_partition_windows"])
        factors = value["factor_spec_hashes"]
        if not isinstance(factors, list) or not all(
            isinstance(item, str) for item in factors
        ):
            raise TypeError("ResearchRunSpec factor_spec_hashes must be strings")
        return cls(
            schema_version=_string(value["schema_version"], name="schema_version"),
            run_id=_string(value["run_id"], name="run_id"),
            version=_string(value["version"], name="version"),
            profile=_string(value["profile"], name="profile"),
            data_partition_hashes=partition_hashes,
            data_partition_roles=partition_roles,
            data_partition_windows=partition_windows,
            data_snapshot_hash=_string(
                value["data_snapshot_hash"], name="data_snapshot_hash"
            ),
            data_request_hash=_string(
                value["data_request_hash"], name="data_request_hash"
            ),
            data_quality_spec_hash=_string(
                value["data_quality_spec_hash"], name="data_quality_spec_hash"
            ),
            factor_spec_hashes=tuple(factors),
            label_spec_hash=_string(value["label_spec_hash"], name="label_spec_hash"),
            validation_spec_hash=_string(
                value["validation_spec_hash"], name="validation_spec_hash"
            ),
            evaluation_spec_hash=_string(
                value["evaluation_spec_hash"], name="evaluation_spec_hash"
            ),
            report_spec_hash=_string(
                value["report_spec_hash"], name="report_spec_hash"
            ),
            model_spec_hash=_optional_string(value["model_spec_hash"]),
            score_spec_hash=_optional_string(value["score_spec_hash"]),
            portfolio_spec_hash=_optional_string(value["portfolio_spec_hash"]),
            risk_spec_hash=_optional_string(value["risk_spec_hash"]),
            cost_model_hash=_optional_string(value["cost_model_hash"]),
            execution_spec_hash=_optional_string(value["execution_spec_hash"]),
            backtest_spec_hash=_optional_string(value["backtest_spec_hash"]),
            robustness_spec_hash=_optional_string(value["robustness_spec_hash"]),
            admission_spec_hash=_optional_string(value["admission_spec_hash"]),
        )


def _safe_code(value: str) -> bool:
    return bool(
        isinstance(value, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value)
    )


def _string_mapping(value: object, *, name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise TypeError(f"ResearchRunSpec {name} must be a string object")
    return dict(value)


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"ResearchRunSpec {name} must be text")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("ResearchRunSpec optional hashes must be strings or null")
    return value


def _window_mapping(value: object) -> Mapping[str, tuple[str, str]]:
    if not isinstance(value, Mapping):
        raise TypeError("ResearchRunSpec data_partition_windows must be an object")
    result: dict[str, tuple[str, str]] = {}
    for key, window in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(window, list)
            or len(window) != 2
            or not all(isinstance(item, str) for item in window)
        ):
            raise TypeError(
                "ResearchRunSpec partition windows must be string [start,end] pairs"
            )
        result[key] = (window[0], window[1])
    return result


__all__ = [
    "RESEARCH_STAGE_ORDER",
    "ResearchRunSpec",
    "ResearchStage",
    "canonical_data_role",
]
