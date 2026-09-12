from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import math
import pandas as pd

from llm_alpha_mining.research.core.hashing import hash_json, require_sha256


class PortfolioConstruction(str, Enum):
    TOP_QUANTILE_EQUAL = "top_quantile_equal"
    CROSS_SECTIONAL_RANK = "cross_sectional_rank"


class ExposureMode(str, Enum):
    LONG_ONLY = "long_only"
    MARKET_NEUTRAL = "market_neutral"


@dataclass(frozen=True, slots=True)
class PortfolioSpec:
    portfolio_id: str
    version: str
    factor_definition_hash: str
    score_hash: str
    universe_hash: str
    construction: PortfolioConstruction | str
    exposure_mode: ExposureMode | str
    selection_fraction: float
    gross_leverage: float
    target_net_exposure: float
    maximum_absolute_weight: float
    maximum_one_way_turnover: float
    minimum_names_per_side: int
    rebalance_policy: str = "each_signal"
    turnover_policy: str = "hard_cap_with_exposure_ramp"
    schema_version: str = "portfolio-spec/v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "construction", PortfolioConstruction(self.construction)
        )
        object.__setattr__(self, "exposure_mode", ExposureMode(self.exposure_mode))
        if self.schema_version != "portfolio-spec/v1":
            raise ValueError("unsupported PortfolioSpec schema")
        if not self.portfolio_id.strip() or not self.version.strip():
            raise ValueError("portfolio id and version are required")
        for name in ("factor_definition_hash", "score_hash", "universe_hash"):
            require_sha256(str(getattr(self, name)), name=f"portfolio {name}")
        if not 0 < self.selection_fraction <= 0.5:
            raise ValueError("portfolio selection_fraction must lie in (0,0.5]")
        for name in (
            "gross_leverage",
            "maximum_absolute_weight",
            "maximum_one_way_turnover",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"portfolio {name} must be positive")
        if abs(self.target_net_exposure) > self.gross_leverage:
            raise ValueError("portfolio net exposure exceeds gross leverage")
        if (
            not isinstance(self.minimum_names_per_side, int)
            or self.minimum_names_per_side <= 0
        ):
            raise ValueError("portfolio minimum_names_per_side must be positive")
        if self.rebalance_policy != "each_signal":
            raise ValueError("unsupported portfolio rebalance policy")
        if self.turnover_policy != "hard_cap_with_exposure_ramp":
            raise ValueError("unsupported portfolio turnover policy")
        mode = self.exposure_mode
        if mode is ExposureMode.LONG_ONLY:
            if (
                self.target_net_exposure < 0
                or self.target_net_exposure != self.gross_leverage
            ):
                raise ValueError(
                    "long-only portfolio net and gross exposure must match"
                )
        elif self.target_net_exposure != 0:
            raise ValueError("market-neutral portfolio requires zero net exposure")

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        construction = self.construction
        mode = self.exposure_mode
        if not isinstance(construction, PortfolioConstruction) or not isinstance(
            mode, ExposureMode
        ):  # pragma: no cover
            raise RuntimeError("portfolio enums were not normalized")
        return {
            "schema_version": self.schema_version,
            "portfolio_id": self.portfolio_id,
            "version": self.version,
            "factor_definition_hash": self.factor_definition_hash,
            "score_hash": self.score_hash,
            "universe_hash": self.universe_hash,
            "construction": construction.value,
            "exposure_mode": mode.value,
            "selection_fraction": self.selection_fraction,
            "gross_leverage": self.gross_leverage,
            "target_net_exposure": self.target_net_exposure,
            "maximum_absolute_weight": self.maximum_absolute_weight,
            "maximum_one_way_turnover": self.maximum_one_way_turnover,
            "minimum_names_per_side": self.minimum_names_per_side,
            "rebalance_policy": self.rebalance_policy,
            "turnover_policy": self.turnover_policy,
        }


@dataclass(frozen=True, slots=True)
class RiskSpec:
    risk_model_id: str
    version: str
    known_at: str
    exposure_hashes: Mapping[str, str]
    maximum_absolute_exposure: Mapping[str, float]
    point_in_time: bool
    availability_hashes: Mapping[str, str] = field(default_factory=dict)
    constraint_policy: str = "enforce"
    schema_version: str = "risk-spec/v1"

    def __post_init__(self) -> None:
        exposures = dict(sorted(self.exposure_hashes.items()))
        availability = dict(sorted(self.availability_hashes.items()))
        limits = dict(sorted(self.maximum_absolute_exposure.items()))
        object.__setattr__(self, "exposure_hashes", MappingProxyType(exposures))
        object.__setattr__(self, "availability_hashes", MappingProxyType(availability))
        object.__setattr__(self, "maximum_absolute_exposure", MappingProxyType(limits))
        if self.schema_version != "risk-spec/v1":
            raise ValueError("unsupported RiskSpec schema")
        if not self.risk_model_id.strip() or not self.version.strip() or not exposures:
            raise ValueError("risk model id, version and exposures are required")
        timestamp = pd.Timestamp(self.known_at)
        if timestamp.tzinfo is None:
            raise ValueError("risk model known_at must be timezone-aware")
        object.__setattr__(self, "known_at", timestamp.isoformat())
        if set(exposures) != set(limits):
            raise ValueError(
                "risk exposure hashes and limits must have identical names"
            )
        for name, digest in exposures.items():
            require_sha256(digest, name=f"risk exposure hash:{name}")
        for name, digest in availability.items():
            require_sha256(digest, name=f"risk availability hash:{name}")
        for name, limit in limits.items():
            if not math.isfinite(limit) or limit < 0:
                raise ValueError(f"risk exposure limit is invalid:{name}")
        if not isinstance(self.point_in_time, bool):
            raise TypeError("risk point_in_time must be boolean")
        if self.constraint_policy not in {"enforce", "diagnostic_only"}:
            raise ValueError("unsupported risk constraint policy")
        if not self.point_in_time and self.constraint_policy == "enforce":
            raise ValueError("non-PIT risk data may only be diagnostic")
        if availability and set(availability) != set(exposures):
            raise ValueError(
                "risk exposure and per-observation availability names differ"
            )
        if self.constraint_policy == "enforce" and not availability:
            raise ValueError(
                "enforced risk data requires per-observation availability hashes"
            )

    @property
    def production_ready(self) -> bool:
        return (
            self.point_in_time
            and self.constraint_policy == "enforce"
            and set(self.availability_hashes) == set(self.exposure_hashes)
        )

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "risk_model_id": self.risk_model_id,
            "version": self.version,
            "known_at": self.known_at,
            "exposure_hashes": dict(self.exposure_hashes),
            "availability_hashes": dict(self.availability_hashes),
            "maximum_absolute_exposure": dict(self.maximum_absolute_exposure),
            "point_in_time": self.point_in_time,
            "constraint_policy": self.constraint_policy,
        }


__all__ = [
    "ExposureMode",
    "PortfolioConstruction",
    "PortfolioSpec",
    "RiskSpec",
]
