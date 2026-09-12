from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd

from llm_alpha_mining.research.core.hashing import hash_json


@dataclass(frozen=True, slots=True)
class RegimePeriod:
    name: str
    start: str
    end: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("regime name is required")
        start = pd.Timestamp(self.start)
        end = pd.Timestamp(self.end)
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("regime boundaries must be timezone-aware")
        if start > end:
            raise ValueError("regime range is inverted")
        object.__setattr__(self, "start", start.isoformat())
        object.__setattr__(self, "end", end.isoformat())

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "start": self.start, "end": self.end}


@dataclass(frozen=True, slots=True)
class RobustnessSpec:
    robustness_id: str
    version: str
    capital_levels: tuple[float, ...]
    execution_delay_sessions: tuple[int, ...]
    regimes: tuple[RegimePeriod, ...]
    pseudo_factor_trials: int
    random_seed: int
    selection_fraction_multipliers: tuple[float, ...] = (0.8, 1.0, 1.2)
    multiple_testing_alpha: float = 0.05
    schema_version: str = "robustness-spec/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "robustness-spec/v1":
            raise ValueError("unsupported RobustnessSpec schema")
        if not self.robustness_id.strip() or not self.version.strip():
            raise ValueError("robustness id and version are required")
        if not self.capital_levels or any(value <= 0 for value in self.capital_levels):
            raise ValueError("robustness capital levels must be positive")
        if tuple(sorted(set(self.capital_levels))) != self.capital_levels:
            raise ValueError("robustness capital levels must be sorted and unique")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in self.execution_delay_sessions
        ):
            raise ValueError("execution delays must be positive sessions")
        if (
            tuple(sorted(set(self.execution_delay_sessions)))
            != self.execution_delay_sessions
        ):
            raise ValueError("execution delays must be sorted and unique")
        if len({period.name for period in self.regimes}) != len(self.regimes):
            raise ValueError("regime names must be unique")
        if (
            not isinstance(self.pseudo_factor_trials, int)
            or self.pseudo_factor_trials < 20
        ):
            raise ValueError("pseudo-factor control requires at least 20 trials")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise TypeError("robustness random_seed must be an integer")
        multipliers = self.selection_fraction_multipliers
        if not multipliers or any(value <= 0 for value in multipliers):
            raise ValueError("selection-fraction multipliers must be positive")
        if tuple(sorted(set(multipliers))) != multipliers:
            raise ValueError("selection-fraction multipliers must be sorted and unique")
        if 1.0 not in multipliers:
            raise ValueError("selection-fraction multipliers must include the base 1.0")
        if not 0 < self.multiple_testing_alpha < 1:
            raise ValueError("multiple-testing alpha must lie in (0,1)")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "robustness_id": self.robustness_id,
            "version": self.version,
            "capital_levels": list(self.capital_levels),
            "execution_delay_sessions": list(self.execution_delay_sessions),
            "regimes": [period.to_dict() for period in self.regimes],
            "pseudo_factor_trials": self.pseudo_factor_trials,
            "random_seed": self.random_seed,
            "selection_fraction_multipliers": list(self.selection_fraction_multipliers),
            "multiple_testing_alpha": self.multiple_testing_alpha,
        }


__all__ = ["RegimePeriod", "RobustnessSpec"]
