from __future__ import annotations

import math
from dataclasses import dataclass

from factor_production.v5.domain.enums import ScoreVisibility, StopReason
from factor_production.v5.orchestration.budget import BudgetController


class HoldoutLeakageError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ObjectiveObservation:
    generation: int
    value: float
    metric: str
    visibility: ScoreVisibility = ScoreVisibility.LOCAL_RESEARCH

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("generation cannot be negative")
        if not math.isfinite(self.value):
            raise ValueError("objective value must be finite")
        if self.visibility is not ScoreVisibility.LOCAL_RESEARCH:
            raise HoldoutLeakageError(
                "external holdout scores cannot drive search stopping or iteration"
            )


@dataclass(frozen=True, slots=True)
class StoppingConfig:
    patience_generations: int = 2
    min_improvement: float = 0.0

    def __post_init__(self) -> None:
        if self.patience_generations <= 0:
            raise ValueError("patience_generations must be positive")
        if self.min_improvement < 0 or not math.isfinite(self.min_improvement):
            raise ValueError("min_improvement must be a finite non-negative number")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "patience_generations": self.patience_generations,
            "min_improvement": self.min_improvement,
        }


@dataclass(frozen=True, slots=True)
class StoppingDecision:
    should_stop: bool
    reason: StopReason
    detail: str


class StopController:
    def __init__(self, config: StoppingConfig) -> None:
        self.config = config
        self._best: float | None = None
        self._stalled = 0
        self._last_generation = -1

    def observe(
        self,
        observation: ObjectiveObservation,
        budget: BudgetController,
    ) -> StoppingDecision:
        if observation.generation <= self._last_generation:
            raise ValueError("objective observations must have strictly increasing generations")
        self._last_generation = observation.generation
        budget_reason = budget.first_exhausted_reason()
        if budget_reason is not StopReason.CONTINUE:
            return StoppingDecision(True, budget_reason, "a pre-registered hard budget was exhausted")

        if self._best is None or observation.value > self._best + self.config.min_improvement:
            self._best = observation.value
            self._stalled = 0
            return StoppingDecision(False, StopReason.CONTINUE, "local objective improved")
        self._stalled += 1
        if self._stalled >= self.config.patience_generations:
            return StoppingDecision(
                True,
                StopReason.PATIENCE_EXHAUSTED,
                f"no local improvement for {self._stalled} generations",
            )
        return StoppingDecision(False, StopReason.CONTINUE, "within pre-registered patience")

