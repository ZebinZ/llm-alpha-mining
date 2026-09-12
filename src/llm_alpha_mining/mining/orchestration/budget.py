from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from llm_alpha_mining.mining.domain.enums import StopReason


class BudgetExceeded(RuntimeError):
    def __init__(self, reason: StopReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    max_candidates: int
    max_generations: int
    max_provider_calls: int
    max_evaluations: int
    max_wall_seconds: int

    def __post_init__(self) -> None:
        for name in (
            "max_candidates",
            "max_generations",
            "max_provider_calls",
            "max_evaluations",
            "max_wall_seconds",
        ):
            _positive_int(getattr(self, name), name)

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BudgetLimits":
        expected = {
            "max_candidates",
            "max_generations",
            "max_provider_calls",
            "max_evaluations",
            "max_wall_seconds",
        }
        if set(value) != expected:
            raise ValueError(
                f"budget fields must be exactly {sorted(expected)}; got {sorted(value)}"
            )
        return cls(**{name: value[name] for name in expected})


@dataclass(frozen=True, slots=True)
class BudgetUsage:
    candidates: int = 0
    generations_started: int = 0
    provider_calls: int = 0
    evaluations: int = 0
    wall_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "candidates",
            "generations_started",
            "provider_calls",
            "evaluations",
            "wall_seconds",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")

    def to_dict(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BudgetUsage":
        expected = {
            "candidates",
            "generations_started",
            "provider_calls",
            "evaluations",
            "wall_seconds",
        }
        if set(value) != expected:
            raise ValueError(
                f"budget usage fields must be exactly {sorted(expected)}; got {sorted(value)}"
            )
        return cls(**{name: value[name] for name in expected})


class BudgetController:
    """Fail-closed reservation API so work cannot start past its budget."""

    def __init__(self, limits: BudgetLimits, usage: BudgetUsage | None = None) -> None:
        self.limits = limits
        self.usage = usage or BudgetUsage()
        self._assert_within_limits(self.usage)

    def _assert_within_limits(self, usage: BudgetUsage) -> None:
        checks = (
            (usage.candidates, self.limits.max_candidates, StopReason.CANDIDATE_BUDGET),
            (
                usage.generations_started,
                self.limits.max_generations,
                StopReason.GENERATION_BUDGET,
            ),
            (
                usage.provider_calls,
                self.limits.max_provider_calls,
                StopReason.PROVIDER_CALL_BUDGET,
            ),
            (
                usage.evaluations,
                self.limits.max_evaluations,
                StopReason.EVALUATION_BUDGET,
            ),
            (
                usage.wall_seconds,
                self.limits.max_wall_seconds,
                StopReason.WALL_TIME_BUDGET,
            ),
        )
        for actual, limit, reason in checks:
            if actual > limit:
                raise BudgetExceeded(
                    reason, f"{reason.value} exceeded: {actual}>{limit}"
                )

    def reserve(
        self,
        *,
        candidates: int = 0,
        generations: int = 0,
        provider_calls: int = 0,
        evaluations: int = 0,
        wall_seconds: float = 0.0,
    ) -> BudgetUsage:
        increments = (
            candidates,
            generations,
            provider_calls,
            evaluations,
            wall_seconds,
        )
        if any(value < 0 for value in increments):
            raise ValueError("budget reservations cannot be negative")
        proposed = BudgetUsage(
            candidates=self.usage.candidates + candidates,
            generations_started=self.usage.generations_started + generations,
            provider_calls=self.usage.provider_calls + provider_calls,
            evaluations=self.usage.evaluations + evaluations,
            wall_seconds=self.usage.wall_seconds + wall_seconds,
        )
        self._assert_within_limits(proposed)
        self.usage = proposed
        return proposed

    def first_exhausted_reason(self) -> StopReason:
        usage = self.usage
        if usage.candidates >= self.limits.max_candidates:
            return StopReason.CANDIDATE_BUDGET
        if usage.generations_started >= self.limits.max_generations:
            return StopReason.GENERATION_BUDGET
        if usage.provider_calls >= self.limits.max_provider_calls:
            return StopReason.PROVIDER_CALL_BUDGET
        if usage.evaluations >= self.limits.max_evaluations:
            return StopReason.EVALUATION_BUDGET
        if usage.wall_seconds >= self.limits.max_wall_seconds:
            return StopReason.WALL_TIME_BUDGET
        return StopReason.CONTINUE
