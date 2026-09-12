from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from factor_production.v5.domain.enums import ScoreVisibility
from factor_production.v5.domain.models import CandidateSpecV5
from factor_production.v5.providers.base import LocalFeedback


class HoldoutIsolationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ScoreObservation:
    candidate_id: str
    generation: int
    metrics: Mapping[str, float]
    scorer: str
    visibility: ScoreVisibility

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.scorer.strip():
            raise ValueError("candidate_id and scorer must not be empty")
        if self.generation < 0:
            raise ValueError("generation cannot be negative")
        normalized: dict[str, float] = {}
        for key, value in self.metrics.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("metric names must be non-empty strings")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"metric {key!r} must be finite")
            normalized[key] = number
        if not normalized:
            raise ValueError("score observation must contain at least one metric")
        object.__setattr__(self, "metrics", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class ScoreBatch:
    batch_id: str
    scorer: str
    visibility: ScoreVisibility
    observations: tuple[ScoreObservation, ...]

    def __post_init__(self) -> None:
        if not self.batch_id.strip() or not self.scorer.strip():
            raise ValueError("batch_id and scorer must not be empty")
        ids = [observation.candidate_id for observation in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("score batch contains duplicate candidate IDs")
        for observation in self.observations:
            if observation.scorer != self.scorer:
                raise ValueError("observation scorer differs from batch scorer")
            if observation.visibility is not self.visibility:
                raise ValueError("observation visibility differs from batch visibility")

    def to_local_feedback(self, metric: str) -> tuple[LocalFeedback, ...]:
        if self.visibility is not ScoreVisibility.LOCAL_RESEARCH:
            raise HoldoutIsolationError(
                "external teacher/official scores cannot be converted into provider feedback"
            )
        feedback: list[LocalFeedback] = []
        for observation in self.observations:
            if metric not in observation.metrics:
                raise KeyError(f"metric {metric!r} is missing for {observation.candidate_id}")
            feedback.append(
                LocalFeedback(
                    candidate_id=observation.candidate_id,
                    metric=metric,
                    value=observation.metrics[metric],
                    generation=observation.generation,
                )
            )
        return tuple(feedback)


class CandidateScorer(ABC):
    name: str
    visibility: ScoreVisibility

    @abstractmethod
    def score(self, candidates: Sequence[CandidateSpecV5]) -> ScoreBatch:
        raise NotImplementedError

