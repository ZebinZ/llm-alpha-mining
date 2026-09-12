from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from llm_alpha_mining.mining.artifacts.hashing import hash_json
from llm_alpha_mining.mining.domain.enums import ScoreVisibility
from llm_alpha_mining.mining.domain.models import CandidateSpec
from llm_alpha_mining.mining.scorers.base import (
    CandidateScorer,
    ScoreBatch,
    ScoreObservation,
)


class MockScorer(CandidateScorer):
    """Deterministic local scorer; never represents an official holdout."""

    visibility = ScoreVisibility.LOCAL_RESEARCH

    def __init__(
        self,
        values: Mapping[str, Mapping[str, float]]
        | Callable[[CandidateSpec], Mapping[str, float]],
        *,
        name: str = "mock_local",
    ) -> None:
        self.values = values
        self.name = name

    def score(self, candidates: Sequence[CandidateSpec]) -> ScoreBatch:
        observations: list[ScoreObservation] = []
        for candidate in candidates:
            if callable(self.values):
                metrics = self.values(candidate)
            else:
                try:
                    metrics = self.values[candidate.candidate_id]
                except KeyError as exc:
                    raise KeyError(
                        f"mock score missing for {candidate.candidate_id}"
                    ) from exc
            observations.append(
                ScoreObservation(
                    candidate_id=candidate.candidate_id,
                    generation=candidate.generation,
                    metrics=metrics,
                    scorer=self.name,
                    visibility=self.visibility,
                )
            )
        batch_hash = hash_json(
            {
                "scorer": self.name,
                "candidates": [candidate.content_hash for candidate in candidates],
            }
        )
        return ScoreBatch(
            batch_id=f"local-{batch_hash[:16]}",
            scorer=self.name,
            visibility=self.visibility,
            observations=tuple(observations),
        )
