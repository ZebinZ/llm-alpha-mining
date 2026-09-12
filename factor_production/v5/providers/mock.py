from __future__ import annotations

from collections.abc import Iterable

from factor_production.v5.domain.models import CandidateSpecV5
from factor_production.v5.providers.base import CandidateProvider, ProposalBatch, ProposalContext


class MockProvider(CandidateProvider):
    """Deterministic in-memory provider for orchestration and contract tests."""

    def __init__(self, candidates: Iterable[CandidateSpecV5], *, name: str = "mock") -> None:
        self.name = name
        self._candidates = tuple(candidates)
        self._emitted: set[str] = set()

    def propose(self, context: ProposalContext) -> ProposalBatch:
        candidates: list[CandidateSpecV5] = []
        for candidate in self._candidates:
            if candidate.candidate_id in self._emitted:
                continue
            if candidate.protocol_hash != context.protocol_hash:
                continue
            if candidate.generation != context.generation:
                continue
            candidates.append(candidate)
            if len(candidates) >= context.request_count:
                break
        self._emitted.update(candidate.candidate_id for candidate in candidates)
        remaining = any(
            candidate.candidate_id not in self._emitted
            and candidate.protocol_hash == context.protocol_hash
            and candidate.generation == context.generation
            for candidate in self._candidates
        )
        return ProposalBatch(self.name, tuple(candidates), exhausted=not remaining)

