from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from factor_production.v5.artifacts.hashing import hash_json
from factor_production.v5.domain.enums import CandidateState, RunState, StopReason
from factor_production.v5.domain.models import CandidateSpecV5
from factor_production.v5.orchestration.budget import BudgetController, BudgetExceeded
from factor_production.v5.orchestration.hypotheses import (
    HypothesisEntry,
    HypothesisRegistry,
    HypothesisRegistryError,
)
from factor_production.v5.orchestration.repository import SQLiteRepository
from factor_production.v5.orchestration.stopping import (
    ObjectiveObservation,
    StopController,
    StoppingDecision,
)
from factor_production.v5.providers.base import (
    CandidateProvider,
    ProposalContext,
    SanitizedFeedback,
)


CAMPAIGN_ENGINE_SCHEMA = "campaign-engine/v1"


class CampaignEngineError(RuntimeError):
    pass


class CampaignStopped(CampaignEngineError):
    pass


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    proposals_per_round: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.proposals_per_round, int)
            or isinstance(self.proposals_per_round, bool)
            or self.proposals_per_round <= 0
        ):
            raise ValueError("proposals_per_round must be a positive integer")

    def to_dict(self) -> dict[str, int]:
        return {"proposals_per_round": self.proposals_per_round}


@dataclass(frozen=True, slots=True)
class RejectedProposal:
    candidate_id: str
    reason_code: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "reason_code": self.reason_code,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class CampaignRoundResult:
    campaign_round: int
    accepted: tuple[CandidateSpecV5, ...]
    rejected: tuple[RejectedProposal, ...]
    provider: str
    provider_exhausted: bool
    context_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "campaign_round": self.campaign_round,
            "accepted": [candidate.content_hash for candidate in self.accepted],
            "rejected": [item.to_dict() for item in self.rejected],
            "provider": self.provider,
            "provider_exhausted": self.provider_exhausted,
            "context_hash": self.context_hash,
        }


class CampaignEngine:
    """P0 multi-round proposal controller with fail-closed persistence.

    Evaluation remains an injected downstream concern.  This controller owns
    only round sequencing, provider budgets, sanitized feedback, lineage,
    campaign-wide semantic de-duplication, candidate registration and stopping.
    """

    def __init__(
        self,
        *,
        run_id: str,
        protocol_hash: str,
        provider: CandidateProvider,
        repository: SQLiteRepository,
        budget: BudgetController,
        stopping: StopController,
        config: CampaignConfig,
        registry: HypothesisRegistry | None = None,
    ) -> None:
        self.run_id = run_id
        self.protocol_hash = protocol_hash
        self.provider = provider
        self.repository = repository
        self.budget = budget
        self.stopping = stopping
        self.config = config
        self.registry = registry if registry is not None else HypothesisRegistry(protocol_hash)
        self._stop_decision: StoppingDecision | None = None
        self._failed = False
        self._observed_rounds: set[int] = set()

        run = repository.get_run(run_id)
        if run.protocol_hash != protocol_hash:
            raise CampaignEngineError("repository run protocol does not match campaign protocol")
        if run.state not in {RunState.INITIALIZED, RunState.RUNNING}:
            raise CampaignEngineError(
                f"campaign cannot start while repository run is {run.state.value}"
            )
        self._synchronize_registry()
        self._next_round = self.registry.next_campaign_round
        self._last_round: int | None = None

    def _synchronize_registry(self) -> None:
        stored = self.repository.list_candidates(self.run_id)
        if not stored:
            if len(self.registry):
                raise CampaignEngineError("non-empty registry has no matching repository candidates")
            return
        if not len(self.registry):
            self.registry.register_many(record.spec for record in stored)
            return
        stored_by_id = {record.candidate_id: record.spec_hash for record in stored}
        registry_by_id = {entry.candidate_id: entry.spec_hash for entry in self.registry.entries}
        if stored_by_id != registry_by_id:
            raise CampaignEngineError("registry and repository candidate ledgers differ")

    @property
    def next_campaign_round(self) -> int:
        return self._next_round

    @property
    def stop_decision(self) -> StoppingDecision | None:
        return self._stop_decision

    @property
    def should_stop(self) -> bool:
        return self._stop_decision is not None and self._stop_decision.should_stop

    def _assert_active(self) -> None:
        if self._failed:
            raise CampaignEngineError("campaign engine is failed closed after a persistence error")
        if self.should_stop:
            raise CampaignStopped(self._stop_decision.detail)

    def validate_feedback(
        self,
        campaign_round: int,
        feedback: Iterable[SanitizedFeedback],
    ) -> tuple[SanitizedFeedback, ...]:
        items = tuple(feedback)
        if any(not isinstance(item, SanitizedFeedback) for item in items):
            raise CampaignEngineError("campaign accepts SanitizedFeedback objects only")
        ids = [item.candidate_id for item in items]
        if len(ids) != len(set(ids)):
            raise CampaignEngineError("feedback contains duplicate candidate IDs")
        for item in items:
            if item.campaign_round >= campaign_round:
                raise CampaignEngineError("feedback must come from a prior campaign round")
            try:
                entry = self.registry.get(item.candidate_id)
            except KeyError as exc:
                raise CampaignEngineError(
                    f"feedback references an unknown candidate: {item.candidate_id}"
                ) from exc
            if item.candidate_hash != entry.spec_hash:
                raise CampaignEngineError(
                    f"feedback hash does not bind to candidate: {item.candidate_id}"
                )
        return items

    def propose_round(
        self,
        campaign_round: int,
        *,
        feedback: Iterable[SanitizedFeedback] = (),
    ) -> CampaignRoundResult:
        self._assert_active()
        if campaign_round != self._next_round:
            raise CampaignEngineError(
                f"campaign rounds must be sequential: expected {self._next_round}, got {campaign_round}"
            )
        sanitized = self.validate_feedback(campaign_round, feedback)
        try:
            self.budget.reserve(generations=1, provider_calls=1)
        except BudgetExceeded as exc:
            self._stop_decision = StoppingDecision(True, exc.reason, str(exc))
            raise CampaignStopped(str(exc)) from exc

        context_payload = {
            "run_id": self.run_id,
            "protocol_hash": self.protocol_hash,
            "campaign_round": campaign_round,
            "request_count": self.config.proposals_per_round,
            "prior_candidate_hashes": list(self.registry.candidate_hashes),
            "sanitized_feedback_hashes": [item.content_hash for item in sanitized],
        }
        context_hash = hash_json(context_payload)
        context = ProposalContext(
            run_id=self.run_id,
            protocol_hash=self.protocol_hash,
            generation=campaign_round,
            request_count=self.config.proposals_per_round,
            prior_candidate_hashes=self.registry.candidate_hashes,
            sanitized_feedback=sanitized,
        )
        try:
            batch = self.provider.propose(context)
        except BaseException:
            self._failed = True
            raise
        if batch.provider != self.provider.name:
            self._failed = True
            raise CampaignEngineError("proposal batch provider does not match configured provider")
        if len(batch.candidates) > self.config.proposals_per_round:
            self._failed = True
            raise CampaignEngineError("provider returned more candidates than requested")

        accepted_with_entries: list[tuple[CandidateSpecV5, HypothesisEntry]] = []
        rejected: list[RejectedProposal] = []
        staged_semantic_hashes: set[str] = set()
        for candidate in batch.candidates:
            if candidate.protocol_hash != self.protocol_hash:
                rejected.append(
                    RejectedProposal(candidate.candidate_id, "protocol_mismatch", "candidate protocol")
                )
                continue
            if candidate.campaign_round != campaign_round:
                rejected.append(
                    RejectedProposal(
                        candidate.candidate_id,
                        "campaign_round_mismatch",
                        f"declared={candidate.campaign_round},expected={campaign_round}",
                    )
                )
                continue
            try:
                entry = self.registry.preflight(candidate)
            except HypothesisRegistryError as exc:
                rejected.append(RejectedProposal(candidate.candidate_id, exc.code, exc.detail))
                continue
            if entry.semantic_hash in staged_semantic_hashes:
                rejected.append(
                    RejectedProposal(
                        candidate.candidate_id,
                        "duplicate_semantic_signal_in_batch",
                        entry.semantic_hash,
                    )
                )
                continue
            staged_semantic_hashes.add(entry.semantic_hash)
            accepted_with_entries.append((candidate, entry))

        try:
            self.budget.reserve(candidates=len(accepted_with_entries))
        except BudgetExceeded as exc:
            self._stop_decision = StoppingDecision(True, exc.reason, str(exc))
            raise CampaignStopped(str(exc)) from exc

        try:
            for candidate, expected_entry in accepted_with_entries:
                self.repository.add_candidate(self.run_id, candidate)
                self.repository.transition_candidate(
                    self.run_id,
                    candidate.candidate_id,
                    CandidateState.PROPOSED,
                    reason="campaign_provider_proposed",
                )
                actual_entry = self.registry.register(candidate)
                if actual_entry != expected_entry:
                    raise CampaignEngineError("hypothesis preflight changed during registration")
        except BaseException:
            # Repository rows are append-only audit evidence.  A partial write
            # is never silently rolled back or retried with altered semantics.
            self._failed = True
            raise

        result = CampaignRoundResult(
            campaign_round=campaign_round,
            accepted=tuple(candidate for candidate, _ in accepted_with_entries),
            rejected=tuple(rejected),
            provider=batch.provider,
            provider_exhausted=batch.exhausted,
            context_hash=context_hash,
        )
        try:
            self.repository.record_event(
                self.run_id,
                "campaign_round_proposed",
                result.to_dict(),
            )
        except BaseException:
            self._failed = True
            raise
        self._last_round = campaign_round
        self._next_round += 1
        return result

    def observe_round(self, observation: ObjectiveObservation) -> StoppingDecision:
        self._assert_active()
        if self._last_round is None or observation.generation != self._last_round:
            raise CampaignEngineError("objective observation must close the most recently proposed round")
        if observation.generation in self._observed_rounds:
            raise CampaignEngineError("campaign round objective was already observed")
        decision = self.stopping.observe(observation, self.budget)
        self._observed_rounds.add(observation.generation)
        if decision.should_stop:
            self._stop_decision = decision
        try:
            self.repository.record_event(
                self.run_id,
                "campaign_round_observed",
                {
                    "campaign_round": observation.generation,
                    "metric": observation.metric,
                    "value": observation.value,
                    "decision": {
                        "should_stop": decision.should_stop,
                        "reason": decision.reason.value,
                        "detail": decision.detail,
                    },
                },
            )
        except BaseException:
            self._failed = True
            raise
        return decision

    def stop_manually(self, detail: str) -> StoppingDecision:
        if not isinstance(detail, str) or not detail.strip():
            raise ValueError("manual stop detail must not be empty")
        if self._stop_decision is None:
            self._stop_decision = StoppingDecision(True, StopReason.MANUAL, detail.strip())
        return self._stop_decision

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": CAMPAIGN_ENGINE_SCHEMA,
            "run_id": self.run_id,
            "protocol_hash": self.protocol_hash,
            "provider": self.provider.name,
            "config": self.config.to_dict(),
            "budget_usage": self.budget.usage.to_dict(),
            "next_campaign_round": self._next_round,
            "hypothesis_registry_hash": self.registry.content_hash,
            "failed_closed": self._failed,
            "stop_reason": (
                self._stop_decision.reason.value if self._stop_decision is not None else None
            ),
        }

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())
