from __future__ import annotations

from pathlib import Path

import pytest

from llm_alpha_mining.mining.domain.enums import (
    CandidateState,
    FeedbackDisposition,
    FeedbackReason,
    ProposalKind,
    RunState,
)
from llm_alpha_mining.mining.domain.models import CandidateSpecError, CandidateSpec
from llm_alpha_mining.mining.orchestration.budget import BudgetController, BudgetLimits
from llm_alpha_mining.mining.orchestration.campaign import (
    CampaignConfig,
    CampaignEngine,
)
from llm_alpha_mining.mining.orchestration.hypotheses import (
    HypothesisRegistry,
    HypothesisRegistryError,
)
from llm_alpha_mining.mining.orchestration.repository import SQLiteRepository
from llm_alpha_mining.mining.orchestration.stopping import (
    ObjectiveObservation,
    StopController,
    StoppingConfig,
)
from llm_alpha_mining.mining.providers import MockProvider, SanitizedFeedback


PROTOCOL_HASH = "a" * 64


def _candidate(
    candidate_id: str,
    expression: str,
    *,
    campaign_round: int = 0,
    lineage_depth: int = 0,
    proposal_kind: ProposalKind = ProposalKind.ROOT,
    parent_ids: tuple[str, ...] = (),
    direction: int = 1,
    explicit: bool = True,
) -> CandidateSpec:
    lineage = (
        {
            "campaign_round": campaign_round,
            "lineage_depth": lineage_depth,
            "proposal_kind": proposal_kind,
        }
        if explicit
        else {}
    )
    return CandidateSpec(
        candidate_id=candidate_id,
        hypothesis=f"mechanism for {candidate_id}",
        expression=expression,
        direction=direction,
        frequency="daily",
        generation=campaign_round,
        family="test_family",
        required_fields=("Returns",),
        protocol_hash=PROTOCOL_HASH,
        provider="mock",
        parent_ids=parent_ids,
        **lineage,
    )


def _budget() -> BudgetController:
    return BudgetController(
        BudgetLimits(
            max_candidates=10,
            max_generations=4,
            max_provider_calls=4,
            max_evaluations=10,
            max_wall_seconds=60,
        )
    )


def test_legacy_candidate_wire_and_hash_are_unchanged() -> None:
    legacy = CandidateSpec(
        candidate_id="legacy",
        hypothesis="h",
        expression="CsRank(Returns)",
        direction=1,
        frequency="daily",
        generation=0,
        family="f",
        required_fields=("Returns",),
        protocol_hash=PROTOCOL_HASH,
        provider="p",
    )

    assert (
        legacy.content_hash
        == "3968ec4fa8dcf944419e674fad1699f89ac1d04d10a41e6ffea40015120933a9"
    )
    assert "campaign_round" not in legacy.to_dict()
    assert legacy.campaign_round == 0
    assert legacy.lineage_depth == 0
    assert legacy.proposal_kind is ProposalKind.ROOT
    assert not legacy.has_explicit_lineage
    assert CandidateSpec.from_dict(legacy.to_dict()).content_hash == legacy.content_hash


def test_explicit_lineage_separates_round_from_depth_and_roundtrips() -> None:
    late_root = _candidate(
        "late_root",
        "CsRank(TsMean(Returns, 5))",
        campaign_round=2,
        lineage_depth=0,
        proposal_kind=ProposalKind.ROOT,
    )
    payload = late_root.to_dict()

    assert payload["generation"] == 2
    assert payload["campaign_round"] == 2
    assert payload["lineage_depth"] == 0
    assert payload["proposal_kind"] == "root"
    assert CandidateSpec.from_dict(payload) == late_root
    with pytest.raises(CandidateSpecError, match="supplied together"):
        CandidateSpec.from_dict({**payload, "lineage_depth": None})
    with pytest.raises(CandidateSpecError, match="exactly one parent"):
        _candidate(
            "bad_mutation",
            "CsRank(TsMean(Returns, 10))",
            campaign_round=2,
            lineage_depth=1,
            proposal_kind=ProposalKind.MUTATION,
        )


def test_hypothesis_registry_checks_lineage_and_semantic_duplicates() -> None:
    registry = HypothesisRegistry(PROTOCOL_HASH)
    root = _candidate("root", "CsRank(TsMean(Returns, 5))")
    registry.register(root)
    mutation = _candidate(
        "mutation",
        "CsRank(TsMean(Returns, 10))",
        campaign_round=1,
        lineage_depth=1,
        proposal_kind=ProposalKind.MUTATION,
        parent_ids=(root.candidate_id,),
    )
    registry.register(mutation)

    duplicate = _candidate(
        "sign_flip_duplicate",
        "CsRank( TsMean(Returns,5) )",
        campaign_round=1,
        lineage_depth=0,
        proposal_kind=ProposalKind.ROOT,
        direction=-1,
    )
    with pytest.raises(HypothesisRegistryError) as duplicate_error:
        registry.register(duplicate)
    assert duplicate_error.value.code == "duplicate_semantic_signal"

    wrong_depth = _candidate(
        "wrong_depth",
        "CsRank(TsStd(Returns, 20))",
        campaign_round=2,
        lineage_depth=2,
        proposal_kind=ProposalKind.MUTATION,
        parent_ids=(root.candidate_id,),
    )
    with pytest.raises(HypothesisRegistryError) as depth_error:
        registry.register(wrong_depth)
    assert depth_error.value.code == "lineage_depth_mismatch"
    assert registry.get("mutation").lineage_depth == 1


def test_sanitized_feedback_has_closed_vocabulary_and_no_raw_metric() -> None:
    candidate = _candidate("root", "CsRank(Returns)")
    feedback = SanitizedFeedback(
        candidate_id=candidate.candidate_id,
        candidate_hash=candidate.content_hash,
        campaign_round=0,
        disposition=FeedbackDisposition.REPAIR,
        reason_codes=(FeedbackReason.COST_FAILURE, FeedbackReason.EXCESS_TURNOVER),
    )

    assert set(feedback.to_dict()) == {
        "candidate_id",
        "candidate_hash",
        "campaign_round",
        "disposition",
        "reason_codes",
    }
    with pytest.raises(ValueError, match="unsupported reason"):
        SanitizedFeedback(
            candidate_id=candidate.candidate_id,
            candidate_hash=candidate.content_hash,
            campaign_round=0,
            disposition="repair",
            reason_codes=("teacher_score",),
        )


def test_campaign_engine_accepts_late_roots_mutations_and_persists_ledger(
    tmp_path: Path,
) -> None:
    root = _candidate("root", "CsRank(TsMean(Returns, 5))")
    late_root = _candidate(
        "late_root",
        "CsRank(TsStd(Returns, 10))",
        campaign_round=1,
        lineage_depth=0,
        proposal_kind=ProposalKind.ROOT,
    )
    mutation = _candidate(
        "mutation",
        "CsRank(TsMean(Returns, 20))",
        campaign_round=1,
        lineage_depth=1,
        proposal_kind=ProposalKind.REPAIR,
        parent_ids=(root.candidate_id,),
    )
    provider = MockProvider((root, late_root, mutation))

    with SQLiteRepository(tmp_path / "control.sqlite3") as repository:
        repository.create_run("run", PROTOCOL_HASH)
        repository.transition_run("run", RunState.INITIALIZED, reason="frozen")
        engine = CampaignEngine(
            run_id="run",
            protocol_hash=PROTOCOL_HASH,
            provider=provider,
            repository=repository,
            budget=_budget(),
            stopping=StopController(StoppingConfig(patience_generations=2)),
            config=CampaignConfig(proposals_per_round=3),
        )

        first = engine.propose_round(0)
        assert first.accepted == (root,)
        assert engine.observe_round(ObjectiveObservation(0, 0.01, "local_pareto"))
        feedback = SanitizedFeedback(
            candidate_id=root.candidate_id,
            candidate_hash=root.content_hash,
            campaign_round=0,
            disposition=FeedbackDisposition.REPAIR,
            reason_codes=(FeedbackReason.COST_FAILURE,),
        )
        second = engine.propose_round(1, feedback=(feedback,))

        assert second.accepted == (late_root, mutation)
        assert engine.registry.get("late_root").lineage_depth == 0
        assert engine.registry.get("mutation").lineage_depth == 1
        assert (
            repository.get_candidate("run", "mutation").state is CandidateState.PROPOSED
        )
        assert repository.verify_integrity("run") == ()
        assert engine.next_campaign_round == 2
        assert len(engine.content_hash) == 64


def test_campaign_engine_rejects_same_signal_resubmitted_with_other_sign(
    tmp_path: Path,
) -> None:
    root = _candidate("root", "CsRank(TsMean(Returns, 5))")
    duplicate = _candidate(
        "duplicate",
        "CsRank( TsMean(Returns, 5) )",
        campaign_round=1,
        lineage_depth=0,
        proposal_kind=ProposalKind.ROOT,
        direction=-1,
    )
    with SQLiteRepository(tmp_path / "control.sqlite3") as repository:
        repository.create_run("run", PROTOCOL_HASH)
        repository.transition_run("run", RunState.INITIALIZED, reason="frozen")
        engine = CampaignEngine(
            run_id="run",
            protocol_hash=PROTOCOL_HASH,
            provider=MockProvider((root, duplicate)),
            repository=repository,
            budget=_budget(),
            stopping=StopController(StoppingConfig()),
            config=CampaignConfig(proposals_per_round=2),
        )
        engine.propose_round(0)
        engine.observe_round(ObjectiveObservation(0, 0.01, "local_pareto"))
        result = engine.propose_round(1)

    assert not result.accepted
    assert result.rejected[0].reason_code == "duplicate_semantic_signal"
