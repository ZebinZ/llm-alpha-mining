from __future__ import annotations

import copy

import pandas as pd
import pytest

from factor_production.v5.domain.enums import ScoreVisibility, StopReason
from factor_production.v5.orchestration.budget import BudgetController, BudgetLimits
from factor_production.v5.orchestration.pareto import (
    DEFAULT_PARETO_OBJECTIVES,
    ObjectiveDirection,
    ParetoCheckpoint,
    ParetoGovernanceError,
    ParetoObjective,
    ParetoObjectiveSpec,
    ParetoObservation,
    ParetoObservationLedger,
    ParetoRoundObservation,
    ParetoStopController,
    ParetoStoppingConfig,
    build_campaign_frontier,
    build_round_frontier,
    epsilon_dominates,
    epsilon_equivalent,
    objective_specs_with_epsilon,
    pareto_frontier,
    select_diverse_shortlist,
)
from factor_production.v5.orchestration.stopping import HoldoutLeakageError


def _metrics(**overrides: float) -> dict[str, float]:
    values = {
        "neutral_ic_strength": 0.02,
        "period_stability": 0.01,
        "net_cost_spread": 0.001,
        "turnover": 0.40,
        "coverage": 0.95,
        "novelty": 0.50,
        "complexity": 3.0,
        "compute_cost": 10.0,
    }
    values.update(overrides)
    return values


def _observation(
    candidate_id: str,
    campaign_round: int = 0,
    **metrics: float,
) -> ParetoObservation:
    return ParetoObservation(candidate_id, campaign_round, _metrics(**metrics))


def _budget(**overrides: int) -> BudgetController:
    values = {
        "max_candidates": 50,
        "max_generations": 10,
        "max_provider_calls": 10,
        "max_evaluations": 50,
        "max_wall_seconds": 100,
    }
    values.update(overrides)
    return BudgetController(BudgetLimits(**values))


def test_objective_contract_is_typed_frozen_complete_and_fail_closed() -> None:
    directions = {item.name: item.direction for item in DEFAULT_PARETO_OBJECTIVES}
    assert directions == {
        ParetoObjective.NEUTRAL_IC_STRENGTH: ObjectiveDirection.MAXIMIZE,
        ParetoObjective.PERIOD_STABILITY: ObjectiveDirection.MAXIMIZE,
        ParetoObjective.NET_COST_SPREAD: ObjectiveDirection.MAXIMIZE,
        ParetoObjective.TURNOVER: ObjectiveDirection.MINIMIZE,
        ParetoObjective.COVERAGE: ObjectiveDirection.MAXIMIZE,
        ParetoObjective.NOVELTY: ObjectiveDirection.MAXIMIZE,
        ParetoObjective.COMPLEXITY: ObjectiveDirection.MINIMIZE,
        ParetoObjective.COMPUTE_COST: ObjectiveDirection.MINIMIZE,
    }
    with pytest.raises(ParetoGovernanceError, match="must use direction"):
        ParetoObjectiveSpec(
            ParetoObjective.TURNOVER, ObjectiveDirection.MAXIMIZE
        )
    with pytest.raises(ParetoGovernanceError, match="finite"):
        ParetoObjectiveSpec(
            ParetoObjective.TURNOVER, ObjectiveDirection.MINIMIZE, float("nan")
        )

    missing = _metrics()
    del missing["coverage"]
    with pytest.raises(ParetoGovernanceError, match="missing=.*coverage"):
        ParetoObservation("missing", 0, missing)
    with pytest.raises(ParetoGovernanceError, match="finite"):
        ParetoObservation("nan", 0, _metrics(turnover=float("inf")))
    with pytest.raises(ParetoGovernanceError, match="extra=.*teacher_score"):
        ParetoObservation("extra", 0, {**_metrics(), "teacher_score": 1.0})

    valid = _observation("frozen")
    with pytest.raises(TypeError):
        valid.metrics["turnover"] = 0.1  # type: ignore[index]


def test_external_visibility_cannot_enter_any_pareto_observation() -> None:
    with pytest.raises(HoldoutLeakageError, match="external teacher/official"):
        ParetoObservation(
            "external",
            0,
            _metrics(),
            visibility=ScoreVisibility.EXTERNAL_HOLDOUT,
        )

    payload = _observation("local").to_dict()
    payload["visibility"] = "external_holdout"
    with pytest.raises(HoldoutLeakageError):
        ParetoObservation.from_dict(payload)


def test_epsilon_dominance_handles_maximize_and_minimize_without_scalarization() -> None:
    specs = objective_specs_with_epsilon(
        {
            ParetoObjective.NEUTRAL_IC_STRENGTH: 0.01,
            ParetoObjective.TURNOVER: 0.01,
        }
    )
    baseline = _observation("baseline")
    inside_epsilon = _observation(
        "inside", neutral_ic_strength=0.025, turnover=0.405
    )
    meaningful = _observation(
        "meaningful", neutral_ic_strength=0.031, turnover=0.405
    )

    assert epsilon_equivalent(inside_epsilon, baseline, specs)
    assert not epsilon_dominates(inside_epsilon, baseline, specs)
    assert epsilon_dominates(meaningful, baseline, specs)
    assert not epsilon_dominates(baseline, meaningful, specs)


def test_round_and_campaign_frontiers_preserve_real_tradeoffs() -> None:
    weak = _observation("weak", neutral_ic_strength=0.01, turnover=0.60)
    strong = _observation("strong", neutral_ic_strength=0.03, turnover=0.40)
    cheap = _observation(
        "cheap", neutral_ic_strength=0.02, turnover=0.20, compute_cost=2.0
    )
    first_round = ParetoRoundObservation(0, (weak, strong, cheap))

    round_frontier = build_round_frontier(first_round)
    assert round_frontier.candidate_ids == ("cheap", "strong")
    assert tuple(item.candidate_id for item in pareto_frontier(first_round.observations)) == (
        "cheap",
        "strong",
    )

    novel = _observation(
        "novel",
        1,
        neutral_ic_strength=0.015,
        turnover=0.30,
        novelty=0.95,
    )
    ledger = ParetoObservationLedger()
    ledger.append(first_round)
    ledger.append(ParetoRoundObservation(1, (novel,)))
    campaign = build_campaign_frontier(ledger)

    assert campaign.scope == "campaign"
    assert campaign.through_round == 1
    assert campaign.candidate_ids == ("cheap", "strong", "novel")

    # A serializable round sequence preserves the latest round even when the
    # latest round has no valid observations.
    empty_latest = build_campaign_frontier(
        (first_round, ParetoRoundObservation(2, ()))
    )
    assert empty_latest.through_round == 2


def test_diverse_shortlist_uses_objective_queues_and_fails_closed_on_correlation() -> None:
    observations = (
        _observation("high_ic", neutral_ic_strength=0.05, turnover=0.80),
        _observation("balanced", neutral_ic_strength=0.04, turnover=0.55),
        _observation("low_turnover", neutral_ic_strength=0.03, turnover=0.20),
        _observation(
            "novel", neutral_ic_strength=0.02, turnover=0.30, novelty=0.99
        ),
    )
    ids = [item.candidate_id for item in observations]
    correlations = pd.DataFrame(float("nan"), index=ids, columns=ids)
    for candidate_id in ids:
        correlations.loc[candidate_id, candidate_id] = 1.0
    correlations.loc["high_ic", "balanced"] = 0.95
    correlations.loc["balanced", "high_ic"] = 0.95
    correlations.loc["high_ic", "low_turnover"] = 0.20
    correlations.loc["low_turnover", "high_ic"] = 0.20
    # Correlations involving `novel` are deliberately missing.

    shortlist = select_diverse_shortlist(
        observations,
        correlations,
        capacity=3,
        max_abs_correlation=0.85,
    )

    assert shortlist.candidate_ids == ("high_ic", "low_turnover")
    reasons = {item.candidate_id: item.reason for item in shortlist.rejected}
    assert reasons["balanced"] == "correlation_limit"
    assert reasons["novel"] == "missing_correlation"
    assert "score" not in shortlist.to_dict()

    asymmetric = correlations.copy()
    asymmetric.loc["low_turnover", "high_ic"] = 0.30
    with pytest.raises(ParetoGovernanceError, match="asymmetric"):
        select_diverse_shortlist(
            observations,
            asymmetric,
            capacity=2,
            max_abs_correlation=0.85,
        )


def test_hash_chained_ledger_roundtrips_and_rejects_tampering_or_replacement() -> None:
    ledger = ParetoObservationLedger()
    first = ledger.append(ParetoRoundObservation(0, (_observation("a"),)))
    second = ledger.append(ParetoRoundObservation(1, (_observation("b", 1),)))

    assert first.previous_hash != first.entry_hash
    assert second.previous_hash == first.entry_hash
    restored = ParetoObservationLedger.from_dict(ledger.to_dict())
    assert restored.content_hash == ledger.content_hash
    assert [item.candidate_id for item in restored.observations] == ["a", "b"]

    tampered = copy.deepcopy(ledger.to_dict())
    tampered["entries"][0]["round_observation"]["observations"][0]["metrics"][
        "turnover"
    ] = 0.01
    with pytest.raises(ParetoGovernanceError, match="hash mismatch"):
        ParetoObservationLedger.from_dict(tampered)

    with pytest.raises(ParetoGovernanceError, match="cannot be replaced"):
        ledger.append(ParetoRoundObservation(2, (_observation("a", 2),)))


def test_stopping_depends_only_on_frontier_patience_and_hard_budget() -> None:
    controller = ParetoStopController(ParetoStoppingConfig(patience_rounds=2))
    budget = _budget()
    first = controller.observe(0, (_observation("a"),), budget)
    equivalent = controller.observe(1, (_observation("b", 1),), budget)
    dominated = controller.observe(
        2,
        (
            _observation(
                "c",
                2,
                neutral_ic_strength=0.01,
                period_stability=0.0,
                net_cost_spread=0.0,
                turnover=0.50,
                coverage=0.90,
                novelty=0.40,
                complexity=4.0,
                compute_cost=12.0,
            ),
        ),
        budget,
    )

    assert first.frontier_progressed and not first.should_stop
    assert not equivalent.frontier_progressed and not equivalent.should_stop
    assert dominated.should_stop
    assert dominated.reason is StopReason.PATIENCE_EXHAUSTED

    hard_budget = _budget(max_generations=1)
    hard_budget.reserve(generations=1)
    budget_controller = ParetoStopController(ParetoStoppingConfig())
    decision = budget_controller.observe(0, (_observation("budget"),), hard_budget)
    assert decision.should_stop
    assert decision.reason is StopReason.GENERATION_BUDGET


def test_checkpoint_is_hash_bound_verifiable_and_restorable() -> None:
    config = ParetoStoppingConfig(patience_rounds=3)
    controller = ParetoStopController(config)
    budget = _budget()
    controller.observe(0, (_observation("a"),), budget)
    controller.observe(1, (_observation("b", 1),), budget)
    ledger = controller.ledger
    checkpoint = controller.checkpoint()

    wire = checkpoint.to_dict()
    restored_checkpoint = ParetoCheckpoint.from_dict(wire)
    restored_checkpoint.verify_against(ledger)
    restored_controller = ParetoStopController.from_checkpoint(
        config, restored_checkpoint, ledger
    )
    assert restored_controller.checkpoint().checkpoint_hash == checkpoint.checkpoint_hash

    third = restored_controller.observe(2, (_observation("c", 2),), budget)
    assert not third.should_stop
    fourth = restored_controller.observe(3, (_observation("d", 3),), budget)
    assert fourth.should_stop
    assert fourth.reason is StopReason.PATIENCE_EXHAUSTED

    tampered = dict(wire)
    tampered["stalled_rounds"] = 0
    with pytest.raises(ParetoGovernanceError, match="checkpoint hash mismatch"):
        ParetoCheckpoint.from_dict(tampered)

    manual = dict(wire)
    manual["stop_reason"] = "manual"
    # Recompute is intentionally omitted: either the policy or the content hash
    # must reject a checkpoint that attempts to add a manual/teacher stop path.
    with pytest.raises(ParetoGovernanceError, match="frontier patience or hard budget"):
        ParetoCheckpoint.from_dict(manual)
