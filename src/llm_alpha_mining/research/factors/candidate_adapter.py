from __future__ import annotations

from collections.abc import Mapping

from llm_alpha_mining.research.core.frequency import FrequencySpec
from llm_alpha_mining.research.factors.spec import (
    AggregationSpec,
    ComplexityBudget,
    FactorProvenance,
    FactorSpec,
    PreprocessStep,
)
from llm_alpha_mining.mining.domain import CandidateSpec
from llm_alpha_mining.mining.domain.enums import FactorFrequency, ProposalKind


def factor_spec_from_candidate(
    candidate: CandidateSpec,
    *,
    economic_rationale: str,
    falsification_criterion: str,
    dataset_id: str,
    snapshot_id: str,
    schema_hash: str,
    availability_hash: str,
    security_contract_hash: str,
    frequency: FrequencySpec,
    operator_registry_version: str,
    operator_registry_digest: str,
    parent_factor_hashes: tuple[str, ...] = (),
    preprocessing: tuple[PreprocessStep, ...] = (),
    complexity_budget: ComplexityBudget | None = None,
) -> FactorSpec:
    """Bind a frozen mining candidate to an executable research definition.

    Data, universe and operator bindings are intentionally supplied by the
    caller: the candidate wire model never contained enough evidence to
    infer them safely.

    R1 JointDaily candidates deliberately use a mechanism-family namespace and
    deterministic preregistration provenance that this generic bridge cannot
    represent.  They must pass through the dedicated logic-bound adapter rather
    than being silently mislabeled as LLM proposals.
    """

    if "r1_joint_daily" in candidate.tags:
        raise ValueError(
            "R1 JointDaily candidates require a dedicated logic-bound adapter"
        )
    if not economic_rationale.strip() or not falsification_criterion.strip():
        raise ValueError("Candidate binding requires research rationale")
    candidate_frequency = candidate.frequency
    expected = (
        FactorFrequency.DAILY if frequency.interval == "1d" else FactorFrequency.MINUTE
    )
    if candidate_frequency is not expected:
        raise ValueError("Candidate/frequency binding differs")
    if len(parent_factor_hashes) != len(candidate.parent_ids):
        raise ValueError(
            "Candidate parent ids require explicit registered factor hashes"
        )
    proposal_kind = candidate.proposal_kind
    if not isinstance(proposal_kind, ProposalKind):  # pragma: no cover
        raise RuntimeError("V5 proposal kind was not normalized")
    origin = {
        ProposalKind.ROOT: "llm",
        ProposalKind.MUTATION: "mutation",
        ProposalKind.REPAIR: "mutation",
        ProposalKind.CROSSOVER: "crossover",
        ProposalKind.COMPOSITE: "crossover",
    }[proposal_kind]
    provenance = FactorProvenance(
        origin=origin,
        actor_id=f"v5:{candidate.provider}",
        model_id=candidate.provider if origin == "llm" else None,
        prompt_hash=candidate.protocol_hash if origin == "llm" else None,
        parent_factor_hashes=parent_factor_hashes,
    )
    return FactorSpec(
        factor_id=candidate.candidate_id,
        version=f"v5-generation-{candidate.generation}",
        family=candidate.family,
        hypothesis=candidate.hypothesis,
        economic_rationale=economic_rationale,
        falsification_criterion=falsification_criterion,
        expression=candidate.expression,
        direction=candidate.direction,
        required_fields=tuple(candidate.required_fields),
        dataset_id=dataset_id,
        snapshot_id=snapshot_id,
        schema_hash=schema_hash,
        availability_hash=availability_hash,
        security_contract_hash=security_contract_hash,
        frequency=frequency,
        aggregation=_aggregation(candidate.aggregation, candidate_frequency),
        operator_registry_version=operator_registry_version,
        operator_registry_digest=operator_registry_digest,
        preprocessing=preprocessing,
        complexity_budget=complexity_budget or ComplexityBudget(),
        provenance=provenance,
    )


def _aggregation(
    value: Mapping[str, object], frequency: FactorFrequency
) -> AggregationSpec:
    method = str(value.get("method", "none"))
    window = str(value.get("window", "full_day"))
    smoothing = value.get("smoothing_span", 0)
    if not isinstance(smoothing, int) or isinstance(smoothing, bool):
        raise TypeError("V5 aggregation smoothing_span must be an integer")
    if frequency is FactorFrequency.DAILY:
        if method != "none" or smoothing != 0:
            raise ValueError("daily V5 factors may not declare intraday aggregation")
        return AggregationSpec()
    if method == "none":
        if smoothing:
            raise ValueError("native minute V5 factor may not declare smoothing")
        return AggregationSpec()
    if method not in {"last", "mean", "sum", "std", "skew", "kurt"}:
        raise ValueError(f"unsupported V5 aggregation method:{method}")
    return AggregationSpec(
        method="calendar_daily",
        window=window,
        smoothing_span=smoothing,
        reducer=method,
    )


__all__ = ["factor_spec_from_candidate"]
