from __future__ import annotations

from enum import Enum


class StringEnum(str, Enum):
    """Enum whose string representation is its stable wire value."""

    def __str__(self) -> str:
        return self.value


class FactorFrequency(StringEnum):
    DAILY = "daily"
    MINUTE = "minute"


class ProposalKind(StringEnum):
    """How a proposal entered a campaign, independent of its campaign round."""

    ROOT = "root"
    MUTATION = "mutation"
    CROSSOVER = "crossover"
    REPAIR = "repair"
    COMPOSITE = "composite"


class FeedbackDisposition(StringEnum):
    """Coarse, local-only actions that may be shown to a proposal provider."""

    PROMOTE = "promote"
    REPAIR = "repair"
    DIVERSIFY = "diversify"
    RETIRE = "retire"
    TECHNICAL_REJECT = "technical_reject"


class FeedbackReason(StringEnum):
    """Closed vocabulary for sanitized local research feedback."""

    LOCAL_GATE_PASS = "local_gate_pass"
    WEAK_SIGNAL = "weak_signal"
    SIGN_INSTABILITY = "sign_instability"
    REGIME_INSTABILITY = "regime_instability"
    EXCESS_TURNOVER = "excess_turnover"
    COST_FAILURE = "cost_failure"
    COVERAGE_FAILURE = "coverage_failure"
    NON_MONOTONIC = "non_monotonic"
    REDUNDANT_SIGNAL = "redundant_signal"
    INVALID_EXPRESSION = "invalid_expression"
    DATA_CONTRACT_FAILURE = "data_contract_failure"


class CandidateState(StringEnum):
    DRAFT = "draft"
    PROPOSED = "proposed"
    VALIDATED = "validated"
    EVALUATING = "evaluating"
    EVALUATED = "evaluated"
    ELIGIBLE = "eligible"
    SELECTED = "selected"
    REJECTED = "rejected"
    EXPORTED = "exported"
    FAILED = "failed"


class RunState(StringEnum):
    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"


class ScoreVisibility(StringEnum):
    """Whether a score is allowed to influence the proposal loop."""

    LOCAL_RESEARCH = "local_research"
    EXTERNAL_HOLDOUT = "external_holdout"


class StopReason(StringEnum):
    CONTINUE = "continue"
    CANDIDATE_BUDGET = "candidate_budget"
    PROVIDER_CALL_BUDGET = "provider_call_budget"
    EVALUATION_BUDGET = "evaluation_budget"
    GENERATION_BUDGET = "generation_budget"
    WALL_TIME_BUDGET = "wall_time_budget"
    PATIENCE_EXHAUSTED = "patience_exhausted"
    MANUAL = "manual"
