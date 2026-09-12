from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class SubmissionGateDecision:
    """Local research decision made before the terminal external audit.

    A pass never means production admission.  During an open campaign the FDR
    value is generation-provisional; it must be recomputed over every evaluated
    candidate before a final submission release can be frozen.
    """

    candidate_id: str
    gate_version: str
    technical_pass: bool
    research_evidence_pass: bool
    provisional_generation_pass: bool
    campaign_closed: bool
    submission_candidate: bool
    technical_failures: tuple[str, ...]
    research_failures: tuple[str, ...]
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["technical_failures"] = list(self.technical_failures)
        payload["research_failures"] = list(self.research_failures)
        return payload


def evaluate_submission_gate(
    analysis: Mapping[str, Any],
    technical_evidence: Mapping[str, bool],
    business_protocol: Mapping[str, Any],
    *,
    neutral_bh_q: float | None,
    campaign_closed: bool,
) -> SubmissionGateDecision:
    """Apply the preregistered V5 gate without constructing a scalar score."""

    governance = business_protocol["selection_governance"]
    hard = governance["hard_technical_gate"]["conditions"]
    research = governance["local_research_evidence_gate"]["conditions"]
    technical_failures: list[str] = []
    research_failures: list[str] = []

    evidence_conditions = (
        "candidate_spec_hash_matches_frozen_registry",
        "protocol_and_input_manifest_hashes_verify",
        "safe_dsl_and_operator_registry_validation_pass",
        "no_future_or_label_field_in_expression",
        "stock_only_universe_applied_before_cross_sectional_operations",
        "zero_non_equity_identifiers_in_factor_cross_sections",
        "forward_return_unknowns_are_not_filled_with_zero",
        "direction_and_aggregation_match_preregistration",
        "evaluation_completed_without_nonfinite_metric_corruption",
    )
    for name in evidence_conditions:
        if not bool(technical_evidence.get(name, False)):
            technical_failures.append(name)

    if analysis.get("status") != "ok":
        technical_failures.append("analysis_status_not_ok")
    if analysis.get("raw_signal_status") != "available":
        technical_failures.append("raw_signal_unavailable")
    if analysis.get("neutral_signal_status") != "available":
        technical_failures.append("neutral_signal_unavailable")
    if not _at_least(analysis.get("raw_week_count"), hard["minimum_raw_week_count"]):
        technical_failures.append("raw_week_count_below_minimum")
    if not _at_least(
        analysis.get("neutral_week_count"), hard["minimum_neutral_week_count"]
    ):
        technical_failures.append("neutral_week_count_below_minimum")
    if not _at_least(analysis.get("coverage_mean"), hard["minimum_raw_coverage_mean"]):
        technical_failures.append("raw_coverage_below_minimum")
    if not _at_least(
        analysis.get("neutral_coverage_mean"),
        hard["minimum_neutral_coverage_mean"],
    ):
        technical_failures.append("neutral_coverage_below_minimum")

    raw_periods = analysis.get("period_raw_ic", {})
    neutral_periods = analysis.get("period_neutral_ic", {})
    if not _at_least(
        raw_periods.get("full_2020_2021"),
        research["minimum_raw_rank_ic_full_2020_2021"],
    ):
        research_failures.append("raw_full_ic_below_minimum")
    if not _at_least(
        neutral_periods.get("full_2020_2021"),
        research["minimum_neutral_rank_ic_full_2020_2021"],
    ):
        research_failures.append("neutral_full_ic_below_minimum")
    for signal_name, periods in (("raw", raw_periods), ("neutral", neutral_periods)):
        for year in ("2020", "2021"):
            if not _strictly_positive(periods.get(year)):
                research_failures.append(f"{signal_name}_{year}_ic_not_positive")

    neutral_halves = [
        _finite(neutral_periods.get(name))
        for name in ("2020H1", "2020H2", "2021H1", "2021H2")
    ]
    positive_halves = sum(value is not None and value > 0 for value in neutral_halves)
    if positive_halves < int(
        research["minimum_positive_neutral_half_years_out_of_four"]
    ):
        research_failures.append("too_few_positive_neutral_half_years")
    if any(value is None for value in neutral_halves) or min(
        value for value in neutral_halves if value is not None
    ) < float(research["minimum_worst_neutral_half_year_rank_ic"]):
        research_failures.append("worst_neutral_half_year_below_minimum")
    if not _at_least(
        analysis.get("positive_week_ratio"),
        research["minimum_raw_positive_week_ratio"],
    ):
        research_failures.append("raw_positive_week_ratio_below_minimum")
    if not _at_least(
        analysis.get("neutral_positive_week_ratio"),
        research["minimum_neutral_positive_week_ratio"],
    ):
        research_failures.append("neutral_positive_week_ratio_below_minimum")
    if not _at_most(
        analysis.get("one_way_weekly_turnover"),
        research["maximum_one_way_weekly_turnover"],
    ):
        research_failures.append("turnover_above_maximum")
    if not _at_least(
        analysis.get("q5_minus_q1_net_20bp_mean"),
        research["minimum_q5_minus_q1_net_20bp_mean"],
    ):
        research_failures.append("net_20bp_spread_below_minimum")
    if not _at_most(
        neutral_bh_q,
        research["maximum_benjamini_hochberg_q_for_primary_neutral_test"],
    ):
        research_failures.append("neutral_primary_test_bh_q_above_maximum")

    technical_failures = list(dict.fromkeys(technical_failures))
    research_failures = list(dict.fromkeys(research_failures))
    technical_pass = not technical_failures
    research_pass = not research_failures
    provisional = technical_pass and research_pass
    submission_candidate = provisional and campaign_closed
    interpretation = (
        "locally_eligible_for_one_frozen_terminal_external_audit"
        if submission_candidate
        else (
            "generation_provisional_pass_pending_campaign_wide_fdr_and_diversity"
            if provisional
            else "local_gate_not_passed"
        )
    )
    return SubmissionGateDecision(
        candidate_id=str(analysis.get("candidate_id", "unknown")),
        gate_version=str(governance["local_gate_version"]),
        technical_pass=technical_pass,
        research_evidence_pass=research_pass,
        provisional_generation_pass=provisional,
        campaign_closed=bool(campaign_closed),
        submission_candidate=submission_candidate,
        technical_failures=tuple(technical_failures),
        research_failures=tuple(research_failures),
        interpretation=interpretation,
    )


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _at_least(value: Any, threshold: Any) -> bool:
    number = _finite(value)
    limit = _finite(threshold)
    return number is not None and limit is not None and number >= limit


def _at_most(value: Any, threshold: Any) -> bool:
    number = _finite(value)
    limit = _finite(threshold)
    return number is not None and limit is not None and number <= limit


def _strictly_positive(value: Any) -> bool:
    number = _finite(value)
    return number is not None and number > 0
