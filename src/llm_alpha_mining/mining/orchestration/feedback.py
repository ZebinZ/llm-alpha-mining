from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from llm_alpha_mining.mining.artifacts.hashing import canonical_json_bytes, hash_file
from llm_alpha_mining.mining.domain.enums import FeedbackDisposition, FeedbackReason
from llm_alpha_mining.mining.domain.models import CandidateSpec
from llm_alpha_mining.mining.providers.base import SanitizedFeedback


SANITIZED_FEEDBACK_BATCH_SCHEMA = "sanitized-feedback-batch/v1"


class FeedbackBuildError(ValueError):
    pass


_SIGN_FAILURES = frozenset(
    {
        "raw_2020_ic_not_positive",
        "raw_2021_ic_not_positive",
        "neutral_2020_ic_not_positive",
        "neutral_2021_ic_not_positive",
        "too_few_positive_neutral_half_years",
        "worst_neutral_half_year_below_minimum",
        "raw_positive_week_ratio_below_minimum",
        "neutral_positive_week_ratio_below_minimum",
    }
)
_WEAK_FAILURES = frozenset(
    {
        "raw_full_ic_below_minimum",
        "neutral_full_ic_below_minimum",
        "neutral_primary_test_bh_q_above_maximum",
    }
)
_TURNOVER_FAILURES = frozenset(
    {
        "turnover_above_maximum",
        "one_way_weekly_turnover_above_maximum",
        "excess_turnover",
    }
)
_COST_FAILURES = frozenset(
    {
        "net_20bp_spread_below_minimum",
        "net_cost_spread_below_minimum",
        "cost_failure",
    }
)
_COVERAGE_FAILURES = frozenset(
    {
        "coverage_below_minimum",
        "neutral_coverage_below_minimum",
        "coverage_failure",
    }
)
_MONOTONIC_FAILURES = frozenset(
    {
        "non_monotonic",
        "quantile_monotonicity_below_minimum",
    }
)
_REDUNDANCY_FAILURES = frozenset(
    {
        "redundant_signal",
        "signal_correlation_above_maximum",
        "family_quota_exceeded",
    }
)
_INVALID_EXPRESSION_FAILURES = frozenset(
    {
        "invalid_expression",
        "dsl_validation_failed",
        "unsupported_operator",
        "future_field_detected",
    }
)


@dataclass(frozen=True, slots=True)
class FeedbackBatch:
    campaign_id: str
    source_campaign_round: int
    source_snapshot_sha256: str
    items: tuple[SanitizedFeedback, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SANITIZED_FEEDBACK_BATCH_SCHEMA,
            "campaign_id": self.campaign_id,
            "source_campaign_round": self.source_campaign_round,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "visibility": "adaptive_local_research_only",
            "contains_raw_metric_values": False,
            "contains_external_teacher_feedback": False,
            "items": [item.to_dict() for item in self.items],
        }


def _validate_snapshot_hash(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FeedbackBuildError("source_snapshot_sha256 must be a SHA-256 digest")
    return value


def _reason_codes(
    technical_pass: bool,
    technical_failures: frozenset[str],
    research_pass: bool,
    research_failures: frozenset[str],
) -> tuple[FeedbackReason, ...]:
    reasons: set[FeedbackReason] = set()
    if not technical_pass:
        if technical_failures & _INVALID_EXPRESSION_FAILURES:
            reasons.add(FeedbackReason.INVALID_EXPRESSION)
        if technical_failures - _INVALID_EXPRESSION_FAILURES or not technical_failures:
            reasons.add(FeedbackReason.DATA_CONTRACT_FAILURE)
    elif research_pass:
        reasons.add(FeedbackReason.LOCAL_GATE_PASS)
    else:
        if research_failures & _WEAK_FAILURES:
            reasons.add(FeedbackReason.WEAK_SIGNAL)
        if research_failures & _SIGN_FAILURES:
            reasons.update(
                {FeedbackReason.SIGN_INSTABILITY, FeedbackReason.REGIME_INSTABILITY}
            )
        if research_failures & _TURNOVER_FAILURES:
            reasons.add(FeedbackReason.EXCESS_TURNOVER)
        if research_failures & _COST_FAILURES:
            reasons.add(FeedbackReason.COST_FAILURE)
        if research_failures & _COVERAGE_FAILURES:
            reasons.add(FeedbackReason.COVERAGE_FAILURE)
        if research_failures & _MONOTONIC_FAILURES:
            reasons.add(FeedbackReason.NON_MONOTONIC)
        if research_failures & _REDUNDANCY_FAILURES:
            reasons.add(FeedbackReason.REDUNDANT_SIGNAL)
        known = (
            _WEAK_FAILURES
            | _SIGN_FAILURES
            | _TURNOVER_FAILURES
            | _COST_FAILURES
            | _COVERAGE_FAILURES
            | _MONOTONIC_FAILURES
            | _REDUNDANCY_FAILURES
        )
        if research_failures - known or not reasons:
            # Unknown future gate names are collapsed to a coarse local reason;
            # their free-form names never cross the provider boundary.
            reasons.add(FeedbackReason.WEAK_SIGNAL)
    order = {item: index for index, item in enumerate(FeedbackReason)}
    return tuple(sorted(reasons, key=order.__getitem__))


def _disposition(
    technical_pass: bool,
    research_pass: bool,
    reasons: tuple[FeedbackReason, ...],
) -> FeedbackDisposition:
    reason_set = frozenset(reasons)
    if not technical_pass:
        return FeedbackDisposition.TECHNICAL_REJECT
    if research_pass:
        return FeedbackDisposition.PROMOTE
    if FeedbackReason.REDUNDANT_SIGNAL in reason_set:
        return FeedbackDisposition.DIVERSIFY
    repairable = {
        FeedbackReason.EXCESS_TURNOVER,
        FeedbackReason.COST_FAILURE,
        FeedbackReason.COVERAGE_FAILURE,
        FeedbackReason.NON_MONOTONIC,
    }
    if reason_set & repairable and FeedbackReason.SIGN_INSTABILITY not in reason_set:
        return FeedbackDisposition.REPAIR
    return FeedbackDisposition.RETIRE


def build_feedback_batch(
    *,
    campaign_id: str,
    source_campaign_round: int,
    source_snapshot_sha256: str,
    specs: Iterable[CandidateSpec],
    gate_decisions: Iterable[Mapping[str, object]],
) -> FeedbackBatch:
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise FeedbackBuildError("campaign_id must not be empty")
    if (
        not isinstance(source_campaign_round, int)
        or isinstance(source_campaign_round, bool)
        or source_campaign_round < 0
    ):
        raise FeedbackBuildError("source_campaign_round must be non-negative")
    snapshot = _validate_snapshot_hash(source_snapshot_sha256)
    spec_items = tuple(specs)
    by_id = {item.candidate_id: item for item in spec_items}
    if not by_id:
        raise FeedbackBuildError("spec registry must not be empty")
    if len(by_id) != len(spec_items):
        raise FeedbackBuildError("spec registry contains duplicate candidate IDs")

    items: list[SanitizedFeedback] = []
    seen: set[str] = set()
    for raw in gate_decisions:
        if not isinstance(raw, Mapping):
            raise FeedbackBuildError("gate decision must be an object")
        candidate_id = raw.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in by_id:
            raise FeedbackBuildError(
                f"gate decision references unknown candidate: {candidate_id!r}"
            )
        if candidate_id in seen:
            raise FeedbackBuildError(f"duplicate gate decision: {candidate_id}")
        seen.add(candidate_id)
        spec = by_id[candidate_id]
        if spec.campaign_round != source_campaign_round:
            raise FeedbackBuildError(
                f"candidate round differs from feedback source: {candidate_id}"
            )
        technical_pass = raw.get("technical_pass")
        research_pass = raw.get("research_evidence_pass")
        if not isinstance(technical_pass, bool) or not isinstance(research_pass, bool):
            raise FeedbackBuildError("gate pass fields must be booleans")
        technical_raw = raw.get("technical_failures", [])
        research_raw = raw.get("research_failures", [])
        if not isinstance(technical_raw, list) or not all(
            isinstance(value, str) for value in technical_raw
        ):
            raise FeedbackBuildError("technical_failures must be a list of strings")
        if not isinstance(research_raw, list) or not all(
            isinstance(value, str) for value in research_raw
        ):
            raise FeedbackBuildError("research_failures must be a list of strings")
        reasons = _reason_codes(
            technical_pass,
            frozenset(technical_raw),
            research_pass,
            frozenset(research_raw),
        )
        items.append(
            SanitizedFeedback(
                candidate_id=candidate_id,
                candidate_hash=spec.content_hash,
                campaign_round=source_campaign_round,
                disposition=_disposition(technical_pass, research_pass, reasons),
                reason_codes=reasons,
            )
        )
    expected_ids = {
        candidate_id
        for candidate_id, spec in by_id.items()
        if spec.campaign_round == source_campaign_round
    }
    if seen != expected_ids:
        raise FeedbackBuildError(
            f"gate decisions do not exactly cover source round: missing={sorted(expected_ids - seen)}, "
            f"extra={sorted(seen - expected_ids)}"
        )
    return FeedbackBatch(
        campaign_id=campaign_id.strip(),
        source_campaign_round=source_campaign_round,
        source_snapshot_sha256=snapshot,
        items=tuple(sorted(items, key=lambda item: item.candidate_id)),
    )


def write_feedback_batch(batch: FeedbackBatch, path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json_bytes(batch.to_dict()) + b"\n")
    digest = hash_file(target)
    target.with_suffix(".sha256").write_text(
        f"{digest}  {target.name}\n", encoding="utf-8"
    )
    return digest


def load_gate_decisions_jsonl(path: str | Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for line_number, raw in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FeedbackBuildError(
                f"invalid gate JSON at line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise FeedbackBuildError(f"gate line {line_number} is not an object")
        rows.append(value)
    return tuple(rows)
