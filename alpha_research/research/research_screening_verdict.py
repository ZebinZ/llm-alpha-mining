"""Deterministic, fail-closed R1 research candidate screening.

This module is deliberately independent from the research runner.  It accepts
only a closed, preregistered candidate denominator and already-computed
development evidence.  It cannot read hidden OOS data, admit a factor, publish
an artifact, or claim production readiness.

The screening receipt is content addressed and replayable.  Zero selected
candidates is a valid outcome: the selection cap is a maximum, never a quota.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
import math
from types import MappingProxyType
from typing import Final, cast

from alpha_research.core.hashing import canonical_json_bytes, hash_json, require_sha256


SCREENING_SPEC_SCHEMA: Final = "r1-research-screening-spec/v1"
SCREENING_EVIDENCE_SCHEMA: Final = "r1-research-screening-evidence/v1"
SCREENING_VERDICT_SCHEMA: Final = "r1-research-screening-verdict/v1"
SCREENING_RECEIPT_SCHEMA: Final = "r1-research-screening-receipt/v1"
SCREENING_FAMILY_SCHEMA: Final = "r1-research-screening-family/v1"
SCREENING_FAMILY_SUMMARY_SCHEMA: Final = "r1-research-screening-family-summary/v1"


class ResearchScreeningDecision(str, Enum):
    """Candidate or batch screening decision."""

    PASS = "pass"
    REJECT = "reject"


class ResearchScreeningReason(str, Enum):
    """Stable, machine-readable screening reason codes."""

    SCREENING_PASSED = "screening_passed"
    INSUFFICIENT_OBSERVATIONS = "insufficient_observations"
    COVERAGE_BELOW_THRESHOLD = "coverage_below_threshold"
    IC_MEAN_BELOW_THRESHOLD = "ic_mean_below_threshold"
    IC_HAC_T_BELOW_THRESHOLD = "ic_hac_t_below_threshold"
    IC_FDR_Q_ABOVE_THRESHOLD = "ic_fdr_q_above_threshold"
    ESTIMATED_COST_ABOVE_THRESHOLD = "estimated_cost_above_threshold"
    TURNOVER_ABOVE_THRESHOLD = "turnover_above_threshold"
    NET_SPREAD_BELOW_THRESHOLD = "net_spread_below_threshold"
    PROVIDER_REGIME_COUNT_BELOW_THRESHOLD = (
        "provider_regime_count_below_threshold"
    )
    PROVIDER_REGIME_FAILURE = "provider_regime_failure"
    PROVIDER_REGIME_IC_BELOW_THRESHOLD = "provider_regime_ic_below_threshold"
    NEGATIVE_CONTROL_TRIALS_BELOW_THRESHOLD = (
        "negative_control_trials_below_threshold"
    )
    NEGATIVE_CONTROL_NOT_BEATEN = "negative_control_not_beaten"
    PARAMETER_VARIANTS_BELOW_THRESHOLD = "parameter_variants_below_threshold"
    PARAMETER_PASS_RATIO_BELOW_THRESHOLD = (
        "parameter_pass_ratio_below_threshold"
    )
    PARAMETER_SIGN_AGREEMENT_BELOW_THRESHOLD = (
        "parameter_sign_agreement_below_threshold"
    )
    PARAMETER_WORST_IC_BELOW_THRESHOLD = (
        "parameter_worst_ic_below_threshold"
    )
    INCREMENTAL_IC_BELOW_THRESHOLD = "incremental_ic_below_threshold"
    INCREMENTAL_HAC_T_BELOW_THRESHOLD = "incremental_hac_t_below_threshold"
    DUPLICATE_CANDIDATE = "duplicate_candidate"
    DUPLICATE_REFERENCE_INVALID = "duplicate_reference_invalid"
    PEER_CORRELATION_ABOVE_THRESHOLD = "peer_correlation_above_threshold"
    SELECTION_CAP_REACHED = "selection_cap_reached"


class ResearchScreeningError(ValueError):
    """Stable fail-closed input or replay error."""

    def __init__(self, code: str, detail: str) -> None:
        if (
            not isinstance(code, str)
            or not code
            or not code[0].isalpha()
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in code)
        ):
            raise ValueError("research screening error code is unsafe")
        self.code = code
        self.detail = str(detail)
        super().__init__(f"{self.code}:{self.detail}")


def _error(code: str, detail: str) -> ResearchScreeningError:
    return ResearchScreeningError(code, detail)


def _exact_fields(value: Mapping[str, object], expected: frozenset[str], *, name: str) -> None:
    if not isinstance(value, Mapping):
        raise _error("invalid_mapping", f"{name} must be a mapping")
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise _error(
            "field_set_mismatch",
            f"{name} fields differ; missing={missing!r}; extra={extra!r}",
        )


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise _error("invalid_text", f"{name} must be non-empty canonical text")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-" for character in value):
        raise _error("invalid_text", f"{name} contains unsupported characters")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _error("invalid_integer", f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error("invalid_number", f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise _error("non_finite_number", f"{name} must be finite")
    return number


def _bounded(value: object, *, name: str, lower: float, upper: float) -> float:
    number = _finite(value, name=name)
    if number < lower or number > upper:
        raise _error("number_out_of_range", f"{name} must be in [{lower}, {upper}]")
    return number


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise _error("invalid_digest", f"{name} must be a lowercase sha256 digest")
    try:
        return cast(str, require_sha256(value, name=name))
    except ValueError as exc:
        raise _error("invalid_digest", str(exc)) from exc


def _research_boundary(
    *,
    research_only: object,
    admission_claim: object,
    production_ready: object,
    hidden_oos_consumed: object,
    release_authorized: object,
) -> None:
    if research_only is not True:
        raise _error("boundary_violation", "screening must remain research-only")
    if admission_claim is not False:
        raise _error("boundary_violation", "screening cannot claim admission")
    if production_ready is not False:
        raise _error("boundary_violation", "screening cannot claim production readiness")
    if hidden_oos_consumed is not False:
        raise _error("hidden_oos_forbidden", "screening cannot consume hidden OOS")
    if release_authorized is not False:
        raise _error("release_forbidden", "screening cannot authorize release")


def _canonical_mapping(payload: bytes, *, name: str) -> Mapping[str, object]:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error("invalid_wire_payload", f"{name} is not canonical JSON") from exc
    if not isinstance(decoded, dict):
        raise _error("invalid_wire_payload", f"{name} must decode to an object")
    if canonical_json_bytes(decoded) != payload:
        raise _error("noncanonical_wire_payload", f"{name} is not canonical JSON")
    return cast(Mapping[str, object], decoded)


@dataclass(frozen=True, slots=True)
class ResearchScreeningFamilyV1:
    """One complete, preregistered family denominator."""

    family_id: str
    candidate_ids: tuple[str, ...]
    schema_version: str = SCREENING_FAMILY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SCREENING_FAMILY_SCHEMA:
            raise _error("unsupported_schema", "unsupported screening family schema")
        object.__setattr__(self, "family_id", _text(self.family_id, name="family_id"))
        candidates = tuple(_text(item, name="candidate_id") for item in self.candidate_ids)
        if not candidates:
            raise _error("empty_family_denominator", "family denominator cannot be empty")
        if len(set(candidates)) != len(candidates):
            raise _error("duplicate_candidate_id", "family candidate ids must be unique")
        if candidates != tuple(sorted(candidates)):
            raise _error("noncanonical_denominator", "family candidate ids must be sorted")
        object.__setattr__(self, "candidate_ids", candidates)

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "family_id": self.family_id,
            "candidate_ids": list(self.candidate_ids),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ResearchScreeningFamilyV1":
        _exact_fields(
            value,
            frozenset({"schema_version", "family_id", "candidate_ids"}),
            name="ResearchScreeningFamilyV1",
        )
        raw_ids = value["candidate_ids"]
        if not isinstance(raw_ids, list):
            raise _error("invalid_sequence", "candidate_ids must be a JSON list")
        return cls(
            schema_version=cast(str, value["schema_version"]),
            family_id=cast(str, value["family_id"]),
            candidate_ids=tuple(cast(Sequence[str], raw_ids)),
        )


@dataclass(frozen=True, slots=True)
class ResearchScreeningSpecV1:
    """Closed screening policy with a complete family denominator."""

    campaign_id: str
    campaign_spec_hash: str
    families: tuple[ResearchScreeningFamilyV1, ...]
    max_selected_candidates: int
    min_observations: int
    min_coverage: float
    min_ic_mean: float
    min_ic_hac_t_stat: float
    max_ic_fdr_q_value: float
    max_estimated_cost_bps: float
    max_turnover: float
    min_net_spread_bps: float
    min_provider_regime_count: int
    min_worst_provider_regime_ic: float
    min_negative_control_trials: int
    max_negative_control_p_value: float
    min_parameter_variants: int
    min_parameter_pass_ratio: float
    min_parameter_sign_agreement: float
    min_worst_parameter_ic: float
    min_incremental_ic: float
    min_incremental_hac_t_stat: float
    max_abs_peer_correlation: float
    research_only: bool = True
    admission_claim: bool = False
    production_ready: bool = False
    hidden_oos_consumed: bool = False
    release_authorized: bool = False
    schema_version: str = SCREENING_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SCREENING_SPEC_SCHEMA:
            raise _error("unsupported_schema", "unsupported screening spec schema")
        object.__setattr__(self, "campaign_id", _text(self.campaign_id, name="campaign_id"))
        object.__setattr__(
            self,
            "campaign_spec_hash",
            _digest(self.campaign_spec_hash, name="campaign_spec_hash"),
        )
        families = tuple(self.families)
        if not families or any(type(item) is not ResearchScreeningFamilyV1 for item in families):
            raise _error("invalid_family_denominator", "families must be exact family records")
        if tuple(item.family_id for item in families) != tuple(
            sorted(item.family_id for item in families)
        ):
            raise _error("noncanonical_denominator", "families must be sorted by family id")
        family_ids = [item.family_id for item in families]
        candidate_ids = [candidate for item in families for candidate in item.candidate_ids]
        if len(set(family_ids)) != len(family_ids):
            raise _error("duplicate_family_id", "family ids must be unique")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise _error("duplicate_candidate_id", "candidate ids must be globally unique")
        object.__setattr__(self, "families", families)
        total = len(candidate_ids)
        maximum = _integer(
            self.max_selected_candidates,
            name="max_selected_candidates",
        )
        if maximum > total:
            raise _error("invalid_selection_cap", "selection cap exceeds denominator")
        object.__setattr__(self, "max_selected_candidates", maximum)
        for name, minimum in (
            ("min_observations", 1),
            ("min_provider_regime_count", 1),
            ("min_negative_control_trials", 1),
            ("min_parameter_variants", 1),
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name=name, minimum=minimum))
        for name in (
            "min_coverage",
            "max_ic_fdr_q_value",
            "max_turnover",
            "max_negative_control_p_value",
            "min_parameter_pass_ratio",
            "min_parameter_sign_agreement",
            "max_abs_peer_correlation",
        ):
            object.__setattr__(self, name, _bounded(getattr(self, name), name=name, lower=0.0, upper=1.0))
        for name in (
            "min_ic_mean",
            "min_worst_provider_regime_ic",
            "min_worst_parameter_ic",
            "min_incremental_ic",
        ):
            object.__setattr__(self, name, _bounded(getattr(self, name), name=name, lower=-1.0, upper=1.0))
        for name in (
            "min_ic_hac_t_stat",
            "min_incremental_hac_t_stat",
            "min_net_spread_bps",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "max_estimated_cost_bps",
            _bounded(
                self.max_estimated_cost_bps,
                name="max_estimated_cost_bps",
                lower=0.0,
                upper=1_000_000.0,
            ),
        )
        _research_boundary(
            research_only=self.research_only,
            admission_claim=self.admission_claim,
            production_ready=self.production_ready,
            hidden_oos_consumed=self.hidden_oos_consumed,
            release_authorized=self.release_authorized,
        )

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate for family in self.families for candidate in family.candidate_ids)

    @property
    def family_by_candidate(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                candidate: family.family_id
                for family in self.families
                for candidate in family.candidate_ids
            }
        )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "campaign_spec_hash": self.campaign_spec_hash,
            "families": [item.to_dict() for item in self.families],
            "max_selected_candidates": self.max_selected_candidates,
            "min_observations": self.min_observations,
            "min_coverage": self.min_coverage,
            "min_ic_mean": self.min_ic_mean,
            "min_ic_hac_t_stat": self.min_ic_hac_t_stat,
            "max_ic_fdr_q_value": self.max_ic_fdr_q_value,
            "max_estimated_cost_bps": self.max_estimated_cost_bps,
            "max_turnover": self.max_turnover,
            "min_net_spread_bps": self.min_net_spread_bps,
            "min_provider_regime_count": self.min_provider_regime_count,
            "min_worst_provider_regime_ic": self.min_worst_provider_regime_ic,
            "min_negative_control_trials": self.min_negative_control_trials,
            "max_negative_control_p_value": self.max_negative_control_p_value,
            "min_parameter_variants": self.min_parameter_variants,
            "min_parameter_pass_ratio": self.min_parameter_pass_ratio,
            "min_parameter_sign_agreement": self.min_parameter_sign_agreement,
            "min_worst_parameter_ic": self.min_worst_parameter_ic,
            "min_incremental_ic": self.min_incremental_ic,
            "min_incremental_hac_t_stat": self.min_incremental_hac_t_stat,
            "max_abs_peer_correlation": self.max_abs_peer_correlation,
            "research_only": self.research_only,
            "admission_claim": self.admission_claim,
            "production_ready": self.production_ready,
            "hidden_oos_consumed": self.hidden_oos_consumed,
            "release_authorized": self.release_authorized,
        }

    def to_wire_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self.to_dict()))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ResearchScreeningSpecV1":
        fields = frozenset(
            {
                "schema_version",
                "campaign_id",
                "campaign_spec_hash",
                "families",
                "max_selected_candidates",
                "min_observations",
                "min_coverage",
                "min_ic_mean",
                "min_ic_hac_t_stat",
                "max_ic_fdr_q_value",
                "max_estimated_cost_bps",
                "max_turnover",
                "min_net_spread_bps",
                "min_provider_regime_count",
                "min_worst_provider_regime_ic",
                "min_negative_control_trials",
                "max_negative_control_p_value",
                "min_parameter_variants",
                "min_parameter_pass_ratio",
                "min_parameter_sign_agreement",
                "min_worst_parameter_ic",
                "min_incremental_ic",
                "min_incremental_hac_t_stat",
                "max_abs_peer_correlation",
                "research_only",
                "admission_claim",
                "production_ready",
                "hidden_oos_consumed",
                "release_authorized",
            }
        )
        _exact_fields(value, fields, name="ResearchScreeningSpecV1")
        raw_families = value["families"]
        if not isinstance(raw_families, list):
            raise _error("invalid_sequence", "families must be a JSON list")
        families = tuple(
            ResearchScreeningFamilyV1.from_mapping(cast(Mapping[str, object], item))
            for item in raw_families
            if isinstance(item, Mapping)
        )
        if len(families) != len(raw_families):
            raise _error("invalid_mapping", "every family must be a mapping")
        return cls(
            schema_version=cast(str, value["schema_version"]),
            campaign_id=cast(str, value["campaign_id"]),
            campaign_spec_hash=cast(str, value["campaign_spec_hash"]),
            families=families,
            max_selected_candidates=cast(int, value["max_selected_candidates"]),
            min_observations=cast(int, value["min_observations"]),
            min_coverage=cast(float, value["min_coverage"]),
            min_ic_mean=cast(float, value["min_ic_mean"]),
            min_ic_hac_t_stat=cast(float, value["min_ic_hac_t_stat"]),
            max_ic_fdr_q_value=cast(float, value["max_ic_fdr_q_value"]),
            max_estimated_cost_bps=cast(float, value["max_estimated_cost_bps"]),
            max_turnover=cast(float, value["max_turnover"]),
            min_net_spread_bps=cast(float, value["min_net_spread_bps"]),
            min_provider_regime_count=cast(int, value["min_provider_regime_count"]),
            min_worst_provider_regime_ic=cast(float, value["min_worst_provider_regime_ic"]),
            min_negative_control_trials=cast(int, value["min_negative_control_trials"]),
            max_negative_control_p_value=cast(float, value["max_negative_control_p_value"]),
            min_parameter_variants=cast(int, value["min_parameter_variants"]),
            min_parameter_pass_ratio=cast(float, value["min_parameter_pass_ratio"]),
            min_parameter_sign_agreement=cast(float, value["min_parameter_sign_agreement"]),
            min_worst_parameter_ic=cast(float, value["min_worst_parameter_ic"]),
            min_incremental_ic=cast(float, value["min_incremental_ic"]),
            min_incremental_hac_t_stat=cast(float, value["min_incremental_hac_t_stat"]),
            max_abs_peer_correlation=cast(float, value["max_abs_peer_correlation"]),
            research_only=cast(bool, value["research_only"]),
            admission_claim=cast(bool, value["admission_claim"]),
            production_ready=cast(bool, value["production_ready"]),
            hidden_oos_consumed=cast(bool, value["hidden_oos_consumed"]),
            release_authorized=cast(bool, value["release_authorized"]),
        )

    @classmethod
    def from_wire_bytes(cls, payload: bytes) -> "ResearchScreeningSpecV1":
        return cls.from_mapping(_canonical_mapping(payload, name="screening spec"))


@dataclass(frozen=True, slots=True)
class ResearchScreeningCandidateEvidenceV1:
    """Development-only, numeric evidence for one closed candidate."""

    candidate_id: str
    family_id: str
    source_evidence_hash: str
    observations: int
    coverage: float
    ic_mean: float
    ic_hac_t_stat: float
    ic_fdr_q_value: float
    estimated_cost_bps: float
    turnover: float
    net_spread_bps: float
    provider_regime_count: int
    provider_regime_pass_count: int
    worst_provider_regime_ic: float
    negative_control_trials: int
    negative_control_p_value: float
    parameter_variant_count: int
    parameter_variant_pass_count: int
    parameter_sign_agreement: float
    worst_parameter_ic: float
    incremental_ic: float
    incremental_hac_t_stat: float
    max_abs_peer_correlation: float
    duplicate_of_candidate_id: str | None = None
    research_only: bool = True
    admission_claim: bool = False
    production_ready: bool = False
    hidden_oos_consumed: bool = False
    release_authorized: bool = False
    schema_version: str = SCREENING_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SCREENING_EVIDENCE_SCHEMA:
            raise _error("unsupported_schema", "unsupported screening evidence schema")
        for name in ("candidate_id", "family_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "source_evidence_hash",
            _digest(self.source_evidence_hash, name="source_evidence_hash"),
        )
        for name in (
            "observations",
            "provider_regime_count",
            "provider_regime_pass_count",
            "negative_control_trials",
            "parameter_variant_count",
            "parameter_variant_pass_count",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name=name))
        if self.provider_regime_pass_count > self.provider_regime_count:
            raise _error("invalid_count", "provider passes exceed provider regimes")
        if self.parameter_variant_pass_count > self.parameter_variant_count:
            raise _error("invalid_count", "parameter passes exceed parameter variants")
        for name in (
            "coverage",
            "ic_fdr_q_value",
            "turnover",
            "negative_control_p_value",
            "parameter_sign_agreement",
            "max_abs_peer_correlation",
        ):
            object.__setattr__(self, name, _bounded(getattr(self, name), name=name, lower=0.0, upper=1.0))
        for name in (
            "ic_mean",
            "worst_provider_regime_ic",
            "worst_parameter_ic",
            "incremental_ic",
        ):
            object.__setattr__(self, name, _bounded(getattr(self, name), name=name, lower=-1.0, upper=1.0))
        for name in ("ic_hac_t_stat", "net_spread_bps", "incremental_hac_t_stat"):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "estimated_cost_bps",
            _bounded(
                self.estimated_cost_bps,
                name="estimated_cost_bps",
                lower=0.0,
                upper=1_000_000.0,
            ),
        )
        if self.duplicate_of_candidate_id is not None:
            object.__setattr__(
                self,
                "duplicate_of_candidate_id",
                _text(self.duplicate_of_candidate_id, name="duplicate_of_candidate_id"),
            )
        _research_boundary(
            research_only=self.research_only,
            admission_claim=self.admission_claim,
            production_ready=self.production_ready,
            hidden_oos_consumed=self.hidden_oos_consumed,
            release_authorized=self.release_authorized,
        )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "family_id": self.family_id,
            "source_evidence_hash": self.source_evidence_hash,
            "observations": self.observations,
            "coverage": self.coverage,
            "ic_mean": self.ic_mean,
            "ic_hac_t_stat": self.ic_hac_t_stat,
            "ic_fdr_q_value": self.ic_fdr_q_value,
            "estimated_cost_bps": self.estimated_cost_bps,
            "turnover": self.turnover,
            "net_spread_bps": self.net_spread_bps,
            "provider_regime_count": self.provider_regime_count,
            "provider_regime_pass_count": self.provider_regime_pass_count,
            "worst_provider_regime_ic": self.worst_provider_regime_ic,
            "negative_control_trials": self.negative_control_trials,
            "negative_control_p_value": self.negative_control_p_value,
            "parameter_variant_count": self.parameter_variant_count,
            "parameter_variant_pass_count": self.parameter_variant_pass_count,
            "parameter_sign_agreement": self.parameter_sign_agreement,
            "worst_parameter_ic": self.worst_parameter_ic,
            "incremental_ic": self.incremental_ic,
            "incremental_hac_t_stat": self.incremental_hac_t_stat,
            "max_abs_peer_correlation": self.max_abs_peer_correlation,
            "duplicate_of_candidate_id": self.duplicate_of_candidate_id,
            "research_only": self.research_only,
            "admission_claim": self.admission_claim,
            "production_ready": self.production_ready,
            "hidden_oos_consumed": self.hidden_oos_consumed,
            "release_authorized": self.release_authorized,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "ResearchScreeningCandidateEvidenceV1":
        _exact_fields(value, frozenset(cls.__dataclass_fields__), name=cls.__name__)
        return cls(**cast(dict[str, object], dict(value)))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ResearchScreeningCandidateVerdictV1:
    """One deterministic candidate decision."""

    candidate_id: str
    family_id: str
    evidence_hash: str
    decision: ResearchScreeningDecision
    reason_codes: tuple[ResearchScreeningReason, ...]
    eligible_before_cap: bool
    schema_version: str = SCREENING_VERDICT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SCREENING_VERDICT_SCHEMA:
            raise _error("unsupported_schema", "unsupported screening verdict schema")
        for name in ("candidate_id", "family_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        object.__setattr__(self, "evidence_hash", _digest(self.evidence_hash, name="evidence_hash"))
        try:
            decision = ResearchScreeningDecision(self.decision)
            reasons = tuple(ResearchScreeningReason(item) for item in self.reason_codes)
        except ValueError as exc:
            raise _error("invalid_verdict", "decision or reason code is invalid") from exc
        if type(self.eligible_before_cap) is not bool:
            raise _error("invalid_verdict", "eligible_before_cap must be boolean")
        if not reasons or len(set(reasons)) != len(reasons):
            raise _error("invalid_verdict", "reason codes must be non-empty and unique")
        if decision is ResearchScreeningDecision.PASS:
            if reasons != (ResearchScreeningReason.SCREENING_PASSED,) or not self.eligible_before_cap:
                raise _error("invalid_verdict", "pass verdict is inconsistent")
        elif ResearchScreeningReason.SCREENING_PASSED in reasons:
            raise _error("invalid_verdict", "reject verdict cannot contain pass reason")
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "reason_codes", reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "family_id": self.family_id,
            "evidence_hash": self.evidence_hash,
            "decision": self.decision.value,
            "reason_codes": [item.value for item in self.reason_codes],
            "eligible_before_cap": self.eligible_before_cap,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "ResearchScreeningCandidateVerdictV1":
        _exact_fields(
            value,
            frozenset(
                {
                    "schema_version",
                    "candidate_id",
                    "family_id",
                    "evidence_hash",
                    "decision",
                    "reason_codes",
                    "eligible_before_cap",
                }
            ),
            name=cls.__name__,
        )
        raw_reasons = value["reason_codes"]
        if not isinstance(raw_reasons, list):
            raise _error("invalid_sequence", "reason_codes must be a JSON list")
        return cls(
            schema_version=cast(str, value["schema_version"]),
            candidate_id=cast(str, value["candidate_id"]),
            family_id=cast(str, value["family_id"]),
            evidence_hash=cast(str, value["evidence_hash"]),
            decision=ResearchScreeningDecision(cast(str, value["decision"])),
            reason_codes=tuple(ResearchScreeningReason(cast(str, item)) for item in raw_reasons),
            eligible_before_cap=cast(bool, value["eligible_before_cap"]),
        )


@dataclass(frozen=True, slots=True)
class ResearchScreeningFamilySummaryV1:
    """Receipt-preserved family denominator and counts."""

    family_id: str
    denominator_candidate_ids: tuple[str, ...]
    tested_count: int
    eligible_count: int
    passed_count: int
    failed_count: int
    schema_version: str = SCREENING_FAMILY_SUMMARY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SCREENING_FAMILY_SUMMARY_SCHEMA:
            raise _error("unsupported_schema", "unsupported family summary schema")
        object.__setattr__(self, "family_id", _text(self.family_id, name="family_id"))
        candidates = tuple(_text(item, name="candidate_id") for item in self.denominator_candidate_ids)
        if not candidates or candidates != tuple(sorted(candidates)) or len(set(candidates)) != len(candidates):
            raise _error("invalid_family_summary", "family denominator differs")
        object.__setattr__(self, "denominator_candidate_ids", candidates)
        for name in ("tested_count", "eligible_count", "passed_count", "failed_count"):
            object.__setattr__(self, name, _integer(getattr(self, name), name=name))
        denominator = len(candidates)
        if (
            self.tested_count != denominator
            or self.failed_count != denominator - self.passed_count
            or self.eligible_count < self.passed_count
            or self.eligible_count > denominator
        ):
            raise _error("invalid_family_summary", "family counts are inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "family_id": self.family_id,
            "denominator_candidate_ids": list(self.denominator_candidate_ids),
            "tested_count": self.tested_count,
            "eligible_count": self.eligible_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "ResearchScreeningFamilySummaryV1":
        _exact_fields(
            value,
            frozenset(
                {
                    "schema_version",
                    "family_id",
                    "denominator_candidate_ids",
                    "tested_count",
                    "eligible_count",
                    "passed_count",
                    "failed_count",
                }
            ),
            name=cls.__name__,
        )
        raw_ids = value["denominator_candidate_ids"]
        if not isinstance(raw_ids, list):
            raise _error("invalid_sequence", "denominator_candidate_ids must be a JSON list")
        return cls(
            schema_version=cast(str, value["schema_version"]),
            family_id=cast(str, value["family_id"]),
            denominator_candidate_ids=tuple(cast(Sequence[str], raw_ids)),
            tested_count=cast(int, value["tested_count"]),
            eligible_count=cast(int, value["eligible_count"]),
            passed_count=cast(int, value["passed_count"]),
            failed_count=cast(int, value["failed_count"]),
        )


@dataclass(frozen=True, slots=True)
class ResearchScreeningReceiptV1:
    """Versioned, content-addressed and replayable screening receipt."""

    campaign_id: str
    screening_spec_hash: str
    evidence_set_hash: str
    batch_decision: ResearchScreeningDecision
    tested_count: int
    eligible_count: int
    passed_count: int
    failed_count: int
    max_selected_candidates: int
    zero_selection_permitted: bool
    family_summaries: tuple[ResearchScreeningFamilySummaryV1, ...]
    candidate_verdicts: tuple[ResearchScreeningCandidateVerdictV1, ...]
    research_only: bool = True
    admission_claim: bool = False
    production_ready: bool = False
    hidden_oos_consumed: bool = False
    release_authorized: bool = False
    schema_version: str = SCREENING_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SCREENING_RECEIPT_SCHEMA:
            raise _error("unsupported_schema", "unsupported screening receipt schema")
        object.__setattr__(self, "campaign_id", _text(self.campaign_id, name="campaign_id"))
        for name in ("screening_spec_hash", "evidence_set_hash"):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        try:
            decision = ResearchScreeningDecision(self.batch_decision)
        except ValueError as exc:
            raise _error("invalid_receipt", "batch decision is invalid") from exc
        object.__setattr__(self, "batch_decision", decision)
        for name in (
            "tested_count",
            "eligible_count",
            "passed_count",
            "failed_count",
            "max_selected_candidates",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name=name))
        if self.zero_selection_permitted is not True:
            raise _error("minimum_quota_forbidden", "zero candidate selection must be permitted")
        summaries = tuple(self.family_summaries)
        verdicts = tuple(self.candidate_verdicts)
        if any(type(item) is not ResearchScreeningFamilySummaryV1 for item in summaries):
            raise _error("invalid_receipt", "family summaries differ")
        if any(type(item) is not ResearchScreeningCandidateVerdictV1 for item in verdicts):
            raise _error("invalid_receipt", "candidate verdicts differ")
        if not summaries or not verdicts:
            raise _error("invalid_receipt", "receipt denominator cannot be empty")
        if tuple(item.family_id for item in summaries) != tuple(sorted(item.family_id for item in summaries)):
            raise _error("invalid_receipt", "family summaries are not canonical")
        if len({item.family_id for item in summaries}) != len(summaries):
            raise _error("invalid_receipt", "family summary ids are not unique")
        if tuple(item.candidate_id for item in verdicts) != tuple(sorted(item.candidate_id for item in verdicts)):
            raise _error("invalid_receipt", "candidate verdicts are not canonical")
        if len({item.candidate_id for item in verdicts}) != len(verdicts):
            raise _error("invalid_receipt", "candidate verdict ids are not unique")
        denominator_owner: dict[str, str] = {}
        for summary in summaries:
            for candidate_id in summary.denominator_candidate_ids:
                if candidate_id in denominator_owner:
                    raise _error(
                        "invalid_receipt",
                        "family candidate denominators are not disjoint",
                    )
                denominator_owner[candidate_id] = summary.family_id
        verdict_by_candidate = {item.candidate_id: item for item in verdicts}
        if set(verdict_by_candidate) != set(denominator_owner):
            raise _error("invalid_receipt", "verdict denominator differs")
        for candidate_id, verdict in verdict_by_candidate.items():
            if verdict.family_id != denominator_owner[candidate_id]:
                raise _error("invalid_receipt", "verdict family binding differs")
        for summary in summaries:
            family_verdicts = [
                verdict_by_candidate[candidate_id]
                for candidate_id in summary.denominator_candidate_ids
            ]
            family_eligible = sum(
                1 for item in family_verdicts if item.eligible_before_cap
            )
            family_passed = sum(
                1
                for item in family_verdicts
                if item.decision is ResearchScreeningDecision.PASS
            )
            if (
                summary.tested_count != len(family_verdicts)
                or summary.eligible_count != family_eligible
                or summary.passed_count != family_passed
                or summary.failed_count != len(family_verdicts) - family_passed
            ):
                raise _error("invalid_receipt", "family verdict counts differ")
        derived_eligible = sum(1 for item in verdicts if item.eligible_before_cap)
        derived_passed = sum(
            1
            for item in verdicts
            if item.decision is ResearchScreeningDecision.PASS
        )
        if (
            self.tested_count != len(verdicts)
            or self.failed_count != self.tested_count - self.passed_count
            or self.eligible_count != derived_eligible
            or self.passed_count != derived_passed
            or self.eligible_count < self.passed_count
            or self.eligible_count > self.tested_count
            or self.passed_count > self.max_selected_candidates
            or sum(item.tested_count for item in summaries) != self.tested_count
            or sum(item.eligible_count for item in summaries) != self.eligible_count
            or sum(item.passed_count for item in summaries) != self.passed_count
            or sum(item.failed_count for item in summaries) != self.failed_count
        ):
            raise _error("invalid_receipt", "receipt counts are inconsistent")
        expected_decision = (
            ResearchScreeningDecision.PASS
            if self.passed_count > 0
            else ResearchScreeningDecision.REJECT
        )
        if decision is not expected_decision:
            raise _error("invalid_receipt", "batch decision differs from selected count")
        object.__setattr__(self, "family_summaries", summaries)
        object.__setattr__(self, "candidate_verdicts", verdicts)
        _research_boundary(
            research_only=self.research_only,
            admission_claim=self.admission_claim,
            production_ready=self.production_ready,
            hidden_oos_consumed=self.hidden_oos_consumed,
            release_authorized=self.release_authorized,
        )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "screening_spec_hash": self.screening_spec_hash,
            "evidence_set_hash": self.evidence_set_hash,
            "batch_decision": self.batch_decision.value,
            "tested_count": self.tested_count,
            "eligible_count": self.eligible_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "max_selected_candidates": self.max_selected_candidates,
            "zero_selection_permitted": self.zero_selection_permitted,
            "family_summaries": [item.to_dict() for item in self.family_summaries],
            "candidate_verdicts": [item.to_dict() for item in self.candidate_verdicts],
            "research_only": self.research_only,
            "admission_claim": self.admission_claim,
            "production_ready": self.production_ready,
            "hidden_oos_consumed": self.hidden_oos_consumed,
            "release_authorized": self.release_authorized,
        }

    def to_wire_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self.to_dict()))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ResearchScreeningReceiptV1":
        _exact_fields(value, frozenset(cls.__dataclass_fields__), name=cls.__name__)
        raw_summaries = value["family_summaries"]
        raw_verdicts = value["candidate_verdicts"]
        if not isinstance(raw_summaries, list) or not isinstance(raw_verdicts, list):
            raise _error("invalid_sequence", "receipt children must be JSON lists")
        if any(not isinstance(item, Mapping) for item in raw_summaries):
            raise _error("invalid_mapping", "every family summary must be a mapping")
        if any(not isinstance(item, Mapping) for item in raw_verdicts):
            raise _error("invalid_mapping", "every candidate verdict must be a mapping")
        return cls(
            schema_version=cast(str, value["schema_version"]),
            campaign_id=cast(str, value["campaign_id"]),
            screening_spec_hash=cast(str, value["screening_spec_hash"]),
            evidence_set_hash=cast(str, value["evidence_set_hash"]),
            batch_decision=ResearchScreeningDecision(cast(str, value["batch_decision"])),
            tested_count=cast(int, value["tested_count"]),
            eligible_count=cast(int, value["eligible_count"]),
            passed_count=cast(int, value["passed_count"]),
            failed_count=cast(int, value["failed_count"]),
            max_selected_candidates=cast(int, value["max_selected_candidates"]),
            zero_selection_permitted=cast(bool, value["zero_selection_permitted"]),
            family_summaries=tuple(
                ResearchScreeningFamilySummaryV1.from_mapping(
                    cast(Mapping[str, object], item)
                )
                for item in raw_summaries
            ),
            candidate_verdicts=tuple(
                ResearchScreeningCandidateVerdictV1.from_mapping(
                    cast(Mapping[str, object], item)
                )
                for item in raw_verdicts
            ),
            research_only=cast(bool, value["research_only"]),
            admission_claim=cast(bool, value["admission_claim"]),
            production_ready=cast(bool, value["production_ready"]),
            hidden_oos_consumed=cast(bool, value["hidden_oos_consumed"]),
            release_authorized=cast(bool, value["release_authorized"]),
        )

    @classmethod
    def from_wire_bytes(
        cls,
        payload: bytes,
        *,
        expected_hash: str | None = None,
    ) -> "ResearchScreeningReceiptV1":
        receipt = cls.from_mapping(_canonical_mapping(payload, name="screening receipt"))
        if expected_hash is not None:
            expected = _digest(expected_hash, name="expected_receipt_hash")
            if receipt.content_hash != expected:
                raise _error("receipt_hash_mismatch", "screening receipt hash differs")
        return receipt


class ResearchScreeningVerdictEngineV1:
    """Evaluate a closed candidate denominator and emit one replayable receipt."""

    __slots__ = ("_spec",)

    _spec: ResearchScreeningSpecV1

    def __init__(self, spec: ResearchScreeningSpecV1) -> None:
        if type(spec) is not ResearchScreeningSpecV1:
            raise TypeError("screening engine requires an exact screening spec")
        object.__setattr__(
            self,
            "_spec",
            ResearchScreeningSpecV1.from_wire_bytes(spec.to_wire_bytes()),
        )

    @property
    def screening_spec_hash(self) -> str:
        return self._spec.content_hash

    def screen(
        self,
        evidence: Sequence[ResearchScreeningCandidateEvidenceV1],
    ) -> ResearchScreeningReceiptV1:
        snapshots: list[ResearchScreeningCandidateEvidenceV1] = []
        for item in evidence:
            if type(item) is not ResearchScreeningCandidateEvidenceV1:
                raise TypeError("screening evidence must use exact V1 records")
            snapshots.append(
                ResearchScreeningCandidateEvidenceV1.from_mapping(item.to_dict())
            )
        observed_ids = [item.candidate_id for item in snapshots]
        if len(set(observed_ids)) != len(observed_ids):
            raise _error("duplicate_candidate_evidence", "candidate evidence is duplicated")
        expected_ids = set(self._spec.candidate_ids)
        observed_set = set(observed_ids)
        if observed_set != expected_ids:
            missing = sorted(expected_ids - observed_set)
            extra = sorted(observed_set - expected_ids)
            raise _error(
                "incomplete_family_denominator",
                f"candidate denominator differs; missing={missing!r}; extra={extra!r}",
            )
        by_candidate = {item.candidate_id: item for item in snapshots}
        family_by_candidate = self._spec.family_by_candidate
        for candidate_id, item in by_candidate.items():
            if item.family_id != family_by_candidate[candidate_id]:
                raise _error("family_binding_mismatch", "candidate family binding differs")

        initial_reasons = {
            candidate_id: self._condition_failures(item, expected_ids)
            for candidate_id, item in by_candidate.items()
        }
        eligible_ids = [
            candidate_id
            for candidate_id, reasons in initial_reasons.items()
            if not reasons
        ]
        eligible_ids.sort(key=lambda candidate_id: self._ranking_key(by_candidate[candidate_id]))
        selected = set(eligible_ids[: self._spec.max_selected_candidates])

        verdicts: list[ResearchScreeningCandidateVerdictV1] = []
        for candidate_id in sorted(by_candidate):
            item = by_candidate[candidate_id]
            reasons = initial_reasons[candidate_id]
            eligible = not reasons
            final_reasons: tuple[ResearchScreeningReason, ...]
            if candidate_id in selected:
                decision = ResearchScreeningDecision.PASS
                final_reasons = (ResearchScreeningReason.SCREENING_PASSED,)
            else:
                decision = ResearchScreeningDecision.REJECT
                final_reasons = (
                    tuple(reasons)
                    if reasons
                    else (ResearchScreeningReason.SELECTION_CAP_REACHED,)
                )
            verdicts.append(
                ResearchScreeningCandidateVerdictV1(
                    candidate_id=candidate_id,
                    family_id=item.family_id,
                    evidence_hash=item.content_hash,
                    decision=decision,
                    reason_codes=final_reasons,
                    eligible_before_cap=eligible,
                )
            )

        verdict_by_id = {item.candidate_id: item for item in verdicts}
        family_summaries: list[ResearchScreeningFamilySummaryV1] = []
        for family in self._spec.families:
            family_verdicts = [verdict_by_id[item] for item in family.candidate_ids]
            passed = sum(
                1
                for item in family_verdicts
                if item.decision is ResearchScreeningDecision.PASS
            )
            family_eligible_count = sum(
                1 for item in family_verdicts if item.eligible_before_cap
            )
            family_summaries.append(
                ResearchScreeningFamilySummaryV1(
                    family_id=family.family_id,
                    denominator_candidate_ids=family.candidate_ids,
                    tested_count=len(family_verdicts),
                    eligible_count=family_eligible_count,
                    passed_count=passed,
                    failed_count=len(family_verdicts) - passed,
                )
            )
        tested_count = len(verdicts)
        passed_count = len(selected)
        evidence_set_hash = cast(
            str,
            hash_json(
                {
                    "schema_version": "r1-research-screening-evidence-set/v1",
                    "screening_spec_hash": self._spec.content_hash,
                    "evidence_hashes": {
                        candidate_id: by_candidate[candidate_id].content_hash
                        for candidate_id in sorted(by_candidate)
                    },
                }
            ),
        )
        return ResearchScreeningReceiptV1(
            campaign_id=self._spec.campaign_id,
            screening_spec_hash=self._spec.content_hash,
            evidence_set_hash=evidence_set_hash,
            batch_decision=(
                ResearchScreeningDecision.PASS
                if passed_count > 0
                else ResearchScreeningDecision.REJECT
            ),
            tested_count=tested_count,
            eligible_count=len(eligible_ids),
            passed_count=passed_count,
            failed_count=tested_count - passed_count,
            max_selected_candidates=self._spec.max_selected_candidates,
            zero_selection_permitted=True,
            family_summaries=tuple(family_summaries),
            candidate_verdicts=tuple(verdicts),
        )

    def verify_and_replay(
        self,
        *,
        receipt_payload: bytes,
        expected_receipt_hash: str,
        evidence: Sequence[ResearchScreeningCandidateEvidenceV1],
    ) -> ResearchScreeningReceiptV1:
        """Verify a pinned receipt and reproduce it from the same evidence."""

        observed = ResearchScreeningReceiptV1.from_wire_bytes(
            receipt_payload,
            expected_hash=expected_receipt_hash,
        )
        if observed.screening_spec_hash != self._spec.content_hash:
            raise _error("screening_spec_mismatch", "receipt screening spec differs")
        replayed = self.screen(evidence)
        if replayed.to_wire_bytes() != observed.to_wire_bytes():
            raise _error("screening_replay_mismatch", "screening replay differs")
        return replayed

    def _condition_failures(
        self,
        item: ResearchScreeningCandidateEvidenceV1,
        expected_ids: set[str],
    ) -> tuple[ResearchScreeningReason, ...]:
        spec = self._spec
        reasons: list[ResearchScreeningReason] = []
        if item.observations < spec.min_observations:
            reasons.append(ResearchScreeningReason.INSUFFICIENT_OBSERVATIONS)
        if item.coverage < spec.min_coverage:
            reasons.append(ResearchScreeningReason.COVERAGE_BELOW_THRESHOLD)
        if item.ic_mean < spec.min_ic_mean:
            reasons.append(ResearchScreeningReason.IC_MEAN_BELOW_THRESHOLD)
        if item.ic_hac_t_stat < spec.min_ic_hac_t_stat:
            reasons.append(ResearchScreeningReason.IC_HAC_T_BELOW_THRESHOLD)
        if item.ic_fdr_q_value > spec.max_ic_fdr_q_value:
            reasons.append(ResearchScreeningReason.IC_FDR_Q_ABOVE_THRESHOLD)
        if item.estimated_cost_bps > spec.max_estimated_cost_bps:
            reasons.append(ResearchScreeningReason.ESTIMATED_COST_ABOVE_THRESHOLD)
        if item.turnover > spec.max_turnover:
            reasons.append(ResearchScreeningReason.TURNOVER_ABOVE_THRESHOLD)
        if item.net_spread_bps < spec.min_net_spread_bps:
            reasons.append(ResearchScreeningReason.NET_SPREAD_BELOW_THRESHOLD)
        if item.provider_regime_count < spec.min_provider_regime_count:
            reasons.append(
                ResearchScreeningReason.PROVIDER_REGIME_COUNT_BELOW_THRESHOLD
            )
        if item.provider_regime_pass_count != item.provider_regime_count:
            reasons.append(ResearchScreeningReason.PROVIDER_REGIME_FAILURE)
        if item.worst_provider_regime_ic < spec.min_worst_provider_regime_ic:
            reasons.append(
                ResearchScreeningReason.PROVIDER_REGIME_IC_BELOW_THRESHOLD
            )
        if item.negative_control_trials < spec.min_negative_control_trials:
            reasons.append(
                ResearchScreeningReason.NEGATIVE_CONTROL_TRIALS_BELOW_THRESHOLD
            )
        if item.negative_control_p_value > spec.max_negative_control_p_value:
            reasons.append(ResearchScreeningReason.NEGATIVE_CONTROL_NOT_BEATEN)
        if item.parameter_variant_count < spec.min_parameter_variants:
            reasons.append(
                ResearchScreeningReason.PARAMETER_VARIANTS_BELOW_THRESHOLD
            )
        parameter_pass_ratio = (
            item.parameter_variant_pass_count / item.parameter_variant_count
            if item.parameter_variant_count
            else 0.0
        )
        if parameter_pass_ratio < spec.min_parameter_pass_ratio:
            reasons.append(
                ResearchScreeningReason.PARAMETER_PASS_RATIO_BELOW_THRESHOLD
            )
        if item.parameter_sign_agreement < spec.min_parameter_sign_agreement:
            reasons.append(
                ResearchScreeningReason.PARAMETER_SIGN_AGREEMENT_BELOW_THRESHOLD
            )
        if item.worst_parameter_ic < spec.min_worst_parameter_ic:
            reasons.append(
                ResearchScreeningReason.PARAMETER_WORST_IC_BELOW_THRESHOLD
            )
        if item.incremental_ic < spec.min_incremental_ic:
            reasons.append(ResearchScreeningReason.INCREMENTAL_IC_BELOW_THRESHOLD)
        if item.incremental_hac_t_stat < spec.min_incremental_hac_t_stat:
            reasons.append(
                ResearchScreeningReason.INCREMENTAL_HAC_T_BELOW_THRESHOLD
            )
        duplicate = item.duplicate_of_candidate_id
        if duplicate is not None:
            reasons.append(ResearchScreeningReason.DUPLICATE_CANDIDATE)
            if duplicate == item.candidate_id or duplicate not in expected_ids:
                reasons.append(ResearchScreeningReason.DUPLICATE_REFERENCE_INVALID)
        if item.max_abs_peer_correlation > spec.max_abs_peer_correlation:
            reasons.append(
                ResearchScreeningReason.PEER_CORRELATION_ABOVE_THRESHOLD
            )
        return tuple(reasons)

    @staticmethod
    def _ranking_key(item: ResearchScreeningCandidateEvidenceV1) -> tuple[float, float, float, str]:
        return (
            -item.incremental_hac_t_stat,
            -item.ic_hac_t_stat,
            -item.net_spread_bps,
            item.candidate_id,
        )


__all__ = [
    "ResearchScreeningCandidateEvidenceV1",
    "ResearchScreeningCandidateVerdictV1",
    "ResearchScreeningDecision",
    "ResearchScreeningError",
    "ResearchScreeningFamilySummaryV1",
    "ResearchScreeningFamilyV1",
    "ResearchScreeningReason",
    "ResearchScreeningReceiptV1",
    "ResearchScreeningSpecV1",
    "ResearchScreeningVerdictEngineV1",
]
