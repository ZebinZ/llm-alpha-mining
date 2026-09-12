from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Mapping, TypeVar, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from alpha_research.core.hashing import hash_frame, hash_json, require_sha256


_ScenarioKey = TypeVar("_ScenarioKey")


class DLReadinessVerdict(str, Enum):
    GO_RESEARCH_ONLY = "go_research_only"
    NO_GO = "no_go"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class ResearchReadinessSpec:
    readiness_id: str
    version: str
    pseudo_label_trials: int
    random_seed: int
    signal_lag_sessions: tuple[int, ...]
    required_perturbation_names: tuple[str, ...]
    minimum_cross_sectional_observations: int = 10
    minimum_valid_dates: int = 20
    maximum_pseudo_label_pvalue: float = 0.05
    minimum_observed_rank_ic: float = 0.0
    minimum_lag_absolute_gap: float = 0.0
    minimum_stability_retention: float = 0.5
    minimum_incremental_rank_ic: float = 0.0
    minimum_incremental_positive_date_fraction: float = 0.5
    schema_version: str = "research-readiness-spec/v2"

    def __post_init__(self) -> None:
        if self.schema_version != "research-readiness-spec/v2":
            raise ValueError("unsupported research readiness specification schema")
        if not self.readiness_id.strip() or not self.version.strip():
            raise ValueError("research readiness id and version are required")
        if (
            not isinstance(self.pseudo_label_trials, int)
            or isinstance(self.pseudo_label_trials, bool)
            or self.pseudo_label_trials < 20
        ):
            raise ValueError("research readiness requires at least 20 pseudo trials")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise TypeError("research readiness random_seed must be an integer")
        lags = tuple(self.signal_lag_sessions)
        if (
            not lags
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in lags
            )
            or lags != tuple(sorted(set(lags)))
        ):
            raise ValueError("signal lags must be positive, sorted and unique")
        object.__setattr__(self, "signal_lag_sessions", lags)
        perturbations = tuple(self.required_perturbation_names)
        if (
            not perturbations
            or any(
                not isinstance(name, str)
                or not name.strip()
                or name != name.strip()
                or name == "__candidate__"
                for name in perturbations
            )
            or perturbations != tuple(sorted(set(perturbations)))
        ):
            raise ValueError(
                "required perturbation names must be non-empty, sorted and unique"
            )
        object.__setattr__(self, "required_perturbation_names", perturbations)
        for name, minimum in (
            ("minimum_cross_sectional_observations", 2),
            ("minimum_valid_dates", 2),
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"research readiness {name} is invalid")
        for name in (
            "maximum_pseudo_label_pvalue",
            "minimum_stability_retention",
            "minimum_incremental_positive_date_fraction",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"research readiness {name} must lie in [0, 1]")
            object.__setattr__(self, name, value)
        if self.maximum_pseudo_label_pvalue == 0.0:
            raise ValueError("maximum pseudo-label p-value must be positive")
        for name in (
            "minimum_observed_rank_ic",
            "minimum_lag_absolute_gap",
            "minimum_incremental_rank_ic",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"research readiness {name} must be non-negative")
            object.__setattr__(self, name, value)

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "readiness_id": self.readiness_id,
            "version": self.version,
            "pseudo_label_trials": self.pseudo_label_trials,
            "random_seed": self.random_seed,
            "signal_lag_sessions": list(self.signal_lag_sessions),
            "required_perturbation_names": list(self.required_perturbation_names),
            "minimum_cross_sectional_observations": (
                self.minimum_cross_sectional_observations
            ),
            "minimum_valid_dates": self.minimum_valid_dates,
            "maximum_pseudo_label_pvalue": self.maximum_pseudo_label_pvalue,
            "minimum_observed_rank_ic": self.minimum_observed_rank_ic,
            "minimum_lag_absolute_gap": self.minimum_lag_absolute_gap,
            "minimum_stability_retention": self.minimum_stability_retention,
            "minimum_incremental_rank_ic": self.minimum_incremental_rank_ic,
            "minimum_incremental_positive_date_fraction": (
                self.minimum_incremental_positive_date_fraction
            ),
        }


@dataclass(frozen=True, slots=True)
class NegativeControlReport:
    readiness_spec: ResearchReadinessSpec
    readiness_spec_hash: str
    candidate_score_hash: str
    label_values_hash: str
    label_validity_hash: str
    scoring_eligibility_hash: str
    observed_rank_ic_mean: float | None
    observed_valid_dates: int
    pseudo_label_pvalue: float | None
    observed_results_hash: str
    null_distribution_hash: str
    lag_results_hash: str
    lag_daily_results_hash: str
    passed: bool | None
    observed_results: pd.DataFrame
    null_distribution: pd.DataFrame
    lag_results: pd.DataFrame
    lag_daily_results: pd.DataFrame
    schema_version: str = "negative-control-report/v2"

    def __post_init__(self) -> None:
        if self.schema_version != "negative-control-report/v2":
            raise ValueError("unsupported negative control report schema")
        _require_hashes(
            self,
            (
                "readiness_spec_hash",
                "candidate_score_hash",
                "label_values_hash",
                "label_validity_hash",
                "scoring_eligibility_hash",
                "observed_results_hash",
                "null_distribution_hash",
                "lag_results_hash",
                "lag_daily_results_hash",
            ),
        )
        if self.readiness_spec.content_hash != self.readiness_spec_hash:
            raise ValueError("negative control specification content differs")
        observed = pd.DataFrame(self.observed_results).copy(deep=True)
        null = pd.DataFrame(self.null_distribution).copy(deep=True)
        lags = pd.DataFrame(self.lag_results).copy(deep=True)
        lag_daily = pd.DataFrame(self.lag_daily_results).copy(deep=True)
        if hash_frame(observed) != self.observed_results_hash:
            raise ValueError("negative control observed result hash differs")
        if hash_frame(null) != self.null_distribution_hash:
            raise ValueError("negative control null distribution hash differs")
        if hash_frame(lags) != self.lag_results_hash:
            raise ValueError("negative control lag results hash differs")
        if hash_frame(lag_daily) != self.lag_daily_results_hash:
            raise ValueError("negative control lag daily results hash differs")
        _optional_finite(self.observed_rank_ic_mean, "observed_rank_ic_mean")
        _optional_probability(self.pseudo_label_pvalue, "pseudo_label_pvalue")
        if self.observed_valid_dates < 0:
            raise ValueError("negative control valid date count is invalid")
        if self.passed is not None and not isinstance(self.passed, bool):
            raise TypeError("negative control passed must be boolean or None")
        object.__setattr__(self, "observed_results", observed)
        object.__setattr__(self, "null_distribution", null)
        object.__setattr__(self, "lag_results", lags)
        object.__setattr__(self, "lag_daily_results", lag_daily)
        _verify_negative_control_report(self, error_type=ValueError)

    @property
    def content_hash(self) -> str:
        self.verify_content()
        return cast(str, hash_json(self.identity_payload()))

    def verify_content(self) -> None:
        if self.readiness_spec.content_hash != self.readiness_spec_hash:
            raise RuntimeError("negative control specification mutated")
        if hash_frame(self.observed_results) != self.observed_results_hash:
            raise RuntimeError("negative control observed results mutated")
        if hash_frame(self.null_distribution) != self.null_distribution_hash:
            raise RuntimeError("negative control null distribution mutated")
        if hash_frame(self.lag_results) != self.lag_results_hash:
            raise RuntimeError("negative control lag results mutated")
        if hash_frame(self.lag_daily_results) != self.lag_daily_results_hash:
            raise RuntimeError("negative control lag daily results mutated")
        _verify_negative_control_report(self, error_type=RuntimeError)

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "readiness_spec_hash": self.readiness_spec_hash,
            "candidate_score_hash": self.candidate_score_hash,
            "label_values_hash": self.label_values_hash,
            "label_validity_hash": self.label_validity_hash,
            "scoring_eligibility_hash": self.scoring_eligibility_hash,
            "observed_rank_ic_mean": self.observed_rank_ic_mean,
            "observed_valid_dates": self.observed_valid_dates,
            "pseudo_label_pvalue": self.pseudo_label_pvalue,
            "observed_results_hash": self.observed_results_hash,
            "null_distribution_hash": self.null_distribution_hash,
            "lag_results_hash": self.lag_results_hash,
            "lag_daily_results_hash": self.lag_daily_results_hash,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class ParameterStabilityReport:
    readiness_spec: ResearchReadinessSpec
    readiness_spec_hash: str
    candidate_score_hash: str
    label_values_hash: str
    label_validity_hash: str
    scoring_eligibility_hash: str
    perturbation_score_hashes: Mapping[str, str]
    scenario_results_hash: str
    daily_results_hash: str
    passed: bool | None
    scenario_results: pd.DataFrame
    daily_results: pd.DataFrame
    schema_version: str = "parameter-stability-report/v2"

    def __post_init__(self) -> None:
        if self.schema_version != "parameter-stability-report/v2":
            raise ValueError("unsupported parameter stability report schema")
        _require_hashes(
            self,
            (
                "readiness_spec_hash",
                "candidate_score_hash",
                "label_values_hash",
                "label_validity_hash",
                "scoring_eligibility_hash",
                "scenario_results_hash",
                "daily_results_hash",
            ),
        )
        if self.readiness_spec.content_hash != self.readiness_spec_hash:
            raise ValueError("parameter stability specification content differs")
        hashes = dict(sorted(self.perturbation_score_hashes.items()))
        if tuple(hashes) != self.readiness_spec.required_perturbation_names:
            raise ValueError("parameter stability perturbation contract differs")
        for name, digest in hashes.items():
            require_sha256(digest, name=f"perturbation score hash:{name}")
        if self.candidate_score_hash in hashes.values() or len(set(hashes.values())) != len(
            hashes
        ):
            raise ValueError("parameter stability perturbations must be distinct")
        frame = pd.DataFrame(self.scenario_results).copy(deep=True)
        daily = pd.DataFrame(self.daily_results).copy(deep=True)
        if hash_frame(frame) != self.scenario_results_hash:
            raise ValueError("parameter stability scenario hash differs")
        if hash_frame(daily) != self.daily_results_hash:
            raise ValueError("parameter stability daily result hash differs")
        if self.passed is not None and not isinstance(self.passed, bool):
            raise TypeError("parameter stability passed must be boolean or None")
        object.__setattr__(self, "perturbation_score_hashes", MappingProxyType(hashes))
        object.__setattr__(self, "scenario_results", frame)
        object.__setattr__(self, "daily_results", daily)
        _verify_parameter_stability_report(self, error_type=ValueError)

    @property
    def content_hash(self) -> str:
        self.verify_content()
        return cast(
            str,
            hash_json(
                {
                    "schema_version": self.schema_version,
                    "readiness_spec_hash": self.readiness_spec_hash,
                    "candidate_score_hash": self.candidate_score_hash,
                    "label_values_hash": self.label_values_hash,
                    "label_validity_hash": self.label_validity_hash,
                    "scoring_eligibility_hash": self.scoring_eligibility_hash,
                    "perturbation_score_hashes": dict(self.perturbation_score_hashes),
                    "scenario_results_hash": self.scenario_results_hash,
                    "daily_results_hash": self.daily_results_hash,
                    "passed": self.passed,
                }
            ),
        )

    def verify_content(self) -> None:
        if self.readiness_spec.content_hash != self.readiness_spec_hash:
            raise RuntimeError("parameter stability specification mutated")
        if hash_frame(self.scenario_results) != self.scenario_results_hash:
            raise RuntimeError("parameter stability scenarios mutated")
        if hash_frame(self.daily_results) != self.daily_results_hash:
            raise RuntimeError("parameter stability daily results mutated")
        _verify_parameter_stability_report(self, error_type=RuntimeError)


@dataclass(frozen=True, slots=True)
class IncrementalBaselineReport:
    readiness_spec: ResearchReadinessSpec
    readiness_spec_hash: str
    candidate_score_hash: str
    baseline_score_hash: str
    label_values_hash: str
    label_validity_hash: str
    scoring_eligibility_hash: str
    daily_results_hash: str
    candidate_rank_ic_mean: float | None
    baseline_rank_ic_mean: float | None
    incremental_rank_ic_mean: float | None
    positive_date_fraction: float | None
    valid_dates: int
    passed: bool | None
    daily_results: pd.DataFrame
    schema_version: str = "incremental-baseline-report/v2"

    def __post_init__(self) -> None:
        if self.schema_version != "incremental-baseline-report/v2":
            raise ValueError("unsupported incremental baseline report schema")
        _require_hashes(
            self,
            (
                "readiness_spec_hash",
                "candidate_score_hash",
                "baseline_score_hash",
                "label_values_hash",
                "label_validity_hash",
                "scoring_eligibility_hash",
                "daily_results_hash",
            ),
        )
        if self.readiness_spec.content_hash != self.readiness_spec_hash:
            raise ValueError("incremental baseline specification content differs")
        frame = pd.DataFrame(self.daily_results).copy(deep=True)
        if hash_frame(frame) != self.daily_results_hash:
            raise ValueError("incremental baseline daily result hash differs")
        for name in (
            "candidate_rank_ic_mean",
            "baseline_rank_ic_mean",
            "incremental_rank_ic_mean",
        ):
            _optional_finite(getattr(self, name), name)
        _optional_probability(self.positive_date_fraction, "positive_date_fraction")
        if self.valid_dates < 0:
            raise ValueError("incremental baseline valid date count is invalid")
        if self.passed is not None and not isinstance(self.passed, bool):
            raise TypeError("incremental baseline passed must be boolean or None")
        object.__setattr__(self, "daily_results", frame)
        _verify_incremental_baseline_report(self, error_type=ValueError)

    @property
    def content_hash(self) -> str:
        self.verify_content()
        return cast(
            str,
            hash_json(
                {
                    "schema_version": self.schema_version,
                    "readiness_spec_hash": self.readiness_spec_hash,
                    "candidate_score_hash": self.candidate_score_hash,
                    "baseline_score_hash": self.baseline_score_hash,
                    "label_values_hash": self.label_values_hash,
                    "label_validity_hash": self.label_validity_hash,
                    "scoring_eligibility_hash": self.scoring_eligibility_hash,
                    "daily_results_hash": self.daily_results_hash,
                    "candidate_rank_ic_mean": self.candidate_rank_ic_mean,
                    "baseline_rank_ic_mean": self.baseline_rank_ic_mean,
                    "incremental_rank_ic_mean": self.incremental_rank_ic_mean,
                    "positive_date_fraction": self.positive_date_fraction,
                    "valid_dates": self.valid_dates,
                    "passed": self.passed,
                }
            ),
        )

    def verify_content(self) -> None:
        if self.readiness_spec.content_hash != self.readiness_spec_hash:
            raise RuntimeError("incremental baseline specification mutated")
        if hash_frame(self.daily_results) != self.daily_results_hash:
            raise RuntimeError("incremental baseline results mutated")
        _verify_incremental_baseline_report(self, error_type=RuntimeError)


@dataclass(frozen=True, slots=True)
class ResearchReadinessBundle:
    readiness_spec_hash: str
    candidate_evidence_hash: str
    baseline_evidence_hash: str
    outer_validation_receipt_hash: str
    negative_control: NegativeControlReport
    parameter_stability: ParameterStabilityReport
    incremental_baseline: IncrementalBaselineReport
    verdict: DLReadinessVerdict | str
    reason_codes: tuple[str, ...]
    research_only: bool = True
    production_ready: bool = False
    schema_version: str = "research-readiness-bundle/v3"

    def __post_init__(self) -> None:
        if self.schema_version != "research-readiness-bundle/v3":
            raise ValueError("unsupported research readiness bundle schema")
        require_sha256(self.readiness_spec_hash, name="readiness spec hash")
        for name in (
            "candidate_evidence_hash",
            "baseline_evidence_hash",
            "outer_validation_receipt_hash",
        ):
            require_sha256(str(getattr(self, name)), name=name)
        for report in (
            self.negative_control,
            self.parameter_stability,
            self.incremental_baseline,
        ):
            if report.readiness_spec_hash != self.readiness_spec_hash:
                raise ValueError("research readiness report specification differs")
        for report in (self.parameter_stability, self.incremental_baseline):
            if (
                report.candidate_score_hash
                != self.negative_control.candidate_score_hash
                or report.label_values_hash != self.negative_control.label_values_hash
                or report.label_validity_hash
                != self.negative_control.label_validity_hash
                or report.scoring_eligibility_hash
                != self.negative_control.scoring_eligibility_hash
            ):
                raise ValueError("research readiness report input lineage differs")
        verdict = DLReadinessVerdict(self.verdict)
        object.__setattr__(self, "verdict", verdict)
        reasons = tuple(self.reason_codes)
        if not reasons or reasons != tuple(sorted(set(reasons))):
            raise ValueError("research readiness reasons must be sorted and unique")
        object.__setattr__(self, "reason_codes", reasons)
        expected_verdict, expected_reasons = _derive_readiness_decision(
            self.negative_control.passed,
            self.parameter_stability.passed,
            self.incremental_baseline.passed,
        )
        if verdict is not expected_verdict or reasons != expected_reasons:
            raise ValueError("research readiness decision differs from report evidence")
        if self.research_only is not True or self.production_ready is not False:
            raise ValueError("research readiness bundle is research-only")

    @property
    def content_hash(self) -> str:
        self.verify_content()
        verdict = self.verdict
        if not isinstance(verdict, DLReadinessVerdict):  # pragma: no cover
            raise RuntimeError("research readiness verdict was not normalized")
        return cast(
            str,
            hash_json(
                {
                    "schema_version": self.schema_version,
                    "readiness_spec_hash": self.readiness_spec_hash,
                    "candidate_evidence_hash": self.candidate_evidence_hash,
                    "baseline_evidence_hash": self.baseline_evidence_hash,
                    "outer_validation_receipt_hash": (
                        self.outer_validation_receipt_hash
                    ),
                    "negative_control_hash": self.negative_control.content_hash,
                    "parameter_stability_hash": self.parameter_stability.content_hash,
                    "incremental_baseline_hash": self.incremental_baseline.content_hash,
                    "verdict": verdict.value,
                    "reason_codes": list(self.reason_codes),
                    "research_only": self.research_only,
                    "production_ready": self.production_ready,
                }
            ),
        )

    def verify_content(self) -> None:
        for report in (
            self.negative_control,
            self.parameter_stability,
            self.incremental_baseline,
        ):
            report.verify_content()
            if report.readiness_spec_hash != self.readiness_spec_hash:
                raise RuntimeError("research readiness report lineage mutated")
        for report in (self.parameter_stability, self.incremental_baseline):
            if (
                report.candidate_score_hash
                != self.negative_control.candidate_score_hash
                or report.label_values_hash != self.negative_control.label_values_hash
                or report.label_validity_hash
                != self.negative_control.label_validity_hash
                or report.scoring_eligibility_hash
                != self.negative_control.scoring_eligibility_hash
            ):
                raise RuntimeError("research readiness input lineage mutated")
        expected_verdict, expected_reasons = _derive_readiness_decision(
            self.negative_control.passed,
            self.parameter_stability.passed,
            self.incremental_baseline.passed,
        )
        if self.verdict is not expected_verdict or self.reason_codes != expected_reasons:
            raise RuntimeError("research readiness decision mutated")


class ResearchReadinessRunner:
    """Decide whether traditional-model evidence justifies DL research.

    This gate never authorizes production or final hidden-OOS evaluation.  It
    only answers whether a more expensive model family is worth researching.
    """

    def run(
        self,
        spec: ResearchReadinessSpec,
        *,
        candidate_evidence_hash: str,
        baseline_evidence_hash: str,
        outer_validation_receipt_hash: str,
        candidate_scores: pd.DataFrame,
        baseline_scores: pd.DataFrame,
        labels: pd.DataFrame,
        label_validity: pd.DataFrame,
        scoring_eligibility: pd.DataFrame,
        perturbation_scores: Mapping[str, pd.DataFrame],
    ) -> ResearchReadinessBundle:
        for name, digest in (
            ("candidate_evidence_hash", candidate_evidence_hash),
            ("baseline_evidence_hash", baseline_evidence_hash),
            ("outer_validation_receipt_hash", outer_validation_receipt_hash),
        ):
            require_sha256(digest, name=name)
        candidate, baseline, target, validity, eligibility = _prepare_axes(
            candidate_scores,
            baseline_scores,
            labels,
            label_validity,
            scoring_eligibility,
        )
        perturbations = _prepare_perturbations(
            perturbation_scores,
            spec=spec,
            reference=candidate,
            eligibility=eligibility,
        )
        negative = _negative_controls(
            spec,
            candidate=candidate,
            labels=target,
            validity=validity,
            eligibility=eligibility,
        )
        stability = _parameter_stability(
            spec,
            candidate=candidate,
            perturbations=perturbations,
            labels=target,
            validity=validity,
            eligibility=eligibility,
        )
        incremental = _incremental_baseline(
            spec,
            candidate=candidate,
            baseline=baseline,
            labels=target,
            validity=validity,
            eligibility=eligibility,
        )
        verdict, reasons = _derive_readiness_decision(
            negative.passed,
            stability.passed,
            incremental.passed,
        )
        return ResearchReadinessBundle(
            readiness_spec_hash=spec.content_hash,
            candidate_evidence_hash=candidate_evidence_hash,
            baseline_evidence_hash=baseline_evidence_hash,
            outer_validation_receipt_hash=outer_validation_receipt_hash,
            negative_control=negative,
            parameter_stability=stability,
            incremental_baseline=incremental,
            verdict=verdict,
            reason_codes=reasons,
        )


def _prepare_axes(
    candidate_scores: pd.DataFrame,
    baseline_scores: pd.DataFrame,
    labels: pd.DataFrame,
    label_validity: pd.DataFrame,
    scoring_eligibility: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = tuple(
        pd.DataFrame(value).copy(deep=True)
        for value in (
            candidate_scores,
            baseline_scores,
            labels,
            label_validity,
            scoring_eligibility,
        )
    )
    candidate, baseline, target, validity, eligibility = frames
    if any(
        not frame.index.equals(candidate.index)
        or not frame.columns.equals(candidate.columns)
        for frame in frames[1:]
    ):
        raise ValueError("research readiness axes differ")
    if (
        not candidate.index.is_unique
        or not candidate.columns.is_unique
        or not candidate.index.is_monotonic_increasing
    ):
        raise ValueError("research readiness axes must be unique and time sorted")
    index = pd.DatetimeIndex(candidate.index)
    if index.tz is None:
        raise ValueError("research readiness timestamps must be timezone-aware")
    _require_strict_boolean_mask(validity, name="label_validity")
    _require_strict_boolean_mask(eligibility, name="scoring_eligibility")
    candidate_finite = np.isfinite(candidate)
    baseline_finite = np.isfinite(baseline)
    if not (candidate_finite.eq(baseline_finite) | ~eligibility).all().all():
        raise ValueError("candidate and baseline prediction membership differs")
    return candidate, baseline, target, validity, eligibility


def _prepare_perturbations(
    values: Mapping[str, pd.DataFrame],
    *,
    spec: ResearchReadinessSpec,
    reference: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> Mapping[str, pd.DataFrame]:
    if any(not isinstance(name, str) for name in values):
        raise TypeError("parameter perturbation names must be strings")
    supplied_names = tuple(sorted(values))
    if supplied_names != spec.required_perturbation_names:
        raise ValueError("parameter perturbation set differs from registered contract")
    output: dict[str, pd.DataFrame] = {}
    reference_finite = np.isfinite(reference)
    reference_hash = hash_frame(reference)
    observed_hashes: set[str] = set()
    active = eligibility.to_numpy(dtype=bool) & reference_finite.to_numpy(dtype=bool)
    reference_active = reference.to_numpy(dtype=float)[active]
    observed_active_values: list[NDArray[np.float64]] = []
    for name, raw in sorted(values.items()):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("parameter perturbation name is invalid")
        frame = pd.DataFrame(raw).copy(deep=True)
        if not frame.index.equals(reference.index) or not frame.columns.equals(
            reference.columns
        ):
            raise ValueError(f"parameter perturbation axes differ:{name}")
        if not (np.isfinite(frame).eq(reference_finite) | ~eligibility).all().all():
            raise ValueError(f"parameter perturbation membership differs:{name}")
        digest = hash_frame(frame)
        if digest == reference_hash:
            raise ValueError(f"parameter perturbation clones candidate:{name}")
        active_values = frame.to_numpy(dtype=float)[active]
        if len(reference_active) and np.array_equal(active_values, reference_active):
            raise ValueError(f"parameter perturbation has no active score change:{name}")
        if digest in observed_hashes:
            raise ValueError(f"parameter perturbation duplicates another scenario:{name}")
        if any(
            np.array_equal(active_values, previous)
            for previous in observed_active_values
        ):
            raise ValueError(
                f"parameter perturbation duplicates active scenario values:{name}"
            )
        observed_hashes.add(digest)
        observed_active_values.append(active_values.copy())
        output[name] = frame
    return MappingProxyType(output)


def _negative_controls(
    spec: ResearchReadinessSpec,
    *,
    candidate: pd.DataFrame,
    labels: pd.DataFrame,
    validity: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> NegativeControlReport:
    observed = _daily_rank_ic(
        candidate,
        labels,
        validity,
        eligibility,
        minimum=spec.minimum_cross_sectional_observations,
    )
    observed_mean = _frame_mean(observed, "rank_ic")
    rng = np.random.default_rng(spec.random_seed)
    null_values: list[float] = []
    for _ in range(spec.pseudo_label_trials):
        pseudo = labels.copy(deep=True)
        for timestamp in labels.index:
            membership = (
                eligibility.loc[timestamp]
                & validity.loc[timestamp]
                & np.isfinite(candidate.loc[timestamp])
                & np.isfinite(labels.loc[timestamp])
            )
            values = pseudo.loc[timestamp, membership].to_numpy(dtype=float).copy()
            rng.shuffle(values)
            pseudo.loc[timestamp, membership] = values
        daily = _daily_rank_ic(
            candidate,
            pseudo,
            validity,
            eligibility,
            minimum=spec.minimum_cross_sectional_observations,
        )
        value = _frame_mean(daily, "rank_ic")
        null_values.append(np.nan if value is None else value)
    null = pd.DataFrame({"rank_ic_mean": null_values})
    finite_null = pd.to_numeric(null["rank_ic_mean"], errors="coerce").dropna()
    pvalue: float | None = None
    if observed_mean is not None and len(finite_null):
        pvalue = float(
            (1 + int((finite_null.abs() >= abs(observed_mean)).sum()))
            / (len(finite_null) + 1)
        )
    lag_rows: list[dict[str, object]] = []
    lag_daily_by_session: dict[int, pd.DataFrame] = {}
    for lag in spec.signal_lag_sessions:
        daily = _daily_rank_ic(
            candidate.shift(lag),
            labels,
            validity,
            eligibility,
            minimum=spec.minimum_cross_sectional_observations,
        )
        lag_daily_by_session[lag] = daily
        lag_mean = _frame_mean(daily, "rank_ic")
        lag_rows.append(
            {
                "lag_sessions": lag,
                "valid_dates": len(daily),
                "rank_ic_mean": lag_mean,
                "absolute_gap_from_observed": (
                    None
                    if observed_mean is None or lag_mean is None
                    else abs(observed_mean) - abs(lag_mean)
                ),
            }
        )
    lags = pd.DataFrame(lag_rows).set_index("lag_sessions")
    lag_daily = _combine_scenario_daily(
        lag_daily_by_session,
        outer_name="lag_sessions",
    )
    lag_values = pd.to_numeric(lags["rank_ic_mean"], errors="coerce").dropna()
    sufficient = (
        observed_mean is not None
        and len(observed) >= spec.minimum_valid_dates
        and pvalue is not None
        and len(finite_null) == spec.pseudo_label_trials
        and len(lag_values) == len(spec.signal_lag_sessions)
    )
    passed: bool | None = None
    if sufficient and observed_mean is not None and pvalue is not None:
        lag_ok = all(
            abs(observed_mean) - abs(float(value)) >= spec.minimum_lag_absolute_gap
            for value in lag_values
        )
        passed = bool(
            observed_mean >= spec.minimum_observed_rank_ic
            and pvalue <= spec.maximum_pseudo_label_pvalue
            and lag_ok
        )
    return NegativeControlReport(
        readiness_spec=spec,
        readiness_spec_hash=spec.content_hash,
        candidate_score_hash=hash_frame(candidate),
        label_values_hash=hash_frame(labels),
        label_validity_hash=hash_frame(validity),
        scoring_eligibility_hash=hash_frame(eligibility),
        observed_rank_ic_mean=observed_mean,
        observed_valid_dates=len(observed),
        pseudo_label_pvalue=pvalue,
        observed_results_hash=hash_frame(observed),
        null_distribution_hash=hash_frame(null),
        lag_results_hash=hash_frame(lags),
        lag_daily_results_hash=hash_frame(lag_daily),
        passed=passed,
        observed_results=observed,
        null_distribution=null,
        lag_results=lags,
        lag_daily_results=lag_daily,
    )


def _parameter_stability(
    spec: ResearchReadinessSpec,
    *,
    candidate: pd.DataFrame,
    perturbations: Mapping[str, pd.DataFrame],
    labels: pd.DataFrame,
    validity: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> ParameterStabilityReport:
    base_daily = _daily_rank_ic(
        candidate,
        labels,
        validity,
        eligibility,
        minimum=spec.minimum_cross_sectional_observations,
    )
    base_mean = _frame_mean(base_daily, "rank_ic")
    rows: list[dict[str, object]] = []
    daily_by_scenario: dict[str, pd.DataFrame] = {"__candidate__": base_daily}
    for name, frame in perturbations.items():
        daily = _daily_rank_ic(
            frame,
            labels,
            validity,
            eligibility,
            minimum=spec.minimum_cross_sectional_observations,
        )
        daily_by_scenario[name] = daily
        value = _frame_mean(daily, "rank_ic")
        retention = (
            None
            if base_mean is None or base_mean == 0.0 or value is None
            else value / base_mean
        )
        rows.append(
            {
                "scenario": name,
                "valid_dates": len(daily),
                "rank_ic_mean": value,
                "retention_ratio": retention,
                "sign_consistent": (
                    None
                    if base_mean is None or value is None
                    else bool(base_mean * value > 0.0)
                ),
            }
        )
    scenarios = pd.DataFrame(rows).set_index("scenario")
    daily_results = _combine_scenario_daily(
        daily_by_scenario,
        outer_name="scenario",
    )
    complete = (
        base_mean is not None
        and len(base_daily) >= spec.minimum_valid_dates
        and all(
            cast(int, row["valid_dates"]) >= spec.minimum_valid_dates
            and row["retention_ratio"] is not None
            and row["sign_consistent"] is not None
            for row in rows
        )
    )
    passed: bool | None = None
    if complete and base_mean is not None:
        passed = bool(
            base_mean >= spec.minimum_observed_rank_ic
            and all(
                bool(row["sign_consistent"])
                and float(cast(float, row["retention_ratio"]))
                >= spec.minimum_stability_retention
                for row in rows
            )
        )
    return ParameterStabilityReport(
        readiness_spec=spec,
        readiness_spec_hash=spec.content_hash,
        candidate_score_hash=hash_frame(candidate),
        label_values_hash=hash_frame(labels),
        label_validity_hash=hash_frame(validity),
        scoring_eligibility_hash=hash_frame(eligibility),
        perturbation_score_hashes={
            name: hash_frame(frame) for name, frame in perturbations.items()
        },
        scenario_results_hash=hash_frame(scenarios),
        daily_results_hash=hash_frame(daily_results),
        passed=passed,
        scenario_results=scenarios,
        daily_results=daily_results,
    )


def _incremental_baseline(
    spec: ResearchReadinessSpec,
    *,
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    labels: pd.DataFrame,
    validity: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> IncrementalBaselineReport:
    candidate_daily = _daily_rank_ic(
        candidate,
        labels,
        validity,
        eligibility,
        minimum=spec.minimum_cross_sectional_observations,
    ).rename(
        columns={
            "observation_count": "candidate_observation_count",
            "rank_ic": "candidate_rank_ic",
        }
    )
    baseline_daily = _daily_rank_ic(
        baseline,
        labels,
        validity,
        eligibility,
        minimum=spec.minimum_cross_sectional_observations,
    ).rename(
        columns={
            "observation_count": "baseline_observation_count",
            "rank_ic": "baseline_rank_ic",
        }
    )
    daily = candidate_daily.join(baseline_daily, how="inner")
    if len(daily):
        if not (
            daily["candidate_observation_count"] == daily["baseline_observation_count"]
        ).all():
            raise ValueError("incremental baseline scoring membership differs")
        daily["incremental_rank_ic"] = (
            daily["candidate_rank_ic"] - daily["baseline_rank_ic"]
        )
    else:
        daily["incremental_rank_ic"] = pd.Series(dtype=float)
    candidate_mean = _frame_mean(daily, "candidate_rank_ic")
    baseline_mean = _frame_mean(daily, "baseline_rank_ic")
    incremental_mean = _frame_mean(daily, "incremental_rank_ic")
    positive_fraction = (
        None if not len(daily) else float((daily["incremental_rank_ic"] > 0.0).mean())
    )
    complete = (
        len(daily) >= spec.minimum_valid_dates
        and candidate_mean is not None
        and baseline_mean is not None
        and incremental_mean is not None
        and positive_fraction is not None
    )
    passed: bool | None = None
    if (
        complete
        and candidate_mean is not None
        and incremental_mean is not None
        and positive_fraction is not None
    ):
        passed = bool(
            candidate_mean >= spec.minimum_observed_rank_ic
            and incremental_mean >= spec.minimum_incremental_rank_ic
            and positive_fraction >= spec.minimum_incremental_positive_date_fraction
        )
    return IncrementalBaselineReport(
        readiness_spec=spec,
        readiness_spec_hash=spec.content_hash,
        candidate_score_hash=hash_frame(candidate),
        baseline_score_hash=hash_frame(baseline),
        label_values_hash=hash_frame(labels),
        label_validity_hash=hash_frame(validity),
        scoring_eligibility_hash=hash_frame(eligibility),
        daily_results_hash=hash_frame(daily),
        candidate_rank_ic_mean=candidate_mean,
        baseline_rank_ic_mean=baseline_mean,
        incremental_rank_ic_mean=incremental_mean,
        positive_date_fraction=positive_fraction,
        valid_dates=len(daily),
        passed=passed,
        daily_results=daily,
    )


def _daily_rank_ic(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    validity: pd.DataFrame,
    eligibility: pd.DataFrame,
    *,
    minimum: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for timestamp in scores.index:
        membership = (
            eligibility.loc[timestamp]
            & validity.loc[timestamp]
            & np.isfinite(scores.loc[timestamp])
            & np.isfinite(labels.loc[timestamp])
        )
        count = int(membership.sum())
        if count < minimum:
            continue
        correlation = scores.loc[timestamp, membership].corr(
            labels.loc[timestamp, membership], method="spearman"
        )
        if np.isfinite(correlation):
            rows.append(
                {
                    "signal_timestamp": timestamp,
                    "observation_count": count,
                    "rank_ic": float(correlation),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["observation_count", "rank_ic"],
            index=pd.DatetimeIndex([], tz=scores.index.tz, name="signal_timestamp"),
        )
    return pd.DataFrame(rows).set_index("signal_timestamp")


def _frame_mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame or not len(frame):
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if not len(values) else float(values.mean())


def _derive_readiness_decision(
    negative_control: bool | None,
    parameter_stability: bool | None,
    incremental_baseline: bool | None,
) -> tuple[DLReadinessVerdict, tuple[str, ...]]:
    states = (negative_control, parameter_stability, incremental_baseline)
    reasons: list[str] = []
    for state, prefix in zip(
        states,
        ("negative_control", "parameter_stability", "incremental_baseline"),
        strict=True,
    ):
        if state is not True:
            reasons.append(f"{prefix}_{'inconclusive' if state is None else 'failed'}")
    if any(state is False for state in states):
        verdict = DLReadinessVerdict.NO_GO
    elif any(state is None for state in states):
        verdict = DLReadinessVerdict.INCONCLUSIVE
    elif all(state is True for state in states):
        verdict = DLReadinessVerdict.GO_RESEARCH_ONLY
        reasons.append("traditional_model_evidence_passed")
    else:  # pragma: no cover - report constructors enforce the tri-state domain.
        raise RuntimeError("research readiness state is invalid")
    return verdict, tuple(sorted(reasons))


def _require_strict_boolean_mask(frame: pd.DataFrame, *, name: str) -> None:
    if any(dtype != np.dtype(bool) for dtype in frame.dtypes):
        raise TypeError(f"{name} must use exact non-nullable bool dtype")
    if frame.isna().any().any():  # Defensive for unusual extension/container inputs.
        raise ValueError(f"{name} must not contain missing values")


def _combine_scenario_daily(
    values: Mapping[_ScenarioKey, pd.DataFrame],
    *,
    outer_name: str,
) -> pd.DataFrame:
    return pd.concat(
        tuple(values.values()),
        keys=tuple(values),
        names=(outer_name, "signal_timestamp"),
    )


def _raise_semantic(error_type: type[Exception], message: str) -> None:
    raise error_type(message)


def _require_optional_float_equal(
    actual: float | None,
    expected: float | None,
    *,
    name: str,
    error_type: type[Exception],
) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            _raise_semantic(error_type, f"{name} differs from stored evidence")
        return
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-15):
        _raise_semantic(error_type, f"{name} differs from stored evidence")


def _require_tristate_equal(
    actual: bool | None,
    expected: bool | None,
    *,
    name: str,
    error_type: type[Exception],
) -> None:
    if actual is not expected:
        _raise_semantic(error_type, f"{name} differs from stored evidence")


def _validate_daily_rank_ic_evidence(
    frame: pd.DataFrame,
    *,
    name: str,
    error_type: type[Exception],
) -> None:
    if tuple(frame.columns) != ("observation_count", "rank_ic"):
        _raise_semantic(error_type, f"{name} columns are invalid")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        _raise_semantic(error_type, f"{name} index is invalid")
    if not len(frame):
        return
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        _raise_semantic(error_type, f"{name} timestamps must be timezone-aware")
    if not pd.api.types.is_integer_dtype(frame["observation_count"].dtype):
        _raise_semantic(error_type, f"{name} observation counts must be integers")
    counts = frame["observation_count"].to_numpy(dtype=np.int64)
    rank_ic = frame["rank_ic"].to_numpy(dtype=float)
    if np.any(counts < 2) or not np.isfinite(rank_ic).all():
        _raise_semantic(error_type, f"{name} contains invalid daily evidence")
    if np.any(np.abs(rank_ic) > 1.0 + 1e-12):
        _raise_semantic(error_type, f"{name} rank IC lies outside [-1, 1]")


def _validate_multi_scenario_daily_evidence(
    frame: pd.DataFrame,
    *,
    outer_name: str,
    allowed_keys: set[object],
    error_type: type[Exception],
) -> None:
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels != 2:
        _raise_semantic(error_type, f"{outer_name} daily evidence index is invalid")
    if tuple(frame.index.names) != (outer_name, "signal_timestamp"):
        _raise_semantic(error_type, f"{outer_name} daily evidence names are invalid")
    if not frame.index.is_unique:
        _raise_semantic(error_type, f"{outer_name} daily evidence is not unique")
    present = set(frame.index.get_level_values(outer_name).unique())
    if not present.issubset(allowed_keys):
        _raise_semantic(error_type, f"{outer_name} daily evidence has extra scenarios")
    for key in present:
        daily = frame.xs(key, level=outer_name, drop_level=True)
        _validate_daily_rank_ic_evidence(
            daily,
            name=f"{outer_name}:{key}",
            error_type=error_type,
        )


def _daily_for_scenario(
    frame: pd.DataFrame,
    *,
    outer_name: str,
    key: object,
) -> pd.DataFrame:
    present = set(frame.index.get_level_values(outer_name).unique())
    if key in present:
        return frame.xs(key, level=outer_name, drop_level=True)
    return pd.DataFrame(columns=("observation_count", "rank_ic"))


def _verify_negative_control_report(
    report: NegativeControlReport,
    *,
    error_type: type[Exception],
) -> None:
    spec = report.readiness_spec
    _validate_daily_rank_ic_evidence(
        report.observed_results,
        name="negative control observed results",
        error_type=error_type,
    )
    if tuple(report.null_distribution.columns) != ("rank_ic_mean",):
        _raise_semantic(error_type, "negative control null distribution columns differ")
    if len(report.null_distribution) != spec.pseudo_label_trials:
        _raise_semantic(error_type, "negative control pseudo trial count differs")
    if not report.null_distribution.index.equals(
        pd.RangeIndex(spec.pseudo_label_trials)
    ):
        _raise_semantic(error_type, "negative control pseudo trial index differs")
    null_values = report.null_distribution["rank_ic_mean"]
    if not pd.api.types.is_numeric_dtype(null_values.dtype):
        _raise_semantic(error_type, "negative control pseudo trials are not numeric")
    numeric_null = null_values.to_numpy(dtype=float)
    if np.isinf(numeric_null).any():
        _raise_semantic(error_type, "negative control pseudo trials contain infinity")
    finite_numeric_null = numeric_null[np.isfinite(numeric_null)]
    if np.any(np.abs(finite_numeric_null) > 1.0 + 1e-12):
        _raise_semantic(error_type, "negative control pseudo trial IC is invalid")
    allowed_lags = set(spec.signal_lag_sessions)
    _validate_multi_scenario_daily_evidence(
        report.lag_daily_results,
        outer_name="lag_sessions",
        allowed_keys=set(allowed_lags),
        error_type=error_type,
    )
    observed_mean = _frame_mean(report.observed_results, "rank_ic")
    _require_optional_float_equal(
        report.observed_rank_ic_mean,
        observed_mean,
        name="negative control observed rank IC mean",
        error_type=error_type,
    )
    observed_valid_dates = len(report.observed_results)
    if report.observed_valid_dates != observed_valid_dates:
        _raise_semantic(error_type, "negative control valid date count differs")
    finite_null = pd.Series(numeric_null).dropna()
    pvalue: float | None = None
    if observed_mean is not None and len(finite_null):
        pvalue = float(
            (1 + int((finite_null.abs() >= abs(observed_mean)).sum()))
            / (len(finite_null) + 1)
        )
    _require_optional_float_equal(
        report.pseudo_label_pvalue,
        pvalue,
        name="negative control pseudo-label p-value",
        error_type=error_type,
    )
    lag_rows: list[dict[str, object]] = []
    for lag in spec.signal_lag_sessions:
        daily = _daily_for_scenario(
            report.lag_daily_results,
            outer_name="lag_sessions",
            key=lag,
        )
        lag_mean = _frame_mean(daily, "rank_ic")
        lag_rows.append(
            {
                "lag_sessions": lag,
                "valid_dates": len(daily),
                "rank_ic_mean": lag_mean,
                "absolute_gap_from_observed": (
                    None
                    if observed_mean is None or lag_mean is None
                    else abs(observed_mean) - abs(lag_mean)
                ),
            }
        )
    expected_lags = pd.DataFrame(lag_rows).set_index("lag_sessions")
    if hash_frame(report.lag_results) != hash_frame(expected_lags):
        _raise_semantic(error_type, "negative control lag summaries differ")
    lag_values = pd.to_numeric(expected_lags["rank_ic_mean"], errors="coerce").dropna()
    sufficient = (
        observed_mean is not None
        and observed_valid_dates >= spec.minimum_valid_dates
        and pvalue is not None
        and len(finite_null) == spec.pseudo_label_trials
        and len(lag_values) == len(spec.signal_lag_sessions)
    )
    passed: bool | None = None
    if sufficient and observed_mean is not None and pvalue is not None:
        lag_ok = all(
            abs(observed_mean) - abs(float(value)) >= spec.minimum_lag_absolute_gap
            for value in lag_values
        )
        passed = bool(
            observed_mean >= spec.minimum_observed_rank_ic
            and pvalue <= spec.maximum_pseudo_label_pvalue
            and lag_ok
        )
    _require_tristate_equal(
        report.passed,
        passed,
        name="negative control passed",
        error_type=error_type,
    )


def _verify_parameter_stability_report(
    report: ParameterStabilityReport,
    *,
    error_type: type[Exception],
) -> None:
    spec = report.readiness_spec
    required = spec.required_perturbation_names
    if tuple(report.perturbation_score_hashes) != required:
        _raise_semantic(error_type, "parameter stability perturbation set differs")
    hashes = tuple(report.perturbation_score_hashes.values())
    if report.candidate_score_hash in hashes or len(set(hashes)) != len(hashes):
        _raise_semantic(error_type, "parameter stability perturbations are not distinct")
    allowed: set[object] = {"__candidate__", *required}
    _validate_multi_scenario_daily_evidence(
        report.daily_results,
        outer_name="scenario",
        allowed_keys=allowed,
        error_type=error_type,
    )
    base_daily = _daily_for_scenario(
        report.daily_results,
        outer_name="scenario",
        key="__candidate__",
    )
    base_mean = _frame_mean(base_daily, "rank_ic")
    rows: list[dict[str, object]] = []
    for name in required:
        daily = _daily_for_scenario(
            report.daily_results,
            outer_name="scenario",
            key=name,
        )
        value = _frame_mean(daily, "rank_ic")
        retention = (
            None
            if base_mean is None or base_mean == 0.0 or value is None
            else value / base_mean
        )
        rows.append(
            {
                "scenario": name,
                "valid_dates": len(daily),
                "rank_ic_mean": value,
                "retention_ratio": retention,
                "sign_consistent": (
                    None
                    if base_mean is None or value is None
                    else bool(base_mean * value > 0.0)
                ),
            }
        )
    expected_scenarios = pd.DataFrame(rows).set_index("scenario")
    if hash_frame(report.scenario_results) != hash_frame(expected_scenarios):
        _raise_semantic(error_type, "parameter stability summaries differ")
    complete = (
        base_mean is not None
        and len(base_daily) >= spec.minimum_valid_dates
        and all(
            cast(int, row["valid_dates"]) >= spec.minimum_valid_dates
            and row["retention_ratio"] is not None
            and row["sign_consistent"] is not None
            for row in rows
        )
    )
    passed: bool | None = None
    if complete and base_mean is not None:
        passed = bool(
            base_mean >= spec.minimum_observed_rank_ic
            and all(
                bool(row["sign_consistent"])
                and float(cast(float, row["retention_ratio"]))
                >= spec.minimum_stability_retention
                for row in rows
            )
        )
    _require_tristate_equal(
        report.passed,
        passed,
        name="parameter stability passed",
        error_type=error_type,
    )


def _verify_incremental_baseline_report(
    report: IncrementalBaselineReport,
    *,
    error_type: type[Exception],
) -> None:
    spec = report.readiness_spec
    daily = report.daily_results
    expected_columns = (
        "candidate_observation_count",
        "candidate_rank_ic",
        "baseline_observation_count",
        "baseline_rank_ic",
        "incremental_rank_ic",
    )
    if tuple(daily.columns) != expected_columns:
        _raise_semantic(error_type, "incremental baseline daily columns differ")
    if not daily.index.is_unique or not daily.index.is_monotonic_increasing:
        _raise_semantic(error_type, "incremental baseline daily index is invalid")
    if len(daily):
        if not isinstance(daily.index, pd.DatetimeIndex) or daily.index.tz is None:
            _raise_semantic(error_type, "incremental baseline timestamps are invalid")
        for column in (
            "candidate_observation_count",
            "baseline_observation_count",
        ):
            if not pd.api.types.is_integer_dtype(daily[column].dtype):
                _raise_semantic(error_type, f"{column} must contain integers")
        if not (
            daily["candidate_observation_count"]
            == daily["baseline_observation_count"]
        ).all():
            _raise_semantic(error_type, "incremental baseline memberships differ")
        if np.any(
            daily["candidate_observation_count"].to_numpy(dtype=np.int64) < 2
        ):
            _raise_semantic(error_type, "incremental baseline counts are invalid")
        candidate_values = daily["candidate_rank_ic"].to_numpy(dtype=float)
        baseline_values = daily["baseline_rank_ic"].to_numpy(dtype=float)
        if (
            not np.isfinite(candidate_values).all()
            or not np.isfinite(baseline_values).all()
            or np.any(np.abs(candidate_values) > 1.0 + 1e-12)
            or np.any(np.abs(baseline_values) > 1.0 + 1e-12)
        ):
            _raise_semantic(error_type, "incremental baseline rank IC is invalid")
        expected_increment = (
            daily["candidate_rank_ic"] - daily["baseline_rank_ic"]
        ).to_numpy(dtype=float)
        stored_increment = daily["incremental_rank_ic"].to_numpy(dtype=float)
        if not np.isfinite(expected_increment).all() or not np.isfinite(
            stored_increment
        ).all():
            _raise_semantic(error_type, "incremental baseline contains non-finite IC")
        if not np.allclose(
            stored_increment,
            expected_increment,
            rtol=0.0,
            atol=1e-15,
        ):
            _raise_semantic(error_type, "incremental baseline daily differences differ")
    candidate_mean = _frame_mean(daily, "candidate_rank_ic")
    baseline_mean = _frame_mean(daily, "baseline_rank_ic")
    incremental_mean = _frame_mean(daily, "incremental_rank_ic")
    positive_fraction = (
        None if not len(daily) else float((daily["incremental_rank_ic"] > 0.0).mean())
    )
    for name, actual, expected in (
        ("candidate rank IC mean", report.candidate_rank_ic_mean, candidate_mean),
        ("baseline rank IC mean", report.baseline_rank_ic_mean, baseline_mean),
        ("incremental rank IC mean", report.incremental_rank_ic_mean, incremental_mean),
        ("positive date fraction", report.positive_date_fraction, positive_fraction),
    ):
        _require_optional_float_equal(
            actual,
            expected,
            name=f"incremental baseline {name}",
            error_type=error_type,
        )
    if report.valid_dates != len(daily):
        _raise_semantic(error_type, "incremental baseline valid date count differs")
    complete = (
        len(daily) >= spec.minimum_valid_dates
        and candidate_mean is not None
        and baseline_mean is not None
        and incremental_mean is not None
        and positive_fraction is not None
    )
    passed: bool | None = None
    if (
        complete
        and candidate_mean is not None
        and incremental_mean is not None
        and positive_fraction is not None
    ):
        passed = bool(
            candidate_mean >= spec.minimum_observed_rank_ic
            and incremental_mean >= spec.minimum_incremental_rank_ic
            and positive_fraction >= spec.minimum_incremental_positive_date_fraction
        )
    _require_tristate_equal(
        report.passed,
        passed,
        name="incremental baseline passed",
        error_type=error_type,
    )


def _require_hashes(value: object, names: tuple[str, ...]) -> None:
    for name in names:
        require_sha256(str(getattr(value, name)), name=name)


def _optional_finite(value: float | None, name: str) -> None:
    if value is not None and not isfinite(float(value)):
        raise ValueError(f"{name} must be finite or None")


def _optional_probability(value: float | None, name: str) -> None:
    if value is not None and (
        not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must lie in [0, 1] or be None")


__all__ = [
    "DLReadinessVerdict",
    "IncrementalBaselineReport",
    "NegativeControlReport",
    "ParameterStabilityReport",
    "ResearchReadinessBundle",
    "ResearchReadinessRunner",
    "ResearchReadinessSpec",
]
