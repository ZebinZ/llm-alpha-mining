from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral, Real
import re
from typing import Mapping

from alpha_research.core.hashing import hash_frame, hash_json, require_sha256
from alpha_research.evaluation import FactorEvaluationReport
from alpha_research.experiments import DataPartition
from alpha_research.market_logic.evidence import (
    CostResilienceCategory,
    CoverageCategory,
    EvidencePartition,
    EvidenceReasonCode,
    EvidenceVerdict,
    EvidenceVisibility,
    LogicEvidenceSummary,
    StabilityCategory,
)
from factor_production.v5.domain.enums import ScoreVisibility


EVIDENCE_CLASSIFICATION_POLICY_SCHEMA = "logic-evidence-classification-policy/v1"
EVIDENCE_DERIVATION_SCHEMA = "logic-evidence-derivation/v1"

_SAFE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")


class EvidenceAdapterFailure(str, Enum):
    """Closed, metric-free failure codes safe to log or display."""

    SOURCE_NOT_ADAPTIVE = "source_not_adaptive"
    VISIBILITY_NOT_LOCAL = "visibility_not_local"
    INVALID_LINEAGE = "invalid_lineage"
    ARTIFACT_BINDING_FAILED = "artifact_binding_failed"
    ARTIFACT_INTEGRITY_FAILED = "artifact_integrity_failed"
    UNSUPPORTED_ARTIFACT = "unsupported_artifact"


class EvidenceAdapterError(ValueError):
    """Fail-closed adapter error that never embeds source metric values."""

    def __init__(self, code: EvidenceAdapterFailure) -> None:
        self.code = EvidenceAdapterFailure(code)
        super().__init__(self.code.value)


@dataclass(frozen=True, slots=True)
class EvidenceClassificationPolicy:
    """Versioned thresholds for deterministic categorical classification.

    These values are governance inputs rather than observed performance.  A
    policy's content hash is included in the evidence derivation hash, so a
    threshold change cannot silently reuse evidence produced by an older rule.
    """

    policy_id: str
    version: str
    minimum_valid_ic_dates: int
    stable_rank_ic_positive_ratio: float
    mixed_rank_ic_positive_ratio: float
    supported_rank_ic_mean: float
    mixed_rank_ic_mean: float
    broad_coverage: float
    adequate_coverage: float
    minimum_incremental_rank_ic: float
    schema_version: str = EVIDENCE_CLASSIFICATION_POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != EVIDENCE_CLASSIFICATION_POLICY_SCHEMA:
            raise ValueError("unsupported evidence classification policy schema")
        if not _SAFE_IDENTIFIER.fullmatch(self.policy_id):
            raise ValueError("invalid evidence classification policy identifier")
        if not _SAFE_VERSION.fullmatch(self.version):
            raise ValueError("invalid evidence classification policy version")
        if (
            not isinstance(self.minimum_valid_ic_dates, int)
            or isinstance(self.minimum_valid_ic_dates, bool)
            or self.minimum_valid_ic_dates <= 0
        ):
            raise ValueError("minimum_valid_ic_dates must be a positive integer")
        numeric = (
            self.stable_rank_ic_positive_ratio,
            self.mixed_rank_ic_positive_ratio,
            self.supported_rank_ic_mean,
            self.mixed_rank_ic_mean,
            self.broad_coverage,
            self.adequate_coverage,
            self.minimum_incremental_rank_ic,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in numeric
        ):
            raise ValueError("evidence classification thresholds must be finite")
        if not (
            0.0
            <= float(self.mixed_rank_ic_positive_ratio)
            <= float(self.stable_rank_ic_positive_ratio)
            <= 1.0
        ):
            raise ValueError("directional ratio thresholds are not ordered")
        if float(self.mixed_rank_ic_mean) > float(self.supported_rank_ic_mean):
            raise ValueError("directional mean thresholds are not ordered")
        if not (
            0.0 <= float(self.adequate_coverage) <= float(self.broad_coverage) <= 1.0
        ):
            raise ValueError("coverage thresholds are not ordered")
        if float(self.minimum_incremental_rank_ic) < 0.0:
            raise ValueError("incremental-signal threshold must be non-negative")

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "version": self.version,
            "minimum_valid_ic_dates": self.minimum_valid_ic_dates,
            "stable_rank_ic_positive_ratio": float(self.stable_rank_ic_positive_ratio),
            "mixed_rank_ic_positive_ratio": float(self.mixed_rank_ic_positive_ratio),
            "supported_rank_ic_mean": float(self.supported_rank_ic_mean),
            "mixed_rank_ic_mean": float(self.mixed_rank_ic_mean),
            "broad_coverage": float(self.broad_coverage),
            "adequate_coverage": float(self.adequate_coverage),
            "minimum_incremental_rank_ic": float(self.minimum_incremental_rank_ic),
        }


@dataclass(frozen=True, slots=True)
class LocalAdaptiveValidationEvidenceAdapter:
    """Convert a local validation report into retrieval-safe evidence.

    Only the framework's sealed ``FactorEvaluationReport`` is accepted.  The
    current backtest and robustness report types do not carry an independently
    verifiable data-partition/experiment binding, so accepting them here would
    permit blind results to be relabelled as adaptive validation.  Their
    categories consequently remain ``NOT_ASSESSED`` until those artifacts gain
    equivalent provenance contracts.
    """

    policy: EvidenceClassificationPolicy

    def adapt(
        self,
        artifact: FactorEvaluationReport,
        *,
        evaluation_artifact_hash: str,
        experiment_hash: str,
        logic_hash: str,
        source_partition: DataPartition | str,
        score_visibility: ScoreVisibility | str,
    ) -> LogicEvidenceSummary:
        partition = _adaptive_partition(source_partition)
        visibility = _local_visibility(score_visibility)
        if partition is not DataPartition.VALIDATION:
            raise EvidenceAdapterError(EvidenceAdapterFailure.SOURCE_NOT_ADAPTIVE)
        if visibility is not ScoreVisibility.LOCAL_RESEARCH:
            raise EvidenceAdapterError(EvidenceAdapterFailure.VISIBILITY_NOT_LOCAL)
        if type(artifact) is not FactorEvaluationReport:
            raise EvidenceAdapterError(EvidenceAdapterFailure.UNSUPPORTED_ARTIFACT)

        try:
            artifact_hash = require_sha256(
                str(evaluation_artifact_hash), name="evaluation artifact hash"
            )
            experiment = require_sha256(
                str(experiment_hash), name="evidence experiment hash"
            )
            logic = require_sha256(str(logic_hash), name="evidence logic hash")
        except (TypeError, ValueError) as exc:
            raise EvidenceAdapterError(EvidenceAdapterFailure.INVALID_LINEAGE) from exc

        _verify_evaluation_artifact(artifact)
        if artifact.content_hash != artifact_hash:
            raise EvidenceAdapterError(EvidenceAdapterFailure.ARTIFACT_BINDING_FAILED)

        classification = _classify(artifact.summary, artifact, self.policy)
        derivation_hash = hash_json(
            {
                "schema_version": EVIDENCE_DERIVATION_SCHEMA,
                "source_artifact_hash": artifact_hash,
                "experiment_hash": experiment,
                "logic_hash": logic,
                "policy_hash": self.policy.content_hash,
                "source_partition": partition.value,
                "score_visibility": visibility.value,
            }
        )
        return LogicEvidenceSummary(
            logic_hash=logic,
            experiment_hash=experiment,
            evidence_artifact_hash=derivation_hash,
            source_partition=EvidencePartition.ADAPTIVE_VALIDATION,
            visibility=EvidenceVisibility.ADAPTIVE_VALIDATION,
            verdict=classification.verdict,
            directional_stability=classification.directional_stability,
            regime_stability=StabilityCategory.NOT_ASSESSED,
            cost_resilience=CostResilienceCategory.NOT_ASSESSED,
            coverage=classification.coverage,
            reason_codes=classification.reason_codes,
        )


@dataclass(frozen=True, slots=True)
class _Classification:
    verdict: EvidenceVerdict
    directional_stability: StabilityCategory
    coverage: CoverageCategory
    reason_codes: tuple[EvidenceReasonCode, ...]


def _adaptive_partition(value: DataPartition | str) -> DataPartition:
    try:
        return DataPartition(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceAdapterError(EvidenceAdapterFailure.SOURCE_NOT_ADAPTIVE) from exc


def _local_visibility(value: ScoreVisibility | str) -> ScoreVisibility:
    try:
        return ScoreVisibility(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceAdapterError(EvidenceAdapterFailure.VISIBILITY_NOT_LOCAL) from exc


def _verify_evaluation_artifact(artifact: FactorEvaluationReport) -> None:
    try:
        valid = (
            hash_frame(artifact.per_date) == artifact.per_date_hash
            and hash_frame(artifact.quantile_returns) == artifact.quantile_returns_hash
            and hash_json(dict(artifact.summary)) == artifact.summary_hash
            and hash_frame(artifact.metric_metadata) == artifact.metric_metadata_hash
        )
    except Exception as exc:
        raise EvidenceAdapterError(
            EvidenceAdapterFailure.ARTIFACT_INTEGRITY_FAILED
        ) from exc
    if not valid:
        raise EvidenceAdapterError(EvidenceAdapterFailure.ARTIFACT_INTEGRITY_FAILED)


def _classify(
    summary: Mapping[str, float | int | None],
    artifact: FactorEvaluationReport,
    policy: EvidenceClassificationPolicy,
) -> _Classification:
    valid_dates = _required_count(summary, "valid_ic_date_count")
    rank_ic_mean = _optional_ratio(summary, "rank_ic_mean", signed=True)
    positive_ratio = _optional_ratio(summary, "rank_ic_positive_ratio", signed=False)
    coverage_mean = _optional_ratio(summary, "coverage_mean", signed=False)
    incremental = _optional_ratio(summary, "incremental_rank_ic_mean", signed=True)

    reasons: list[EvidenceReasonCode] = []
    if valid_dates < policy.minimum_valid_ic_dates:
        directional = StabilityCategory.NOT_ASSESSED
        coverage = CoverageCategory.NOT_ASSESSED
        reasons.append(EvidenceReasonCode.INSUFFICIENT_EVIDENCE)
        return _Classification(
            verdict=EvidenceVerdict.INCONCLUSIVE,
            directional_stability=directional,
            coverage=coverage,
            reason_codes=tuple(reasons),
        )
    if rank_ic_mean is None or positive_ratio is None or coverage_mean is None:
        directional = StabilityCategory.NOT_ASSESSED
        coverage = CoverageCategory.NOT_ASSESSED
        reasons.append(EvidenceReasonCode.INSUFFICIENT_EVIDENCE)
        return _Classification(
            verdict=EvidenceVerdict.INCONCLUSIVE,
            directional_stability=directional,
            coverage=coverage,
            reason_codes=tuple(reasons),
        )

    if rank_ic_mean >= float(policy.supported_rank_ic_mean) and positive_ratio >= float(
        policy.stable_rank_ic_positive_ratio
    ):
        directional = StabilityCategory.STABLE
        reasons.append(EvidenceReasonCode.DIRECTION_CONSISTENT)
    elif rank_ic_mean >= float(policy.mixed_rank_ic_mean) and positive_ratio >= float(
        policy.mixed_rank_ic_positive_ratio
    ):
        directional = StabilityCategory.MIXED
        reasons.append(EvidenceReasonCode.DIRECTION_MIXED)
    else:
        directional = StabilityCategory.UNSTABLE
        reasons.append(EvidenceReasonCode.DIRECTION_UNSTABLE)

    if coverage_mean >= float(policy.broad_coverage):
        coverage = CoverageCategory.BROAD
        reasons.append(EvidenceReasonCode.COVERAGE_ADEQUATE)
    elif coverage_mean >= float(policy.adequate_coverage):
        coverage = CoverageCategory.ADEQUATE
        reasons.append(EvidenceReasonCode.COVERAGE_ADEQUATE)
    else:
        coverage = CoverageCategory.SPARSE
        reasons.append(EvidenceReasonCode.COVERAGE_SPARSE)

    redundant = False
    if artifact.reference_factor_hashes:
        if incremental is None:
            reasons.append(EvidenceReasonCode.INSUFFICIENT_EVIDENCE)
        elif incremental >= float(policy.minimum_incremental_rank_ic):
            reasons.append(EvidenceReasonCode.INCREMENTAL_SIGNAL)
        else:
            redundant = True
            reasons.append(EvidenceReasonCode.REDUNDANT_SIGNAL)

    if (
        directional is StabilityCategory.UNSTABLE
        or (coverage is CoverageCategory.SPARSE)
        or redundant
    ):
        verdict = EvidenceVerdict.REJECTED
    elif (
        directional is StabilityCategory.STABLE
        and coverage in {CoverageCategory.BROAD, CoverageCategory.ADEQUATE}
        and EvidenceReasonCode.INSUFFICIENT_EVIDENCE not in reasons
    ):
        verdict = EvidenceVerdict.SUPPORTED
    else:
        verdict = EvidenceVerdict.MIXED
    return _Classification(
        verdict=verdict,
        directional_stability=directional,
        coverage=coverage,
        reason_codes=tuple(reasons),
    )


def _required_count(summary: Mapping[str, float | int | None], name: str) -> int:
    if name not in summary:
        raise EvidenceAdapterError(EvidenceAdapterFailure.ARTIFACT_INTEGRITY_FAILED)
    value = summary[name]
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise EvidenceAdapterError(EvidenceAdapterFailure.ARTIFACT_INTEGRITY_FAILED)
    return int(value)


def _optional_ratio(
    summary: Mapping[str, float | int | None],
    name: str,
    *,
    signed: bool,
) -> float | None:
    if name not in summary:
        raise EvidenceAdapterError(EvidenceAdapterFailure.ARTIFACT_INTEGRITY_FAILED)
    value = summary[name]
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise EvidenceAdapterError(EvidenceAdapterFailure.ARTIFACT_INTEGRITY_FAILED)
    number = float(value)
    lower = -1.0 if signed else 0.0
    if not lower <= number <= 1.0:
        raise EvidenceAdapterError(EvidenceAdapterFailure.ARTIFACT_INTEGRITY_FAILED)
    return number


__all__ = [
    "EVIDENCE_CLASSIFICATION_POLICY_SCHEMA",
    "EVIDENCE_DERIVATION_SCHEMA",
    "EvidenceAdapterError",
    "EvidenceAdapterFailure",
    "EvidenceClassificationPolicy",
    "LocalAdaptiveValidationEvidenceAdapter",
]
