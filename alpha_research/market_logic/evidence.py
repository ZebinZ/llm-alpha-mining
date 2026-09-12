from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from alpha_research.core.hashing import hash_json, require_sha256


LOGIC_EVIDENCE_SCHEMA = "logic-evidence-summary/v1"


class EvidencePartition(str, Enum):
    """Partitions allowed to contribute categorical research evidence.

    Discovery/train observations deliberately do not enter this sidecar.  The
    adaptive memory may learn from adaptive validation.  Blind evaluations and
    teacher feedback are retained only for audit.
    """

    ADAPTIVE_VALIDATION = "adaptive_validation"
    TEST = "test"
    HOLDOUT = "holdout"
    TEACHER = "teacher"


class EvidenceVisibility(str, Enum):
    ADAPTIVE_VALIDATION = "adaptive_validation"
    AUDIT_ONLY = "audit_only"


class EvidenceVerdict(str, Enum):
    SUPPORTED = "supported"
    MIXED = "mixed"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class StabilityCategory(str, Enum):
    STABLE = "stable"
    MIXED = "mixed"
    UNSTABLE = "unstable"
    NOT_ASSESSED = "not_assessed"


class CostResilienceCategory(str, Enum):
    RESILIENT = "resilient"
    BORDERLINE = "borderline"
    FRAGILE = "fragile"
    NOT_ASSESSED = "not_assessed"


class CoverageCategory(str, Enum):
    BROAD = "broad"
    ADEQUATE = "adequate"
    SPARSE = "sparse"
    NOT_ASSESSED = "not_assessed"


class EvidenceReasonCode(str, Enum):
    """Closed vocabulary safe to expose to the adaptive research loop."""

    DIRECTION_CONSISTENT = "direction_consistent"
    DIRECTION_MIXED = "direction_mixed"
    DIRECTION_UNSTABLE = "direction_unstable"
    REGIME_CONSISTENT = "regime_consistent"
    REGIME_SENSITIVE = "regime_sensitive"
    COST_RESILIENT = "cost_resilient"
    COST_SENSITIVE = "cost_sensitive"
    COVERAGE_ADEQUATE = "coverage_adequate"
    COVERAGE_SPARSE = "coverage_sparse"
    INCREMENTAL_SIGNAL = "incremental_signal"
    REDUNDANT_SIGNAL = "redundant_signal"
    ECONOMIC_LOGIC_SUPPORTED = "economic_logic_supported"
    COMPLEXITY_EXCESSIVE = "complexity_excessive"
    DATA_QUALITY_CONCERN = "data_quality_concern"
    TIMING_CONCERN = "timing_concern"
    OVERFIT_CONCERN = "overfit_concern"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    LEAKAGE_GUARD_FAILED = "leakage_guard_failed"
    PROTOCOL_VIOLATION = "protocol_violation"
    EXTERNAL_REVIEW_SUPPORTED = "external_review_supported"
    EXTERNAL_REVIEW_REJECTED = "external_review_rejected"


@dataclass(frozen=True, slots=True)
class LogicEvidenceSummary:
    """Leakage-safe categorical evidence attached to one market logic.

    The fixed schema intentionally has no numeric performance, market-date,
    security, free-text, or arbitrary-metadata fields.  Exact IC, Sharpe,
    returns, dates, and instruments stay in sealed experiment artifacts.
    """

    logic_hash: str
    experiment_hash: str
    evidence_artifact_hash: str
    source_partition: EvidencePartition | str
    visibility: EvidenceVisibility | str
    verdict: EvidenceVerdict | str
    directional_stability: StabilityCategory | str
    regime_stability: StabilityCategory | str
    cost_resilience: CostResilienceCategory | str
    coverage: CoverageCategory | str
    reason_codes: tuple[EvidenceReasonCode | str, ...]
    schema_version: str = LOGIC_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != LOGIC_EVIDENCE_SCHEMA:
            raise ValueError("unsupported LogicEvidenceSummary schema")
        for name in ("logic_hash", "experiment_hash", "evidence_artifact_hash"):
            require_sha256(str(getattr(self, name)), name=f"logic evidence {name}")

        partition = EvidencePartition(self.source_partition)
        visibility = EvidenceVisibility(self.visibility)
        verdict = EvidenceVerdict(self.verdict)
        directional = StabilityCategory(self.directional_stability)
        regime = StabilityCategory(self.regime_stability)
        cost = CostResilienceCategory(self.cost_resilience)
        coverage = CoverageCategory(self.coverage)
        reasons = tuple(EvidenceReasonCode(item) for item in self.reason_codes)
        if not reasons:
            raise ValueError("logic evidence requires at least one closed reason code")
        if len(reasons) != len(set(reasons)):
            raise ValueError("logic evidence reason codes must be unique")
        reasons = tuple(sorted(reasons, key=lambda item: item.value))

        if visibility is EvidenceVisibility.ADAPTIVE_VALIDATION:
            if partition is not EvidencePartition.ADAPTIVE_VALIDATION:
                raise ValueError(
                    "only adaptive-validation evidence may be retrieval-visible"
                )
        elif (
            partition
            in {
                EvidencePartition.TEST,
                EvidencePartition.HOLDOUT,
                EvidencePartition.TEACHER,
            }
            and visibility is not EvidenceVisibility.AUDIT_ONLY
        ):
            raise ValueError("blind or teacher evidence must remain audit-only")

        object.__setattr__(self, "source_partition", partition)
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "verdict", verdict)
        object.__setattr__(self, "directional_stability", directional)
        object.__setattr__(self, "regime_stability", regime)
        object.__setattr__(self, "cost_resilience", cost)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "reason_codes", reasons)

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "logic_hash": self.logic_hash,
            "experiment_hash": self.experiment_hash,
            "evidence_artifact_hash": self.evidence_artifact_hash,
            "source_partition": EvidencePartition(self.source_partition).value,
            "visibility": EvidenceVisibility(self.visibility).value,
            "verdict": EvidenceVerdict(self.verdict).value,
            "directional_stability": StabilityCategory(
                self.directional_stability
            ).value,
            "regime_stability": StabilityCategory(self.regime_stability).value,
            "cost_resilience": CostResilienceCategory(self.cost_resilience).value,
            "coverage": CoverageCategory(self.coverage).value,
            "reason_codes": [
                EvidenceReasonCode(item).value for item in self.reason_codes
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LogicEvidenceSummary":
        expected = {
            "schema_version",
            "logic_hash",
            "experiment_hash",
            "evidence_artifact_hash",
            "source_partition",
            "visibility",
            "verdict",
            "directional_stability",
            "regime_stability",
            "cost_resilience",
            "coverage",
            "reason_codes",
        }
        if set(payload) != expected:
            raise ValueError(
                "logic evidence payload fields differ from the sealed schema"
            )
        raw_reasons = payload["reason_codes"]
        if not isinstance(raw_reasons, list) or not all(
            isinstance(item, str) for item in raw_reasons
        ):
            raise ValueError("logic evidence reason_codes must be a JSON string list")
        scalar_names = expected.difference({"reason_codes"})
        if not all(isinstance(payload[name], str) for name in scalar_names):
            raise ValueError("logic evidence scalar fields must be strings")
        return cls(
            schema_version=str(payload["schema_version"]),
            logic_hash=str(payload["logic_hash"]),
            experiment_hash=str(payload["experiment_hash"]),
            evidence_artifact_hash=str(payload["evidence_artifact_hash"]),
            source_partition=str(payload["source_partition"]),
            visibility=str(payload["visibility"]),
            verdict=str(payload["verdict"]),
            directional_stability=str(payload["directional_stability"]),
            regime_stability=str(payload["regime_stability"]),
            cost_resilience=str(payload["cost_resilience"]),
            coverage=str(payload["coverage"]),
            reason_codes=tuple(str(item) for item in raw_reasons),
        )


__all__ = [
    "CoverageCategory",
    "CostResilienceCategory",
    "EvidencePartition",
    "EvidenceReasonCode",
    "EvidenceVerdict",
    "EvidenceVisibility",
    "LOGIC_EVIDENCE_SCHEMA",
    "LogicEvidenceSummary",
    "StabilityCategory",
]
