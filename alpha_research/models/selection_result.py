from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
from types import MappingProxyType
from typing import Mapping, cast

import pandas as pd

from alpha_research.core.hashing import hash_json, require_sha256
from alpha_research.models.artifact import ModelArtifactBundle, ModelResultDescriptor
from alpha_research.models.selection_scoring_evidence import (
    FoldPredictionRankEvidence,
    FoldScoringCohortEvidence,
    FoldScoringEvidenceVerifier,
    VerifiedFoldScoringEvidence,
)
from alpha_research.models.selection_spec import (
    NESTED_SELECTION_TIE_BREAK,
    ModelCandidateTemplate,
    NestedPurgedSelectionSpec,
)
from alpha_research.validation import (
    SplitFoldReceipt,
    ValidationFoldSpec,
    ValidationReceipt,
    ValidationSpec,
)


def _mean(values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError("mean requires values")
    return fsum(values) / len(values)


@dataclass(frozen=True, slots=True)
class OuterTrainingSliceReceipt:
    outer_validation_spec_hash: str
    outer_fold_receipt: SplitFoldReceipt
    outer_fold_receipt_hash: str
    train_signals: tuple[str, ...]
    sliced_feature_hashes: Mapping[str, str]
    sliced_label_values_hash: str
    sliced_label_windows_hash: str
    sliced_label_validity_hash: str
    sliced_label_diagnostics_hash: str
    scoring_eligibility_hash: str
    sliced_calendar_hash: str
    schema_version: str = "outer-training-slice-receipt/v2"

    def __post_init__(self) -> None:
        if self.schema_version != "outer-training-slice-receipt/v2":
            raise ValueError("unsupported outer training slice receipt schema")
        if not isinstance(self.outer_fold_receipt, SplitFoldReceipt):
            raise TypeError("outer training slice fold receipt type differs")
        for name in (
            "outer_validation_spec_hash",
            "outer_fold_receipt_hash",
            "sliced_label_values_hash",
            "sliced_label_windows_hash",
            "sliced_label_validity_hash",
            "sliced_label_diagnostics_hash",
            "scoring_eligibility_hash",
            "sliced_calendar_hash",
        ):
            require_sha256(
                str(getattr(self, name)), name=f"outer training slice {name}"
            )
        if self.outer_fold_receipt.content_hash != self.outer_fold_receipt_hash:
            raise ValueError("outer training slice fold receipt hash differs")
        sliced = _hash_mapping(self.sliced_feature_hashes, "sliced features")
        object.__setattr__(self, "sliced_feature_hashes", MappingProxyType(sliced))
        signals = tuple(self.train_signals)
        if signals != self.outer_fold_receipt.train_signals:
            raise ValueError("outer training slice signals differ from fold receipt")
        if not signals:
            raise ValueError("outer training slice requires train signals")
        if len(signals) != len(set(signals)):
            raise ValueError("outer training slice train signals must be unique")
        forbidden = set(self.outer_fold_receipt.validation_signals)
        forbidden.update(self.outer_fold_receipt.purged_train_signals)
        forbidden.update(self.outer_fold_receipt.excluded_validation_signals)
        if forbidden.intersection(signals):
            raise ValueError("outer training slice contains forbidden signals")
        object.__setattr__(self, "train_signals", signals)

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "outer_validation_spec_hash": self.outer_validation_spec_hash,
            "outer_fold_receipt": self.outer_fold_receipt.to_dict(),
            "outer_fold_receipt_hash": self.outer_fold_receipt_hash,
            "train_signals": list(self.train_signals),
            "sliced_feature_hashes": dict(self.sliced_feature_hashes),
            "sliced_label_values_hash": self.sliced_label_values_hash,
            "sliced_label_windows_hash": self.sliced_label_windows_hash,
            "sliced_label_validity_hash": self.sliced_label_validity_hash,
            "sliced_label_diagnostics_hash": self.sliced_label_diagnostics_hash,
            "scoring_eligibility_hash": self.scoring_eligibility_hash,
            "sliced_calendar_hash": self.sliced_calendar_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "OuterTrainingSliceReceipt":
        expected = {
            "schema_version",
            "outer_validation_spec_hash",
            "outer_fold_receipt",
            "outer_fold_receipt_hash",
            "train_signals",
            "sliced_feature_hashes",
            "sliced_label_values_hash",
            "sliced_label_windows_hash",
            "sliced_label_validity_hash",
            "sliced_label_diagnostics_hash",
            "scoring_eligibility_hash",
            "sliced_calendar_hash",
        }
        if set(value) != expected:
            raise ValueError("OuterTrainingSliceReceipt wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            outer_validation_spec_hash=_text(
                value["outer_validation_spec_hash"],
                "outer_validation_spec_hash",
            ),
            outer_fold_receipt=SplitFoldReceipt.from_mapping(
                _mapping(value["outer_fold_receipt"], "outer_fold_receipt")
            ),
            outer_fold_receipt_hash=_text(
                value["outer_fold_receipt_hash"], "outer_fold_receipt_hash"
            ),
            train_signals=_text_tuple(value["train_signals"], "train_signals"),
            sliced_feature_hashes=_string_mapping(
                value["sliced_feature_hashes"], "sliced_feature_hashes"
            ),
            sliced_label_values_hash=_text(
                value["sliced_label_values_hash"], "sliced_label_values_hash"
            ),
            sliced_label_windows_hash=_text(
                value["sliced_label_windows_hash"], "sliced_label_windows_hash"
            ),
            sliced_label_validity_hash=_text(
                value["sliced_label_validity_hash"], "sliced_label_validity_hash"
            ),
            sliced_label_diagnostics_hash=_text(
                value["sliced_label_diagnostics_hash"],
                "sliced_label_diagnostics_hash",
            ),
            scoring_eligibility_hash=_text(
                value["scoring_eligibility_hash"], "scoring_eligibility_hash"
            ),
            sliced_calendar_hash=_text(
                value["sliced_calendar_hash"], "sliced_calendar_hash"
            ),
        )


@dataclass(frozen=True, slots=True)
class RankICObservation:
    signal_timestamp: str
    observation_count: int
    security_membership_hash: str
    rank_ic: float
    schema_version: str = "rank-ic-observation/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "rank-ic-observation/v1":
            raise ValueError("unsupported RankIC observation schema")
        timestamp = pd.Timestamp(self.signal_timestamp)
        if timestamp.tzinfo is None:
            raise ValueError("RankIC observation timestamp must be timezone-aware")
        object.__setattr__(self, "signal_timestamp", timestamp.isoformat())
        if (
            not isinstance(self.observation_count, int)
            or isinstance(self.observation_count, bool)
            or self.observation_count < 2
        ):
            raise ValueError("RankIC observation count must be at least two")
        require_sha256(
            self.security_membership_hash,
            name="RankIC observation security_membership_hash",
        )
        value = float(self.rank_ic)
        if not isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("RankIC observation value must lie in [-1, 1]")
        object.__setattr__(self, "rank_ic", value)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "signal_timestamp": self.signal_timestamp,
            "observation_count": self.observation_count,
            "security_membership_hash": self.security_membership_hash,
            "rank_ic": self.rank_ic,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RankICObservation":
        expected = {
            "schema_version",
            "signal_timestamp",
            "observation_count",
            "security_membership_hash",
            "rank_ic",
        }
        if set(value) != expected:
            raise ValueError("RankICObservation wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            signal_timestamp=_text(value["signal_timestamp"], "signal_timestamp"),
            observation_count=_integer(value["observation_count"], "observation_count"),
            security_membership_hash=_text(
                value["security_membership_hash"], "security_membership_hash"
            ),
            rank_ic=_number(value["rank_ic"], "rank_ic"),
        )


@dataclass(frozen=True, slots=True)
class FoldRankICReceipt:
    fold_id: str
    scoring_cohort_evidence: FoldScoringCohortEvidence
    scoring_cohort_evidence_hash: str
    prediction_rank_evidence: FoldPredictionRankEvidence
    prediction_rank_evidence_hash: str
    verified_scoring_evidence: VerifiedFoldScoringEvidence
    verified_scoring_evidence_hash: str
    observations: tuple[RankICObservation, ...]
    observations_hash: str
    rank_ic_mean: float
    schema_version: str = "fold-rank-ic-receipt/v2"

    def __post_init__(self) -> None:
        if self.schema_version != "fold-rank-ic-receipt/v2":
            raise ValueError("unsupported fold RankIC receipt schema")
        if not self.fold_id.strip():
            raise ValueError("fold RankIC receipt requires a fold id")
        if not isinstance(self.scoring_cohort_evidence, FoldScoringCohortEvidence):
            raise TypeError("fold RankIC scoring cohort evidence type differs")
        if not isinstance(self.prediction_rank_evidence, FoldPredictionRankEvidence):
            raise TypeError("fold RankIC prediction rank evidence type differs")
        if not isinstance(self.verified_scoring_evidence, VerifiedFoldScoringEvidence):
            raise TypeError("fold RankIC verified scoring evidence type differs")
        for name in (
            "scoring_cohort_evidence_hash",
            "prediction_rank_evidence_hash",
            "verified_scoring_evidence_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"fold RankIC {name}")
        cohort = self.scoring_cohort_evidence
        prediction = self.prediction_rank_evidence
        if cohort.content_hash != self.scoring_cohort_evidence_hash:
            raise ValueError("fold RankIC scoring cohort evidence hash differs")
        if prediction.content_hash != self.prediction_rank_evidence_hash:
            raise ValueError("fold RankIC prediction rank evidence hash differs")
        verified = FoldScoringEvidenceVerifier().verify(
            cohort,
            prediction,
            expected_fold_id=self.fold_id,
            expected_validation_receipt_hash=cohort.validation_receipt_hash,
            expected_scoring_spec_hash=cohort.scoring_spec_hash,
            expected_label_values_hash=cohort.label_values_hash,
            expected_label_validity_hash=cohort.label_validity_hash,
            expected_eligibility_hash=cohort.eligibility_hash,
            expected_model_result_hash=prediction.model_result_hash,
            expected_predictions_hash=prediction.predictions_hash,
            expected_cohort_evidence_hash=self.scoring_cohort_evidence_hash,
            expected_prediction_evidence_hash=self.prediction_rank_evidence_hash,
        )
        if (
            self.verified_scoring_evidence.content_hash
            != self.verified_scoring_evidence_hash
            or verified.to_dict() != self.verified_scoring_evidence.to_dict()
        ):
            raise ValueError("fold RankIC verified scoring evidence differs")
        observations = tuple(self.observations)
        if not observations or not all(
            isinstance(item, RankICObservation) for item in observations
        ):
            raise TypeError("fold RankIC receipt observations have invalid types")
        timestamps = tuple(item.signal_timestamp for item in observations)
        if timestamps != tuple(sorted(set(timestamps))):
            raise ValueError("fold RankIC observations must be sorted and unique")
        object.__setattr__(self, "observations", observations)
        require_sha256(self.observations_hash, name="fold RankIC observations_hash")
        expected_hash = hash_json([item.to_dict() for item in observations])
        if expected_hash != self.observations_hash:
            raise ValueError("fold RankIC observation hash differs")
        expected_observations = tuple(
            RankICObservation(
                signal_timestamp=item.signal_timestamp,
                observation_count=item.observation_count,
                security_membership_hash=item.security_membership_hash,
                rank_ic=item.rank_ic,
            )
            for item in verified.daily_scores
        )
        if tuple(item.to_dict() for item in observations) != tuple(
            item.to_dict() for item in expected_observations
        ):
            raise ValueError("fold RankIC observations differ from scoring evidence")
        expected_mean = _mean(tuple(item.rank_ic for item in observations))
        value = float(self.rank_ic_mean)
        if (
            not isfinite(value)
            or value != expected_mean
            or value != verified.fold_rank_ic_mean
        ):
            raise ValueError("fold RankIC mean differs from observations")
        object.__setattr__(self, "rank_ic_mean", value)

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fold_id": self.fold_id,
            "scoring_cohort_evidence": self.scoring_cohort_evidence.to_dict(),
            "scoring_cohort_evidence_hash": self.scoring_cohort_evidence_hash,
            "prediction_rank_evidence": self.prediction_rank_evidence.to_dict(),
            "prediction_rank_evidence_hash": self.prediction_rank_evidence_hash,
            "verified_scoring_evidence": self.verified_scoring_evidence.to_dict(),
            "verified_scoring_evidence_hash": self.verified_scoring_evidence_hash,
            "observations": [item.to_dict() for item in self.observations],
            "observations_hash": self.observations_hash,
            "rank_ic_mean": self.rank_ic_mean,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FoldRankICReceipt":
        expected = {
            "schema_version",
            "fold_id",
            "scoring_cohort_evidence",
            "scoring_cohort_evidence_hash",
            "prediction_rank_evidence",
            "prediction_rank_evidence_hash",
            "verified_scoring_evidence",
            "verified_scoring_evidence_hash",
            "observations",
            "observations_hash",
            "rank_ic_mean",
        }
        if set(value) != expected:
            raise ValueError("FoldRankICReceipt wire fields differ")
        observations = _mapping_list(value["observations"], "observations")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            fold_id=_text(value["fold_id"], "fold_id"),
            scoring_cohort_evidence=FoldScoringCohortEvidence.from_mapping(
                _mapping(
                    value["scoring_cohort_evidence"],
                    "scoring_cohort_evidence",
                )
            ),
            scoring_cohort_evidence_hash=_text(
                value["scoring_cohort_evidence_hash"],
                "scoring_cohort_evidence_hash",
            ),
            prediction_rank_evidence=FoldPredictionRankEvidence.from_mapping(
                _mapping(
                    value["prediction_rank_evidence"],
                    "prediction_rank_evidence",
                )
            ),
            prediction_rank_evidence_hash=_text(
                value["prediction_rank_evidence_hash"],
                "prediction_rank_evidence_hash",
            ),
            verified_scoring_evidence=VerifiedFoldScoringEvidence.from_mapping(
                _mapping(
                    value["verified_scoring_evidence"],
                    "verified_scoring_evidence",
                )
            ),
            verified_scoring_evidence_hash=_text(
                value["verified_scoring_evidence_hash"],
                "verified_scoring_evidence_hash",
            ),
            observations=tuple(
                RankICObservation.from_mapping(item) for item in observations
            ),
            observations_hash=_text(value["observations_hash"], "observations_hash"),
            rank_ic_mean=_number(value["rank_ic_mean"], "rank_ic_mean"),
        )


@dataclass(frozen=True, slots=True)
class CandidateScoreReceipt:
    outer_fold_id: str
    candidate_id: str
    candidate_template_hash: str
    materialized_model_spec_hash: str
    outer_training_slice_hash: str
    inner_validation_spec_hash: str
    inner_validation_receipt_hash: str
    inner_label_values_hash: str
    inner_model_result: ModelResultDescriptor
    inner_model_result_hash: str
    preprocess_spec_hash: str
    fold_scores: tuple[FoldRankICReceipt, ...]
    selection_score: float
    complexity_rank: int
    consumed_fold_evaluations: int
    schema_version: str = "candidate-score-receipt/v2"

    def __post_init__(self) -> None:
        if self.schema_version != "candidate-score-receipt/v2":
            raise ValueError("unsupported candidate score receipt schema")
        if not self.outer_fold_id.strip() or not self.candidate_id.strip():
            raise ValueError("candidate score identity is required")
        for name in (
            "candidate_template_hash",
            "materialized_model_spec_hash",
            "outer_training_slice_hash",
            "inner_validation_spec_hash",
            "inner_validation_receipt_hash",
            "inner_label_values_hash",
            "inner_model_result_hash",
            "preprocess_spec_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"candidate score {name}")
        if not isinstance(self.inner_model_result, ModelResultDescriptor):
            raise TypeError("candidate inner model result descriptor type differs")
        descriptor = self.inner_model_result
        if (
            _model_result_hash(descriptor) != descriptor.result_hash
            or descriptor.result_hash != self.inner_model_result_hash
        ):
            raise ValueError("candidate inner model result hash differs")
        if (
            descriptor.model_spec_hash != self.materialized_model_spec_hash
            or descriptor.validation_receipt_hash != self.inner_validation_receipt_hash
            or descriptor.label_values_hash != self.inner_label_values_hash
            or descriptor.preprocess_spec_hash != self.preprocess_spec_hash
        ):
            raise ValueError("candidate inner model result lineage differs")
        scores = tuple(self.fold_scores)
        if not scores or not all(
            isinstance(item, FoldRankICReceipt) for item in scores
        ):
            raise TypeError("candidate fold scores have invalid types")
        if len({item.fold_id for item in scores}) != len(scores):
            raise ValueError("candidate fold score ids must be unique")
        object.__setattr__(self, "fold_scores", scores)
        if descriptor.fold_count != len(scores):
            raise ValueError("candidate inner model result fold count differs")
        for fold_score in scores:
            cohort = fold_score.scoring_cohort_evidence
            prediction = fold_score.prediction_rank_evidence
            if (
                cohort.label_values_hash != self.inner_label_values_hash
                or prediction.model_result_hash != self.inner_model_result_hash
                or prediction.predictions_hash != descriptor.predictions_hash
            ):
                raise ValueError("candidate fold score model/label evidence differs")
        expected_score = _mean(tuple(item.rank_ic_mean for item in scores))
        selection_score = float(self.selection_score)
        if not isfinite(selection_score) or selection_score != expected_score:
            raise ValueError("candidate selection score differs from fold means")
        object.__setattr__(self, "selection_score", selection_score)
        if (
            not isinstance(self.complexity_rank, int)
            or isinstance(self.complexity_rank, bool)
            or self.complexity_rank < 0
        ):
            raise ValueError("candidate score complexity_rank must be non-negative")
        if (
            not isinstance(self.consumed_fold_evaluations, int)
            or isinstance(self.consumed_fold_evaluations, bool)
            or self.consumed_fold_evaluations != len(scores)
        ):
            raise ValueError("candidate consumed fold evaluations differ")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    @property
    def selection_key(self) -> tuple[float, int, str]:
        return (-self.selection_score, self.complexity_rank, self.candidate_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "outer_fold_id": self.outer_fold_id,
            "candidate_id": self.candidate_id,
            "candidate_template_hash": self.candidate_template_hash,
            "materialized_model_spec_hash": self.materialized_model_spec_hash,
            "outer_training_slice_hash": self.outer_training_slice_hash,
            "inner_validation_spec_hash": self.inner_validation_spec_hash,
            "inner_validation_receipt_hash": self.inner_validation_receipt_hash,
            "inner_label_values_hash": self.inner_label_values_hash,
            "inner_model_result": self.inner_model_result.to_dict(),
            "inner_model_result_hash": self.inner_model_result_hash,
            "preprocess_spec_hash": self.preprocess_spec_hash,
            "fold_scores": [item.to_dict() for item in self.fold_scores],
            "selection_score": self.selection_score,
            "complexity_rank": self.complexity_rank,
            "consumed_fold_evaluations": self.consumed_fold_evaluations,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CandidateScoreReceipt":
        expected = {
            "schema_version",
            "outer_fold_id",
            "candidate_id",
            "candidate_template_hash",
            "materialized_model_spec_hash",
            "outer_training_slice_hash",
            "inner_validation_spec_hash",
            "inner_validation_receipt_hash",
            "inner_label_values_hash",
            "inner_model_result",
            "inner_model_result_hash",
            "preprocess_spec_hash",
            "fold_scores",
            "selection_score",
            "complexity_rank",
            "consumed_fold_evaluations",
        }
        if set(value) != expected:
            raise ValueError("CandidateScoreReceipt wire fields differ")
        scores = _mapping_list(value["fold_scores"], "fold_scores")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            outer_fold_id=_text(value["outer_fold_id"], "outer_fold_id"),
            candidate_id=_text(value["candidate_id"], "candidate_id"),
            candidate_template_hash=_text(
                value["candidate_template_hash"], "candidate_template_hash"
            ),
            materialized_model_spec_hash=_text(
                value["materialized_model_spec_hash"],
                "materialized_model_spec_hash",
            ),
            outer_training_slice_hash=_text(
                value["outer_training_slice_hash"], "outer_training_slice_hash"
            ),
            inner_validation_spec_hash=_text(
                value["inner_validation_spec_hash"], "inner_validation_spec_hash"
            ),
            inner_validation_receipt_hash=_text(
                value["inner_validation_receipt_hash"],
                "inner_validation_receipt_hash",
            ),
            inner_label_values_hash=_text(
                value["inner_label_values_hash"], "inner_label_values_hash"
            ),
            inner_model_result=ModelResultDescriptor.from_mapping(
                _mapping(value["inner_model_result"], "inner_model_result")
            ),
            inner_model_result_hash=_text(
                value["inner_model_result_hash"], "inner_model_result_hash"
            ),
            preprocess_spec_hash=_text(
                value["preprocess_spec_hash"], "preprocess_spec_hash"
            ),
            fold_scores=tuple(FoldRankICReceipt.from_mapping(item) for item in scores),
            selection_score=_number(value["selection_score"], "selection_score"),
            complexity_rank=_integer(value["complexity_rank"], "complexity_rank"),
            consumed_fold_evaluations=_integer(
                value["consumed_fold_evaluations"],
                "consumed_fold_evaluations",
            ),
        )


@dataclass(frozen=True, slots=True)
class OuterFoldSelectionReceipt:
    selection_spec_hash: str
    selection_spec: NestedPurgedSelectionSpec
    outer_fold_id: str
    outer_training_slice_hash: str
    outer_training_slice: OuterTrainingSliceReceipt
    inner_validation_receipt_hash: str
    inner_validation_receipt: ValidationReceipt
    candidate_scores: tuple[CandidateScoreReceipt, ...]
    selected_candidate_id: str
    selected_candidate_template_hash: str
    selected_score: float
    tie_break_policy: str = NESTED_SELECTION_TIE_BREAK
    schema_version: str = "outer-fold-selection-receipt/v2"

    def __post_init__(self) -> None:
        if self.schema_version != "outer-fold-selection-receipt/v2":
            raise ValueError("unsupported outer fold selection receipt schema")
        require_sha256(self.selection_spec_hash, name="selection receipt spec hash")
        if not isinstance(self.selection_spec, NestedPurgedSelectionSpec):
            raise TypeError("selection receipt specification type differs")
        if self.selection_spec.content_hash != self.selection_spec_hash:
            raise ValueError("selection receipt specification hash differs")
        require_sha256(
            self.outer_training_slice_hash,
            name="selection receipt outer training slice hash",
        )
        if not isinstance(self.outer_training_slice, OuterTrainingSliceReceipt):
            raise TypeError("selection receipt outer training slice type differs")
        if self.outer_training_slice.content_hash != self.outer_training_slice_hash:
            raise ValueError("selection receipt outer training slice hash differs")
        require_sha256(
            self.inner_validation_receipt_hash,
            name="selection receipt inner validation receipt hash",
        )
        if not isinstance(self.inner_validation_receipt, ValidationReceipt):
            raise TypeError("selection receipt inner validation receipt type differs")
        if (
            self.inner_validation_receipt.content_hash
            != self.inner_validation_receipt_hash
        ):
            raise ValueError("selection receipt inner validation receipt hash differs")
        if not self.outer_fold_id.strip() or not self.selected_candidate_id.strip():
            raise ValueError("outer fold selection identity is required")
        require_sha256(
            self.selected_candidate_template_hash,
            name="selection receipt candidate template hash",
        )
        matching_plans = tuple(
            item
            for item in self.selection_spec.outer_plans
            if item.outer_fold_id == self.outer_fold_id
        )
        if len(matching_plans) != 1:
            raise ValueError("selection receipt outer fold plan differs")
        plan = matching_plans[0]
        inner_spec_hash = plan.inner_validation_spec.content_hash
        if (
            self.outer_training_slice.outer_validation_spec_hash
            != self.selection_spec.outer_validation_spec_hash
            or self.outer_training_slice.outer_fold_receipt.fold_id
            != self.outer_fold_id
        ):
            raise ValueError("selection receipt outer training slice lineage differs")
        expected_feature_names = tuple(
            sorted(
                {
                    name
                    for candidate in self.selection_spec.candidates
                    for name in candidate.feature_names
                }
            )
        )
        if (
            tuple(self.outer_training_slice.sliced_feature_hashes)
            != expected_feature_names
        ):
            raise ValueError(
                "selection receipt outer training feature contract differs"
            )
        inner_receipt = self.inner_validation_receipt
        if (
            inner_receipt.validation_spec_hash != inner_spec_hash
            or inner_receipt.labels_hash
            != self.outer_training_slice.sliced_label_values_hash
            or inner_receipt.windows_hash
            != self.outer_training_slice.sliced_label_windows_hash
            or inner_receipt.calendar_hash
            != self.outer_training_slice.sliced_calendar_hash
        ):
            raise ValueError("selection receipt inner validation lineage differs")
        expected_inner_fold_ids = tuple(
            item.fold_id for item in plan.inner_validation_spec.folds
        )
        if (
            tuple(item.fold_id for item in inner_receipt.folds)
            != expected_inner_fold_ids
        ):
            raise ValueError("selection receipt inner validation fold order differs")
        if any(
            len(item.train_signals) < plan.inner_validation_spec.min_train_signals
            or len(item.validation_signals)
            < plan.inner_validation_spec.min_validation_signals
            for item in inner_receipt.folds
        ):
            raise ValueError("selection receipt inner validation signal count differs")
        _require_inner_membership_within_outer_slice(
            inner_receipt,
            outer_training_slice=self.outer_training_slice,
        )
        scores = tuple(self.candidate_scores)
        if not scores or not all(
            isinstance(item, CandidateScoreReceipt) for item in scores
        ):
            raise TypeError("outer fold candidate scores have invalid types")
        if len({item.candidate_id for item in scores}) != len(scores):
            raise ValueError("outer fold candidate score ids must be unique")
        candidates = self.selection_spec.candidates
        if tuple(item.candidate_id for item in scores) != tuple(
            item.candidate_id for item in candidates
        ):
            raise ValueError("outer fold candidate score contract is incomplete")
        if any(
            item.outer_fold_id != self.outer_fold_id
            or item.outer_training_slice_hash != self.outer_training_slice_hash
            or item.inner_validation_spec_hash != inner_spec_hash
            or item.inner_validation_receipt_hash != self.inner_validation_receipt_hash
            or item.inner_label_values_hash != inner_receipt.labels_hash
            for item in scores
        ):
            raise ValueError("outer fold candidate score lineage differs")
        for candidate, score in zip(candidates, scores, strict=True):
            _verify_candidate_score_contract(
                score,
                candidate=candidate,
                selection_spec=self.selection_spec,
                outer_fold_id=self.outer_fold_id,
                outer_training_slice=self.outer_training_slice,
                inner_validation_spec=plan.inner_validation_spec,
                inner_validation_receipt=inner_receipt,
            )
        reference_membership = _candidate_score_membership(scores[0])
        if any(
            _candidate_score_membership(item) != reference_membership
            for item in scores[1:]
        ):
            raise ValueError("outer fold candidate scoring membership differs")
        reference_cohorts = _candidate_score_cohort_hashes(scores[0])
        if any(
            _candidate_score_cohort_hashes(item) != reference_cohorts
            for item in scores[1:]
        ):
            raise ValueError("outer fold candidate scoring cohort evidence differs")
        object.__setattr__(self, "candidate_scores", scores)
        if self.tie_break_policy != NESTED_SELECTION_TIE_BREAK:
            raise ValueError("outer fold selection tie-break policy differs")
        expected = min(scores, key=lambda item: item.selection_key)
        if (
            expected.candidate_id != self.selected_candidate_id
            or expected.candidate_template_hash != self.selected_candidate_template_hash
            or float(self.selected_score) != expected.selection_score
        ):
            raise ValueError("outer fold selected candidate differs from score ranking")
        object.__setattr__(self, "selected_score", float(self.selected_score))

    @property
    def consumed_inner_evaluations(self) -> int:
        return sum(item.consumed_fold_evaluations for item in self.candidate_scores)

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "selection_spec_hash": self.selection_spec_hash,
            "selection_spec": self.selection_spec.to_dict(),
            "outer_fold_id": self.outer_fold_id,
            "outer_training_slice_hash": self.outer_training_slice_hash,
            "outer_training_slice": self.outer_training_slice.to_dict(),
            "inner_validation_receipt_hash": self.inner_validation_receipt_hash,
            "inner_validation_receipt": self.inner_validation_receipt.to_dict(),
            "candidate_scores": [item.to_dict() for item in self.candidate_scores],
            "selected_candidate_id": self.selected_candidate_id,
            "selected_candidate_template_hash": (self.selected_candidate_template_hash),
            "selected_score": self.selected_score,
            "tie_break_policy": self.tie_break_policy,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "OuterFoldSelectionReceipt":
        expected = {
            "schema_version",
            "selection_spec_hash",
            "selection_spec",
            "outer_fold_id",
            "outer_training_slice_hash",
            "outer_training_slice",
            "inner_validation_receipt_hash",
            "inner_validation_receipt",
            "candidate_scores",
            "selected_candidate_id",
            "selected_candidate_template_hash",
            "selected_score",
            "tie_break_policy",
        }
        if set(value) != expected:
            raise ValueError("OuterFoldSelectionReceipt wire fields differ")
        scores = _mapping_list(value["candidate_scores"], "candidate_scores")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            selection_spec_hash=_text(
                value["selection_spec_hash"], "selection_spec_hash"
            ),
            selection_spec=NestedPurgedSelectionSpec.from_mapping(
                _mapping(value["selection_spec"], "selection_spec")
            ),
            outer_fold_id=_text(value["outer_fold_id"], "outer_fold_id"),
            outer_training_slice_hash=_text(
                value["outer_training_slice_hash"], "outer_training_slice_hash"
            ),
            outer_training_slice=OuterTrainingSliceReceipt.from_mapping(
                _mapping(value["outer_training_slice"], "outer_training_slice")
            ),
            inner_validation_receipt_hash=_text(
                value["inner_validation_receipt_hash"],
                "inner_validation_receipt_hash",
            ),
            inner_validation_receipt=ValidationReceipt.from_mapping(
                _mapping(
                    value["inner_validation_receipt"],
                    "inner_validation_receipt",
                )
            ),
            candidate_scores=tuple(
                CandidateScoreReceipt.from_mapping(item) for item in scores
            ),
            selected_candidate_id=_text(
                value["selected_candidate_id"], "selected_candidate_id"
            ),
            selected_candidate_template_hash=_text(
                value["selected_candidate_template_hash"],
                "selected_candidate_template_hash",
            ),
            selected_score=_number(value["selected_score"], "selected_score"),
            tie_break_policy=_text(value["tie_break_policy"], "tie_break_policy"),
        )


@dataclass(frozen=True, slots=True)
class OuterFoldEvaluationResult:
    selection_receipt_hash: str
    outer_fold_id: str
    selected_candidate_id: str
    projected_outer_validation_spec_hash: str
    projected_outer_validation_receipt_hash: str
    materialized_model_spec_hash: str
    model_result_hash: str
    model_artifact: ModelArtifactBundle
    model_artifact_hash: str
    outer_score: FoldRankICReceipt
    outer_evaluation_ordinal: int = 1
    schema_version: str = "outer-fold-evaluation-result/v2"

    def __post_init__(self) -> None:
        if self.schema_version != "outer-fold-evaluation-result/v2":
            raise ValueError("unsupported outer fold evaluation result schema")
        if not self.outer_fold_id.strip() or not self.selected_candidate_id.strip():
            raise ValueError("outer fold evaluation identity is required")
        for name in (
            "selection_receipt_hash",
            "projected_outer_validation_spec_hash",
            "projected_outer_validation_receipt_hash",
            "materialized_model_spec_hash",
            "model_result_hash",
            "model_artifact_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"outer evaluation {name}")
        if not isinstance(self.model_artifact, ModelArtifactBundle):
            raise TypeError("outer fold model artifact type differs")
        if self.model_artifact.content_hash != self.model_artifact_hash:
            raise ValueError("outer fold model artifact hash differs")
        if (
            _model_result_hash(self.model_artifact.model_result)
            != self.model_artifact.model_result_hash
        ):
            raise ValueError("outer fold model result descriptor hash differs")
        if (
            self.model_artifact.validation_receipt.validation_spec_hash
            != self.projected_outer_validation_spec_hash
            or self.model_artifact.validation_receipt_hash
            != self.projected_outer_validation_receipt_hash
            or self.model_artifact.model_spec_hash != self.materialized_model_spec_hash
            or self.model_artifact.model_result_hash != self.model_result_hash
        ):
            raise ValueError("outer fold model artifact lineage differs")
        artifact_folds = self.model_artifact.validation_receipt.folds
        if len(artifact_folds) != 1 or artifact_folds[0].fold_id != self.outer_fold_id:
            raise ValueError("outer fold model artifact must contain its single fold")
        if not isinstance(self.outer_score, FoldRankICReceipt):
            raise TypeError("outer fold score type differs")
        if self.outer_score.fold_id != self.outer_fold_id:
            raise ValueError("outer fold score id differs")
        if self.outer_evaluation_ordinal != 1:
            raise ValueError("outer validation may be evaluated exactly once")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "selection_receipt_hash": self.selection_receipt_hash,
            "outer_fold_id": self.outer_fold_id,
            "selected_candidate_id": self.selected_candidate_id,
            "projected_outer_validation_spec_hash": (
                self.projected_outer_validation_spec_hash
            ),
            "projected_outer_validation_receipt_hash": (
                self.projected_outer_validation_receipt_hash
            ),
            "materialized_model_spec_hash": self.materialized_model_spec_hash,
            "model_result_hash": self.model_result_hash,
            "model_artifact": self.model_artifact.to_dict(),
            "model_artifact_hash": self.model_artifact_hash,
            "outer_score": self.outer_score.to_dict(),
            "outer_evaluation_ordinal": self.outer_evaluation_ordinal,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "OuterFoldEvaluationResult":
        expected = {
            "schema_version",
            "selection_receipt_hash",
            "outer_fold_id",
            "selected_candidate_id",
            "projected_outer_validation_spec_hash",
            "projected_outer_validation_receipt_hash",
            "materialized_model_spec_hash",
            "model_result_hash",
            "model_artifact",
            "model_artifact_hash",
            "outer_score",
            "outer_evaluation_ordinal",
        }
        if set(value) != expected:
            raise ValueError("OuterFoldEvaluationResult wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            selection_receipt_hash=_text(
                value["selection_receipt_hash"], "selection_receipt_hash"
            ),
            outer_fold_id=_text(value["outer_fold_id"], "outer_fold_id"),
            selected_candidate_id=_text(
                value["selected_candidate_id"], "selected_candidate_id"
            ),
            projected_outer_validation_spec_hash=_text(
                value["projected_outer_validation_spec_hash"],
                "projected_outer_validation_spec_hash",
            ),
            projected_outer_validation_receipt_hash=_text(
                value["projected_outer_validation_receipt_hash"],
                "projected_outer_validation_receipt_hash",
            ),
            materialized_model_spec_hash=_text(
                value["materialized_model_spec_hash"],
                "materialized_model_spec_hash",
            ),
            model_result_hash=_text(value["model_result_hash"], "model_result_hash"),
            model_artifact=ModelArtifactBundle.from_mapping(
                _mapping(value["model_artifact"], "model_artifact")
            ),
            model_artifact_hash=_text(
                value["model_artifact_hash"], "model_artifact_hash"
            ),
            outer_score=FoldRankICReceipt.from_mapping(
                _mapping(value["outer_score"], "outer_score")
            ),
            outer_evaluation_ordinal=_integer(
                value["outer_evaluation_ordinal"], "outer_evaluation_ordinal"
            ),
        )


@dataclass(frozen=True, slots=True)
class NestedSelectionResult:
    selection_spec_hash: str
    selection_spec: NestedPurgedSelectionSpec
    outer_validation_spec_hash: str
    outer_validation_spec: ValidationSpec
    outer_validation_receipt_hash: str
    outer_validation_receipt: ValidationReceipt
    source_feature_hashes: Mapping[str, str]
    source_label_values_hash: str
    source_label_validity_hash: str
    source_scoring_eligibility_hash: str
    selections: tuple[OuterFoldSelectionReceipt, ...]
    outer_evaluations: tuple[OuterFoldEvaluationResult, ...]
    required_fold_evaluations: int
    consumed_fold_evaluations: int
    research_only: bool = True
    production_ready: bool = False
    schema_version: str = "nested-selection-result/v2"

    def __post_init__(self) -> None:
        if self.schema_version != "nested-selection-result/v2":
            raise ValueError("unsupported nested selection result schema")
        if not isinstance(self.selection_spec, NestedPurgedSelectionSpec):
            raise TypeError("nested result selection specification type differs")
        if not isinstance(self.outer_validation_spec, ValidationSpec):
            raise TypeError("nested result outer validation specification type differs")
        if not isinstance(self.outer_validation_receipt, ValidationReceipt):
            raise TypeError("nested result outer validation receipt type differs")
        for name in (
            "selection_spec_hash",
            "outer_validation_spec_hash",
            "outer_validation_receipt_hash",
            "source_label_values_hash",
            "source_label_validity_hash",
            "source_scoring_eligibility_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"nested result {name}")
        if self.selection_spec.content_hash != self.selection_spec_hash:
            raise ValueError("nested result selection specification hash differs")
        if (
            self.outer_validation_spec.content_hash != self.outer_validation_spec_hash
            or self.selection_spec.outer_validation_spec_hash
            != self.outer_validation_spec_hash
        ):
            raise ValueError(
                "nested result outer validation specification hash differs"
            )
        if (
            self.outer_validation_receipt.content_hash
            != self.outer_validation_receipt_hash
            or self.outer_validation_receipt.validation_spec_hash
            != self.outer_validation_spec_hash
        ):
            raise ValueError("nested result outer validation receipt hash differs")
        if self.source_label_values_hash != self.outer_validation_receipt.labels_hash:
            raise ValueError("nested result source label values hash differs")
        features = _hash_mapping(self.source_feature_hashes, "source features")
        expected_feature_names = tuple(
            sorted(
                {
                    name
                    for candidate in self.selection_spec.candidates
                    for name in candidate.feature_names
                }
            )
        )
        if tuple(features) != expected_feature_names:
            raise ValueError("nested result source feature contract differs")
        object.__setattr__(self, "source_feature_hashes", MappingProxyType(features))
        selections = tuple(self.selections)
        evaluations = tuple(self.outer_evaluations)
        if not selections or not all(
            isinstance(item, OuterFoldSelectionReceipt) for item in selections
        ):
            raise TypeError("nested result selections have invalid types")
        if not all(isinstance(item, OuterFoldEvaluationResult) for item in evaluations):
            raise TypeError("nested result outer evaluations have invalid types")
        selection_ids = tuple(item.outer_fold_id for item in selections)
        evaluation_ids = tuple(item.outer_fold_id for item in evaluations)
        contract_fold_ids = tuple(
            item.outer_fold_id for item in self.selection_spec.outer_plans
        )
        outer_spec_fold_ids = tuple(
            item.fold_id for item in self.outer_validation_spec.folds
        )
        outer_receipt_fold_ids = tuple(
            item.fold_id for item in self.outer_validation_receipt.folds
        )
        if len(set(selection_ids)) != len(selection_ids):
            raise ValueError("nested result outer fold ids must be unique")
        if not (
            selection_ids
            == evaluation_ids
            == contract_fold_ids
            == outer_spec_fold_ids
            == outer_receipt_fold_ids
        ):
            raise ValueError("nested result outer fold contract/order differs")
        for selection, evaluation, fold_spec, fold_receipt in zip(
            selections,
            evaluations,
            self.outer_validation_spec.folds,
            self.outer_validation_receipt.folds,
            strict=True,
        ):
            if (
                selection.selection_spec_hash != self.selection_spec_hash
                or selection.selection_spec.content_hash != self.selection_spec_hash
                or selection.inner_validation_receipt.label_spec_hash
                != self.outer_validation_receipt.label_spec_hash
                or evaluation.selection_receipt_hash != selection.content_hash
                or evaluation.selected_candidate_id != selection.selected_candidate_id
            ):
                raise ValueError("nested result selection/evaluation lineage differs")
            if (
                selection.outer_training_slice.outer_validation_spec_hash
                != self.outer_validation_spec_hash
                or selection.outer_training_slice.outer_fold_receipt_hash
                != fold_receipt.content_hash
                or selection.outer_training_slice.outer_fold_receipt.content_hash
                != fold_receipt.content_hash
            ):
                raise ValueError("nested result outer training membership differs")
            _verify_outer_evaluation_contract(
                evaluation,
                selection=selection,
                selection_spec=self.selection_spec,
                outer_validation_spec=self.outer_validation_spec,
                outer_fold_spec=fold_spec,
                outer_fold_receipt=fold_receipt,
                outer_validation_receipt=self.outer_validation_receipt,
                source_feature_hashes=features,
                source_label_values_hash=self.source_label_values_hash,
                source_label_validity_hash=self.source_label_validity_hash,
                source_scoring_eligibility_hash=(self.source_scoring_eligibility_hash),
            )
        object.__setattr__(self, "selections", selections)
        object.__setattr__(self, "outer_evaluations", evaluations)
        for name in ("required_fold_evaluations", "consumed_fold_evaluations"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"nested result {name} must be positive")
        expected_consumed = sum(
            item.consumed_inner_evaluations for item in selections
        ) + len(evaluations)
        if (
            self.consumed_fold_evaluations != expected_consumed
            or self.consumed_fold_evaluations != self.required_fold_evaluations
            or self.required_fold_evaluations
            != self.selection_spec.required_fold_evaluations
            or len(evaluations) != self.selection_spec.maximum_outer_evaluations
        ):
            raise ValueError("nested result fold evaluation accounting differs")
        if (
            not isinstance(self.research_only, bool)
            or not self.research_only
            or not isinstance(self.production_ready, bool)
            or self.production_ready
        ):
            raise ValueError("nested result assurance boundary differs")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "selection_spec_hash": self.selection_spec_hash,
            "selection_spec": self.selection_spec.to_dict(),
            "outer_validation_spec_hash": self.outer_validation_spec_hash,
            "outer_validation_spec": self.outer_validation_spec.to_dict(),
            "outer_validation_receipt_hash": self.outer_validation_receipt_hash,
            "outer_validation_receipt": self.outer_validation_receipt.to_dict(),
            "source_feature_hashes": dict(self.source_feature_hashes),
            "source_label_values_hash": self.source_label_values_hash,
            "source_label_validity_hash": self.source_label_validity_hash,
            "source_scoring_eligibility_hash": self.source_scoring_eligibility_hash,
            "selections": [item.to_dict() for item in self.selections],
            "outer_evaluations": [item.to_dict() for item in self.outer_evaluations],
            "required_fold_evaluations": self.required_fold_evaluations,
            "consumed_fold_evaluations": self.consumed_fold_evaluations,
            "research_only": self.research_only,
            "production_ready": self.production_ready,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "NestedSelectionResult":
        expected = {
            "schema_version",
            "selection_spec_hash",
            "selection_spec",
            "outer_validation_spec_hash",
            "outer_validation_spec",
            "outer_validation_receipt_hash",
            "outer_validation_receipt",
            "source_feature_hashes",
            "source_label_values_hash",
            "source_label_validity_hash",
            "source_scoring_eligibility_hash",
            "selections",
            "outer_evaluations",
            "required_fold_evaluations",
            "consumed_fold_evaluations",
            "research_only",
            "production_ready",
        }
        if set(value) != expected:
            raise ValueError("NestedSelectionResult wire fields differ")
        selections = _mapping_list(value["selections"], "selections")
        evaluations = _mapping_list(value["outer_evaluations"], "outer_evaluations")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            selection_spec_hash=_text(
                value["selection_spec_hash"], "selection_spec_hash"
            ),
            selection_spec=NestedPurgedSelectionSpec.from_mapping(
                _mapping(value["selection_spec"], "selection_spec")
            ),
            outer_validation_spec_hash=_text(
                value["outer_validation_spec_hash"], "outer_validation_spec_hash"
            ),
            outer_validation_spec=ValidationSpec.from_mapping(
                _mapping(value["outer_validation_spec"], "outer_validation_spec")
            ),
            outer_validation_receipt_hash=_text(
                value["outer_validation_receipt_hash"],
                "outer_validation_receipt_hash",
            ),
            outer_validation_receipt=ValidationReceipt.from_mapping(
                _mapping(
                    value["outer_validation_receipt"],
                    "outer_validation_receipt",
                )
            ),
            source_feature_hashes=_string_mapping(
                value["source_feature_hashes"], "source_feature_hashes"
            ),
            source_label_values_hash=_text(
                value["source_label_values_hash"], "source_label_values_hash"
            ),
            source_label_validity_hash=_text(
                value["source_label_validity_hash"],
                "source_label_validity_hash",
            ),
            source_scoring_eligibility_hash=_text(
                value["source_scoring_eligibility_hash"],
                "source_scoring_eligibility_hash",
            ),
            selections=tuple(
                OuterFoldSelectionReceipt.from_mapping(item) for item in selections
            ),
            outer_evaluations=tuple(
                OuterFoldEvaluationResult.from_mapping(item) for item in evaluations
            ),
            required_fold_evaluations=_integer(
                value["required_fold_evaluations"], "required_fold_evaluations"
            ),
            consumed_fold_evaluations=_integer(
                value["consumed_fold_evaluations"], "consumed_fold_evaluations"
            ),
            research_only=_boolean(value["research_only"], "research_only"),
            production_ready=_boolean(value["production_ready"], "production_ready"),
        )


def _model_result_hash(descriptor: ModelResultDescriptor) -> str:
    return descriptor.recomputed_result_hash


def _require_inner_membership_within_outer_slice(
    receipt: ValidationReceipt,
    *,
    outer_training_slice: OuterTrainingSliceReceipt,
) -> None:
    allowed = set(outer_training_slice.train_signals)
    forbidden = set(outer_training_slice.outer_fold_receipt.validation_signals)
    forbidden.update(outer_training_slice.outer_fold_receipt.purged_train_signals)
    forbidden.update(
        outer_training_slice.outer_fold_receipt.excluded_validation_signals
    )
    for fold in receipt.folds:
        members = set(fold.train_signals)
        members.update(fold.validation_signals)
        members.update(fold.purged_train_signals)
        members.update(fold.excluded_validation_signals)
        if not members.issubset(allowed) or members.intersection(forbidden):
            raise ValueError("inner validation receipt escapes outer training slice")


def _verify_candidate_score_contract(
    score: CandidateScoreReceipt,
    *,
    candidate: ModelCandidateTemplate,
    selection_spec: NestedPurgedSelectionSpec,
    outer_fold_id: str,
    outer_training_slice: OuterTrainingSliceReceipt,
    inner_validation_spec: ValidationSpec,
    inner_validation_receipt: ValidationReceipt,
) -> None:
    if (
        score.candidate_id != candidate.candidate_id
        or score.candidate_template_hash != candidate.content_hash
        or score.complexity_rank != candidate.complexity_rank
        or score.preprocess_spec_hash != candidate.preprocessing.content_hash
    ):
        raise ValueError("candidate score template contract differs")
    expected_fold_ids = tuple(item.fold_id for item in inner_validation_receipt.folds)
    if tuple(item.fold_id for item in score.fold_scores) != expected_fold_ids:
        raise ValueError("candidate score inner fold order differs")
    if not set(candidate.feature_names).issubset(
        outer_training_slice.sliced_feature_hashes
    ):
        raise ValueError("candidate score feature contract is unavailable")
    expected_model = candidate.materialize(
        selection_id=selection_spec.selection_id,
        selection_version=selection_spec.version,
        outer_fold_id=outer_fold_id,
        stage="inner_selection",
        feature_signal_hashes={
            name: outer_training_slice.sliced_feature_hashes[name]
            for name in candidate.feature_names
        },
        label_spec_hash=inner_validation_receipt.label_spec_hash,
        validation_spec_hash=inner_validation_spec.content_hash,
    )
    if expected_model.content_hash != score.materialized_model_spec_hash:
        raise ValueError("candidate score materialized model contract differs")
    scoring = selection_spec.scoring
    for fold_score, fold_receipt in zip(
        score.fold_scores,
        inner_validation_receipt.folds,
        strict=True,
    ):
        cohort = fold_score.scoring_cohort_evidence
        prediction = fold_score.prediction_rank_evidence
        if (
            cohort.fold_id != fold_receipt.fold_id
            or cohort.validation_receipt_hash != inner_validation_receipt.content_hash
            or cohort.scoring_spec_hash != selection_spec.scoring.content_hash
            or cohort.label_values_hash != outer_training_slice.sliced_label_values_hash
            or cohort.label_validity_hash
            != outer_training_slice.sliced_label_validity_hash
            or cohort.eligibility_hash != outer_training_slice.scoring_eligibility_hash
            or prediction.model_result_hash != score.inner_model_result_hash
            or prediction.predictions_hash != score.inner_model_result.predictions_hash
        ):
            raise ValueError("candidate score RankIC evidence lineage differs")
        if len(fold_score.observations) < scoring.minimum_valid_dates_per_inner_fold:
            raise ValueError("candidate score has insufficient valid RankIC dates")
        allowed_signals = set(fold_receipt.validation_signals)
        if any(
            item.signal_timestamp not in allowed_signals
            or item.observation_count < scoring.minimum_cross_sectional_observations
            for item in fold_score.observations
        ):
            raise ValueError("candidate score RankIC observation contract differs")


def _verify_outer_evaluation_contract(
    evaluation: OuterFoldEvaluationResult,
    *,
    selection: OuterFoldSelectionReceipt,
    selection_spec: NestedPurgedSelectionSpec,
    outer_validation_spec: ValidationSpec,
    outer_fold_spec: ValidationFoldSpec,
    outer_fold_receipt: SplitFoldReceipt,
    outer_validation_receipt: ValidationReceipt,
    source_feature_hashes: Mapping[str, str],
    source_label_values_hash: str,
    source_label_validity_hash: str,
    source_scoring_eligibility_hash: str,
) -> None:
    candidate = _candidate_contract_by_id(
        selection_spec,
        selection.selected_candidate_id,
    )
    if selection.selected_candidate_template_hash != candidate.content_hash:
        raise ValueError("outer evaluation selected candidate contract differs")
    projected_spec = _project_outer_validation_spec(
        selection_spec=selection_spec,
        outer_validation_spec=outer_validation_spec,
        fold=outer_fold_spec,
    )
    if evaluation.projected_outer_validation_spec_hash != projected_spec.content_hash:
        raise ValueError("outer evaluation projected specification differs")
    expected_model = candidate.materialize(
        selection_id=selection_spec.selection_id,
        selection_version=selection_spec.version,
        outer_fold_id=outer_fold_spec.fold_id,
        stage="outer_evaluation",
        feature_signal_hashes={
            name: source_feature_hashes[name] for name in candidate.feature_names
        },
        label_spec_hash=outer_validation_receipt.label_spec_hash,
        validation_spec_hash=projected_spec.content_hash,
    )
    artifact = evaluation.model_artifact
    if (
        expected_model.content_hash != evaluation.materialized_model_spec_hash
        or artifact.model_spec.to_dict() != expected_model.to_dict()
        or artifact.preprocess_spec.to_dict() != candidate.preprocessing.to_dict()
        or artifact.preprocess_spec_hash != candidate.preprocessing.content_hash
    ):
        raise ValueError("outer evaluation selected model contract differs")
    artifact_receipt = artifact.validation_receipt
    if (
        artifact.label_values_hash != source_label_values_hash
        or artifact_receipt.labels_hash != source_label_values_hash
        or artifact_receipt.label_spec_hash != outer_validation_receipt.label_spec_hash
        or artifact_receipt.windows_hash != outer_validation_receipt.windows_hash
        or artifact_receipt.calendar_hash != outer_validation_receipt.calendar_hash
        or len(artifact_receipt.folds) != 1
        or artifact_receipt.folds[0].content_hash != outer_fold_receipt.content_hash
    ):
        raise ValueError("outer evaluation label/fold lineage differs")
    if _model_result_hash(artifact.model_result) != artifact.model_result.result_hash:
        raise ValueError("outer evaluation model result hash differs")
    scoring = selection_spec.scoring
    cohort = evaluation.outer_score.scoring_cohort_evidence
    prediction = evaluation.outer_score.prediction_rank_evidence
    if (
        cohort.fold_id != outer_fold_receipt.fold_id
        or cohort.validation_receipt_hash
        != evaluation.projected_outer_validation_receipt_hash
        or cohort.scoring_spec_hash != scoring.content_hash
        or cohort.label_values_hash != source_label_values_hash
        or cohort.label_validity_hash != source_label_validity_hash
        or cohort.eligibility_hash != source_scoring_eligibility_hash
        or prediction.model_result_hash != evaluation.model_result_hash
        or prediction.predictions_hash != artifact.predictions_hash
    ):
        raise ValueError("outer evaluation RankIC evidence lineage differs")
    if len(evaluation.outer_score.observations) < (
        scoring.minimum_valid_dates_per_inner_fold
    ):
        raise ValueError("outer evaluation has insufficient valid RankIC dates")
    allowed_signals = set(outer_fold_receipt.validation_signals)
    if any(
        item.signal_timestamp not in allowed_signals
        or item.observation_count < scoring.minimum_cross_sectional_observations
        for item in evaluation.outer_score.observations
    ):
        raise ValueError("outer evaluation RankIC observation contract differs")


def _candidate_contract_by_id(
    spec: NestedPurgedSelectionSpec,
    candidate_id: str,
) -> ModelCandidateTemplate:
    matching = tuple(
        item for item in spec.candidates if item.candidate_id == candidate_id
    )
    if len(matching) != 1:
        raise ValueError("selection candidate contract is unavailable")
    return matching[0]


def _project_outer_validation_spec(
    *,
    selection_spec: NestedPurgedSelectionSpec,
    outer_validation_spec: ValidationSpec,
    fold: ValidationFoldSpec,
) -> ValidationSpec:
    return ValidationSpec(
        validation_id=(
            f"{outer_validation_spec.validation_id}:nested:"
            f"{selection_spec.selection_id}:{fold.fold_id}"
        ),
        version=selection_spec.version,
        folds=(fold,),
        embargo_sessions=outer_validation_spec.embargo_sessions,
        purge_overlapping_labels=outer_validation_spec.purge_overlapping_labels,
        min_train_signals=outer_validation_spec.min_train_signals,
        min_validation_signals=outer_validation_spec.min_validation_signals,
        train_role=outer_validation_spec.train_role,
        validation_role=outer_validation_spec.validation_role,
        method=outer_validation_spec.method,
    )


def _hash_mapping(value: Mapping[str, str], name: str) -> dict[str, str]:
    normalized = dict(sorted(value.items()))
    if not normalized or not all(
        isinstance(key, str) and key.strip() and isinstance(item, str)
        for key, item in normalized.items()
    ):
        raise TypeError(f"{name} must be a non-empty string mapping")
    for key, item in normalized.items():
        require_sha256(item, name=f"{name}:{key}")
    return normalized


def _candidate_score_membership(
    score: CandidateScoreReceipt,
) -> tuple[tuple[str, tuple[tuple[str, int, str], ...]], ...]:
    return tuple(
        (
            fold.fold_id,
            tuple(
                (
                    item.signal_timestamp,
                    item.observation_count,
                    item.security_membership_hash,
                )
                for item in fold.observations
            ),
        )
        for fold in score.fold_scores
    )


def _candidate_score_cohort_hashes(
    score: CandidateScoreReceipt,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (fold.fold_id, fold.scoring_cohort_evidence_hash) for fold in score.fold_scores
    )


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _mapping_list(value: object, name: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) and all(isinstance(key, str) for key in item)
        for item in value
    ):
        raise TypeError(f"{name} must be an object array")
    return tuple(cast(Mapping[str, object], item) for item in value)


def _text_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{name} must be a text array")
    return tuple(value)


def _string_mapping(value: object, name: str) -> dict[str, str]:
    mapping = _mapping(value, name)
    if not all(
        isinstance(key, str) and isinstance(item, str) for key, item in mapping.items()
    ):
        raise TypeError(f"{name} must be a string object")
    return {str(key): cast(str, item) for key, item in mapping.items()}


__all__ = [
    "CandidateScoreReceipt",
    "FoldRankICReceipt",
    "NestedSelectionResult",
    "OuterFoldEvaluationResult",
    "OuterFoldSelectionReceipt",
    "OuterTrainingSliceReceipt",
    "RankICObservation",
]
