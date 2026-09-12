from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isclose, isfinite, sqrt
from typing import Mapping, cast

import pandas as pd

from alpha_research.core.hashing import hash_json, require_sha256


def _mean(values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError("mean requires values")
    return fsum(values) / len(values)


def _canonical_timestamp(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return cast(str, timestamp.isoformat())


def _identity(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty canonical text")
    return value


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    return cast(str, require_sha256(value, name=name))


def _average_rank_vector(value: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a numeric sequence")
    if len(value) < 2:
        raise ValueError(f"{name} must contain at least two ranks")
    ranks: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise TypeError(f"{name} must contain only numbers")
        rank = float(item)
        if not isfinite(rank):
            raise ValueError(f"{name} must contain only finite ranks")
        ranks.append(rank)

    # Validate the exact shape of an average-rank vector, including ties.  For
    # a tie group occupying ordinal positions p..q, every member must equal
    # (p + q) / 2.  This is stronger than checking only range and total sum.
    ordered = sorted(ranks)
    start = 1
    cursor = 0
    while cursor < len(ordered):
        rank = ordered[cursor]
        end_cursor = cursor + 1
        while end_cursor < len(ordered) and ordered[end_cursor] == rank:
            end_cursor += 1
        count = end_cursor - cursor
        expected = (start + (start + count - 1)) / 2.0
        if rank != expected:
            raise ValueError(f"{name} is not a canonical average-rank vector")
        start += count
        cursor = end_cursor
    return tuple(ranks)


def _security_members(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not all(
        isinstance(item, str) for item in value
    ):
        raise TypeError(f"{name} must be a text sequence")
    members = tuple(cast(str, item) for item in value)
    if len(members) < 2:
        raise ValueError(f"{name} must contain at least two securities")
    if any(not item or item != item.strip() for item in members):
        raise ValueError(f"{name} contains a non-canonical security id")
    if members != tuple(sorted(set(members))):
        raise ValueError(f"{name} must be sorted and unique")
    return members


@dataclass(frozen=True, slots=True)
class ScoringCohortRankVector:
    signal_timestamp: str
    security_members: tuple[str, ...]
    target_ranks: tuple[float, ...]
    schema_version: str = "scoring-cohort-rank-vector/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "scoring-cohort-rank-vector/v1":
            raise ValueError("unsupported scoring cohort rank vector schema")
        object.__setattr__(
            self,
            "signal_timestamp",
            _canonical_timestamp(
                self.signal_timestamp, name="scoring cohort signal_timestamp"
            ),
        )
        members = _security_members(
            self.security_members, name="scoring cohort security_members"
        )
        ranks = _average_rank_vector(
            self.target_ranks, name="scoring cohort target_ranks"
        )
        if len(members) != len(ranks):
            raise ValueError("scoring cohort members and target ranks differ")
        object.__setattr__(self, "security_members", members)
        object.__setattr__(self, "target_ranks", ranks)

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "signal_timestamp": self.signal_timestamp,
            "security_members": list(self.security_members),
            "target_ranks": list(self.target_ranks),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ScoringCohortRankVector":
        expected = {
            "schema_version",
            "signal_timestamp",
            "security_members",
            "target_ranks",
        }
        if set(value) != expected:
            raise ValueError("ScoringCohortRankVector wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            signal_timestamp=_text(value["signal_timestamp"], "signal_timestamp"),
            security_members=_text_tuple(value["security_members"], "security_members"),
            target_ranks=_number_tuple(value["target_ranks"], "target_ranks"),
        )


@dataclass(frozen=True, slots=True)
class PredictionRankVector:
    signal_timestamp: str
    security_members: tuple[str, ...]
    prediction_ranks: tuple[float, ...]
    schema_version: str = "prediction-rank-vector/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "prediction-rank-vector/v1":
            raise ValueError("unsupported prediction rank vector schema")
        object.__setattr__(
            self,
            "signal_timestamp",
            _canonical_timestamp(
                self.signal_timestamp, name="prediction rank signal_timestamp"
            ),
        )
        members = _security_members(
            self.security_members, name="prediction rank security_members"
        )
        ranks = _average_rank_vector(
            self.prediction_ranks, name="prediction rank prediction_ranks"
        )
        if len(members) != len(ranks):
            raise ValueError("prediction rank members and ranks differ")
        object.__setattr__(self, "security_members", members)
        object.__setattr__(self, "prediction_ranks", ranks)

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "signal_timestamp": self.signal_timestamp,
            "security_members": list(self.security_members),
            "prediction_ranks": list(self.prediction_ranks),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PredictionRankVector":
        expected = {
            "schema_version",
            "signal_timestamp",
            "security_members",
            "prediction_ranks",
        }
        if set(value) != expected:
            raise ValueError("PredictionRankVector wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            signal_timestamp=_text(value["signal_timestamp"], "signal_timestamp"),
            security_members=_text_tuple(value["security_members"], "security_members"),
            prediction_ranks=_number_tuple(
                value["prediction_ranks"], "prediction_ranks"
            ),
        )


@dataclass(frozen=True, slots=True)
class FoldScoringCohortEvidence:
    fold_id: str
    validation_receipt_hash: str
    scoring_spec_hash: str
    label_values_hash: str
    label_validity_hash: str
    eligibility_hash: str
    rank_vectors: tuple[ScoringCohortRankVector, ...]
    schema_version: str = "fold-scoring-cohort-evidence/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "fold-scoring-cohort-evidence/v1":
            raise ValueError("unsupported fold scoring cohort evidence schema")
        object.__setattr__(self, "fold_id", _identity(self.fold_id, name="fold_id"))
        for name in (
            "validation_receipt_hash",
            "scoring_spec_hash",
            "label_values_hash",
            "label_validity_hash",
            "eligibility_hash",
        ):
            _digest(getattr(self, name), name=f"scoring cohort {name}")
        vectors = tuple(self.rank_vectors)
        if not vectors or not all(
            isinstance(item, ScoringCohortRankVector) for item in vectors
        ):
            raise TypeError("scoring cohort rank_vectors have invalid types")
        timestamps = tuple(item.signal_timestamp for item in vectors)
        if timestamps != tuple(sorted(set(timestamps))):
            raise ValueError("scoring cohort rank vectors must be timestamp sorted")
        object.__setattr__(self, "rank_vectors", vectors)

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fold_id": self.fold_id,
            "validation_receipt_hash": self.validation_receipt_hash,
            "scoring_spec_hash": self.scoring_spec_hash,
            "label_values_hash": self.label_values_hash,
            "label_validity_hash": self.label_validity_hash,
            "eligibility_hash": self.eligibility_hash,
            "rank_vectors": [item.to_dict() for item in self.rank_vectors],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FoldScoringCohortEvidence":
        expected = {
            "schema_version",
            "fold_id",
            "validation_receipt_hash",
            "scoring_spec_hash",
            "label_values_hash",
            "label_validity_hash",
            "eligibility_hash",
            "rank_vectors",
        }
        if set(value) != expected:
            raise ValueError("FoldScoringCohortEvidence wire fields differ")
        vectors = _mapping_tuple(value["rank_vectors"], "rank_vectors")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            fold_id=_text(value["fold_id"], "fold_id"),
            validation_receipt_hash=_text(
                value["validation_receipt_hash"], "validation_receipt_hash"
            ),
            scoring_spec_hash=_text(value["scoring_spec_hash"], "scoring_spec_hash"),
            label_values_hash=_text(value["label_values_hash"], "label_values_hash"),
            label_validity_hash=_text(
                value["label_validity_hash"], "label_validity_hash"
            ),
            eligibility_hash=_text(value["eligibility_hash"], "eligibility_hash"),
            rank_vectors=tuple(
                ScoringCohortRankVector.from_mapping(item) for item in vectors
            ),
        )


@dataclass(frozen=True, slots=True)
class FoldPredictionRankEvidence:
    fold_id: str
    cohort_evidence_hash: str
    model_result_hash: str
    predictions_hash: str
    rank_vectors: tuple[PredictionRankVector, ...]
    schema_version: str = "fold-prediction-rank-evidence/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "fold-prediction-rank-evidence/v1":
            raise ValueError("unsupported fold prediction rank evidence schema")
        object.__setattr__(self, "fold_id", _identity(self.fold_id, name="fold_id"))
        for name in (
            "cohort_evidence_hash",
            "model_result_hash",
            "predictions_hash",
        ):
            _digest(getattr(self, name), name=f"prediction rank {name}")
        vectors = tuple(self.rank_vectors)
        if not vectors or not all(
            isinstance(item, PredictionRankVector) for item in vectors
        ):
            raise TypeError("prediction rank_vectors have invalid types")
        timestamps = tuple(item.signal_timestamp for item in vectors)
        if timestamps != tuple(sorted(set(timestamps))):
            raise ValueError("prediction rank vectors must be timestamp sorted")
        object.__setattr__(self, "rank_vectors", vectors)

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fold_id": self.fold_id,
            "cohort_evidence_hash": self.cohort_evidence_hash,
            "model_result_hash": self.model_result_hash,
            "predictions_hash": self.predictions_hash,
            "rank_vectors": [item.to_dict() for item in self.rank_vectors],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FoldPredictionRankEvidence":
        expected = {
            "schema_version",
            "fold_id",
            "cohort_evidence_hash",
            "model_result_hash",
            "predictions_hash",
            "rank_vectors",
        }
        if set(value) != expected:
            raise ValueError("FoldPredictionRankEvidence wire fields differ")
        vectors = _mapping_tuple(value["rank_vectors"], "rank_vectors")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            fold_id=_text(value["fold_id"], "fold_id"),
            cohort_evidence_hash=_text(
                value["cohort_evidence_hash"], "cohort_evidence_hash"
            ),
            model_result_hash=_text(value["model_result_hash"], "model_result_hash"),
            predictions_hash=_text(value["predictions_hash"], "predictions_hash"),
            rank_vectors=tuple(
                PredictionRankVector.from_mapping(item) for item in vectors
            ),
        )


@dataclass(frozen=True, slots=True)
class DailySpearmanEvidence:
    signal_timestamp: str
    observation_count: int
    security_membership_hash: str
    target_ranks_hash: str
    prediction_ranks_hash: str
    rank_ic: float
    schema_version: str = "daily-spearman-evidence/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "daily-spearman-evidence/v1":
            raise ValueError("unsupported daily Spearman evidence schema")
        object.__setattr__(
            self,
            "signal_timestamp",
            _canonical_timestamp(
                self.signal_timestamp, name="daily Spearman signal_timestamp"
            ),
        )
        if (
            not isinstance(self.observation_count, int)
            or isinstance(self.observation_count, bool)
            or self.observation_count < 2
        ):
            raise ValueError("daily Spearman observation_count must be at least two")
        for name in (
            "security_membership_hash",
            "target_ranks_hash",
            "prediction_ranks_hash",
        ):
            _digest(getattr(self, name), name=f"daily Spearman {name}")
        value = float(self.rank_ic)
        if not isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("daily Spearman rank_ic must lie in [-1, 1]")
        object.__setattr__(self, "rank_ic", value)

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "signal_timestamp": self.signal_timestamp,
            "observation_count": self.observation_count,
            "security_membership_hash": self.security_membership_hash,
            "target_ranks_hash": self.target_ranks_hash,
            "prediction_ranks_hash": self.prediction_ranks_hash,
            "rank_ic": self.rank_ic,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "DailySpearmanEvidence":
        expected = {
            "schema_version",
            "signal_timestamp",
            "observation_count",
            "security_membership_hash",
            "target_ranks_hash",
            "prediction_ranks_hash",
            "rank_ic",
        }
        if set(value) != expected:
            raise ValueError("DailySpearmanEvidence wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            signal_timestamp=_text(value["signal_timestamp"], "signal_timestamp"),
            observation_count=_integer(value["observation_count"], "observation_count"),
            security_membership_hash=_text(
                value["security_membership_hash"], "security_membership_hash"
            ),
            target_ranks_hash=_text(value["target_ranks_hash"], "target_ranks_hash"),
            prediction_ranks_hash=_text(
                value["prediction_ranks_hash"], "prediction_ranks_hash"
            ),
            rank_ic=_number(value["rank_ic"], "rank_ic"),
        )


@dataclass(frozen=True, slots=True)
class VerifiedFoldScoringEvidence:
    fold_id: str
    cohort_evidence_hash: str
    prediction_evidence_hash: str
    daily_scores: tuple[DailySpearmanEvidence, ...]
    daily_scores_hash: str
    fold_rank_ic_mean: float
    schema_version: str = "verified-fold-scoring-evidence/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "verified-fold-scoring-evidence/v1":
            raise ValueError("unsupported verified fold scoring evidence schema")
        object.__setattr__(self, "fold_id", _identity(self.fold_id, name="fold_id"))
        _digest(self.cohort_evidence_hash, name="verified cohort_evidence_hash")
        _digest(self.prediction_evidence_hash, name="verified prediction_evidence_hash")
        scores = tuple(self.daily_scores)
        if not scores or not all(
            isinstance(item, DailySpearmanEvidence) for item in scores
        ):
            raise TypeError("verified daily_scores have invalid types")
        timestamps = tuple(item.signal_timestamp for item in scores)
        if timestamps != tuple(sorted(set(timestamps))):
            raise ValueError("verified daily scores must be timestamp sorted")
        object.__setattr__(self, "daily_scores", scores)
        _digest(self.daily_scores_hash, name="verified daily_scores_hash")
        if self.daily_scores_hash != hash_json([item.to_dict() for item in scores]):
            raise ValueError("verified daily scores hash differs")
        expected_mean = _mean(tuple(item.rank_ic for item in scores))
        mean = float(self.fold_rank_ic_mean)
        if not isfinite(mean) or mean != expected_mean:
            raise ValueError("verified fold RankIC mean differs")
        object.__setattr__(self, "fold_rank_ic_mean", mean)

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fold_id": self.fold_id,
            "cohort_evidence_hash": self.cohort_evidence_hash,
            "prediction_evidence_hash": self.prediction_evidence_hash,
            "daily_scores": [item.to_dict() for item in self.daily_scores],
            "daily_scores_hash": self.daily_scores_hash,
            "fold_rank_ic_mean": self.fold_rank_ic_mean,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "VerifiedFoldScoringEvidence":
        expected = {
            "schema_version",
            "fold_id",
            "cohort_evidence_hash",
            "prediction_evidence_hash",
            "daily_scores",
            "daily_scores_hash",
            "fold_rank_ic_mean",
        }
        if set(value) != expected:
            raise ValueError("VerifiedFoldScoringEvidence wire fields differ")
        scores = _mapping_tuple(value["daily_scores"], "daily_scores")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            fold_id=_text(value["fold_id"], "fold_id"),
            cohort_evidence_hash=_text(
                value["cohort_evidence_hash"], "cohort_evidence_hash"
            ),
            prediction_evidence_hash=_text(
                value["prediction_evidence_hash"], "prediction_evidence_hash"
            ),
            daily_scores=tuple(
                DailySpearmanEvidence.from_mapping(item) for item in scores
            ),
            daily_scores_hash=_text(value["daily_scores_hash"], "daily_scores_hash"),
            fold_rank_ic_mean=_number(value["fold_rank_ic_mean"], "fold_rank_ic_mean"),
        )


class FoldScoringEvidenceVerifier:
    """Verify lineage and derive signed daily Spearman scores from rank vectors."""

    def verify(
        self,
        cohort: FoldScoringCohortEvidence,
        prediction: FoldPredictionRankEvidence,
        *,
        expected_fold_id: str,
        expected_validation_receipt_hash: str,
        expected_scoring_spec_hash: str,
        expected_label_values_hash: str,
        expected_label_validity_hash: str,
        expected_eligibility_hash: str,
        expected_model_result_hash: str,
        expected_predictions_hash: str,
        expected_cohort_evidence_hash: str,
        expected_prediction_evidence_hash: str,
    ) -> VerifiedFoldScoringEvidence:
        if not isinstance(cohort, FoldScoringCohortEvidence):
            raise TypeError("scoring cohort evidence type differs")
        if not isinstance(prediction, FoldPredictionRankEvidence):
            raise TypeError("prediction rank evidence type differs")
        fold_id = _identity(expected_fold_id, name="expected_fold_id")
        expected = {
            "validation_receipt_hash": _digest(
                expected_validation_receipt_hash,
                name="expected_validation_receipt_hash",
            ),
            "scoring_spec_hash": _digest(
                expected_scoring_spec_hash, name="expected_scoring_spec_hash"
            ),
            "label_values_hash": _digest(
                expected_label_values_hash, name="expected_label_values_hash"
            ),
            "label_validity_hash": _digest(
                expected_label_validity_hash, name="expected_label_validity_hash"
            ),
            "eligibility_hash": _digest(
                expected_eligibility_hash, name="expected_eligibility_hash"
            ),
        }
        expected_model = _digest(
            expected_model_result_hash, name="expected_model_result_hash"
        )
        expected_predictions = _digest(
            expected_predictions_hash, name="expected_predictions_hash"
        )
        expected_cohort = _digest(
            expected_cohort_evidence_hash, name="expected_cohort_evidence_hash"
        )
        expected_prediction = _digest(
            expected_prediction_evidence_hash,
            name="expected_prediction_evidence_hash",
        )
        if cohort.fold_id != fold_id or prediction.fold_id != fold_id:
            raise ValueError("scoring evidence fold differs")
        for name, digest in expected.items():
            if getattr(cohort, name) != digest:
                raise ValueError(f"scoring cohort {name} differs")
        if cohort.content_hash != expected_cohort:
            raise ValueError("scoring cohort evidence hash differs")
        if prediction.cohort_evidence_hash != cohort.content_hash:
            raise ValueError("prediction cohort evidence hash differs")
        if prediction.model_result_hash != expected_model:
            raise ValueError("prediction model result hash differs")
        if prediction.predictions_hash != expected_predictions:
            raise ValueError("prediction values hash differs")
        if prediction.content_hash != expected_prediction:
            raise ValueError("prediction rank evidence hash differs")

        cohort_vectors = cohort.rank_vectors
        prediction_vectors = prediction.rank_vectors
        if len(cohort_vectors) != len(prediction_vectors):
            raise ValueError("scoring evidence date count differs")

        scores: list[DailySpearmanEvidence] = []
        for target, predicted in zip(cohort_vectors, prediction_vectors, strict=True):
            if target.signal_timestamp != predicted.signal_timestamp:
                raise ValueError("scoring evidence timestamp differs")
            if target.security_members != predicted.security_members:
                raise ValueError("scoring evidence security order differs")
            if len(target.target_ranks) != len(predicted.prediction_ranks):
                raise ValueError("scoring evidence rank vector length differs")
            rank_ic = _pearson_of_ranks(target.target_ranks, predicted.prediction_ranks)
            scores.append(
                DailySpearmanEvidence(
                    signal_timestamp=target.signal_timestamp,
                    observation_count=len(target.security_members),
                    security_membership_hash=hash_json(list(target.security_members)),
                    target_ranks_hash=hash_json(list(target.target_ranks)),
                    prediction_ranks_hash=hash_json(list(predicted.prediction_ranks)),
                    rank_ic=rank_ic,
                )
            )
        frozen = tuple(scores)
        return VerifiedFoldScoringEvidence(
            fold_id=fold_id,
            cohort_evidence_hash=cohort.content_hash,
            prediction_evidence_hash=prediction.content_hash,
            daily_scores=frozen,
            daily_scores_hash=hash_json([item.to_dict() for item in frozen]),
            fold_rank_ic_mean=_mean(tuple(item.rank_ic for item in frozen)),
        )


def _pearson_of_ranks(
    target_ranks: tuple[float, ...], prediction_ranks: tuple[float, ...]
) -> float:
    if len(target_ranks) != len(prediction_ranks) or len(target_ranks) < 2:
        raise ValueError("Spearman rank vectors differ")
    target_mean = _mean(target_ranks)
    prediction_mean = _mean(prediction_ranks)
    target_centered = tuple(item - target_mean for item in target_ranks)
    prediction_centered = tuple(item - prediction_mean for item in prediction_ranks)
    target_ss = fsum(item * item for item in target_centered)
    prediction_ss = fsum(item * item for item in prediction_centered)
    if target_ss == 0.0 or prediction_ss == 0.0:
        raise ValueError("Spearman rank vector has zero variance")
    covariance = fsum(
        left * right
        for left, right in zip(target_centered, prediction_centered, strict=True)
    )
    value = covariance / sqrt(target_ss * prediction_ss)
    if not isfinite(value):  # pragma: no cover - guarded by finite inputs
        raise ValueError("Spearman score is non-finite")
    if isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-15):
        return 1.0
    if isclose(value, -1.0, rel_tol=0.0, abs_tol=1e-15):
        return -1.0
    if not -1.0 <= value <= 1.0:  # pragma: no cover - numerical guard
        raise ValueError("Spearman score lies outside [-1, 1]")
    return value


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


def _text_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{name} must be a text array")
    return tuple(value)


def _number_tuple(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
    ):
        raise TypeError(f"{name} must be a numeric array")
    return tuple(float(item) for item in value)


def _mapping_tuple(value: object, name: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) and all(isinstance(key, str) for key in item)
        for item in value
    ):
        raise TypeError(f"{name} must be an object array")
    return tuple(cast(Mapping[str, object], item) for item in value)


__all__ = [
    "DailySpearmanEvidence",
    "FoldPredictionRankEvidence",
    "FoldScoringCohortEvidence",
    "FoldScoringEvidenceVerifier",
    "PredictionRankVector",
    "ScoringCohortRankVector",
    "VerifiedFoldScoringEvidence",
]
