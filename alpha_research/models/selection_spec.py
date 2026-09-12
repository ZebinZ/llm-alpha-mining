from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Mapping, cast

from alpha_research.core.hashing import hash_json, require_sha256
from alpha_research.models.preprocessing import (
    ModelPreprocessKind,
    ModelPreprocessSpec,
)
from alpha_research.models.spec import ModelEstimator, ModelSpec
from alpha_research.validation import ValidationSpec


MAXIMUM_NESTED_SELECTION_CANDIDATES = 12
MAXIMUM_MODEL_CANDIDATE_FEATURES = 1_024
MODEL_CANDIDATE_COMPLEXITY_POLICY = "model-candidate-complexity-policy/v1"
NESTED_SELECTION_TIE_BREAK = "score_desc_complexity_asc_candidate_id_asc"
_MATERIALIZATION_STAGES = frozenset({"inner_selection", "outer_evaluation"})
_IDENTITY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_ESTIMATOR_COMPLEXITY_TIER = {
    ModelEstimator.FACTOR_ONLY: 0,
    ModelEstimator.OLS: 1,
    ModelEstimator.RIDGE: 2,
    ModelEstimator.ELASTIC_NET: 3,
    ModelEstimator.EXTRA_TREES: 4,
}
_PREPROCESS_COMPLEXITY_TIER = {
    ModelPreprocessKind.IDENTITY: 0,
    ModelPreprocessKind.STANDARDIZE: 1,
}
_MAXIMUM_TREE_ESTIMATORS = 2_048
_MAXIMUM_TREE_DEPTH = 64
_LEAF_GRANULARITY_LIMIT = MAXIMUM_MODEL_CANDIDATE_FEATURES


def _encode_tree_complexity(
    *,
    n_estimators: int,
    max_depth: int,
    effective_features: int,
    leaf_granularity: int,
) -> int:
    return (
        (n_estimators * (_MAXIMUM_TREE_DEPTH + 1) + max_depth)
        * (MAXIMUM_MODEL_CANDIDATE_FEATURES + 1)
        + effective_features
    ) * (_LEAF_GRANULARITY_LIMIT + 1) + leaf_granularity


_MAXIMUM_TREE_COMPLEXITY = _encode_tree_complexity(
    n_estimators=_MAXIMUM_TREE_ESTIMATORS,
    max_depth=_MAXIMUM_TREE_DEPTH,
    effective_features=MAXIMUM_MODEL_CANDIDATE_FEATURES,
    leaf_granularity=_LEAF_GRANULARITY_LIMIT,
)
_PREPROCESS_COMPLEXITY_STRIDE = _MAXIMUM_TREE_COMPLEXITY + 1
_FEATURE_COMPLEXITY_STRIDE = (
    len(_PREPROCESS_COMPLEXITY_TIER) * _PREPROCESS_COMPLEXITY_STRIDE
)
_ESTIMATOR_COMPLEXITY_STRIDE = (
    MAXIMUM_MODEL_CANDIDATE_FEATURES + 1
) * _FEATURE_COMPLEXITY_STRIDE


def derive_model_candidate_complexity_rank(
    *,
    estimator: ModelEstimator | str,
    feature_count: int,
    preprocessing: ModelPreprocessSpec,
    hyperparameters: Mapping[str, float | int],
) -> int:
    """Return the deterministic structural-complexity ordinal for policy v1.

    Lower values win a score tie.  The ordering is estimator family, feature
    count, preprocessing, then ExtraTrees resource shape.  Regularization
    strengths (including ``alpha`` and ``l1_ratio``) and convergence budgets
    are deliberately excluded because they do not change model structure.
    """

    normalized_estimator = ModelEstimator(estimator)
    if (
        not isinstance(feature_count, int)
        or isinstance(feature_count, bool)
        or not 1 <= feature_count <= MAXIMUM_MODEL_CANDIDATE_FEATURES
    ):
        raise ValueError("model candidate feature_count exceeds complexity policy")
    if not isinstance(preprocessing, ModelPreprocessSpec):
        raise TypeError("model candidate preprocessing type differs")
    preprocess_kind = preprocessing.kind
    if not isinstance(preprocess_kind, ModelPreprocessKind):  # pragma: no cover
        raise RuntimeError("model preprocessing kind was not normalized")

    tree_complexity = 0
    if normalized_estimator is ModelEstimator.EXTRA_TREES:
        n_estimators = _positive_integer_parameter(hyperparameters, "n_estimators")
        max_depth = _positive_integer_parameter(hyperparameters, "max_depth")
        min_samples_leaf = _positive_integer_parameter(
            hyperparameters, "min_samples_leaf"
        )
        if n_estimators > _MAXIMUM_TREE_ESTIMATORS:
            raise ValueError("extra-trees n_estimators exceeds complexity policy")
        if max_depth > _MAXIMUM_TREE_DEPTH:
            raise ValueError("extra-trees max_depth exceeds complexity policy")
        maximum_features = hyperparameters.get("max_features")
        if (
            not isinstance(maximum_features, (int, float))
            or isinstance(maximum_features, bool)
            or not 0 < float(maximum_features) <= 1
        ):
            raise ValueError("extra-trees max_features differs for complexity policy")
        maximum_features_value = float(maximum_features)
        numerator, denominator = maximum_features_value.as_integer_ratio()
        effective_features = max(1, feature_count * numerator // denominator)
        leaf_granularity = min(
            _LEAF_GRANULARITY_LIMIT,
            max(
                1,
                (_LEAF_GRANULARITY_LIMIT + min_samples_leaf - 1) // min_samples_leaf,
            ),
        )
        tree_complexity = _encode_tree_complexity(
            n_estimators=n_estimators,
            max_depth=max_depth,
            effective_features=effective_features,
            leaf_granularity=leaf_granularity,
        )

    return (
        _ESTIMATOR_COMPLEXITY_TIER[normalized_estimator] * _ESTIMATOR_COMPLEXITY_STRIDE
        + feature_count * _FEATURE_COMPLEXITY_STRIDE
        + _PREPROCESS_COMPLEXITY_TIER[preprocess_kind] * _PREPROCESS_COMPLEXITY_STRIDE
        + tree_complexity
    )


def _positive_integer_parameter(
    hyperparameters: Mapping[str, float | int], name: str
) -> int:
    value = hyperparameters.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"extra-trees {name} differs for complexity policy")
    return value


@dataclass(frozen=True, slots=True)
class RankICSelectionSpec:
    minimum_cross_sectional_observations: int = 5
    minimum_valid_dates_per_inner_fold: int = 1
    correlation_method: str = "spearman"
    date_weighting: str = "equal"
    fold_weighting: str = "equal"
    sign_policy: str = "signed_no_flip"
    schema_version: str = "rank-ic-selection-spec/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "rank-ic-selection-spec/v1":
            raise ValueError("unsupported RankIC selection schema")
        for name in (
            "minimum_cross_sectional_observations",
            "minimum_valid_dates_per_inner_fold",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"RankIC selection {name} must be positive")
        if self.minimum_cross_sectional_observations < 2:
            raise ValueError("RankIC selection requires at least two securities")
        if self.correlation_method != "spearman":
            raise ValueError("RankIC selection method must be spearman")
        if self.date_weighting != "equal" or self.fold_weighting != "equal":
            raise ValueError("RankIC selection weighting must be equal")
        if self.sign_policy != "signed_no_flip":
            raise ValueError("RankIC selection may not flip or absolutize scores")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "minimum_cross_sectional_observations": (
                self.minimum_cross_sectional_observations
            ),
            "minimum_valid_dates_per_inner_fold": (
                self.minimum_valid_dates_per_inner_fold
            ),
            "correlation_method": self.correlation_method,
            "date_weighting": self.date_weighting,
            "fold_weighting": self.fold_weighting,
            "sign_policy": self.sign_policy,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RankICSelectionSpec":
        expected = {
            "schema_version",
            "minimum_cross_sectional_observations",
            "minimum_valid_dates_per_inner_fold",
            "correlation_method",
            "date_weighting",
            "fold_weighting",
            "sign_policy",
        }
        if set(value) != expected:
            raise ValueError("RankICSelectionSpec wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            minimum_cross_sectional_observations=_integer(
                value["minimum_cross_sectional_observations"],
                "minimum_cross_sectional_observations",
            ),
            minimum_valid_dates_per_inner_fold=_integer(
                value["minimum_valid_dates_per_inner_fold"],
                "minimum_valid_dates_per_inner_fold",
            ),
            correlation_method=_text(value["correlation_method"], "correlation_method"),
            date_weighting=_text(value["date_weighting"], "date_weighting"),
            fold_weighting=_text(value["fold_weighting"], "fold_weighting"),
            sign_policy=_text(value["sign_policy"], "sign_policy"),
        )


@dataclass(frozen=True, slots=True)
class ModelCandidateTemplate:
    candidate_id: str
    estimator: ModelEstimator | str
    feature_names: tuple[str, ...]
    hyperparameters: Mapping[str, float | int]
    preprocessing: ModelPreprocessSpec
    fit_intercept: bool
    random_seed: int
    complexity_rank: int
    schema_version: str = "model-candidate-template/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "model-candidate-template/v1":
            raise ValueError("unsupported model candidate template schema")
        _require_identity(self.candidate_id, "model candidate id")
        estimator = ModelEstimator(self.estimator)
        object.__setattr__(self, "estimator", estimator)
        features = tuple(self.feature_names)
        if (
            not features
            or any(not isinstance(name, str) or not name.strip() for name in features)
            or features != tuple(sorted(set(features)))
        ):
            raise ValueError("model candidate feature names must be sorted and unique")
        object.__setattr__(self, "feature_names", features)
        if not isinstance(self.hyperparameters, Mapping):
            raise TypeError("model candidate hyperparameters must be a numeric object")
        parameters = dict(self.hyperparameters)
        if not all(
            isinstance(name, str)
            and name.strip()
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            for name, value in parameters.items()
        ):
            raise TypeError("model candidate hyperparameters must be numeric")
        if not isinstance(self.preprocessing, ModelPreprocessSpec):
            raise TypeError("model candidate preprocessing type differs")
        if not isinstance(self.fit_intercept, bool):
            raise TypeError("model candidate fit_intercept must be boolean")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise TypeError("model candidate random_seed must be an integer")
        if (
            not isinstance(self.complexity_rank, int)
            or isinstance(self.complexity_rank, bool)
            or self.complexity_rank < 0
        ):
            raise ValueError("model candidate complexity_rank must be non-negative")
        # Reuse the authoritative model contract rather than duplicating its
        # estimator-specific parameter and resource rules.
        authoritative_spec = ModelSpec(
            model_id=f"candidate-contract:{self.candidate_id}",
            version="1",
            estimator=estimator,
            feature_signal_hashes={name: "0" * 64 for name in features},
            label_spec_hash="1" * 64,
            validation_spec_hash="2" * 64,
            hyperparameters=parameters,
            fit_intercept=self.fit_intercept,
            random_seed=self.random_seed,
        )
        object.__setattr__(
            self,
            "hyperparameters",
            MappingProxyType(dict(authoritative_spec.hyperparameters)),
        )
        object.__setattr__(
            self,
            "complexity_rank",
            derive_model_candidate_complexity_rank(
                estimator=estimator,
                feature_count=len(features),
                preprocessing=self.preprocessing,
                hyperparameters=authoritative_spec.hyperparameters,
            ),
        )

    @property
    def semantic_hash(self) -> str:
        estimator = self.estimator
        if not isinstance(estimator, ModelEstimator):  # pragma: no cover
            raise RuntimeError("model candidate estimator was not normalized")
        return cast(
            str,
            hash_json(
                {
                    "estimator": estimator.value,
                    "feature_names": list(self.feature_names),
                    "hyperparameters": dict(self.hyperparameters),
                    "preprocessing": self.preprocessing.to_dict(),
                    "fit_intercept": self.fit_intercept,
                }
            ),
        )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        estimator = self.estimator
        if not isinstance(estimator, ModelEstimator):  # pragma: no cover
            raise RuntimeError("model candidate estimator was not normalized")
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "estimator": estimator.value,
            "feature_names": list(self.feature_names),
            "hyperparameters": dict(self.hyperparameters),
            "preprocessing": self.preprocessing.to_dict(),
            "fit_intercept": self.fit_intercept,
            "random_seed": self.random_seed,
            "complexity_rank": self.complexity_rank,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ModelCandidateTemplate":
        expected = {
            "schema_version",
            "candidate_id",
            "estimator",
            "feature_names",
            "hyperparameters",
            "preprocessing",
            "fit_intercept",
            "random_seed",
            "complexity_rank",
        }
        if set(value) != expected:
            raise ValueError("ModelCandidateTemplate wire fields differ")
        raw_parameters = _mapping(value["hyperparameters"], "hyperparameters")
        if not all(
            isinstance(name, str)
            and isinstance(item, (int, float))
            and not isinstance(item, bool)
            for name, item in raw_parameters.items()
        ):
            raise TypeError("model candidate hyperparameters must be numeric")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            candidate_id=_text(value["candidate_id"], "candidate_id"),
            estimator=_text(value["estimator"], "estimator"),
            feature_names=_text_tuple(value["feature_names"], "feature_names"),
            hyperparameters={
                str(name): cast(float | int, item)
                for name, item in raw_parameters.items()
            },
            preprocessing=ModelPreprocessSpec.from_mapping(
                _mapping(value["preprocessing"], "preprocessing")
            ),
            fit_intercept=_boolean(value["fit_intercept"], "fit_intercept"),
            random_seed=_integer(value["random_seed"], "random_seed"),
            complexity_rank=_integer(value["complexity_rank"], "complexity_rank"),
        )

    def materialize(
        self,
        *,
        selection_id: str,
        selection_version: str,
        outer_fold_id: str,
        stage: str,
        feature_signal_hashes: Mapping[str, str],
        label_spec_hash: str,
        validation_spec_hash: str,
    ) -> ModelSpec:
        _require_identity(selection_id, "materialized selection id")
        _require_identity(selection_version, "materialized selection version")
        _require_identity(outer_fold_id, "materialized outer fold id")
        if stage not in _MATERIALIZATION_STAGES:
            raise ValueError("materialized candidate stage differs")
        hashes = dict(feature_signal_hashes)
        if set(hashes) != set(self.feature_names):
            raise ValueError("materialized candidate feature names differ")
        return ModelSpec(
            model_id=(f"{selection_id}:{outer_fold_id}:{stage}:{self.candidate_id}"),
            version=selection_version,
            estimator=self.estimator,
            feature_signal_hashes={name: hashes[name] for name in self.feature_names},
            label_spec_hash=label_spec_hash,
            validation_spec_hash=validation_spec_hash,
            hyperparameters=dict(self.hyperparameters),
            fit_intercept=self.fit_intercept,
            random_seed=self.random_seed,
        )


@dataclass(frozen=True, slots=True)
class OuterFoldSelectionPlan:
    outer_fold_id: str
    inner_validation_spec: ValidationSpec
    schema_version: str = "outer-fold-selection-plan/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "outer-fold-selection-plan/v1":
            raise ValueError("unsupported outer fold selection plan schema")
        _require_identity(self.outer_fold_id, "outer fold selection plan id")
        if not isinstance(self.inner_validation_spec, ValidationSpec):
            raise TypeError("inner validation specification type differs")
        _require_identity(
            self.inner_validation_spec.validation_id,
            "inner validation id",
        )
        _require_identity(
            self.inner_validation_spec.version,
            "inner validation version",
        )
        for fold in self.inner_validation_spec.folds:
            _require_identity(fold.fold_id, "inner validation fold id")
        if len(self.inner_validation_spec.folds) < 2:
            raise ValueError("nested selection requires at least two inner folds")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "outer_fold_id": self.outer_fold_id,
            "inner_validation_spec": self.inner_validation_spec.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "OuterFoldSelectionPlan":
        expected = {
            "schema_version",
            "outer_fold_id",
            "inner_validation_spec",
        }
        if set(value) != expected:
            raise ValueError("OuterFoldSelectionPlan wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            outer_fold_id=_text(value["outer_fold_id"], "outer_fold_id"),
            inner_validation_spec=ValidationSpec.from_mapping(
                _mapping(value["inner_validation_spec"], "inner_validation_spec")
            ),
        )


@dataclass(frozen=True, slots=True)
class NestedPurgedSelectionSpec:
    selection_id: str
    version: str
    outer_validation_spec_hash: str
    candidates: tuple[ModelCandidateTemplate, ...]
    outer_plans: tuple[OuterFoldSelectionPlan, ...]
    scoring: RankICSelectionSpec
    maximum_fold_evaluations: int
    maximum_outer_evaluations: int
    complexity_policy: str = MODEL_CANDIDATE_COMPLEXITY_POLICY
    tie_break_policy: str = NESTED_SELECTION_TIE_BREAK
    schema_version: str = "nested-purged-selection-spec/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "nested-purged-selection-spec/v1":
            raise ValueError("unsupported nested purged selection schema")
        _require_identity(self.selection_id, "nested selection id")
        _require_identity(self.version, "nested selection version")
        require_sha256(
            self.outer_validation_spec_hash,
            name="nested selection outer_validation_spec_hash",
        )
        candidates = tuple(self.candidates)
        if not all(isinstance(item, ModelCandidateTemplate) for item in candidates):
            raise TypeError("nested selection candidates have invalid types")
        candidates = tuple(sorted(candidates, key=lambda item: item.candidate_id))
        if not candidates:
            raise ValueError("nested selection requires candidates")
        if len(candidates) > MAXIMUM_NESTED_SELECTION_CANDIDATES:
            raise ValueError("nested selection candidate count exceeds 12")
        if len({item.candidate_id for item in candidates}) != len(candidates):
            raise ValueError("nested selection candidate ids must be unique")
        if len({item.semantic_hash for item in candidates}) != len(candidates):
            raise ValueError("nested selection candidate semantics must be unique")
        if len({item.random_seed for item in candidates}) != 1:
            raise ValueError("nested selection candidate random_seed must be identical")
        object.__setattr__(self, "candidates", candidates)
        plans = tuple(self.outer_plans)
        if not plans or not all(
            isinstance(item, OuterFoldSelectionPlan) for item in plans
        ):
            raise TypeError("nested selection outer plans have invalid types")
        if len({item.outer_fold_id for item in plans}) != len(plans):
            raise ValueError("nested selection outer fold ids must be unique")
        object.__setattr__(self, "outer_plans", plans)
        if not isinstance(self.scoring, RankICSelectionSpec):
            raise TypeError("nested selection scoring type differs")
        for name in ("maximum_fold_evaluations", "maximum_outer_evaluations"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"nested selection {name} must be positive")
        if self.maximum_outer_evaluations != len(plans):
            raise ValueError("nested selection outer evaluation budget differs")
        if self.maximum_fold_evaluations < self.required_fold_evaluations:
            raise ValueError("nested selection fold evaluation budget is insufficient")
        if self.complexity_policy != MODEL_CANDIDATE_COMPLEXITY_POLICY:
            raise ValueError("nested selection complexity policy differs")
        if self.tie_break_policy != NESTED_SELECTION_TIE_BREAK:
            raise ValueError("nested selection tie-break policy differs")

    @property
    def required_fold_evaluations(self) -> int:
        inner = sum(
            len(self.candidates) * len(plan.inner_validation_spec.folds)
            for plan in self.outer_plans
        )
        return inner + len(self.outer_plans)

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "selection_id": self.selection_id,
            "version": self.version,
            "outer_validation_spec_hash": self.outer_validation_spec_hash,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "outer_plans": [plan.to_dict() for plan in self.outer_plans],
            "scoring": self.scoring.to_dict(),
            "maximum_fold_evaluations": self.maximum_fold_evaluations,
            "maximum_outer_evaluations": self.maximum_outer_evaluations,
            "complexity_policy": self.complexity_policy,
            "tie_break_policy": self.tie_break_policy,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "NestedPurgedSelectionSpec":
        expected = {
            "schema_version",
            "selection_id",
            "version",
            "outer_validation_spec_hash",
            "candidates",
            "outer_plans",
            "scoring",
            "maximum_fold_evaluations",
            "maximum_outer_evaluations",
            "complexity_policy",
            "tie_break_policy",
        }
        fields = set(value)
        legacy_expected = expected - {"complexity_policy"}
        if fields != expected and fields != legacy_expected:
            raise ValueError("NestedPurgedSelectionSpec wire fields differ")
        candidates = _mapping_list(value["candidates"], "candidates")
        plans = _mapping_list(value["outer_plans"], "outer_plans")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            selection_id=_text(value["selection_id"], "selection_id"),
            version=_text(value["version"], "version"),
            outer_validation_spec_hash=_text(
                value["outer_validation_spec_hash"],
                "outer_validation_spec_hash",
            ),
            candidates=tuple(
                ModelCandidateTemplate.from_mapping(item) for item in candidates
            ),
            outer_plans=tuple(
                OuterFoldSelectionPlan.from_mapping(item) for item in plans
            ),
            scoring=RankICSelectionSpec.from_mapping(
                _mapping(value["scoring"], "scoring")
            ),
            maximum_fold_evaluations=_integer(
                value["maximum_fold_evaluations"], "maximum_fold_evaluations"
            ),
            maximum_outer_evaluations=_integer(
                value["maximum_outer_evaluations"], "maximum_outer_evaluations"
            ),
            complexity_policy=(
                _text(value["complexity_policy"], "complexity_policy")
                if "complexity_policy" in value
                else MODEL_CANDIDATE_COMPLEXITY_POLICY
            ),
            tie_break_policy=_text(value["tie_break_policy"], "tie_break_policy"),
        )


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    return value


def _require_identity(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    if value != value.strip() or _IDENTITY_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{name} must match [A-Za-z0-9][A-Za-z0-9._-]* without delimiters"
        )
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


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


__all__ = [
    "MAXIMUM_MODEL_CANDIDATE_FEATURES",
    "MAXIMUM_NESTED_SELECTION_CANDIDATES",
    "MODEL_CANDIDATE_COMPLEXITY_POLICY",
    "NESTED_SELECTION_TIE_BREAK",
    "ModelCandidateTemplate",
    "NestedPurgedSelectionSpec",
    "OuterFoldSelectionPlan",
    "RankICSelectionSpec",
    "derive_model_candidate_complexity_rank",
]
