from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Mapping, cast

from alpha_research.core.hashing import hash_json, require_sha256


class ModelEstimator(str, Enum):
    FACTOR_ONLY = "factor_only"
    OLS = "ols"
    RIDGE = "ridge"
    ELASTIC_NET = "elastic_net"
    EXTRA_TREES = "extra_trees"


def _canonical_finite_float(value: float | int, *, name: str) -> float:
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise ValueError(f"model hyperparameter {name} must be finite") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"model hyperparameter {name} must be finite")
    return 0.0 if normalized == 0.0 else normalized


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    version: str
    estimator: ModelEstimator | str
    feature_signal_hashes: Mapping[str, str]
    label_spec_hash: str
    validation_spec_hash: str
    hyperparameters: Mapping[str, float | int]
    fit_intercept: bool = True
    random_seed: int = 0
    missing_policy: str = "complete_case"
    sample_weighting: str = "equal_observation"
    schema_version: str = "model-spec/v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "estimator", ModelEstimator(self.estimator))
        features = dict(sorted(self.feature_signal_hashes.items()))
        object.__setattr__(self, "feature_signal_hashes", MappingProxyType(features))
        if self.schema_version != "model-spec/v1":
            raise ValueError("unsupported ModelSpec schema")
        if not self.model_id.strip() or not self.version.strip():
            raise ValueError("model id and version are required")
        if not features or any(not name.strip() for name in features):
            raise ValueError("model requires named feature signals")
        for name, digest in features.items():
            require_sha256(digest, name=f"model feature hash:{name}")
        require_sha256(self.label_spec_hash, name="model label_spec_hash")
        require_sha256(self.validation_spec_hash, name="model validation_spec_hash")
        if not isinstance(self.fit_intercept, bool):
            raise TypeError("model fit_intercept must be boolean")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise TypeError("model random_seed must be an integer")
        if self.missing_policy != "complete_case":
            raise ValueError("Phase 2 models require complete-case observations")
        if self.sample_weighting != "equal_observation":
            raise ValueError("unsupported model sample weighting")
        hyperparameters = self._normalize_hyperparameters(self.hyperparameters)
        object.__setattr__(self, "hyperparameters", MappingProxyType(hyperparameters))

    def _normalize_hyperparameters(
        self, hyperparameters: Mapping[str, float | int]
    ) -> dict[str, float | int]:
        estimator = self.estimator
        if not isinstance(estimator, ModelEstimator):  # pragma: no cover
            raise RuntimeError("model estimator was not normalized")
        if not isinstance(hyperparameters, Mapping):
            raise TypeError("model hyperparameters must be a numeric object")
        parameters = dict(hyperparameters)
        if not all(isinstance(name, str) and name.strip() for name in parameters):
            raise TypeError("model hyperparameter names must be non-empty text")
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in parameters.values()
        ):
            raise TypeError("model hyperparameters must be numeric, not boolean")
        for name, value in parameters.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"model hyperparameter {name} must be finite")
        expected: set[str]
        if estimator in {ModelEstimator.FACTOR_ONLY, ModelEstimator.OLS}:
            expected = set()
        elif estimator is ModelEstimator.RIDGE:
            expected = {"alpha"}
        elif estimator is ModelEstimator.ELASTIC_NET:
            expected = {"alpha", "l1_ratio", "max_iter"}
        else:
            expected = {
                "max_depth",
                "max_features",
                "min_samples_leaf",
                "n_estimators",
            }
        if set(parameters) != expected:
            raise ValueError(
                "model hyperparameter fields differ:expected="
                + ",".join(sorted(expected))
            )
        if (
            estimator is ModelEstimator.FACTOR_ONLY
            and len(self.feature_signal_hashes) != 1
        ):
            raise ValueError("factor-only baseline requires exactly one feature")
        if estimator in {ModelEstimator.RIDGE, ModelEstimator.ELASTIC_NET}:
            alpha = _canonical_finite_float(parameters["alpha"], name="alpha")
            parameters["alpha"] = alpha
            if alpha <= 0:
                raise ValueError("regularized model alpha must be positive")
        if estimator is ModelEstimator.ELASTIC_NET:
            ratio = _canonical_finite_float(parameters["l1_ratio"], name="l1_ratio")
            parameters["l1_ratio"] = ratio
            maximum_iterations = parameters["max_iter"]
            if not 0 <= ratio <= 1:
                raise ValueError("elastic-net l1_ratio must lie in [0, 1]")
            if (
                not isinstance(maximum_iterations, int)
                or isinstance(maximum_iterations, bool)
                or maximum_iterations <= 0
            ):
                raise ValueError("elastic-net max_iter must be a positive integer")
        if estimator is ModelEstimator.EXTRA_TREES:
            if self.fit_intercept:
                raise ValueError("extra-trees fit_intercept must be false")
            for name in ("n_estimators", "max_depth", "min_samples_leaf"):
                value = parameters[name]
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise ValueError(f"extra-trees {name} must be a positive integer")
            if int(parameters["n_estimators"]) > 2_048:
                raise ValueError("extra-trees n_estimators exceeds resource limit")
            if int(parameters["max_depth"]) > 64:
                raise ValueError("extra-trees max_depth exceeds resource limit")
            maximum_features = _canonical_finite_float(
                parameters["max_features"], name="max_features"
            )
            parameters["max_features"] = maximum_features
            if not 0 < maximum_features <= 1:
                raise ValueError("extra-trees max_features must lie in (0, 1]")
        return dict(sorted(parameters.items()))

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        estimator = self.estimator
        if not isinstance(estimator, ModelEstimator):  # pragma: no cover
            raise RuntimeError("model estimator was not normalized")
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "version": self.version,
            "estimator": estimator.value,
            "feature_signal_hashes": dict(self.feature_signal_hashes),
            "label_spec_hash": self.label_spec_hash,
            "validation_spec_hash": self.validation_spec_hash,
            "hyperparameters": dict(self.hyperparameters),
            "fit_intercept": self.fit_intercept,
            "random_seed": self.random_seed,
            "missing_policy": self.missing_policy,
            "sample_weighting": self.sample_weighting,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ModelSpec":
        expected = {
            "schema_version",
            "model_id",
            "version",
            "estimator",
            "feature_signal_hashes",
            "label_spec_hash",
            "validation_spec_hash",
            "hyperparameters",
            "fit_intercept",
            "random_seed",
            "missing_policy",
            "sample_weighting",
        }
        if set(value) != expected:
            raise ValueError("ModelSpec wire fields differ")
        feature_hashes = value["feature_signal_hashes"]
        hyperparameters = value["hyperparameters"]
        if not isinstance(feature_hashes, Mapping) or not all(
            isinstance(name, str) and isinstance(digest, str)
            for name, digest in feature_hashes.items()
        ):
            raise TypeError("ModelSpec feature hashes must be a string object")
        if not isinstance(hyperparameters, Mapping) or not all(
            isinstance(name, str)
            and isinstance(item, (int, float))
            and not isinstance(item, bool)
            for name, item in hyperparameters.items()
        ):
            raise TypeError("ModelSpec hyperparameters must be numeric")
        fit_intercept = value["fit_intercept"]
        random_seed = value["random_seed"]
        if not isinstance(fit_intercept, bool):
            raise TypeError("ModelSpec fit_intercept must be boolean")
        if not isinstance(random_seed, int) or isinstance(random_seed, bool):
            raise TypeError("ModelSpec random_seed must be integer")
        return cls(
            schema_version=str(value["schema_version"]),
            model_id=str(value["model_id"]),
            version=str(value["version"]),
            estimator=str(value["estimator"]),
            feature_signal_hashes={
                str(key): str(item) for key, item in feature_hashes.items()
            },
            label_spec_hash=str(value["label_spec_hash"]),
            validation_spec_hash=str(value["validation_spec_hash"]),
            hyperparameters={str(key): item for key, item in hyperparameters.items()},
            fit_intercept=fit_intercept,
            random_seed=random_seed,
            missing_policy=str(value["missing_policy"]),
            sample_weighting=str(value["sample_weighting"]),
        )


__all__ = ["ModelEstimator", "ModelSpec"]
