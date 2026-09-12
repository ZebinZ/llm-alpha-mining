from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypeAlias, cast

import numpy as np
import numpy.typing as npt
from sklearn.base import RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge

from alpha_research.core.hashing import hash_json
from alpha_research.models.spec import ModelEstimator, ModelSpec


_MAXIMUM_TREE_NODES = 100_000
_MAXIMUM_FOREST_NODES = 250_000


@dataclass(frozen=True, slots=True)
class FactorOnlyState:
    feature_index: int = 0
    schema_version: str = "factor-only-state/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "factor-only-state/v1":
            raise ValueError("unsupported factor-only state schema")
        if self.feature_index != 0:
            raise ValueError("factor-only state requires feature index zero")

    def predict(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        matrix = _matrix(values, feature_count=1)
        return matrix[:, 0].astype(float, copy=True)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "feature_index": self.feature_index,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FactorOnlyState":
        if set(value) != {"schema_version", "feature_index"}:
            raise ValueError("FactorOnlyState wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            feature_index=_integer(value["feature_index"], "feature_index"),
        )


@dataclass(frozen=True, slots=True)
class LinearRegressionState:
    coefficients: tuple[float, ...]
    intercept: float
    schema_version: str = "linear-regression-state/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "linear-regression-state/v1":
            raise ValueError("unsupported linear-regression state schema")
        coefficients = _coerce_float_tuple(self.coefficients, "coefficients")
        intercept = _coerce_float(self.intercept, "intercept")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "intercept", intercept)
        if not coefficients or not np.isfinite(
            np.asarray(coefficients, dtype=float)
        ).all() or not np.isfinite(intercept):
            raise ValueError("linear-regression state parameters must be finite")

    def predict(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        matrix = _matrix(values, feature_count=len(self.coefficients))
        return (
            matrix @ np.asarray(self.coefficients, dtype=float) + self.intercept
        ).astype(float)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "LinearRegressionState":
        if set(value) != {"schema_version", "coefficients", "intercept"}:
            raise ValueError("LinearRegressionState wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            coefficients=_float_tuple(value["coefficients"], "coefficients"),
            intercept=_number(value["intercept"], "intercept"),
        )


@dataclass(frozen=True, slots=True)
class RegressionTreeState:
    children_left: tuple[int, ...]
    children_right: tuple[int, ...]
    feature: tuple[int, ...]
    threshold: tuple[float, ...]
    value: tuple[float, ...]
    schema_version: str = "regression-tree-state/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "regression-tree-state/v1":
            raise ValueError("unsupported regression-tree state schema")
        children_left = _coerce_integer_tuple(self.children_left, "children_left")
        children_right = _coerce_integer_tuple(
            self.children_right, "children_right"
        )
        feature = _coerce_integer_tuple(self.feature, "feature")
        threshold = _coerce_float_tuple(self.threshold, "threshold")
        value = _coerce_float_tuple(self.value, "value")
        object.__setattr__(self, "children_left", children_left)
        object.__setattr__(self, "children_right", children_right)
        object.__setattr__(self, "feature", feature)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "value", value)
        node_count = len(children_left)
        if node_count <= 0 or node_count > _MAXIMUM_TREE_NODES:
            raise ValueError("regression-tree node count is invalid")
        if any(
            len(values) != node_count
            for values in (
                self.children_right,
                self.feature,
                self.threshold,
                self.value,
            )
        ):
            raise ValueError("regression-tree arrays differ in length")
        if not np.isfinite(np.asarray(threshold, dtype=float)).all() or not np.isfinite(
            np.asarray(value, dtype=float)
        ).all():
            raise ValueError("regression-tree numeric state must be finite")
        self._verify_topology()

    @property
    def maximum_depth(self) -> int:
        maximum = 0
        stack: list[tuple[int, int]] = [(0, 0)]
        while stack:
            node, depth = stack.pop()
            maximum = max(maximum, depth)
            left = self.children_left[node]
            if left != -1:
                stack.append((left, depth + 1))
                stack.append((self.children_right[node], depth + 1))
        return maximum

    def _verify_topology(self) -> None:
        node_count = len(self.children_left)
        seen: set[int] = set()
        stack = [0]
        while stack:
            node = stack.pop()
            if node in seen:
                raise ValueError("regression-tree topology is cyclic or shared")
            if node < 0 or node >= node_count:
                raise ValueError("regression-tree child index is out of bounds")
            seen.add(node)
            left = self.children_left[node]
            right = self.children_right[node]
            if left == -1 or right == -1:
                if left != -1 or right != -1:
                    raise ValueError("regression-tree leaf children differ")
                continue
            if left == right:
                raise ValueError("regression-tree internal children must differ")
            stack.extend((right, left))
        if len(seen) != node_count:
            raise ValueError("regression-tree contains unreachable nodes")

    def predict(
        self,
        values: npt.NDArray[np.float64],
        *,
        feature_count: int,
    ) -> npt.NDArray[np.float64]:
        # sklearn tree predictors coerce inputs to float32 before comparing
        # against their float64 thresholds.  Mirroring that detail is required
        # for bit-exact replay at threshold boundaries.
        matrix = _matrix(values, feature_count=feature_count).astype(
            np.float32, copy=False
        )
        left = np.asarray(self.children_left, dtype=np.int64)
        right = np.asarray(self.children_right, dtype=np.int64)
        feature = np.asarray(self.feature, dtype=np.int64)
        threshold = np.asarray(self.threshold, dtype=float)
        terminal = np.asarray(self.value, dtype=float)
        nodes: npt.NDArray[np.int64] = np.zeros(len(matrix), dtype=np.int64)
        for _ in range(len(left)):
            active = left[nodes] != -1
            if not bool(active.any()):
                return terminal[nodes].astype(float, copy=True)
            rows = np.flatnonzero(active)
            current = nodes[rows]
            selected_features = feature[current]
            if bool(
                ((selected_features < 0) | (selected_features >= feature_count)).any()
            ):
                raise ValueError("regression-tree feature index is out of bounds")
            go_left = matrix[rows, selected_features] <= threshold[current]
            nodes[rows] = np.where(go_left, left[current], right[current])
        raise RuntimeError("regression-tree traversal did not terminate")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "children_left": list(self.children_left),
            "children_right": list(self.children_right),
            "feature": list(self.feature),
            "threshold": list(self.threshold),
            "value": list(self.value),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RegressionTreeState":
        expected = {
            "schema_version",
            "children_left",
            "children_right",
            "feature",
            "threshold",
            "value",
        }
        if set(value) != expected:
            raise ValueError("RegressionTreeState wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            children_left=_integer_tuple(value["children_left"], "children_left"),
            children_right=_integer_tuple(
                value["children_right"], "children_right"
            ),
            feature=_integer_tuple(value["feature"], "feature"),
            threshold=_float_tuple(value["threshold"], "threshold"),
            value=_float_tuple(value["value"], "value"),
        )


@dataclass(frozen=True, slots=True)
class ExtraTreesRegressionState:
    feature_count: int
    trees: tuple[RegressionTreeState, ...]
    aggregation: str = "arithmetic_mean"
    schema_version: str = "extra-trees-regression-state/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "extra-trees-regression-state/v1":
            raise ValueError("unsupported extra-trees state schema")
        if (
            not isinstance(self.feature_count, int)
            or isinstance(self.feature_count, bool)
            or self.feature_count <= 0
        ):
            raise ValueError("extra-trees state feature count must be positive")
        trees = tuple(self.trees)
        if not all(isinstance(tree, RegressionTreeState) for tree in trees):
            raise TypeError("extra-trees state trees must be regression tree states")
        object.__setattr__(self, "trees", trees)
        if not trees or len(trees) > 2_048:
            raise ValueError("extra-trees state tree count is invalid")
        if sum(len(tree.children_left) for tree in trees) > _MAXIMUM_FOREST_NODES:
            raise ValueError("extra-trees state total node count exceeds resource limit")
        if self.aggregation != "arithmetic_mean":
            raise ValueError("extra-trees state aggregation differs")
        for tree in self.trees:
            internal = [
                feature
                for left, feature in zip(tree.children_left, tree.feature, strict=True)
                if left != -1
            ]
            if any(feature < 0 or feature >= self.feature_count for feature in internal):
                raise ValueError("extra-trees state feature index is out of bounds")

    def predict(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        matrix = _matrix(values, feature_count=self.feature_count)
        prediction: npt.NDArray[np.float64] = np.zeros(len(matrix), dtype=float)
        for tree in self.trees:
            prediction += tree.predict(matrix, feature_count=self.feature_count)
        prediction /= float(len(self.trees))
        return prediction

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "feature_count": self.feature_count,
            "aggregation": self.aggregation,
            "trees": [tree.to_dict() for tree in self.trees],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ExtraTreesRegressionState":
        expected = {"schema_version", "feature_count", "aggregation", "trees"}
        if set(value) != expected:
            raise ValueError("ExtraTreesRegressionState wire fields differ")
        raw_trees = value["trees"]
        if not isinstance(raw_trees, list) or not all(
            isinstance(item, Mapping) for item in raw_trees
        ):
            raise TypeError("trees must be an object array")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            feature_count=_integer(value["feature_count"], "feature_count"),
            aggregation=_text(value["aggregation"], "aggregation"),
            trees=tuple(RegressionTreeState.from_mapping(item) for item in raw_trees),
        )


PortableEstimatorState: TypeAlias = (
    FactorOnlyState | LinearRegressionState | ExtraTreesRegressionState
)


def portable_state_from_estimator(
    spec: ModelSpec,
    estimator: RegressorMixin | None,
) -> PortableEstimatorState:
    if spec.estimator is ModelEstimator.FACTOR_ONLY:
        if estimator is not None:
            raise ValueError("factor-only artifact estimator must be absent")
        return FactorOnlyState()
    if estimator is None:
        raise ValueError("fitted artifact estimator is absent")
    if spec.estimator is ModelEstimator.EXTRA_TREES:
        if not isinstance(estimator, ExtraTreesRegressor):
            raise TypeError("extra-trees artifact estimator type differs")
        trees = tuple(_tree_state(item.tree_) for item in estimator.estimators_)
        if len(trees) != int(spec.hyperparameters["n_estimators"]):
            raise ValueError("extra-trees fitted tree count differs")
        return ExtraTreesRegressionState(
            feature_count=len(spec.feature_signal_hashes),
            trees=trees,
        )
    expected_linear_type: type[LinearRegression] | type[Ridge] | type[ElasticNet]
    if spec.estimator is ModelEstimator.OLS:
        expected_linear_type = LinearRegression
    elif spec.estimator is ModelEstimator.RIDGE:
        expected_linear_type = Ridge
    elif spec.estimator is ModelEstimator.ELASTIC_NET:
        expected_linear_type = ElasticNet
    else:  # pragma: no cover - ModelEstimator is exhaustively handled above.
        raise RuntimeError("unsupported portable estimator")
    if not isinstance(estimator, expected_linear_type):
        raise TypeError("linear artifact estimator type differs")
    coefficients = np.asarray(estimator.coef_, dtype=float).reshape(-1)
    intercept = float(estimator.intercept_)
    if len(coefficients) != len(spec.feature_signal_hashes):
        raise ValueError("linear artifact coefficient dimension differs")
    return LinearRegressionState(
        coefficients=tuple(float(value) for value in coefficients),
        intercept=intercept,
    )


def portable_state_from_mapping(
    value: Mapping[str, object],
) -> PortableEstimatorState:
    schema = _text(value.get("schema_version"), "schema_version")
    if schema == "factor-only-state/v1":
        return FactorOnlyState.from_mapping(value)
    if schema == "linear-regression-state/v1":
        return LinearRegressionState.from_mapping(value)
    if schema == "extra-trees-regression-state/v1":
        return ExtraTreesRegressionState.from_mapping(value)
    raise ValueError("unsupported portable estimator state schema")


def portable_state_hash(state: PortableEstimatorState) -> str:
    return cast(str, hash_json(state.to_dict()))


def _tree_state(tree: object) -> RegressionTreeState:
    children_left = np.asarray(getattr(tree, "children_left"), dtype=np.int64)
    children_right = np.asarray(getattr(tree, "children_right"), dtype=np.int64)
    feature = np.asarray(getattr(tree, "feature"), dtype=np.int64)
    threshold = np.asarray(getattr(tree, "threshold"), dtype=float)
    raw_value = np.asarray(getattr(tree, "value"), dtype=float)
    if raw_value.ndim != 3 or raw_value.shape[1:] != (1, 1):
        raise ValueError("portable regression tree must have one numeric output")
    return RegressionTreeState(
        children_left=tuple(int(value) for value in children_left),
        children_right=tuple(int(value) for value in children_right),
        feature=tuple(int(value) for value in feature),
        threshold=tuple(float(value) for value in threshold),
        value=tuple(float(value) for value in raw_value[:, 0, 0]),
    )


def _matrix(
    values: npt.NDArray[np.float64], *, feature_count: int
) -> npt.NDArray[np.float64]:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != feature_count:
        raise ValueError("portable model input dimension differs")
    if len(matrix) == 0 or not np.isfinite(matrix).all():
        raise ValueError("portable model input must be non-empty and finite")
    return matrix


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


def _coerce_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(
        value, (bool, np.bool_)
    ):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _coerce_float_tuple(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a numeric sequence")
    return tuple(_coerce_float(item, name) for item in value)


def _coerce_integer_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)) or not all(
        isinstance(item, (int, np.integer)) and not isinstance(item, (bool, np.bool_))
        for item in value
    ):
        raise TypeError(f"{name} must be an integer sequence")
    return tuple(int(item) for item in value)


def _integer_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise TypeError(f"{name} must be an integer array")
    return tuple(value)


def _float_tuple(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, (int, float)) and not isinstance(item, bool)
        for item in value
    ):
        raise TypeError(f"{name} must be a numeric array")
    return tuple(float(item) for item in value)


__all__ = [
    "ExtraTreesRegressionState",
    "FactorOnlyState",
    "LinearRegressionState",
    "PortableEstimatorState",
    "RegressionTreeState",
    "portable_state_from_estimator",
    "portable_state_from_mapping",
    "portable_state_hash",
]
