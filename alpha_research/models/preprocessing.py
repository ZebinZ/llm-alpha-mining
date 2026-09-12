from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from alpha_research.core.hashing import hash_frame, hash_json, require_sha256


class ModelPreprocessKind(str, Enum):
    IDENTITY = "identity"
    STANDARDIZE = "standardize"


@dataclass(frozen=True, slots=True)
class ModelPreprocessSpec:
    """A deliberately small, replayable preprocessing contract."""

    kind: ModelPreprocessKind | str = ModelPreprocessKind.IDENTITY
    zero_variance_policy: str = "unit_scale"
    schema_version: str = "model-preprocess-spec/v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ModelPreprocessKind(self.kind))
        if self.schema_version != "model-preprocess-spec/v1":
            raise ValueError("unsupported model preprocessing schema")
        if self.zero_variance_policy != "unit_scale":
            raise ValueError("unsupported zero-variance preprocessing policy")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, str]:
        kind = self.kind
        if not isinstance(kind, ModelPreprocessKind):  # pragma: no cover
            raise RuntimeError("model preprocessing kind was not normalized")
        return {
            "schema_version": self.schema_version,
            "kind": kind.value,
            "zero_variance_policy": self.zero_variance_policy,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ModelPreprocessSpec":
        expected = {"schema_version", "kind", "zero_variance_policy"}
        if set(value) != expected:
            raise ValueError("ModelPreprocessSpec wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            kind=_text(value["kind"], "kind"),
            zero_variance_policy=_text(
                value["zero_variance_policy"], "zero_variance_policy"
            ),
        )


@dataclass(frozen=True, slots=True)
class FoldTransformReceipt:
    """Content-addressed proof of one fold-local fit and transform."""

    fold_id: str
    preprocess_spec_hash: str
    feature_names: tuple[str, ...]
    train_sample_index_hash: str
    validation_sample_index_hash: str
    train_input_hash: str
    validation_input_hash: str
    location: tuple[float, ...]
    scale: tuple[float, ...]
    train_output_hash: str
    validation_output_hash: str
    train_observation_count: int
    validation_observation_count: int
    schema_version: str = "fold-transform-receipt/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "fold-transform-receipt/v1":
            raise ValueError("unsupported fold transform receipt schema")
        if not self.fold_id.strip():
            raise ValueError("fold transform receipt requires a fold id")
        require_sha256(
            self.preprocess_spec_hash, name="fold transform preprocess_spec_hash"
        )
        for name, value in (
            ("train_sample_index_hash", self.train_sample_index_hash),
            ("validation_sample_index_hash", self.validation_sample_index_hash),
            ("train_input_hash", self.train_input_hash),
            ("validation_input_hash", self.validation_input_hash),
            ("train_output_hash", self.train_output_hash),
            ("validation_output_hash", self.validation_output_hash),
        ):
            require_sha256(value, name=f"fold transform {name}")
        feature_names = tuple(self.feature_names)
        location = tuple(float(value) for value in self.location)
        scale = tuple(float(value) for value in self.scale)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "scale", scale)
        if not feature_names or any(not name.strip() for name in feature_names):
            raise ValueError("fold transform receipt requires named features")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("fold transform feature names must be unique")
        if len(location) != len(feature_names) or len(scale) != len(feature_names):
            raise ValueError("fold transform statistics differ from feature dimension")
        if (
            not np.isfinite(np.asarray(location, dtype=float)).all()
            or not np.isfinite(np.asarray(scale, dtype=float)).all()
            or any(value <= 0 for value in scale)
        ):
            raise ValueError("fold transform statistics must be finite with positive scale")
        if (
            not isinstance(self.train_observation_count, int)
            or isinstance(self.train_observation_count, bool)
            or self.train_observation_count <= 0
            or not isinstance(self.validation_observation_count, int)
            or isinstance(self.validation_observation_count, bool)
            or self.validation_observation_count <= 0
        ):
            raise ValueError("fold transform observation counts must be positive integers")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fold_id": self.fold_id,
            "preprocess_spec_hash": self.preprocess_spec_hash,
            "feature_names": list(self.feature_names),
            "train_sample_index_hash": self.train_sample_index_hash,
            "validation_sample_index_hash": self.validation_sample_index_hash,
            "train_input_hash": self.train_input_hash,
            "validation_input_hash": self.validation_input_hash,
            "location": list(self.location),
            "scale": list(self.scale),
            "train_output_hash": self.train_output_hash,
            "validation_output_hash": self.validation_output_hash,
            "train_observation_count": self.train_observation_count,
            "validation_observation_count": self.validation_observation_count,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FoldTransformReceipt":
        expected = {
            "schema_version",
            "fold_id",
            "preprocess_spec_hash",
            "feature_names",
            "train_sample_index_hash",
            "validation_sample_index_hash",
            "train_input_hash",
            "validation_input_hash",
            "location",
            "scale",
            "train_output_hash",
            "validation_output_hash",
            "train_observation_count",
            "validation_observation_count",
        }
        if set(value) != expected:
            raise ValueError("FoldTransformReceipt wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            fold_id=_text(value["fold_id"], "fold_id"),
            preprocess_spec_hash=_text(
                value["preprocess_spec_hash"], "preprocess_spec_hash"
            ),
            feature_names=_text_tuple(value["feature_names"], "feature_names"),
            train_sample_index_hash=_text(
                value["train_sample_index_hash"], "train_sample_index_hash"
            ),
            validation_sample_index_hash=_text(
                value["validation_sample_index_hash"],
                "validation_sample_index_hash",
            ),
            train_input_hash=_text(value["train_input_hash"], "train_input_hash"),
            validation_input_hash=_text(
                value["validation_input_hash"], "validation_input_hash"
            ),
            location=_float_tuple(value["location"], "location"),
            scale=_float_tuple(value["scale"], "scale"),
            train_output_hash=_text(
                value["train_output_hash"], "train_output_hash"
            ),
            validation_output_hash=_text(
                value["validation_output_hash"], "validation_output_hash"
            ),
            train_observation_count=_integer(
                value["train_observation_count"], "train_observation_count"
            ),
            validation_observation_count=_integer(
                value["validation_observation_count"],
                "validation_observation_count",
            ),
        )


class FoldLocalPreprocessor:
    """Fit preprocessing on a training fold and apply it to both fold partitions."""

    def fit_transform(
        self,
        spec: ModelPreprocessSpec,
        *,
        fold_id: str,
        feature_names: tuple[str, ...],
        x_train: npt.NDArray[np.float64],
        x_validation: npt.NDArray[np.float64],
        train_index: pd.Index,
        validation_index: pd.Index,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        FoldTransformReceipt,
    ]:
        train = _matrix_frame(x_train, train_index, feature_names, "train")
        validation = _matrix_frame(
            x_validation, validation_index, feature_names, "validation"
        )
        kind = spec.kind
        if kind is ModelPreprocessKind.IDENTITY:
            location: npt.NDArray[np.float64] = np.zeros(
                len(feature_names), dtype=float
            )
            scale: npt.NDArray[np.float64] = np.ones(
                len(feature_names), dtype=float
            )
        elif kind is ModelPreprocessKind.STANDARDIZE:
            location = train.to_numpy(dtype=float).mean(axis=0)
            scale = train.to_numpy(dtype=float).std(axis=0, ddof=0)
            scale = np.where(scale <= np.finfo(float).eps, 1.0, scale)
        else:  # pragma: no cover
            raise RuntimeError("unsupported normalized preprocessing kind")
        transformed_train = (train.to_numpy(dtype=float) - location) / scale
        transformed_validation = (
            validation.to_numpy(dtype=float) - location
        ) / scale
        if not np.isfinite(transformed_train).all() or not np.isfinite(
            transformed_validation
        ).all():
            raise ValueError("model preprocessing produced non-finite values")
        train_output = pd.DataFrame(
            transformed_train, index=train.index, columns=train.columns
        )
        validation_output = pd.DataFrame(
            transformed_validation,
            index=validation.index,
            columns=validation.columns,
        )
        receipt = FoldTransformReceipt(
            fold_id=fold_id,
            preprocess_spec_hash=spec.content_hash,
            feature_names=feature_names,
            train_sample_index_hash=_hash_index(train.index),
            validation_sample_index_hash=_hash_index(validation.index),
            train_input_hash=hash_frame(train),
            validation_input_hash=hash_frame(validation),
            location=tuple(float(value) for value in location),
            scale=tuple(float(value) for value in scale),
            train_output_hash=hash_frame(train_output),
            validation_output_hash=hash_frame(validation_output),
            train_observation_count=len(train),
            validation_observation_count=len(validation),
        )
        return (
            transformed_train.astype(float, copy=True),
            transformed_validation.astype(float, copy=True),
            receipt,
        )

    def verify(
        self,
        receipt: FoldTransformReceipt,
        spec: ModelPreprocessSpec,
        *,
        feature_names: tuple[str, ...],
        x_train: npt.NDArray[np.float64],
        x_validation: npt.NDArray[np.float64],
        train_index: pd.Index,
        validation_index: pd.Index,
    ) -> FoldTransformReceipt:
        _, _, recomputed = self.fit_transform(
            spec,
            fold_id=receipt.fold_id,
            feature_names=feature_names,
            x_train=x_train,
            x_validation=x_validation,
            train_index=train_index,
            validation_index=validation_index,
        )
        if recomputed.content_hash != receipt.content_hash:
            raise ValueError(
                "fold_transform_receipt_recomputation_mismatch:"
                f"{receipt.fold_id}:expected={recomputed.content_hash}:"
                f"observed={receipt.content_hash}"
            )
        return receipt


def _matrix_frame(
    values: npt.NDArray[np.float64],
    index: pd.Index,
    feature_names: tuple[str, ...],
    partition: str,
) -> pd.DataFrame:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
        raise ValueError(f"fold transform {partition} matrix dimension differs")
    if len(matrix) != len(index) or len(matrix) == 0:
        raise ValueError(f"fold transform {partition} index dimension differs")
    if not np.isfinite(matrix).all():
        raise ValueError(f"fold transform {partition} values must be finite")
    return pd.DataFrame(matrix, index=index.copy(), columns=feature_names)


def _hash_index(index: pd.Index) -> str:
    marker = pd.DataFrame({"present": np.ones(len(index), dtype=np.int8)}, index=index)
    return cast(str, hash_frame(marker))


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _text_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{name} must be a text array")
    return tuple(value)


def _float_tuple(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, (int, float)) and not isinstance(item, bool)
        for item in value
    ):
        raise TypeError(f"{name} must be a numeric array")
    return tuple(float(item) for item in value)


__all__ = [
    "FoldLocalPreprocessor",
    "FoldTransformReceipt",
    "ModelPreprocessKind",
    "ModelPreprocessSpec",
]
