from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, cast

import numpy as np
import pandas as pd

from alpha_research.core.hashing import canonical_json_bytes, hash_json, require_sha256
from alpha_research.core.immutable_json import (
    load_immutable_json_document,
    write_immutable_json_document,
)
from alpha_research.models.model_state import (
    ExtraTreesRegressionState,
    FactorOnlyState,
    LinearRegressionState,
    PortableEstimatorState,
    portable_state_from_mapping,
    portable_state_hash,
)
from alpha_research.models.preprocessing import (
    FoldTransformReceipt,
    ModelPreprocessSpec,
)
from alpha_research.models.spec import ModelEstimator, ModelSpec
from alpha_research.validation import ValidationReceipt


_ARTIFACT_SUFFIX = ".model-artifact-v1.json"
_MAXIMUM_ARTIFACT_BYTES = 64 * 1024 * 1024
_MEDIA_TYPE = "application/vnd.alpha-research.model-artifact+json"
MODEL_RESULT_SCHEMA_VERSION = "model-result/v3"
MODEL_RESULT_DESCRIPTOR_SCHEMA_VERSION = "model-result-descriptor/v2"


def model_result_content_hash(
    *,
    model_spec_hash: str,
    validation_receipt_hash: str,
    label_values_hash: str,
    predictions_hash: str,
    coefficients_hash: str,
    feature_importance_hash: str,
    diagnostics_hash: str,
    preprocess_spec_hash: str,
    fold_transform_receipt_hashes: tuple[str, ...],
    prediction_row_count: int,
    prediction_security_count: int,
    fold_count: int,
    schema_version: str = MODEL_RESULT_SCHEMA_VERSION,
) -> str:
    """Recompute the complete ModelResult identity from its wire descriptor."""

    if schema_version != MODEL_RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported model result content-hash schema")
    fields = {
        "model_spec_hash": model_spec_hash,
        "validation_receipt_hash": validation_receipt_hash,
        "label_values_hash": label_values_hash,
        "predictions_hash": predictions_hash,
        "coefficients_hash": coefficients_hash,
        "feature_importance_hash": feature_importance_hash,
        "diagnostics_hash": diagnostics_hash,
        "preprocess_spec_hash": preprocess_spec_hash,
    }
    for name, digest in fields.items():
        require_sha256(digest, name=f"model result {name}")
    receipts = tuple(fold_transform_receipt_hashes)
    if not receipts:
        raise ValueError("model result requires fold transform receipts")
    for digest in receipts:
        require_sha256(digest, name="model result fold transform receipt hash")
    counts = {
        "prediction_row_count": prediction_row_count,
        "prediction_security_count": prediction_security_count,
        "fold_count": fold_count,
    }
    for name, value in counts.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"model result {name} is invalid")
    if fold_count != len(receipts):
        raise ValueError("model result fold count differs")
    return cast(
        str,
        hash_json(
            {
                "schema_version": schema_version,
                **fields,
                "fold_transform_receipt_hashes": list(receipts),
                **counts,
            }
        ),
    )


@dataclass(frozen=True, slots=True)
class ModelRuntimeFingerprint:
    python_implementation: str
    python_version: str
    numpy_version: str
    pandas_version: str
    scipy_version: str
    scikit_learn_version: str
    platform_system: str
    platform_machine: str
    numerical_backend_hash: str
    inference_thread_policy: str = "caller_serial"
    schema_version: str = "model-runtime-fingerprint/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "model-runtime-fingerprint/v1":
            raise ValueError("unsupported model runtime fingerprint schema")
        if self.python_implementation != "CPython":
            raise ValueError("model runtime requires CPython")
        for name in (
            "python_version",
            "numpy_version",
            "pandas_version",
            "scipy_version",
            "scikit_learn_version",
            "platform_system",
            "platform_machine",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value.strip()
                or value.strip() != value
            ):
                raise ValueError(f"model runtime {name} is invalid")
        require_sha256(
            self.numerical_backend_hash,
            name="model runtime numerical_backend_hash",
        )
        if self.inference_thread_policy != "caller_serial":
            raise ValueError("model runtime inference thread policy differs")

    @classmethod
    def current(cls) -> ModelRuntimeFingerprint:
        configuration = np.__config__.show(mode="dicts")
        return cls(
            python_implementation=platform.python_implementation(),
            python_version=platform.python_version(),
            numpy_version=importlib.metadata.version("numpy"),
            pandas_version=importlib.metadata.version("pandas"),
            scipy_version=importlib.metadata.version("scipy"),
            scikit_learn_version=importlib.metadata.version("scikit-learn"),
            platform_system=platform.system(),
            platform_machine=platform.machine(),
            numerical_backend_hash=cast(str, hash_json(configuration)),
        )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def verify_current(self) -> None:
        if self.current().content_hash != self.content_hash:
            raise ValueError("model_artifact_runtime_fingerprint_differs")

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "pandas_version": self.pandas_version,
            "scipy_version": self.scipy_version,
            "scikit_learn_version": self.scikit_learn_version,
            "platform_system": self.platform_system,
            "platform_machine": self.platform_machine,
            "numerical_backend_hash": self.numerical_backend_hash,
            "inference_thread_policy": self.inference_thread_policy,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelRuntimeFingerprint:
        expected = {
            "schema_version",
            "python_implementation",
            "python_version",
            "numpy_version",
            "pandas_version",
            "scipy_version",
            "scikit_learn_version",
            "platform_system",
            "platform_machine",
            "numerical_backend_hash",
            "inference_thread_policy",
        }
        if set(value) != expected:
            raise ValueError("ModelRuntimeFingerprint wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            python_implementation=_text(
                value["python_implementation"], "python_implementation"
            ),
            python_version=_text(value["python_version"], "python_version"),
            numpy_version=_text(value["numpy_version"], "numpy_version"),
            pandas_version=_text(value["pandas_version"], "pandas_version"),
            scipy_version=_text(value["scipy_version"], "scipy_version"),
            scikit_learn_version=_text(
                value["scikit_learn_version"], "scikit_learn_version"
            ),
            platform_system=_text(value["platform_system"], "platform_system"),
            platform_machine=_text(value["platform_machine"], "platform_machine"),
            numerical_backend_hash=_text(
                value["numerical_backend_hash"], "numerical_backend_hash"
            ),
            inference_thread_policy=_text(
                value["inference_thread_policy"], "inference_thread_policy"
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelResultDescriptor:
    result_hash: str
    model_spec_hash: str
    validation_receipt_hash: str
    label_values_hash: str
    predictions_hash: str
    coefficients_hash: str
    feature_importance_hash: str
    diagnostics_hash: str
    preprocess_spec_hash: str
    fold_transform_receipt_hashes: tuple[str, ...]
    prediction_row_count: int
    prediction_security_count: int
    fold_count: int
    schema_version: str = MODEL_RESULT_DESCRIPTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_RESULT_DESCRIPTOR_SCHEMA_VERSION:
            raise ValueError("unsupported model result descriptor schema")
        for name in (
            "result_hash",
            "model_spec_hash",
            "validation_receipt_hash",
            "label_values_hash",
            "predictions_hash",
            "coefficients_hash",
            "feature_importance_hash",
            "diagnostics_hash",
            "preprocess_spec_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"model result {name}")
        receipts = tuple(self.fold_transform_receipt_hashes)
        if not receipts:
            raise ValueError("model result descriptor requires fold receipts")
        for receipt in receipts:
            require_sha256(receipt, name="model result fold transform receipt hash")
        object.__setattr__(self, "fold_transform_receipt_hashes", receipts)
        for name in (
            "prediction_row_count",
            "prediction_security_count",
            "fold_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"model result descriptor {name} is invalid")
        if self.fold_count != len(receipts):
            raise ValueError("model result descriptor fold count differs")
        if self.recomputed_result_hash != self.result_hash:
            raise ValueError("model result descriptor result hash differs")

    @property
    def recomputed_result_hash(self) -> str:
        return model_result_content_hash(
            model_spec_hash=self.model_spec_hash,
            validation_receipt_hash=self.validation_receipt_hash,
            label_values_hash=self.label_values_hash,
            predictions_hash=self.predictions_hash,
            coefficients_hash=self.coefficients_hash,
            feature_importance_hash=self.feature_importance_hash,
            diagnostics_hash=self.diagnostics_hash,
            preprocess_spec_hash=self.preprocess_spec_hash,
            fold_transform_receipt_hashes=self.fold_transform_receipt_hashes,
            prediction_row_count=self.prediction_row_count,
            prediction_security_count=self.prediction_security_count,
            fold_count=self.fold_count,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "result_hash": self.result_hash,
            "model_spec_hash": self.model_spec_hash,
            "validation_receipt_hash": self.validation_receipt_hash,
            "label_values_hash": self.label_values_hash,
            "predictions_hash": self.predictions_hash,
            "coefficients_hash": self.coefficients_hash,
            "feature_importance_hash": self.feature_importance_hash,
            "diagnostics_hash": self.diagnostics_hash,
            "preprocess_spec_hash": self.preprocess_spec_hash,
            "fold_transform_receipt_hashes": list(self.fold_transform_receipt_hashes),
            "prediction_row_count": self.prediction_row_count,
            "prediction_security_count": self.prediction_security_count,
            "fold_count": self.fold_count,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelResultDescriptor:
        expected = {
            "schema_version",
            "result_hash",
            "model_spec_hash",
            "validation_receipt_hash",
            "label_values_hash",
            "predictions_hash",
            "coefficients_hash",
            "feature_importance_hash",
            "diagnostics_hash",
            "preprocess_spec_hash",
            "fold_transform_receipt_hashes",
            "prediction_row_count",
            "prediction_security_count",
            "fold_count",
        }
        if set(value) != expected:
            raise ValueError("ModelResultDescriptor wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            result_hash=_text(value["result_hash"], "result_hash"),
            model_spec_hash=_text(value["model_spec_hash"], "model_spec_hash"),
            validation_receipt_hash=_text(
                value["validation_receipt_hash"], "validation_receipt_hash"
            ),
            label_values_hash=_text(value["label_values_hash"], "label_values_hash"),
            predictions_hash=_text(value["predictions_hash"], "predictions_hash"),
            coefficients_hash=_text(value["coefficients_hash"], "coefficients_hash"),
            feature_importance_hash=_text(
                value["feature_importance_hash"], "feature_importance_hash"
            ),
            diagnostics_hash=_text(value["diagnostics_hash"], "diagnostics_hash"),
            preprocess_spec_hash=_text(
                value["preprocess_spec_hash"], "preprocess_spec_hash"
            ),
            fold_transform_receipt_hashes=_text_tuple(
                value["fold_transform_receipt_hashes"],
                "fold_transform_receipt_hashes",
            ),
            prediction_row_count=_integer(
                value["prediction_row_count"], "prediction_row_count"
            ),
            prediction_security_count=_integer(
                value["prediction_security_count"], "prediction_security_count"
            ),
            fold_count=_integer(value["fold_count"], "fold_count"),
        )


@dataclass(frozen=True, slots=True)
class FoldModelArtifact:
    fold_id: str
    estimator: ModelEstimator | str
    transform_receipt: FoldTransformReceipt
    estimator_state: PortableEstimatorState
    estimator_state_hash: str
    schema_version: str = "fold-model-artifact/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "fold-model-artifact/v1":
            raise ValueError("unsupported fold model artifact schema")
        if not isinstance(self.transform_receipt, FoldTransformReceipt):
            raise TypeError("fold model transform receipt type differs")
        if not self.fold_id.strip() or self.fold_id != self.transform_receipt.fold_id:
            raise ValueError("fold model artifact id differs")
        estimator = ModelEstimator(self.estimator)
        object.__setattr__(self, "estimator", estimator)
        require_sha256(
            self.estimator_state_hash, name="fold model estimator_state_hash"
        )
        if portable_state_hash(self.estimator_state) != self.estimator_state_hash:
            raise ValueError("fold model estimator state hash differs")
        state = self.estimator_state
        if estimator is ModelEstimator.FACTOR_ONLY:
            compatible = isinstance(state, FactorOnlyState)
        elif estimator is ModelEstimator.EXTRA_TREES:
            compatible = isinstance(state, ExtraTreesRegressionState)
        else:
            compatible = isinstance(state, LinearRegressionState)
        if not compatible:
            raise TypeError("fold model estimator/state kind differs")
        feature_count = len(self.transform_receipt.feature_names)
        if isinstance(state, FactorOnlyState) and feature_count != 1:
            raise ValueError("fold model factor-only feature dimension differs")
        if (
            isinstance(state, LinearRegressionState)
            and len(state.coefficients) != feature_count
        ):
            raise ValueError("fold model linear feature dimension differs")
        if (
            isinstance(state, ExtraTreesRegressionState)
            and state.feature_count != feature_count
        ):
            raise ValueError("fold model tree feature dimension differs")

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        values = pd.DataFrame(frame).copy(deep=True)
        expected_columns = pd.Index(self.transform_receipt.feature_names)
        if not values.columns.equals(expected_columns):
            raise ValueError("model artifact inference feature order differs")
        if values.columns.has_duplicates or len(values) == 0:
            raise ValueError("model artifact inference frame is invalid")
        matrix = values.to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError("model artifact inference values must be finite")
        location = np.asarray(self.transform_receipt.location, dtype=float)
        scale = np.asarray(self.transform_receipt.scale, dtype=float)
        transformed = (matrix - location) / scale
        if not np.isfinite(transformed).all():
            raise ValueError("model artifact preprocessing produced non-finite values")
        prediction = self.estimator_state.predict(transformed)
        if not np.isfinite(prediction).all():
            raise ValueError("model artifact prediction produced non-finite values")
        return pd.Series(prediction, index=values.index.copy(), name="prediction")

    def to_dict(self) -> dict[str, object]:
        estimator = self.estimator
        if not isinstance(estimator, ModelEstimator):  # pragma: no cover
            raise RuntimeError("fold model estimator was not normalized")
        return {
            "schema_version": self.schema_version,
            "fold_id": self.fold_id,
            "estimator": estimator.value,
            "transform_receipt": self.transform_receipt.to_dict(),
            "estimator_state": self.estimator_state.to_dict(),
            "estimator_state_hash": self.estimator_state_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> FoldModelArtifact:
        expected = {
            "schema_version",
            "fold_id",
            "estimator",
            "transform_receipt",
            "estimator_state",
            "estimator_state_hash",
        }
        if set(value) != expected:
            raise ValueError("FoldModelArtifact wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            fold_id=_text(value["fold_id"], "fold_id"),
            estimator=_text(value["estimator"], "estimator"),
            transform_receipt=FoldTransformReceipt.from_mapping(
                _mapping(value["transform_receipt"], "transform_receipt")
            ),
            estimator_state=portable_state_from_mapping(
                _mapping(value["estimator_state"], "estimator_state")
            ),
            estimator_state_hash=_text(
                value["estimator_state_hash"], "estimator_state_hash"
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelArtifactBundle:
    model_spec: ModelSpec
    model_spec_hash: str
    validation_receipt: ValidationReceipt
    validation_receipt_hash: str
    label_values_hash: str
    model_result: ModelResultDescriptor
    model_result_hash: str
    predictions_hash: str
    preprocess_spec: ModelPreprocessSpec
    preprocess_spec_hash: str
    runtime: ModelRuntimeFingerprint
    folds: tuple[FoldModelArtifact, ...]
    research_only: bool = True
    production_ready: bool = False
    schema_version: str = "model-artifact-bundle/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "model-artifact-bundle/v1":
            raise ValueError("unsupported model artifact bundle schema")
        if not isinstance(self.model_spec, ModelSpec):
            raise TypeError("model artifact specification type differs")
        if not isinstance(self.validation_receipt, ValidationReceipt):
            raise TypeError("model artifact validation receipt type differs")
        if not isinstance(self.model_result, ModelResultDescriptor):
            raise TypeError("model artifact result descriptor type differs")
        if not isinstance(self.preprocess_spec, ModelPreprocessSpec):
            raise TypeError("model artifact preprocessing type differs")
        if not isinstance(self.runtime, ModelRuntimeFingerprint):
            raise TypeError("model artifact runtime type differs")
        for name in (
            "model_spec_hash",
            "validation_receipt_hash",
            "label_values_hash",
            "model_result_hash",
            "predictions_hash",
            "preprocess_spec_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"model artifact {name}")
        if self.model_spec.content_hash != self.model_spec_hash:
            raise ValueError("model artifact specification hash differs")
        if self.validation_receipt.content_hash != self.validation_receipt_hash:
            raise ValueError("model artifact validation receipt hash differs")
        if self.preprocess_spec.content_hash != self.preprocess_spec_hash:
            raise ValueError("model artifact preprocessing hash differs")
        if self.model_result.result_hash != self.model_result_hash:
            raise ValueError("model artifact result hash differs")
        if self.model_result.predictions_hash != self.predictions_hash:
            raise ValueError("model artifact prediction hash differs")
        if (
            self.validation_receipt.validation_spec_hash
            != self.model_spec.validation_spec_hash
        ):
            raise ValueError("model artifact validation specification binding differs")
        if self.validation_receipt.label_spec_hash != self.model_spec.label_spec_hash:
            raise ValueError("model artifact label specification binding differs")
        if self.validation_receipt.labels_hash != self.label_values_hash:
            raise ValueError("model artifact label values binding differs")
        if (
            self.model_result.model_spec_hash != self.model_spec_hash
            or self.model_result.validation_receipt_hash != self.validation_receipt_hash
            or self.model_result.label_values_hash != self.label_values_hash
            or self.model_result.preprocess_spec_hash != self.preprocess_spec_hash
        ):
            raise ValueError("model artifact result lineage binding differs")
        folds = tuple(self.folds)
        if not all(isinstance(fold, FoldModelArtifact) for fold in folds):
            raise TypeError("model artifact folds must be fold model artifacts")
        expected_fold_ids = tuple(
            fold.fold_id for fold in self.validation_receipt.folds
        )
        if tuple(fold.fold_id for fold in folds) != expected_fold_ids:
            raise ValueError(
                "model artifact fold order differs from validation receipt"
            )
        if tuple(fold.transform_receipt.content_hash for fold in folds) != (
            self.model_result.fold_transform_receipt_hashes
        ):
            raise ValueError("model artifact transform receipt lineage differs")
        expected_features = tuple(self.model_spec.feature_signal_hashes)
        for fold in folds:
            if fold.estimator is not self.model_spec.estimator:
                raise ValueError("model artifact fold estimator differs")
            if fold.transform_receipt.feature_names != expected_features:
                raise ValueError("model artifact fold feature order differs")
            if fold.transform_receipt.preprocess_spec_hash != self.preprocess_spec_hash:
                raise ValueError("model artifact fold preprocessing binding differs")
            state = fold.estimator_state
            if isinstance(state, ExtraTreesRegressionState):
                if len(state.trees) != int(
                    self.model_spec.hyperparameters["n_estimators"]
                ):
                    raise ValueError("model artifact fitted tree count differs")
                if any(
                    tree.maximum_depth
                    > int(self.model_spec.hyperparameters["max_depth"])
                    for tree in state.trees
                ):
                    raise ValueError("model artifact fitted tree depth differs")
        if (
            not isinstance(self.research_only, bool)
            or not self.research_only
            or not isinstance(self.production_ready, bool)
            or self.production_ready
        ):
            raise ValueError("model artifact assurance boundary differs")
        object.__setattr__(self, "folds", folds)

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def predict_fold(self, fold_id: str, frame: pd.DataFrame) -> pd.Series:
        self.runtime.verify_current()
        matching = [fold for fold in self.folds if fold.fold_id == fold_id]
        if len(matching) != 1:
            raise KeyError(fold_id)
        return matching[0].predict(frame)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_spec": self.model_spec.to_dict(),
            "model_spec_hash": self.model_spec_hash,
            "validation_receipt": self.validation_receipt.to_dict(),
            "validation_receipt_hash": self.validation_receipt_hash,
            "label_values_hash": self.label_values_hash,
            "model_result": self.model_result.to_dict(),
            "model_result_hash": self.model_result_hash,
            "predictions_hash": self.predictions_hash,
            "preprocess_spec": self.preprocess_spec.to_dict(),
            "preprocess_spec_hash": self.preprocess_spec_hash,
            "runtime": self.runtime.to_dict(),
            "folds": [fold.to_dict() for fold in self.folds],
            "research_only": self.research_only,
            "production_ready": self.production_ready,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelArtifactBundle:
        expected = {
            "schema_version",
            "model_spec",
            "model_spec_hash",
            "validation_receipt",
            "validation_receipt_hash",
            "label_values_hash",
            "model_result",
            "model_result_hash",
            "predictions_hash",
            "preprocess_spec",
            "preprocess_spec_hash",
            "runtime",
            "folds",
            "research_only",
            "production_ready",
        }
        if set(value) != expected:
            raise ValueError("ModelArtifactBundle wire fields differ")
        raw_folds = value["folds"]
        if not isinstance(raw_folds, list) or not all(
            isinstance(item, Mapping) for item in raw_folds
        ):
            raise TypeError("model artifact folds must be an object array")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            model_spec=ModelSpec.from_mapping(
                _mapping(value["model_spec"], "model_spec")
            ),
            model_spec_hash=_text(value["model_spec_hash"], "model_spec_hash"),
            validation_receipt=ValidationReceipt.from_mapping(
                _mapping(value["validation_receipt"], "validation_receipt")
            ),
            validation_receipt_hash=_text(
                value["validation_receipt_hash"], "validation_receipt_hash"
            ),
            label_values_hash=_text(value["label_values_hash"], "label_values_hash"),
            model_result=ModelResultDescriptor.from_mapping(
                _mapping(value["model_result"], "model_result")
            ),
            model_result_hash=_text(value["model_result_hash"], "model_result_hash"),
            predictions_hash=_text(value["predictions_hash"], "predictions_hash"),
            preprocess_spec=ModelPreprocessSpec.from_mapping(
                _mapping(value["preprocess_spec"], "preprocess_spec")
            ),
            preprocess_spec_hash=_text(
                value["preprocess_spec_hash"], "preprocess_spec_hash"
            ),
            runtime=ModelRuntimeFingerprint.from_mapping(
                _mapping(value["runtime"], "runtime")
            ),
            folds=tuple(FoldModelArtifact.from_mapping(item) for item in raw_folds),
            research_only=_boolean(value["research_only"], "research_only"),
            production_ready=_boolean(value["production_ready"], "production_ready"),
        )


@dataclass(frozen=True, slots=True)
class ModelArtifactReference:
    artifact_id: str
    document_sha256: str
    filename: str
    size_bytes: int
    media_type: str = _MEDIA_TYPE
    schema_version: str = "model-artifact-reference/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "model-artifact-reference/v1":
            raise ValueError("unsupported model artifact reference schema")
        require_sha256(self.artifact_id, name="model artifact id")
        require_sha256(self.document_sha256, name="model artifact document sha256")
        if self.filename != f"{self.artifact_id}{_ARTIFACT_SUFFIX}":
            raise ValueError("model artifact filename differs")
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes <= 0
            or self.size_bytes > _MAXIMUM_ARTIFACT_BYTES
        ):
            raise ValueError("model artifact size is invalid")
        if self.media_type != _MEDIA_TYPE:
            raise ValueError("model artifact media type differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "document_sha256": self.document_sha256,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelArtifactReference:
        expected = {
            "schema_version",
            "artifact_id",
            "document_sha256",
            "filename",
            "size_bytes",
            "media_type",
        }
        if set(value) != expected:
            raise ValueError("ModelArtifactReference wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            artifact_id=_text(value["artifact_id"], "artifact_id"),
            document_sha256=_text(value["document_sha256"], "document_sha256"),
            filename=_text(value["filename"], "filename"),
            size_bytes=_integer(value["size_bytes"], "size_bytes"),
            media_type=_text(value["media_type"], "media_type"),
        )


def publish_model_artifact(
    artifact: ModelArtifactBundle,
    output_root: str | Path,
) -> ModelArtifactReference:
    if not isinstance(artifact, ModelArtifactBundle):
        raise TypeError("model artifact bundle type differs")
    artifact.runtime.verify_current()
    artifact_id = artifact.content_hash
    filename = f"{artifact_id}{_ARTIFACT_SUFFIX}"
    wire = artifact.to_dict()
    payload = canonical_json_bytes(wire) + b"\n"
    if len(payload) > _MAXIMUM_ARTIFACT_BYTES:
        raise ValueError("model artifact exceeds serialized size limit")
    write_immutable_json_document(
        output_root,
        filename=filename,
        value=wire,
    )
    return ModelArtifactReference(
        artifact_id=artifact_id,
        document_sha256=hashlib.sha256(payload).hexdigest(),
        filename=filename,
        size_bytes=len(payload),
    )


def load_model_artifact(
    path: str | Path,
    reference: ModelArtifactReference,
) -> ModelArtifactBundle:
    if not isinstance(reference, ModelArtifactReference):
        raise TypeError("model artifact reference type differs")
    target = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    value = load_immutable_json_document(
        target,
        expected_filename=reference.filename,
        expected_document_sha256=reference.document_sha256,
        maximum_bytes=_MAXIMUM_ARTIFACT_BYTES,
    )
    payload_size = len(canonical_json_bytes(dict(value))) + 1
    if payload_size != reference.size_bytes:
        raise ValueError("model artifact size differs")
    artifact = ModelArtifactBundle.from_mapping(value)
    if artifact.content_hash != reference.artifact_id:
        raise ValueError("model artifact content hash differs")
    artifact.runtime.verify_current()
    return artifact


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object")
    return MappingProxyType(dict(value))


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    return value


def _text_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{name} must be a text array")
    return tuple(value)


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


__all__ = [
    "FoldModelArtifact",
    "MODEL_RESULT_DESCRIPTOR_SCHEMA_VERSION",
    "MODEL_RESULT_SCHEMA_VERSION",
    "ModelArtifactBundle",
    "ModelArtifactReference",
    "ModelResultDescriptor",
    "ModelRuntimeFingerprint",
    "load_model_artifact",
    "model_result_content_hash",
    "publish_model_artifact",
]
