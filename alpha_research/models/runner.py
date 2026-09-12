from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge

from alpha_research.core.frequency import TradingCalendar
from alpha_research.core.hashing import hash_frame, require_sha256
from alpha_research.labels import LabelResult
from alpha_research.models.artifact import (
    FoldModelArtifact,
    MODEL_RESULT_SCHEMA_VERSION,
    ModelArtifactBundle,
    ModelResultDescriptor,
    ModelRuntimeFingerprint,
    model_result_content_hash,
)
from alpha_research.models.model_state import (
    portable_state_from_estimator,
    portable_state_hash,
)
from alpha_research.models.model_execution_governance import (
    ModelExecutionContext,
    ModelFitAttemptLedger,
)
from alpha_research.models.preprocessing import (
    FoldLocalPreprocessor,
    FoldTransformReceipt,
    ModelPreprocessSpec,
)
from alpha_research.models.spec import ModelEstimator, ModelSpec
from alpha_research.validation import (
    SplitFoldReceipt,
    ValidationReceipt,
    ValidationReceiptVerifier,
    ValidationSpec,
)


@dataclass(frozen=True, slots=True)
class ModelResult:
    model_spec_hash: str
    validation_receipt_hash: str
    label_values_hash: str
    predictions_hash: str
    coefficients_hash: str
    feature_importance_hash: str
    diagnostics_hash: str
    preprocess_spec_hash: str
    fold_transform_receipts: tuple[FoldTransformReceipt, ...]
    predictions: pd.DataFrame
    coefficients: pd.DataFrame
    feature_importance: pd.DataFrame
    diagnostics: pd.DataFrame
    schema_version: str = MODEL_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported model result schema")
        require_sha256(
            self.preprocess_spec_hash, name="model result preprocess_spec_hash"
        )
        predictions = pd.DataFrame(self.predictions).copy(deep=True)
        coefficients = pd.DataFrame(self.coefficients).copy(deep=True)
        importance = pd.DataFrame(self.feature_importance).copy(deep=True)
        diagnostics = pd.DataFrame(self.diagnostics).copy(deep=True)
        if hash_frame(predictions) != self.predictions_hash:
            raise ValueError("model prediction hash differs")
        if hash_frame(coefficients) != self.coefficients_hash:
            raise ValueError("model coefficient hash differs")
        if hash_frame(importance) != self.feature_importance_hash:
            raise ValueError("model feature-importance hash differs")
        if hash_frame(diagnostics) != self.diagnostics_hash:
            raise ValueError("model diagnostics hash differs")
        receipts = tuple(self.fold_transform_receipts)
        receipt_fold_ids = tuple(receipt.fold_id for receipt in receipts)
        coefficient_fold_ids = tuple(str(value) for value in coefficients.index)
        if receipt_fold_ids != coefficient_fold_ids:
            raise ValueError("model transform receipt folds differ")
        importance_fold_ids = tuple(str(value) for value in importance.index)
        diagnostic_fold_ids = tuple(str(value) for value in diagnostics.index)
        if receipt_fold_ids != importance_fold_ids:
            raise ValueError("model feature-importance folds differ")
        if receipt_fold_ids != diagnostic_fold_ids:
            raise ValueError("model diagnostic folds differ")
        if any(
            receipt.preprocess_spec_hash != self.preprocess_spec_hash
            for receipt in receipts
        ):
            raise ValueError("model transform receipt preprocessing binding differs")
        if len(set(receipt_fold_ids)) != len(receipt_fold_ids):
            raise ValueError("model transform receipt fold ids must be unique")
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "feature_importance", importance)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "fold_transform_receipts", receipts)
        # Make the complete v3 identity part of construction-time validation,
        # rather than deferring malformed lineage or shape counts until publish.
        _ = self.content_hash

    @property
    def content_hash(self) -> str:
        result_hash: str = model_result_content_hash(
            model_spec_hash=self.model_spec_hash,
            validation_receipt_hash=self.validation_receipt_hash,
            label_values_hash=self.label_values_hash,
            predictions_hash=self.predictions_hash,
            coefficients_hash=self.coefficients_hash,
            feature_importance_hash=self.feature_importance_hash,
            diagnostics_hash=self.diagnostics_hash,
            preprocess_spec_hash=self.preprocess_spec_hash,
            fold_transform_receipt_hashes=tuple(
                receipt.content_hash for receipt in self.fold_transform_receipts
            ),
            prediction_row_count=len(self.predictions),
            prediction_security_count=len(self.predictions.columns),
            fold_count=len(self.fold_transform_receipts),
            schema_version=self.schema_version,
        )
        return result_hash

    def verify_content(self) -> None:
        if hash_frame(self.predictions) != self.predictions_hash:
            raise RuntimeError("model predictions changed after construction")
        if hash_frame(self.coefficients) != self.coefficients_hash:
            raise RuntimeError("model coefficients changed after construction")
        if hash_frame(self.feature_importance) != self.feature_importance_hash:
            raise RuntimeError("model feature importance changed after construction")
        if hash_frame(self.diagnostics) != self.diagnostics_hash:
            raise RuntimeError("model diagnostics changed after construction")

    def descriptor(self) -> dict[str, object]:
        self.verify_content()
        return {
            "schema_version": self.schema_version,
            "result_hash": self.content_hash,
            "model_spec_hash": self.model_spec_hash,
            "validation_receipt_hash": self.validation_receipt_hash,
            "label_values_hash": self.label_values_hash,
            "predictions_hash": self.predictions_hash,
            "coefficients_hash": self.coefficients_hash,
            "feature_importance_hash": self.feature_importance_hash,
            "diagnostics_hash": self.diagnostics_hash,
            "preprocess_spec_hash": self.preprocess_spec_hash,
            "fold_transform_receipts": [
                receipt.to_dict() for receipt in self.fold_transform_receipts
            ],
            "prediction_row_count": len(self.predictions),
            "prediction_security_count": len(self.predictions.columns),
            "fold_count": len(self.coefficients),
        }


@dataclass(frozen=True, slots=True)
class _FoldFitOutcome:
    estimator: RegressorMixin | None
    predicted: npt.NDArray[np.float64]
    coefficients: npt.NDArray[np.float64]
    intercept: float
    importance: npt.NDArray[np.float64]
    coefficient_semantics: str
    importance_method: str
    thread_policy: str
    transform_receipt: FoldTransformReceipt
    validation_index: pd.MultiIndex
    train_observation_count: int
    validation_observation_count: int


@dataclass(frozen=True, slots=True)
class _FoldMaterializationOutcome:
    predictions: pd.DataFrame
    coefficient_row: dict[str, float | str]
    importance_row: dict[str, float | str]
    diagnostic_row: dict[str, float | int | str]
    transform_receipt: FoldTransformReceipt
    artifact: FoldModelArtifact | None


class TimeSeriesModelRunner:
    """Fit deterministic baselines on receipted train signals only."""

    def fit_predict_verified(
        self,
        spec: ModelSpec,
        feature_signals: Mapping[str, pd.DataFrame],
        labels: LabelResult,
        validation: ValidationReceipt,
        *,
        validation_spec: ValidationSpec,
        calendar: TradingCalendar,
        preprocessing: ModelPreprocessSpec | None = None,
        execution_ledger: ModelFitAttemptLedger | None = None,
        execution_context: ModelExecutionContext | None = None,
    ) -> ModelResult:
        """Run the strict entry after recomputing every validation fold."""

        verified = ValidationReceiptVerifier().verify(
            validation,
            spec=validation_spec,
            labels=labels,
            calendar=calendar,
        )
        return self._fit_predict(
            spec,
            feature_signals,
            labels,
            verified,
            preprocessing=preprocessing,
            execution_ledger=execution_ledger,
            execution_context=execution_context,
        )

    def fit_predict_verified_with_artifact(
        self,
        spec: ModelSpec,
        feature_signals: Mapping[str, pd.DataFrame],
        labels: LabelResult,
        validation: ValidationReceipt,
        *,
        validation_spec: ValidationSpec,
        calendar: TradingCalendar,
        preprocessing: ModelPreprocessSpec | None = None,
        execution_ledger: ModelFitAttemptLedger | None = None,
        execution_context: ModelExecutionContext | None = None,
    ) -> tuple[ModelResult, ModelArtifactBundle]:
        """Run the verified path and capture safe, portable fitted states."""

        verified = ValidationReceiptVerifier().verify(
            validation,
            spec=validation_spec,
            labels=labels,
            calendar=calendar,
        )
        result, artifact = self._execute(
            spec,
            feature_signals,
            labels,
            verified,
            preprocessing=preprocessing,
            capture_artifact=True,
            execution_ledger=execution_ledger,
            execution_context=execution_context,
        )
        if artifact is None:  # pragma: no cover - protected by capture_artifact.
            raise RuntimeError("verified model artifact capture failed")
        return result, artifact

    def fit_predict(
        self,
        spec: ModelSpec,
        feature_signals: Mapping[str, pd.DataFrame],
        labels: LabelResult,
        validation: ValidationReceipt,
        *,
        preprocessing: ModelPreprocessSpec | None = None,
        execution_ledger: ModelFitAttemptLedger | None = None,
        execution_context: ModelExecutionContext | None = None,
    ) -> ModelResult:
        """Run legacy deterministic baselines on a caller-supplied receipt.

        Non-linear models are deliberately excluded because their formal
        research path must recompute the purged validation receipt first.
        """

        if spec.estimator is ModelEstimator.EXTRA_TREES:
            raise ValueError("tree_model_requires_verified_validation")
        return self._fit_predict(
            spec,
            feature_signals,
            labels,
            validation,
            preprocessing=preprocessing,
            execution_ledger=execution_ledger,
            execution_context=execution_context,
        )

    def _fit_predict(
        self,
        spec: ModelSpec,
        feature_signals: Mapping[str, pd.DataFrame],
        labels: LabelResult,
        validation: ValidationReceipt,
        *,
        preprocessing: ModelPreprocessSpec | None = None,
        execution_ledger: ModelFitAttemptLedger | None = None,
        execution_context: ModelExecutionContext | None = None,
    ) -> ModelResult:
        result, artifact = self._execute(
            spec,
            feature_signals,
            labels,
            validation,
            preprocessing=preprocessing,
            capture_artifact=False,
            execution_ledger=execution_ledger,
            execution_context=execution_context,
        )
        if artifact is not None:  # pragma: no cover - protected by capture flag.
            raise RuntimeError("unexpected model artifact capture")
        return result

    def _execute(
        self,
        spec: ModelSpec,
        feature_signals: Mapping[str, pd.DataFrame],
        labels: LabelResult,
        validation: ValidationReceipt,
        *,
        preprocessing: ModelPreprocessSpec | None,
        capture_artifact: bool,
        execution_ledger: ModelFitAttemptLedger | None,
        execution_context: ModelExecutionContext | None,
    ) -> tuple[ModelResult, ModelArtifactBundle | None]:
        if (execution_ledger is None) != (execution_context is None):
            raise ValueError(
                "model execution ledger and context must be supplied together"
            )
        features = self._validate_bindings(spec, feature_signals, labels, validation)
        preprocess_spec = preprocessing or ModelPreprocessSpec()
        preprocessor = FoldLocalPreprocessor()
        prediction_rows: list[pd.DataFrame] = []
        coefficient_rows: list[dict[str, float | str]] = []
        importance_rows: list[dict[str, float | str]] = []
        diagnostic_rows: list[dict[str, float | int | str]] = []
        transform_receipts: list[FoldTransformReceipt] = []
        fold_artifacts: list[FoldModelArtifact] = []
        for fold in validation.folds:
            materialized = _materialize_fold_with_governance(
                spec=spec,
                preprocess_spec=preprocess_spec,
                preprocessor=preprocessor,
                features=features,
                labels=labels,
                fold=fold,
                capture_artifact=capture_artifact,
                validation_receipt_hash=validation.content_hash,
                execution_ledger=execution_ledger,
                execution_context=execution_context,
            )
            transform_receipts.append(materialized.transform_receipt)
            prediction_rows.append(materialized.predictions)
            coefficient_rows.append(materialized.coefficient_row)
            importance_rows.append(materialized.importance_row)
            diagnostic_rows.append(materialized.diagnostic_row)
            if materialized.artifact is not None:
                fold_artifacts.append(materialized.artifact)
        predictions = pd.concat(prediction_rows).sort_index(kind="stable")
        if predictions.index.duplicated().any():
            raise ValueError("model folds produced duplicate validation timestamps")
        predictions = predictions.reindex(columns=next(iter(features.values())).columns)
        coefficients = pd.DataFrame(coefficient_rows).set_index("fold_id")
        feature_importance = pd.DataFrame(importance_rows).set_index("fold_id")
        diagnostics = pd.DataFrame(diagnostic_rows).set_index("fold_id")
        result = ModelResult(
            model_spec_hash=spec.content_hash,
            validation_receipt_hash=validation.content_hash,
            label_values_hash=labels.labels_hash,
            predictions_hash=hash_frame(predictions),
            coefficients_hash=hash_frame(coefficients),
            feature_importance_hash=hash_frame(feature_importance),
            diagnostics_hash=hash_frame(diagnostics),
            preprocess_spec_hash=preprocess_spec.content_hash,
            fold_transform_receipts=tuple(transform_receipts),
            predictions=predictions,
            coefficients=coefficients,
            feature_importance=feature_importance,
            diagnostics=diagnostics,
        )
        if not capture_artifact:
            return result, None
        descriptor = ModelResultDescriptor(
            result_hash=result.content_hash,
            model_spec_hash=result.model_spec_hash,
            validation_receipt_hash=result.validation_receipt_hash,
            label_values_hash=result.label_values_hash,
            predictions_hash=result.predictions_hash,
            coefficients_hash=result.coefficients_hash,
            feature_importance_hash=result.feature_importance_hash,
            diagnostics_hash=result.diagnostics_hash,
            preprocess_spec_hash=result.preprocess_spec_hash,
            fold_transform_receipt_hashes=tuple(
                receipt.content_hash for receipt in result.fold_transform_receipts
            ),
            prediction_row_count=len(result.predictions),
            prediction_security_count=len(result.predictions.columns),
            fold_count=len(result.fold_transform_receipts),
        )
        artifact = ModelArtifactBundle(
            model_spec=spec,
            model_spec_hash=spec.content_hash,
            validation_receipt=validation,
            validation_receipt_hash=validation.content_hash,
            label_values_hash=labels.labels_hash,
            model_result=descriptor,
            model_result_hash=result.content_hash,
            predictions_hash=result.predictions_hash,
            preprocess_spec=preprocess_spec,
            preprocess_spec_hash=preprocess_spec.content_hash,
            runtime=ModelRuntimeFingerprint.current(),
            folds=tuple(fold_artifacts),
        )
        return result, artifact

    @staticmethod
    def _validate_bindings(
        spec: ModelSpec,
        feature_signals: Mapping[str, pd.DataFrame],
        labels: LabelResult,
        validation: ValidationReceipt,
    ) -> dict[str, pd.DataFrame]:
        labels.verify_content()
        if set(feature_signals) != set(spec.feature_signal_hashes):
            raise ValueError("model feature names differ from specification")
        features = {
            name: pd.DataFrame(feature_signals[name]).copy(deep=True)
            for name in sorted(feature_signals)
        }
        reference = next(iter(features.values()))
        for name, frame in features.items():
            if not frame.index.equals(reference.index) or not frame.columns.equals(
                reference.columns
            ):
                raise ValueError(f"model feature axes differ:{name}")
            if hash_frame(frame) != spec.feature_signal_hashes[name]:
                raise ValueError(f"model feature hash differs:{name}")
        if (
            not labels.labels.index.equals(reference.index)
            or not labels.labels.columns.equals(reference.columns)
            or not labels.validity.index.equals(reference.index)
            or not labels.validity.columns.equals(reference.columns)
        ):
            raise ValueError("model feature/label/validity axes differ")
        if labels.label_spec_hash != spec.label_spec_hash:
            raise ValueError("model label specification binding differs")
        if validation.validation_spec_hash != spec.validation_spec_hash:
            raise ValueError("model validation specification binding differs")
        if validation.label_spec_hash != labels.label_spec_hash:
            raise ValueError("model validation/label specification differs")
        if validation.labels_hash != labels.labels_hash:
            raise ValueError("model validation/label values differ")
        if validation.windows_hash != labels.windows_hash:
            raise ValueError("model validation/label windows differ")
        return features


def _materialize_fold_with_governance(
    *,
    spec: ModelSpec,
    preprocess_spec: ModelPreprocessSpec,
    preprocessor: FoldLocalPreprocessor,
    features: Mapping[str, pd.DataFrame],
    labels: LabelResult,
    fold: SplitFoldReceipt,
    capture_artifact: bool,
    validation_receipt_hash: str,
    execution_ledger: ModelFitAttemptLedger | None,
    execution_context: ModelExecutionContext | None,
) -> _FoldMaterializationOutcome:
    if execution_ledger is None:
        if execution_context is not None:  # pragma: no cover - checked by caller.
            raise RuntimeError("unexpected model execution context")
        return _materialize_fold(
            spec=spec,
            preprocess_spec=preprocess_spec,
            preprocessor=preprocessor,
            features=features,
            labels=labels,
            fold=fold,
            capture_artifact=capture_artifact,
        )
    if execution_context is None:  # pragma: no cover - checked by caller.
        raise RuntimeError("missing model execution context")
    # Reservation happens before the first fold-local preprocessing operation.
    # Success is terminalized only after the fitted estimator has been converted
    # into complete fold evidence (portable state when requested, predictions,
    # coefficients, importance and diagnostics).  An abrupt process death leaves
    # an open but still consumed attempt.
    with execution_ledger.attempt(
        context=execution_context,
        model_spec_hash=spec.content_hash,
        validation_receipt_hash=validation_receipt_hash,
        fold_id=fold.fold_id,
    ):
        return _materialize_fold(
            spec=spec,
            preprocess_spec=preprocess_spec,
            preprocessor=preprocessor,
            features=features,
            labels=labels,
            fold=fold,
            capture_artifact=capture_artifact,
        )


def _materialize_fold(
    *,
    spec: ModelSpec,
    preprocess_spec: ModelPreprocessSpec,
    preprocessor: FoldLocalPreprocessor,
    features: Mapping[str, pd.DataFrame],
    labels: LabelResult,
    fold: SplitFoldReceipt,
    capture_artifact: bool,
) -> _FoldMaterializationOutcome:
    fold_fit = _prepare_and_fit_fold(
        spec=spec,
        preprocess_spec=preprocess_spec,
        preprocessor=preprocessor,
        features=features,
        labels=labels,
        fold=fold,
    )
    transform_receipt = fold_fit.transform_receipt
    artifact: FoldModelArtifact | None = None
    if capture_artifact:
        state = portable_state_from_estimator(spec, fold_fit.estimator)
        artifact = FoldModelArtifact(
            fold_id=fold.fold_id,
            estimator=spec.estimator,
            transform_receipt=transform_receipt,
            estimator_state=state,
            estimator_state_hash=portable_state_hash(state),
        )
    fold_predictions = pd.Series(
        fold_fit.predicted,
        index=fold_fit.validation_index,
    ).unstack("security")
    fold_predictions.index.name = "signal_timestamp"
    if len(fold_fit.coefficients) != len(features):
        raise RuntimeError("model coefficient dimension differs")
    coefficient_row: dict[str, float | str] = {
        "fold_id": fold.fold_id,
        "intercept": fold_fit.intercept,
    }
    coefficient_row.update(
        {
            name: float(fold_fit.coefficients[offset])
            for offset, name in enumerate(features)
        }
    )
    if len(fold_fit.importance) != len(features):
        raise RuntimeError("model feature-importance dimension differs")
    importance_row: dict[str, float | str] = {"fold_id": fold.fold_id}
    importance_row.update(
        {
            name: float(fold_fit.importance[offset])
            for offset, name in enumerate(features)
        }
    )
    diagnostic_target, diagnostic_mask = _paired_targets(
        labels,
        fold_fit.validation_index,
    )
    diagnostic_prediction = fold_fit.predicted[diagnostic_mask]
    residual = diagnostic_target - diagnostic_prediction
    calibration_slope, calibration_intercept = _calibration(
        diagnostic_prediction,
        diagnostic_target,
    )
    diagnostic_row: dict[str, float | int | str] = {
        "fold_id": fold.fold_id,
        "train_observation_count": fold_fit.train_observation_count,
        "validation_observation_count": fold_fit.validation_observation_count,
        "validation_rmse": _rmse(residual),
        "validation_mae": _mae(residual),
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
        "coefficient_semantics": fold_fit.coefficient_semantics,
        "feature_importance_method": fold_fit.importance_method,
        "thread_policy": fold_fit.thread_policy,
        "internal_holdout": "disabled",
    }
    return _FoldMaterializationOutcome(
        predictions=fold_predictions,
        coefficient_row=coefficient_row,
        importance_row=importance_row,
        diagnostic_row=diagnostic_row,
        transform_receipt=transform_receipt,
        artifact=artifact,
    )


def _prepare_and_fit_fold(
    *,
    spec: ModelSpec,
    preprocess_spec: ModelPreprocessSpec,
    preprocessor: FoldLocalPreprocessor,
    features: Mapping[str, pd.DataFrame],
    labels: LabelResult,
    fold: SplitFoldReceipt,
) -> _FoldFitOutcome:
    reference_index = pd.DatetimeIndex(next(iter(features.values())).index)
    train_timestamps = _receipt_timestamps(fold.train_signals, reference_index)
    validation_timestamps = _receipt_timestamps(
        fold.validation_signals, reference_index
    )
    x_train, y_train, train_index = _training_samples(
        features, labels, train_timestamps
    )
    x_validation, validation_index = _prediction_samples(
        features, validation_timestamps
    )
    if len(x_train) <= len(features):
        raise ValueError(f"insufficient_model_train_observations:{fold.fold_id}")
    if len(x_validation) == 0:
        raise ValueError(f"insufficient_model_validation_observations:{fold.fold_id}")
    return _fit_fold(
        spec=spec,
        preprocess_spec=preprocess_spec,
        preprocessor=preprocessor,
        fold_id=fold.fold_id,
        feature_names=tuple(features),
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        train_index=train_index,
        validation_index=validation_index,
    )


def _fit_fold(
    *,
    spec: ModelSpec,
    preprocess_spec: ModelPreprocessSpec,
    preprocessor: FoldLocalPreprocessor,
    fold_id: str,
    feature_names: tuple[str, ...],
    x_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
    x_validation: npt.NDArray[np.float64],
    train_index: pd.Index,
    validation_index: pd.Index,
) -> _FoldFitOutcome:
    transformed_train, transformed_validation, transform_receipt = (
        preprocessor.fit_transform(
            preprocess_spec,
            fold_id=fold_id,
            feature_names=feature_names,
            x_train=x_train,
            x_validation=x_validation,
            train_index=train_index,
            validation_index=validation_index,
        )
    )
    estimator: RegressorMixin | None
    predicted: npt.NDArray[np.float64]
    if spec.estimator is ModelEstimator.FACTOR_ONLY:
        estimator = None
        predicted = transformed_validation[:, 0].astype(float)
        coefficients = np.array([1.0], dtype=float)
        intercept = 0.0
        importance = np.array([1.0], dtype=float)
        coefficient_semantics = "identity_passthrough"
        importance_method = "identity_single_feature"
        thread_policy = "not_applicable"
    else:
        estimator = _build_estimator(spec)
        estimator.fit(transformed_train, y_train)
        predicted = np.asarray(estimator.predict(transformed_validation), dtype=float)
        if spec.estimator is ModelEstimator.EXTRA_TREES:
            if not isinstance(estimator, ExtraTreesRegressor):
                raise RuntimeError("extra-trees estimator type differs")
            coefficients = np.full(len(feature_names), np.nan, dtype=float)
            intercept = float("nan")
            importance = _normalized_importance(
                np.asarray(estimator.feature_importances_, dtype=float)
            )
            coefficient_semantics = "not_applicable"
            importance_method = "tree_impurity_decrease"
            thread_policy = "single_thread"
        else:
            coefficients = np.asarray(estimator.coef_, dtype=float).reshape(-1)
            intercept = float(estimator.intercept_)
            importance = _normalized_importance(coefficients)
            coefficient_semantics = "linear"
            importance_method = "normalized_absolute_coefficient"
            thread_policy = "not_applicable"
    return _FoldFitOutcome(
        estimator=estimator,
        predicted=predicted,
        coefficients=coefficients,
        intercept=intercept,
        importance=importance,
        coefficient_semantics=coefficient_semantics,
        importance_method=importance_method,
        thread_policy=thread_policy,
        transform_receipt=transform_receipt,
        validation_index=validation_index,
        train_observation_count=len(x_train),
        validation_observation_count=len(x_validation),
    )


def _training_samples(
    features: Mapping[str, pd.DataFrame],
    labels: LabelResult,
    timestamps: pd.DatetimeIndex,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    pd.MultiIndex,
]:
    x, feature_index = _prediction_samples(features, timestamps)
    label_series = labels.labels.stack(future_stack=True).reindex(feature_index)
    validity = labels.validity.stack(future_stack=True).reindex(feature_index)
    paired = validity.fillna(False).astype(bool).to_numpy(dtype=bool) & np.isfinite(
        pd.to_numeric(label_series, errors="coerce").to_numpy(dtype=float)
    )
    sample_index = feature_index[paired]
    y = pd.to_numeric(label_series.loc[sample_index], errors="coerce").to_numpy(
        dtype=float
    )
    return x[paired], y, sample_index


def _prediction_samples(
    features: Mapping[str, pd.DataFrame],
    timestamps: pd.DatetimeIndex,
) -> tuple[npt.NDArray[np.float64], pd.MultiIndex]:
    """Build the PIT prediction cohort without consulting future labels."""

    reference = next(iter(features.values()))
    if len(timestamps.difference(reference.index)):
        raise ValueError("model split signals are unavailable in feature panels")
    feature_series = [
        frame.reindex(index=timestamps, columns=reference.columns).stack(
            future_stack=True
        )
        for frame in features.values()
    ]
    complete: npt.NDArray[np.bool_] = np.ones(len(feature_series[0]), dtype=bool)
    for values in feature_series:
        complete &= np.isfinite(
            pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
        )
    sample_index = feature_series[0].index[complete]
    sample_index = sample_index.set_names(["signal_timestamp", "security"])
    x = np.column_stack(
        [
            pd.to_numeric(values.loc[sample_index], errors="raise").to_numpy(
                dtype=float
            )
            for values in feature_series
        ]
    )
    return x, sample_index


def _paired_targets(
    labels: LabelResult,
    prediction_index: pd.MultiIndex,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Pair labels only after the feature-determined predictions exist."""

    label_series = labels.labels.stack(future_stack=True).reindex(prediction_index)
    validity = labels.validity.stack(future_stack=True).reindex(prediction_index)
    target = pd.to_numeric(label_series, errors="coerce").to_numpy(dtype=float)
    paired = validity.fillna(False).astype(bool).to_numpy(dtype=bool) & np.isfinite(
        target
    )
    return target[paired], paired


def _build_estimator(spec: ModelSpec) -> RegressorMixin:
    estimator = spec.estimator
    if estimator is ModelEstimator.OLS:
        return LinearRegression(fit_intercept=spec.fit_intercept)
    if estimator is ModelEstimator.RIDGE:
        return Ridge(
            alpha=float(spec.hyperparameters["alpha"]),
            fit_intercept=spec.fit_intercept,
            random_state=spec.random_seed,
        )
    if estimator is ModelEstimator.ELASTIC_NET:
        return ElasticNet(
            alpha=float(spec.hyperparameters["alpha"]),
            l1_ratio=float(spec.hyperparameters["l1_ratio"]),
            fit_intercept=spec.fit_intercept,
            max_iter=int(spec.hyperparameters["max_iter"]),
            random_state=spec.random_seed,
            selection="cyclic",
        )
    if estimator is ModelEstimator.EXTRA_TREES:
        return ExtraTreesRegressor(
            n_estimators=int(spec.hyperparameters["n_estimators"]),
            max_depth=int(spec.hyperparameters["max_depth"]),
            min_samples_leaf=int(spec.hyperparameters["min_samples_leaf"]),
            max_features=float(spec.hyperparameters["max_features"]),
            criterion="squared_error",
            bootstrap=False,
            oob_score=False,
            n_jobs=1,
            random_state=spec.random_seed,
            warm_start=False,
            ccp_alpha=0.0,
            max_leaf_nodes=None,
        )
    raise RuntimeError("unsupported normalized estimator")  # pragma: no cover


def _normalized_importance(
    values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    absolute = np.abs(np.asarray(values, dtype=float).reshape(-1))
    if not np.isfinite(absolute).all():
        raise ValueError("model feature importance must be finite")
    total = float(absolute.sum())
    if total <= np.finfo(float).eps:
        return np.zeros_like(absolute, dtype=float)
    return (absolute / total).astype(float)


def _receipt_timestamps(
    values: tuple[str, ...], reference: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    parsed = pd.DatetimeIndex(pd.to_datetime(list(values)))
    if reference.tz is None or parsed.tz is None:
        raise ValueError("model split timestamps must be timezone-aware")
    return parsed.tz_convert(reference.tz)


def _calibration(
    predicted: npt.NDArray[np.float64],
    observed: npt.NDArray[np.float64],
) -> tuple[float, float]:
    if len(predicted) < 2 or float(np.std(predicted)) <= np.finfo(float).eps:
        return float("nan"), float("nan")
    design = np.column_stack([predicted, np.ones(len(predicted))])
    slope, intercept = np.linalg.lstsq(design, observed, rcond=None)[0]
    return float(slope), float(intercept)


def _rmse(residual: npt.NDArray[np.float64]) -> float:
    if len(residual) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(residual))))


def _mae(residual: npt.NDArray[np.float64]) -> float:
    if len(residual) == 0:
        return float("nan")
    return float(np.mean(np.abs(residual)))


__all__ = ["ModelResult", "TimeSeriesModelRunner"]
