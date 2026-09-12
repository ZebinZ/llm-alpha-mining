"""Typed upstream artifacts consumed by governed model training.

The three producers mirror the real parent stages.  In particular, the
factor-stage method has no label input and binds scoring eligibility only to
the factor contract's data authority.  This makes label-derived eligibility
an invalid API path rather than a descriptive convention.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
from types import MappingProxyType
from typing import Mapping, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from alpha_research.core.frequency import TradingCalendar
from alpha_research.core.hashing import hash_frame, hash_json, require_sha256
from alpha_research.labels import LabelResult
from alpha_research.research.artifacts import StageContract
from alpha_research.research.lineage import ModelFeatureBinding
from alpha_research.research.model_training_inputs import (
    SCORING_ELIGIBILITY_SOURCE_SCHEMA,
    DataFrameArtifactReferenceV2,
    ModelTrainingInputError,
    ScoringEligibilityPolicyV1,
)
from alpha_research.research.spec import ResearchStage
from alpha_research.validation import (
    ValidationReceipt,
    ValidationReceiptVerifier,
    ValidationSpec,
    validation_calendar_hash,
)
from factor_production.v5.artifacts import ArtifactStore


_PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
_LABEL_BUNDLE_SCHEMA = "model-label-artifacts/v1"
_VALIDATION_BUNDLE_SCHEMA = "model-validation-artifacts/v1"


def _error(code: str, detail: str) -> ModelTrainingInputError:
    return ModelTrainingInputError(code, detail)


def _object(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise _error("invalid_upstream_bundle", f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _exact_fields(
    value: Mapping[str, object], expected: frozenset[str], *, name: str
) -> None:
    if frozenset(value) != expected:
        raise _error("upstream_bundle_fields_differ", f"{name} fields differ")


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise _error("invalid_upstream_bundle", f"{name} must be a hash")
    try:
        return cast(str, require_sha256(value, name=name))
    except ValueError as exc:
        raise _error("invalid_upstream_bundle", str(exc)) from exc


def _reference(value: object, *, name: str) -> DataFrameArtifactReferenceV2:
    try:
        return DataFrameArtifactReferenceV2.from_mapping(_object(value, name=name))
    except ModelTrainingInputError as exc:
        raise _error("invalid_upstream_bundle", f"{name} is invalid") from exc


def _require_role(
    reference: DataFrameArtifactReferenceV2, *, role: str, name: str
) -> None:
    if not isinstance(reference, DataFrameArtifactReferenceV2):
        raise _error("invalid_upstream_bundle", f"{name} reference type differs")
    if reference.role != role:
        raise _error("artifact_role_mismatch", f"{name} role differs")


@dataclass(frozen=True, slots=True)
class FactorModelTrainingArtifactsV1:
    """Strict factor-stage feature and label-free eligibility payload."""

    feature_artifacts: Mapping[str, DataFrameArtifactReferenceV2]
    scoring_eligibility_policy: ScoringEligibilityPolicyV1
    scoring_eligibility_hash: str
    scoring_eligibility: DataFrameArtifactReferenceV2

    def __post_init__(self) -> None:
        if not isinstance(self.feature_artifacts, Mapping):
            raise _error(
                "invalid_upstream_bundle", "feature artifacts must be a mapping"
            )
        features = dict(sorted(self.feature_artifacts.items()))
        if not features or any(
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or not isinstance(reference, DataFrameArtifactReferenceV2)
            for name, reference in features.items()
        ):
            raise _error("invalid_upstream_bundle", "feature artifacts are invalid")
        for name, reference in features.items():
            _require_role(reference, role="model_feature", name=f"feature:{name}")
        if not isinstance(self.scoring_eligibility_policy, ScoringEligibilityPolicyV1):
            raise _error("invalid_upstream_bundle", "eligibility policy type differs")
        eligibility_hash = _digest(
            self.scoring_eligibility_hash,
            name="scoring_eligibility_hash",
        )
        _require_role(
            self.scoring_eligibility,
            role="scoring_eligibility",
            name="scoring eligibility",
        )
        expected_shape = (
            self.scoring_eligibility.row_count,
            self.scoring_eligibility.column_count,
            self.scoring_eligibility.index_hash,
            self.scoring_eligibility.columns_hash,
        )
        if any(
            (
                reference.row_count,
                reference.column_count,
                reference.index_hash,
                reference.columns_hash,
            )
            != expected_shape
            for reference in features.values()
        ):
            raise _error(
                "upstream_panel_mismatch",
                "feature and eligibility artifact axes differ",
            )
        logical_names = [
            *(reference.logical_name for reference in features.values()),
            self.scoring_eligibility.logical_name,
        ]
        if len(logical_names) != len(set(logical_names)):
            raise _error("invalid_upstream_bundle", "artifact logical names repeat")
        object.__setattr__(self, "feature_artifacts", MappingProxyType(features))
        object.__setattr__(self, "scoring_eligibility_hash", eligibility_hash)

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_result_payload()))

    def to_result_payload(self) -> dict[str, object]:
        eligibility = self.scoring_eligibility
        policy = self.scoring_eligibility_policy
        return {
            "model_feature_artifacts": {
                name: reference.to_dict()
                for name, reference in self.feature_artifacts.items()
            },
            "scoring_eligibility_source": {
                "schema_version": SCORING_ELIGIBILITY_SOURCE_SCHEMA,
                "policy": policy.to_dict(),
                "policy_hash": policy.content_hash,
                "frame_hash": self.scoring_eligibility_hash,
                "artifact_reference": eligibility.to_dict(),
                "artifact_reference_hash": eligibility.content_hash,
            },
        }

    @classmethod
    def from_result_payload(
        cls, value: Mapping[str, object]
    ) -> "FactorModelTrainingArtifactsV1":
        _exact_fields(
            value,
            frozenset({"model_feature_artifacts", "scoring_eligibility_source"}),
            name="factor model-training result payload",
        )
        raw_features = _object(
            value["model_feature_artifacts"], name="model_feature_artifacts"
        )
        features = {
            name: _reference(reference, name=f"feature:{name}")
            for name, reference in raw_features.items()
        }
        source = _object(
            value["scoring_eligibility_source"],
            name="scoring_eligibility_source",
        )
        _exact_fields(
            source,
            frozenset(
                {
                    "schema_version",
                    "policy",
                    "policy_hash",
                    "frame_hash",
                    "artifact_reference",
                    "artifact_reference_hash",
                }
            ),
            name="scoring_eligibility_source",
        )
        if source["schema_version"] != SCORING_ELIGIBILITY_SOURCE_SCHEMA:
            raise _error("invalid_upstream_bundle", "eligibility source schema differs")
        try:
            policy = ScoringEligibilityPolicyV1.from_mapping(
                _object(source["policy"], name="eligibility policy")
            )
        except ModelTrainingInputError as exc:
            raise _error(
                "invalid_upstream_bundle", "eligibility policy is invalid"
            ) from exc
        eligibility = _reference(
            source["artifact_reference"], name="eligibility artifact reference"
        )
        if source["policy_hash"] != policy.content_hash:
            raise _error("upstream_hash_mismatch", "eligibility policy hash differs")
        if source["artifact_reference_hash"] != eligibility.content_hash:
            raise _error(
                "upstream_hash_mismatch",
                "eligibility artifact reference hash differs",
            )
        return cls(
            feature_artifacts=features,
            scoring_eligibility_policy=policy,
            scoring_eligibility_hash=_digest(
                source["frame_hash"], name="eligibility frame_hash"
            ),
            scoring_eligibility=eligibility,
        )


@dataclass(frozen=True, slots=True)
class LabelModelTrainingArtifactsV1:
    """Strict label-stage identity plus four immutable parquet references."""

    label_spec_hash: str
    label_view_hash: str
    label_benchmark_hash: str | None
    label_values_hash: str
    label_windows_hash: str
    label_validity_hash: str
    label_diagnostics_hash: str
    label_values: DataFrameArtifactReferenceV2
    label_windows: DataFrameArtifactReferenceV2
    label_validity: DataFrameArtifactReferenceV2
    label_diagnostics: DataFrameArtifactReferenceV2

    def __post_init__(self) -> None:
        for name in (
            "label_spec_hash",
            "label_view_hash",
            "label_values_hash",
            "label_windows_hash",
            "label_validity_hash",
            "label_diagnostics_hash",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if self.label_benchmark_hash is not None:
            object.__setattr__(
                self,
                "label_benchmark_hash",
                _digest(self.label_benchmark_hash, name="label_benchmark_hash"),
            )
        references = (
            ("label_values", self.label_values),
            ("label_windows", self.label_windows),
            ("label_validity", self.label_validity),
            ("label_diagnostics", self.label_diagnostics),
        )
        for role, reference in references:
            _require_role(reference, role=role, name=role)
        values_shape = (
            self.label_values.row_count,
            self.label_values.column_count,
            self.label_values.index_hash,
            self.label_values.columns_hash,
        )
        validity_shape = (
            self.label_validity.row_count,
            self.label_validity.column_count,
            self.label_validity.index_hash,
            self.label_validity.columns_hash,
        )
        if validity_shape != values_shape:
            raise _error("upstream_panel_mismatch", "label validity axes differ")
        if any(
            reference.row_count != self.label_values.row_count
            for reference in (self.label_windows, self.label_diagnostics)
        ):
            raise _error("upstream_panel_mismatch", "label row counts differ")
        logical_names = [reference.logical_name for _, reference in references]
        if len(logical_names) != len(set(logical_names)):
            raise _error("invalid_upstream_bundle", "label logical names repeat")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_result_payload()))

    def to_result_payload(self) -> dict[str, object]:
        return {
            "model_label_artifacts": {
                "schema_version": _LABEL_BUNDLE_SCHEMA,
                "label_spec_hash": self.label_spec_hash,
                "label_view_hash": self.label_view_hash,
                "label_benchmark_hash": self.label_benchmark_hash,
                "label_values_hash": self.label_values_hash,
                "label_windows_hash": self.label_windows_hash,
                "label_validity_hash": self.label_validity_hash,
                "label_diagnostics_hash": self.label_diagnostics_hash,
                "label_values": self.label_values.to_dict(),
                "label_windows": self.label_windows.to_dict(),
                "label_validity": self.label_validity.to_dict(),
                "label_diagnostics": self.label_diagnostics.to_dict(),
            }
        }

    @classmethod
    def from_result_payload(
        cls, value: Mapping[str, object]
    ) -> "LabelModelTrainingArtifactsV1":
        _exact_fields(
            value,
            frozenset({"model_label_artifacts"}),
            name="label model-training result payload",
        )
        bundle = _object(value["model_label_artifacts"], name="model_label_artifacts")
        expected = frozenset(
            {
                "schema_version",
                "label_spec_hash",
                "label_view_hash",
                "label_benchmark_hash",
                "label_values_hash",
                "label_windows_hash",
                "label_validity_hash",
                "label_diagnostics_hash",
                "label_values",
                "label_windows",
                "label_validity",
                "label_diagnostics",
            }
        )
        _exact_fields(bundle, expected, name="model_label_artifacts")
        if bundle["schema_version"] != _LABEL_BUNDLE_SCHEMA:
            raise _error("invalid_upstream_bundle", "label bundle schema differs")
        benchmark = bundle["label_benchmark_hash"]
        if benchmark is not None and not isinstance(benchmark, str):
            raise _error("invalid_upstream_bundle", "label benchmark hash is invalid")
        return cls(
            label_spec_hash=_digest(bundle["label_spec_hash"], name="label_spec_hash"),
            label_view_hash=_digest(bundle["label_view_hash"], name="label_view_hash"),
            label_benchmark_hash=benchmark,
            label_values_hash=_digest(
                bundle["label_values_hash"], name="label_values_hash"
            ),
            label_windows_hash=_digest(
                bundle["label_windows_hash"], name="label_windows_hash"
            ),
            label_validity_hash=_digest(
                bundle["label_validity_hash"], name="label_validity_hash"
            ),
            label_diagnostics_hash=_digest(
                bundle["label_diagnostics_hash"], name="label_diagnostics_hash"
            ),
            label_values=_reference(bundle["label_values"], name="label_values"),
            label_windows=_reference(bundle["label_windows"], name="label_windows"),
            label_validity=_reference(bundle["label_validity"], name="label_validity"),
            label_diagnostics=_reference(
                bundle["label_diagnostics"], name="label_diagnostics"
            ),
        )


@dataclass(frozen=True, slots=True)
class ValidationModelTrainingArtifactsV1:
    """Strict resolver-compatible validation contract bundle."""

    validation_spec: ValidationSpec
    validation_receipt: ValidationReceipt
    trading_calendar: TradingCalendar

    def __post_init__(self) -> None:
        if not isinstance(self.validation_spec, ValidationSpec):
            raise _error("invalid_upstream_bundle", "validation spec type differs")
        if not isinstance(self.validation_receipt, ValidationReceipt):
            raise _error("invalid_upstream_bundle", "validation receipt type differs")
        if not isinstance(self.trading_calendar, TradingCalendar):
            raise _error("invalid_upstream_bundle", "trading calendar type differs")
        receipt = self.validation_receipt
        if receipt.validation_spec_hash != self.validation_spec.content_hash:
            raise _error("upstream_hash_mismatch", "validation spec hash differs")
        if receipt.calendar_hash != validation_calendar_hash(self.trading_calendar):
            raise _error("upstream_hash_mismatch", "validation calendar hash differs")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_result_payload()))

    def to_result_payload(self) -> dict[str, object]:
        return {
            "model_validation_artifacts": {
                "schema_version": _VALIDATION_BUNDLE_SCHEMA,
                "validation_spec_hash": self.validation_spec.content_hash,
                "validation_spec": self.validation_spec.to_dict(),
                "validation_receipt_hash": self.validation_receipt.content_hash,
                "validation_receipt": self.validation_receipt.to_dict(),
                "trading_calendar_content_hash": self.trading_calendar.content_hash,
                "trading_calendar": self.trading_calendar.to_dict(),
            }
        }

    @classmethod
    def from_result_payload(
        cls, value: Mapping[str, object]
    ) -> "ValidationModelTrainingArtifactsV1":
        _exact_fields(
            value,
            frozenset({"model_validation_artifacts"}),
            name="validation model-training result payload",
        )
        bundle = _object(
            value["model_validation_artifacts"], name="model_validation_artifacts"
        )
        expected = frozenset(
            {
                "schema_version",
                "validation_spec_hash",
                "validation_spec",
                "validation_receipt_hash",
                "validation_receipt",
                "trading_calendar_content_hash",
                "trading_calendar",
            }
        )
        _exact_fields(bundle, expected, name="model_validation_artifacts")
        if bundle["schema_version"] != _VALIDATION_BUNDLE_SCHEMA:
            raise _error("invalid_upstream_bundle", "validation bundle schema differs")
        try:
            spec = ValidationSpec.from_mapping(
                _object(bundle["validation_spec"], name="validation_spec")
            )
            receipt = ValidationReceipt.from_mapping(
                _object(bundle["validation_receipt"], name="validation_receipt")
            )
            calendar = TradingCalendar.from_mapping(
                _object(bundle["trading_calendar"], name="trading_calendar")
            )
        except (TypeError, ValueError) as exc:
            raise _error(
                "invalid_upstream_bundle", "validation payload is invalid"
            ) from exc
        expected_hashes = {
            "validation_spec_hash": spec.content_hash,
            "validation_receipt_hash": receipt.content_hash,
            "trading_calendar_content_hash": calendar.content_hash,
        }
        if any(bundle[name] != digest for name, digest in expected_hashes.items()):
            raise _error("upstream_hash_mismatch", "validation bundle hash differs")
        return cls(
            validation_spec=spec,
            validation_receipt=receipt,
            trading_calendar=calendar,
        )


class ModelTrainingUpstreamBundleProducer:
    """Publish immutable parent-stage inputs without crossing stage authority."""

    def __init__(self, artifact_store: ArtifactStore) -> None:
        if type(artifact_store) is not ArtifactStore:
            raise TypeError("upstream bundle producer requires an exact ArtifactStore")
        self._artifact_store = artifact_store

    def publish_factor_inputs(
        self,
        *,
        factor_contract: StageContract,
        feature_bindings: tuple[ModelFeatureBinding, ...],
        feature_frames: Mapping[str, pd.DataFrame],
        scoring_eligibility_policy: ScoringEligibilityPolicyV1,
        scoring_eligibility: pd.DataFrame,
    ) -> FactorModelTrainingArtifactsV1:
        """Publish features and eligibility using factor-only authority."""

        if not isinstance(factor_contract, StageContract) or (
            ResearchStage(factor_contract.stage) is not ResearchStage.FACTOR_GENERATION
        ):
            raise _error(
                "eligibility_authority_mismatch", "factor contract is required"
            )
        bindings = tuple(feature_bindings)
        if not bindings or not all(
            isinstance(binding, ModelFeatureBinding) for binding in bindings
        ):
            raise _error("invalid_upstream_bundle", "feature bindings are invalid")
        binding_by_name = {binding.feature_name: binding for binding in bindings}
        if len(binding_by_name) != len(bindings):
            raise _error("invalid_upstream_bundle", "feature binding names repeat")
        if not isinstance(feature_frames, Mapping) or set(feature_frames) != set(
            binding_by_name
        ):
            raise _error("invalid_upstream_bundle", "feature frame set differs")
        allowed_factors = {
            digest
            for role, digest in factor_contract.component_bindings.items()
            if role.startswith("factor:")
        }
        if any(binding.factor_spec_hash not in allowed_factors for binding in bindings):
            raise _error("eligibility_authority_mismatch", "feature factor is unbound")
        if not isinstance(scoring_eligibility_policy, ScoringEligibilityPolicyV1):
            raise _error("invalid_upstream_bundle", "eligibility policy type differs")
        allowed_sources = tuple(
            sorted(
                digest
                for role, digest in factor_contract.component_bindings.items()
                if role.startswith("data:")
            )
        )
        if scoring_eligibility_policy.source_data_hashes != allowed_sources:
            raise _error(
                "eligibility_authority_mismatch",
                "eligibility sources differ from factor data authority",
            )
        eligibility = _boolean_panel(scoring_eligibility)
        frames = {
            name: _feature_panel(feature_frames[name], name=name)
            for name in sorted(feature_frames)
        }
        for name, frame in frames.items():
            binding = binding_by_name[name]
            if hash_frame(frame) != binding.signal_hash:
                raise _error("upstream_hash_mismatch", f"feature:{name} signal differs")
            _require_same_axes(frame, eligibility, name=f"feature:{name}")
            values = frame.to_numpy(dtype=float, na_value=np.nan)
            if not np.isfinite(values[eligibility.to_numpy(dtype=bool)]).all():
                raise _error(
                    "invalid_upstream_bundle",
                    f"feature:{name} has non-finite eligible values",
                )
        serialized = {
            name: _parquet_bytes(frame, name=f"feature:{name}")
            for name, frame in frames.items()
        }
        eligibility_payload = _parquet_bytes(eligibility, name="scoring eligibility")
        feature_references = {
            name: self._publish_frame(
                logical_name=f"model.feature.{name}",
                role="model_feature",
                frame=frames[name],
                payload=serialized[name],
            )
            for name in frames
        }
        eligibility_reference = self._publish_frame(
            logical_name="model.scoring_eligibility",
            role="scoring_eligibility",
            frame=eligibility,
            payload=eligibility_payload,
        )
        return FactorModelTrainingArtifactsV1(
            feature_artifacts=feature_references,
            scoring_eligibility_policy=scoring_eligibility_policy,
            scoring_eligibility_hash=hash_frame(eligibility),
            scoring_eligibility=eligibility_reference,
        )

    def publish_label_inputs(
        self, *, label_result: LabelResult
    ) -> LabelModelTrainingArtifactsV1:
        if not isinstance(label_result, LabelResult):
            raise _error("invalid_upstream_bundle", "label result type differs")
        try:
            label_result.verify_content()
        except RuntimeError as exc:
            raise _error("upstream_hash_mismatch", "label result changed") from exc
        labels = pd.DataFrame(label_result.labels).copy(deep=True)
        windows = pd.DataFrame(label_result.label_windows).copy(deep=True)
        validity = _boolean_panel(label_result.validity)
        diagnostics = pd.DataFrame(label_result.diagnostics).copy(deep=True)
        _require_same_axes(validity, labels, name="label validity")
        if len(windows.index) != len(labels.index) or len(diagnostics.index) != len(
            labels.index
        ):
            raise _error("upstream_panel_mismatch", "label row counts differ")
        frames = {
            "label_values": ("model.label.values", labels),
            "label_windows": ("model.label.windows", windows),
            "label_validity": ("model.label.validity", validity),
            "label_diagnostics": ("model.label.diagnostics", diagnostics),
        }
        payloads = {
            role: _parquet_bytes(frame, name=role)
            for role, (_, frame) in frames.items()
        }
        references = {
            role: self._publish_frame(
                logical_name=logical_name,
                role=role,
                frame=frame,
                payload=payloads[role],
            )
            for role, (logical_name, frame) in frames.items()
        }
        return LabelModelTrainingArtifactsV1(
            label_spec_hash=label_result.label_spec_hash,
            label_view_hash=label_result.label_view_hash,
            label_benchmark_hash=label_result.benchmark_hash,
            label_values_hash=label_result.labels_hash,
            label_windows_hash=label_result.windows_hash,
            label_validity_hash=label_result.validity_hash,
            label_diagnostics_hash=label_result.diagnostics_hash,
            label_values=references["label_values"],
            label_windows=references["label_windows"],
            label_validity=references["label_validity"],
            label_diagnostics=references["label_diagnostics"],
        )

    def publish_validation_inputs(
        self,
        *,
        validation_spec: ValidationSpec,
        validation_receipt: ValidationReceipt,
        trading_calendar: TradingCalendar,
        label_result: LabelResult,
    ) -> ValidationModelTrainingArtifactsV1:
        """Build the v1 resolver bundle after full receipt recomputation.

        The current resolver contract intentionally carries typed spec,
        receipt and calendar documents rather than an invented validation
        parquet artifact.
        """

        if not isinstance(label_result, LabelResult):
            raise _error("invalid_upstream_bundle", "label result type differs")
        try:
            ValidationReceiptVerifier().verify(
                validation_receipt,
                spec=validation_spec,
                labels=label_result,
                calendar=trading_calendar,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise _error(
                "validation_receipt_recomputation_mismatch",
                "validation receipt cannot be reproduced",
            ) from exc
        return ValidationModelTrainingArtifactsV1(
            validation_spec=validation_spec,
            validation_receipt=validation_receipt,
            trading_calendar=trading_calendar,
        )

    def _publish_frame(
        self,
        *,
        logical_name: str,
        role: str,
        frame: pd.DataFrame,
        payload: bytes,
    ) -> DataFrameArtifactReferenceV2:
        record = self._artifact_store.put_bytes(
            logical_name,
            payload,
            media_type=_PARQUET_MEDIA_TYPE,
            role=role,
        )
        return DataFrameArtifactReferenceV2.bind_frame(
            record=record,
            frame=frame,
            parquet_payload=payload,
        )


def _boolean_panel(value: object) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise _error("invalid_upstream_bundle", "boolean panel type differs")
    frame = value.copy(deep=True)
    if (
        frame.empty
        or frame.isna().any(axis=None)
        or not all(is_bool_dtype(dtype) for dtype in frame.dtypes)
    ):
        raise _error(
            "invalid_boolean_panel",
            "boolean panel must be complete, non-empty and boolean",
        )
    return frame


def _feature_panel(value: object, *, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise _error("invalid_upstream_bundle", f"feature:{name} type differs")
    frame = value.copy(deep=True)
    if frame.empty or any(
        is_bool_dtype(dtype) or is_complex_dtype(dtype) or not is_numeric_dtype(dtype)
        for dtype in frame.dtypes
    ):
        raise _error("invalid_upstream_bundle", f"feature:{name} must be numeric")
    return frame


def _require_same_axes(
    value: pd.DataFrame, expected: pd.DataFrame, *, name: str
) -> None:
    if not value.index.equals(expected.index) or not value.columns.equals(
        expected.columns
    ):
        raise _error("upstream_panel_mismatch", f"{name} axes differ")


def _parquet_bytes(frame: pd.DataFrame, *, name: str) -> bytes:
    buffer = io.BytesIO()
    try:
        frame.to_parquet(
            buffer,
            engine="pyarrow",
            compression="zstd",
            index=True,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise _error(
            "parquet_serialization_failed", f"{name} cannot be serialized"
        ) from exc
    payload = buffer.getvalue()
    if not payload:
        raise _error("parquet_serialization_failed", f"{name} payload is empty")
    return payload


__all__ = [
    "FactorModelTrainingArtifactsV1",
    "LabelModelTrainingArtifactsV1",
    "ModelTrainingUpstreamBundleProducer",
    "ValidationModelTrainingArtifactsV1",
]
