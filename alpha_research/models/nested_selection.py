from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from types import MappingProxyType
from typing import Mapping, cast

import numpy as np
import pandas as pd

from alpha_research.core.frequency import TradingCalendar
from alpha_research.core.hashing import hash_frame, hash_json
from alpha_research.labels import LabelResult
from alpha_research.models.artifact import ModelResultDescriptor
from alpha_research.models.nested_selection_manifest import (
    NestedSelectionPhaseOneManifest,
    nested_inner_selection_execution_context,
)
from alpha_research.models.nested_outer_evaluation_store import (
    CompletedOuterEvaluation,
    NestedOuterEvaluationStore,
    OuterEvaluationClaim,
    outer_evaluation_execution_context,
)
from alpha_research.models.model_execution_governance import (
    ModelExecutionContext,
    ModelFitAttemptLedger,
)
from alpha_research.models.runner import ModelResult, TimeSeriesModelRunner
from alpha_research.models.selection_scoring_evidence import (
    FoldPredictionRankEvidence,
    FoldScoringCohortEvidence,
    FoldScoringEvidenceVerifier,
    PredictionRankVector,
    ScoringCohortRankVector,
)
from alpha_research.models.selection_result import (
    CandidateScoreReceipt,
    FoldRankICReceipt,
    NestedSelectionResult,
    OuterFoldEvaluationResult,
    OuterFoldSelectionReceipt,
    OuterTrainingSliceReceipt,
    RankICObservation,
)
from alpha_research.models.selection_spec import (
    ModelCandidateTemplate,
    NestedPurgedSelectionSpec,
    OuterFoldSelectionPlan,
    RankICSelectionSpec,
)
from alpha_research.validation import (
    PurgedWalkForwardSplitter,
    SplitFoldReceipt,
    ValidationFoldSpec,
    ValidationReceipt,
    ValidationReceiptVerifier,
    ValidationSpec,
)


@dataclass(frozen=True, slots=True)
class _PreparedSource:
    features: Mapping[str, pd.DataFrame]
    feature_hashes: Mapping[str, str]
    scoring_eligibility: pd.DataFrame
    scoring_eligibility_hash: str


@dataclass(frozen=True, slots=True)
class _PreparedOuterFold:
    fold_spec: ValidationFoldSpec
    fold_receipt: SplitFoldReceipt
    plan: OuterFoldSelectionPlan
    features: Mapping[str, pd.DataFrame]
    scoring_eligibility: pd.DataFrame
    labels: LabelResult
    calendar: TradingCalendar
    slice_receipt: OuterTrainingSliceReceipt
    inner_receipt: ValidationReceipt


class NestedPurgedModelSelector:
    """Select models on physical outer-train slices, then evaluate once.

    The execution is deliberately split into two global phases.  Every outer
    fold's inner selection is completed and content-addressed before any outer
    validation model is fitted or scored.  This prevents an orchestrator from
    feeding an early outer result back into later candidate selection.
    """

    def __init__(self, runner: TimeSeriesModelRunner | None = None) -> None:
        self._runner = runner or TimeSeriesModelRunner()

    def run(
        self,
        spec: NestedPurgedSelectionSpec,
        *,
        feature_signals: Mapping[str, pd.DataFrame],
        labels: LabelResult,
        outer_validation_spec: ValidationSpec,
        outer_validation_receipt: ValidationReceipt,
        calendar: TradingCalendar,
        scoring_eligibility: pd.DataFrame | None = None,
    ) -> NestedSelectionResult:
        """Research convenience wrapper around the explicit two-phase API.

        The outer folds here are registered development-validation folds, not a
        final locked test or hidden institutional holdout.  A final protected
        partition must never be supplied to this in-process API; it belongs
        behind an institution-owned terminal evaluator.  Callers that need a
        strict orchestration barrier can persist ``select_inner_phase`` first
        and invoke ``evaluate_outer_phase`` only after every selection is
        sealed.
        """

        manifest = self.select_inner_phase(
            spec,
            feature_signals=feature_signals,
            labels=labels,
            outer_validation_spec=outer_validation_spec,
            outer_validation_receipt=outer_validation_receipt,
            calendar=calendar,
            scoring_eligibility=scoring_eligibility,
        )
        return self.evaluate_outer_phase(
            manifest,
            feature_signals=feature_signals,
            labels=labels,
            calendar=calendar,
            scoring_eligibility=scoring_eligibility,
        )

    def select_inner_phase(
        self,
        spec: NestedPurgedSelectionSpec,
        *,
        feature_signals: Mapping[str, pd.DataFrame],
        labels: LabelResult,
        outer_validation_spec: ValidationSpec,
        outer_validation_receipt: ValidationReceipt,
        calendar: TradingCalendar,
        scoring_eligibility: pd.DataFrame | None = None,
    ) -> NestedSelectionPhaseOneManifest:
        """Complete and seal every inner selection without outer evaluation."""

        if not isinstance(spec, NestedPurgedSelectionSpec):
            raise TypeError("nested selection specification type differs")
        if spec.outer_validation_spec_hash != outer_validation_spec.content_hash:
            raise ValueError("nested selection outer validation specification differs")
        if spec.required_fold_evaluations > spec.maximum_fold_evaluations:
            # Also checked by the immutable spec.  Keep the execution boundary
            # fail-closed in case a forged object bypasses normal construction.
            raise ValueError("nested selection fold evaluation budget is insufficient")
        labels.verify_content()
        source_labels = _detach_labels(labels)
        source = _prepare_source(
            spec,
            feature_signals,
            source_labels,
            scoring_eligibility=scoring_eligibility,
        )
        verified_outer = ValidationReceiptVerifier().verify(
            outer_validation_receipt,
            spec=outer_validation_spec,
            labels=source_labels,
            calendar=calendar,
        )
        aligned = _align_outer_plans(
            spec,
            outer_validation_spec=outer_validation_spec,
            outer_validation_receipt=verified_outer,
        )
        required_inner_evaluations = (
            spec.required_fold_evaluations - spec.maximum_outer_evaluations
        )
        inner_execution_context = nested_inner_selection_execution_context(
            selection_spec_hash=spec.content_hash,
            outer_validation_receipt_hash=verified_outer.content_hash,
            source_feature_hashes=source.feature_hashes,
            source_label_values_hash=source_labels.labels_hash,
            source_label_validity_hash=source_labels.validity_hash,
            source_scoring_eligibility_hash=source.scoring_eligibility_hash,
        )
        inner_execution_ledger = ModelFitAttemptLedger(
            context=inner_execution_context,
            maximum_attempts=required_inner_evaluations,
        )

        # Phase one: candidate selection receives only physical outer-train
        # slices.  No outer validation feature or label values are passed to it.
        selections: list[OuterFoldSelectionReceipt] = []
        for fold_spec, fold_receipt, plan in aligned:
            prepared = _prepare_outer_fold(
                outer_validation_spec_hash=outer_validation_spec.content_hash,
                fold_spec=fold_spec,
                fold_receipt=fold_receipt,
                plan=plan,
                source_features=source.features,
                source_scoring_eligibility=source.scoring_eligibility,
                labels=source_labels,
                calendar=calendar,
            )
            selection = self._select_outer_fold(
                spec,
                prepared,
                execution_ledger=inner_execution_ledger,
                execution_context=inner_execution_context,
            )
            selections.append(selection)

        sealed_selections = tuple(selections)
        if len(sealed_selections) != len(aligned):  # pragma: no cover
            raise RuntimeError("nested selection phase did not seal every outer fold")

        consumed_inner = sum(
            item.consumed_inner_evaluations for item in sealed_selections
        )
        inner_execution_snapshot = inner_execution_ledger.snapshot()
        return NestedSelectionPhaseOneManifest(
            selection_spec_hash=spec.content_hash,
            selection_spec=spec,
            outer_validation_spec_hash=outer_validation_spec.content_hash,
            outer_validation_spec=outer_validation_spec,
            outer_validation_receipt_hash=verified_outer.content_hash,
            outer_validation_receipt=verified_outer,
            source_feature_hashes=source.feature_hashes,
            source_label_values_hash=source_labels.labels_hash,
            source_label_validity_hash=source_labels.validity_hash,
            source_scoring_eligibility_hash=source.scoring_eligibility_hash,
            selections=sealed_selections,
            required_inner_fold_evaluations=required_inner_evaluations,
            consumed_inner_fold_evaluations=consumed_inner,
            inner_execution_snapshot_hash=inner_execution_snapshot.content_hash,
            inner_execution_snapshot=inner_execution_snapshot,
            planned_outer_fold_evaluations=spec.maximum_outer_evaluations,
        )


    def evaluate_outer_phase(
        self,
        manifest: NestedSelectionPhaseOneManifest,
        *,
        feature_signals: Mapping[str, pd.DataFrame],
        labels: LabelResult,
        calendar: TradingCalendar,
        scoring_eligibility: pd.DataFrame | None = None,
        execution_ledger: ModelFitAttemptLedger | None = None,
        execution_context: ModelExecutionContext | None = None,
    ) -> NestedSelectionResult:
        """Evaluate only selections recovered from an immutable phase-one seal."""

        if (execution_ledger is None) != (execution_context is None):
            raise ValueError(
                "nested outer execution ledger and context must be supplied together"
            )

        if not isinstance(manifest, NestedSelectionPhaseOneManifest):
            raise TypeError("nested selection phase-one manifest type differs")
        spec = manifest.selection_spec
        outer_validation_spec = manifest.outer_validation_spec
        source_labels = _detach_labels(labels)
        source_labels.verify_content()
        source = _prepare_source(
            spec,
            feature_signals,
            source_labels,
            scoring_eligibility=scoring_eligibility,
        )
        if (
            dict(source.feature_hashes) != dict(manifest.source_feature_hashes)
            or source_labels.labels_hash != manifest.source_label_values_hash
            or source_labels.validity_hash != manifest.source_label_validity_hash
            or source.scoring_eligibility_hash
            != manifest.source_scoring_eligibility_hash
        ):
            raise ValueError("nested selection phase-one source lineage differs")
        verified_outer = ValidationReceiptVerifier().verify(
            manifest.outer_validation_receipt,
            spec=outer_validation_spec,
            labels=source_labels,
            calendar=calendar,
        )
        if verified_outer.content_hash != manifest.outer_validation_receipt_hash:
            raise ValueError("nested selection phase-one outer receipt differs")
        aligned = _align_outer_plans(
            spec,
            outer_validation_spec=outer_validation_spec,
            outer_validation_receipt=verified_outer,
        )
        prepared_folds: list[_PreparedOuterFold] = []
        for fold_spec, fold_receipt, plan in aligned:
            prepared_folds.append(
                _prepare_outer_fold(
                    outer_validation_spec_hash=outer_validation_spec.content_hash,
                    fold_spec=fold_spec,
                    fold_receipt=fold_receipt,
                    plan=plan,
                    source_features=source.features,
                    source_scoring_eligibility=source.scoring_eligibility,
                    labels=source_labels,
                    calendar=calendar,
                )
            )
        for selection, prepared in zip(
            manifest.selections, prepared_folds, strict=True
        ):
            if (
                selection.outer_training_slice_hash
                != prepared.slice_receipt.content_hash
                or selection.inner_validation_receipt_hash
                != prepared.inner_receipt.content_hash
            ):
                raise ValueError(
                    "nested selection phase-one physical slice lineage differs"
                )

        # Phase two: only now may the selected candidate for each fold touch the
        # corresponding outer validation values.  Unselected candidates are
        # never evaluated on the outer fold.
        evaluations = tuple(
            self._evaluate_outer_fold(
                spec,
                selection=selection,
                prepared=prepared,
                source=source,
                labels=source_labels,
                outer_validation_spec=outer_validation_spec,
                calendar=calendar,
                execution_ledger=execution_ledger,
                execution_context=execution_context,
            )
            for selection, prepared in zip(
                manifest.selections, prepared_folds, strict=True
            )
        )
        return NestedSelectionResult(
            selection_spec_hash=spec.content_hash,
            selection_spec=spec,
            outer_validation_spec_hash=outer_validation_spec.content_hash,
            outer_validation_spec=outer_validation_spec,
            outer_validation_receipt_hash=verified_outer.content_hash,
            outer_validation_receipt=verified_outer,
            source_feature_hashes=source.feature_hashes,
            source_label_values_hash=source_labels.labels_hash,
            source_label_validity_hash=source_labels.validity_hash,
            source_scoring_eligibility_hash=source.scoring_eligibility_hash,
            selections=manifest.selections,
            outer_evaluations=evaluations,
            required_fold_evaluations=spec.required_fold_evaluations,
            consumed_fold_evaluations=spec.required_fold_evaluations,
        )

    def evaluate_outer_phase_managed(
        self,
        manifest: NestedSelectionPhaseOneManifest,
        *,
        store: NestedOuterEvaluationStore,
        evaluator_protocol_hash: str,
        runtime_fingerprint_hash: str,
        feature_signals: Mapping[str, pd.DataFrame],
        labels: LabelResult,
        calendar: TradingCalendar,
        scoring_eligibility: pd.DataFrame | None = None,
    ) -> NestedSelectionResult:
        """Run or load one trusted-local, single-consumption outer evaluation.

        The durable claim is acquired before this method inspects outer inputs.
        An interrupted claim therefore remains uncertain and is never retried
        automatically. This is research workflow governance, not an
        institution-owned hidden-OOS security boundary.
        """

        return self.evaluate_outer_phase_managed_record(
            manifest,
            store=store,
            evaluator_protocol_hash=evaluator_protocol_hash,
            runtime_fingerprint_hash=runtime_fingerprint_hash,
            feature_signals=feature_signals,
            labels=labels,
            calendar=calendar,
            scoring_eligibility=scoring_eligibility,
        ).result

    def evaluate_outer_phase_managed_record(
        self,
        manifest: NestedSelectionPhaseOneManifest,
        *,
        store: NestedOuterEvaluationStore,
        evaluator_protocol_hash: str,
        runtime_fingerprint_hash: str,
        feature_signals: Mapping[str, pd.DataFrame],
        labels: LabelResult,
        calendar: TradingCalendar,
        scoring_eligibility: pd.DataFrame | None = None,
    ) -> CompletedOuterEvaluation:
        """Run or load managed outer evaluation as an audited record.

        A trusted human/audit controller uses this record-returning method so
        its normal execution surface can expose only completion metadata.
        Agent-facing code must never receive this selector or store directly.
        """

        if not isinstance(store, NestedOuterEvaluationStore):
            raise TypeError("nested outer evaluation store type differs")
        access = store.claim_or_load(
            manifest,
            evaluator_protocol_hash=evaluator_protocol_hash,
            runtime_fingerprint_hash=runtime_fingerprint_hash,
        )
        if isinstance(access, CompletedOuterEvaluation):
            return access
        if not isinstance(access, OuterEvaluationClaim):  # pragma: no cover
            raise RuntimeError("nested outer evaluation access type differs")
        execution_context = outer_evaluation_execution_context(access)
        execution_ledger = ModelFitAttemptLedger(
            context=execution_context,
            maximum_attempts=manifest.planned_outer_fold_evaluations,
        )
        result = self.evaluate_outer_phase(
            manifest,
            feature_signals=feature_signals,
            labels=labels,
            calendar=calendar,
            scoring_eligibility=scoring_eligibility,
            execution_ledger=execution_ledger,
            execution_context=execution_context,
        )
        execution_snapshot = execution_ledger.snapshot()
        return store.publish_completed(
            access,
            manifest=manifest,
            result=result,
            execution_snapshot=execution_snapshot,
        )

    def _select_outer_fold(
        self,
        spec: NestedPurgedSelectionSpec,
        prepared: _PreparedOuterFold,
        *,
        execution_ledger: ModelFitAttemptLedger,
        execution_context: ModelExecutionContext,
    ) -> OuterFoldSelectionReceipt:
        candidate_scores: list[CandidateScoreReceipt] = []
        for candidate in spec.candidates:
            candidate_features = {
                name: prepared.features[name].copy(deep=True)
                for name in candidate.feature_names
            }
            candidate_labels = _detach_labels(prepared.labels)
            model_spec = candidate.materialize(
                selection_id=spec.selection_id,
                selection_version=spec.version,
                outer_fold_id=prepared.fold_spec.fold_id,
                stage="inner_selection",
                feature_signal_hashes={
                    name: prepared.slice_receipt.sliced_feature_hashes[name]
                    for name in candidate.feature_names
                },
                label_spec_hash=prepared.labels.label_spec_hash,
                validation_spec_hash=prepared.plan.inner_validation_spec.content_hash,
            )
            result = self._runner.fit_predict_verified(
                model_spec,
                candidate_features,
                candidate_labels,
                prepared.inner_receipt,
                validation_spec=prepared.plan.inner_validation_spec,
                calendar=prepared.calendar,
                preprocessing=candidate.preprocessing,
                execution_ledger=execution_ledger,
                execution_context=execution_context,
            )
            _verify_runner_inputs(
                candidate_features,
                expected_hashes={
                    name: prepared.slice_receipt.sliced_feature_hashes[name]
                    for name in candidate.feature_names
                },
                labels=candidate_labels,
            )
            _verify_result_lineage(
                result,
                model_spec_hash=model_spec.content_hash,
                validation_receipt_hash=prepared.inner_receipt.content_hash,
                label_values_hash=prepared.labels.labels_hash,
                preprocess_spec_hash=candidate.preprocessing.content_hash,
            )
            fold_scores = _score_model_result(
                result,
                labels=prepared.labels,
                receipt=prepared.inner_receipt,
                scoring=spec.scoring,
                scoring_eligibility=prepared.scoring_eligibility,
            )
            candidate_scores.append(
                CandidateScoreReceipt(
                    outer_fold_id=prepared.fold_spec.fold_id,
                    candidate_id=candidate.candidate_id,
                    candidate_template_hash=candidate.content_hash,
                    materialized_model_spec_hash=model_spec.content_hash,
                    outer_training_slice_hash=prepared.slice_receipt.content_hash,
                    inner_validation_spec_hash=(
                        prepared.plan.inner_validation_spec.content_hash
                    ),
                    inner_validation_receipt_hash=prepared.inner_receipt.content_hash,
                    inner_label_values_hash=result.label_values_hash,
                    inner_model_result=_describe_model_result(result),
                    inner_model_result_hash=result.content_hash,
                    preprocess_spec_hash=candidate.preprocessing.content_hash,
                    fold_scores=fold_scores,
                    selection_score=_mean(
                        tuple(item.rank_ic_mean for item in fold_scores)
                    ),
                    complexity_rank=candidate.complexity_rank,
                    consumed_fold_evaluations=len(fold_scores),
                )
            )
        selected = min(candidate_scores, key=lambda item: item.selection_key)
        return OuterFoldSelectionReceipt(
            selection_spec_hash=spec.content_hash,
            selection_spec=spec,
            outer_fold_id=prepared.fold_spec.fold_id,
            outer_training_slice_hash=prepared.slice_receipt.content_hash,
            outer_training_slice=prepared.slice_receipt,
            inner_validation_receipt_hash=prepared.inner_receipt.content_hash,
            inner_validation_receipt=prepared.inner_receipt,
            candidate_scores=tuple(candidate_scores),
            selected_candidate_id=selected.candidate_id,
            selected_candidate_template_hash=selected.candidate_template_hash,
            selected_score=selected.selection_score,
        )

    def _evaluate_outer_fold(
        self,
        spec: NestedPurgedSelectionSpec,
        *,
        selection: OuterFoldSelectionReceipt,
        prepared: _PreparedOuterFold,
        source: _PreparedSource,
        labels: LabelResult,
        outer_validation_spec: ValidationSpec,
        calendar: TradingCalendar,
        execution_ledger: ModelFitAttemptLedger | None,
        execution_context: ModelExecutionContext | None,
    ) -> OuterFoldEvaluationResult:
        selected = _candidate_by_id(spec, selection.selected_candidate_id)
        projected_spec = _project_single_outer_spec(
            selection_spec=spec,
            outer_validation_spec=outer_validation_spec,
            fold=prepared.fold_spec,
        )
        projected_receipt = PurgedWalkForwardSplitter().split(
            projected_spec, labels, calendar
        )
        if (
            len(projected_receipt.folds) != 1
            or projected_receipt.folds[0].content_hash
            != prepared.fold_receipt.content_hash
        ):
            raise ValueError("projected outer validation membership differs")
        candidate_features = {
            name: source.features[name].copy(deep=True)
            for name in selected.feature_names
        }
        evaluation_labels = _detach_labels(labels)
        model_spec = selected.materialize(
            selection_id=spec.selection_id,
            selection_version=spec.version,
            outer_fold_id=prepared.fold_spec.fold_id,
            stage="outer_evaluation",
            feature_signal_hashes={
                name: source.feature_hashes[name] for name in selected.feature_names
            },
            label_spec_hash=labels.label_spec_hash,
            validation_spec_hash=projected_spec.content_hash,
        )
        result, artifact = self._runner.fit_predict_verified_with_artifact(
            model_spec,
            candidate_features,
            evaluation_labels,
            projected_receipt,
            validation_spec=projected_spec,
            calendar=calendar,
            preprocessing=selected.preprocessing,
            execution_ledger=execution_ledger,
            execution_context=execution_context,
        )
        _verify_runner_inputs(
            candidate_features,
            expected_hashes={
                name: source.feature_hashes[name] for name in selected.feature_names
            },
            labels=evaluation_labels,
        )
        _verify_result_lineage(
            result,
            model_spec_hash=model_spec.content_hash,
            validation_receipt_hash=projected_receipt.content_hash,
            label_values_hash=labels.labels_hash,
            preprocess_spec_hash=selected.preprocessing.content_hash,
        )
        outer_scores = _score_model_result(
            result,
            labels=labels,
            receipt=projected_receipt,
            scoring=spec.scoring,
            scoring_eligibility=source.scoring_eligibility,
        )
        if len(outer_scores) != 1:  # pragma: no cover - projected spec is one fold.
            raise RuntimeError("projected outer evaluation produced multiple folds")
        return OuterFoldEvaluationResult(
            selection_receipt_hash=selection.content_hash,
            outer_fold_id=prepared.fold_spec.fold_id,
            selected_candidate_id=selected.candidate_id,
            projected_outer_validation_spec_hash=projected_spec.content_hash,
            projected_outer_validation_receipt_hash=projected_receipt.content_hash,
            materialized_model_spec_hash=model_spec.content_hash,
            model_result_hash=result.content_hash,
            model_artifact=artifact,
            model_artifact_hash=artifact.content_hash,
            outer_score=outer_scores[0],
        )


def verify_nested_selection_phase_one_manifest_inputs(
    manifest: NestedSelectionPhaseOneManifest,
    *,
    spec: NestedPurgedSelectionSpec,
    feature_signals: Mapping[str, pd.DataFrame],
    labels: LabelResult,
    outer_validation_spec: ValidationSpec,
    outer_validation_receipt: ValidationReceipt,
    calendar: TradingCalendar,
    scoring_eligibility: pd.DataFrame | None = None,
) -> None:
    """Recompute every phase-one source commitment without fitting a model."""

    if not isinstance(manifest, NestedSelectionPhaseOneManifest):
        raise TypeError("nested selection phase-one manifest type differs")
    if not isinstance(spec, NestedPurgedSelectionSpec):
        raise TypeError("nested selection specification type differs")
    if (
        manifest.selection_spec_hash != spec.content_hash
        or manifest.selection_spec != spec
    ):
        raise ValueError("nested selection phase-one specification differs")
    if (
        manifest.outer_validation_spec_hash != outer_validation_spec.content_hash
        or manifest.outer_validation_spec != outer_validation_spec
        or manifest.outer_validation_receipt_hash
        != outer_validation_receipt.content_hash
        or manifest.outer_validation_receipt != outer_validation_receipt
    ):
        raise ValueError("nested selection phase-one validation lineage differs")
    labels.verify_content()
    source_labels = _detach_labels(labels)
    source = _prepare_source(
        spec,
        feature_signals,
        source_labels,
        scoring_eligibility=scoring_eligibility,
    )
    if (
        dict(source.feature_hashes) != dict(manifest.source_feature_hashes)
        or source_labels.labels_hash != manifest.source_label_values_hash
        or source_labels.validity_hash != manifest.source_label_validity_hash
        or source.scoring_eligibility_hash
        != manifest.source_scoring_eligibility_hash
    ):
        raise ValueError("nested selection phase-one source lineage differs")
    verified_outer = ValidationReceiptVerifier().verify(
        outer_validation_receipt,
        spec=outer_validation_spec,
        labels=source_labels,
        calendar=calendar,
    )
    if verified_outer.content_hash != manifest.outer_validation_receipt_hash:
        raise ValueError("nested selection phase-one outer receipt differs")
    _align_outer_plans(
        spec,
        outer_validation_spec=outer_validation_spec,
        outer_validation_receipt=verified_outer,
    )


def _prepare_source(
    spec: NestedPurgedSelectionSpec,
    feature_signals: Mapping[str, pd.DataFrame],
    labels: LabelResult,
    *,
    scoring_eligibility: pd.DataFrame | None,
) -> _PreparedSource:
    required_names = tuple(
        sorted({name for item in spec.candidates for name in item.feature_names})
    )
    if set(feature_signals) != set(required_names):
        raise ValueError("nested selection feature names differ from candidate space")
    features = {
        name: pd.DataFrame(feature_signals[name]).copy(deep=True)
        for name in required_names
    }
    reference = next(iter(features.values()))
    reference_index = pd.DatetimeIndex(reference.index)
    if (
        reference_index.tz is None
        or not reference_index.is_unique
        or not reference_index.is_monotonic_increasing
    ):
        raise ValueError(
            "nested selection feature index must be aware, sorted and unique"
        )
    for name, frame in features.items():
        if not frame.index.equals(reference.index) or not frame.columns.equals(
            reference.columns
        ):
            raise ValueError(f"nested selection feature axes differ:{name}")
        if not frame.columns.is_unique:
            raise ValueError(f"nested selection feature columns are not unique:{name}")
        try:
            features[name] = frame.apply(pd.to_numeric, errors="raise").astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"nested selection feature values are not numeric:{name}"
            ) from error
    label_index = pd.DatetimeIndex(labels.labels.index)
    if (
        not labels.labels.index.equals(labels.validity.index)
        or not labels.labels.columns.equals(labels.validity.columns)
        or not labels.labels.index.equals(reference.index)
        or not labels.labels.columns.equals(reference.columns)
        or not labels.labels.columns.is_unique
        or not label_index.is_unique
    ):
        raise ValueError("nested selection feature/label/validity axes differ")
    if len(reference.columns) < 2:
        raise ValueError("nested selection requires at least two securities")
    # The prediction/scoring cohort is fixed before candidate generation and
    # never inferred from the union of candidate feature sets.  Otherwise an
    # unselected candidate could change every other candidate's sample merely
    # by carrying a different missing-value pattern.  Labels still participate
    # only after prediction when targets are paired for scoring.
    if scoring_eligibility is None:
        eligibility = pd.DataFrame(
            True,
            index=reference.index,
            columns=reference.columns,
        )
    else:
        eligibility = pd.DataFrame(scoring_eligibility).copy(deep=True)
        if not eligibility.index.equals(
            reference.index
        ) or not eligibility.columns.equals(reference.columns):
            raise ValueError("nested selection scoring eligibility axes differ")
        if eligibility.isna().to_numpy().any():
            raise ValueError(
                "nested selection scoring eligibility contains missing values"
            )
        eligibility_values = eligibility.to_numpy(dtype=object)
        if not all(
            isinstance(value, (bool, np.bool_)) for value in eligibility_values.ravel()
        ):
            raise ValueError("nested selection scoring eligibility is not boolean")
        eligibility = eligibility.astype(bool)
    eligibility_mask = eligibility.to_numpy(dtype=bool)
    for candidate in spec.candidates:
        for name in candidate.feature_names:
            values = features[name].to_numpy(dtype=float)
            if not np.isfinite(values[eligibility_mask]).all():
                raise ValueError(
                    f"candidate_feature_coverage_incomplete:{candidate.candidate_id}:{name}"
                )
    features = {
        name: frame.where(eligibility).copy(deep=True)
        for name, frame in features.items()
    }
    hashes = {name: hash_frame(frame) for name, frame in features.items()}
    return _PreparedSource(
        features=MappingProxyType(features),
        feature_hashes=MappingProxyType(hashes),
        scoring_eligibility=eligibility.copy(deep=True),
        scoring_eligibility_hash=hash_frame(eligibility),
    )


def _detach_labels(labels: LabelResult) -> LabelResult:
    """Take one verified in-memory snapshot before nested execution begins."""

    return LabelResult(
        label_spec_hash=labels.label_spec_hash,
        label_view_hash=labels.label_view_hash,
        benchmark_hash=labels.benchmark_hash,
        labels_hash=labels.labels_hash,
        windows_hash=labels.windows_hash,
        validity_hash=labels.validity_hash,
        diagnostics_hash=labels.diagnostics_hash,
        labels=labels.labels,
        label_windows=labels.label_windows,
        validity=labels.validity,
        diagnostics=labels.diagnostics,
    )


def _align_outer_plans(
    spec: NestedPurgedSelectionSpec,
    *,
    outer_validation_spec: ValidationSpec,
    outer_validation_receipt: ValidationReceipt,
) -> tuple[tuple[ValidationFoldSpec, SplitFoldReceipt, OuterFoldSelectionPlan], ...]:
    spec_ids = tuple(item.fold_id for item in outer_validation_spec.folds)
    receipt_ids = tuple(item.fold_id for item in outer_validation_receipt.folds)
    plan_ids = tuple(item.outer_fold_id for item in spec.outer_plans)
    if plan_ids != spec_ids or receipt_ids != spec_ids:
        raise ValueError("nested selection outer fold plans differ")
    return tuple(
        zip(
            outer_validation_spec.folds,
            outer_validation_receipt.folds,
            spec.outer_plans,
            strict=True,
        )
    )


def _prepare_outer_fold(
    *,
    outer_validation_spec_hash: str,
    fold_spec: ValidationFoldSpec,
    fold_receipt: SplitFoldReceipt,
    plan: OuterFoldSelectionPlan,
    source_features: Mapping[str, pd.DataFrame],
    source_scoring_eligibility: pd.DataFrame,
    labels: LabelResult,
    calendar: TradingCalendar,
) -> _PreparedOuterFold:
    _validate_inner_boundaries(plan.inner_validation_spec, fold_spec)
    reference_index = pd.DatetimeIndex(next(iter(source_features.values())).index)
    train_index = _receipt_index(
        fold_receipt.train_signals,
        reference=reference_index,
        name=f"outer_train:{fold_spec.fold_id}",
    )
    sliced_features = {
        name: frame.loc[train_index].copy(deep=True)
        for name, frame in source_features.items()
    }
    sliced_labels = _slice_labels(labels, train_index)
    sliced_eligibility = source_scoring_eligibility.loc[train_index].copy(deep=True)
    sliced_calendar = _slice_calendar(calendar, fold_spec)
    sliced_hashes = {name: hash_frame(frame) for name, frame in sliced_features.items()}
    slice_receipt = OuterTrainingSliceReceipt(
        outer_validation_spec_hash=outer_validation_spec_hash,
        outer_fold_receipt=fold_receipt,
        outer_fold_receipt_hash=fold_receipt.content_hash,
        train_signals=fold_receipt.train_signals,
        sliced_feature_hashes=sliced_hashes,
        sliced_label_values_hash=sliced_labels.labels_hash,
        sliced_label_windows_hash=sliced_labels.windows_hash,
        sliced_label_validity_hash=sliced_labels.validity_hash,
        sliced_label_diagnostics_hash=sliced_labels.diagnostics_hash,
        scoring_eligibility_hash=hash_frame(sliced_eligibility),
        sliced_calendar_hash=_validation_calendar_hash(sliced_calendar),
    )
    forbidden = set(fold_receipt.validation_signals)
    forbidden.update(fold_receipt.purged_train_signals)
    forbidden.update(fold_receipt.excluded_validation_signals)
    _assert_sliced_axes(
        train_index,
        sliced_features=sliced_features,
        sliced_labels=sliced_labels,
        scoring_eligibility=sliced_eligibility,
        forbidden=forbidden,
    )
    inner_receipt = PurgedWalkForwardSplitter().split(
        plan.inner_validation_spec,
        sliced_labels,
        sliced_calendar,
    )
    if inner_receipt.calendar_hash != slice_receipt.sliced_calendar_hash:
        raise ValueError("inner validation calendar lineage differs")
    _assert_inner_membership(
        inner_receipt,
        outer_train=set(fold_receipt.train_signals),
        outer_forbidden=forbidden,
    )
    return _PreparedOuterFold(
        fold_spec=fold_spec,
        fold_receipt=fold_receipt,
        plan=plan,
        features=MappingProxyType(sliced_features),
        scoring_eligibility=sliced_eligibility,
        labels=sliced_labels,
        calendar=sliced_calendar,
        slice_receipt=slice_receipt,
        inner_receipt=inner_receipt,
    )


def _slice_labels(labels: LabelResult, train_index: pd.DatetimeIndex) -> LabelResult:
    label_index = pd.DatetimeIndex(labels.labels.index)
    if len(train_index.difference(label_index)):
        raise ValueError("outer train signals are unavailable in label values")
    values = labels.labels.loc[train_index].copy(deep=True)
    validity = labels.validity.loc[train_index].copy(deep=True)
    windows = _slice_signal_frame(
        labels.label_windows, train_index, name="label_windows"
    )
    diagnostics = _slice_signal_frame(
        labels.diagnostics, train_index, name="label_diagnostics"
    )
    return LabelResult(
        label_spec_hash=labels.label_spec_hash,
        label_view_hash=labels.label_view_hash,
        benchmark_hash=labels.benchmark_hash,
        labels_hash=hash_frame(values),
        windows_hash=hash_frame(windows),
        validity_hash=hash_frame(validity),
        diagnostics_hash=hash_frame(diagnostics),
        labels=values,
        label_windows=windows,
        validity=validity,
        diagnostics=diagnostics,
    )


def _slice_signal_frame(
    frame: pd.DataFrame,
    train_index: pd.DatetimeIndex,
    *,
    name: str,
) -> pd.DataFrame:
    if "signal_timestamp" not in frame.columns:
        raise ValueError(f"{name} is missing signal_timestamp")
    working = pd.DataFrame(frame).copy(deep=True)
    signals = pd.to_datetime(working["signal_timestamp"], errors="raise")
    if not isinstance(signals, pd.Series) or signals.dt.tz is None:
        raise ValueError(f"{name} signal timestamps must be timezone-aware")
    if signals.duplicated().any():
        raise ValueError(f"{name} signal timestamps must be unique")
    working["signal_timestamp"] = signals
    indexed = working.set_index("signal_timestamp", drop=False)
    if len(train_index.difference(pd.DatetimeIndex(indexed.index))):
        raise ValueError(f"outer train signals are unavailable in {name}")
    selected = indexed.loc[train_index].copy(deep=True).reset_index(drop=True)
    return selected.reindex(columns=working.columns)


def _slice_calendar(
    calendar: TradingCalendar, fold_spec: ValidationFoldSpec
) -> TradingCalendar:
    start = pd.Timestamp(fold_spec.train_start).tz_convert(calendar.timezone)
    end = pd.Timestamp(fold_spec.train_end).tz_convert(calendar.timezone)
    start_day = start.strftime("%Y%m%d")
    end_day = end.strftime("%Y%m%d")
    sessions = tuple(item for item in calendar.sessions if start_day <= item <= end_day)
    if not sessions or start_day not in sessions or end_day not in sessions:
        raise ValueError("outer train boundary is outside calendar")
    return TradingCalendar(
        calendar_id=calendar.calendar_id,
        timezone=calendar.timezone,
        sessions=sessions,
        session=calendar.session,
    )


def _validation_calendar_hash(calendar: TradingCalendar) -> str:
    return cast(
        str,
        hash_json(
            {
                "calendar_id": calendar.calendar_id,
                "timezone": calendar.timezone,
                "sessions": list(calendar.sessions),
                "session_id": calendar.session.session_id,
                "intervals": [list(item) for item in calendar.session.intervals],
            }
        ),
    )


def _receipt_index(
    signals: tuple[str, ...],
    *,
    reference: pd.DatetimeIndex,
    name: str,
) -> pd.DatetimeIndex:
    if reference.tz is None:
        raise ValueError("reference feature index must be timezone-aware")
    values = pd.DatetimeIndex(tuple(pd.Timestamp(item) for item in signals))
    if values.tz is None:
        raise ValueError(f"{name} signals must be timezone-aware")
    values = values.tz_convert(reference.tz)
    if not values.is_unique or not values.is_monotonic_increasing:
        raise ValueError(f"{name} signals must be sorted and unique")
    if len(values.difference(reference)):
        raise ValueError(f"{name} signals are unavailable")
    return values


def _assert_sliced_axes(
    train_index: pd.DatetimeIndex,
    *,
    sliced_features: Mapping[str, pd.DataFrame],
    sliced_labels: LabelResult,
    scoring_eligibility: pd.DataFrame,
    forbidden: set[str],
) -> None:
    expected = tuple(item.isoformat() for item in train_index)
    for name, frame in sliced_features.items():
        if not frame.index.equals(train_index):
            raise ValueError(f"outer training feature slice order differs:{name}")
    if not sliced_labels.labels.index.equals(train_index):
        raise ValueError("outer training label slice order differs")
    if not scoring_eligibility.index.equals(
        train_index
    ) or not scoring_eligibility.columns.equals(sliced_labels.labels.columns):
        raise ValueError("outer training scoring eligibility axes differ")
    if forbidden.intersection(expected):
        raise ValueError("outer training slice contains forbidden signals")


def _validate_inner_boundaries(
    inner: ValidationSpec, outer_fold: ValidationFoldSpec
) -> None:
    outer_start = pd.Timestamp(outer_fold.train_start)
    outer_end = pd.Timestamp(outer_fold.train_end)
    for fold in inner.folds:
        boundaries = (
            pd.Timestamp(fold.train_start),
            pd.Timestamp(fold.train_end),
            pd.Timestamp(fold.validation_start),
            pd.Timestamp(fold.validation_end),
        )
        if any(value < outer_start or value > outer_end for value in boundaries):
            raise ValueError(
                f"inner validation falls outside outer train:{outer_fold.fold_id}:"
                f"{fold.fold_id}"
            )


def _assert_inner_membership(
    receipt: ValidationReceipt,
    *,
    outer_train: set[str],
    outer_forbidden: set[str],
) -> None:
    for fold in receipt.folds:
        members = set(fold.train_signals)
        members.update(fold.validation_signals)
        members.update(fold.purged_train_signals)
        members.update(fold.excluded_validation_signals)
        if not members.issubset(outer_train):
            raise ValueError(f"inner receipt escapes outer train:{fold.fold_id}")
        if members.intersection(outer_forbidden):
            raise ValueError(
                f"inner receipt contains outer forbidden signals:{fold.fold_id}"
            )


def _score_model_result(
    result: ModelResult,
    *,
    labels: LabelResult,
    receipt: ValidationReceipt,
    scoring: RankICSelectionSpec,
    scoring_eligibility: pd.DataFrame,
) -> tuple[FoldRankICReceipt, ...]:
    result.verify_content()
    label_index = pd.DatetimeIndex(labels.labels.index)
    if not scoring_eligibility.index.equals(
        labels.labels.index
    ) or not scoring_eligibility.columns.equals(labels.labels.columns):
        raise ValueError("RankIC scoring eligibility axes differ")
    eligibility_hash = hash_frame(scoring_eligibility)
    output: list[FoldRankICReceipt] = []
    for fold in receipt.folds:
        timestamps = _receipt_index(
            fold.validation_signals,
            reference=label_index,
            name=f"RankIC:{fold.fold_id}",
        )
        cohort_vectors: list[ScoringCohortRankVector] = []
        prediction_vectors: list[PredictionRankVector] = []
        common = labels.labels.columns
        for timestamp in timestamps:
            prediction_membership = scoring_eligibility.loc[timestamp, common].to_numpy(
                dtype=bool
            )
            if timestamp not in result.predictions.index:
                raise ValueError(f"RankIC prediction row is missing:{fold.fold_id}")
            prediction = pd.to_numeric(
                result.predictions.loc[timestamp, common], errors="coerce"
            ).to_numpy(dtype=float)
            if not np.isfinite(prediction[prediction_membership]).all():
                raise ValueError(
                    f"RankIC prediction cohort is incomplete:{fold.fold_id}"
                )
            target = pd.to_numeric(
                labels.labels.loc[timestamp, common], errors="coerce"
            ).to_numpy(dtype=float)
            label_membership = labels.validity.loc[timestamp, common].to_numpy(
                dtype=bool
            ) & np.isfinite(target)
            membership = prediction_membership & label_membership
            count = int(membership.sum())
            if count < scoring.minimum_cross_sectional_observations:
                continue
            if not np.isfinite(target[membership]).all():
                raise ValueError(
                    f"RankIC eligible values are non-finite:{fold.fold_id}"
                )
            member_positions = np.flatnonzero(membership)
            member_ids = np.asarray(
                [str(common[position]) for position in member_positions],
                dtype=object,
            )
            order = np.argsort(member_ids, kind="stable")
            sorted_positions = member_positions[order]
            security_members = tuple(
                str(common[position]) for position in sorted_positions
            )
            if len(security_members) != len(set(security_members)):
                raise ValueError(
                    f"RankIC security identifiers are not unique:{fold.fold_id}"
                )
            target_ranks = tuple(
                float(item)
                for item in pd.Series(target[sorted_positions]).rank(method="average")
            )
            prediction_ranks = tuple(
                float(item)
                for item in pd.Series(prediction[sorted_positions]).rank(
                    method="average"
                )
            )
            cohort_vectors.append(
                ScoringCohortRankVector(
                    signal_timestamp=timestamp.isoformat(),
                    security_members=security_members,
                    target_ranks=target_ranks,
                )
            )
            prediction_vectors.append(
                PredictionRankVector(
                    signal_timestamp=timestamp.isoformat(),
                    security_members=security_members,
                    prediction_ranks=prediction_ranks,
                )
            )
        if len(cohort_vectors) < scoring.minimum_valid_dates_per_inner_fold:
            raise ValueError(
                f"insufficient_rank_ic_dates:{fold.fold_id}:{len(cohort_vectors)}"
            )
        cohort = FoldScoringCohortEvidence(
            fold_id=fold.fold_id,
            validation_receipt_hash=receipt.content_hash,
            scoring_spec_hash=scoring.content_hash,
            label_values_hash=labels.labels_hash,
            label_validity_hash=labels.validity_hash,
            eligibility_hash=eligibility_hash,
            rank_vectors=tuple(cohort_vectors),
        )
        prediction_evidence = FoldPredictionRankEvidence(
            fold_id=fold.fold_id,
            cohort_evidence_hash=cohort.content_hash,
            model_result_hash=result.content_hash,
            predictions_hash=result.predictions_hash,
            rank_vectors=tuple(prediction_vectors),
        )
        verified = FoldScoringEvidenceVerifier().verify(
            cohort,
            prediction_evidence,
            expected_fold_id=fold.fold_id,
            expected_validation_receipt_hash=receipt.content_hash,
            expected_scoring_spec_hash=scoring.content_hash,
            expected_label_values_hash=labels.labels_hash,
            expected_label_validity_hash=labels.validity_hash,
            expected_eligibility_hash=eligibility_hash,
            expected_model_result_hash=result.content_hash,
            expected_predictions_hash=result.predictions_hash,
            expected_cohort_evidence_hash=cohort.content_hash,
            expected_prediction_evidence_hash=prediction_evidence.content_hash,
        )
        observations = tuple(
            RankICObservation(
                signal_timestamp=item.signal_timestamp,
                observation_count=item.observation_count,
                security_membership_hash=item.security_membership_hash,
                rank_ic=item.rank_ic,
            )
            for item in verified.daily_scores
        )
        output.append(
            FoldRankICReceipt(
                fold_id=fold.fold_id,
                scoring_cohort_evidence=cohort,
                scoring_cohort_evidence_hash=cohort.content_hash,
                prediction_rank_evidence=prediction_evidence,
                prediction_rank_evidence_hash=prediction_evidence.content_hash,
                verified_scoring_evidence=verified,
                verified_scoring_evidence_hash=verified.content_hash,
                observations=observations,
                observations_hash=hash_json([item.to_dict() for item in observations]),
                rank_ic_mean=verified.fold_rank_ic_mean,
            )
        )
    return tuple(output)


def _verify_runner_inputs(
    features: Mapping[str, pd.DataFrame],
    *,
    expected_hashes: Mapping[str, str],
    labels: LabelResult,
) -> None:
    if set(features) != set(expected_hashes) or any(
        hash_frame(frame) != expected_hashes[name] for name, frame in features.items()
    ):
        raise RuntimeError("model runner mutated nested selection feature inputs")
    try:
        labels.verify_content()
    except RuntimeError as error:
        raise RuntimeError(
            "model runner mutated nested selection label inputs"
        ) from error


def _verify_result_lineage(
    result: ModelResult,
    *,
    model_spec_hash: str,
    validation_receipt_hash: str,
    label_values_hash: str,
    preprocess_spec_hash: str,
) -> None:
    if (
        result.model_spec_hash != model_spec_hash
        or result.validation_receipt_hash != validation_receipt_hash
        or result.label_values_hash != label_values_hash
        or result.preprocess_spec_hash != preprocess_spec_hash
    ):
        raise ValueError("nested selection model result lineage differs")


def _describe_model_result(result: ModelResult) -> ModelResultDescriptor:
    """Freeze the complete portable lineage descriptor for an inner result."""

    result.verify_content()
    return ModelResultDescriptor(
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


def _project_single_outer_spec(
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


def _candidate_by_id(
    spec: NestedPurgedSelectionSpec, candidate_id: str
) -> ModelCandidateTemplate:
    matching = tuple(
        item for item in spec.candidates if item.candidate_id == candidate_id
    )
    if len(matching) != 1:  # pragma: no cover - immutable spec guarantees uniqueness.
        raise RuntimeError("selected candidate identity is unavailable")
    return matching[0]


def _mean(values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError("mean requires values")
    return fsum(values) / len(values)


__all__ = ["NestedPurgedModelSelector"]
