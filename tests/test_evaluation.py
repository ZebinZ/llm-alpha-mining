from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from llm_alpha_mining.research.core.frequency import FrequencySpec
from llm_alpha_mining.research.core.hashing import hash_frame
from llm_alpha_mining.research.evaluation import EvaluationSpec, FactorEvaluationSuite
from llm_alpha_mining.research.factors import (
    STRICT_FACTOR_ENGINE_EXECUTION_POLICY_HASH,
    AggregationSpec,
    ComplexityBudget,
    FactorProvenance,
    FactorResult,
    FactorSpec,
)
from llm_alpha_mining.research.factors.complexity import FactorComplexity
from llm_alpha_mining.research.labels import (
    EntryEvent,
    LabelResult,
    LabelSpec,
    LabelTask,
    PriceObservationEvent,
)
from llm_alpha_mining.research.validation import SplitFoldReceipt, ValidationReceipt
from llm_alpha_mining.mining.dsl import OperatorRegistry


def _fixture():
    timestamps = pd.date_range(
        "2020-07-01 15:00:00",
        periods=8,
        freq="B",
        tz="Asia/Shanghai",
        name="signal_timestamp",
    )
    securities = pd.Index([f"{number:06d}" for number in range(1, 11)])
    base = np.arange(1, 11, dtype=float)
    signal = pd.DataFrame(
        [base + offset * 0.01 for offset in range(len(timestamps))],
        index=timestamps,
        columns=securities,
    )
    label_values = signal * 0.02
    validity = pd.DataFrame(True, index=timestamps, columns=securities)
    windows = pd.DataFrame(
        {
            "signal_timestamp": timestamps,
            "label_start": timestamps + pd.Timedelta(days=1),
            "label_end": timestamps + pd.Timedelta(days=2),
        }
    )
    diagnostics = pd.DataFrame(
        {
            "signal_timestamp": timestamps,
            "security_count": 10,
            "valid_label_count": 10,
            "invalid_label_count": 0,
        }
    )
    frequency = FrequencySpec.daily()
    label_spec = LabelSpec(
        label_id="evaluation-label",
        version="1",
        task=LabelTask.RETURN,
        horizon_sessions=1,
        information_cutoff_event="signal_session_close",
        entry_event=EntryEvent.NEXT_SESSION_OPEN,
        entry_lag_sessions=1,
        price_field="Open",
        price_observation_event=PriceObservationEvent.SESSION_OPEN,
        adjustment_field=None,
        amount_field="Amount",
        require_execution_tradability=True,
        benchmark_id=None,
        benchmark_snapshot_id=None,
        direction_threshold=0.0,
        annualize_volatility=False,
        missing_policy="invalidate_label",
        dataset_id="evaluation-data",
        snapshot_id="a" * 64,
        schema_hash="b" * 64,
        availability_hash="c" * 64,
        security_contract_hash="d" * 64,
        frequency=frequency,
    )
    labels = LabelResult(
        label_spec_hash=label_spec.content_hash,
        label_view_hash="d" * 64,
        benchmark_hash=None,
        labels_hash=hash_frame(label_values),
        windows_hash=hash_frame(windows),
        validity_hash=hash_frame(validity),
        diagnostics_hash=hash_frame(diagnostics),
        labels=label_values,
        label_windows=windows,
        validity=validity,
        diagnostics=diagnostics,
    )
    mask = pd.DataFrame(True, index=timestamps, columns=securities)
    registry = OperatorRegistry.dataframe_pit_v3(mask, mask)
    factor_spec = FactorSpec(
        factor_id="evaluation_factor",
        version="1",
        family="test",
        hypothesis="Higher signal ranks predict higher next-session returns.",
        economic_rationale="Synthetic monotone fixture for the governed evaluator.",
        falsification_criterion="Reject unless validation rank IC is positive.",
        expression="Signal",
        direction=1,
        required_fields=("Signal",),
        dataset_id="evaluation-data",
        snapshot_id="a" * 64,
        schema_hash="b" * 64,
        availability_hash="c" * 64,
        security_contract_hash="f" * 64,
        frequency=frequency,
        aggregation=AggregationSpec(),
        operator_registry_version=registry.version,
        operator_registry_digest=registry.digest,
        preprocessing=(),
        complexity_budget=ComplexityBudget(),
        provenance=FactorProvenance(origin="human", actor_id="test"),
    )
    factor = FactorResult(
        factor_spec_hash=factor_spec.content_hash,
        semantic_hash=factor_spec.semantic_hash,
        definition_hash=factor_spec.definition_hash,
        factor_view_hash="e" * 64,
        execution_policy_hash=STRICT_FACTOR_ENGINE_EXECUTION_POLICY_HASH,
        operator_registry_digest=registry.digest,
        complexity=FactorComplexity(
            ast_nodes=1,
            call_count=0,
            maximum_call_depth=0,
            maximum_window=0,
            field_count=1,
            estimated_relative_ops_per_cell=0,
        ),
        signal_hash=hash_frame(signal),
        runtime_seconds=0.001,
        admission_eligible=True,
        signal=signal,
    )
    validation_timestamps = timestamps[2:7]
    split = SplitFoldReceipt(
        fold_id="fold-1",
        train_signals=tuple(timestamp.isoformat() for timestamp in timestamps[:2]),
        validation_signals=tuple(
            timestamp.isoformat() for timestamp in validation_timestamps
        ),
        purged_train_signals=(),
        excluded_validation_signals=(),
    )
    validation = ValidationReceipt(
        validation_spec_hash="f" * 64,
        label_spec_hash=labels.label_spec_hash,
        labels_hash=labels.labels_hash,
        windows_hash=labels.windows_hash,
        calendar_hash="0" * 64,
        folds=(split,),
    )
    return factor_spec, factor, label_spec, labels, validation


def test_evaluation_reports_ic_quantiles_monotonicity_and_serial_metrics() -> None:
    factor_spec, factor, label_spec, labels, validation = _fixture()
    report = FactorEvaluationSuite().evaluate(
        EvaluationSpec(
            evaluation_id="test",
            version="1",
            quantile_count=5,
            minimum_cross_sectional_observations=10,
        ),
        factor_spec,
        factor,
        label_spec,
        labels,
        validation,
        fold_id="fold-1",
    )
    assert report.summary["rank_ic_mean"] == pytest.approx(1.0)
    assert report.summary["pearson_ic_mean"] == pytest.approx(1.0)
    assert report.summary["rank_ic_positive_ratio"] == pytest.approx(1.0)
    assert report.summary["quantile_monotonicity"] == pytest.approx(1.0)
    assert report.summary["top_bottom_spread_mean"] > 0.0
    assert report.summary["coverage_mean"] == pytest.approx(1.0)
    assert report.summary["rank_turnover_mean"] == pytest.approx(0.0)
    assert report.summary["factor_autocorrelation_mean"] == pytest.approx(1.0)
    assert set(report.metric_metadata.columns) == {
        "sample_count",
        "unit",
        "status",
        "source_artifact_hash",
    }
    assert report.metric_metadata.loc["rank_ic_mean", "sample_count"] == 5
    assert report.content_hash


def test_evaluation_uses_only_receipted_validation_signals() -> None:
    factor_spec, factor, label_spec, labels, validation = _fixture()
    evaluator = FactorEvaluationSuite()
    evaluation_spec = EvaluationSpec(
        evaluation_id="test-validation-only",
        version="1",
        quantile_count=5,
        minimum_cross_sectional_observations=10,
    )
    baseline = evaluator.evaluate(
        evaluation_spec,
        factor_spec,
        factor,
        label_spec,
        labels,
        validation,
        fold_id="fold-1",
    )
    changed_signal = factor.signal.copy()
    changed_signal.iloc[:2] = changed_signal.iloc[:2, ::-1].to_numpy()
    changed = replace(
        factor,
        signal=changed_signal,
        signal_hash=hash_frame(changed_signal),
    )
    repeated = evaluator.evaluate(
        evaluation_spec,
        factor_spec,
        changed,
        label_spec,
        labels,
        validation,
        fold_id="fold-1",
    )
    pd.testing.assert_frame_equal(baseline.per_date, repeated.per_date)
    pd.testing.assert_frame_equal(baseline.quantile_returns, repeated.quantile_returns)
    assert dict(baseline.summary) == dict(repeated.summary)


def test_direction_reference_incremental_ic_and_binding_fail_closed() -> None:
    factor_spec, factor, label_spec, labels, validation = _fixture()
    evaluation_spec = EvaluationSpec(
        evaluation_id="test-reference",
        version="1",
        quantile_count=5,
        minimum_cross_sectional_observations=10,
    )
    reference = factor.signal * 3.0
    report = FactorEvaluationSuite().evaluate(
        evaluation_spec,
        factor_spec,
        factor,
        label_spec,
        labels,
        validation,
        fold_id="fold-1",
        reference_factors={"existing_factor": reference},
        decay_labels={"1_session": labels},
    )
    assert report.summary["factor_correlation__existing_factor"] == pytest.approx(1.0)
    assert report.summary["incremental_rank_ic_mean"] is None
    assert report.summary["rank_ic_decay__1_session"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="factor specification binding"):
        FactorEvaluationSuite().evaluate(
            evaluation_spec,
            replace(factor_spec, factor_id="different_factor"),
            factor,
            label_spec,
            labels,
            validation,
            fold_id="fold-1",
        )


def test_negative_direction_is_applied_without_learning_from_validation() -> None:
    factor_spec, factor, label_spec, labels, validation = _fixture()
    negative_labels_frame = -labels.labels
    negative_labels = replace(
        labels,
        labels=negative_labels_frame,
        labels_hash=hash_frame(negative_labels_frame),
    )
    negative_validation = replace(
        validation,
        labels_hash=negative_labels.labels_hash,
    )
    negative_spec = replace(factor_spec, direction=-1)
    negative_factor = replace(
        factor,
        factor_spec_hash=negative_spec.content_hash,
    )
    report = FactorEvaluationSuite().evaluate(
        EvaluationSpec(
            evaluation_id="negative-direction",
            version="1",
            quantile_count=5,
            minimum_cross_sectional_observations=10,
        ),
        negative_spec,
        negative_factor,
        label_spec,
        negative_labels,
        negative_validation,
        fold_id="fold-1",
    )
    assert report.summary["rank_ic_mean"] == pytest.approx(1.0)
    assert report.summary["raw_rank_ic_mean"] == pytest.approx(-1.0)


@pytest.mark.parametrize(
    "invalid_validity",
    [
        lambda validity: validity.astype("boolean").mask(
            pd.DataFrame(
                [[True]],
                index=validity.index[:1],
                columns=validity.columns[:1],
            )
        ),
        lambda validity: validity.astype(object),
        lambda validity: validity.astype(int),
    ],
    ids=["nan", "object", "integer"],
)
def test_evaluation_rejects_non_complete_boolean_validity(
    invalid_validity,
) -> None:
    factor_spec, factor, label_spec, labels, validation = _fixture()
    attacked = invalid_validity(labels.validity)
    object.__setattr__(labels, "validity", attacked)
    object.__setattr__(labels, "validity_hash", hash_frame(attacked))
    labels.verify_content()

    with pytest.raises(TypeError, match="complete boolean panel"):
        FactorEvaluationSuite().evaluate(
            EvaluationSpec(
                evaluation_id="strict-boolean-validity",
                version="1",
                quantile_count=5,
                minimum_cross_sectional_observations=10,
            ),
            factor_spec,
            factor,
            label_spec,
            labels,
            validation,
            fold_id="fold-1",
        )
