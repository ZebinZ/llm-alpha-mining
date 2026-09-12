from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from alpha_research.core.frequency import FrequencySpec, SessionSpec, TradingCalendar
from alpha_research.core.hashing import hash_frame, hash_json
from alpha_research.labels import (
    EntryEvent,
    LabelBuilder,
    LabelDataView,
    LabelSpec,
    LabelTask,
    PriceObservationEvent,
)
from alpha_research.validation import (
    PurgedWalkForwardSplitter,
    SplitFoldReceipt,
    ValidationFoldSpec,
    ValidationReceiptVerifier,
    ValidationSpec,
)


SECURITIES = pd.Index(["000001", "000002", "399001"], name="security")


def _label_fixture(*, blocked_exit: bool = False):
    timestamps = pd.date_range(
        "2020-01-02 15:00:00",
        periods=30,
        freq="B",
        tz="Asia/Shanghai",
        name="timestamp",
    )
    step = np.arange(len(timestamps), dtype=float)
    open_panel = pd.DataFrame(
        {
            "000001": 10.0 * np.power(1.01, step),
            "000002": 20.0 * np.power(0.995, step),
            "399001": 100.0 * np.power(1.002, step),
        },
        index=timestamps,
    )
    close_panel = pd.DataFrame(
        {
            "000001": 10.1 * np.exp(0.001 * step + 0.003 * np.sin(step)),
            "000002": 19.9 * np.exp(-0.0005 * step + 0.002 * np.cos(step)),
            "399001": 100.2 * np.exp(0.0002 * step),
        },
        index=timestamps,
    )
    adjustment = pd.DataFrame(1.0, index=timestamps, columns=SECURITIES)
    amount = pd.DataFrame(100.0, index=timestamps, columns=SECURITIES)
    identity = pd.DataFrame(True, index=timestamps, columns=SECURITIES)
    identity["399001"] = False
    execution = identity.copy()
    if blocked_exit:
        # Signal at row zero enters at row one and exits at row two.
        execution.loc[timestamps[2], "000002"] = False
        amount.loc[timestamps[2], "000002"] = 0.0
    fields = {
        "AdjFactor": adjustment.where(identity),
        "Amount": amount.where(identity),
        "Close": close_panel.where(identity),
        "Open": open_panel.where(identity),
    }
    snapshot_id = "a" * 64
    schema_hash = "b" * 64
    security_hash = "c" * 64
    frequency = FrequencySpec.daily()
    payload = {
        "snapshot_id": snapshot_id,
        "schema_hash": schema_hash,
        "frequency_hash": frequency.content_hash,
        "availability_hash": "e" * 64,
        "security_contract_hash": security_hash,
        "production_ready": True,
        "fields": {name: hash_frame(value) for name, value in sorted(fields.items())},
        "field_observation_events": {
            "Close": PriceObservationEvent.SESSION_CLOSE.value,
            "Open": PriceObservationEvent.SESSION_OPEN.value,
        },
        "instrument_identity_mask": hash_frame(identity),
        "execution_tradability_mask": hash_frame(execution),
    }
    view = LabelDataView(
        snapshot_id=snapshot_id,
        schema_hash=schema_hash,
        frequency_hash=frequency.content_hash,
        availability_hash="e" * 64,
        security_contract_hash=security_hash,
        production_ready=True,
        fields=fields,
        field_observation_events={
            "Close": PriceObservationEvent.SESSION_CLOSE,
            "Open": PriceObservationEvent.SESSION_OPEN,
        },
        instrument_identity_mask=identity,
        execution_tradability_mask=execution,
        view_hash=hash_json(payload),
    )
    return timestamps, frequency, view, open_panel, close_panel


def _spec(
    frequency: FrequencySpec,
    view: LabelDataView,
    *,
    task: LabelTask = LabelTask.RETURN,
    horizon: int = 1,
) -> LabelSpec:
    volatility = task is LabelTask.VOLATILITY
    excess = task is LabelTask.EXCESS_RETURN
    return LabelSpec(
        label_id=f"test_{task.value}_{horizon}",
        version="1",
        task=task,
        horizon_sessions=horizon,
        information_cutoff_event="signal_session_close",
        entry_event=(
            EntryEvent.SIGNAL_SESSION_CLOSE
            if volatility
            else EntryEvent.NEXT_SESSION_OPEN
        ),
        entry_lag_sessions=0 if volatility else 1,
        price_field="Close" if volatility else "Open",
        price_observation_event=(
            PriceObservationEvent.SESSION_CLOSE
            if volatility
            else PriceObservationEvent.SESSION_OPEN
        ),
        adjustment_field="AdjFactor",
        amount_field=None if volatility else "Amount",
        require_execution_tradability=not volatility,
        benchmark_id="000300" if excess else None,
        benchmark_snapshot_id="d" * 64 if excess else None,
        direction_threshold=0.0,
        annualize_volatility=volatility,
        missing_policy="invalidate_label",
        dataset_id="phase2-label-data",
        snapshot_id=view.snapshot_id,
        schema_hash=view.schema_hash,
        availability_hash="e" * 64,
        security_contract_hash=view.security_contract_hash,
        frequency=frequency,
    )


def test_next_open_label_has_explicit_window_and_future_tradability_only_invalidates_label() -> (
    None
):
    timestamps, frequency, view, open_panel, _ = _label_fixture(blocked_exit=True)
    spec = _spec(frequency, view)
    result = LabelBuilder().build(
        spec,
        view,
        signal_timestamps=timestamps[:5],
    )
    expected = (
        open_panel.loc[timestamps[2], "000001"]
        / open_panel.loc[timestamps[1], "000001"]
        - 1.0
    )
    assert result.labels.loc[timestamps[0], "000001"] == pytest.approx(expected)
    assert pd.isna(result.labels.loc[timestamps[0], "000002"])
    assert not result.validity.loc[timestamps[0], "000002"]
    assert pd.isna(result.labels.loc[timestamps[0], "399001"])
    first_window = result.label_windows.iloc[0]
    assert first_window["information_cutoff"] == timestamps[0]
    assert first_window["entry_timestamp"] == timestamps[1]
    assert first_window["exit_timestamp"] == timestamps[2]


def test_excess_direction_and_volatility_tasks_are_bound_and_deterministic() -> None:
    timestamps, frequency, view, open_panel, _ = _label_fixture()
    benchmark = pd.Series(
        100.0 * np.power(1.003, np.arange(len(timestamps))), index=timestamps
    )
    excess_spec = _spec(frequency, view, task=LabelTask.EXCESS_RETURN, horizon=5)
    excess = LabelBuilder().build(
        excess_spec,
        view,
        signal_timestamps=timestamps[:5],
        benchmark_adjusted_prices=benchmark,
        benchmark_snapshot_id="d" * 64,
    )
    stock_return = (
        open_panel.loc[timestamps[6], "000001"]
        / open_panel.loc[timestamps[1], "000001"]
        - 1.0
    )
    benchmark_return = benchmark.loc[timestamps[6]] / benchmark.loc[timestamps[1]] - 1.0
    assert excess.labels.iloc[0, 0] == pytest.approx(stock_return - benchmark_return)
    with pytest.raises(ValueError, match="benchmark snapshot"):
        LabelBuilder().build(
            excess_spec,
            view,
            signal_timestamps=timestamps[:2],
            benchmark_adjusted_prices=benchmark,
            benchmark_snapshot_id="f" * 64,
        )

    direction = LabelBuilder().build(
        _spec(frequency, view, task=LabelTask.DIRECTION),
        view,
        signal_timestamps=timestamps[:5],
    )
    assert direction.labels.loc[timestamps[0], "000001"] == 1.0
    assert direction.labels.loc[timestamps[0], "000002"] == 0.0

    volatility_spec = _spec(frequency, view, task=LabelTask.VOLATILITY, horizon=5)
    first = LabelBuilder().build(
        volatility_spec, view, signal_timestamps=timestamps[:4]
    )
    second = LabelBuilder().build(
        volatility_spec, view, signal_timestamps=timestamps[:4]
    )
    assert first.labels_hash == second.labels_hash
    assert first.labels.loc[timestamps[0], "000001"] > 0.0


@pytest.mark.parametrize("horizon", [1, 5, 20])
def test_return_horizons_match_manual_next_open_calculation(horizon: int) -> None:
    timestamps, frequency, view, open_panel, _ = _label_fixture()
    result = LabelBuilder().build(
        _spec(frequency, view, horizon=horizon),
        view,
        signal_timestamps=timestamps[:2],
    )
    expected = (
        open_panel.loc[timestamps[1 + horizon], "000001"]
        / open_panel.loc[timestamps[1], "000001"]
        - 1.0
    )
    assert result.labels.loc[timestamps[0], "000001"] == pytest.approx(expected)
    window = result.label_windows.iloc[0]
    assert window["entry_timestamp"] == timestamps[1]
    assert window["exit_timestamp"] == timestamps[1 + horizon]


def test_label_spec_rejects_same_close_return_and_fail_open_missing_policy() -> None:
    _, frequency, view, _, _ = _label_fixture()
    base = _spec(frequency, view)
    with pytest.raises(ValueError, match="return/direction labels"):
        replace(
            base,
            entry_event=EntryEvent.SIGNAL_SESSION_CLOSE,
            entry_lag_sessions=0,
        )
    with pytest.raises(ValueError, match="fail closed"):
        replace(base, missing_policy="drop_missing_and_continue")


def test_price_observation_semantics_are_explicit_and_bound_to_view_hash() -> None:
    timestamps, frequency, view, _, _ = _label_fixture()
    base = _spec(frequency, view)
    with pytest.raises(ValueError, match="session-open price observation"):
        replace(
            base,
            price_observation_event=PriceObservationEvent.SESSION_CLOSE,
        )

    # The spec's declared event is valid for a next-open label, but the selected
    # field is authenticated as a session-close observation by LabelDataView.
    with pytest.raises(ValueError, match="price observation event differs"):
        LabelBuilder().build(
            replace(base, price_field="Close"),
            view,
            signal_timestamps=timestamps[:2],
        )

    with pytest.raises(ValueError, match="view hash differs"):
        replace(
            view,
            field_observation_events={
                "Close": PriceObservationEvent.SESSION_OPEN,
                "Open": PriceObservationEvent.SESSION_OPEN,
            },
        )


def test_label_artifact_mutation_is_detected_before_validation() -> None:
    timestamps, frequency, view, _, _ = _label_fixture()
    labels = LabelBuilder().build(
        _spec(frequency, view), view, signal_timestamps=timestamps[:5]
    )
    labels.labels.iloc[0, 0] = 999.0
    with pytest.raises(RuntimeError, match="label values changed"):
        labels.verify_content()


def test_label_result_has_one_stable_shared_scientific_identity() -> None:
    timestamps, frequency, view, _, _ = _label_fixture()
    labels = LabelBuilder().build(
        _spec(frequency, view), view, signal_timestamps=timestamps[:5]
    )

    assert labels.content_hash == hash_json(labels.scientific_descriptor())
    assert labels.scientific_descriptor() == {
        "label_spec_hash": labels.label_spec_hash,
        "label_view_hash": labels.label_view_hash,
        "benchmark_hash": labels.benchmark_hash,
        "labels_hash": labels.labels_hash,
        "windows_hash": labels.windows_hash,
        "validity_hash": labels.validity_hash,
        "diagnostics_hash": labels.diagnostics_hash,
    }


@pytest.mark.parametrize(
    "invalid_mask",
    [
        lambda mask: mask.astype(float).mask(
            pd.DataFrame(
                [[True, False, False]],
                index=mask.index[:1],
                columns=mask.columns,
            )
        ),
        lambda mask: mask.astype(object),
        lambda mask: mask.astype(int),
    ],
    ids=["nan", "object", "integer"],
)
@pytest.mark.parametrize(
    "mask_field",
    ["instrument_identity_mask", "execution_tradability_mask"],
)
def test_label_view_rejects_non_complete_boolean_masks(
    invalid_mask,
    mask_field: str,
) -> None:
    _, _, view, _, _ = _label_fixture()
    bad = invalid_mask(getattr(view, mask_field))

    with pytest.raises(TypeError, match="complete boolean panel"):
        replace(view, **{mask_field: bad})


@pytest.mark.parametrize(
    "invalid_validity",
    [
        lambda validity: validity.astype(float).mask(
            pd.DataFrame(
                [[True, False, False]],
                index=validity.index[:1],
                columns=validity.columns,
            )
        ),
        lambda validity: validity.astype(object),
        lambda validity: validity.astype(int),
    ],
    ids=["nan", "object", "integer"],
)
def test_label_result_rejects_non_complete_boolean_validity(
    invalid_validity,
) -> None:
    timestamps, frequency, view, _, _ = _label_fixture()
    labels = LabelBuilder().build(
        _spec(frequency, view),
        view,
        signal_timestamps=timestamps[:5],
    )

    with pytest.raises(TypeError, match="complete boolean panel"):
        replace(labels, validity=invalid_validity(labels.validity))


def test_purged_walk_forward_uses_complete_label_windows_and_embargo() -> None:
    timestamps, frequency, view, _, _ = _label_fixture()
    labels = LabelBuilder().build(
        _spec(frequency, view),
        view,
        signal_timestamps=timestamps[:-2],
    )
    fold = ValidationFoldSpec(
        fold_id="fold-1",
        train_start=timestamps[0].isoformat(),
        train_end=timestamps[10].isoformat(),
        validation_start=timestamps[12].isoformat(),
        validation_end=timestamps[20].isoformat(),
    )
    spec = ValidationSpec(
        validation_id="phase2-test",
        version="1",
        folds=(fold,),
        embargo_sessions=1,
    )
    calendar = TradingCalendar(
        calendar_id="SSE_SZSE",
        timezone="Asia/Shanghai",
        sessions=tuple(timestamp.strftime("%Y%m%d") for timestamp in timestamps),
        session=SessionSpec.cn_a_share_regular(),
    )
    receipt = PurgedWalkForwardSplitter().split(spec, labels, calendar)
    result = receipt.folds[0]
    assert result.train_signals[-1] == timestamps[8].isoformat()
    assert set(result.purged_train_signals) == {
        timestamps[9].isoformat(),
        timestamps[10].isoformat(),
    }
    assert result.validation_signals[0] == timestamps[12].isoformat()
    assert result.validation_signals[-1] == timestamps[18].isoformat()
    assert set(result.excluded_validation_signals) == {
        timestamps[19].isoformat(),
        timestamps[20].isoformat(),
    }
    assert (
        receipt.content_hash
        == PurgedWalkForwardSplitter().split(spec, labels, calendar).content_hash
    )
    assert (
        ValidationReceiptVerifier().verify(
            receipt,
            spec=spec,
            labels=labels,
            calendar=calendar,
        )
        == receipt
    )


def test_validation_receipt_verifier_rejects_forged_fold_membership() -> None:
    timestamps, frequency, view, _, _ = _label_fixture()
    labels = LabelBuilder().build(
        _spec(frequency, view),
        view,
        signal_timestamps=timestamps[:-2],
    )
    spec = ValidationSpec(
        validation_id="phase2-verifier-test",
        version="1",
        folds=(
            ValidationFoldSpec(
                fold_id="fold-1",
                train_start=timestamps[0].isoformat(),
                train_end=timestamps[10].isoformat(),
                validation_start=timestamps[12].isoformat(),
                validation_end=timestamps[20].isoformat(),
            ),
        ),
        embargo_sessions=1,
    )
    calendar = TradingCalendar(
        calendar_id="SSE_SZSE",
        timezone="Asia/Shanghai",
        sessions=tuple(timestamp.strftime("%Y%m%d") for timestamp in timestamps),
        session=SessionSpec.cn_a_share_regular(),
    )
    receipt = PurgedWalkForwardSplitter().split(spec, labels, calendar)
    original = receipt.folds[0]
    forged_fold = SplitFoldReceipt(
        fold_id=original.fold_id,
        train_signals=original.train_signals[:-1],
        validation_signals=original.validation_signals,
        purged_train_signals=original.purged_train_signals
        + original.train_signals[-1:],
        excluded_validation_signals=original.excluded_validation_signals,
    )
    forged = replace(receipt, folds=(forged_fold,))

    with pytest.raises(ValueError, match="recomputation_mismatch"):
        ValidationReceiptVerifier().verify(
            forged,
            spec=spec,
            labels=labels,
            calendar=calendar,
        )


def test_validation_rejects_random_split_and_unsatisfied_embargo() -> None:
    timestamps, frequency, view, _, _ = _label_fixture()
    labels = LabelBuilder().build(
        _spec(frequency, view), view, signal_timestamps=timestamps[:-2]
    )
    with pytest.raises(ValueError, match="purged_walk_forward"):
        ValidationSpec(
            validation_id="bad-random",
            version="1",
            folds=(
                ValidationFoldSpec(
                    fold_id="fold",
                    train_start=timestamps[0].isoformat(),
                    train_end=timestamps[10].isoformat(),
                    validation_start=timestamps[12].isoformat(),
                    validation_end=timestamps[20].isoformat(),
                ),
            ),
            method="random_kfold",
        )
    fold = ValidationFoldSpec(
        fold_id="fold",
        train_start=timestamps[0].isoformat(),
        train_end=timestamps[10].isoformat(),
        validation_start=timestamps[11].isoformat(),
        validation_end=timestamps[20].isoformat(),
    )
    spec = ValidationSpec(
        validation_id="bad-embargo",
        version="1",
        folds=(fold,),
        embargo_sessions=1,
    )
    calendar = TradingCalendar(
        calendar_id="SSE_SZSE",
        timezone="Asia/Shanghai",
        sessions=tuple(timestamp.strftime("%Y%m%d") for timestamp in timestamps),
        session=SessionSpec.cn_a_share_regular(),
    )
    with pytest.raises(ValueError, match="embargo_not_satisfied"):
        PurgedWalkForwardSplitter().split(spec, labels, calendar)


def test_validation_rejects_overlapping_validation_ranges() -> None:
    timestamps, _, _, _, _ = _label_fixture()
    first = ValidationFoldSpec(
        fold_id="fold-1",
        train_start=timestamps[0].isoformat(),
        train_end=timestamps[5].isoformat(),
        validation_start=timestamps[8].isoformat(),
        validation_end=timestamps[14].isoformat(),
    )
    overlapping = ValidationFoldSpec(
        fold_id="fold-2",
        train_start=timestamps[0].isoformat(),
        train_end=timestamps[10].isoformat(),
        validation_start=timestamps[14].isoformat(),
        validation_end=timestamps[20].isoformat(),
    )

    with pytest.raises(ValueError, match="disjoint and non-overlapping"):
        ValidationSpec(
            validation_id="overlapping-validation-folds",
            version="1",
            folds=(first, overlapping),
        )
