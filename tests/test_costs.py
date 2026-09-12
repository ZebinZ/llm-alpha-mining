from __future__ import annotations

import pytest

from llm_alpha_mining.research.costs import (
    CostCalibration,
    CostEstimator,
    CostModel,
    TradeSide,
)


def _proxy(**changes) -> CostModel:
    values = {
        "cost_model_id": "proxy-test",
        "version": "1",
        "calibration": CostCalibration.PROXY,
        "commission_bps_buy": 2.0,
        "commission_bps_sell": 2.0,
        "minimum_commission": 5.0,
        "stamp_tax_bps_sell": 5.0,
        "fallback_half_spread_bps": 3.0,
        "slippage_bps": 1.0,
        "impact_coefficient": 0.5,
        "impact_exponent": 1.0,
        "annual_borrow_bps": 100.0,
        "require_observed_spread": False,
        "require_observed_volatility": False,
        "require_adv": False,
    }
    values.update(changes)
    return CostModel(**values)


def test_zero_cost_reference_is_exactly_zero() -> None:
    model = CostModel.zero()
    cost = CostEstimator().estimate_trade(
        model,
        side=TradeSide.BUY,
        notional=1_000_000.0,
        participation_rate=None,
        volatility=None,
        observed_half_spread_bps=None,
    )
    assert cost.total == 0.0
    assert model.production_ready is False


def test_sell_cost_has_stamp_tax_and_cost_is_monotone_in_participation() -> None:
    model = _proxy()
    estimator = CostEstimator()
    buy = estimator.estimate_trade(
        model,
        side="buy",
        notional=1_000_000.0,
        participation_rate=0.01,
        volatility=0.02,
        observed_half_spread_bps=2.0,
    )
    sell = estimator.estimate_trade(
        model,
        side="sell",
        notional=1_000_000.0,
        participation_rate=0.01,
        volatility=0.02,
        observed_half_spread_bps=2.0,
    )
    larger = estimator.estimate_trade(
        model,
        side="buy",
        notional=1_000_000.0,
        participation_rate=0.10,
        volatility=0.02,
        observed_half_spread_bps=2.0,
    )
    assert sell.stamp_tax > 0.0
    assert sell.total > buy.total
    assert larger.market_impact > buy.market_impact


def test_calibrated_model_fails_closed_without_observed_inputs() -> None:
    calibrated = _proxy(
        calibration=CostCalibration.CALIBRATED,
        require_observed_spread=True,
        require_observed_volatility=True,
        require_adv=True,
        calibrated_at="2026-07-19T00:00:00+08:00",
        calibration_start_at="2026-01-01T00:00:00+08:00",
        calibration_end_at="2026-06-30T23:59:59+08:00",
        calibration_artifact_hash="a" * 64,
        calibration_data_snapshot_id="b" * 64,
    )
    assert calibrated.production_ready is True
    with pytest.raises(ValueError, match="participation_rate"):
        CostEstimator().estimate_trade(
            calibrated,
            side="buy",
            notional=100_000.0,
            participation_rate=None,
            volatility=0.02,
            observed_half_spread_bps=2.0,
        )


def test_proxy_cannot_claim_production_readiness() -> None:
    assert _proxy().production_ready is False
    with pytest.raises(ValueError, match="calibrated_at"):
        _proxy(calibration=CostCalibration.CALIBRATED)
    with pytest.raises(ValueError, match="calibration artifact"):
        _proxy(
            calibration=CostCalibration.CALIBRATED,
            require_observed_spread=True,
            require_observed_volatility=True,
            require_adv=True,
            calibrated_at="2026-07-19T00:00:00+08:00",
            calibration_start_at="2026-01-01T00:00:00+08:00",
            calibration_end_at="2026-06-30T23:59:59+08:00",
        )


def test_calibrated_model_binds_ordered_point_in_time_window() -> None:
    with pytest.raises(ValueError, match="calibration_start_at"):
        _proxy(
            calibration=CostCalibration.CALIBRATED,
            require_observed_spread=True,
            require_observed_volatility=True,
            require_adv=True,
            calibrated_at="2026-07-19T00:00:00+08:00",
            calibration_end_at="2026-06-30T23:59:59+08:00",
            calibration_artifact_hash="a" * 64,
            calibration_data_snapshot_id="b" * 64,
        )
    with pytest.raises(ValueError, match="calibration_end_at"):
        _proxy(
            calibration=CostCalibration.CALIBRATED,
            require_observed_spread=True,
            require_observed_volatility=True,
            require_adv=True,
            calibrated_at="2026-07-19T00:00:00+08:00",
            calibration_start_at="2026-01-01T00:00:00+08:00",
            calibration_artifact_hash="a" * 64,
            calibration_data_snapshot_id="b" * 64,
        )
    with pytest.raises(ValueError, match="earlier than calibration_end_at"):
        _proxy(
            calibration=CostCalibration.CALIBRATED,
            require_observed_spread=True,
            require_observed_volatility=True,
            require_adv=True,
            calibrated_at="2026-07-19T00:00:00+08:00",
            calibration_start_at="2026-06-30T23:59:59+08:00",
            calibration_end_at="2026-01-01T00:00:00+08:00",
            calibration_artifact_hash="a" * 64,
            calibration_data_snapshot_id="b" * 64,
        )
    with pytest.raises(ValueError, match="cannot precede"):
        _proxy(
            calibration=CostCalibration.CALIBRATED,
            require_observed_spread=True,
            require_observed_volatility=True,
            require_adv=True,
            calibrated_at="2026-06-01T00:00:00+08:00",
            calibration_start_at="2026-01-01T00:00:00+08:00",
            calibration_end_at="2026-06-30T23:59:59+08:00",
            calibration_artifact_hash="a" * 64,
            calibration_data_snapshot_id="b" * 64,
        )
    with pytest.raises(ValueError, match="non-calibrated"):
        _proxy(calibration_start_at="2026-01-01T00:00:00+08:00")


def test_cost_model_schema_v2_serializes_calibration_window() -> None:
    calibrated = _proxy(
        calibration=CostCalibration.CALIBRATED,
        require_observed_spread=True,
        require_observed_volatility=True,
        require_adv=True,
        calibrated_at="2026-07-19T00:00:00+08:00",
        calibration_start_at="2026-01-01T00:00:00+08:00",
        calibration_end_at="2026-06-30T23:59:59+08:00",
        calibration_artifact_hash="a" * 64,
        calibration_data_snapshot_id="b" * 64,
    )
    payload = calibrated.to_dict()
    assert payload["schema_version"] == "cost-model/v2"
    assert payload["calibration_start_at"] == "2026-01-01T00:00:00+08:00"
    assert payload["calibration_end_at"] == "2026-06-30T23:59:59+08:00"
