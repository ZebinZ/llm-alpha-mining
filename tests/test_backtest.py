from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from llm_alpha_mining.research.backtest import (
    BacktestDataView,
    BacktestEngine,
    BacktestSpec,
    ExecutionSpec,
)
from llm_alpha_mining.research.core.hashing import hash_frame
from llm_alpha_mining.research.costs import CostCalibration, CostModel
from llm_alpha_mining.research.portfolio import (
    ExposureMode,
    PortfolioBuilder,
    PortfolioConstruction,
    PortfolioSpec,
    RiskSpec,
)


def _event_timestamp(timestamp: pd.Timestamp, local_time: str) -> pd.Timestamp:
    local_date = timestamp.tz_convert("Asia/Shanghai").date().isoformat()
    return pd.Timestamp(f"{local_date}T{local_time}").tz_localize("Asia/Shanghai")


def _pit_availability(
    timestamps: pd.DatetimeIndex,
    securities: pd.Index,
) -> dict[str, pd.DataFrame]:
    def panel(local_time: str) -> pd.DataFrame:
        values = [
            [_event_timestamp(timestamp, local_time)] * len(securities)
            for timestamp in timestamps
        ]
        return pd.DataFrame(values, index=timestamps, columns=securities)

    at_open = panel("09:30:00")
    at_close = panel("15:00:00")
    return {
        "open_price": at_open.copy(),
        "close_price": at_close.copy(),
        "traded_amount": at_close.copy(),
        "adv_amount": at_close.copy(),
        "volatility": at_close.copy(),
        "half_spread_bps": at_close.copy(),
        "execution_tradability": at_open.copy(),
        "execution_reason": at_open.copy(),
        "split_ratio": at_open.copy(),
        "cash_dividend": at_open.copy(),
    }


def _copy_data_with_availability(
    data: BacktestDataView,
    known_at_panels: dict[str, pd.DataFrame],
    *,
    split_ratio: pd.DataFrame | None = None,
    cash_dividend: pd.DataFrame | None = None,
) -> BacktestDataView:
    return BacktestDataView.create(
        snapshot_id=data.snapshot_id,
        security_contract_hash=data.security_contract_hash,
        benchmark_id=data.benchmark_id,
        benchmark_snapshot_id=data.benchmark_snapshot_id,
        benchmark_close=data.benchmark_close,
        open_price=data.open_price,
        close_price=data.close_price,
        traded_amount=data.traded_amount,
        adv_amount=data.adv_amount,
        volatility=data.volatility,
        half_spread_bps=data.half_spread_bps,
        execution_tradability=data.execution_tradability,
        execution_reason=data.execution_reason,
        split_ratio=data.split_ratio if split_ratio is None else split_ratio,
        cash_dividend=data.cash_dividend if cash_dividend is None else cash_dividend,
        known_at_panels=known_at_panels,
        production_ready=True,
    )


def _fixture(
    *,
    tradable: bool = True,
    traded_amount: float = 1_000_000_000.0,
    production_ready: bool = False,
    nontradable_reason: str = "not_tradable",
):
    timestamps = pd.date_range(
        "2020-10-09 15:00:00",
        periods=6,
        freq="B",
        tz="Asia/Shanghai",
        name="timestamp",
    )
    securities = pd.Index(["000001", "000002"])
    scores = pd.DataFrame(
        [[2.0, 1.0]] * len(timestamps), index=timestamps, columns=securities
    )
    universe = pd.DataFrame(True, index=timestamps, columns=securities)
    portfolio_spec = PortfolioSpec(
        portfolio_id="long-only-test",
        version="1",
        factor_definition_hash="a" * 64,
        score_hash=hash_frame(scores),
        universe_hash=hash_frame(universe),
        construction=PortfolioConstruction.TOP_QUANTILE_EQUAL,
        exposure_mode=ExposureMode.LONG_ONLY,
        selection_fraction=0.5,
        gross_leverage=1.0,
        target_net_exposure=1.0,
        maximum_absolute_weight=1.0,
        maximum_one_way_turnover=1.0,
        minimum_names_per_side=1,
    )
    risk_exposure = pd.DataFrame(0.0, index=timestamps, columns=securities)
    risk_known_at = pd.DataFrame(
        [[timestamp] * len(securities) for timestamp in timestamps],
        index=timestamps,
        columns=securities,
    )
    risk_spec = (
        RiskSpec(
            risk_model_id="research-pit-risk",
            version="1",
            known_at=timestamps[0].isoformat(),
            exposure_hashes={"beta": hash_frame(risk_exposure)},
            availability_hashes={"beta": hash_frame(risk_known_at)},
            maximum_absolute_exposure={"beta": 0.1},
            point_in_time=True,
        )
        if production_ready
        else None
    )
    portfolio = PortfolioBuilder().build(
        portfolio_spec,
        scores,
        universe,
        risk_spec=risk_spec,
        risk_exposures=None if risk_spec is None else {"beta": risk_exposure},
        risk_known_at=None if risk_spec is None else {"beta": risk_known_at},
    )
    open_price = pd.DataFrame(
        {
            "000001": [10.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            "000002": [10.0] * 6,
        },
        index=timestamps,
    )
    close_price = open_price.copy()
    amount = pd.DataFrame(traded_amount, index=timestamps, columns=securities)
    adv = pd.DataFrame(1_000_000_000.0, index=timestamps, columns=securities)
    volatility = pd.DataFrame(0.02, index=timestamps, columns=securities)
    spread = pd.DataFrame(2.0, index=timestamps, columns=securities)
    execution = pd.DataFrame(True, index=timestamps, columns=securities)
    execution.iloc[1, 0] = tradable
    execution_reason = pd.DataFrame("tradable", index=timestamps, columns=securities)
    if not tradable:
        execution_reason.iloc[1, 0] = nontradable_reason
    data = BacktestDataView.create(
        snapshot_id="b" * 64,
        security_contract_hash="c" * 64,
        open_price=open_price,
        close_price=close_price,
        traded_amount=amount,
        adv_amount=adv,
        volatility=volatility,
        half_spread_bps=spread,
        execution_tradability=execution,
        execution_reason=execution_reason,
        known_at_panels=(
            _pit_availability(timestamps, securities) if production_ready else None
        ),
        production_ready=production_ready,
    )
    return timestamps, portfolio, data


def _proxy() -> CostModel:
    return CostModel(
        cost_model_id="backtest-proxy",
        version="1",
        calibration=CostCalibration.PROXY,
        commission_bps_buy=2.0,
        commission_bps_sell=2.0,
        minimum_commission=0.0,
        stamp_tax_bps_sell=5.0,
        fallback_half_spread_bps=2.0,
        slippage_bps=1.0,
        impact_coefficient=0.25,
        impact_exponent=1.0,
        annual_borrow_bps=0.0,
        require_observed_spread=False,
        require_observed_volatility=False,
        require_adv=False,
    )


def _spec(
    portfolio,
    cost,
    data,
    *,
    participation=1.0,
    corporate_action_policy="pre_adjusted",
):
    execution = ExecutionSpec(
        execution_id="next-open-test",
        version="1",
        maximum_participation_rate=participation,
        lot_size=1,
        corporate_action_policy=corporate_action_policy,
    )
    return BacktestSpec(
        backtest_id="research-test",
        version="1",
        portfolio_spec_hash=portfolio.portfolio_spec_hash,
        portfolio_weights_hash=portfolio.weights_hash,
        portfolio_result_hash=portfolio.content_hash,
        data_view_hash=data.view_hash,
        cost_model_hash=cost.content_hash,
        execution=execution,
        initial_capital=100_000.0,
    )


def test_zero_cost_fully_filled_reference_and_accounting_invariant() -> None:
    timestamps, portfolio, data = _fixture()
    cost = CostModel.zero()
    result = BacktestEngine().run(_spec(portfolio, cost, data), portfolio, cost, data)
    assert result.positions.loc[timestamps[1], "000001"] == pytest.approx(10_000.0)
    assert result.nav.loc[timestamps[1], "gross_nav"] == pytest.approx(100_000.0)
    assert result.nav.loc[timestamps[2], "gross_nav"] == pytest.approx(110_000.0)
    pd.testing.assert_series_equal(
        result.nav["gross_nav"], result.nav["net_nav"], check_names=False
    )
    assert result.nav["total_cost"].sum() == 0.0
    assert result.metrics["gross_annual_return"] > 0.0
    assert "net_information_ratio" in result.metrics
    assert "average_holding_period_sessions" in result.metrics
    assert result.portfolio_result_hash == portfolio.content_hash
    result.verify_content()


def test_same_fills_have_identical_gross_and_lower_net_under_costs() -> None:
    _, portfolio, data = _fixture()
    zero = CostModel.zero()
    proxy = _proxy()
    gross = BacktestEngine().run(_spec(portfolio, zero, data), portfolio, zero, data)
    net = BacktestEngine().run(_spec(portfolio, proxy, data), portfolio, proxy, data)
    pd.testing.assert_frame_equal(gross.positions, net.positions)
    pd.testing.assert_series_equal(
        gross.nav["gross_nav"], net.nav["gross_nav"], check_names=False
    )
    assert (net.nav["net_nav"] <= net.nav["gross_nav"] + 1e-10).all()
    assert net.metrics["net_annual_return"] < net.metrics["gross_annual_return"]
    difference = net.nav["gross_nav"] - net.nav["net_nav"]
    np.testing.assert_allclose(
        difference.to_numpy(), net.nav["cumulative_cost"].to_numpy(), atol=1e-8
    )


def test_suspension_is_recorded_and_position_is_not_silently_created() -> None:
    timestamps, portfolio, data = _fixture(
        tradable=False, nontradable_reason="suspended"
    )
    cost = CostModel.zero()
    result = BacktestEngine().run(_spec(portfolio, cost, data), portfolio, cost, data)
    first = result.orders.loc[
        result.orders["execution_session_timestamp"] == timestamps[1]
    ].iloc[0]
    assert first["status"] == "suspended"
    assert result.positions.loc[timestamps[1], "000001"] == 0.0
    # The next rebalance recalculates the remaining target and can fill later.
    assert result.positions.loc[timestamps[2], "000001"] > 0.0


@pytest.mark.parametrize("reason", ["limit_up", "limit_down", "security_ineligible"])
def test_nontradable_reason_is_preserved_in_order_audit(reason: str) -> None:
    timestamps, portfolio, data = _fixture(tradable=False, nontradable_reason=reason)
    cost = CostModel.zero()
    result = BacktestEngine().run(_spec(portfolio, cost, data), portfolio, cost, data)
    order = result.orders.loc[
        result.orders["execution_session_timestamp"] == timestamps[1]
    ].iloc[0]
    assert order["status"] == reason
    assert result.positions.loc[timestamps[1], "000001"] == 0.0


def test_participation_cap_creates_audited_partial_fill() -> None:
    timestamps, portfolio, data = _fixture(traded_amount=10_000.0)
    cost = CostModel.zero()
    result = BacktestEngine().run(
        _spec(portfolio, cost, data, participation=0.10), portfolio, cost, data
    )
    order = result.orders.loc[
        result.orders["execution_session_timestamp"] == timestamps[1]
    ].iloc[0]
    fill = result.fills.loc[
        result.fills["execution_session_timestamp"] == timestamps[1]
    ].iloc[0]
    assert order["status"] == "partial_fill"
    assert fill["filled_shares"] == pytest.approx(100.0)
    assert result.positions.loc[timestamps[1], "000001"] == pytest.approx(100.0)


def test_next_open_capacity_and_cost_inputs_use_only_prior_session_data() -> None:
    timestamps, portfolio, data = _fixture()
    amount = data.traded_amount.copy()
    amount.loc[timestamps[0], "000001"] = 10_000.0
    amount.loc[timestamps[1], "000001"] = 1_000_000_000.0
    guarded = BacktestDataView.create(
        snapshot_id=data.snapshot_id,
        security_contract_hash=data.security_contract_hash,
        open_price=data.open_price,
        close_price=data.close_price,
        traded_amount=amount,
        adv_amount=data.adv_amount,
        volatility=data.volatility,
        half_spread_bps=data.half_spread_bps,
        execution_tradability=data.execution_tradability,
        execution_reason=data.execution_reason,
    )
    result = BacktestEngine().run(
        _spec(portfolio, CostModel.zero(), guarded, participation=0.10),
        portfolio,
        CostModel.zero(),
        guarded,
    )
    first = result.orders.loc[
        result.orders["execution_session_timestamp"] == timestamps[1]
    ].iloc[0]
    assert first["capacity_shares"] == pytest.approx(100.0)
    assert first["liquidity_observation_timestamp"] == timestamps[0]
    assert first["risk_observation_timestamp"] == timestamps[0]


def test_explicit_split_and_dividend_preserve_accounting() -> None:
    timestamps, portfolio, data = _fixture()
    split = data.split_ratio.copy()
    dividend = data.cash_dividend.copy()
    split.loc[timestamps[2], "000001"] = 2.0
    dividend.loc[timestamps[2], "000001"] = 1.0
    open_price = data.open_price.copy()
    close_price = data.close_price.copy()
    open_price.loc[timestamps[2] :, "000001"] /= 2.0
    close_price.loc[timestamps[2] :, "000001"] /= 2.0
    explicit = BacktestDataView.create(
        snapshot_id=data.snapshot_id,
        security_contract_hash=data.security_contract_hash,
        open_price=open_price,
        close_price=close_price,
        traded_amount=data.traded_amount,
        adv_amount=data.adv_amount,
        volatility=data.volatility,
        half_spread_bps=data.half_spread_bps,
        execution_tradability=data.execution_tradability,
        split_ratio=split,
        cash_dividend=dividend,
    )
    cost = CostModel.zero()
    result = BacktestEngine().run(
        _spec(
            portfolio,
            cost,
            explicit,
            corporate_action_policy="explicit_panels",
        ),
        portfolio,
        cost,
        explicit,
    )
    assert result.positions.loc[timestamps[2], "000001"] >= 20_000.0
    assert result.nav.loc[timestamps[2], "gross_nav"] > 100_000.0
    assert np.isfinite(result.nav[["gross_nav", "net_nav"]].to_numpy()).all()


def test_backtest_bindings_fail_closed() -> None:
    _, portfolio, data = _fixture()
    cost = CostModel.zero()
    with pytest.raises(ValueError, match="weights binding"):
        BacktestEngine().run(
            replace(_spec(portfolio, cost, data), portfolio_weights_hash="f" * 64),
            portfolio,
            cost,
            data,
        )


def test_production_gate_requires_pit_risk_calibrated_cost_and_ready_data() -> None:
    timestamps, portfolio, data = _fixture(production_ready=True)
    calibrated = CostModel(
        cost_model_id="calibrated-production-cost",
        version="1",
        calibration=CostCalibration.CALIBRATED,
        commission_bps_buy=2.0,
        commission_bps_sell=2.0,
        minimum_commission=0.0,
        stamp_tax_bps_sell=5.0,
        fallback_half_spread_bps=0.0,
        slippage_bps=1.0,
        impact_coefficient=0.25,
        impact_exponent=1.0,
        annual_borrow_bps=0.0,
        require_observed_spread=True,
        require_observed_volatility=True,
        require_adv=True,
        calibrated_at=(timestamps[0] - pd.Timedelta(days=1)).isoformat(),
        calibration_start_at=(timestamps[0] - pd.Timedelta(days=120)).isoformat(),
        calibration_end_at=(timestamps[0] - pd.Timedelta(days=2)).isoformat(),
        calibration_artifact_hash="e" * 64,
        calibration_data_snapshot_id="f" * 64,
    )
    ready = BacktestEngine().run(
        _spec(portfolio, calibrated, data), portfolio, calibrated, data
    )
    assert portfolio.production_ready is True
    assert data.production_ready is True
    assert ready.production_ready is True

    proxy = _proxy()
    research_only = BacktestEngine().run(
        _spec(portfolio, proxy, data), portfolio, proxy, data
    )
    assert research_only.production_ready is False


def test_production_data_requires_hash_bound_per_observation_availability() -> None:
    _, _, research_data = _fixture()
    with pytest.raises(ValueError, match="per-observation availability"):
        _copy_data_with_availability(research_data, {})


def test_production_execution_uses_real_open_clock_and_rejects_future_tradability() -> (
    None
):
    timestamps, portfolio, data = _fixture(production_ready=True)
    cost = CostModel.zero()
    result = BacktestEngine().run(_spec(portfolio, cost, data), portfolio, cost, data)
    first = result.orders.loc[
        result.orders["execution_session_timestamp"] == timestamps[1]
    ].iloc[0]
    assert first["execution_timestamp"] == _event_timestamp(timestamps[1], "09:30:00")

    availability = {name: panel.copy() for name, panel in data.known_at_panels.items()}
    availability["execution_tradability"].loc[timestamps[1], "000001"] = (
        _event_timestamp(timestamps[1], "09:31:00")
    )
    future = _copy_data_with_availability(data, availability)
    with pytest.raises(
        ValueError, match="future backtest availability:execution_tradability"
    ):
        BacktestEngine().run(_spec(portfolio, cost, future), portfolio, cost, future)


def test_production_execution_rejects_future_prior_liquidity_and_missing_risk() -> None:
    timestamps, portfolio, data = _fixture(production_ready=True)
    cost = CostModel.zero()
    availability = {name: panel.copy() for name, panel in data.known_at_panels.items()}
    availability["traded_amount"].loc[timestamps[0], "000001"] = _event_timestamp(
        timestamps[1], "09:31:00"
    )
    future_liquidity = _copy_data_with_availability(data, availability)
    with pytest.raises(ValueError, match="future backtest availability:traded_amount"):
        BacktestEngine().run(
            _spec(portfolio, cost, future_liquidity),
            portfolio,
            cost,
            future_liquidity,
        )

    availability = {name: panel.copy() for name, panel in data.known_at_panels.items()}
    availability["volatility"].loc[timestamps[0], "000001"] = pd.NaT
    missing_risk = _copy_data_with_availability(data, availability)
    with pytest.raises(ValueError, match="missing backtest availability:volatility"):
        BacktestEngine().run(
            _spec(portfolio, cost, missing_risk), portfolio, cost, missing_risk
        )


def test_production_valuation_and_corporate_actions_reject_future_inputs() -> None:
    timestamps, portfolio, data = _fixture(production_ready=True)
    cost = CostModel.zero()
    availability = {name: panel.copy() for name, panel in data.known_at_panels.items()}
    availability["close_price"].loc[timestamps[1], "000001"] = _event_timestamp(
        timestamps[1], "15:00:01"
    )
    future_close = _copy_data_with_availability(data, availability)
    with pytest.raises(ValueError, match="future backtest availability:close_price"):
        BacktestEngine().run(
            _spec(portfolio, cost, future_close), portfolio, cost, future_close
        )

    availability = {name: panel.copy() for name, panel in data.known_at_panels.items()}
    availability["split_ratio"].loc[timestamps[2], "000001"] = _event_timestamp(
        timestamps[2], "09:30:01"
    )
    future_action = _copy_data_with_availability(data, availability)
    with pytest.raises(ValueError, match="future backtest availability:split_ratio"):
        BacktestEngine().run(
            _spec(
                portfolio,
                cost,
                future_action,
                corporate_action_policy="explicit_panels",
            ),
            portfolio,
            cost,
            future_action,
        )


def test_backtest_view_detects_availability_mutation() -> None:
    _, _, data = _fixture(production_ready=True)
    data.known_at_panels["open_price"].iloc[0, 0] = pd.NaT
    with pytest.raises(RuntimeError, match="content changed"):
        data.verify_content()


def test_calibrated_cost_must_be_available_before_first_evaluation() -> None:
    timestamps, portfolio, data = _fixture(production_ready=True)
    valid = CostModel(
        cost_model_id="calibrated-production-cost",
        version="1",
        calibration=CostCalibration.CALIBRATED,
        commission_bps_buy=2.0,
        commission_bps_sell=2.0,
        minimum_commission=0.0,
        stamp_tax_bps_sell=5.0,
        fallback_half_spread_bps=0.0,
        slippage_bps=1.0,
        impact_coefficient=0.25,
        impact_exponent=1.0,
        annual_borrow_bps=0.0,
        require_observed_spread=True,
        require_observed_volatility=True,
        require_adv=True,
        calibrated_at=(timestamps[0] - pd.Timedelta(days=1)).isoformat(),
        calibration_start_at=(timestamps[0] - pd.Timedelta(days=120)).isoformat(),
        calibration_end_at=(timestamps[0] - pd.Timedelta(days=2)).isoformat(),
        calibration_artifact_hash="e" * 64,
        calibration_data_snapshot_id="f" * 64,
    )
    leaked_window = replace(
        valid,
        calibration_end_at=timestamps[0].isoformat(),
        calibrated_at=timestamps[0].isoformat(),
    )
    with pytest.raises(ValueError, match="calibration_end_at.*strictly earlier"):
        BacktestEngine().run(
            _spec(portfolio, leaked_window, data),
            portfolio,
            leaked_window,
            data,
        )

    unavailable_artifact = replace(
        valid,
        calibrated_at=timestamps[0].isoformat(),
    )
    with pytest.raises(ValueError, match="calibrated_at.*strictly earlier"):
        BacktestEngine().run(
            _spec(portfolio, unavailable_artifact, data),
            portfolio,
            unavailable_artifact,
            data,
        )
