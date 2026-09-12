from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from alpha_research.backtest.data import BacktestDataView
from alpha_research.backtest.metrics import MetricValue, performance_metrics
from alpha_research.backtest.spec import BacktestSpec
from alpha_research.core.hashing import hash_frame, hash_json
from alpha_research.costs import CostEstimator, CostModel, TradeSide
from alpha_research.portfolio import PortfolioResult


@dataclass(frozen=True, slots=True)
class BacktestResult:
    backtest_spec_hash: str
    data_view_hash: str
    portfolio_result_hash: str
    nav_hash: str
    positions_hash: str
    cash_hash: str
    orders_hash: str
    fills_hash: str
    cost_breakdown_hash: str
    metrics_hash: str
    metric_metadata_hash: str
    production_ready: bool
    nav: pd.DataFrame
    positions: pd.DataFrame
    cash: pd.DataFrame
    orders: pd.DataFrame
    fills: pd.DataFrame
    cost_breakdown: pd.DataFrame
    metrics: Mapping[str, MetricValue]
    metric_metadata: pd.DataFrame

    def __post_init__(self) -> None:
        frames = {
            "nav": pd.DataFrame(self.nav).copy(deep=True),
            "positions": pd.DataFrame(self.positions).copy(deep=True),
            "cash": pd.DataFrame(self.cash).copy(deep=True),
            "orders": pd.DataFrame(self.orders).copy(deep=True),
            "fills": pd.DataFrame(self.fills).copy(deep=True),
            "cost_breakdown": pd.DataFrame(self.cost_breakdown).copy(deep=True),
            "metric_metadata": pd.DataFrame(self.metric_metadata).copy(deep=True),
        }
        for name, frame in frames.items():
            if hash_frame(frame) != getattr(self, f"{name}_hash"):
                raise ValueError(f"backtest {name} hash differs")
            object.__setattr__(self, name, frame)
        metrics = dict(self.metrics)
        if hash_json(metrics) != self.metrics_hash:
            raise ValueError("backtest metrics hash differs")
        object.__setattr__(self, "metrics", MappingProxyType(metrics))

    @property
    def content_hash(self) -> str:
        return hash_json(
            {
                "backtest_spec_hash": self.backtest_spec_hash,
                "data_view_hash": self.data_view_hash,
                "portfolio_result_hash": self.portfolio_result_hash,
                "nav_hash": self.nav_hash,
                "positions_hash": self.positions_hash,
                "cash_hash": self.cash_hash,
                "orders_hash": self.orders_hash,
                "fills_hash": self.fills_hash,
                "cost_breakdown_hash": self.cost_breakdown_hash,
                "metrics_hash": self.metrics_hash,
                "metric_metadata_hash": self.metric_metadata_hash,
                "production_ready": self.production_ready,
            }
        )

    def verify_content(self) -> None:
        for name in (
            "nav",
            "positions",
            "cash",
            "orders",
            "fills",
            "cost_breakdown",
            "metric_metadata",
        ):
            if hash_frame(getattr(self, name)) != getattr(self, f"{name}_hash"):
                raise RuntimeError(f"backtest {name} changed after construction")
        if hash_json(dict(self.metrics)) != self.metrics_hash:
            raise RuntimeError("backtest metrics changed after construction")


class BacktestEngine:
    def run(
        self,
        spec: BacktestSpec,
        portfolio: PortfolioResult,
        cost_model: CostModel,
        data: BacktestDataView,
    ) -> BacktestResult:
        portfolio.verify_content()
        data.verify_content()
        if spec.portfolio_spec_hash != portfolio.portfolio_spec_hash:
            raise ValueError("backtest portfolio specification binding differs")
        if spec.portfolio_weights_hash != portfolio.weights_hash:
            raise ValueError("backtest portfolio weights binding differs")
        if spec.portfolio_result_hash != portfolio.content_hash:
            raise ValueError("backtest portfolio result binding differs")
        if spec.data_view_hash != data.view_hash:
            raise ValueError("backtest data view binding differs")
        if spec.cost_model_hash != cost_model.content_hash:
            raise ValueError("backtest cost model binding differs")
        cost_model.assert_available_before(data.open_price.index[0])
        weights = portfolio.weights
        if len(weights.index.difference(data.open_price.index)):
            raise ValueError("portfolio signals are outside backtest calendar")
        if len(weights.columns.difference(data.open_price.columns)):
            raise ValueError("portfolio securities are outside backtest data")
        if not spec.execution.allow_short and (weights < -1e-14).any().any():
            raise ValueError("backtest execution forbids requested short positions")
        if spec.execution.corporate_action_policy == "pre_adjusted":
            if (
                not (data.split_ratio == 1.0).all().all()
                or not (data.cash_dividend == 0.0).all().all()
            ):
                raise ValueError(
                    "pre-adjusted execution cannot apply explicit corporate actions"
                )
        return self._simulate(spec, portfolio, cost_model, data)

    @staticmethod
    def _simulate(
        spec: BacktestSpec,
        portfolio: PortfolioResult,
        cost_model: CostModel,
        data: BacktestDataView,
    ) -> BacktestResult:
        index = data.open_price.index
        securities = data.open_price.columns
        positions = pd.Series(0.0, index=securities)
        gross_cash = float(spec.initial_capital)
        net_cash = float(spec.initial_capital)
        previous_close = pd.Series(np.nan, index=securities)
        previous_close_known_at = pd.Series(pd.NaT, index=securities, dtype=object)
        cumulative_cost = 0.0
        position_rows: list[pd.Series] = []
        cash_rows: list[dict[str, float]] = []
        nav_rows: list[dict[str, float]] = []
        order_rows: list[dict[str, object]] = []
        fill_rows: list[dict[str, object]] = []
        cost_rows: list[dict[str, float | pd.Timestamp]] = []
        estimator = CostEstimator()
        for session_offset, timestamp in enumerate(index):
            execution_event_timestamp = _session_event_timestamp(
                timestamp,
                local_time=spec.execution.order_time_local,
                timezone=spec.execution.exchange_timezone,
            )
            valuation_event_timestamp = _session_event_timestamp(
                timestamp,
                local_time=spec.execution.valuation_time_local,
                timezone=spec.execution.exchange_timezone,
            )
            open_price = _positive(data.open_price.loc[timestamp])
            close_price = _positive(data.close_price.loc[timestamp])
            if spec.execution.corporate_action_policy == "explicit_panels":
                held_before_actions = positions != 0
                for security in securities[held_before_actions]:
                    data.require_available(
                        "cash_dividend",
                        timestamp,
                        security,
                        as_of=execution_event_timestamp,
                    )
                    data.require_available(
                        "split_ratio",
                        timestamp,
                        security,
                        as_of=execution_event_timestamp,
                    )
                    raw_dividend = float(data.cash_dividend.loc[timestamp, security])
                    raw_split = float(data.split_ratio.loc[timestamp, security])
                    if data.production_ready and (
                        not np.isfinite(raw_dividend) or raw_dividend < 0
                    ):
                        raise ValueError(
                            "held position has invalid cash-dividend input"
                        )
                    if data.production_ready and (
                        not np.isfinite(raw_split) or raw_split <= 0
                    ):
                        raise ValueError("held position has invalid split-ratio input")
                dividend = _nonnegative(data.cash_dividend.loc[timestamp]).fillna(0.0)
                dividend_cash = float((positions * dividend).sum())
                gross_cash += dividend_cash
                net_cash += dividend_cash
                split = _positive(data.split_ratio.loc[timestamp]).fillna(1.0)
                positions *= split
            mark_open = open_price.fillna(previous_close)
            held_without_mark = (positions != 0) & ~np.isfinite(mark_open)
            if held_without_mark.any():
                raise ValueError("held position lacks a non-future valuation price")
            for security in securities[positions != 0]:
                if np.isfinite(open_price[security]):
                    data.require_available(
                        "open_price",
                        timestamp,
                        security,
                        as_of=execution_event_timestamp,
                    )
                else:
                    _require_lineage_available(
                        previous_close_known_at[security],
                        as_of=execution_event_timestamp,
                        field="previous_close",
                        observation_timestamp=timestamp,
                        security=security,
                        production_ready=data.production_ready,
                    )
            gross_nav_open = float(
                gross_cash + (positions * mark_open.fillna(0.0)).sum()
            )
            session_cost = {
                "commission": 0.0,
                "stamp_tax": 0.0,
                "spread": 0.0,
                "slippage": 0.0,
                "market_impact": 0.0,
                "borrow": 0.0,
            }
            signal_offset = session_offset - spec.execution.execution_lag_sessions
            if signal_offset >= 0:
                signal_timestamp = index[signal_offset]
                liquidity_offset = (
                    session_offset - spec.execution.liquidity_observation_lag_sessions
                )
                risk_offset = (
                    session_offset - spec.execution.risk_observation_lag_sessions
                )
                if liquidity_offset < 0 or risk_offset < 0:  # pragma: no cover
                    raise RuntimeError(
                        "execution inputs have no pre-session observation"
                    )
                liquidity_timestamp = index[liquidity_offset]
                risk_timestamp = index[risk_offset]
                if signal_timestamp in portfolio.weights.index:
                    target_weight = (
                        portfolio.weights.reindex(columns=securities)
                        .loc[signal_timestamp]
                        .fillna(0.0)
                    )
                    active_for_target = (positions != 0) | (target_weight != 0)
                    for security in securities[active_for_target]:
                        data.require_available(
                            "open_price",
                            timestamp,
                            security,
                            as_of=execution_event_timestamp,
                        )
                    target_value = target_weight * gross_nav_open
                    target_shares = target_value.div(open_price)
                    if not spec.execution.allow_short:
                        target_shares = target_shares.clip(lower=0.0)
                    desired = target_shares - positions
                    for security in securities:
                        requested = float(desired[security])
                        if not np.isfinite(requested) or abs(requested) < 1e-12:
                            continue
                        order_id = f"{timestamp.isoformat()}::{security}"
                        side = TradeSide.BUY if requested > 0 else TradeSide.SELL
                        data.require_available(
                            "execution_tradability",
                            timestamp,
                            security,
                            as_of=execution_event_timestamp,
                        )
                        tradable = bool(
                            data.execution_tradability.loc[timestamp, security]
                        )
                        price = (
                            float(open_price[security])
                            if np.isfinite(open_price[security])
                            else np.nan
                        )
                        reason = "filled"
                        filled = 0.0
                        capacity_shares = 0.0
                        if not tradable:
                            data.require_available(
                                "execution_reason",
                                timestamp,
                                security,
                                as_of=execution_event_timestamp,
                            )
                            reason = str(data.execution_reason.loc[timestamp, security])
                        elif not np.isfinite(price) or price <= 0:
                            reason = "missing_open_price"
                        else:
                            # Full-session amount, ADV, close volatility, and
                            # close spread are not known at the next open.  The
                            # execution contract therefore binds explicit
                            # pre-session lags and never reads today's totals.
                            data.require_available(
                                "traded_amount",
                                liquidity_timestamp,
                                security,
                                as_of=execution_event_timestamp,
                            )
                            traded_amount = float(
                                data.traded_amount.loc[liquidity_timestamp, security]
                            )
                            if not np.isfinite(traded_amount) or traded_amount <= 0:
                                reason = "missing_or_zero_traded_amount"
                            else:
                                capacity_shares = (
                                    traded_amount
                                    * spec.execution.maximum_participation_rate
                                    / price
                                )
                                filled = np.sign(requested) * min(
                                    abs(requested), capacity_shares
                                )
                                if (
                                    filled < 0
                                    and spec.execution.enforce_t_plus_one
                                    and not spec.execution.allow_short
                                ):
                                    filled = -min(
                                        abs(filled),
                                        max(0.0, float(positions[security])),
                                    )
                                filled = _round_lot(filled, spec.execution.lot_size)
                                if abs(filled) < 1e-12:
                                    reason = "below_lot_or_capacity"
                                elif abs(filled) + 1e-12 < abs(requested):
                                    reason = "partial_fill"
                        order_rows.append(
                            {
                                "order_id": order_id,
                                "signal_timestamp": signal_timestamp,
                                "execution_session_timestamp": timestamp,
                                "execution_timestamp": execution_event_timestamp,
                                "security": security,
                                "side": side.value,
                                "requested_shares": requested,
                                "capacity_shares": capacity_shares,
                                "liquidity_observation_timestamp": liquidity_timestamp,
                                "risk_observation_timestamp": risk_timestamp,
                                "status": reason,
                            }
                        )
                        if abs(filled) < 1e-12:
                            continue
                        notional = abs(filled * price)
                        data.require_available(
                            "adv_amount",
                            liquidity_timestamp,
                            security,
                            as_of=execution_event_timestamp,
                        )
                        data.require_available(
                            "volatility",
                            risk_timestamp,
                            security,
                            as_of=execution_event_timestamp,
                        )
                        data.require_available(
                            "half_spread_bps",
                            risk_timestamp,
                            security,
                            as_of=execution_event_timestamp,
                        )
                        adv = float(data.adv_amount.loc[liquidity_timestamp, security])
                        participation = (
                            notional / adv if np.isfinite(adv) and adv > 0 else None
                        )
                        volatility = _optional(
                            data.volatility.loc[risk_timestamp, security]
                        )
                        spread = _optional(
                            data.half_spread_bps.loc[risk_timestamp, security]
                        )
                        cost = estimator.estimate_trade(
                            cost_model,
                            side=side,
                            notional=notional,
                            participation_rate=participation,
                            volatility=volatility,
                            observed_half_spread_bps=spread,
                        )
                        signed_notional = filled * price
                        gross_cash -= signed_notional
                        net_cash -= signed_notional + cost.total
                        positions[security] += filled
                        for name in (
                            "commission",
                            "stamp_tax",
                            "spread",
                            "slippage",
                            "market_impact",
                        ):
                            session_cost[name] += float(getattr(cost, name))
                        fill_rows.append(
                            {
                                "order_id": order_id,
                                "execution_session_timestamp": timestamp,
                                "execution_timestamp": execution_event_timestamp,
                                "security": security,
                                "side": side.value,
                                "filled_shares": filled,
                                "fill_price": price,
                                "filled_notional": notional,
                                "participation_rate": participation,
                                "liquidity_observation_timestamp": liquidity_timestamp,
                                "risk_observation_timestamp": risk_timestamp,
                                "total_cost": cost.total,
                                "status": reason,
                            }
                        )
            mark_close = close_price.fillna(open_price).fillna(previous_close)
            held_without_close = (positions != 0) & ~np.isfinite(mark_close)
            if held_without_close.any():
                raise ValueError("held position lacks close valuation price")
            mark_close_known_at = pd.Series(pd.NaT, index=securities, dtype=object)
            for security in securities[positions != 0]:
                if np.isfinite(close_price[security]):
                    mark_close_known_at[security] = data.require_available(
                        "close_price",
                        timestamp,
                        security,
                        as_of=valuation_event_timestamp,
                    )
                elif np.isfinite(open_price[security]):
                    mark_close_known_at[security] = data.require_available(
                        "open_price",
                        timestamp,
                        security,
                        as_of=valuation_event_timestamp,
                    )
                else:
                    _require_lineage_available(
                        previous_close_known_at[security],
                        as_of=valuation_event_timestamp,
                        field="previous_close",
                        observation_timestamp=timestamp,
                        security=security,
                        production_ready=data.production_ready,
                    )
                    mark_close_known_at[security] = previous_close_known_at[security]
            short_value = float(
                (-positions.clip(upper=0.0) * mark_close.fillna(0.0)).sum()
            )
            borrow = estimator.estimate_daily_borrow(
                cost_model, short_market_value=short_value
            )
            net_cash -= borrow
            session_cost["borrow"] = borrow
            transaction_cost = sum(
                session_cost[name]
                for name in (
                    "commission",
                    "stamp_tax",
                    "spread",
                    "slippage",
                    "market_impact",
                )
            )
            total_cost = transaction_cost + borrow
            cumulative_cost += total_cost
            market_value = float((positions * mark_close.fillna(0.0)).sum())
            gross_nav = gross_cash + market_value
            net_nav = net_cash + market_value
            if gross_nav <= 0 or net_nav <= 0:
                raise ValueError("backtest NAV became non-positive")
            if not np.isclose(
                gross_nav - net_nav,
                cumulative_cost,
                rtol=1e-10,
                atol=max(1e-8, spec.initial_capital * 1e-12),
            ):
                raise RuntimeError("gross/net/cost accounting invariant failed")
            position_rows.append(positions.copy())
            cash_rows.append(
                {
                    "gross_cash": gross_cash,
                    "net_cash": net_cash,
                    "position_market_value": market_value,
                }
            )
            nav_rows.append(
                {
                    "gross_nav": gross_nav,
                    "net_nav": net_nav,
                    "transaction_cost": transaction_cost,
                    "borrow_cost": borrow,
                    "total_cost": total_cost,
                    "cumulative_cost": cumulative_cost,
                }
            )
            cost_rows.append({"timestamp": timestamp, **session_cost})
            previous_close = mark_close
            previous_close_known_at = mark_close_known_at
        positions_frame = pd.DataFrame(position_rows, index=index, columns=securities)
        cash_frame = pd.DataFrame(cash_rows, index=index)
        nav_frame = pd.DataFrame(nav_rows, index=index)
        nav_frame["gross_return"] = nav_frame["gross_nav"].pct_change(fill_method=None)
        nav_frame["net_return"] = nav_frame["net_nav"].pct_change(fill_method=None)
        nav_frame["benchmark_return"] = data.benchmark_close.pct_change(
            fill_method=None
        )
        nav_frame["gross_excess_return"] = (
            nav_frame["gross_return"] - nav_frame["benchmark_return"]
        )
        nav_frame["net_excess_return"] = (
            nav_frame["net_return"] - nav_frame["benchmark_return"]
        )
        orders = pd.DataFrame(order_rows, columns=_ORDER_COLUMNS)
        fills = pd.DataFrame(fill_rows, columns=_FILL_COLUMNS)
        costs = pd.DataFrame(cost_rows).set_index("timestamp")
        metrics, metric_metadata = performance_metrics(
            nav_frame,
            fills,
            annualization=spec.annualization_sessions,
            risk_free_rate=spec.risk_free_rate,
        )
        return BacktestResult(
            backtest_spec_hash=spec.content_hash,
            data_view_hash=data.view_hash,
            portfolio_result_hash=portfolio.content_hash,
            nav_hash=hash_frame(nav_frame),
            positions_hash=hash_frame(positions_frame),
            cash_hash=hash_frame(cash_frame),
            orders_hash=hash_frame(orders),
            fills_hash=hash_frame(fills),
            cost_breakdown_hash=hash_frame(costs),
            metrics_hash=hash_json(metrics),
            metric_metadata_hash=hash_frame(metric_metadata),
            production_ready=(
                portfolio.production_ready
                and cost_model.production_ready
                and data.production_ready
            ),
            nav=nav_frame,
            positions=positions_frame,
            cash=cash_frame,
            orders=orders,
            fills=fills,
            cost_breakdown=costs,
            metrics=metrics,
            metric_metadata=metric_metadata,
        )


_ORDER_COLUMNS = [
    "order_id",
    "signal_timestamp",
    "execution_session_timestamp",
    "execution_timestamp",
    "security",
    "side",
    "requested_shares",
    "capacity_shares",
    "liquidity_observation_timestamp",
    "risk_observation_timestamp",
    "status",
]
_FILL_COLUMNS = [
    "order_id",
    "execution_session_timestamp",
    "execution_timestamp",
    "security",
    "side",
    "filled_shares",
    "fill_price",
    "filled_notional",
    "participation_rate",
    "liquidity_observation_timestamp",
    "risk_observation_timestamp",
    "total_cost",
    "status",
]


def _positive(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return numeric.where(numeric > 0).astype(float)


def _nonnegative(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return numeric.where(numeric >= 0).astype(float)


def _optional(value: object) -> float | None:
    numeric = float(str(value))
    return numeric if np.isfinite(numeric) else None


def _round_lot(shares: float, lot_size: int) -> float:
    return float(np.sign(shares) * np.floor(abs(shares) / lot_size) * lot_size)


def _session_event_timestamp(
    session_timestamp: pd.Timestamp,
    *,
    local_time: str,
    timezone: str,
) -> pd.Timestamp:
    session = pd.Timestamp(session_timestamp)
    if session.tzinfo is None:  # guarded by BacktestDataView
        raise ValueError("backtest session timestamp must be timezone-aware")
    local_date = session.tz_convert(timezone).date().isoformat()
    return pd.Timestamp(f"{local_date}T{local_time}").tz_localize(timezone)


def _require_lineage_available(
    available_at: object,
    *,
    as_of: pd.Timestamp,
    field: str,
    observation_timestamp: pd.Timestamp,
    security: object,
    production_ready: bool,
) -> None:
    if not production_ready:
        return
    if pd.isna(available_at):
        raise ValueError(
            f"missing backtest availability:{field}:"
            f"{observation_timestamp}:{security}"
        )
    timestamp = pd.Timestamp(available_at)
    if timestamp.tzinfo is None:
        raise ValueError(f"backtest availability must be timezone-aware:{field}")
    if timestamp > as_of:
        raise ValueError(
            f"future backtest availability:{field}:"
            f"available={timestamp.isoformat()}:use={as_of.isoformat()}:"
            f"observation={observation_timestamp}:security={security}"
        )


__all__ = ["BacktestEngine", "BacktestResult"]
