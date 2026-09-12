from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from zoneinfo import ZoneInfo

import math

from alpha_research.core.hashing import hash_json, require_sha256


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    execution_id: str
    version: str
    signal_event: str = "signal_session_close"
    order_event: str = "next_session_open"
    valuation_event: str = "session_close"
    execution_lag_sessions: int = 1
    liquidity_observation_lag_sessions: int = 1
    risk_observation_lag_sessions: int = 1
    exchange_timezone: str = "Asia/Shanghai"
    order_time_local: str = "09:30:00"
    valuation_time_local: str = "15:00:00"
    maximum_participation_rate: float = 0.10
    lot_size: int = 1
    allow_short: bool = False
    enforce_t_plus_one: bool = True
    unfilled_policy: str = "recalculate_next_rebalance"
    missing_execution_policy: str = "fail_closed_no_fill"
    corporate_action_policy: str = "pre_adjusted"
    schema_version: str = "execution-spec/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "execution-spec/v1":
            raise ValueError("unsupported ExecutionSpec schema")
        if not self.execution_id.strip() or not self.version.strip():
            raise ValueError("execution id and version are required")
        if self.signal_event != "signal_session_close":
            raise ValueError("Phase 3 execution requires close-known signals")
        if (
            self.order_event != "next_session_open"
            or self.valuation_event != "session_close"
        ):
            raise ValueError(
                "Phase 3 execution supports next-open orders and close valuation"
            )
        try:
            ZoneInfo(self.exchange_timezone)
        except Exception as exc:
            raise ValueError("execution exchange_timezone is invalid") from exc
        try:
            order_time = time.fromisoformat(self.order_time_local)
            valuation_time = time.fromisoformat(self.valuation_time_local)
        except ValueError as exc:
            raise ValueError("execution local event time is invalid") from exc
        if order_time.tzinfo is not None or valuation_time.tzinfo is not None:
            raise ValueError("execution local event times must not include a timezone")
        if order_time >= valuation_time:
            raise ValueError("execution order time must precede valuation time")
        if self.execution_lag_sessions < 1:
            raise ValueError("execution lag must be at least one session")
        if self.liquidity_observation_lag_sessions < 1:
            raise ValueError(
                "next-open capacity must use liquidity known before the execution session"
            )
        if self.risk_observation_lag_sessions < 1:
            raise ValueError(
                "next-open costs must use volatility/spread known before execution"
            )
        if not 0 < self.maximum_participation_rate <= 1:
            raise ValueError("maximum participation must lie in (0,1]")
        if (
            not isinstance(self.lot_size, int)
            or isinstance(self.lot_size, bool)
            or self.lot_size <= 0
        ):
            raise ValueError("execution lot_size must be a positive integer")
        if not isinstance(self.allow_short, bool) or not isinstance(
            self.enforce_t_plus_one, bool
        ):
            raise TypeError("execution short/T+1 flags must be boolean")
        if self.unfilled_policy != "recalculate_next_rebalance":
            raise ValueError("unsupported execution unfilled policy")
        if self.missing_execution_policy != "fail_closed_no_fill":
            raise ValueError("execution must fail closed on unavailable instruments")
        if self.corporate_action_policy not in {"pre_adjusted", "explicit_panels"}:
            raise ValueError("unsupported corporate action policy")

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "execution_id": self.execution_id,
            "version": self.version,
            "signal_event": self.signal_event,
            "order_event": self.order_event,
            "valuation_event": self.valuation_event,
            "execution_lag_sessions": self.execution_lag_sessions,
            "liquidity_observation_lag_sessions": self.liquidity_observation_lag_sessions,
            "risk_observation_lag_sessions": self.risk_observation_lag_sessions,
            "exchange_timezone": self.exchange_timezone,
            "order_time_local": self.order_time_local,
            "valuation_time_local": self.valuation_time_local,
            "maximum_participation_rate": self.maximum_participation_rate,
            "lot_size": self.lot_size,
            "allow_short": self.allow_short,
            "enforce_t_plus_one": self.enforce_t_plus_one,
            "unfilled_policy": self.unfilled_policy,
            "missing_execution_policy": self.missing_execution_policy,
            "corporate_action_policy": self.corporate_action_policy,
        }


@dataclass(frozen=True, slots=True)
class BacktestSpec:
    backtest_id: str
    version: str
    portfolio_spec_hash: str
    portfolio_weights_hash: str
    portfolio_result_hash: str
    data_view_hash: str
    cost_model_hash: str
    execution: ExecutionSpec
    initial_capital: float
    annualization_sessions: int = 252
    risk_free_rate: float = 0.0
    accounting_policy: str = "dual_gross_net_same_fills"
    schema_version: str = "backtest-spec/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "backtest-spec/v1":
            raise ValueError("unsupported BacktestSpec schema")
        if not self.backtest_id.strip() or not self.version.strip():
            raise ValueError("backtest id and version are required")
        for name in (
            "portfolio_spec_hash",
            "portfolio_weights_hash",
            "portfolio_result_hash",
            "data_view_hash",
            "cost_model_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"backtest {name}")
        if not math.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("backtest initial capital must be positive")
        if (
            not isinstance(self.annualization_sessions, int)
            or self.annualization_sessions <= 0
        ):
            raise ValueError("backtest annualization_sessions must be positive")
        if not math.isfinite(self.risk_free_rate):
            raise ValueError("backtest risk-free rate must be finite")
        if self.accounting_policy != "dual_gross_net_same_fills":
            raise ValueError("unsupported backtest accounting policy")

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "backtest_id": self.backtest_id,
            "version": self.version,
            "portfolio_spec_hash": self.portfolio_spec_hash,
            "portfolio_weights_hash": self.portfolio_weights_hash,
            "portfolio_result_hash": self.portfolio_result_hash,
            "data_view_hash": self.data_view_hash,
            "cost_model_hash": self.cost_model_hash,
            "execution": self.execution.to_dict(),
            "initial_capital": self.initial_capital,
            "annualization_sessions": self.annualization_sessions,
            "risk_free_rate": self.risk_free_rate,
            "accounting_policy": self.accounting_policy,
        }


__all__ = ["BacktestSpec", "ExecutionSpec"]
