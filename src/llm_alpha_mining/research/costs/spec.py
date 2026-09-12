from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import math
import pandas as pd

from llm_alpha_mining.research.core.hashing import hash_json, require_sha256


class CostCalibration(str, Enum):
    ZERO = "zero"
    PROXY = "proxy"
    CALIBRATED = "calibrated"


class TradeSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class CostModel:
    cost_model_id: str
    version: str
    calibration: CostCalibration | str
    commission_bps_buy: float
    commission_bps_sell: float
    minimum_commission: float
    stamp_tax_bps_sell: float
    fallback_half_spread_bps: float
    slippage_bps: float
    impact_coefficient: float
    impact_exponent: float
    annual_borrow_bps: float
    require_observed_spread: bool
    require_observed_volatility: bool
    require_adv: bool
    calibrated_at: str | None = None
    calibration_start_at: str | None = None
    calibration_end_at: str | None = None
    calibration_artifact_hash: str | None = None
    calibration_data_snapshot_id: str | None = None
    schema_version: str = "cost-model/v2"

    def __post_init__(self) -> None:
        object.__setattr__(self, "calibration", CostCalibration(self.calibration))
        if self.schema_version != "cost-model/v2":
            raise ValueError("unsupported CostModel schema")
        if not self.cost_model_id.strip() or not self.version.strip():
            raise ValueError("cost model id and version are required")
        nonnegative = (
            "commission_bps_buy",
            "commission_bps_sell",
            "minimum_commission",
            "stamp_tax_bps_sell",
            "fallback_half_spread_bps",
            "slippage_bps",
            "impact_coefficient",
            "annual_borrow_bps",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"cost model {name} must be finite and non-negative")
        if not math.isfinite(self.impact_exponent) or self.impact_exponent <= 0:
            raise ValueError("cost impact_exponent must be positive")
        flags = (
            self.require_observed_spread,
            self.require_observed_volatility,
            self.require_adv,
        )
        if not all(isinstance(item, bool) for item in flags):
            raise TypeError("cost data requirements must be boolean")
        calibration = self.calibration
        if not isinstance(calibration, CostCalibration):  # pragma: no cover
            raise RuntimeError("cost calibration was not normalized")
        if calibration is CostCalibration.ZERO:
            numeric = [float(getattr(self, name)) for name in nonnegative]
            if any(value != 0 for value in numeric) or any(flags):
                raise ValueError("zero cost model must have zero parameters")
        if calibration is CostCalibration.CALIBRATED:
            if self.calibrated_at is None:
                raise ValueError("calibrated cost model requires calibrated_at")
            if self.calibration_start_at is None:
                raise ValueError("calibrated cost model requires calibration_start_at")
            if self.calibration_end_at is None:
                raise ValueError("calibrated cost model requires calibration_end_at")
            if self.calibration_artifact_hash is None:
                raise ValueError(
                    "calibrated cost model requires a calibration artifact"
                )
            if self.calibration_data_snapshot_id is None:
                raise ValueError(
                    "calibrated cost model requires a calibration data snapshot"
                )
            require_sha256(
                self.calibration_artifact_hash,
                name="cost calibration_artifact_hash",
            )
            require_sha256(
                self.calibration_data_snapshot_id,
                name="cost calibration_data_snapshot_id",
            )
            if not all(flags):
                raise ValueError(
                    "calibrated cost model must require observed microstructure inputs"
                )
            calibrated_at = _aware_timestamp(
                self.calibrated_at, name="cost calibrated_at"
            )
            calibration_start_at = _aware_timestamp(
                self.calibration_start_at, name="cost calibration_start_at"
            )
            calibration_end_at = _aware_timestamp(
                self.calibration_end_at, name="cost calibration_end_at"
            )
            if calibration_start_at >= calibration_end_at:
                raise ValueError(
                    "cost calibration_start_at must be earlier than calibration_end_at"
                )
            if calibration_end_at > calibrated_at:
                raise ValueError("cost calibrated_at cannot precede calibration_end_at")
            object.__setattr__(self, "calibrated_at", calibrated_at.isoformat())
            object.__setattr__(
                self, "calibration_start_at", calibration_start_at.isoformat()
            )
            object.__setattr__(
                self, "calibration_end_at", calibration_end_at.isoformat()
            )
        elif any(
            value is not None
            for value in (
                self.calibrated_at,
                self.calibration_start_at,
                self.calibration_end_at,
                self.calibration_artifact_hash,
                self.calibration_data_snapshot_id,
            )
        ):
            raise ValueError(
                "non-calibrated cost model cannot bind calibration evidence"
            )

    @classmethod
    def zero(cls) -> "CostModel":
        return cls(
            cost_model_id="zero-cost-reference",
            version="1",
            calibration=CostCalibration.ZERO,
            commission_bps_buy=0.0,
            commission_bps_sell=0.0,
            minimum_commission=0.0,
            stamp_tax_bps_sell=0.0,
            fallback_half_spread_bps=0.0,
            slippage_bps=0.0,
            impact_coefficient=0.0,
            impact_exponent=1.0,
            annual_borrow_bps=0.0,
            require_observed_spread=False,
            require_observed_volatility=False,
            require_adv=False,
        )

    @property
    def production_ready(self) -> bool:
        return self.calibration is CostCalibration.CALIBRATED

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def assert_available_before(self, evaluation_start_at: object) -> None:
        """Fail closed when calibrated costs were unavailable at evaluation start.

        A calibrated model is point-in-time admissible only when both the last
        calibration observation and the completed calibration artifact predate
        the first evaluated timestamp.  Proxy and zero-cost research models do
        not carry calibration evidence and remain unaffected.
        """

        if self.calibration is not CostCalibration.CALIBRATED:
            return
        evaluation_start = _aware_timestamp(
            evaluation_start_at, name="cost evaluation_start_at"
        )
        calibration_end = _aware_timestamp(
            self.calibration_end_at, name="cost calibration_end_at"
        )
        calibrated_at = _aware_timestamp(self.calibrated_at, name="cost calibrated_at")
        if calibration_end >= evaluation_start:
            raise ValueError(
                "cost calibration_end_at must be strictly earlier than "
                "the first evaluation timestamp"
            )
        if calibrated_at >= evaluation_start:
            raise ValueError(
                "cost calibrated_at must be strictly earlier than "
                "the first evaluation timestamp"
            )

    def to_dict(self) -> dict[str, object]:
        calibration = self.calibration
        if not isinstance(calibration, CostCalibration):  # pragma: no cover
            raise RuntimeError("cost calibration was not normalized")
        return {
            "schema_version": self.schema_version,
            "cost_model_id": self.cost_model_id,
            "version": self.version,
            "calibration": calibration.value,
            "commission_bps_buy": self.commission_bps_buy,
            "commission_bps_sell": self.commission_bps_sell,
            "minimum_commission": self.minimum_commission,
            "stamp_tax_bps_sell": self.stamp_tax_bps_sell,
            "fallback_half_spread_bps": self.fallback_half_spread_bps,
            "slippage_bps": self.slippage_bps,
            "impact_coefficient": self.impact_coefficient,
            "impact_exponent": self.impact_exponent,
            "annual_borrow_bps": self.annual_borrow_bps,
            "require_observed_spread": self.require_observed_spread,
            "require_observed_volatility": self.require_observed_volatility,
            "require_adv": self.require_adv,
            "calibrated_at": self.calibrated_at,
            "calibration_start_at": self.calibration_start_at,
            "calibration_end_at": self.calibration_end_at,
            "calibration_artifact_hash": self.calibration_artifact_hash,
            "calibration_data_snapshot_id": self.calibration_data_snapshot_id,
        }


def _aware_timestamp(value: object, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    commission: float
    stamp_tax: float
    spread: float
    slippage: float
    market_impact: float
    borrow: float = 0.0

    @property
    def total(self) -> float:
        return float(
            self.commission
            + self.stamp_tax
            + self.spread
            + self.slippage
            + self.market_impact
            + self.borrow
        )


class CostEstimator:
    def estimate_trade(
        self,
        model: CostModel,
        *,
        side: TradeSide | str,
        notional: float,
        participation_rate: float | None,
        volatility: float | None,
        observed_half_spread_bps: float | None,
    ) -> CostBreakdown:
        side = TradeSide(side)
        if not math.isfinite(notional) or notional < 0:
            raise ValueError("trade notional must be finite and non-negative")
        if notional == 0:
            return CostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0)
        if model.calibration is CostCalibration.ZERO:
            return CostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0)
        participation = _input(
            participation_rate,
            required=model.require_adv,
            name="participation_rate",
            minimum=0.0,
        )
        volatility_value = _input(
            volatility,
            required=model.require_observed_volatility,
            name="volatility",
            minimum=0.0,
        )
        if observed_half_spread_bps is None:
            if model.require_observed_spread:
                raise ValueError("observed spread is required by the cost model")
            spread_bps = model.fallback_half_spread_bps
        else:
            spread_bps = _input(
                observed_half_spread_bps,
                required=True,
                name="observed_half_spread_bps",
                minimum=0.0,
            )
        commission_bps = (
            model.commission_bps_buy
            if side is TradeSide.BUY
            else model.commission_bps_sell
        )
        commission = max(notional * commission_bps / 10_000.0, model.minimum_commission)
        stamp = (
            notional * model.stamp_tax_bps_sell / 10_000.0
            if side is TradeSide.SELL
            else 0.0
        )
        impact_rate = (
            model.impact_coefficient
            * (participation**model.impact_exponent)
            * volatility_value
        )
        return CostBreakdown(
            commission=float(commission),
            stamp_tax=float(stamp),
            spread=float(notional * spread_bps / 10_000.0),
            slippage=float(notional * model.slippage_bps / 10_000.0),
            market_impact=float(notional * impact_rate),
        )

    def estimate_daily_borrow(
        self, model: CostModel, *, short_market_value: float
    ) -> float:
        if not math.isfinite(short_market_value) or short_market_value < 0:
            raise ValueError("short market value must be finite and non-negative")
        return float(short_market_value * model.annual_borrow_bps / 10_000.0 / 252.0)


def _input(
    value: float | None,
    *,
    required: bool,
    name: str,
    minimum: float,
) -> float:
    if value is None:
        if required:
            raise ValueError(f"{name} is required by the cost model")
        return 0.0
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < minimum:
        raise ValueError(f"{name} is invalid")
    return numeric


__all__ = [
    "CostBreakdown",
    "CostCalibration",
    "CostEstimator",
    "CostModel",
    "TradeSide",
]
