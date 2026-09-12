from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from llm_alpha_mining.research.core.hashing import hash_frame, hash_json, require_sha256


@dataclass(frozen=True, slots=True)
class BacktestDataView:
    snapshot_id: str
    security_contract_hash: str
    benchmark_id: str
    benchmark_snapshot_id: str
    benchmark_close: pd.Series
    open_price: pd.DataFrame
    close_price: pd.DataFrame
    traded_amount: pd.DataFrame
    adv_amount: pd.DataFrame
    volatility: pd.DataFrame
    half_spread_bps: pd.DataFrame
    execution_tradability: pd.DataFrame
    execution_reason: pd.DataFrame
    split_ratio: pd.DataFrame
    cash_dividend: pd.DataFrame
    availability_hashes: Mapping[str, str]
    known_at_panels: Mapping[str, pd.DataFrame]
    production_ready: bool
    view_hash: str
    schema_version: str = "backtest-data-view/v2"

    def __post_init__(self) -> None:
        if self.schema_version != "backtest-data-view/v2":
            raise ValueError("unsupported BacktestDataView schema")
        require_sha256(self.snapshot_id, name="backtest snapshot_id")
        require_sha256(
            self.security_contract_hash, name="backtest security_contract_hash"
        )
        if not self.benchmark_id.strip():
            raise ValueError("backtest benchmark_id is required")
        require_sha256(
            self.benchmark_snapshot_id, name="backtest benchmark_snapshot_id"
        )
        panels = self._copied_panels()
        reference = panels["open_price"]
        if (
            not isinstance(reference.index, pd.DatetimeIndex)
            or reference.index.tz is None
        ):
            raise ValueError("backtest timestamps must be timezone-aware")
        if not reference.index.is_unique or not reference.index.is_monotonic_increasing:
            raise ValueError("backtest timestamps must be sorted and unique")
        for name, panel in panels.items():
            if not panel.index.equals(reference.index) or not panel.columns.equals(
                reference.columns
            ):
                raise ValueError(f"backtest data axes differ:{name}")
        benchmark = pd.to_numeric(
            pd.Series(self.benchmark_close).copy(deep=True), errors="coerce"
        ).astype(float)
        if not benchmark.index.equals(reference.index):
            raise ValueError("backtest benchmark calendar differs from assets")
        if (~np.isfinite(benchmark) | (benchmark <= 0)).any():
            raise ValueError("backtest benchmark prices must be positive and complete")
        raw_tradability = panels["execution_tradability"]
        if raw_tradability.isna().any().any():
            raise ValueError("backtest execution_tradability must be complete")
        if (
            not raw_tradability.map(lambda value: isinstance(value, (bool, np.bool_)))
            .all()
            .all()
        ):
            raise TypeError("backtest execution_tradability must contain booleans")
        panels["execution_tradability"] = raw_tradability.astype(bool)
        reason = panels["execution_reason"].astype(str)
        allowed_reasons = {
            "tradable",
            "not_tradable",
            "suspended",
            "limit_up",
            "limit_down",
            "security_ineligible",
        }
        observed_reasons = set(reason.stack().unique())
        if not observed_reasons.issubset(allowed_reasons):
            raise ValueError("backtest execution_reason contains an unsupported value")
        tradable_reason = reason == "tradable"
        if not tradable_reason.equals(panels["execution_tradability"]):
            raise ValueError("backtest execution_reason differs from tradability mask")
        panels["execution_reason"] = reason
        availability = {
            name: pd.DataFrame(panel).copy(deep=True)
            for name, panel in sorted(self.known_at_panels.items())
        }
        hashes = dict(sorted(self.availability_hashes.items()))
        unknown = set(availability).difference(_AVAILABILITY_FIELDS)
        if unknown:
            raise ValueError(
                "backtest availability contains unsupported fields:"
                + ",".join(sorted(unknown))
            )
        if set(hashes) != set(availability):
            raise ValueError("backtest availability panel/hash names differ")
        if self.production_ready and set(availability) != set(_AVAILABILITY_FIELDS):
            missing = set(_AVAILABILITY_FIELDS).difference(availability)
            raise ValueError(
                "production backtest data requires per-observation availability:"
                + ",".join(sorted(missing))
            )
        for name, known_at in availability.items():
            if not known_at.index.equals(
                reference.index
            ) or not known_at.columns.equals(reference.columns):
                raise ValueError(f"backtest availability axes differ:{name}")
            _validate_availability_panel(known_at, name=name)
            require_sha256(hashes[name], name=f"backtest availability hash:{name}")
            if hash_frame(known_at) != hashes[name]:
                raise ValueError(f"backtest availability hash differs:{name}")
        payload = _payload(
            self.snapshot_id,
            self.security_contract_hash,
            panels,
            self.production_ready,
            self.schema_version,
            benchmark_id=self.benchmark_id,
            benchmark_snapshot_id=self.benchmark_snapshot_id,
            benchmark_close=benchmark,
            availability=availability,
        )
        if hash_json(payload) != self.view_hash:
            raise ValueError("backtest data view hash differs")
        for name, panel in panels.items():
            object.__setattr__(self, name, panel)
        object.__setattr__(self, "benchmark_close", benchmark)
        object.__setattr__(self, "availability_hashes", MappingProxyType(hashes))
        object.__setattr__(self, "known_at_panels", MappingProxyType(availability))

    @classmethod
    def create(
        cls,
        *,
        snapshot_id: str,
        security_contract_hash: str,
        open_price: pd.DataFrame,
        close_price: pd.DataFrame,
        traded_amount: pd.DataFrame,
        adv_amount: pd.DataFrame,
        volatility: pd.DataFrame,
        half_spread_bps: pd.DataFrame,
        execution_tradability: pd.DataFrame,
        execution_reason: pd.DataFrame | None = None,
        split_ratio: pd.DataFrame | None = None,
        cash_dividend: pd.DataFrame | None = None,
        benchmark_id: str = "cash-zero-return",
        benchmark_snapshot_id: str | None = None,
        benchmark_close: pd.Series | None = None,
        known_at_panels: Mapping[str, pd.DataFrame] | None = None,
        production_ready: bool = False,
    ) -> "BacktestDataView":
        reference = pd.DataFrame(open_price)
        split = (
            pd.DataFrame(1.0, index=reference.index, columns=reference.columns)
            if split_ratio is None
            else pd.DataFrame(split_ratio)
        )
        dividend = (
            pd.DataFrame(0.0, index=reference.index, columns=reference.columns)
            if cash_dividend is None
            else pd.DataFrame(cash_dividend)
        )
        tradability = pd.DataFrame(execution_tradability).astype(bool)
        reason = (
            pd.DataFrame(
                np.where(tradability, "tradable", "not_tradable"),
                index=reference.index,
                columns=reference.columns,
            )
            if execution_reason is None
            else pd.DataFrame(execution_reason)
        )
        panels = {
            "open_price": pd.DataFrame(open_price),
            "close_price": pd.DataFrame(close_price),
            "traded_amount": pd.DataFrame(traded_amount),
            "adv_amount": pd.DataFrame(adv_amount),
            "volatility": pd.DataFrame(volatility),
            "half_spread_bps": pd.DataFrame(half_spread_bps),
            "execution_tradability": tradability,
            "execution_reason": reason,
            "split_ratio": split,
            "cash_dividend": dividend,
        }
        benchmark = (
            pd.Series(1.0, index=reference.index, name=benchmark_id)
            if benchmark_close is None
            else pd.Series(benchmark_close).copy()
        )
        bound_benchmark_snapshot = benchmark_snapshot_id or snapshot_id
        availability = {
            name: pd.DataFrame(panel).copy(deep=True)
            for name, panel in sorted((known_at_panels or {}).items())
        }
        availability_hashes = {
            name: hash_frame(panel) for name, panel in availability.items()
        }
        payload = _payload(
            snapshot_id,
            security_contract_hash,
            panels,
            production_ready,
            "backtest-data-view/v2",
            benchmark_id=benchmark_id,
            benchmark_snapshot_id=bound_benchmark_snapshot,
            benchmark_close=benchmark,
            availability=availability,
        )
        return cls(
            snapshot_id=snapshot_id,
            security_contract_hash=security_contract_hash,
            benchmark_id=benchmark_id,
            benchmark_snapshot_id=bound_benchmark_snapshot,
            benchmark_close=benchmark,
            availability_hashes=availability_hashes,
            known_at_panels=availability,
            production_ready=production_ready,
            view_hash=hash_json(payload),
            **panels,
        )

    def _copied_panels(self) -> dict[str, pd.DataFrame]:
        return {
            name: pd.DataFrame(getattr(self, name)).copy(deep=True)
            for name in (
                "open_price",
                "close_price",
                "traded_amount",
                "adv_amount",
                "volatility",
                "half_spread_bps",
                "execution_tradability",
                "execution_reason",
                "split_ratio",
                "cash_dividend",
            )
        }

    def verify_content(self) -> None:
        payload = _payload(
            self.snapshot_id,
            self.security_contract_hash,
            self._copied_panels(),
            self.production_ready,
            self.schema_version,
            benchmark_id=self.benchmark_id,
            benchmark_snapshot_id=self.benchmark_snapshot_id,
            benchmark_close=self.benchmark_close,
            availability=self._copied_availability(),
        )
        if hash_json(payload) != self.view_hash:
            raise RuntimeError("backtest data view content changed after construction")

    def _copied_availability(self) -> dict[str, pd.DataFrame]:
        return {
            name: pd.DataFrame(panel).copy(deep=True)
            for name, panel in self.known_at_panels.items()
        }

    def availability_at(
        self,
        field: str,
        observation_timestamp: pd.Timestamp,
        security: object,
    ) -> pd.Timestamp | None:
        """Return the immutable PIT timestamp for one value.

        Research-only views may omit availability metadata.  A view that claims
        production readiness must provide an explicit, timezone-aware timestamp
        for every value that the engine actually consumes.
        """

        if not self.production_ready:
            return None
        if field not in self.known_at_panels:
            raise ValueError(f"missing backtest availability panel:{field}")
        value = self.known_at_panels[field].loc[observation_timestamp, security]
        if pd.isna(value):
            raise ValueError(
                f"missing backtest availability:{field}:"
                f"{observation_timestamp}:{security}"
            )
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:  # guarded at construction; defense in depth
            raise ValueError(f"backtest availability must be timezone-aware:{field}")
        return timestamp

    def require_available(
        self,
        field: str,
        observation_timestamp: pd.Timestamp,
        security: object,
        *,
        as_of: pd.Timestamp,
    ) -> pd.Timestamp | None:
        available_at = self.availability_at(field, observation_timestamp, security)
        if available_at is None:
            return None
        use_time = pd.Timestamp(as_of)
        if use_time.tzinfo is None:
            raise ValueError("backtest use timestamp must be timezone-aware")
        if available_at > use_time:
            raise ValueError(
                f"future backtest availability:{field}:"
                f"available={available_at.isoformat()}:use={use_time.isoformat()}:"
                f"observation={observation_timestamp}:security={security}"
            )
        return available_at


def _payload(
    snapshot_id: str,
    security_contract_hash: str,
    panels: dict[str, pd.DataFrame],
    production_ready: bool,
    schema_version: str,
    *,
    benchmark_id: str,
    benchmark_snapshot_id: str,
    benchmark_close: pd.Series,
    availability: Mapping[str, pd.DataFrame],
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "snapshot_id": snapshot_id,
        "security_contract_hash": security_contract_hash,
        "production_ready": production_ready,
        "benchmark_id": benchmark_id,
        "benchmark_snapshot_id": benchmark_snapshot_id,
        "benchmark_close_hash": hash_frame(benchmark_close.to_frame("benchmark_close")),
        "panels": {name: hash_frame(panel) for name, panel in sorted(panels.items())},
        "availability_hashes": {
            name: hash_frame(panel) for name, panel in sorted(availability.items())
        },
    }


_AVAILABILITY_FIELDS = (
    "open_price",
    "close_price",
    "traded_amount",
    "adv_amount",
    "volatility",
    "half_spread_bps",
    "execution_tradability",
    "execution_reason",
    "split_ratio",
    "cash_dividend",
)


def _validate_availability_panel(panel: pd.DataFrame, *, name: str) -> None:
    for value in panel.to_numpy(dtype=object).ravel():
        if pd.isna(value):
            continue
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            raise ValueError(f"backtest availability must be timezone-aware:{name}")


__all__ = ["BacktestDataView"]
