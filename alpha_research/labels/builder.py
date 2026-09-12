from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from alpha_research.core.hashing import hash_frame, hash_json
from alpha_research.labels.spec import LabelSpec, LabelTask
from alpha_research.labels.view import LabelDataView


@dataclass(frozen=True, slots=True)
class LabelResult:
    label_spec_hash: str
    label_view_hash: str
    benchmark_hash: str | None
    labels_hash: str
    windows_hash: str
    validity_hash: str
    diagnostics_hash: str
    labels: pd.DataFrame
    label_windows: pd.DataFrame
    validity: pd.DataFrame
    diagnostics: pd.DataFrame

    def __post_init__(self) -> None:
        labels = pd.DataFrame(self.labels).copy(deep=True)
        windows = pd.DataFrame(self.label_windows).copy(deep=True)
        validity = _complete_boolean_panel(self.validity, name="label validity")
        diagnostics = pd.DataFrame(self.diagnostics).copy(deep=True)
        if hash_frame(labels) != self.labels_hash:
            raise ValueError("label values hash differs")
        if hash_frame(windows) != self.windows_hash:
            raise ValueError("label windows hash differs")
        if hash_frame(validity) != self.validity_hash:
            raise ValueError("label validity hash differs")
        if hash_frame(diagnostics) != self.diagnostics_hash:
            raise ValueError("label diagnostics hash differs")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "label_windows", windows)
        object.__setattr__(self, "validity", validity)
        object.__setattr__(self, "diagnostics", diagnostics)

    def verify_content(self) -> None:
        if hash_frame(self.labels) != self.labels_hash:
            raise RuntimeError("label values changed after construction")
        if hash_frame(self.label_windows) != self.windows_hash:
            raise RuntimeError("label windows changed after construction")
        if hash_frame(self.validity) != self.validity_hash:
            raise RuntimeError("label validity changed after construction")
        if hash_frame(self.diagnostics) != self.diagnostics_hash:
            raise RuntimeError("label diagnostics changed after construction")

    def scientific_descriptor(self) -> dict[str, str | None]:
        """Return the stable identity shared by every label consumer.

        The field set intentionally matches the already-published legacy
        label-result hash.  Centralizing it here prevents evaluation,
        market-logic and materialization paths from silently drifting while
        preserving existing content addresses.
        """

        self.verify_content()
        return {
            "label_spec_hash": self.label_spec_hash,
            "label_view_hash": self.label_view_hash,
            "benchmark_hash": self.benchmark_hash,
            "labels_hash": self.labels_hash,
            "windows_hash": self.windows_hash,
            "validity_hash": self.validity_hash,
            "diagnostics_hash": self.diagnostics_hash,
        }

    @property
    def content_hash(self) -> str:
        return hash_json(self.scientific_descriptor())


class LabelBuilder:
    """Build labels without allowing future validity to define signal membership."""

    def build(
        self,
        spec: LabelSpec,
        view: LabelDataView,
        *,
        signal_timestamps: pd.DatetimeIndex,
        benchmark_adjusted_prices: pd.Series | None = None,
        benchmark_snapshot_id: str | None = None,
    ) -> LabelResult:
        view.verify_content()
        self._validate_bindings(spec, view)
        index = _daily_index(view)
        signals = pd.DatetimeIndex(signal_timestamps)
        if signals.tz is None:
            raise ValueError("label signal timestamps must be timezone-aware")
        if not signals.is_unique or not signals.is_monotonic_increasing:
            raise ValueError("label signal timestamps must be sorted and unique")
        missing_signals = signals.difference(index)
        if len(missing_signals):
            raise ValueError("label signals fall outside the data calendar")
        benchmark_hash: str | None = None
        benchmark = None
        if spec.task is LabelTask.EXCESS_RETURN:
            if benchmark_adjusted_prices is None:
                raise ValueError("excess-return label requires benchmark prices")
            if benchmark_snapshot_id != spec.benchmark_snapshot_id:
                raise ValueError("benchmark snapshot binding differs")
            benchmark = _benchmark_series(benchmark_adjusted_prices, index)
            benchmark_hash = hash_frame(benchmark.to_frame("benchmark"))
        elif benchmark_adjusted_prices is not None or benchmark_snapshot_id is not None:
            raise ValueError("benchmark input supplied to a non-excess label")

        price = _finite_panel(view.field_panel(spec.price_field))
        adjustment = (
            pd.DataFrame(1.0, index=price.index, columns=price.columns)
            if spec.adjustment_field is None
            else _finite_panel(view.field_panel(spec.adjustment_field))
        )
        adjusted_price = price * adjustment
        amount = (
            None
            if spec.amount_field is None
            else _finite_panel(view.field_panel(spec.amount_field))
        )
        position = {timestamp: offset for offset, timestamp in enumerate(index)}
        label_rows: list[pd.Series] = []
        validity_rows: list[pd.Series] = []
        windows: list[dict[str, object]] = []
        diagnostics: list[dict[str, object]] = []
        output_index: list[pd.Timestamp] = []
        for signal in signals:
            signal_position = position[signal]
            entry_position = signal_position + spec.entry_lag_sessions
            exit_position = entry_position + spec.horizon_sessions
            if exit_position >= len(index):
                continue
            entry = index[entry_position]
            exit_time = index[exit_position]
            if spec.task is LabelTask.VOLATILITY:
                values, valid = _future_volatility(
                    adjusted_price.iloc[entry_position : exit_position + 1],
                    annualize=spec.annualize_volatility,
                )
            else:
                entry_price = adjusted_price.iloc[entry_position]
                exit_price = adjusted_price.iloc[exit_position]
                valid = (
                    np.isfinite(entry_price)
                    & np.isfinite(exit_price)
                    & (entry_price > 0)
                    & (exit_price > 0)
                )
                if spec.require_execution_tradability:
                    execution = view.execution_tradability_mask
                    valid &= (
                        execution.iloc[entry_position] & execution.iloc[exit_position]
                    )
                    if amount is None:  # guarded by LabelSpec
                        raise RuntimeError("execution amount was not supplied")
                    valid &= (
                        np.isfinite(amount.iloc[entry_position])
                        & np.isfinite(amount.iloc[exit_position])
                        & (amount.iloc[entry_position] > 0)
                        & (amount.iloc[exit_position] > 0)
                    )
                raw_return = exit_price / entry_price - 1.0
                if spec.task is LabelTask.EXCESS_RETURN:
                    if benchmark is None:  # pragma: no cover
                        raise RuntimeError("benchmark was not normalized")
                    benchmark_return = (
                        benchmark.iloc[exit_position] / benchmark.iloc[entry_position]
                        - 1.0
                    )
                    if not np.isfinite(benchmark_return):
                        valid &= False
                    values = raw_return - benchmark_return
                elif spec.task is LabelTask.DIRECTION:
                    values = (raw_return > spec.direction_threshold).astype(float)
                else:
                    values = raw_return
                values = values.where(valid)
            label_rows.append(values.astype(float))
            validity_rows.append(valid.astype(bool))
            output_index.append(signal)
            windows.append(
                {
                    "signal_timestamp": signal,
                    "information_cutoff": signal,
                    "entry_timestamp": entry,
                    "exit_timestamp": exit_time,
                    "label_start": entry,
                    "label_end": exit_time,
                    "horizon_sessions": spec.horizon_sessions,
                    "holding_interval": "half_open_[entry,exit)",
                }
            )
            diagnostics.append(
                {
                    "signal_timestamp": signal,
                    "security_count": len(valid),
                    "valid_label_count": int(valid.sum()),
                    "invalid_label_count": int((~valid).sum()),
                }
            )
        output = pd.DatetimeIndex(output_index, name="signal_timestamp")
        labels = pd.DataFrame(
            label_rows, index=output, columns=price.columns, dtype=float
        )
        validity = pd.DataFrame(
            validity_rows, index=output, columns=price.columns, dtype=bool
        )
        window_frame = pd.DataFrame(windows)
        diagnostic_frame = pd.DataFrame(diagnostics)
        return LabelResult(
            label_spec_hash=spec.content_hash,
            label_view_hash=view.view_hash,
            benchmark_hash=benchmark_hash,
            labels_hash=hash_frame(labels),
            windows_hash=hash_frame(window_frame),
            validity_hash=hash_frame(validity),
            diagnostics_hash=hash_frame(diagnostic_frame),
            labels=labels,
            label_windows=window_frame,
            validity=validity,
            diagnostics=diagnostic_frame,
        )

    @staticmethod
    def _validate_bindings(spec: LabelSpec, view: LabelDataView) -> None:
        if spec.snapshot_id != view.snapshot_id:
            raise ValueError("label snapshot binding differs")
        if spec.schema_hash != view.schema_hash:
            raise ValueError("label schema binding differs")
        if spec.frequency.content_hash != view.frequency_hash:
            raise ValueError("label frequency binding differs")
        if spec.availability_hash != view.availability_hash:
            raise ValueError("label availability binding differs")
        if spec.security_contract_hash != view.security_contract_hash:
            raise ValueError("label security contract binding differs")
        try:
            view_price_event = view.field_observation_event(spec.price_field)
        except KeyError as exc:
            raise ValueError(str(exc)) from None
        if spec.price_observation_event is not view_price_event:
            raise ValueError(
                "label price observation event differs from the authenticated data view"
            )
        required = {spec.price_field}
        if spec.adjustment_field is not None:
            required.add(spec.adjustment_field)
        if spec.amount_field is not None:
            required.add(spec.amount_field)
        missing = sorted(required.difference(view.fields))
        if missing:
            raise ValueError("label view is missing fields:" + ",".join(missing))


def _daily_index(view: LabelDataView) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(next(iter(view.fields.values())).index)
    if index.tz is None:
        raise ValueError("label data index must be timezone-aware")
    local_dates = index.tz_convert("Asia/Shanghai").normalize()
    if not local_dates.is_unique:
        raise ValueError("Phase 2 LabelBuilder currently requires one bar per session")
    if not index.is_monotonic_increasing:
        raise ValueError("label data index must be chronologically sorted")
    return index


def _finite_panel(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )


def _complete_boolean_panel(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    values = pd.DataFrame(frame).copy(deep=True)
    if values.isna().to_numpy().any() or any(
        not is_bool_dtype(dtype) for dtype in values.dtypes
    ):
        raise TypeError(f"{name} must be a complete boolean panel")
    return values.astype(bool)


def _benchmark_series(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    values = pd.Series(series).copy()
    values.index = pd.DatetimeIndex(values.index)
    if values.index.tz is None:
        raise ValueError("benchmark price index must be timezone-aware")
    values = pd.to_numeric(values.reindex(index), errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    return values.where(values > 0).astype(float)


def _future_volatility(
    adjusted_price: pd.DataFrame,
    *,
    annualize: bool,
) -> tuple[pd.Series, pd.Series]:
    valid_prices = np.isfinite(adjusted_price) & (adjusted_price > 0)
    complete = valid_prices.all(axis=0)
    log_returns = np.log(adjusted_price).diff().iloc[1:]
    values = log_returns.std(axis=0, ddof=0)
    if annualize:
        values = values * np.sqrt(252.0)
    valid = complete & np.isfinite(values)
    return values.where(valid), valid.astype(bool)


__all__ = ["LabelBuilder", "LabelResult"]
