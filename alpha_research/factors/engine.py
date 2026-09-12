from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from alpha_research.core.hashing import hash_frame, hash_json, require_sha256
from alpha_research.factors.complexity import (
    FactorComplexity,
    analyze_expression,
    enforce_complexity,
)
from alpha_research.factors.spec import AggregationSpec, FactorSpec, PreprocessKind
from alpha_research.factors.view import FactorDataView
from factor_production.v5.dsl import OperatorRegistry, SafeExpressionInterpreter


@dataclass(frozen=True, slots=True)
class FactorEngineExecutionPolicy:
    """Content-addressed scientific execution semantics for the factor engine."""

    require_point_in_time: bool
    operator_registry_factory: str = "OperatorRegistry.dataframe_pit_v3"
    preprocessing_order: str = (
        "expression_then_aggregation_then_declared_preprocessing"
    )
    nonfinite_output_policy: str = "replace_posneg_inf_with_nan"
    schema_version: str = "factor-engine-execution-policy/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "factor-engine-execution-policy/v1":
            raise ValueError("unsupported factor engine execution policy")
        if not isinstance(self.require_point_in_time, bool):
            raise TypeError("factor engine PIT policy must be boolean")
        expected = {
            "operator_registry_factory": "OperatorRegistry.dataframe_pit_v3",
            "preprocessing_order": (
                "expression_then_aggregation_then_declared_preprocessing"
            ),
            "nonfinite_output_policy": "replace_posneg_inf_with_nan",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"factor engine {name} differs")

    @property
    def content_hash(self) -> str:
        return cast(
            str,
            hash_json(
                {
                    "schema_version": self.schema_version,
                    "require_point_in_time": self.require_point_in_time,
                    "operator_registry_factory": self.operator_registry_factory,
                    "preprocessing_order": self.preprocessing_order,
                    "nonfinite_output_policy": self.nonfinite_output_policy,
                }
            ),
        )


STRICT_FACTOR_ENGINE_EXECUTION_POLICY = FactorEngineExecutionPolicy(
    require_point_in_time=True
)
STRICT_FACTOR_ENGINE_EXECUTION_POLICY_HASH = (
    STRICT_FACTOR_ENGINE_EXECUTION_POLICY.content_hash
)


@dataclass(frozen=True, slots=True)
class FactorResult:
    factor_spec_hash: str
    semantic_hash: str
    definition_hash: str
    factor_view_hash: str
    execution_policy_hash: str
    operator_registry_digest: str
    complexity: FactorComplexity
    signal_hash: str
    runtime_seconds: float
    admission_eligible: bool
    signal: pd.DataFrame

    def __post_init__(self) -> None:
        for name in (
            "factor_spec_hash",
            "semantic_hash",
            "definition_hash",
            "factor_view_hash",
            "execution_policy_hash",
            "operator_registry_digest",
            "signal_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"factor result {name}")
        if (
            not isinstance(self.runtime_seconds, (int, float))
            or isinstance(self.runtime_seconds, bool)
            or not math.isfinite(float(self.runtime_seconds))
            or float(self.runtime_seconds) < 0.0
        ):
            raise ValueError("factor result runtime must be finite and non-negative")
        copied = pd.DataFrame(self.signal).copy(deep=True)
        if hash_frame(copied) != self.signal_hash:
            raise ValueError("factor result signal hash differs")
        object.__setattr__(self, "runtime_seconds", float(self.runtime_seconds))
        object.__setattr__(self, "signal", copied)

    def verify_content(self) -> None:
        if hash_frame(self.signal) != self.signal_hash:
            raise RuntimeError("factor result content changed after construction")

    def descriptor(self) -> dict[str, object]:
        """Return operational metadata, including non-deterministic runtime."""

        return {
            **self.scientific_descriptor(),
            "runtime_seconds": self.runtime_seconds,
        }

    def scientific_descriptor(self) -> dict[str, object]:
        """Return deterministic scientific identity, excluding telemetry.

        Wall-clock runtime is useful operational evidence but changes between
        identical executions.  It must therefore never enter a content hash
        that identifies the scientific result.
        """

        self.verify_content()
        return {
            "schema_version": "factor-result-scientific-descriptor/v1",
            "factor_spec_hash": self.factor_spec_hash,
            "semantic_hash": self.semantic_hash,
            "definition_hash": self.definition_hash,
            "factor_view_hash": self.factor_view_hash,
            "execution_policy_hash": self.execution_policy_hash,
            "operator_registry_digest": self.operator_registry_digest,
            "complexity": self.complexity.to_dict(),
            "signal_hash": self.signal_hash,
            "admission_eligible": self.admission_eligible,
            "row_count": len(self.signal),
            "security_count": len(self.signal.columns),
        }

    @property
    def scientific_hash(self) -> str:
        return cast(str, hash_json(self.scientific_descriptor()))


class FactorEngine:
    """One authoritative, PIT-aware execution path for declarative factors."""

    def __init__(
        self,
        *,
        require_point_in_time: bool | None = None,
        execution_policy: FactorEngineExecutionPolicy | None = None,
    ) -> None:
        """Create an engine with an optional strict row-availability override.

        The default is strict for views built by ``from_market_view`` and keeps
        legacy hand-constructed views executable for research replay only.  A
        production caller can set ``require_point_in_time=True`` explicitly;
        no execution mode ever ignores panels that are present.
        """

        if execution_policy is not None:
            if type(execution_policy) is not FactorEngineExecutionPolicy:
                raise TypeError(
                    "factor engine requires an exact execution policy"
                )
            if require_point_in_time is not None:
                raise ValueError(
                    "factor engine PIT override and execution policy are exclusive"
                )
        elif require_point_in_time is not None and not isinstance(
            require_point_in_time,
            bool,
        ):
            raise TypeError("factor engine PIT override must be boolean or None")
        self._require_point_in_time = require_point_in_time
        self._execution_policy = execution_policy

    def evaluate(self, spec: FactorSpec, view: FactorDataView) -> FactorResult:
        view.verify_content()
        self._validate_bindings(spec, view)
        require_point_in_time = (
            self._execution_policy.require_point_in_time
            if self._execution_policy is not None
            else (
                view.strict_point_in_time
                if self._require_point_in_time is None
                else self._require_point_in_time
            )
        )
        execution_policy = self._execution_policy or FactorEngineExecutionPolicy(
            require_point_in_time=bool(require_point_in_time)
        )
        availability_mask = view.point_in_time_mask(require=require_point_in_time)
        history_mask = view.history_mask & availability_mask
        cross_section_mask = view.cross_section_mask & availability_mask
        registry = OperatorRegistry.dataframe_pit_v3(
            cross_section_mask,
            history_mask,
        )
        if registry.version != spec.operator_registry_version:
            raise ValueError("factor operator registry version differs")
        if registry.digest != spec.operator_registry_digest:
            raise ValueError("factor operator registry digest differs")
        interpreter = SafeExpressionInterpreter(
            registry,
            max_call_depth=spec.complexity_budget.maximum_call_depth,
            maximum_window=spec.complexity_budget.maximum_window,
        )
        validation = interpreter.validate(
            spec.expression,
            allowed_fields=frozenset(spec.required_fields),
        )
        if not validation.is_valid:
            raise ValueError(
                "factor_expression_rejected:" + ";".join(validation.reasons)
            )
        expression_fields = tuple(validation.fields)
        exposure_fields = tuple(
            sorted(
                {field for step in spec.preprocessing for field in step.exposure_fields}
            )
        )
        if (
            tuple(sorted({*expression_fields, *exposure_fields}))
            != spec.required_fields
        ):
            raise ValueError("factor required_fields differ from expression fields")
        complexity = analyze_expression(spec.expression, registry=registry)
        enforce_complexity(complexity, spec.complexity_budget)
        started = time.perf_counter()
        fields = {
            name: view.field_panel(name).where(availability_mask)
            for name in expression_fields
        }
        signal = interpreter.evaluate(spec.expression, fields)
        signal = signal.where(cross_section_mask)
        signal, output_mask = _aggregate_signal(
            signal,
            cross_section_mask,
            spec.aggregation,
            timezone=spec.frequency.timezone,
        )
        for step in spec.preprocessing:
            if step.kind is PreprocessKind.WINSORIZE_QUANTILE:
                if step.lower is None or step.upper is None:  # pragma: no cover
                    raise RuntimeError("winsorization parameters were not normalized")
                signal = _winsorize(
                    signal, lower=float(step.lower), upper=float(step.upper)
                )
            elif step.kind is PreprocessKind.ZSCORE:
                signal = _zscore(signal)
            elif step.kind is PreprocessKind.NEUTRALIZE_OLS:
                if spec.aggregation.method != "none":
                    raise ValueError(
                        "neutralization after intraday aggregation requires daily exposures"
                    )
                signal = _neutralize_ols(
                    signal,
                    {
                        name: view.field_panel(name).where(availability_mask)
                        for name in step.exposure_fields
                    },
                    ridge=step.ridge,
                )
            signal = signal.where(output_mask)
        signal = signal.replace([np.inf, -np.inf], np.nan).astype(float)
        runtime = time.perf_counter() - started
        return FactorResult(
            factor_spec_hash=spec.content_hash,
            semantic_hash=spec.semantic_hash,
            definition_hash=spec.definition_hash,
            factor_view_hash=view.view_hash,
            execution_policy_hash=execution_policy.content_hash,
            operator_registry_digest=registry.digest,
            complexity=complexity,
            signal_hash=hash_frame(signal),
            runtime_seconds=float(runtime),
            admission_eligible=(
                view.production_ready
                and view.has_row_availability
                and bool(require_point_in_time)
            ),
            signal=signal,
        )

    @staticmethod
    def _validate_bindings(spec: FactorSpec, view: FactorDataView) -> None:
        if spec.snapshot_id != view.snapshot_id:
            raise ValueError("factor snapshot binding differs")
        if spec.schema_hash != view.schema_hash:
            raise ValueError("factor schema binding differs")
        if spec.frequency.content_hash != view.frequency_hash:
            raise ValueError("factor frequency binding differs")
        if spec.availability_hash != view.availability_hash:
            raise ValueError("factor availability binding differs")
        if spec.security_contract_hash != view.security_contract_hash:
            raise ValueError("factor security contract binding differs")
        missing = sorted(set(spec.required_fields).difference(view.fields))
        if missing:
            raise ValueError("factor view is missing fields:" + ",".join(missing))


def _winsorize(frame: pd.DataFrame, *, lower: float, upper: float) -> pd.DataFrame:
    low = frame.quantile(lower, axis=1)
    high = frame.quantile(upper, axis=1)
    return frame.clip(lower=low, upper=high, axis=0)


def _zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    standard_deviation = frame.std(axis=1, ddof=0).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(standard_deviation, axis=0)


def _neutralize_ols(
    signal: pd.DataFrame,
    exposures: dict[str, pd.DataFrame],
    *,
    ridge: float,
) -> pd.DataFrame:
    result = pd.DataFrame(
        np.nan, index=signal.index, columns=signal.columns, dtype=float
    )
    for timestamp in signal.index:
        valid = np.isfinite(signal.loc[timestamp])
        for exposure in exposures.values():
            valid &= np.isfinite(exposure.loc[timestamp])
        count = int(valid.sum())
        if count <= len(exposures) + 1:
            continue
        design = np.column_stack(
            [
                np.ones(count),
                *[
                    exposure.loc[timestamp, valid].to_numpy(dtype=float)
                    for exposure in exposures.values()
                ],
            ]
        )
        target = signal.loc[timestamp, valid].to_numpy(dtype=float)
        penalty = np.eye(design.shape[1]) * ridge
        penalty[0, 0] = 0.0
        coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ target
        result.loc[timestamp, valid] = target - design @ coefficients
    return result


def _aggregate_signal(
    signal: pd.DataFrame,
    mask: pd.DataFrame,
    aggregation: AggregationSpec,
    *,
    timezone: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if aggregation.method == "none":
        return signal, mask
    index = pd.DatetimeIndex(signal.index)
    if index.tz is None:
        raise ValueError("factor aggregation requires timezone-aware timestamps")
    local = index.tz_convert(timezone)
    selected = _intraday_window(local, aggregation.window)
    if not bool(selected.any()):
        raise ValueError(
            f"factor aggregation window has no observations:{aggregation.window}"
        )
    values = signal.loc[selected]
    masks = mask.loc[selected]
    local_selected = local[selected]
    group_dates = local_selected.strftime("%Y%m%d")
    value_rows: list[pd.Series] = []
    mask_rows: list[pd.Series] = []
    output_index: list[pd.Timestamp] = []
    for date in pd.unique(group_dates):
        positions = np.flatnonzero(group_dates == date)
        group = values.iloc[positions]
        group_mask = masks.iloc[positions]
        value_rows.append(_reduce_group(group, aggregation.reducer))
        mask_rows.append(group_mask.iloc[-1].astype(bool))
        output_index.append(pd.Timestamp(group.index[-1]))
    daily = pd.DataFrame(
        value_rows, index=pd.DatetimeIndex(output_index), columns=signal.columns
    )
    daily_mask = pd.DataFrame(
        mask_rows, index=daily.index, columns=signal.columns, dtype=bool
    )
    daily = daily.where(daily_mask)
    if aggregation.smoothing_span:
        daily = daily.ewm(
            span=aggregation.smoothing_span,
            adjust=False,
            min_periods=aggregation.smoothing_span,
        ).mean()
        daily = daily.where(daily_mask)
    return daily, daily_mask


def _intraday_window(
    index: pd.DatetimeIndex,
    window: str,
) -> npt.NDArray[np.bool_]:
    minutes = index.hour * 60 + index.minute
    intervals = {
        "full_day": (9 * 60 + 30, 15 * 60),
        "open30": (9 * 60 + 30, 10 * 60),
        "midday30": (11 * 60, 11 * 60 + 30),
        "postlunch30": (13 * 60, 13 * 60 + 30),
        "close30": (14 * 60 + 30, 15 * 60),
        "close60": (14 * 60, 15 * 60),
    }
    try:
        start, end = intervals[window]
    except KeyError:
        raise ValueError(f"unsupported intraday aggregation window:{window}") from None
    return cast(
        npt.NDArray[np.bool_],
        np.asarray((minutes >= start) & (minutes <= end), dtype=bool),
    )


def _reduce_group(group: pd.DataFrame, reducer: str) -> pd.Series:
    if reducer == "last":
        return group.iloc[-1]
    if reducer == "mean":
        return group.mean(axis=0)
    if reducer == "sum":
        return group.sum(axis=0, min_count=1)
    if reducer == "std":
        return group.std(axis=0, ddof=0)
    if reducer == "skew":
        return group.skew(axis=0)
    if reducer == "kurt":
        return group.kurt(axis=0)
    raise RuntimeError("unsupported normalized aggregation reducer")  # pragma: no cover


__all__ = [
    "FactorEngine",
    "FactorEngineExecutionPolicy",
    "FactorResult",
    "STRICT_FACTOR_ENGINE_EXECUTION_POLICY",
    "STRICT_FACTOR_ENGINE_EXECUTION_POLICY_HASH",
]
