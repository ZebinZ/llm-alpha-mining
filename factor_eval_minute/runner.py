from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .aggregation import aggregate_intraday, aggregate_interday
from .config import MinuteEvalConfig
from .data_adapter import MinuteDataAdapter
from .evaluator import evaluate_minute_factor
from .expression_engine import ExpressionEngine
from .schemas import ExpressionEvaluationResult


KNOWN_FIELDS = ["Open", "High", "Low", "Close", "Volume", "Amount", "VWAP", "Returns"]


def _extract_field_names(expression: str, known_fields: Sequence[str]) -> list[str]:
    return [field for field in known_fields if field in expression]


def evaluate_expression(
    expression: str,
    dates: Sequence[str],
    config: MinuteEvalConfig | None = None,
    future_return_field: str = "Returns",
    intraday_method: str = "last_valid",
) -> ExpressionEvaluationResult:
    cfg = config or MinuteEvalConfig()
    adapter = MinuteDataAdapter(cfg)
    engine = ExpressionEngine()
    needed = sorted(set(_extract_field_names(expression, KNOWN_FIELDS) + [future_return_field]))

    minute_factors: dict[str, pd.DataFrame] = {}
    minute_details: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, float]] = []
    for date in dates:
        panel = adapter.load_day(date, fields=needed)
        factor = engine.evaluate(expression, panel.fields)
        minute_factors[date] = factor
        eval_result = evaluate_minute_factor(factor, panel.fields[future_return_field], group_count=5)
        minute_details[date] = eval_result.by_minute
        summaries.append(eval_result.summary)

    daily_signal = aggregate_intraday(minute_factors, method=intraday_method)
    minute_summary = _average_summaries(summaries)
    return ExpressionEvaluationResult(
        minute_factors=minute_factors,
        minute_summary=minute_summary,
        minute_details=minute_details,
        daily_signal=daily_signal,
    )


def compute_daily_signal_only(
    expression: str,
    dates: Sequence[str],
    config: MinuteEvalConfig | None = None,
    intraday_method: str = "last_valid",
) -> pd.DataFrame:
    cfg = config or MinuteEvalConfig()
    adapter = MinuteDataAdapter(cfg)
    engine = ExpressionEngine()
    needed = sorted(set(_extract_field_names(expression, KNOWN_FIELDS)))

    minute_factors: dict[str, pd.DataFrame] = {}
    for date in dates:
        panel = adapter.load_day(date, fields=needed)
        minute_factors[date] = engine.evaluate(expression, panel.fields)
    return aggregate_intraday(minute_factors, method=intraday_method)


def evaluate_factor_values(
    minute_factors: dict[str, pd.DataFrame],
    future_returns: dict[str, pd.DataFrame],
    intraday_method: str = "last_valid",
    group_count: int = 5,
) -> ExpressionEvaluationResult:
    minute_details = {}
    summaries = []
    for date, factor in minute_factors.items():
        eval_result = evaluate_minute_factor(factor, future_returns[date], group_count=group_count)
        minute_details[date] = eval_result.by_minute
        summaries.append(eval_result.summary)
    return ExpressionEvaluationResult(
        minute_factors=minute_factors,
        minute_summary=_average_summaries(summaries),
        minute_details=minute_details,
        daily_signal=aggregate_intraday(minute_factors, method=intraday_method),
    )


def evaluate_aggregated_signal(
    daily_signal: pd.DataFrame,
    method: str = "mean",
    window: int = 5,
    normalize_before_agg: str | None = "rank",
) -> pd.DataFrame:
    return aggregate_interday(
        daily_signal,
        method=method,
        window=window,
        normalize_before_agg=normalize_before_agg,
    )


def _average_summaries(summaries: list[dict[str, float]]) -> dict[str, float]:
    if not summaries:
        return {}
    keys = summaries[0].keys()
    return {key: float(pd.Series([item[key] for item in summaries]).mean()) for key in keys}
