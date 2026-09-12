from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


MetricValue = float | int | None


def performance_metrics(
    nav: pd.DataFrame,
    fills: pd.DataFrame,
    *,
    annualization: int,
    risk_free_rate: float,
) -> tuple[dict[str, MetricValue], pd.DataFrame]:
    metrics: dict[str, MetricValue] = {}
    metadata: list[dict[str, object]] = []
    for prefix in ("gross", "net"):
        values = pd.to_numeric(nav[f"{prefix}_nav"], errors="coerce")
        returns = values.pct_change(fill_method=None).dropna()
        metric_values = _one_path(
            values,
            returns,
            annualization=annualization,
            risk_free_rate=risk_free_rate,
        )
        for name, value in metric_values.items():
            key = f"{prefix}_{name}"
            metrics[key] = value
            metadata.append(
                {
                    "metric": key,
                    "sample_count": int(len(returns)),
                    "unit": _unit(name),
                    "status": "ok" if value is not None else "insufficient_data",
                }
            )
        if "benchmark_return" in nav:
            benchmark = pd.to_numeric(nav["benchmark_return"], errors="coerce")
            active = returns - benchmark.reindex(returns.index)
            tracking_error = (
                float(active.std(ddof=1) * np.sqrt(annualization))
                if len(active.dropna()) > 1
                else None
            )
            active_std = float(active.std(ddof=1)) if len(active.dropna()) > 1 else 0.0
            information_ratio = (
                float(active.mean() / active_std * np.sqrt(annualization))
                if active_std > 0
                else None
            )
            for name, value in (
                ("tracking_error", tracking_error),
                ("information_ratio", information_ratio),
                (
                    "annualized_arithmetic_excess_return",
                    (
                        float(active.mean() * annualization)
                        if len(active.dropna())
                        else None
                    ),
                ),
            ):
                key = f"{prefix}_{name}"
                metrics[key] = value
                metadata.append(
                    {
                        "metric": key,
                        "sample_count": int(len(active.dropna())),
                        "unit": (
                            "ratio"
                            if name == "information_ratio"
                            else "decimal_return_or_rate"
                        ),
                        "status": "ok" if value is not None else "insufficient_data",
                    }
                )
    total_notional = (
        float(pd.to_numeric(fills["filled_notional"], errors="coerce").abs().sum())
        if len(fills)
        else 0.0
    )
    average_nav = float(pd.to_numeric(nav["gross_nav"], errors="coerce").mean())
    metrics["one_way_turnover"] = (
        None if average_nav <= 0 else total_notional / (2.0 * average_nav)
    )
    metrics["total_transaction_cost"] = float(nav["transaction_cost"].sum())
    metrics["total_borrow_cost"] = float(nav["borrow_cost"].sum())
    metrics["total_cost"] = float(nav["total_cost"].sum())
    metrics["average_holding_period_sessions"] = _average_holding_sessions(
        fills, pd.DatetimeIndex(nav.index)
    )
    for name in (
        "one_way_turnover",
        "total_transaction_cost",
        "total_borrow_cost",
        "total_cost",
        "average_holding_period_sessions",
    ):
        metadata.append(
            {
                "metric": name,
                "sample_count": int(
                    len(fills) if name == "one_way_turnover" else len(nav)
                ),
                "unit": (
                    "ratio"
                    if name == "one_way_turnover"
                    else (
                        "sessions"
                        if name == "average_holding_period_sessions"
                        else "currency"
                    )
                ),
                "status": "ok" if metrics[name] is not None else "insufficient_data",
            }
        )
    return dict(sorted(metrics.items())), pd.DataFrame(metadata).set_index("metric")


def _one_path(
    nav: pd.Series,
    returns: pd.Series,
    *,
    annualization: int,
    risk_free_rate: float,
) -> Mapping[str, MetricValue]:
    periods = len(returns)
    start = float(nav.iloc[0])
    end = float(nav.iloc[-1])
    annual_return = None
    if periods > 0 and start > 0 and end > 0:
        annual_return = float((end / start) ** (annualization / periods) - 1.0)
    volatility = (
        float(returns.std(ddof=1) * np.sqrt(annualization)) if periods > 1 else None
    )
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / annualization) - 1.0
    standard_deviation = float(returns.std(ddof=1)) if periods > 1 else float("nan")
    sharpe = (
        float((returns.mean() - daily_rf) / standard_deviation * np.sqrt(annualization))
        if periods > 1 and standard_deviation > 0
        else None
    )
    downside = np.minimum(returns.to_numpy(dtype=float) - daily_rf, 0.0)
    downside_deviation = (
        float(np.sqrt(np.mean(np.square(downside)))) if periods else 0.0
    )
    sortino = (
        float((returns.mean() - daily_rf) / downside_deviation * np.sqrt(annualization))
        if periods and downside_deviation > 0
        else None
    )
    drawdown = nav / nav.cummax() - 1.0
    maximum_drawdown = float(drawdown.min()) if len(drawdown) else None
    calmar = (
        float(annual_return / abs(maximum_drawdown))
        if annual_return is not None
        and maximum_drawdown is not None
        and maximum_drawdown < 0
        else None
    )
    positive = returns.loc[returns > 0]
    negative = returns.loc[returns < 0]
    profit_loss = (
        float(positive.mean() / abs(negative.mean()))
        if len(positive) and len(negative) and negative.mean() != 0
        else None
    )
    return {
        "annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "maximum_drawdown": maximum_drawdown,
        "calmar": calmar,
        "maximum_drawdown_duration": _maximum_drawdown_duration(drawdown),
        "win_rate": float((returns > 0).mean()) if periods else None,
        "profit_loss_ratio": profit_loss,
    }


def _maximum_drawdown_duration(drawdown: pd.Series) -> int:
    longest = 0
    current = 0
    for value in drawdown:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _unit(name: str) -> str:
    if name == "maximum_drawdown_duration":
        return "sessions"
    if name in {"sharpe", "sortino", "calmar", "profit_loss_ratio"}:
        return "ratio"
    return "decimal_return_or_rate"


def _average_holding_sessions(
    fills: pd.DataFrame, calendar: pd.DatetimeIndex
) -> float | None:
    if not len(fills):
        return None
    positions = {timestamp: offset for offset, timestamp in enumerate(calendar)}
    lots: dict[str, list[list[float]]] = {}
    weighted_age = 0.0
    closed_quantity = 0.0
    ordered = fills.sort_values("execution_timestamp", kind="stable")
    for _, row in ordered.iterrows():
        timestamp = pd.Timestamp(row["execution_timestamp"])
        offset = positions.get(timestamp)
        if offset is None:
            continue
        security = str(row["security"])
        quantity = float(row["filled_shares"])
        queue = lots.setdefault(security, [])
        while quantity != 0 and queue and np.sign(quantity) != np.sign(queue[0][0]):
            lot_quantity, opened = queue[0]
            matched = min(abs(quantity), abs(lot_quantity))
            weighted_age += matched * max(0.0, offset - opened)
            closed_quantity += matched
            lot_quantity -= np.sign(lot_quantity) * matched
            quantity -= np.sign(quantity) * matched
            if abs(lot_quantity) < 1e-12:
                queue.pop(0)
            else:
                queue[0][0] = lot_quantity
        if abs(quantity) >= 1e-12:
            queue.append([quantity, float(offset)])
    final_offset = len(calendar) - 1
    for queue in lots.values():
        for quantity, opened in queue:
            matched = abs(quantity)
            weighted_age += matched * max(0.0, final_offset - opened)
            closed_quantity += matched
    return None if closed_quantity == 0 else float(weighted_age / closed_quantity)


__all__ = ["MetricValue", "performance_metrics"]
