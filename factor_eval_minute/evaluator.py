from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MinuteEvaluationResult:
    by_minute: pd.DataFrame
    summary: dict[str, float]


def _row_corr(left: pd.DataFrame, right: pd.DataFrame) -> pd.Series:
    aligned_left, aligned_right = left.align(right, join="inner", axis=1)
    return aligned_left.corrwith(aligned_right, axis=1)


def _safe_ir(series: pd.Series) -> float:
    std = series.std()
    if pd.isna(std) or std == 0:
        return np.nan
    return float(series.mean() / std)


def evaluate_minute_factor(
    factor: pd.DataFrame,
    future_return: pd.DataFrame,
    group_count: int = 10,
) -> MinuteEvaluationResult:
    factor, future_return = factor.align(future_return, join="inner", axis=0)
    factor, future_return = factor.align(future_return, join="inner", axis=1)

    rank_factor = factor.rank(axis=1, pct=True)
    rank_return = future_return.rank(axis=1, pct=True)

    by_minute = pd.DataFrame(index=factor.index)
    by_minute["ic"] = _row_corr(factor, future_return)
    by_minute["rank_ic"] = _row_corr(rank_factor, rank_return)

    for group_idx in range(group_count):
        low = group_idx / group_count
        high = (group_idx + 1) / group_count
        mask = (rank_factor > low) & (rank_factor <= high)
        by_minute[f"group_{group_idx + 1}_ret"] = future_return.where(mask).mean(axis=1)

    by_minute["nan_ratio"] = factor.isna().mean(axis=1)
    by_minute["zero_ratio"] = (factor == 0).mean(axis=1)

    summary = {
        "ic_mean": float(by_minute["ic"].mean()),
        "rank_ic_mean": float(by_minute["rank_ic"].mean()),
        "rank_ic_ir": _safe_ir(by_minute["rank_ic"]),
        "nan_ratio_mean": float(by_minute["nan_ratio"].mean()),
        "zero_ratio_mean": float(by_minute["zero_ratio"].mean()),
    }
    return MinuteEvaluationResult(by_minute=by_minute, summary=summary)
