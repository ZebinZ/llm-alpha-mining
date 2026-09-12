from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-12


def Div(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    denominator = y.where(y.abs() > EPS)
    return x / denominator


def Add(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return x + y


def Sub(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return x - y


def Mul(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return x * y


def Neg(x: pd.DataFrame) -> pd.DataFrame:
    return -x


def TsRank(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window=window, min_periods=window).apply(
        lambda arr: pd.Series(arr).rank(pct=True).iloc[-1],
        raw=False,
    )


def CsRank(x: pd.DataFrame) -> pd.DataFrame:
    return x.rank(axis=1, pct=True)


def TsMean(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window=window, min_periods=window).mean()


def TsStd(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window=window, min_periods=window).std()


def TsDelta(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x - x.shift(window)


def TsDelay(x: pd.DataFrame, periods: int) -> pd.DataFrame:
    return x.shift(periods)


def TsCorr(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window=window, min_periods=window).corr(y)


def TsAutoCorr(x: pd.DataFrame, window: int, lag: int) -> pd.DataFrame:
    return x.rolling(window=window, min_periods=window).corr(x.shift(lag))


def TsSkew(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window=window, min_periods=window).skew()


def TsKurt(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window=window, min_periods=window).kurt()


def Greater(x: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return x > threshold


def Less(x: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return x < threshold


def _if_else_v1(cond: pd.DataFrame, x: pd.DataFrame, y: Any) -> pd.DataFrame:
    """Replay the original x-axis conditional semantics.

    This implementation is intentionally retained for immutable replays of the
    ``dataframe_v1`` and ``dataframe_pit_v1/v2`` registries.  It cannot handle
    two scalar branches, but changing it in place would make an old registry
    digest describe different behavior.
    """

    if not isinstance(y, pd.DataFrame):
        y = pd.DataFrame(y, index=x.index, columns=x.columns)
    return x.where(cond, y)


def _if_else_v2(cond: pd.DataFrame, x: Any, y: Any) -> pd.DataFrame:
    """Select between frame or scalar branches on the condition's exact axes.

    ``cond`` is the authoritative point-in-time panel.  Scalar branches are
    broadcast to its axes and DataFrame branches are reindexed to them before
    selection.  This prevents pandas from introducing an axis union and makes
    ``IfElse(condition_frame, 1, 0)`` a well-defined daily DSL expression.
    """

    if not isinstance(cond, pd.DataFrame):
        raise TypeError("IfElse condition must be a pandas DataFrame")

    def align_branch(value: Any, *, branch: str) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            return value.reindex(index=cond.index, columns=cond.columns)
        if np.isscalar(value):
            return pd.DataFrame(value, index=cond.index, columns=cond.columns)
        raise TypeError(
            f"IfElse {branch} branch must be a scalar or pandas DataFrame"
        )

    when_true = align_branch(x, branch="true")
    when_false = align_branch(y, branch="false")
    return when_true.where(cond, when_false)


def IfElse(cond: pd.DataFrame, x: Any, y: Any) -> pd.DataFrame:
    """Current condition-axis broadcasting semantics for conditional panels."""

    return _if_else_v2(cond, x, y)
