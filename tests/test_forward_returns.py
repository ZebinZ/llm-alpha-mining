from __future__ import annotations

import numpy as np
import pandas as pd

from llm_alpha_mining.mining.evaluation import compute_forward_returns


def _windows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_date": "20200101",
                "label_start": "20200102",
                "label_end": "20200103",
            }
        ]
    )


def test_unexplained_missing_forward_return_is_not_zero_filled() -> None:
    returns = pd.DataFrame(
        {
            "000001": [0.10, np.nan],
            "000002": [0.10, 0.20],
        },
        index=["20200102", "20200103"],
    )
    result = compute_forward_returns(
        returns,
        ["20200102", "20200103"],
        _windows(),
    )
    assert np.isnan(result.period_returns.loc["20200101", "000001"])
    assert np.isclose(result.period_returns.loc["20200101", "000002"], 0.32)
    assert result.diagnostics.loc["20200101", "invalid_unexplained_missing_count"] == 1


def test_only_explicit_point_in_time_suspension_can_fill_zero() -> None:
    returns = pd.DataFrame(
        {"000001": [0.10, np.nan]},
        index=["20200102", "20200103"],
    )
    suspension = pd.DataFrame(
        {"000001": [False, True]},
        index=["20200102", "20200103"],
    )
    result = compute_forward_returns(
        returns,
        ["20200102", "20200103"],
        _windows(),
        explicit_suspension=suspension,
    )
    assert np.isclose(result.period_returns.loc["20200101", "000001"], 0.10)
    assert result.diagnostics.loc["20200101", "explicit_suspension_fill_count"] == 1
