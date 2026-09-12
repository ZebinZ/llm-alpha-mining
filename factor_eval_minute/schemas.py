from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ExpressionEvaluationResult:
    minute_factors: dict[str, pd.DataFrame]
    minute_summary: dict[str, float]
    minute_details: dict[str, pd.DataFrame]
    daily_signal: pd.DataFrame
