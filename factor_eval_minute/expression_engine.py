from __future__ import annotations

from typing import Mapping

import pandas as pd

from . import operators as op


class ExpressionEngine:
    def __init__(self) -> None:
        names = [
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Neg",
            "TsRank",
            "CsRank",
            "TsMean",
            "TsStd",
            "TsDelta",
            "TsDelay",
            "TsCorr",
            "TsAutoCorr",
            "TsSkew",
            "TsKurt",
            "Greater",
            "Less",
            "IfElse",
        ]
        self._functions = {name: getattr(op, name) for name in names}

    def evaluate(self, expression: str, fields: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        context = dict(self._functions)
        context.update(fields)
        result = eval(expression, {"__builtins__": {}}, context)
        if not isinstance(result, pd.DataFrame):
            raise TypeError(f"Expression did not return a DataFrame: {type(result)!r}")
        return result
