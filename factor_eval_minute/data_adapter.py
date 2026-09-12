from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import MinuteEvalConfig
from .panel_cache import PanelCache


@dataclass(frozen=True)
class MinutePanel:
    date: str
    minutes: pd.Index
    stocks: pd.Index
    fields: dict[str, pd.DataFrame]


class MinuteDataAdapter:
    def __init__(self, config: MinuteEvalConfig):
        self.config = config

    def load_day(self, date: str, fields: Iterable[str]) -> MinutePanel:
        field_list = list(fields)
        if self.config.cache_root is not None:
            cache = PanelCache(self.config.cache_root)
            if cache.is_cached(date, field_list):
                minutes, stocks, matrices = cache.load_fields(date, field_list)
                return MinutePanel(date=date, minutes=minutes, stocks=stocks, fields=matrices)
            panel = self._load_day_from_raw(date, field_list)
            cache.write_fields(date, panel.minutes, panel.stocks, panel.fields)
            minutes, stocks, matrices = cache.load_fields(date, field_list)
            return MinutePanel(date=date, minutes=minutes, stocks=stocks, fields=matrices)

        return self._load_day_from_raw(date, field_list)

    def _load_day_from_raw(self, date: str, fields: Iterable[str]) -> MinutePanel:
        path = self.config.data_root / f"{date}.feather"
        if not path.exists():
            raise FileNotFoundError(path)

        raw = pd.read_feather(path)
        prepared = self._prepare_raw(raw)
        result: dict[str, pd.DataFrame] = {}
        for field in fields:
            result[field] = self._field_to_matrix(prepared, field)

        minutes = pd.Index(sorted(prepared[self.config.raw_columns.time].unique()), name="minute")
        stocks = pd.Index(sorted(prepared[self.config.raw_columns.stock].unique()), name="stock")
        return MinutePanel(date=date, minutes=minutes, stocks=stocks, fields=result)

    def _prepare_raw(self, raw: pd.DataFrame) -> pd.DataFrame:
        c = self.config.raw_columns
        df = raw.copy()
        df = df.sort_values([c.stock, c.time])
        volume = df[c.volume].astype(float)
        amount = df[c.amount].astype(float)
        df["VWAP"] = np.where(volume.abs() > 0, amount / volume, np.nan)
        close = df[c.close].astype(float)
        df["Returns"] = close.groupby(df[c.stock]).pct_change()
        df["NextReturns"] = close.groupby(df[c.stock]).shift(-1) / close - 1
        return df

    def _field_to_matrix(self, df: pd.DataFrame, field: str) -> pd.DataFrame:
        column = self._resolve_field(field)
        c = self.config.raw_columns
        matrix = df.pivot(index=c.time, columns=c.stock, values=column)
        matrix = matrix.sort_index().sort_index(axis=1)
        matrix.index.name = "minute"
        matrix.columns.name = "stock"
        return matrix.astype(float)

    def _resolve_field(self, field: str) -> str:
        mapping = {
            "Open": self.config.raw_columns.open,
            "High": self.config.raw_columns.high,
            "Low": self.config.raw_columns.low,
            "Close": self.config.raw_columns.close,
            "Volume": self.config.raw_columns.volume,
            "Amount": self.config.raw_columns.amount,
            "VWAP": "VWAP",
            "Returns": "Returns",
            "NextReturns": "NextReturns",
        }
        if field not in mapping:
            raise KeyError(f"Unknown field: {field}")
        return mapping[field]
