from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RawColumnNames:
    time: str = "time"
    stock: str = "stock_code"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str = "volume"
    amount: str = "amount"


@dataclass(frozen=True)
class MinuteEvalConfig:
    data_root: Path = Path("data/raw_minute")
    cache_root: Path | None = None
    date_format: str = "%Y-%m-%d"
    device: str = "cpu"
    raw_columns: RawColumnNames = field(default_factory=RawColumnNames)
    min_group_count: int = 5
