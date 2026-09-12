from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


class PanelCache:
    def __init__(self, root: Path):
        self.root = Path(root)

    def is_cached(self, date: str, fields: Iterable[str]) -> bool:
        day_dir = self._day_dir(date)
        required = [day_dir / "minutes.npy", day_dir / "stocks.npy"]
        required.extend(day_dir / f"{field}.npy" for field in fields)
        return all(path.exists() for path in required)

    def load_fields(self, date: str, fields: Iterable[str]) -> tuple[pd.Index, pd.Index, dict[str, pd.DataFrame]]:
        day_dir = self._day_dir(date)
        minutes = pd.Index(np.load(day_dir / "minutes.npy", allow_pickle=True), name="minute")
        stocks = pd.Index(np.load(day_dir / "stocks.npy", allow_pickle=True), name="stock")
        matrices: dict[str, pd.DataFrame] = {}
        for field in fields:
            values = np.load(day_dir / f"{field}.npy", allow_pickle=False)
            matrices[field] = pd.DataFrame(values, index=minutes, columns=stocks, dtype=float)
        return minutes, stocks, matrices

    def write_fields(
        self,
        date: str,
        minutes: pd.Index,
        stocks: pd.Index,
        fields: dict[str, pd.DataFrame],
    ) -> None:
        day_dir = self._day_dir(date)
        day_dir.mkdir(parents=True, exist_ok=True)
        np.save(day_dir / "minutes.npy", minutes.to_numpy())
        np.save(day_dir / "stocks.npy", stocks.to_numpy())
        for name, matrix in fields.items():
            aligned = matrix.reindex(index=minutes, columns=stocks)
            np.save(day_dir / f"{name}.npy", aligned.to_numpy(dtype=np.float32))

    def _day_dir(self, date: str) -> Path:
        return self.root / date
