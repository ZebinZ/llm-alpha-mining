from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from alpha_research.core.data import DataBatch
from alpha_research.core.hashing import hash_json


@dataclass(frozen=True, slots=True)
class MarketDataView:
    """File-free factor input assembled from authenticated DataBatch objects."""

    snapshot_id: str
    schema_hash: str
    frequency_hash: str
    availability_hash: str
    frame: pd.DataFrame

    @classmethod
    def from_batches(cls, batches: Iterable[DataBatch]) -> "MarketDataView":
        items = tuple(batches)
        if not items:
            raise ValueError("market data view requires at least one batch")
        for item in items:
            item.verify_content()
        snapshot_ids = {item.snapshot_id for item in items}
        schema_hashes = {item.schema_hash for item in items}
        frequency_hashes = {item.frequency_hash for item in items}
        availability_hashes = {hash_json(item.availability.to_dict()) for item in items}
        request_hashes = {item.request_hash for item in items}
        if any(
            len(values) != 1
            for values in (
                snapshot_ids,
                schema_hashes,
                frequency_hashes,
                availability_hashes,
                request_hashes,
            )
        ):
            raise ValueError("batches from different contracts cannot share one view")
        sequence = [item.sequence for item in items]
        if sequence != list(range(len(items))):
            raise ValueError("batch sequence must be complete and ordered")
        frame = pd.concat([item.frame for item in items], ignore_index=True)
        if frame.duplicated(["timestamp", "security"]).any():
            raise ValueError(
                "market data view contains duplicate timestamp/security keys"
            )
        frame = frame.sort_values(["timestamp", "security"], kind="stable").reset_index(
            drop=True
        )
        return cls(
            snapshot_id=next(iter(snapshot_ids)),
            schema_hash=next(iter(schema_hashes)),
            frequency_hash=next(iter(frequency_hashes)),
            availability_hash=next(iter(availability_hashes)),
            frame=frame,
        )

    def field_panel(self, field: str) -> pd.DataFrame:
        if field not in self.frame:
            raise KeyError(field)
        return (
            self.frame.pivot(index="timestamp", columns="security", values=field)
            .sort_index()
            .sort_index(axis=1)
        )


__all__ = ["MarketDataView"]
