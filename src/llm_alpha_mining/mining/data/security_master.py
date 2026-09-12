from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from llm_alpha_mining.mining.data.universe import universe_mask_from_fast_cache


SECURITY_MASTER_GUARD_VERSION = "pit_stock_master_fast_mask_intersection_v1"


@dataclass(frozen=True, slots=True)
class SecurityMasterAudit:
    """Compact, JSON-serializable audit for one cleaned universe build."""

    guard_version: str
    target_date_count: int
    target_security_count: int
    stock_master_date_count: int
    stock_master_security_count: int
    fast_mask_date_count: int
    fast_mask_security_count: int
    mapped_in_stock_master_count: int
    mapped_in_fast_mask_count: int
    mapped_in_both_count: int
    unmapped_in_either_count: int
    ever_eligible_count: int
    excluded_security_count: int
    excluded_399_security_count: int
    target_dates_before_stock_master_count: int
    target_dates_before_fast_mask_count: int
    eligible_cell_count: int
    total_cell_count: int
    stock_master_first_date: str
    stock_master_last_date: str
    fast_mask_first_date: str
    fast_mask_last_date: str
    excluded_399_examples: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["excluded_399_examples"] = list(self.excluded_399_examples)
        return payload


@dataclass(frozen=True, slots=True)
class CleanStockUniverse:
    """An aligned fail-closed stock mask plus its construction audit."""

    mask: pd.DataFrame
    audit: SecurityMasterAudit


def load_stock_security_master(path: str | Path) -> pd.DataFrame:
    """Load and normalize ``frdata/stock_name.pkl``.

    The source is a date-by-code table.  A non-null value means that the code
    is a stock on that date; the name text itself is intentionally not used.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"stock security master is missing: {source}")
    return normalize_stock_security_master(pd.read_pickle(source))


def normalize_stock_security_master(stock_name: pd.DataFrame) -> pd.DataFrame:
    """Convert a stock-name panel into a normalized point-in-time bool mask."""

    source = _normalize_axes(stock_name, source_name="stock security master")
    if source.empty or len(source.columns) == 0:
        raise ValueError("stock security master must not be empty")
    # The data contract is deliberately exact: non-null means stock that day.
    result = source.notna().astype(bool)
    result.index.name = "date"
    result.columns.name = "stock"
    return result


def normalize_point_in_time_mask(
    mask: pd.DataFrame,
    *,
    source_name: str = "point-in-time mask",
) -> pd.DataFrame:
    """Normalize a boolean PIT mask without inventing membership observations."""

    source = _normalize_axes(mask, source_name=source_name)
    if source.empty or len(source.columns) == 0:
        raise ValueError(f"{source_name} must not be empty")
    result = source.fillna(False).astype(bool)
    result.index.name = "date"
    result.columns.name = "stock"
    return result


def align_point_in_time_mask(
    mask: pd.DataFrame,
    dates: Sequence[object] | pd.Index,
    securities: Sequence[object] | pd.Index,
    *,
    source_name: str = "point-in-time mask",
) -> pd.DataFrame:
    """Align a PIT mask using past observations only.

    A source row can be carried forward to a later target date.  It is never
    backfilled into an earlier date.  Target codes absent from the source and
    dates before the first source row are therefore always ``False``.
    """

    source = normalize_point_in_time_mask(mask, source_name=source_name)
    target_dates = _normalize_target_dates(dates)
    target_securities = _normalize_target_securities(securities)

    source = source.reindex(columns=target_securities, fill_value=False).astype(
        "boolean"
    )
    combined_dates = source.index.union(target_dates).sort_values()
    # ffill is an as-of lookup from the past.  In particular, leading target
    # dates remain NA and become False; no bfill is present anywhere here.
    aligned = source.reindex(combined_dates).ffill().reindex(target_dates)
    aligned = aligned.fillna(False).astype(bool)
    aligned.index.name = "date"
    aligned.columns.name = "stock"
    return aligned


def build_clean_stock_universe(
    dates: Sequence[object] | pd.Index,
    securities: Sequence[object] | pd.Index,
    *,
    stock_security_master: pd.DataFrame,
    fast_cache_mask: pd.DataFrame,
) -> CleanStockUniverse:
    """Intersect the stock master and fast-cache PIT eligibility mask.

    Both channels are required.  Absence from either source fails closed.  The
    returned mask is intended to be applied to raw fields *before* any
    cross-sectional rank, z-score, neutralization, or portfolio grouping.
    """

    target_dates = _normalize_target_dates(dates)
    target_securities = _normalize_target_securities(securities)
    stock_master = normalize_point_in_time_mask(
        stock_security_master,
        source_name="stock security master",
    )
    fast_mask = normalize_point_in_time_mask(
        fast_cache_mask,
        source_name="fast cache PIT mask",
    )

    master_aligned = align_point_in_time_mask(
        stock_master,
        target_dates,
        target_securities,
        source_name="stock security master",
    )
    fast_aligned = align_point_in_time_mask(
        fast_mask,
        target_dates,
        target_securities,
        source_name="fast cache PIT mask",
    )
    combined = (master_aligned & fast_aligned).astype(bool)

    target_set = set(target_securities)
    master_codes = set(stock_master.columns)
    fast_codes = set(fast_mask.columns)
    mapped_master = target_set & master_codes
    mapped_fast = target_set & fast_codes
    mapped_both = target_set & master_codes & fast_codes
    ever_eligible = combined.any(axis=0)
    excluded = target_securities[~ever_eligible.to_numpy()].tolist()
    excluded_399 = sorted(code for code in excluded if str(code).startswith("399"))

    audit = SecurityMasterAudit(
        guard_version=SECURITY_MASTER_GUARD_VERSION,
        target_date_count=len(target_dates),
        target_security_count=len(target_securities),
        stock_master_date_count=len(stock_master.index),
        stock_master_security_count=len(stock_master.columns),
        fast_mask_date_count=len(fast_mask.index),
        fast_mask_security_count=len(fast_mask.columns),
        mapped_in_stock_master_count=len(mapped_master),
        mapped_in_fast_mask_count=len(mapped_fast),
        mapped_in_both_count=len(mapped_both),
        unmapped_in_either_count=len(target_set - mapped_both),
        ever_eligible_count=int(ever_eligible.sum()),
        excluded_security_count=len(excluded),
        excluded_399_security_count=len(excluded_399),
        target_dates_before_stock_master_count=int(
            (target_dates < stock_master.index.min()).sum()
        ),
        target_dates_before_fast_mask_count=int(
            (target_dates < fast_mask.index.min()).sum()
        ),
        eligible_cell_count=int(combined.to_numpy(dtype=bool).sum()),
        total_cell_count=int(combined.size),
        stock_master_first_date=str(stock_master.index.min()),
        stock_master_last_date=str(stock_master.index.max()),
        fast_mask_first_date=str(fast_mask.index.min()),
        fast_mask_last_date=str(fast_mask.index.max()),
        excluded_399_examples=tuple(excluded_399[:10]),
    )
    return CleanStockUniverse(mask=combined, audit=audit)


def build_clean_stock_universe_from_fast_cache(
    dates: Sequence[object] | pd.Index,
    securities: Sequence[object] | pd.Index,
    *,
    stock_security_master: pd.DataFrame | str | Path,
    fast_cache: Mapping[str, object],
) -> CleanStockUniverse:
    """Convenience adapter for the authoritative V5 runner inputs."""

    if isinstance(stock_security_master, (str, Path)):
        master = load_stock_security_master(stock_security_master)
    else:
        master = normalize_stock_security_master(stock_security_master)
    return build_clean_stock_universe(
        dates,
        securities,
        stock_security_master=master,
        fast_cache_mask=universe_mask_from_fast_cache(fast_cache),
    )


def apply_clean_stock_universe(
    frame: pd.DataFrame,
    clean_mask: pd.DataFrame,
) -> pd.DataFrame:
    """Mask raw date-by-security values before cross-sectional operations."""

    values = _normalize_axes(frame, source_name="factor frame")
    aligned = align_point_in_time_mask(
        clean_mask,
        values.index,
        values.columns,
        source_name="clean stock universe",
    )
    return values.where(aligned)


def _normalize_axes(frame: pd.DataFrame, *, source_name: str) -> pd.DataFrame:
    out = pd.DataFrame(frame).copy()
    out.index = pd.Index([_normalize_date(item) for item in out.index], name="date")
    out.columns = pd.Index(
        [_normalize_security(item) for item in out.columns], name="stock"
    )
    if not out.index.is_unique:
        raise ValueError(f"{source_name} dates are not unique after normalization")
    if not out.columns.is_unique:
        raise ValueError(f"{source_name} securities are not unique after normalization")
    return out.sort_index()


def _normalize_target_dates(
    dates: Sequence[object] | pd.Index,
) -> pd.Index:
    result = pd.Index([_normalize_date(item) for item in dates], name="date")
    if not result.is_unique:
        raise ValueError("target dates are not unique after normalization")
    return result


def _normalize_target_securities(
    securities: Sequence[object] | pd.Index,
) -> pd.Index:
    result = pd.Index([_normalize_security(item) for item in securities], name="stock")
    if not result.is_unique:
        raise ValueError("target securities are not unique after normalization")
    return result


def _normalize_security(value: object) -> str:
    text = str(value).strip()
    return text.zfill(6) if text.isdigit() and len(text) <= 6 else text


def _normalize_date(value: object) -> str:
    text = str(value).strip()
    compact = text.replace("-", "").replace("/", "").replace(".", "")
    if compact.isdigit() and len(compact) >= 8:
        return compact[:8]
    return pd.Timestamp(value).strftime("%Y%m%d")
