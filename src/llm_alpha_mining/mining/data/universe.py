from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


UNIVERSE_GUARD_VERSION = "pit_allowed_axis_before_rank_v1"


def universe_mask_from_fast_cache(cache: Mapping[str, object]) -> pd.DataFrame:
    """Return the stock-universe channel from a fast-evaluator cache.

    ``benchmark_ret`` is deliberately ignored.  A code such as ``000905`` can
    identify both a listed stock and an index in different source channels, so
    code blacklists are not a safe way to separate securities from benchmarks.
    The point-in-time stock mask is the authoritative source instead.
    """

    if "mask" not in cache:
        raise KeyError("fast evaluator cache missing stock-universe mask")
    mask = _normalize_mask(pd.DataFrame(cache["mask"]))
    if mask.empty or len(mask.columns) == 0:
        raise ValueError("fast evaluator cache has an empty stock-universe mask")
    return mask


def allowed_axis_from_mask(mask: pd.DataFrame) -> tuple[str, ...]:
    """Return securities that are eligible at least once in the PIT mask."""

    normalized = _normalize_mask(mask)
    ever_eligible = normalized.any(axis=0)
    return tuple(str(item) for item in normalized.columns[ever_eligible])


def build_point_in_time_universe_mask(
    dates: Sequence[object] | pd.Index,
    securities: Sequence[object] | pd.Index,
    *,
    point_in_time_mask: pd.DataFrame | None = None,
    allowed_axis: Sequence[object] | pd.Index | None = None,
) -> pd.DataFrame:
    """Align an allowed security axis and optional PIT mask to target axes.

    Mask observations are carried forward only, never backward.  Dates before
    the first mask observation are warm-up dates: the static ``allowed_axis``
    applies there, but no future membership row is backfilled.  When
    ``allowed_axis`` is omitted and a PIT mask is supplied, securities that are
    eligible at least once become the allowed axis; all-false contaminant
    columns therefore remain excluded during warm-up too.  Supplying neither
    preserves the legacy all-columns behavior.
    """

    target_dates = pd.Index([_normalize_date(item) for item in dates], name="date")
    target_securities = pd.Index(
        [_normalize_security(item) for item in securities], name="stock"
    )
    if not target_dates.is_unique or not target_securities.is_unique:
        raise ValueError("target universe axes must be unique after normalization")

    normalized_mask = (
        _normalize_mask(point_in_time_mask) if point_in_time_mask is not None else None
    )
    if allowed_axis is None:
        allowed = (
            set(allowed_axis_from_mask(normalized_mask))
            if normalized_mask is not None
            else set(target_securities)
        )
    else:
        allowed = {_normalize_security(item) for item in allowed_axis}

    static_allowed = pd.Series(
        target_securities.isin(allowed),
        index=target_securities,
        dtype=bool,
    )
    result = pd.DataFrame(
        [static_allowed.to_numpy()] * len(target_dates),
        index=target_dates,
        columns=target_securities,
        dtype=bool,
    )
    if normalized_mask is None or normalized_mask.empty or target_dates.empty:
        return result

    source = normalized_mask.reindex(
        columns=target_securities, fill_value=False
    ).astype("boolean")
    combined_dates = source.index.union(target_dates).sort_values()
    asof = source.reindex(combined_dates).ffill().reindex(target_dates)
    has_observed_mask = target_dates >= str(source.index.min())
    if bool(has_observed_mask.any()):
        observed_dates = target_dates[has_observed_mask]
        observed = asof.loc[observed_dates].fillna(False).astype(bool)
        result.loc[observed_dates] = observed & static_allowed
    return result.astype(bool)


def apply_universe_mask(
    frame: pd.DataFrame,
    *,
    point_in_time_mask: pd.DataFrame | None = None,
    allowed_axis: Sequence[object] | pd.Index | None = None,
) -> pd.DataFrame:
    """Set unauthorized observations to ``NaN`` without dropping target axes."""

    values = pd.DataFrame(frame).copy()
    values.index = [_normalize_date(item) for item in values.index]
    values.columns = [_normalize_security(item) for item in values.columns]
    if not values.index.is_unique or not values.columns.is_unique:
        raise ValueError("factor axes must be unique after normalization")
    mask = build_point_in_time_universe_mask(
        values.index,
        values.columns,
        point_in_time_mask=point_in_time_mask,
        allowed_axis=allowed_axis,
    )
    return values.where(mask)


def _normalize_mask(mask: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(mask).copy()
    out.index = [_normalize_date(item) for item in out.index]
    out.columns = [_normalize_security(item) for item in out.columns]
    if not out.index.is_unique or not out.columns.is_unique:
        raise ValueError("universe mask axes must be unique after normalization")
    return out.sort_index().fillna(False).astype(bool)


def _normalize_security(value: object) -> str:
    text = str(value).strip()
    return text.zfill(6) if text.isdigit() and len(text) <= 6 else text


def _normalize_date(value: object) -> str:
    text = str(value).strip()
    compact = text.replace("-", "").replace("/", "").replace(".", "")
    if compact.isdigit() and len(compact) >= 8:
        return compact[:8]
    return pd.Timestamp(value).strftime("%Y%m%d")
