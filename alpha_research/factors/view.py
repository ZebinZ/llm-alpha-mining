from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import pandas as pd

from alpha_research.core.hashing import hash_frame, hash_json
from alpha_research.data.views import MarketDataView
from factor_production.v5.data.security_contract import (
    SecurityDataContract,
    security_data_contract_hash,
)


@dataclass(frozen=True, slots=True)
class FactorDataView:
    snapshot_id: str
    schema_hash: str
    frequency_hash: str
    availability_hash: str
    security_contract_hash: str
    production_ready: bool
    degraded_reasons: tuple[str, ...]
    fields: Mapping[str, pd.DataFrame]
    history_mask: pd.DataFrame
    cross_section_mask: pd.DataFrame
    view_hash: str
    effective_at_panel: pd.DataFrame | None = None
    known_at_panel: pd.DataFrame | None = None
    strict_point_in_time: bool = False

    def __post_init__(self) -> None:
        copied_fields = {
            name: pd.DataFrame(value).copy(deep=True)
            for name, value in self.fields.items()
        }
        history = _complete_boolean_mask(
            self.history_mask,
            name="history",
        )
        cross_section = _complete_boolean_mask(
            self.cross_section_mask,
            name="cross_section",
        )
        effective_at = (
            None
            if self.effective_at_panel is None
            else pd.DataFrame(self.effective_at_panel).copy(deep=True)
        )
        known_at = (
            None
            if self.known_at_panel is None
            else pd.DataFrame(self.known_at_panel).copy(deep=True)
        )
        if set(copied_fields) == set():
            raise ValueError("factor data view requires at least one field")
        reference = next(iter(copied_fields.values()))
        for name, value in copied_fields.items():
            if not value.index.equals(reference.index) or not value.columns.equals(
                reference.columns
            ):
                raise ValueError(f"factor view field axes differ:{name}")
        for name, mask in (("history", history), ("cross_section", cross_section)):
            if not mask.index.equals(reference.index) or not mask.columns.equals(
                reference.columns
            ):
                raise ValueError(f"factor view {name} mask axes differ")
        if (effective_at is None) != (known_at is None):
            raise ValueError(
                "factor view requires both effective_at and known_at panels"
            )
        for name, panel in (
            ("effective_at", effective_at),
            ("known_at", known_at),
        ):
            if panel is not None and (
                not panel.index.equals(reference.index)
                or not panel.columns.equals(reference.columns)
            ):
                raise ValueError(f"factor view {name} panel axes differ")
        if self.strict_point_in_time and effective_at is None:
            raise ValueError(
                "strict factor view requires row-level effective_at and known_at"
            )
        payload = _view_payload(
            snapshot_id=self.snapshot_id,
            schema_hash=self.schema_hash,
            frequency_hash=self.frequency_hash,
            availability_hash=self.availability_hash,
            security_contract_hash=self.security_contract_hash,
            production_ready=self.production_ready,
            degraded_reasons=self.degraded_reasons,
            fields=copied_fields,
            history_mask=history,
            cross_section_mask=cross_section,
            effective_at_panel=effective_at,
            known_at_panel=known_at,
            strict_point_in_time=self.strict_point_in_time,
        )
        if hash_json(payload) != self.view_hash:
            raise ValueError("factor data view hash differs from content")
        object.__setattr__(self, "fields", MappingProxyType(copied_fields))
        object.__setattr__(self, "history_mask", history)
        object.__setattr__(self, "cross_section_mask", cross_section)
        object.__setattr__(self, "effective_at_panel", effective_at)
        object.__setattr__(self, "known_at_panel", known_at)

    @classmethod
    def from_market_view(
        cls,
        market: MarketDataView,
        security_contract: SecurityDataContract,
        *,
        required_fields: tuple[str, ...],
        timezone: str = "Asia/Shanghai",
        allow_degraded_research: bool = False,
        upstream_production_ready: bool = True,
        upstream_degraded_reasons: tuple[str, ...] = (),
    ) -> "FactorDataView":
        audit = security_contract.audit
        production_ready, degraded_reasons = _combined_assurance(
            security_contract_ready=audit.production_ready,
            security_contract_reasons=audit.degraded_reasons,
            upstream_production_ready=upstream_production_ready,
            upstream_degraded_reasons=upstream_degraded_reasons,
        )
        if not production_ready and not allow_degraded_research:
            reasons = ",".join(degraded_reasons) or "unspecified"
            raise PermissionError(
                "factor_view_not_production_ready:" + reasons
            )
        fields = {
            field: market.field_panel(field)
            for field in tuple(sorted(set(required_fields)))
        }
        effective_at = market.field_panel("effective_at")
        known_at = market.field_panel("known_at")
        reference = next(iter(fields.values()))
        history = _broadcast_daily_mask(
            security_contract.instrument_identity_mask,
            reference.index,
            reference.columns,
            timezone=timezone,
        )
        cross_section = _broadcast_daily_mask(
            security_contract.signal_universe_mask,
            reference.index,
            reference.columns,
            timezone=timezone,
        )
        fields = {name: value.where(history) for name, value in fields.items()}
        contract_hash = security_data_contract_hash(security_contract)
        payload = _view_payload(
            snapshot_id=market.snapshot_id,
            schema_hash=market.schema_hash,
            frequency_hash=market.frequency_hash,
            availability_hash=market.availability_hash,
            security_contract_hash=contract_hash,
            production_ready=production_ready,
            degraded_reasons=degraded_reasons,
            fields=fields,
            history_mask=history,
            cross_section_mask=cross_section,
            effective_at_panel=effective_at,
            known_at_panel=known_at,
            strict_point_in_time=True,
        )
        return cls(
            snapshot_id=market.snapshot_id,
            schema_hash=market.schema_hash,
            frequency_hash=market.frequency_hash,
            availability_hash=market.availability_hash,
            security_contract_hash=contract_hash,
            production_ready=production_ready,
            degraded_reasons=degraded_reasons,
            fields=fields,
            history_mask=history,
            cross_section_mask=cross_section,
            view_hash=hash_json(payload),
            effective_at_panel=effective_at,
            known_at_panel=known_at,
            strict_point_in_time=True,
        )

    @classmethod
    def from_market_view_with_domain(
        cls,
        market: MarketDataView,
        security_contract: SecurityDataContract,
        *,
        required_fields: tuple[str, ...],
        domain_mask: pd.DataFrame,
        timezone: str = "Asia/Shanghai",
        allow_degraded_research: bool = False,
        upstream_production_ready: bool = True,
        upstream_degraded_reasons: tuple[str, ...] = (),
    ) -> "FactorDataView":
        """Build a PIT view on a declared sub-domain of the signal universe.

        Structural provider coverage can be narrower than the stock universe
        (for example, an Order channel that is available only for one venue
        after a documented cutover).  The domain is therefore an execution
        input and part of the resulting view hash.  It may only remove cells
        from the canonical signal universe; it can never add securities or
        dates that the security contract did not authorize.
        """

        base = cls.from_market_view(
            market,
            security_contract,
            required_fields=required_fields,
            timezone=timezone,
            allow_degraded_research=allow_degraded_research,
            upstream_production_ready=upstream_production_ready,
            upstream_degraded_reasons=upstream_degraded_reasons,
        )
        domain = pd.DataFrame(domain_mask).copy(deep=True)
        if (
            not domain.index.equals(base.cross_section_mask.index)
            or not domain.columns.equals(base.cross_section_mask.columns)
        ):
            raise ValueError("factor domain mask axes differ")
        if domain.isna().any().any() or any(
            str(dtype) not in {"bool", "boolean"} for dtype in domain.dtypes
        ):
            raise TypeError("factor domain mask must be a complete boolean panel")
        domain = domain.astype(bool)
        outside = domain & ~base.cross_section_mask
        if bool(outside.to_numpy(dtype=bool).any()):
            raise ValueError("factor domain exceeds the security signal universe")
        cross_section = base.cross_section_mask & domain
        payload = _view_payload(
            snapshot_id=base.snapshot_id,
            schema_hash=base.schema_hash,
            frequency_hash=base.frequency_hash,
            availability_hash=base.availability_hash,
            security_contract_hash=base.security_contract_hash,
            production_ready=base.production_ready,
            degraded_reasons=base.degraded_reasons,
            fields=base.fields,
            history_mask=base.history_mask,
            cross_section_mask=cross_section,
            effective_at_panel=base.effective_at_panel,
            known_at_panel=base.known_at_panel,
            strict_point_in_time=base.strict_point_in_time,
        )
        return cls(
            snapshot_id=base.snapshot_id,
            schema_hash=base.schema_hash,
            frequency_hash=base.frequency_hash,
            availability_hash=base.availability_hash,
            security_contract_hash=base.security_contract_hash,
            production_ready=base.production_ready,
            degraded_reasons=base.degraded_reasons,
            fields=base.fields,
            history_mask=base.history_mask,
            cross_section_mask=cross_section,
            view_hash=hash_json(payload),
            effective_at_panel=base.effective_at_panel,
            known_at_panel=base.known_at_panel,
            strict_point_in_time=base.strict_point_in_time,
        )

    def field_panel(self, field: str) -> pd.DataFrame:
        try:
            return self.fields[field].copy(deep=True)
        except KeyError:
            raise KeyError(f"factor view field is unavailable:{field}") from None

    @property
    def has_row_availability(self) -> bool:
        return self.effective_at_panel is not None and self.known_at_panel is not None

    def point_in_time_mask(self, *, require: bool = False) -> pd.DataFrame:
        """Return cell-level availability at each signal timestamp.

        A late snapshot-level ``as_of`` is deliberately not accepted as a
        substitute for this panel: every security and observation timestamp is
        checked independently.  Missing cells are unavailable.  Legacy views
        without panels may only run when ``require`` is false and are never
        eligible for production admission.
        """

        if not self.has_row_availability:
            if require:
                raise PermissionError(
                    "factor_row_availability_required_for_strict_execution"
                )
            return pd.DataFrame(
                True,
                index=self.history_mask.index,
                columns=self.history_mask.columns,
                dtype=bool,
            )
        if self.effective_at_panel is None or self.known_at_panel is None:
            raise RuntimeError("factor row availability state is inconsistent")
        index = pd.DatetimeIndex(self.history_mask.index)
        if index.tz is None:
            raise ValueError("factor signal timestamps must be timezone-aware")
        effective = _availability_as_utc(self.effective_at_panel, name="effective_at")
        known = _availability_as_utc(self.known_at_panel, name="known_at")
        cutoff = pd.Series(index.tz_convert("UTC"), index=index)
        visible = effective.le(cutoff, axis=0) & known.le(cutoff, axis=0)
        return visible.fillna(False).astype(bool)

    def verify_content(self) -> None:
        payload = _view_payload(
            snapshot_id=self.snapshot_id,
            schema_hash=self.schema_hash,
            frequency_hash=self.frequency_hash,
            availability_hash=self.availability_hash,
            security_contract_hash=self.security_contract_hash,
            production_ready=self.production_ready,
            degraded_reasons=self.degraded_reasons,
            fields=self.fields,
            history_mask=self.history_mask,
            cross_section_mask=self.cross_section_mask,
            effective_at_panel=self.effective_at_panel,
            known_at_panel=self.known_at_panel,
            strict_point_in_time=self.strict_point_in_time,
        )
        if hash_json(payload) != self.view_hash:
            raise RuntimeError("factor data view content changed after construction")


def _combined_assurance(
    *,
    security_contract_ready: bool,
    security_contract_reasons: tuple[str, ...],
    upstream_production_ready: bool,
    upstream_degraded_reasons: tuple[str, ...],
) -> tuple[bool, tuple[str, ...]]:
    if not isinstance(upstream_production_ready, bool):
        raise TypeError("factor upstream production readiness must be boolean")
    reasons = tuple(
        dict.fromkeys(
            (
                *(str(item) for item in security_contract_reasons),
                *(str(item) for item in upstream_degraded_reasons),
            )
        )
    )
    if any(not item or item != item.strip() for item in reasons):
        raise ValueError("factor degraded reasons must be canonical and non-empty")
    production_ready = bool(security_contract_ready and upstream_production_ready)
    if not production_ready and not reasons:
        raise ValueError("degraded factor view requires at least one reason")
    if production_ready and reasons:
        raise ValueError("production-ready factor view cannot carry degraded reasons")
    return production_ready, reasons


def _complete_boolean_mask(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    values = pd.DataFrame(frame).copy(deep=True)
    if values.isna().any().any() or any(
        str(dtype) not in {"bool", "boolean"} for dtype in values.dtypes
    ):
        raise TypeError(
            f"factor view {name} mask must be a complete boolean panel"
        )
    return values.astype(bool)


def _broadcast_daily_mask(
    mask: pd.DataFrame,
    timestamps: pd.Index,
    securities: pd.Index,
    *,
    timezone: str,
) -> pd.DataFrame:
    source = pd.DataFrame(mask).copy()
    source.index = [_date_text(item) for item in source.index]
    source.columns = [_security(item) for item in source.columns]
    if not source.index.is_unique or not source.columns.is_unique:
        raise ValueError("security contract mask axes are ambiguous")
    index = pd.DatetimeIndex(timestamps)
    if index.tz is None:
        raise ValueError("factor view timestamps must be timezone-aware")
    dates = index.tz_convert(timezone).strftime("%Y%m%d")
    columns = pd.Index([_security(item) for item in securities])
    aligned = source.reindex(index=dates, columns=columns).fillna(False).astype(bool)
    aligned.index = timestamps
    aligned.columns = securities
    return aligned


def _view_payload(
    *,
    snapshot_id: str,
    schema_hash: str,
    frequency_hash: str,
    availability_hash: str,
    security_contract_hash: str,
    production_ready: bool,
    degraded_reasons: tuple[str, ...],
    fields: Mapping[str, pd.DataFrame],
    history_mask: pd.DataFrame,
    cross_section_mask: pd.DataFrame,
    effective_at_panel: pd.DataFrame | None = None,
    known_at_panel: pd.DataFrame | None = None,
    strict_point_in_time: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "snapshot_id": snapshot_id,
        "schema_hash": schema_hash,
        "frequency_hash": frequency_hash,
        "availability_hash": availability_hash,
        "security_contract_hash": security_contract_hash,
        "production_ready": production_ready,
        "degraded_reasons": list(degraded_reasons),
        "fields": {name: hash_frame(value) for name, value in sorted(fields.items())},
        "history_mask": hash_frame(history_mask),
        "cross_section_mask": hash_frame(cross_section_mask),
    }
    if effective_at_panel is not None or known_at_panel is not None:
        if effective_at_panel is None or known_at_panel is None:
            raise ValueError("factor availability panels must be paired")
        payload.update(
            {
                "effective_at_panel": hash_frame(effective_at_panel),
                "known_at_panel": hash_frame(known_at_panel),
                "strict_point_in_time": bool(strict_point_in_time),
            }
        )
    elif strict_point_in_time:
        payload["strict_point_in_time"] = True
    return payload


def _availability_as_utc(panel: pd.DataFrame, *, name: str) -> pd.DataFrame:
    converted = pd.DataFrame(index=panel.index, columns=panel.columns)
    for column in panel.columns:
        values = panel[column]
        for value in values.dropna():
            timestamp = pd.Timestamp(value)
            if timestamp.tzinfo is None:
                raise ValueError(f"factor {name} timestamps must be timezone-aware")
        converted[column] = pd.to_datetime(values, errors="raise", utc=True)
    return converted


def _date_text(value: object) -> str:
    text = str(value).strip().replace("-", "").replace("/", "").replace(".", "")
    return (
        text[:8]
        if len(text) >= 8 and text[:8].isdigit()
        else pd.Timestamp(value).strftime("%Y%m%d")
    )


def _security(value: object) -> str:
    text = str(value).strip()
    return text.zfill(6) if text.isdigit() and len(text) <= 6 else text


__all__ = ["FactorDataView"]
