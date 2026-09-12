from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import pandas as pd
from pandas.api.types import is_bool_dtype

from llm_alpha_mining.research.core.hashing import hash_frame, hash_json
from llm_alpha_mining.research.data.views import MarketDataView
from llm_alpha_mining.research.labels.spec import PriceObservationEvent
from llm_alpha_mining.mining.data.security_contract import (
    SecurityDataContract,
    security_data_contract_hash,
)


@dataclass(frozen=True, slots=True)
class LabelDataView:
    snapshot_id: str
    schema_hash: str
    frequency_hash: str
    availability_hash: str
    security_contract_hash: str
    production_ready: bool
    fields: Mapping[str, pd.DataFrame]
    field_observation_events: Mapping[str, PriceObservationEvent | str]
    instrument_identity_mask: pd.DataFrame
    execution_tradability_mask: pd.DataFrame
    view_hash: str

    def __post_init__(self) -> None:
        fields = {
            name: pd.DataFrame(value).copy(deep=True)
            for name, value in self.fields.items()
        }
        if not fields:
            raise ValueError("label data view requires fields")
        observation_events = {
            str(name): PriceObservationEvent(event)
            for name, event in self.field_observation_events.items()
        }
        if not observation_events or any(
            not name.strip() for name in observation_events
        ):
            raise ValueError("label data view requires named price observation events")
        unknown_semantics = sorted(set(observation_events).difference(fields))
        if unknown_semantics:
            raise ValueError(
                "label observation semantics reference unavailable fields:"
                + ",".join(unknown_semantics)
            )
        reference = next(iter(fields.values()))
        identity = _complete_boolean_panel(
            self.instrument_identity_mask,
            name="instrument identity mask",
        )
        execution = _complete_boolean_panel(
            self.execution_tradability_mask,
            name="execution tradability mask",
        )
        for name, value in fields.items():
            if not value.index.equals(reference.index) or not value.columns.equals(
                reference.columns
            ):
                raise ValueError(f"label view field axes differ:{name}")
        if not identity.index.equals(reference.index) or not identity.columns.equals(
            reference.columns
        ):
            raise ValueError("label identity mask axes differ")
        if not execution.index.equals(reference.index) or not execution.columns.equals(
            reference.columns
        ):
            raise ValueError("label execution mask axes differ")
        payload = _payload(
            snapshot_id=self.snapshot_id,
            schema_hash=self.schema_hash,
            frequency_hash=self.frequency_hash,
            availability_hash=self.availability_hash,
            security_contract_hash=self.security_contract_hash,
            production_ready=self.production_ready,
            fields=fields,
            observation_events=observation_events,
            identity=identity,
            execution=execution,
        )
        if hash_json(payload) != self.view_hash:
            raise ValueError("label view hash differs from content")
        object.__setattr__(self, "fields", MappingProxyType(fields))
        object.__setattr__(
            self,
            "field_observation_events",
            MappingProxyType(dict(sorted(observation_events.items()))),
        )
        object.__setattr__(self, "instrument_identity_mask", identity)
        object.__setattr__(self, "execution_tradability_mask", execution)

    @classmethod
    def from_market_view(
        cls,
        market: MarketDataView,
        security_contract: SecurityDataContract,
        *,
        required_fields: tuple[str, ...],
        field_observation_events: Mapping[str, PriceObservationEvent | str],
        timezone: str = "Asia/Shanghai",
        allow_degraded_research: bool = False,
    ) -> "LabelDataView":
        if not security_contract.audit.production_ready and not allow_degraded_research:
            raise PermissionError("label_view_security_contract_not_production_ready")
        fields = {
            name: market.field_panel(name)
            for name in tuple(sorted(set(required_fields)))
        }
        if not fields:
            raise ValueError("label view required_fields must not be empty")
        observation_events = {
            str(name): PriceObservationEvent(event)
            for name, event in field_observation_events.items()
        }
        unknown_semantics = sorted(set(observation_events).difference(fields))
        if unknown_semantics:
            raise ValueError(
                "label observation semantics reference unavailable fields:"
                + ",".join(unknown_semantics)
            )
        if not observation_events:
            raise ValueError("label view requires explicit price observation semantics")
        reference = next(iter(fields.values()))
        identity = _broadcast(
            security_contract.instrument_identity_mask,
            reference.index,
            reference.columns,
            timezone=timezone,
        )
        execution = _broadcast(
            security_contract.execution_tradability_mask,
            reference.index,
            reference.columns,
            timezone=timezone,
        )
        fields = {name: value.where(identity) for name, value in fields.items()}
        contract_hash = security_data_contract_hash(security_contract)
        payload = _payload(
            snapshot_id=market.snapshot_id,
            schema_hash=market.schema_hash,
            frequency_hash=market.frequency_hash,
            availability_hash=market.availability_hash,
            security_contract_hash=contract_hash,
            production_ready=security_contract.audit.production_ready,
            fields=fields,
            observation_events=observation_events,
            identity=identity,
            execution=execution,
        )
        return cls(
            snapshot_id=market.snapshot_id,
            schema_hash=market.schema_hash,
            frequency_hash=market.frequency_hash,
            availability_hash=market.availability_hash,
            security_contract_hash=contract_hash,
            production_ready=security_contract.audit.production_ready,
            fields=fields,
            field_observation_events=observation_events,
            instrument_identity_mask=identity,
            execution_tradability_mask=execution,
            view_hash=hash_json(payload),
        )

    def field_panel(self, field: str) -> pd.DataFrame:
        try:
            return self.fields[field].copy(deep=True)
        except KeyError:
            raise KeyError(f"label view field is unavailable:{field}") from None

    def field_observation_event(self, field: str) -> PriceObservationEvent:
        try:
            return PriceObservationEvent(self.field_observation_events[field])
        except KeyError:
            raise KeyError(
                f"label view field lacks observation-event semantics:{field}"
            ) from None

    def verify_content(self) -> None:
        payload = _payload(
            snapshot_id=self.snapshot_id,
            schema_hash=self.schema_hash,
            frequency_hash=self.frequency_hash,
            availability_hash=self.availability_hash,
            security_contract_hash=self.security_contract_hash,
            production_ready=self.production_ready,
            fields=self.fields,
            observation_events=self.field_observation_events,
            identity=self.instrument_identity_mask,
            execution=self.execution_tradability_mask,
        )
        if hash_json(payload) != self.view_hash:
            raise RuntimeError("label data view content changed after construction")


def _broadcast(
    source_mask: pd.DataFrame,
    timestamps: pd.Index,
    securities: pd.Index,
    *,
    timezone: str,
) -> pd.DataFrame:
    source = _complete_boolean_panel(source_mask, name="security source mask")
    source.index = [_date(item) for item in source.index]
    source.columns = [_security(item) for item in source.columns]
    if not source.index.is_unique or not source.columns.is_unique:
        raise ValueError("label security mask axes are ambiguous")
    index = pd.DatetimeIndex(timestamps)
    if index.tz is None:
        raise ValueError("label timestamps must be timezone-aware")
    dates = index.tz_convert(timezone).strftime("%Y%m%d")
    columns = pd.Index([_security(item) for item in securities])
    result = source.reindex(index=dates, columns=columns, fill_value=False)
    result.index = timestamps
    result.columns = securities
    return result


def _complete_boolean_panel(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    values = pd.DataFrame(frame).copy(deep=True)
    if values.isna().to_numpy().any() or any(
        not is_bool_dtype(dtype) for dtype in values.dtypes
    ):
        raise TypeError(f"label {name} must be a complete boolean panel")
    return values.astype(bool)


def _payload(
    *,
    snapshot_id: str,
    schema_hash: str,
    frequency_hash: str,
    availability_hash: str,
    security_contract_hash: str,
    production_ready: bool,
    fields: Mapping[str, pd.DataFrame],
    observation_events: Mapping[str, PriceObservationEvent | str],
    identity: pd.DataFrame,
    execution: pd.DataFrame,
) -> dict[str, object]:
    return {
        "snapshot_id": snapshot_id,
        "schema_hash": schema_hash,
        "frequency_hash": frequency_hash,
        "availability_hash": availability_hash,
        "security_contract_hash": security_contract_hash,
        "production_ready": production_ready,
        "fields": {name: hash_frame(value) for name, value in sorted(fields.items())},
        "field_observation_events": {
            name: PriceObservationEvent(event).value
            for name, event in sorted(observation_events.items())
        },
        "instrument_identity_mask": hash_frame(identity),
        "execution_tradability_mask": hash_frame(execution),
    }


def _date(value: object) -> str:
    text = str(value).strip().replace("-", "").replace("/", "").replace(".", "")
    return (
        text[:8]
        if len(text) >= 8 and text[:8].isdigit()
        else pd.Timestamp(value).strftime("%Y%m%d")
    )


def _security(value: object) -> str:
    text = str(value).strip()
    return text.zfill(6) if text.isdigit() and len(text) <= 6 else text


__all__ = ["LabelDataView"]
