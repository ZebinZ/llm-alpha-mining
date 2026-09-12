from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import re
from typing import cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from alpha_research.core.hashing import hash_frame, hash_json
from alpha_research.high_frequency.timestamp_format import (
    local_session_dates_yyyymmdd,
)


class SnapshotRemediationError(ValueError):
    """Fail-closed error raised by the explicit snapshot remediation stage."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}:{self.detail}")


class SnapshotRemediationMode(str, Enum):
    MASK_TO_NULL = "mask_to_null"
    BOUNDED_PAST_FILL = "bounded_past_fill"
    ORDERFLOW_RECONSTRUCT = "orderflow_reconstruct"


SNAPSHOT_ECONOMIC_FIELDS = (
    "pre_close",
    "open",
    "high",
    "low",
    "latest",
    "cumulative_volume",
    "cumulative_amount",
    "cumulative_trade_count",
    *(f"bid_price_{level}" for level in range(1, 6)),
    *(f"bid_size_{level}" for level in range(1, 6)),
    *(f"ask_price_{level}" for level in range(1, 6)),
    *(f"ask_size_{level}" for level in range(1, 6)),
)


@dataclass(frozen=True, slots=True)
class SnapshotRemediationSpec:
    """Versioned research-only policy for rows already quarantined by QC.

    The baseline is ``mask_to_null``.  Bounded past-fill is an explicit
    sensitivity analysis and never restores production eligibility.  True
    order-flow reconstruction belongs to a separately authorized event-replay
    stage and is deliberately unavailable here.
    """

    mode: SnapshotRemediationMode | str = SnapshotRemediationMode.MASK_TO_NULL
    maximum_past_fill_gap: str = "3s"
    timezone: str = "Asia/Shanghai"
    schema_version: str = "snapshot-remediation-spec/v1"
    research_only: bool = True
    admission_claim: bool = False
    production_ready: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != "snapshot-remediation-spec/v1":
            raise ValueError("unsupported snapshot remediation specification")
        object.__setattr__(self, "mode", SnapshotRemediationMode(self.mode))
        if pd.Timedelta(self.maximum_past_fill_gap) <= pd.Timedelta(0):
            raise ValueError("snapshot remediation fill gap must be positive")
        ZoneInfo(self.timezone)
        if (
            self.research_only is not True
            or self.admission_claim is not False
            or self.production_ready is not False
        ):
            raise ValueError("snapshot remediation must remain research-only")

    @property
    def content_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        mode = self.mode
        if not isinstance(mode, SnapshotRemediationMode):  # pragma: no cover
            raise RuntimeError("snapshot remediation mode was not normalized")
        return {
            "schema_version": self.schema_version,
            "mode": mode.value,
            "maximum_past_fill_gap": self.maximum_past_fill_gap,
            "timezone": self.timezone,
            "economic_fields": list(SNAPSHOT_ECONOMIC_FIELDS),
            "preserve_quarantine": True,
            "research_only": self.research_only,
            "admission_claim": self.admission_claim,
            "production_ready": self.production_ready,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SnapshotRemediationSpec:
        required = {
            "schema_version",
            "mode",
            "maximum_past_fill_gap",
            "timezone",
            "economic_fields",
            "preserve_quarantine",
            "research_only",
            "admission_claim",
            "production_ready",
        }
        if set(value) != required:
            raise ValueError("snapshot remediation specification keys differ")
        fields = value["economic_fields"]
        if not isinstance(fields, list) or tuple(map(str, fields)) != (
            SNAPSHOT_ECONOMIC_FIELDS
        ):
            raise ValueError("snapshot remediation economic fields differ")
        if value["preserve_quarantine"] is not True:
            raise ValueError("snapshot remediation must preserve quarantine")
        result = cls(
            mode=str(value["mode"]),
            maximum_past_fill_gap=str(value["maximum_past_fill_gap"]),
            timezone=str(value["timezone"]),
            schema_version=str(value["schema_version"]),
            research_only=value["research_only"] is True,
            admission_claim=value["admission_claim"] is True,
            production_ready=value["production_ready"] is True,
        )
        if result.to_dict() != dict(value):
            raise ValueError("snapshot remediation specification is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class SnapshotRemediationReceipt:
    receipt_id: str
    spec_hash: str
    input_frame_hash: str
    output_frame_hash: str
    input_rows: int
    output_rows: int
    quarantined_rows: int
    remediated_rows: int
    masked_rows: int
    past_filled_rows: int
    unresolved_rows: int
    mode: str
    research_only: bool = True
    admission_claim: bool = False
    production_ready: bool = False
    schema_version: str = "snapshot-remediation-receipt/v1"

    def __post_init__(self) -> None:
        for name in (
            "receipt_id",
            "spec_hash",
            "input_frame_hash",
            "output_frame_hash",
            "mode",
            "schema_version",
        ):
            if type(getattr(self, name)) is not str:
                raise TypeError(
                    f"snapshot remediation receipt {name} must be a string"
                )
        if self.schema_version != "snapshot-remediation-receipt/v1":
            raise ValueError("unsupported snapshot remediation receipt")
        if re.fullmatch(r"[0-9a-f]{64}", self.spec_hash) is None:
            raise ValueError("snapshot remediation spec hash is invalid")
        if re.fullmatch(r"[0-9a-f]{64}", self.input_frame_hash) is None:
            raise ValueError("snapshot remediation input hash is invalid")
        if re.fullmatch(r"[0-9a-f]{64}", self.output_frame_hash) is None:
            raise ValueError("snapshot remediation output hash is invalid")
        SnapshotRemediationMode(self.mode)
        counts = (
            self.input_rows,
            self.output_rows,
            self.quarantined_rows,
            self.remediated_rows,
            self.masked_rows,
            self.past_filled_rows,
            self.unresolved_rows,
        )
        if any(type(value) is not int for value in counts):
            raise TypeError("snapshot remediation counts must be exact integers")
        if any(value < 0 for value in counts):
            raise ValueError("snapshot remediation counts must be nonnegative")
        if self.input_rows != self.output_rows:
            raise ValueError("snapshot remediation must preserve row count")
        if self.remediated_rows + self.unresolved_rows != self.quarantined_rows:
            raise ValueError("snapshot remediation quarantine accounting differs")
        if self.masked_rows + self.past_filled_rows != self.remediated_rows:
            raise ValueError("snapshot remediation method accounting differs")
        if (
            self.research_only is not True
            or self.admission_claim is not False
            or self.production_ready is not False
        ):
            raise ValueError("snapshot remediation receipt must remain research-only")
        if self.receipt_id != hash_json(self.identity_payload()):
            raise ValueError("snapshot remediation receipt id differs")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "spec_hash": self.spec_hash,
            "input_frame_hash": self.input_frame_hash,
            "output_frame_hash": self.output_frame_hash,
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "quarantined_rows": self.quarantined_rows,
            "remediated_rows": self.remediated_rows,
            "masked_rows": self.masked_rows,
            "past_filled_rows": self.past_filled_rows,
            "unresolved_rows": self.unresolved_rows,
            "mode": self.mode,
            "research_only": self.research_only,
            "admission_claim": self.admission_claim,
            "production_ready": self.production_ready,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.identity_payload(), "receipt_id": self.receipt_id}

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> SnapshotRemediationReceipt:
        """Rebuild one receipt from its exact canonical public field set."""

        expected = frozenset(cls.__dataclass_fields__)
        observed = frozenset(value)
        if observed != expected:
            raise ValueError(
                "snapshot remediation receipt fields differ; "
                f"missing={sorted(expected - observed)!r}; "
                f"extra={sorted(observed - expected)!r}"
            )

        text_fields = (
            "receipt_id",
            "spec_hash",
            "input_frame_hash",
            "output_frame_hash",
            "mode",
            "schema_version",
        )
        for name in text_fields:
            if not isinstance(value[name], str):
                raise TypeError(
                    f"snapshot remediation receipt {name} must be a string"
                )

        count_fields = (
            "input_rows",
            "output_rows",
            "quarantined_rows",
            "remediated_rows",
            "masked_rows",
            "past_filled_rows",
            "unresolved_rows",
        )
        for name in count_fields:
            item = value[name]
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(
                    f"snapshot remediation receipt {name} must be an integer"
                )

        boolean_fields = (
            "research_only",
            "admission_claim",
            "production_ready",
        )
        for name in boolean_fields:
            if type(value[name]) is not bool:
                raise TypeError(
                    f"snapshot remediation receipt {name} must be a boolean"
                )

        result = cls(
            receipt_id=cast(str, value["receipt_id"]),
            spec_hash=cast(str, value["spec_hash"]),
            input_frame_hash=cast(str, value["input_frame_hash"]),
            output_frame_hash=cast(str, value["output_frame_hash"]),
            input_rows=cast(int, value["input_rows"]),
            output_rows=cast(int, value["output_rows"]),
            quarantined_rows=cast(int, value["quarantined_rows"]),
            remediated_rows=cast(int, value["remediated_rows"]),
            masked_rows=cast(int, value["masked_rows"]),
            past_filled_rows=cast(int, value["past_filled_rows"]),
            unresolved_rows=cast(int, value["unresolved_rows"]),
            mode=cast(str, value["mode"]),
            research_only=cast(bool, value["research_only"]),
            admission_claim=cast(bool, value["admission_claim"]),
            production_ready=cast(bool, value["production_ready"]),
            schema_version=cast(str, value["schema_version"]),
        )
        if result.to_dict() != dict(value):
            raise ValueError("snapshot remediation receipt is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class SnapshotRemediationResult:
    frame: pd.DataFrame
    receipt: SnapshotRemediationReceipt

    def __post_init__(self) -> None:
        values = pd.DataFrame(self.frame).copy(deep=True)
        if hash_frame(values) != self.receipt.output_frame_hash:
            raise ValueError("snapshot remediation output hash differs")
        if len(values) != self.receipt.output_rows:
            raise ValueError("snapshot remediation output row count differs")
        object.__setattr__(self, "frame", values)


class SnapshotRemediator:
    """Apply one explicit, research-only disposition to quarantined rows."""

    _identity_fields = (
        "trading_date",
        "security",
        "venue",
        "session_phase",
        "source_file",
        "source_row_ordinal",
        "timestamp",
        "effective_at",
        "known_at",
    )
    _fill_partition_fields = (
        "trading_date",
        "security",
        "venue",
        "session_phase",
        "provider_regime_id",
    )

    def __init__(self, spec: SnapshotRemediationSpec | None = None) -> None:
        self.spec = spec or SnapshotRemediationSpec()

    def apply(self, frame: pd.DataFrame) -> SnapshotRemediationResult:
        values = pd.DataFrame(frame).copy(deep=True)
        self._validate(values)
        input_hash = hash_frame(values)
        mode = SnapshotRemediationMode(self.spec.mode)
        if mode is SnapshotRemediationMode.ORDERFLOW_RECONSTRUCT:
            raise SnapshotRemediationError(
                "orderflow_reconstruction_unavailable",
                "requires exact combined sequence, full LOB semantics, and a "
                "separately authorized event-replay executor",
            )
        quarantined = values["qc_row_quarantined"].astype(bool)
        values["remediation_mode"] = mode.value
        values["remediation_spec_hash"] = self.spec.content_hash
        values["remediation_applied"] = pd.Series(False, index=values.index, dtype=bool)
        values["remediation_unresolved"] = quarantined.copy()
        values["remediation_source_row_ordinal"] = pd.Series(
            pd.NA, index=values.index, dtype="Int64"
        )
        values["remediation_source_file"] = pd.Series(
            pd.NA, index=values.index, dtype="string"
        )
        values["remediation_research_eligible"] = (~quarantined).astype(bool)
        # This stage is not an eligibility override.  A caller may have
        # attached downstream gates already, so a quarantined row must not
        # retain an earlier true eligibility value after remediation.
        for field in (
            "research_eligible",
            "counter_research_eligible",
            "book_production_eligible",
            "counter_production_eligible",
            "production_eligible",
        ):
            if field in values:
                values.loc[quarantined, field] = False

        masked_rows = 0
        past_filled_rows = 0
        if mode is SnapshotRemediationMode.MASK_TO_NULL:
            values.loc[quarantined, SNAPSHOT_ECONOMIC_FIELDS] = np.nan
            values.loc[quarantined, "remediation_applied"] = True
            values.loc[quarantined, "remediation_unresolved"] = False
            masked_rows = int(quarantined.sum())
        elif mode is SnapshotRemediationMode.BOUNDED_PAST_FILL:
            filled = self._bounded_past_fill(values, quarantined)
            unresolved = quarantined & ~filled
            values.loc[unresolved, SNAPSHOT_ECONOMIC_FIELDS] = np.nan
            values.loc[filled | unresolved, "remediation_applied"] = True
            values.loc[filled | unresolved, "remediation_unresolved"] = False
            past_filled_rows = int(filled.sum())
            masked_rows = int(unresolved.sum())
        else:  # pragma: no cover - exhaustive enum guard
            raise RuntimeError("unsupported snapshot remediation mode")

        # A remediated value is useful only for sensitivity analysis.  It can
        # never silently restore content integrity or downstream eligibility.
        values.loc[quarantined, "source_content_integrity_verified"] = False
        remediated = values["remediation_applied"].astype(bool)
        unresolved = values["remediation_unresolved"].astype(bool)
        output_hash = hash_frame(values)
        payload = {
            "schema_version": "snapshot-remediation-receipt/v1",
            "spec_hash": self.spec.content_hash,
            "input_frame_hash": input_hash,
            "output_frame_hash": output_hash,
            "input_rows": len(frame),
            "output_rows": len(values),
            "quarantined_rows": int(quarantined.sum()),
            "remediated_rows": int(remediated.sum()),
            "masked_rows": masked_rows,
            "past_filled_rows": past_filled_rows,
            "unresolved_rows": int(unresolved.sum()),
            "mode": mode.value,
            "research_only": True,
            "admission_claim": False,
            "production_ready": False,
        }
        receipt = SnapshotRemediationReceipt(
            receipt_id=hash_json(payload),
            **payload,
        )
        return SnapshotRemediationResult(values, receipt)

    def _validate(self, values: pd.DataFrame) -> None:
        required = {
            *self._identity_fields,
            "qc_row_quarantined",
            "qc_quarantine_reasons",
            "qc_quarantine_evidence_id",
            "source_content_integrity_verified",
            *SNAPSHOT_ECONOMIC_FIELDS,
        }
        if self.spec.mode is SnapshotRemediationMode.BOUNDED_PAST_FILL:
            required.update(self._fill_partition_fields)
        missing = sorted(required.difference(values.columns))
        if missing:
            raise SnapshotRemediationError(
                "missing_fields", ",".join(missing)
            )
        if not values.index.is_unique:
            raise SnapshotRemediationError(
                "duplicate_frame_index", "frame index must be unique"
            )
        nonempty = [
            "trading_date",
            "security",
            "venue",
            "session_phase",
            "source_file",
        ]
        if self.spec.mode is SnapshotRemediationMode.BOUNDED_PAST_FILL:
            nonempty.extend(self._fill_partition_fields)
        if any(
            values[field].isna().any()
            or values[field].astype(str).str.strip().eq("").any()
            for field in dict.fromkeys(nonempty)
        ):
            raise SnapshotRemediationError(
                "invalid_identity", "identity and fill partition fields must be present"
            )
        quarantined = values["qc_row_quarantined"]
        if not is_bool_dtype(quarantined.dtype) or quarantined.isna().any():
            raise SnapshotRemediationError(
                "invalid_quarantine_flag", "must be non-null boolean"
            )
        reasons = values["qc_quarantine_reasons"]
        evidence = values["qc_quarantine_evidence_id"]
        if reasons.isna().any() or evidence.isna().any():
            raise SnapshotRemediationError(
                "invalid_quarantine_evidence", "must not be null"
            )
        reason_present = reasons.astype(str).ne("")
        evidence_text = evidence.astype(str)
        evidence_present = evidence_text.ne("")
        if (
            reason_present.ne(quarantined).any()
            or evidence_present.ne(quarantined).any()
            or (
                quarantined
                & ~evidence_text.str.fullmatch(r"[0-9a-f]{64}", na=False)
            ).any()
        ):
            raise SnapshotRemediationError(
                "invalid_quarantine_evidence", "flags and evidence differ"
            )
        timestamps = pd.to_datetime(values["timestamp"], errors="raise")
        effective = pd.to_datetime(values["effective_at"], errors="raise")
        known = pd.to_datetime(values["known_at"], errors="raise")
        if (
            timestamps.dt.tz is None
            or effective.dt.tz is None
            or known.dt.tz is None
        ):
            raise SnapshotRemediationError(
                "naive_timestamp",
                "timestamp, effective_at, and known_at must be timezone-aware",
            )
        if (effective != timestamps).any():
            raise SnapshotRemediationError(
                "invalid_effective_at", "effective_at must equal timestamp"
            )
        if (known < timestamps).any():
            raise SnapshotRemediationError(
                "invalid_availability", "known_at precedes timestamp"
            )
        local_date = local_session_dates_yyyymmdd(
            timestamps, timezone=self.spec.timezone
        )
        if values["trading_date"].astype(str).ne(local_date).any():
            raise SnapshotRemediationError(
                "invalid_trading_date", "trading_date differs from local timestamp date"
            )
        values["timestamp"] = timestamps
        values["effective_at"] = effective
        values["known_at"] = known
        integrity = values["source_content_integrity_verified"]
        if not is_bool_dtype(integrity.dtype) or integrity.isna().any():
            raise SnapshotRemediationError(
                "invalid_content_integrity",
                "source_content_integrity_verified must be non-null boolean",
            )
        ordinal = pd.to_numeric(values["source_row_ordinal"], errors="raise")
        if ordinal.isna().any() or (ordinal < 0).any() or ordinal.mod(1).ne(0).any():
            raise SnapshotRemediationError(
                "invalid_source_order", "source row ordinal is invalid"
            )
        values["source_row_ordinal"] = ordinal.astype("int64")
        source_key = [
            "trading_date",
            "source_file",
            "security",
            "venue",
            "source_row_ordinal",
        ]
        if values.duplicated(source_key, keep=False).any():
            raise SnapshotRemediationError(
                "duplicate_source_identity", "source row identity is duplicated"
            )
        ordered = values.sort_values(source_key, kind="stable")
        backward = ordered.groupby(
            ["trading_date", "source_file", "security", "venue"],
            sort=False,
            observed=True,
            dropna=False,
        )["timestamp"].diff().lt(pd.Timedelta(0))
        if backward.any():
            raise SnapshotRemediationError(
                "backward_timestamp", "timestamp moved backward in source order"
            )

    def _bounded_past_fill(
        self,
        values: pd.DataFrame,
        quarantined: pd.Series,
    ) -> pd.Series:
        filled = pd.Series(False, index=values.index, dtype=bool)
        maximum_gap = pd.Timedelta(self.spec.maximum_past_fill_gap)
        ordered = values.sort_values(
            [
                *self._fill_partition_fields,
                "timestamp",
                "known_at",
                "source_file",
                "source_row_ordinal",
            ],
            kind="stable",
        )
        groups = ordered.groupby(
            list(self._fill_partition_fields),
            sort=False,
            observed=True,
            dropna=False,
        )
        for _, group in groups:
            prior_index: object | None = None
            for index in group.index:
                if (
                    not bool(quarantined.loc[index])
                    and bool(values.loc[index, "source_content_integrity_verified"])
                ):
                    prior_index = index
                    continue
                if prior_index is None:
                    continue
                current_timestamp = pd.Timestamp(values.loc[index, "timestamp"])
                prior_timestamp = pd.Timestamp(values.loc[prior_index, "timestamp"])
                current_known = pd.Timestamp(values.loc[index, "known_at"])
                prior_known = pd.Timestamp(values.loc[prior_index, "known_at"])
                gap = current_timestamp - prior_timestamp
                if gap <= pd.Timedelta(0) or gap > maximum_gap:
                    continue
                if prior_known > current_known:
                    continue
                economic_fields = list(SNAPSHOT_ECONOMIC_FIELDS)
                values.loc[index, economic_fields] = values.loc[
                    prior_index, economic_fields
                ].to_numpy()
                values.loc[index, "remediation_source_row_ordinal"] = int(
                    values.loc[prior_index, "source_row_ordinal"]
                )
                values.loc[index, "remediation_source_file"] = str(
                    values.loc[prior_index, "source_file"]
                )
                filled.loc[index] = True
        return filled


__all__ = [
    "SNAPSHOT_ECONOMIC_FIELDS",
    "SnapshotRemediationError",
    "SnapshotRemediationMode",
    "SnapshotRemediationReceipt",
    "SnapshotRemediationResult",
    "SnapshotRemediationSpec",
    "SnapshotRemediator",
]
