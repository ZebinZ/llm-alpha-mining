from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_research.core.data import DataBatch, DataSchema
from alpha_research.core.frequency import TradingCalendar
from alpha_research.core.hashing import canonical_json_bytes, hash_json


class QualitySeverity(str, Enum):
    WARNING = "warning"
    ERROR = "error"


class QualityDecision(str, Enum):
    ACCEPTED = "accepted"
    QUARANTINED = "quarantined"


@dataclass(frozen=True, slots=True)
class DataQualityIssue:
    code: str
    severity: QualitySeverity
    field: str | None
    count: int
    detail: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", QualitySeverity(self.severity))
        if not self.code or self.count <= 0 or not self.detail:
            raise ValueError("invalid data quality issue")

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "field": self.field,
            "count": self.count,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class DataQualityPolicy:
    maximum_missing_ratio: float = 0.25
    maximum_warning_missing_ratio: float = 0.05
    require_sorted_keys: bool = True
    reject_infinity: bool = True
    reject_out_of_session: bool = True

    def __post_init__(self) -> None:
        if (
            not 0
            <= self.maximum_warning_missing_ratio
            <= self.maximum_missing_ratio
            <= 1
        ):
            raise ValueError("missing-ratio thresholds are invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "maximum_missing_ratio": self.maximum_missing_ratio,
            "maximum_warning_missing_ratio": self.maximum_warning_missing_ratio,
            "require_sorted_keys": self.require_sorted_keys,
            "reject_infinity": self.reject_infinity,
            "reject_out_of_session": self.reject_out_of_session,
        }


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    report_id: str
    batch_id: str
    snapshot_id: str
    schema_hash: str
    policy_hash: str
    row_count: int
    duplicate_key_count: int
    issues: tuple[DataQualityIssue, ...]
    field_missing_ratio: tuple[tuple[str, float], ...]

    @property
    def decision(self) -> QualityDecision:
        return (
            QualityDecision.QUARANTINED
            if any(issue.severity is QualitySeverity.ERROR for issue in self.issues)
            else QualityDecision.ACCEPTED
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "snapshot_id": self.snapshot_id,
            "schema_hash": self.schema_hash,
            "policy_hash": self.policy_hash,
            "row_count": self.row_count,
            "duplicate_key_count": self.duplicate_key_count,
            "issues": [issue.to_dict() for issue in self.issues],
            "field_missing_ratio": [list(item) for item in self.field_missing_ratio],
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "report_id": self.report_id,
            "decision": self.decision.value,
        }


@dataclass(frozen=True, slots=True)
class QualityReceipt:
    receipt_id: str
    report_id: str
    batch_id: str
    snapshot_id: str
    decision: QualityDecision
    quarantined_artifact: str | None

    @classmethod
    def create(
        cls,
        report: DataQualityReport,
        *,
        quarantined_artifact: str | None = None,
    ) -> "QualityReceipt":
        payload = {
            "report_id": report.report_id,
            "batch_id": report.batch_id,
            "snapshot_id": report.snapshot_id,
            "decision": report.decision.value,
            "quarantined_artifact": quarantined_artifact,
        }
        return cls(
            receipt_id=hash_json(payload),
            report_id=report.report_id,
            batch_id=report.batch_id,
            snapshot_id=report.snapshot_id,
            decision=report.decision,
            quarantined_artifact=quarantined_artifact,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "receipt_id": self.receipt_id,
            "report_id": self.report_id,
            "batch_id": self.batch_id,
            "snapshot_id": self.snapshot_id,
            "decision": self.decision.value,
            "quarantined_artifact": self.quarantined_artifact,
        }


class DataQualityEngine:
    def __init__(self, policy: DataQualityPolicy | None = None):
        self.policy = policy or DataQualityPolicy()

    def evaluate(
        self,
        batch: DataBatch,
        *,
        schema: DataSchema,
        calendar: TradingCalendar,
    ) -> DataQualityReport:
        batch.verify_content()
        if batch.schema_hash != schema.content_hash:
            raise ValueError("quality schema does not match batch")
        if batch.calendar_id != calendar.calendar_id:
            raise ValueError("quality calendar does not match batch")
        frame = batch.frame
        issues: list[DataQualityIssue] = []
        key_fields = list(schema.key_fields or ())
        uses_legacy_bar_key = key_fields == [
            schema.timestamp_field,
            schema.security_field,
        ]
        duplicate_count = int(frame.duplicated(key_fields, keep=False).sum())
        if duplicate_count:
            issues.append(
                DataQualityIssue(
                    (
                        "duplicate_timestamp_security"
                        if uses_legacy_bar_key
                        else "duplicate_schema_key"
                    ),
                    QualitySeverity.ERROR,
                    None,
                    duplicate_count,
                    "schema key fields must identify unique records",
                )
            )
        if self.policy.require_sorted_keys and not frame.equals(
            frame.sort_values(key_fields, kind="stable").reset_index(drop=True)
        ):
            issues.append(
                DataQualityIssue(
                    (
                        "unsorted_timestamp_security"
                        if uses_legacy_bar_key
                        else "unsorted_schema_key"
                    ),
                    QualitySeverity.ERROR,
                    None,
                    max(1, len(frame)),
                    "records must be sorted by the declared schema key fields",
                )
            )

        timestamps = frame[schema.timestamp_field]
        if getattr(timestamps.dt, "tz", None) is None:
            issues.append(
                DataQualityIssue(
                    "naive_timestamp",
                    QualitySeverity.ERROR,
                    schema.timestamp_field,
                    max(1, len(frame)),
                    "timestamps must carry an explicit timezone",
                )
            )
        else:
            out_of_session = 0
            wrong_timezone = str(timestamps.dt.tz) != calendar.timezone
            for timestamp in timestamps.drop_duplicates():
                try:
                    calendar.assert_timestamp(pd.Timestamp(timestamp))
                except ValueError:
                    out_of_session += int((timestamps == timestamp).sum())
            if wrong_timezone:
                issues.append(
                    DataQualityIssue(
                        "timestamp_timezone_mismatch",
                        QualitySeverity.ERROR,
                        schema.timestamp_field,
                        max(1, len(frame)),
                        f"expected timezone {calendar.timezone}, got {timestamps.dt.tz}",
                    )
                )
            if out_of_session:
                issues.append(
                    DataQualityIssue(
                        "timestamp_outside_calendar_or_session",
                        (
                            QualitySeverity.ERROR
                            if self.policy.reject_out_of_session
                            else QualitySeverity.WARNING
                        ),
                        schema.timestamp_field,
                        out_of_session,
                        "records fall outside the declared trading calendar/session",
                    )
                )

        ratios: list[tuple[str, float]] = []
        for field_name in batch.requested_fields:
            field = schema.field_map[field_name]
            series = frame[field_name]
            if field.applicability_field is not None:
                applicable = frame[field.applicability_field].fillna(False).astype(bool)
                quality_series = series.loc[applicable]
            else:
                quality_series = series
            ratio = float(quality_series.isna().mean()) if len(quality_series) else 0.0
            ratios.append((field_name, ratio))
            if ratio > self.policy.maximum_missing_ratio:
                issues.append(
                    DataQualityIssue(
                        "excessive_missing_values",
                        QualitySeverity.ERROR,
                        field_name,
                        max(1, int(quality_series.isna().sum())),
                        f"missing ratio {ratio:.6f} exceeds {self.policy.maximum_missing_ratio:.6f}",
                    )
                )
            elif ratio > self.policy.maximum_warning_missing_ratio:
                issues.append(
                    DataQualityIssue(
                        "elevated_missing_values",
                        QualitySeverity.WARNING,
                        field_name,
                        max(1, int(quality_series.isna().sum())),
                        f"missing ratio {ratio:.6f} exceeds warning threshold",
                    )
                )
            if pd.api.types.is_numeric_dtype(series.dtype):
                numeric = pd.to_numeric(quality_series, errors="coerce")
                infinity_count = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
                if infinity_count and self.policy.reject_infinity:
                    issues.append(
                        DataQualityIssue(
                            "infinite_numeric_value",
                            QualitySeverity.ERROR,
                            field_name,
                            infinity_count,
                            "positive/negative infinity is forbidden",
                        )
                    )
                finite = numeric[np.isfinite(numeric)]
                if field.minimum is not None:
                    count = int((finite < field.minimum).sum())
                    if count:
                        issues.append(
                            DataQualityIssue(
                                "below_field_minimum",
                                QualitySeverity.ERROR,
                                field_name,
                                count,
                                f"values are below {field.minimum}",
                            )
                        )
                if field.maximum is not None:
                    count = int((finite > field.maximum).sum())
                    if count:
                        issues.append(
                            DataQualityIssue(
                                "above_field_maximum",
                                QualitySeverity.ERROR,
                                field_name,
                                count,
                                f"values are above {field.maximum}",
                            )
                        )

        ohlc = {"Open", "High", "Low", "Close"}
        if ohlc.issubset(frame.columns):
            open_price = pd.to_numeric(frame["Open"], errors="coerce")
            high_price = pd.to_numeric(frame["High"], errors="coerce")
            low_price = pd.to_numeric(frame["Low"], errors="coerce")
            close_price = pd.to_numeric(frame["Close"], errors="coerce")
            inconsistent = (
                (high_price < low_price)
                | (high_price < open_price)
                | (high_price < close_price)
                | (low_price > open_price)
                | (low_price > close_price)
            )
            count = int(inconsistent.fillna(False).sum())
            if count:
                issues.append(
                    DataQualityIssue(
                        "inconsistent_ohlc_envelope",
                        QualitySeverity.ERROR,
                        None,
                        count,
                        "OHLC rows must satisfy low <= open/close <= high",
                    )
                )

        if "Adj" in frame.columns:
            ordered = frame.sort_values(
                [schema.security_field, schema.timestamp_field], kind="stable"
            )
            adjustment = pd.to_numeric(ordered["Adj"], errors="coerce")
            previous = adjustment.groupby(
                ordered[schema.security_field], sort=False
            ).shift(1)
            ratio = pd.concat(
                [adjustment / previous, previous / adjustment], axis=1
            ).max(axis=1)
            count = int((ratio > 100.0).fillna(False).sum())
            if count:
                issues.append(
                    DataQualityIssue(
                        "implausible_adjustment_step",
                        QualitySeverity.ERROR,
                        "Adj",
                        count,
                        "adjacent adjustment-factor ratio exceeds 100x",
                    )
                )

        issues_tuple = tuple(
            sorted(
                issues,
                key=lambda item: (item.severity.value, item.code, item.field or ""),
            )
        )
        payload = {
            "batch_id": batch.batch_id,
            "snapshot_id": batch.snapshot_id,
            "schema_hash": schema.content_hash,
            "policy_hash": hash_json(self.policy.to_dict()),
            "row_count": len(frame),
            "duplicate_key_count": duplicate_count,
            "issues": [issue.to_dict() for issue in issues_tuple],
            "field_missing_ratio": [list(item) for item in ratios],
        }
        return DataQualityReport(
            report_id=hash_json(payload),
            batch_id=batch.batch_id,
            snapshot_id=batch.snapshot_id,
            schema_hash=schema.content_hash,
            policy_hash=hash_json(self.policy.to_dict()),
            row_count=len(frame),
            duplicate_key_count=duplicate_count,
            issues=issues_tuple,
            field_missing_ratio=tuple(ratios),
        )


class QuarantineStore:
    """Persist rejected data and its receipt without mutating the source."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        batch: DataBatch,
        report: DataQualityReport,
    ) -> QualityReceipt:
        artifact: str | None = None
        if report.decision is QualityDecision.QUARANTINED:
            directory = self.root / report.report_id
            directory.mkdir(exist_ok=False)
            data_path = directory / "records.csv"
            batch.frame.to_csv(data_path, index=False)
            artifact = str(data_path)
        receipt = QualityReceipt.create(report, quarantined_artifact=artifact)
        receipt_path = self.root / f"{receipt.receipt_id}.json"
        receipt_path.write_bytes(
            canonical_json_bytes(
                {
                    "receipt": receipt.to_dict(),
                    "report": report.to_dict(),
                    "batch": batch.to_descriptor(),
                }
            )
            + b"\n"
        )
        return receipt


__all__ = [
    "DataQualityEngine",
    "DataQualityIssue",
    "DataQualityPolicy",
    "DataQualityReport",
    "QualityDecision",
    "QualityReceipt",
    "QualitySeverity",
    "QuarantineStore",
]
