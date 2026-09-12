from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, cast

import pandas as pd

from alpha_research.core.frequency import TradingCalendar
from alpha_research.core.hashing import hash_json, require_sha256
from alpha_research.labels import LabelResult
from alpha_research.validation.spec import ValidationFoldSpec, ValidationSpec


@dataclass(frozen=True, slots=True)
class SplitFoldReceipt:
    fold_id: str
    train_signals: tuple[str, ...]
    validation_signals: tuple[str, ...]
    purged_train_signals: tuple[str, ...]
    excluded_validation_signals: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fold_id, str) or not self.fold_id.strip():
            raise ValueError("split fold receipt requires a fold id")
        for name in (
            "train_signals",
            "validation_signals",
            "purged_train_signals",
            "excluded_validation_signals",
        ):
            values = tuple(getattr(self, name))
            if any(not isinstance(item, str) or not item.strip() for item in values):
                raise TypeError(f"split fold {name} must contain non-empty text")
            if len(values) != len(set(values)):
                raise ValueError(f"split fold {name} must be unique")
            object.__setattr__(self, name, values)

    def to_dict(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "train_signals": list(self.train_signals),
            "validation_signals": list(self.validation_signals),
            "purged_train_signals": list(self.purged_train_signals),
            "excluded_validation_signals": list(self.excluded_validation_signals),
        }

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "SplitFoldReceipt":
        expected = {
            "fold_id",
            "train_signals",
            "validation_signals",
            "purged_train_signals",
            "excluded_validation_signals",
        }
        if set(value) != expected:
            raise ValueError("SplitFoldReceipt wire fields differ")
        return cls(
            fold_id=_text(value["fold_id"], "fold_id"),
            train_signals=_text_tuple(value["train_signals"], "train_signals"),
            validation_signals=_text_tuple(
                value["validation_signals"], "validation_signals"
            ),
            purged_train_signals=_text_tuple(
                value["purged_train_signals"], "purged_train_signals"
            ),
            excluded_validation_signals=_text_tuple(
                value["excluded_validation_signals"],
                "excluded_validation_signals",
            ),
        )


@dataclass(frozen=True, slots=True)
class ValidationReceipt:
    validation_spec_hash: str
    label_spec_hash: str
    labels_hash: str
    windows_hash: str
    calendar_hash: str
    folds: tuple[SplitFoldReceipt, ...]
    schema_version: str = "validation-receipt/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "validation-receipt/v1":
            raise ValueError("unsupported validation receipt schema")
        for name in (
            "validation_spec_hash",
            "label_spec_hash",
            "labels_hash",
            "windows_hash",
            "calendar_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"validation receipt {name}")
        folds = tuple(self.folds)
        if not all(isinstance(fold, SplitFoldReceipt) for fold in folds):
            raise TypeError("validation receipt folds must be split fold receipts")
        if not folds:
            raise ValueError("validation receipt requires folds")
        if len({fold.fold_id for fold in folds}) != len(folds):
            raise ValueError("validation receipt fold ids must be unique")
        object.__setattr__(self, "folds", folds)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "validation_spec_hash": self.validation_spec_hash,
            "label_spec_hash": self.label_spec_hash,
            "labels_hash": self.labels_hash,
            "windows_hash": self.windows_hash,
            "calendar_hash": self.calendar_hash,
            "folds": [fold.to_dict() for fold in self.folds],
        }

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ValidationReceipt":
        expected = {
            "schema_version",
            "validation_spec_hash",
            "label_spec_hash",
            "labels_hash",
            "windows_hash",
            "calendar_hash",
            "folds",
        }
        if set(value) != expected:
            raise ValueError("ValidationReceipt wire fields differ")
        raw_folds = value["folds"]
        if not isinstance(raw_folds, list) or not all(
            isinstance(item, Mapping) for item in raw_folds
        ):
            raise TypeError("validation receipt folds must be an object array")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            validation_spec_hash=_text(
                value["validation_spec_hash"], "validation_spec_hash"
            ),
            label_spec_hash=_text(value["label_spec_hash"], "label_spec_hash"),
            labels_hash=_text(value["labels_hash"], "labels_hash"),
            windows_hash=_text(value["windows_hash"], "windows_hash"),
            calendar_hash=_text(value["calendar_hash"], "calendar_hash"),
            folds=tuple(SplitFoldReceipt.from_mapping(item) for item in raw_folds),
        )


class ValidationReceiptVerifier:
    """Recompute a validation receipt from its authoritative inputs.

    Comparing only caller-supplied hashes is insufficient because a receipt
    can contain a different set of train/validation timestamps while retaining
    otherwise valid label and specification bindings.  This verifier rebuilds
    every fold from the immutable specification, label windows and calendar,
    then requires the complete receipt to match.
    """

    def verify(
        self,
        receipt: ValidationReceipt,
        *,
        spec: ValidationSpec,
        labels: LabelResult,
        calendar: TradingCalendar,
    ) -> ValidationReceipt:
        expected = PurgedWalkForwardSplitter().split(spec, labels, calendar)
        if receipt.content_hash != expected.content_hash:
            raise ValueError("validation_receipt_recomputation_mismatch")
        return expected


def validation_calendar_hash(calendar: TradingCalendar) -> str:
    """Return the legacy projection committed by ValidationReceipt v1.

    This deliberately differs from ``TradingCalendar.content_hash`` because
    the receipt predates the calendar's nested ``session`` wire.  Publishing
    the projection in one place prevents validators from accidentally treating
    the two identities as interchangeable.
    """

    if not isinstance(calendar, TradingCalendar):
        raise TypeError("validation calendar hash requires TradingCalendar")
    return hash_json(
        {
            "calendar_id": calendar.calendar_id,
            "timezone": calendar.timezone,
            "sessions": list(calendar.sessions),
            "session_id": calendar.session.session_id,
            "intervals": [list(item) for item in calendar.session.intervals],
        }
    )


class PurgedWalkForwardSplitter:
    """Create chronological folds using the realized label intervals.

    A signal is admitted only when its complete label window is contained in
    the requested role period.  This prevents a late training signal from
    borrowing outcomes that occur in validation.  The embargo is an explicit
    number of calendar sessions between the two role periods.
    """

    def split(
        self,
        spec: ValidationSpec,
        labels: LabelResult,
        calendar: TradingCalendar,
    ) -> ValidationReceipt:
        labels.verify_content()
        windows = _normalize_windows(labels)
        calendar_positions = _calendar_positions(calendar)
        calendar_hash = validation_calendar_hash(calendar)
        receipts = tuple(
            self._split_fold(
                fold,
                spec=spec,
                windows=windows,
                timezone=calendar.timezone,
                calendar_positions=calendar_positions,
            )
            for fold in spec.folds
        )
        return ValidationReceipt(
            validation_spec_hash=spec.content_hash,
            label_spec_hash=labels.label_spec_hash,
            labels_hash=labels.labels_hash,
            windows_hash=labels.windows_hash,
            calendar_hash=calendar_hash,
            folds=receipts,
        )

    @staticmethod
    def _split_fold(
        fold: ValidationFoldSpec,
        *,
        spec: ValidationSpec,
        windows: pd.DataFrame,
        timezone: str,
        calendar_positions: dict[str, int],
    ) -> SplitFoldReceipt:
        train_start = pd.Timestamp(fold.train_start)
        train_end = pd.Timestamp(fold.train_end)
        validation_start = pd.Timestamp(fold.validation_start)
        validation_end = pd.Timestamp(fold.validation_end)
        _require_embargo(
            train_end,
            validation_start,
            embargo_sessions=spec.embargo_sessions,
            timezone=timezone,
            calendar_positions=calendar_positions,
        )

        train_candidate = windows.loc[
            windows["signal_timestamp"].between(train_start, train_end)
        ]
        train_keep = (train_candidate["label_start"] >= train_start) & (
            train_candidate["label_end"] <= train_end
        )
        validation_candidate = windows.loc[
            windows["signal_timestamp"].between(validation_start, validation_end)
        ]
        validation_keep = (validation_candidate["label_start"] >= validation_start) & (
            validation_candidate["label_end"] <= validation_end
        )
        train = train_candidate.loc[train_keep]
        validation = validation_candidate.loc[validation_keep]

        # Defensive interval purge.  Walk-forward periods are already disjoint,
        # but this makes the non-overlap condition part of the receipt logic.
        if not validation.empty and not train.empty:
            validation_interval_start = validation["label_start"].min()
            validation_interval_end = validation["label_end"].max()
            overlap = (train["label_start"] < validation_interval_end) & (
                train["label_end"] > validation_interval_start
            )
            train_keep_indices = train.index[~overlap]
            purged_overlap_indices = train.index[overlap]
            train = train.loc[train_keep_indices]
        else:
            purged_overlap_indices = pd.Index([])

        purged_indices = train_candidate.index[~train_keep].union(
            purged_overlap_indices
        )
        excluded_validation = validation_candidate.index[~validation_keep]
        if len(train) < spec.min_train_signals:
            raise ValueError(f"insufficient_train_signals:{fold.fold_id}:{len(train)}")
        if len(validation) < spec.min_validation_signals:
            raise ValueError(
                f"insufficient_validation_signals:{fold.fold_id}:{len(validation)}"
            )
        return SplitFoldReceipt(
            fold_id=fold.fold_id,
            train_signals=_iso_tuple(train["signal_timestamp"]),
            validation_signals=_iso_tuple(validation["signal_timestamp"]),
            purged_train_signals=_iso_tuple(
                windows.loc[purged_indices, "signal_timestamp"]
            ),
            excluded_validation_signals=_iso_tuple(
                windows.loc[excluded_validation, "signal_timestamp"]
            ),
        )


def _normalize_windows(labels: LabelResult) -> pd.DataFrame:
    required = {"signal_timestamp", "label_start", "label_end"}
    missing = sorted(required.difference(labels.label_windows.columns))
    if missing:
        raise ValueError("label windows are missing fields:" + ",".join(missing))
    windows = labels.label_windows.copy(deep=True)
    for name in sorted(required):
        values = pd.to_datetime(windows[name], errors="raise")
        if values.dt.tz is None:
            raise ValueError(f"label window {name} must be timezone-aware")
        windows[name] = values
    if windows["signal_timestamp"].duplicated().any():
        raise ValueError("label windows contain duplicate signals")
    label_index = pd.DatetimeIndex(labels.labels.index)
    if label_index.tz is None:
        raise ValueError("label result index must be timezone-aware")
    if set(windows["signal_timestamp"]) != set(label_index):
        raise ValueError("label values and window signals differ")
    if (windows["label_start"] < windows["signal_timestamp"]).any():
        raise ValueError("label starts before its signal")
    if (windows["label_end"] <= windows["label_start"]).any():
        raise ValueError("label interval must be positive")
    return windows.sort_values("signal_timestamp", kind="stable").reset_index(drop=True)


def _calendar_positions(calendar: TradingCalendar) -> dict[str, int]:
    return {session: offset for offset, session in enumerate(calendar.sessions)}


def _require_embargo(
    train_end: pd.Timestamp,
    validation_start: pd.Timestamp,
    *,
    embargo_sessions: int,
    timezone: str,
    calendar_positions: dict[str, int],
) -> None:
    train_date = train_end.tz_convert(timezone).strftime("%Y%m%d")
    validation_date = validation_start.tz_convert(timezone).strftime("%Y%m%d")
    try:
        gap = calendar_positions[validation_date] - calendar_positions[train_date] - 1
    except KeyError as error:
        raise ValueError(
            f"validation boundary is outside calendar:{error.args[0]}"
        ) from None
    if gap < embargo_sessions:
        raise ValueError(
            f"embargo_not_satisfied:required={embargo_sessions}:actual={gap}"
        )


def _iso_tuple(values: pd.Series) -> tuple[str, ...]:
    return tuple(pd.Timestamp(value).isoformat() for value in values)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    return value


def _text_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{name} must be a text array")
    return tuple(value)


__all__ = [
    "PurgedWalkForwardSplitter",
    "SplitFoldReceipt",
    "ValidationReceipt",
    "ValidationReceiptVerifier",
]
