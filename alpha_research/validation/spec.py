from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Mapping, cast

import pandas as pd

from alpha_research.core.data import DataRole
from alpha_research.core.hashing import hash_json


def _aware(value: str, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp


@dataclass(frozen=True, slots=True)
class ValidationFoldSpec:
    fold_id: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str

    def __post_init__(self) -> None:
        if not self.fold_id.strip():
            raise ValueError("validation fold_id must not be empty")
        train_start = _aware(self.train_start, name="train_start")
        train_end = _aware(self.train_end, name="train_end")
        validation_start = _aware(self.validation_start, name="validation_start")
        validation_end = _aware(self.validation_end, name="validation_end")
        if train_start > train_end:
            raise ValueError("validation train range is inverted")
        if validation_start > validation_end:
            raise ValueError("validation range is inverted")
        if train_end >= validation_start:
            raise ValueError("walk-forward train must end before validation starts")

    def to_dict(self) -> dict[str, str]:
        return {
            "fold_id": self.fold_id,
            "train_start": pd.Timestamp(self.train_start).isoformat(),
            "train_end": pd.Timestamp(self.train_end).isoformat(),
            "validation_start": pd.Timestamp(self.validation_start).isoformat(),
            "validation_end": pd.Timestamp(self.validation_end).isoformat(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ValidationFoldSpec":
        expected = {
            "fold_id",
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
        }
        if set(value) != expected:
            raise ValueError("ValidationFoldSpec wire fields differ")
        return cls(
            fold_id=_text(value["fold_id"], "fold_id"),
            train_start=_text(value["train_start"], "train_start"),
            train_end=_text(value["train_end"], "train_end"),
            validation_start=_text(value["validation_start"], "validation_start"),
            validation_end=_text(value["validation_end"], "validation_end"),
        )


@dataclass(frozen=True, slots=True)
class ValidationSpec:
    validation_id: str
    version: str
    folds: tuple[ValidationFoldSpec, ...]
    embargo_sessions: int = 0
    purge_overlapping_labels: bool = True
    min_train_signals: int = 1
    min_validation_signals: int = 1
    train_role: DataRole | str = DataRole.TRAIN
    validation_role: DataRole | str = DataRole.VALIDATION
    method: str = "purged_walk_forward"
    schema_version: str = "validation-spec/v1"

    def __post_init__(self) -> None:
        folds = tuple(self.folds)
        if not all(isinstance(fold, ValidationFoldSpec) for fold in folds):
            raise TypeError("validation folds have invalid types")
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "train_role", DataRole(self.train_role))
        object.__setattr__(self, "validation_role", DataRole(self.validation_role))
        if self.schema_version != "validation-spec/v1":
            raise ValueError("unsupported ValidationSpec schema")
        if self.method != "purged_walk_forward":
            raise ValueError("only purged_walk_forward validation is permitted")
        if not self.validation_id.strip() or not self.version.strip():
            raise ValueError("validation id and version are required")
        if not self.folds:
            raise ValueError("validation requires at least one fold")
        if len({fold.fold_id for fold in self.folds}) != len(self.folds):
            raise ValueError("validation fold ids must be unique")
        if not self.purge_overlapping_labels:
            raise ValueError("financial validation may not disable label purging")
        if (
            not isinstance(self.embargo_sessions, int)
            or isinstance(self.embargo_sessions, bool)
            or self.embargo_sessions < 0
        ):
            raise ValueError("embargo_sessions must be a non-negative integer")
        for name in ("min_train_signals", "min_validation_signals"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.train_role is not DataRole.TRAIN:
            raise ValueError("validation train_role must be train")
        if self.validation_role is not DataRole.VALIDATION:
            raise ValueError("validation validation_role must be validation")
        ordered = sorted(
            self.folds,
            key=lambda fold: pd.Timestamp(fold.validation_start).value,
        )
        if list(self.folds) != ordered:
            raise ValueError("validation folds must be chronologically ordered")
        for previous, current in pairwise(ordered):
            if pd.Timestamp(current.validation_start) <= pd.Timestamp(
                previous.validation_end
            ):
                raise ValueError(
                    "validation fold ranges must be disjoint and non-overlapping"
                )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        train_role = self.train_role
        validation_role = self.validation_role
        if not isinstance(train_role, DataRole) or not isinstance(
            validation_role, DataRole
        ):  # pragma: no cover
            raise RuntimeError("validation roles were not normalized")
        return {
            "schema_version": self.schema_version,
            "validation_id": self.validation_id,
            "version": self.version,
            "method": self.method,
            "folds": [fold.to_dict() for fold in self.folds],
            "embargo_sessions": self.embargo_sessions,
            "purge_overlapping_labels": self.purge_overlapping_labels,
            "min_train_signals": self.min_train_signals,
            "min_validation_signals": self.min_validation_signals,
            "train_role": train_role.value,
            "validation_role": validation_role.value,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ValidationSpec":
        expected = {
            "schema_version",
            "validation_id",
            "version",
            "method",
            "folds",
            "embargo_sessions",
            "purge_overlapping_labels",
            "min_train_signals",
            "min_validation_signals",
            "train_role",
            "validation_role",
        }
        if set(value) != expected:
            raise ValueError("ValidationSpec wire fields differ")
        raw_folds = value["folds"]
        if not isinstance(raw_folds, list) or not all(
            isinstance(item, Mapping) for item in raw_folds
        ):
            raise TypeError("ValidationSpec folds must be an object array")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            validation_id=_text(value["validation_id"], "validation_id"),
            version=_text(value["version"], "version"),
            method=_text(value["method"], "method"),
            folds=tuple(ValidationFoldSpec.from_mapping(item) for item in raw_folds),
            embargo_sessions=_integer(value["embargo_sessions"], "embargo_sessions"),
            purge_overlapping_labels=_boolean(
                value["purge_overlapping_labels"], "purge_overlapping_labels"
            ),
            min_train_signals=_integer(value["min_train_signals"], "min_train_signals"),
            min_validation_signals=_integer(
                value["min_validation_signals"], "min_validation_signals"
            ),
            train_role=_text(value["train_role"], "train_role"),
            validation_role=_text(value["validation_role"], "validation_role"),
        )


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


__all__ = ["ValidationFoldSpec", "ValidationSpec"]
