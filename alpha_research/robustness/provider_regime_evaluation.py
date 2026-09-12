"""Historical v1 evaluation across one global provider cutover boundary.

The supplied L2 archive changes its XSHG order semantics on 2021-06-07.
This module treats the dates before, on, and after a configured cutover as
three different scientific domains.  It never fills or standardizes across
those domains and never forms a turnover transition across their boundaries.

The evaluator consumes only already-materialized, daily research observations.
It performs no raw-data I/O and grants no admission or production authority.

This module is retained for historical receipt compatibility.  Its global
``pre/boundary/post`` partition is not valid R1 launch evidence because the
known 2021-06-07 semantic change applies only to XSHG ORDER.  New R1 work must
use :mod:`alpha_research.robustness.provider_semantic_strata_v2`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import cast

import numpy as np
import pandas as pd

from alpha_research.core.hashing import (
    canonical_json_bytes,
    hash_frame,
    hash_json,
    require_sha256,
)


_SPEC_SCHEMA = "provider-regime-evaluation-spec/v1"
_RECEIPT_SCHEMA = "provider-regime-evaluation-receipt/v1"
_METRICS_SCHEMA = "provider-regime-segment-metrics/v1"
_OBSERVATION_COLUMNS = (
    "trading_date",
    "security_id",
    "provider_regime",
    "signal",
    "label",
    "eligible",
)


class ProviderRegimeEvaluationError(ValueError):
    """Stable fail-closed error for provider-regime evaluation."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}:{detail}")


def _error(code: str, detail: str) -> ProviderRegimeEvaluationError:
    return ProviderRegimeEvaluationError(code, detail)


class ProviderRegime(str, Enum):
    PRE_CUTOVER = "pre_cutover"
    CUTOVER_BOUNDARY = "cutover_boundary"
    POST_CUTOVER = "post_cutover"


_REQUIRED_REGIMES = tuple(ProviderRegime)


class BoundaryDayPolicy(str, Enum):
    ISOLATE = "isolate_cutover_date_as_boundary_segment"


class StandardizationPolicy(str, Enum):
    NONE = "none"
    WITHIN_REGIME = "within_provider_regime_and_trading_date_only"


class FillPolicy(str, Enum):
    NONE = "preserve_nulls_no_fill"
    WITHIN_REGIME_PAST_ONLY = "within_provider_regime_past_only"


@dataclass(frozen=True, slots=True)
class ProviderRegimeEvaluationSpec:
    """Historical v1 policy; deliberately non-R1-launchable."""

    evaluation_id: str
    version: str
    cutover_date: str
    minimum_cross_sectional_observations: int
    boundary_day_policy: BoundaryDayPolicy | str = BoundaryDayPolicy.ISOLATE
    standardization_policy: StandardizationPolicy | str = (
        StandardizationPolicy.WITHIN_REGIME
    )
    fill_policy: FillPolicy | str = FillPolicy.NONE
    required_regimes: tuple[ProviderRegime | str, ...] = _REQUIRED_REGIMES
    turnover_pair_policy: str = "within_provider_regime_only"
    research_only: bool = True
    admission_claim: bool = False
    schema_version: str = _SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _SPEC_SCHEMA:
            raise ValueError("unsupported provider-regime evaluation spec schema")
        if (
            not isinstance(self.evaluation_id, str)
            or not self.evaluation_id
            or self.evaluation_id != self.evaluation_id.strip()
        ):
            raise ValueError("evaluation_id must be a canonical non-empty string")
        if (
            not isinstance(self.version, str)
            or not self.version
            or self.version != self.version.strip()
        ):
            raise ValueError("version must be a canonical non-empty string")
        object.__setattr__(
            self,
            "cutover_date",
            _canonical_date(self.cutover_date, name="cutover_date"),
        )
        if (
            not isinstance(self.minimum_cross_sectional_observations, int)
            or isinstance(self.minimum_cross_sectional_observations, bool)
            or self.minimum_cross_sectional_observations < 2
        ):
            raise ValueError(
                "minimum_cross_sectional_observations must be an integer >= 2"
            )
        object.__setattr__(
            self,
            "boundary_day_policy",
            BoundaryDayPolicy(self.boundary_day_policy),
        )
        object.__setattr__(
            self,
            "standardization_policy",
            StandardizationPolicy(self.standardization_policy),
        )
        object.__setattr__(self, "fill_policy", FillPolicy(self.fill_policy))
        regimes = tuple(ProviderRegime(value) for value in self.required_regimes)
        if regimes != _REQUIRED_REGIMES:
            raise ValueError(
                "provider-regime evaluation requires pre/boundary/post in order"
            )
        object.__setattr__(self, "required_regimes", regimes)
        if self.turnover_pair_policy != "within_provider_regime_only":
            raise ValueError("turnover pairs must not cross provider regimes")
        if self.research_only is not True:
            raise ValueError("provider-regime evaluation must remain research-only")
        if self.admission_claim is not False:
            raise ValueError("provider-regime evaluation cannot claim admission")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    @property
    def r1_launchable(self) -> bool:
        """V1's global date split is never valid R1 launch evidence."""

        return False

    def to_dict(self) -> dict[str, object]:
        boundary = cast(BoundaryDayPolicy, self.boundary_day_policy)
        standardization = cast(StandardizationPolicy, self.standardization_policy)
        fill = cast(FillPolicy, self.fill_policy)
        regimes = cast(tuple[ProviderRegime, ...], self.required_regimes)
        return {
            "schema_version": self.schema_version,
            "evaluation_id": self.evaluation_id,
            "version": self.version,
            "cutover_date": self.cutover_date,
            "minimum_cross_sectional_observations": (
                self.minimum_cross_sectional_observations
            ),
            "boundary_day_policy": boundary.value,
            "standardization_policy": standardization.value,
            "fill_policy": fill.value,
            "required_regimes": [regime.value for regime in regimes],
            "turnover_pair_policy": self.turnover_pair_policy,
            "research_only": self.research_only,
            "admission_claim": self.admission_claim,
        }

    def to_wire_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self.to_dict()))

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "ProviderRegimeEvaluationSpec":
        expected = {
            "schema_version",
            "evaluation_id",
            "version",
            "cutover_date",
            "minimum_cross_sectional_observations",
            "boundary_day_policy",
            "standardization_policy",
            "fill_policy",
            "required_regimes",
            "turnover_pair_policy",
            "research_only",
            "admission_claim",
        }
        if set(value) != expected:
            raise ValueError("ProviderRegimeEvaluationSpec wire fields differ")
        raw_regimes = value["required_regimes"]
        if not isinstance(raw_regimes, list):
            raise TypeError("required_regimes must be a list")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            evaluation_id=_text(value["evaluation_id"], "evaluation_id"),
            version=_text(value["version"], "version"),
            cutover_date=_text(value["cutover_date"], "cutover_date"),
            minimum_cross_sectional_observations=_integer(
                value["minimum_cross_sectional_observations"],
                "minimum_cross_sectional_observations",
            ),
            boundary_day_policy=_text(
                value["boundary_day_policy"], "boundary_day_policy"
            ),
            standardization_policy=_text(
                value["standardization_policy"], "standardization_policy"
            ),
            fill_policy=_text(value["fill_policy"], "fill_policy"),
            required_regimes=tuple(
                _text(item, "required_regimes item") for item in raw_regimes
            ),
            turnover_pair_policy=_text(
                value["turnover_pair_policy"], "turnover_pair_policy"
            ),
            research_only=_boolean(value["research_only"], "research_only"),
            admission_claim=_boolean(value["admission_claim"], "admission_claim"),
        )

    @classmethod
    def from_wire_bytes(cls, payload: bytes) -> "ProviderRegimeEvaluationSpec":
        value = _canonical_wire_mapping(payload, name="provider-regime spec")
        result = cls.from_mapping(value)
        if result.to_wire_bytes() != payload:
            raise ValueError("provider-regime spec wire is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class ProviderRegimeSegmentMetrics:
    """Auditable metrics for exactly one segment or the combined view."""

    scope_id: str
    start_date: str
    end_date: str
    date_count: int
    intended_observations: int
    valid_observations: int
    coverage: float | None
    valid_rank_ic_dates: int
    rank_ic_mean: float | None
    turnover_transition_count: int
    rank_turnover_mean: float | None
    input_hash: str
    daily_rank_ic_evidence_hash: str
    turnover_evidence_hash: str
    schema_version: str = _METRICS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _METRICS_SCHEMA:
            raise ValueError("unsupported provider-regime metrics schema")
        allowed_scopes = {regime.value for regime in _REQUIRED_REGIMES} | {
            "all_segments"
        }
        if self.scope_id not in allowed_scopes:
            raise ValueError("provider-regime metrics scope is unsupported")
        start = _canonical_date(self.start_date, name="metrics start_date")
        end = _canonical_date(self.end_date, name="metrics end_date")
        if start > end:
            raise ValueError("provider-regime metrics date range is inverted")
        object.__setattr__(self, "start_date", start)
        object.__setattr__(self, "end_date", end)
        for name in (
            "date_count",
            "intended_observations",
            "valid_observations",
            "valid_rank_ic_dates",
            "turnover_transition_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.date_count < 1:
            raise ValueError("provider-regime metrics require at least one date")
        if self.valid_observations > self.intended_observations:
            raise ValueError("valid observations exceed intended observations")
        if self.valid_rank_ic_dates > self.date_count:
            raise ValueError("valid RankIC dates exceed date count")
        if self.turnover_transition_count > max(self.date_count - 1, 0):
            raise ValueError("turnover transitions exceed within-scope date pairs")
        _validate_ratio(self.coverage, name="coverage")
        _validate_range(self.rank_ic_mean, -1.0, 1.0, name="rank_ic_mean")
        _validate_range(
            self.rank_turnover_mean,
            0.0,
            1.0,
            name="rank_turnover_mean",
        )
        for name in ("coverage", "rank_ic_mean", "rank_turnover_mean"):
            value = cast(float | None, getattr(self, name))
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    0.0 if float(value) == 0.0 else float(value),
                )
        expected_coverage = (
            None
            if self.intended_observations == 0
            else self.valid_observations / self.intended_observations
        )
        if not _optional_close(self.coverage, expected_coverage):
            raise ValueError("provider-regime coverage differs from counts")
        if (self.valid_rank_ic_dates == 0) is not (self.rank_ic_mean is None):
            raise ValueError("RankIC mean presence differs from valid date count")
        if (self.turnover_transition_count == 0) is not (
            self.rank_turnover_mean is None
        ):
            raise ValueError("turnover mean presence differs from transition count")
        for name in (
            "input_hash",
            "daily_rank_ic_evidence_hash",
            "turnover_evidence_hash",
        ):
            require_sha256(getattr(self, name), name=f"provider-regime {name}")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope_id": self.scope_id,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "date_count": self.date_count,
            "intended_observations": self.intended_observations,
            "valid_observations": self.valid_observations,
            "coverage": self.coverage,
            "valid_rank_ic_dates": self.valid_rank_ic_dates,
            "rank_ic_mean": self.rank_ic_mean,
            "turnover_transition_count": self.turnover_transition_count,
            "rank_turnover_mean": self.rank_turnover_mean,
            "input_hash": self.input_hash,
            "daily_rank_ic_evidence_hash": self.daily_rank_ic_evidence_hash,
            "turnover_evidence_hash": self.turnover_evidence_hash,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "ProviderRegimeSegmentMetrics":
        expected = {
            "schema_version",
            "scope_id",
            "start_date",
            "end_date",
            "date_count",
            "intended_observations",
            "valid_observations",
            "coverage",
            "valid_rank_ic_dates",
            "rank_ic_mean",
            "turnover_transition_count",
            "rank_turnover_mean",
            "input_hash",
            "daily_rank_ic_evidence_hash",
            "turnover_evidence_hash",
        }
        if set(value) != expected:
            raise ValueError("ProviderRegimeSegmentMetrics wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            scope_id=_text(value["scope_id"], "scope_id"),
            start_date=_text(value["start_date"], "start_date"),
            end_date=_text(value["end_date"], "end_date"),
            date_count=_integer(value["date_count"], "date_count"),
            intended_observations=_integer(
                value["intended_observations"], "intended_observations"
            ),
            valid_observations=_integer(
                value["valid_observations"], "valid_observations"
            ),
            coverage=_optional_number(value["coverage"], "coverage"),
            valid_rank_ic_dates=_integer(
                value["valid_rank_ic_dates"], "valid_rank_ic_dates"
            ),
            rank_ic_mean=_optional_number(value["rank_ic_mean"], "rank_ic_mean"),
            turnover_transition_count=_integer(
                value["turnover_transition_count"], "turnover_transition_count"
            ),
            rank_turnover_mean=_optional_number(
                value["rank_turnover_mean"], "rank_turnover_mean"
            ),
            input_hash=_text(value["input_hash"], "input_hash"),
            daily_rank_ic_evidence_hash=_text(
                value["daily_rank_ic_evidence_hash"],
                "daily_rank_ic_evidence_hash",
            ),
            turnover_evidence_hash=_text(
                value["turnover_evidence_hash"], "turnover_evidence_hash"
            ),
        )


@dataclass(frozen=True, slots=True)
class ProviderRegimeEvaluationReceipt:
    """Historical research-only result; never valid R1 launch evidence."""

    evaluation_spec_hash: str
    input_hash: str
    segment_metrics: tuple[ProviderRegimeSegmentMetrics, ...]
    combined_metrics: ProviderRegimeSegmentMetrics
    boundary_day_policy: str
    standardization_policy: str
    fill_policy: str
    cross_regime_standardization_allowed: bool = False
    cross_regime_ffill_allowed: bool = False
    cross_regime_turnover_pairs_allowed: bool = False
    research_only: bool = True
    admission_claim: bool = False
    schema_version: str = _RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _RECEIPT_SCHEMA:
            raise ValueError("unsupported provider-regime receipt schema")
        require_sha256(
            self.evaluation_spec_hash,
            name="provider-regime receipt evaluation_spec_hash",
        )
        require_sha256(self.input_hash, name="provider-regime receipt input_hash")
        expected_scopes = tuple(regime.value for regime in _REQUIRED_REGIMES)
        if tuple(item.scope_id for item in self.segment_metrics) != expected_scopes:
            raise ValueError("provider-regime receipt segments are incomplete or unordered")
        if self.combined_metrics.scope_id != "all_segments":
            raise ValueError("provider-regime combined metrics scope differs")
        if self.boundary_day_policy != BoundaryDayPolicy.ISOLATE.value:
            raise ValueError("provider-regime receipt boundary policy differs")
        StandardizationPolicy(self.standardization_policy)
        FillPolicy(self.fill_policy)
        if self.cross_regime_standardization_allowed is not False:
            raise ValueError("cross-regime standardization cannot be authorized")
        if self.cross_regime_ffill_allowed is not False:
            raise ValueError("cross-regime ffill cannot be authorized")
        if self.cross_regime_turnover_pairs_allowed is not False:
            raise ValueError("cross-regime turnover pairs cannot be authorized")
        if self.research_only is not True:
            raise ValueError("provider-regime receipt must remain research-only")
        if self.admission_claim is not False:
            raise ValueError("provider-regime receipt cannot claim admission")
        if self.input_hash != self.combined_metrics.input_hash:
            raise ValueError(
                "provider-regime receipt input hash differs from combined metrics"
            )
        self._validate_combined_counts()

    def _validate_combined_counts(self) -> None:
        combined = self.combined_metrics
        segments = self.segment_metrics
        for name in (
            "date_count",
            "intended_observations",
            "valid_observations",
            "valid_rank_ic_dates",
            "turnover_transition_count",
        ):
            if getattr(combined, name) != sum(getattr(item, name) for item in segments):
                raise ValueError(f"combined {name} differs from segment totals")
        rank_values = [
            (item.rank_ic_mean, item.valid_rank_ic_dates) for item in segments
        ]
        turnover_values = [
            (item.rank_turnover_mean, item.turnover_transition_count)
            for item in segments
        ]
        if not _optional_close(
            combined.rank_ic_mean,
            _weighted_optional_mean(rank_values),
        ):
            raise ValueError("combined RankIC differs from segment evidence")
        if not _optional_close(
            combined.rank_turnover_mean,
            _weighted_optional_mean(turnover_values),
        ):
            raise ValueError("combined turnover differs from segment evidence")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    @property
    def r1_launchable(self) -> bool:
        """V1 receipts cannot authorize R1 launch or admission."""

        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evaluation_spec_hash": self.evaluation_spec_hash,
            "input_hash": self.input_hash,
            "segment_metrics": [item.to_dict() for item in self.segment_metrics],
            "combined_metrics": self.combined_metrics.to_dict(),
            "boundary_day_policy": self.boundary_day_policy,
            "standardization_policy": self.standardization_policy,
            "fill_policy": self.fill_policy,
            "cross_regime_standardization_allowed": (
                self.cross_regime_standardization_allowed
            ),
            "cross_regime_ffill_allowed": self.cross_regime_ffill_allowed,
            "cross_regime_turnover_pairs_allowed": (
                self.cross_regime_turnover_pairs_allowed
            ),
            "research_only": self.research_only,
            "admission_claim": self.admission_claim,
        }

    def to_wire_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self.to_dict()))

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "ProviderRegimeEvaluationReceipt":
        expected = {
            "schema_version",
            "evaluation_spec_hash",
            "input_hash",
            "segment_metrics",
            "combined_metrics",
            "boundary_day_policy",
            "standardization_policy",
            "fill_policy",
            "cross_regime_standardization_allowed",
            "cross_regime_ffill_allowed",
            "cross_regime_turnover_pairs_allowed",
            "research_only",
            "admission_claim",
        }
        if set(value) != expected:
            raise ValueError("ProviderRegimeEvaluationReceipt wire fields differ")
        raw_segments = value["segment_metrics"]
        if not isinstance(raw_segments, list):
            raise TypeError("segment_metrics must be a list")
        combined = value["combined_metrics"]
        if not isinstance(combined, Mapping):
            raise TypeError("combined_metrics must be an object")
        segments: list[ProviderRegimeSegmentMetrics] = []
        for raw in raw_segments:
            if not isinstance(raw, Mapping):
                raise TypeError("segment_metrics entries must be objects")
            segments.append(ProviderRegimeSegmentMetrics.from_mapping(raw))
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            evaluation_spec_hash=_text(
                value["evaluation_spec_hash"], "evaluation_spec_hash"
            ),
            input_hash=_text(value["input_hash"], "input_hash"),
            segment_metrics=tuple(segments),
            combined_metrics=ProviderRegimeSegmentMetrics.from_mapping(combined),
            boundary_day_policy=_text(
                value["boundary_day_policy"], "boundary_day_policy"
            ),
            standardization_policy=_text(
                value["standardization_policy"], "standardization_policy"
            ),
            fill_policy=_text(value["fill_policy"], "fill_policy"),
            cross_regime_standardization_allowed=_boolean(
                value["cross_regime_standardization_allowed"],
                "cross_regime_standardization_allowed",
            ),
            cross_regime_ffill_allowed=_boolean(
                value["cross_regime_ffill_allowed"],
                "cross_regime_ffill_allowed",
            ),
            cross_regime_turnover_pairs_allowed=_boolean(
                value["cross_regime_turnover_pairs_allowed"],
                "cross_regime_turnover_pairs_allowed",
            ),
            research_only=_boolean(value["research_only"], "research_only"),
            admission_claim=_boolean(value["admission_claim"], "admission_claim"),
        )

    @classmethod
    def from_wire_bytes(cls, payload: bytes) -> "ProviderRegimeEvaluationReceipt":
        value = _canonical_wire_mapping(payload, name="provider-regime receipt")
        result = cls.from_mapping(value)
        if result.to_wire_bytes() != payload:
            raise ValueError("provider-regime receipt wire is not canonical")
        return result


def evaluate_provider_regimes(
    spec: ProviderRegimeEvaluationSpec,
    observations: pd.DataFrame,
) -> ProviderRegimeEvaluationReceipt:
    """Evaluate mandatory provider segments without cross-segment transforms."""

    if type(spec) is not ProviderRegimeEvaluationSpec:
        raise TypeError("spec must be an exact ProviderRegimeEvaluationSpec")
    source_frame = _canonical_observations(spec, observations)
    frame = _preprocess_signals(spec, source_frame)
    segment_metrics: list[ProviderRegimeSegmentMetrics] = []
    daily_evidence: list[dict[str, object]] = []
    turnover_evidence: list[dict[str, object]] = []
    for regime in _REQUIRED_REGIMES:
        segment = frame.loc[frame["provider_regime"].eq(regime.value)].copy()
        source_segment = source_frame.loc[
            source_frame["provider_regime"].eq(regime.value)
        ].copy()
        metrics, daily_rows, turnover_rows = _evaluate_scope(
            segment,
            source_frame=source_segment,
            scope_id=regime.value,
            minimum=spec.minimum_cross_sectional_observations,
        )
        segment_metrics.append(metrics)
        daily_evidence.extend(daily_rows)
        turnover_evidence.extend(turnover_rows)
    combined = _combined_metrics(
        frame,
        source_frame,
        tuple(segment_metrics),
        daily_evidence=daily_evidence,
        turnover_evidence=turnover_evidence,
    )
    standardization = cast(StandardizationPolicy, spec.standardization_policy)
    fill = cast(FillPolicy, spec.fill_policy)
    boundary = cast(BoundaryDayPolicy, spec.boundary_day_policy)
    return ProviderRegimeEvaluationReceipt(
        evaluation_spec_hash=spec.content_hash,
        input_hash=cast(str, hash_frame(source_frame)),
        segment_metrics=tuple(segment_metrics),
        combined_metrics=combined,
        boundary_day_policy=boundary.value,
        standardization_policy=standardization.value,
        fill_policy=fill.value,
    )


def verify_provider_regime_evaluation_receipt(
    spec: ProviderRegimeEvaluationSpec,
    observations: pd.DataFrame,
    receipt: ProviderRegimeEvaluationReceipt,
) -> ProviderRegimeEvaluationReceipt:
    """Recompute and verify a receipt against its exact spec and source input."""

    if type(receipt) is not ProviderRegimeEvaluationReceipt:
        raise TypeError("receipt must be an exact ProviderRegimeEvaluationReceipt")
    expected = evaluate_provider_regimes(spec, observations)
    if expected.to_wire_bytes() != receipt.to_wire_bytes():
        raise _error(
            "receipt_recomputation_mismatch",
            f"expected={expected.content_hash}:actual={receipt.content_hash}",
        )
    return receipt


def _canonical_observations(
    spec: ProviderRegimeEvaluationSpec,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.DataFrame(observations).copy().reset_index(drop=True)
    if tuple(frame.columns) != _OBSERVATION_COLUMNS:
        raise _error(
            "observation_columns_differ",
            f"expected={','.join(_OBSERVATION_COLUMNS)}",
        )
    if frame.empty:
        raise _error("observations_empty", "provider-regime observations are empty")
    trading_dates = [
        _canonical_date(value, name="trading_date")
        for value in frame["trading_date"].tolist()
    ]
    frame["trading_date"] = pd.Series(trading_dates, dtype="string")
    security_ids = frame["security_id"].tolist()
    if any(
        not isinstance(value, str)
        or not value
        or value != value.strip()
        for value in security_ids
    ):
        raise _error("security_id_invalid", "security_id must be canonical strings")
    frame["security_id"] = pd.Series(security_ids, dtype="string")
    if frame.duplicated(["trading_date", "security_id"]).any():
        raise _error(
            "duplicate_observation",
            "trading_date/security_id observations must be unique",
        )
    for column in ("signal", "label"):
        if pd.api.types.is_bool_dtype(frame[column].dtype):
            raise _error("numeric_column_invalid", f"{column} cannot be boolean")
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise _error("numeric_column_invalid", column) from exc
        values = frame[column].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise _error("numeric_column_nonfinite", f"{column} contains infinity")
    eligible = frame["eligible"].tolist()
    if any(not isinstance(value, (bool, np.bool_)) for value in eligible):
        raise _error("eligibility_invalid", "eligible must be non-null boolean")
    frame["eligible"] = frame["eligible"].astype(bool)
    expected_regimes = [
        _classify_regime(value, cutover_date=spec.cutover_date).value
        for value in trading_dates
    ]
    declared_regimes = frame["provider_regime"].tolist()
    for row_number, (declared, expected) in enumerate(
        zip(declared_regimes, expected_regimes, strict=True)
    ):
        if declared != expected:
            raise _error(
                "provider_regime_misclassified",
                f"row={row_number}:declared={declared}:expected={expected}",
            )
    frame["provider_regime"] = pd.Series(expected_regimes, dtype="string")
    present = set(expected_regimes)
    missing = [
        regime.value for regime in _REQUIRED_REGIMES if regime.value not in present
    ]
    if missing:
        raise _error("required_segment_missing", ",".join(missing))
    return frame.loc[:, list(_OBSERVATION_COLUMNS)].sort_values(
        ["trading_date", "security_id"], kind="stable", ignore_index=True
    )


def _preprocess_signals(
    spec: ProviderRegimeEvaluationSpec,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the only permitted transforms inside explicit regime groups."""

    result = frame.copy()
    fill_policy = cast(FillPolicy, spec.fill_policy)
    if fill_policy is FillPolicy.WITHIN_REGIME_PAST_ONLY:
        result["signal"] = result.groupby(
            ["provider_regime", "security_id"],
            sort=False,
            observed=True,
        )["signal"].ffill()
    standardization = cast(StandardizationPolicy, spec.standardization_policy)
    if standardization is StandardizationPolicy.WITHIN_REGIME:
        grouped = result.groupby(
            ["provider_regime", "trading_date"],
            sort=False,
            observed=True,
        )["signal"]
        location = grouped.transform("mean")
        scale = grouped.transform(lambda values: values.std(ddof=0))
        result["signal"] = (result["signal"] - location) / scale.where(scale > 0.0)
    return result


def _evaluate_scope(
    frame: pd.DataFrame,
    *,
    source_frame: pd.DataFrame,
    scope_id: str,
    minimum: int,
) -> tuple[
    ProviderRegimeSegmentMetrics,
    list[dict[str, object]],
    list[dict[str, object]],
]:
    dates = sorted(cast(list[str], frame["trading_date"].unique().tolist()))
    intended = int(frame["eligible"].sum())
    valid = (
        frame["eligible"]
        & frame["signal"].notna()
        & frame["label"].notna()
    )
    valid_observations = int(valid.sum())
    daily_rows = _daily_rank_ic(frame, minimum=minimum, scope_id=scope_id)
    valid_ics = [
        cast(float, row["rank_ic"])
        for row in daily_rows
        if row["rank_ic"] is not None
    ]
    turnover_rows = _within_scope_turnover(frame, scope_id=scope_id)
    turnovers = [cast(float, row["rank_turnover"]) for row in turnover_rows]
    metrics = ProviderRegimeSegmentMetrics(
        scope_id=scope_id,
        start_date=dates[0],
        end_date=dates[-1],
        date_count=len(dates),
        intended_observations=intended,
        valid_observations=valid_observations,
        coverage=(
            None if intended == 0 else float(valid_observations / intended)
        ),
        valid_rank_ic_dates=len(valid_ics),
        rank_ic_mean=(None if not valid_ics else float(np.mean(valid_ics))),
        turnover_transition_count=len(turnovers),
        rank_turnover_mean=(
            None if not turnovers else float(np.mean(turnovers))
        ),
        input_hash=cast(str, hash_frame(source_frame.reset_index(drop=True))),
        daily_rank_ic_evidence_hash=cast(str, hash_json(daily_rows)),
        turnover_evidence_hash=cast(str, hash_json(turnover_rows)),
    )
    return metrics, daily_rows, turnover_rows


def _daily_rank_ic(
    frame: pd.DataFrame,
    *,
    minimum: int,
    scope_id: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trading_date, raw_day in frame.groupby("trading_date", sort=True):
        day = raw_day.loc[
            raw_day["eligible"]
            & raw_day["signal"].notna()
            & raw_day["label"].notna()
        ]
        rank_ic: float | None = None
        if len(day) >= minimum:
            correlation = day["signal"].corr(day["label"], method="spearman")
            if correlation is not None and math.isfinite(float(correlation)):
                rank_ic = float(correlation)
        rows.append(
            {
                "scope_id": scope_id,
                "trading_date": cast(str, trading_date),
                "observation_count": int(len(day)),
                "rank_ic": rank_ic,
            }
        )
    return rows


def _within_scope_turnover(
    frame: pd.DataFrame,
    *,
    scope_id: str,
) -> list[dict[str, object]]:
    ranks: dict[str, pd.Series] = {}
    for trading_date, raw_day in frame.groupby("trading_date", sort=True):
        day = raw_day.loc[
            raw_day["eligible"] & raw_day["signal"].notna(),
            ["security_id", "signal"],
        ].set_index("security_id")
        ranks[cast(str, trading_date)] = day["signal"].rank(
            method="average", pct=True
        )
    rows: list[dict[str, object]] = []
    dates = sorted(ranks)
    for previous_date, current_date in zip(dates, dates[1:], strict=False):
        previous = ranks[previous_date]
        current = ranks[current_date]
        common = previous.index.intersection(current.index, sort=True)
        if len(common) < 2:
            continue
        value = float((current.loc[common] - previous.loc[common]).abs().mean())
        rows.append(
            {
                "scope_id": scope_id,
                "previous_date": previous_date,
                "current_date": current_date,
                "common_security_count": int(len(common)),
                "rank_turnover": value,
            }
        )
    return rows


def _combined_metrics(
    frame: pd.DataFrame,
    source_frame: pd.DataFrame,
    segments: tuple[ProviderRegimeSegmentMetrics, ...],
    *,
    daily_evidence: list[dict[str, object]],
    turnover_evidence: list[dict[str, object]],
) -> ProviderRegimeSegmentMetrics:
    dates = sorted(cast(list[str], frame["trading_date"].unique().tolist()))
    intended = sum(item.intended_observations for item in segments)
    valid = sum(item.valid_observations for item in segments)
    rank_count = sum(item.valid_rank_ic_dates for item in segments)
    turnover_count = sum(item.turnover_transition_count for item in segments)
    return ProviderRegimeSegmentMetrics(
        scope_id="all_segments",
        start_date=dates[0],
        end_date=dates[-1],
        date_count=sum(item.date_count for item in segments),
        intended_observations=intended,
        valid_observations=valid,
        coverage=None if intended == 0 else float(valid / intended),
        valid_rank_ic_dates=rank_count,
        rank_ic_mean=_weighted_optional_mean(
            [(item.rank_ic_mean, item.valid_rank_ic_dates) for item in segments]
        ),
        turnover_transition_count=turnover_count,
        rank_turnover_mean=_weighted_optional_mean(
            [
                (item.rank_turnover_mean, item.turnover_transition_count)
                for item in segments
            ]
        ),
        input_hash=cast(str, hash_frame(source_frame)),
        daily_rank_ic_evidence_hash=cast(str, hash_json(daily_evidence)),
        turnover_evidence_hash=cast(str, hash_json(turnover_evidence)),
    )


def _classify_regime(
    trading_date: str,
    *,
    cutover_date: str,
) -> ProviderRegime:
    if trading_date < cutover_date:
        return ProviderRegime.PRE_CUTOVER
    if trading_date == cutover_date:
        return ProviderRegime.CUTOVER_BOUNDARY
    return ProviderRegime.POST_CUTOVER


def _canonical_date(value: object, *, name: str) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise ValueError(f"{name} must be an ISO date string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date string") from exc
    canonical = parsed.isoformat()
    if canonical != value:
        raise ValueError(f"{name} must be a canonical ISO date")
    return canonical


def _weighted_optional_mean(
    values: list[tuple[float | None, int]],
) -> float | None:
    denominator = sum(weight for value, weight in values if value is not None)
    if denominator == 0:
        return None
    weighted_total = 0.0
    for value, weight in values:
        if value is not None:
            weighted_total += value * weight
    return weighted_total / denominator


def _validate_ratio(value: float | None, *, name: str) -> None:
    _validate_range(value, 0.0, 1.0, name=name)


def _validate_range(
    value: float | None,
    lower: float,
    upper: float,
    *,
    name: str,
) -> None:
    if value is None:
        return
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not lower <= float(value) <= upper
    ):
        raise ValueError(f"{name} must lie in [{lower},{upper}]")


def _optional_close(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _canonical_wire_mapping(payload: bytes, *, name: str) -> Mapping[str, object]:
    if not isinstance(payload, bytes):
        raise TypeError(f"{name} payload must be bytes")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} wire is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} wire root must be an object")
    if canonical_json_bytes(value) != payload:
        raise ValueError(f"{name} wire is not canonical")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _optional_number(value: object, name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric or null")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


__all__ = [
    "BoundaryDayPolicy",
    "FillPolicy",
    "ProviderRegime",
    "ProviderRegimeEvaluationError",
    "ProviderRegimeEvaluationReceipt",
    "ProviderRegimeEvaluationSpec",
    "ProviderRegimeSegmentMetrics",
    "StandardizationPolicy",
    "evaluate_provider_regimes",
    "verify_provider_regime_evaluation_receipt",
]
