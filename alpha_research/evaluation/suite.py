from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from alpha_research.core.hashing import hash_frame, hash_json
from alpha_research.evaluation.spec import EvaluationSpec
from alpha_research.factors import FactorResult, FactorSpec
from alpha_research.labels import LabelResult, LabelSpec
from alpha_research.validation import ValidationReceipt


MetricValue = float | int | None


@dataclass(frozen=True, slots=True)
class FactorEvaluationReport:
    evaluation_spec_hash: str
    factor_spec_hash: str
    factor_signal_hash: str
    label_spec_hash: str
    label_values_hash: str
    validation_receipt_hash: str
    reference_factor_hashes: Mapping[str, str]
    decay_label_result_hashes: Mapping[str, str]
    fold_id: str
    per_date_hash: str
    quantile_returns_hash: str
    summary_hash: str
    metric_metadata_hash: str
    per_date: pd.DataFrame
    quantile_returns: pd.DataFrame
    summary: Mapping[str, MetricValue]
    metric_metadata: pd.DataFrame
    schema_version: str = "factor-evaluation-report/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "factor-evaluation-report/v1":
            raise ValueError("unsupported factor evaluation report schema")
        per_date = pd.DataFrame(self.per_date).copy(deep=True)
        quantiles = pd.DataFrame(self.quantile_returns).copy(deep=True)
        summary = dict(self.summary)
        reference_hashes = dict(sorted(self.reference_factor_hashes.items()))
        decay_hashes = dict(sorted(self.decay_label_result_hashes.items()))
        metadata = pd.DataFrame(self.metric_metadata).copy(deep=True)
        if hash_frame(per_date) != self.per_date_hash:
            raise ValueError("evaluation per-date hash differs")
        if hash_frame(quantiles) != self.quantile_returns_hash:
            raise ValueError("evaluation quantile hash differs")
        if hash_json(summary) != self.summary_hash:
            raise ValueError("evaluation summary hash differs")
        if hash_frame(metadata) != self.metric_metadata_hash:
            raise ValueError("evaluation metric-metadata hash differs")
        object.__setattr__(self, "per_date", per_date)
        object.__setattr__(self, "quantile_returns", quantiles)
        object.__setattr__(self, "summary", MappingProxyType(summary))
        object.__setattr__(
            self, "reference_factor_hashes", MappingProxyType(reference_hashes)
        )
        object.__setattr__(
            self, "decay_label_result_hashes", MappingProxyType(decay_hashes)
        )
        object.__setattr__(self, "metric_metadata", metadata)

    @property
    def content_hash(self) -> str:
        return cast(
            str,
            hash_json(
                {
                    "schema_version": self.schema_version,
                    "evaluation_spec_hash": self.evaluation_spec_hash,
                    "factor_spec_hash": self.factor_spec_hash,
                    "factor_signal_hash": self.factor_signal_hash,
                    "label_spec_hash": self.label_spec_hash,
                    "label_values_hash": self.label_values_hash,
                    "validation_receipt_hash": self.validation_receipt_hash,
                    "reference_factor_hashes": dict(self.reference_factor_hashes),
                    "decay_label_result_hashes": dict(self.decay_label_result_hashes),
                    "fold_id": self.fold_id,
                    "per_date_hash": self.per_date_hash,
                    "quantile_returns_hash": self.quantile_returns_hash,
                    "summary_hash": self.summary_hash,
                    "metric_metadata_hash": self.metric_metadata_hash,
                }
            ),
        )


class FactorEvaluationSuite:
    """Evaluate one frozen factor only on signals assigned to validation."""

    def evaluate(
        self,
        evaluation_spec: EvaluationSpec,
        factor_spec: FactorSpec,
        factor: FactorResult,
        label_spec: LabelSpec,
        labels: LabelResult,
        validation: ValidationReceipt,
        *,
        fold_id: str,
        reference_factors: Mapping[str, pd.DataFrame] | None = None,
        decay_labels: Mapping[str, LabelResult] | None = None,
    ) -> FactorEvaluationReport:
        _validate_bindings(factor_spec, factor, label_spec, labels, validation)
        references = {
            name: pd.DataFrame(value).copy(deep=True)
            for name, value in (reference_factors or {}).items()
        }
        decay = dict(decay_labels or {})
        for result in decay.values():
            result.verify_content()
        try:
            fold = next(item for item in validation.folds if item.fold_id == fold_id)
        except StopIteration:
            raise KeyError(f"validation fold is unavailable:{fold_id}") from None
        factor_index = pd.DatetimeIndex(factor.signal.index)
        timestamps = pd.DatetimeIndex(pd.to_datetime(list(fold.validation_signals)))
        if factor_index.tz is None or timestamps.tz is None:
            raise ValueError("evaluation timestamps must be timezone-aware")
        timestamps = timestamps.tz_convert(factor_index.tz)
        signal, target, valid = _aligned_panels(factor.signal, labels, timestamps)
        directed = signal * int(factor_spec.direction)
        per_date, quantile_returns = _per_date_metrics(
            directed,
            signal,
            target,
            valid,
            evaluation_spec,
        )
        summary: dict[str, MetricValue] = _summarize(
            per_date,
            quantile_returns,
            quantile_count=evaluation_spec.quantile_count,
        )
        summary.update(_serial_dependence(directed, valid))
        summary.update(
            _reference_metrics(
                directed,
                target,
                valid,
                references,
                timestamps=timestamps,
                minimum=evaluation_spec.minimum_cross_sectional_observations,
            )
        )
        summary.update(
            _decay_metrics(
                directed,
                decay,
                timestamps=timestamps,
                minimum=evaluation_spec.minimum_cross_sectional_observations,
            )
        )
        summary = {key: summary[key] for key in sorted(summary)}
        per_date_hash = hash_frame(per_date)
        quantile_returns_hash = hash_frame(quantile_returns)
        metric_metadata = _metric_metadata(
            summary,
            per_date=per_date,
            quantiles=quantile_returns,
            per_date_hash=per_date_hash,
            quantile_hash=quantile_returns_hash,
        )
        return FactorEvaluationReport(
            evaluation_spec_hash=evaluation_spec.content_hash,
            factor_spec_hash=factor_spec.content_hash,
            factor_signal_hash=factor.signal_hash,
            label_spec_hash=label_spec.content_hash,
            label_values_hash=labels.labels_hash,
            validation_receipt_hash=validation.content_hash,
            reference_factor_hashes={
                name: hash_frame(value) for name, value in sorted(references.items())
            },
            decay_label_result_hashes={
                name: _label_result_hash(value) for name, value in sorted(decay.items())
            },
            fold_id=fold_id,
            per_date_hash=per_date_hash,
            quantile_returns_hash=quantile_returns_hash,
            summary_hash=hash_json(summary),
            metric_metadata_hash=hash_frame(metric_metadata),
            per_date=per_date,
            quantile_returns=quantile_returns,
            summary=summary,
            metric_metadata=metric_metadata,
        )


def _label_result_hash(result: LabelResult) -> str:
    return result.content_hash


def _validate_bindings(
    factor_spec: FactorSpec,
    factor: FactorResult,
    label_spec: LabelSpec,
    labels: LabelResult,
    validation: ValidationReceipt,
) -> None:
    factor.verify_content()
    labels.verify_content()
    if factor.factor_spec_hash != factor_spec.content_hash:
        raise ValueError("evaluation factor specification binding differs")
    if factor.semantic_hash != factor_spec.semantic_hash:
        raise ValueError("evaluation factor semantic binding differs")
    if factor.definition_hash != factor_spec.definition_hash:
        raise ValueError("evaluation factor definition binding differs")
    if labels.label_spec_hash != label_spec.content_hash:
        raise ValueError("evaluation label specification binding differs")
    if validation.label_spec_hash != labels.label_spec_hash:
        raise ValueError("evaluation validation/label binding differs")
    if validation.labels_hash != labels.labels_hash:
        raise ValueError("evaluation validation/label values differ")
    if validation.windows_hash != labels.windows_hash:
        raise ValueError("evaluation validation/label windows differ")


def _aligned_panels(
    factor: pd.DataFrame,
    labels: LabelResult,
    timestamps: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing_factor = timestamps.difference(factor.index)
    missing_label = timestamps.difference(labels.labels.index)
    if len(missing_factor) or len(missing_label):
        raise ValueError("validation signals are unavailable in factor or label panels")
    columns = factor.columns.intersection(labels.labels.columns, sort=False)
    if len(columns) == 0:
        raise ValueError("factor and label panels have no common securities")
    signal = factor.reindex(index=timestamps, columns=columns).astype(float)
    target = labels.labels.reindex(index=timestamps, columns=columns).astype(float)
    validity = _complete_boolean_panel(
        labels.validity,
        name="evaluation label validity",
    ).reindex(index=timestamps, columns=columns, fill_value=False)
    validity &= np.isfinite(signal) & np.isfinite(target)
    return signal, target, validity


def _per_date_metrics(
    directed: pd.DataFrame,
    raw: pd.DataFrame,
    target: pd.DataFrame,
    validity: pd.DataFrame,
    spec: EvaluationSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, MetricValue]] = []
    quantile_rows: list[dict[str, float]] = []
    for timestamp in directed.index:
        valid = validity.loc[timestamp]
        x = directed.loc[timestamp, valid]
        raw_x = raw.loc[timestamp, valid]
        y = target.loc[timestamp, valid]
        potential = int(np.isfinite(raw.loc[timestamp]).sum())
        row: dict[str, MetricValue] = {
            "observation_count": int(valid.sum()),
            "potential_signal_count": potential,
            "coverage": _safe_ratio(int(valid.sum()), potential),
            "pearson_ic": None,
            "rank_ic": None,
            "raw_rank_ic": None,
        }
        quantiles = {
            f"Q{number}": float("nan") for number in range(1, spec.quantile_count + 1)
        }
        if len(x) >= spec.minimum_cross_sectional_observations:
            row["pearson_ic"] = _correlation(x, y, method="pearson")
            row["rank_ic"] = _correlation(x, y, method="spearman")
            row["raw_rank_ic"] = _correlation(raw_x, y, method="spearman")
            ranks = x.rank(method=spec.quantile_tie_method)
            bucket = np.ceil(ranks / len(ranks) * spec.quantile_count).astype(int)
            bucket = bucket.clip(1, spec.quantile_count)
            for number in range(1, spec.quantile_count + 1):
                selected = y.loc[bucket == number]
                if len(selected):
                    quantiles[f"Q{number}"] = float(selected.mean())
        metric_rows.append(row)
        quantile_rows.append(quantiles)
    per_date = pd.DataFrame(metric_rows, index=directed.index)
    quantile_frame = pd.DataFrame(quantile_rows, index=directed.index)
    return per_date, quantile_frame


def _summarize(
    per_date: pd.DataFrame,
    quantiles: pd.DataFrame,
    *,
    quantile_count: int,
) -> dict[str, MetricValue]:
    pearson = pd.to_numeric(per_date["pearson_ic"], errors="coerce").dropna()
    rank_ic = pd.to_numeric(per_date["rank_ic"], errors="coerce").dropna()
    raw_rank_ic = pd.to_numeric(per_date["raw_rank_ic"], errors="coerce").dropna()
    coverage = pd.to_numeric(per_date["coverage"], errors="coerce").dropna()
    rank_std = float(rank_ic.std(ddof=1)) if len(rank_ic) > 1 else float("nan")
    quantile_means = quantiles.mean(axis=0, skipna=True)
    spread = quantiles[f"Q{quantile_count}"] - quantiles["Q1"]
    monotonicity = _correlation(
        pd.Series(range(1, quantile_count + 1), dtype=float),
        quantile_means.reset_index(drop=True),
        method="spearman",
    )
    summary: dict[str, MetricValue] = {
        "date_count": int(len(per_date)),
        "valid_ic_date_count": int(len(rank_ic)),
        "pearson_ic_mean": _mean_or_none(pearson),
        "pearson_ic_std": _std_or_none(pearson),
        "rank_ic_mean": _mean_or_none(rank_ic),
        "rank_ic_std": _finite_or_none(rank_std),
        "rank_ic_ir": (
            _finite_or_none(float(rank_ic.mean()) / rank_std)
            if len(rank_ic) > 1 and rank_std > 0
            else None
        ),
        "rank_ic_positive_ratio": (
            _finite_or_none(float((rank_ic > 0).mean())) if len(rank_ic) else None
        ),
        "raw_rank_ic_mean": _mean_or_none(raw_rank_ic),
        "coverage_mean": _mean_or_none(coverage),
        "top_bottom_spread_mean": _mean_or_none(spread.dropna()),
        "quantile_monotonicity": _finite_or_none(monotonicity),
    }
    for name, value in quantile_means.items():
        summary[f"{str(name).lower()}_return_mean"] = _finite_or_none(float(value))
    return summary


def _serial_dependence(
    directed: pd.DataFrame,
    validity: pd.DataFrame,
) -> dict[str, MetricValue]:
    rank_turnovers: list[float] = []
    autocorrelations: list[float] = []
    ranks = directed.rank(axis=1, method="average", pct=True)
    for offset in range(1, len(directed)):
        common = validity.iloc[offset - 1] & validity.iloc[offset]
        if int(common.sum()) < 2:
            continue
        previous_rank = ranks.iloc[offset - 1].loc[common]
        current_rank = ranks.iloc[offset].loc[common]
        rank_turnovers.append(float((current_rank - previous_rank).abs().mean()))
        autocorrelation = _correlation(
            directed.iloc[offset - 1].loc[common],
            directed.iloc[offset].loc[common],
            method="spearman",
        )
        if autocorrelation is not None:
            autocorrelations.append(autocorrelation)
    return {
        "rank_turnover_mean": _mean_or_none(pd.Series(rank_turnovers, dtype=float)),
        "factor_autocorrelation_mean": _mean_or_none(
            pd.Series(autocorrelations, dtype=float)
        ),
    }


def _reference_metrics(
    signal: pd.DataFrame,
    target: pd.DataFrame,
    validity: pd.DataFrame,
    references: Mapping[str, pd.DataFrame],
    *,
    timestamps: pd.DatetimeIndex,
    minimum: int,
) -> dict[str, MetricValue]:
    output: dict[str, MetricValue] = {}
    aligned_references: dict[str, pd.DataFrame] = {}
    for name, value in sorted(references.items()):
        reference = pd.DataFrame(value).reindex(
            index=timestamps, columns=signal.columns
        )
        aligned_references[name] = reference
        correlations: list[float] = []
        for timestamp in timestamps:
            valid = validity.loc[timestamp] & np.isfinite(reference.loc[timestamp])
            correlation = _correlation(
                signal.loc[timestamp, valid],
                reference.loc[timestamp, valid],
                method="spearman",
            )
            if correlation is not None:
                correlations.append(correlation)
        output[f"factor_correlation__{name}"] = _mean_or_none(
            pd.Series(correlations, dtype=float)
        )
    if not aligned_references:
        output["incremental_rank_ic_mean"] = None
        return output
    incremental: list[float] = []
    for timestamp in timestamps:
        valid = validity.loc[timestamp].copy()
        for reference in aligned_references.values():
            valid &= np.isfinite(reference.loc[timestamp])
        if int(valid.sum()) < max(minimum, len(aligned_references) + 2):
            continue
        design = np.column_stack(
            [
                np.ones(int(valid.sum())),
                *[
                    reference.loc[timestamp, valid].to_numpy(dtype=float)
                    for reference in aligned_references.values()
                ],
            ]
        )
        values = signal.loc[timestamp, valid].to_numpy(dtype=float)
        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
        residual_values = values - design @ coefficients
        scale = max(1.0, float(np.max(np.abs(values))))
        if float(np.std(residual_values)) <= np.finfo(float).eps * scale * 32:
            continue
        residual = pd.Series(residual_values)
        correlation = _correlation(
            residual,
            target.loc[timestamp, valid].reset_index(drop=True),
            method="spearman",
        )
        if correlation is not None:
            incremental.append(correlation)
    output["incremental_rank_ic_mean"] = _mean_or_none(
        pd.Series(incremental, dtype=float)
    )
    return output


def _decay_metrics(
    signal: pd.DataFrame,
    decay_labels: Mapping[str, LabelResult],
    *,
    timestamps: pd.DatetimeIndex,
    minimum: int,
) -> dict[str, MetricValue]:
    output: dict[str, MetricValue] = {}
    for horizon, result in sorted(decay_labels.items()):
        if len(timestamps.difference(result.labels.index)):
            raise ValueError(f"decay label lacks validation timestamps:{horizon}")
        target = result.labels.reindex(index=timestamps, columns=signal.columns)
        validity = _complete_boolean_panel(
            result.validity,
            name=f"decay label validity:{horizon}",
        ).reindex(index=timestamps, columns=signal.columns, fill_value=False)
        values: list[float] = []
        for timestamp in timestamps:
            valid = (
                validity.loc[timestamp]
                & np.isfinite(signal.loc[timestamp])
                & np.isfinite(target.loc[timestamp])
            )
            if int(valid.sum()) < minimum:
                continue
            correlation = _correlation(
                signal.loc[timestamp, valid],
                target.loc[timestamp, valid],
                method="spearman",
            )
            if correlation is not None:
                values.append(correlation)
        output[f"rank_ic_decay__{horizon}"] = _mean_or_none(
            pd.Series(values, dtype=float)
        )
    return output


def _correlation(
    left: pd.Series,
    right: pd.Series,
    *,
    method: str,
) -> float | None:
    left_values = pd.to_numeric(left, errors="coerce")
    right_values = pd.to_numeric(right, errors="coerce")
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    left_values = left_values.loc[valid]
    right_values = right_values.loc[valid]
    if len(left_values) < 2 or left_values.nunique() < 2 or right_values.nunique() < 2:
        return None
    return _finite_or_none(float(left_values.corr(right_values, method=method)))


def _complete_boolean_panel(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    values = pd.DataFrame(frame).copy(deep=True)
    if values.isna().to_numpy().any() or any(
        not is_bool_dtype(dtype) for dtype in values.dtypes
    ):
        raise TypeError(f"{name} must be a complete boolean panel")
    return values.astype(bool)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else float(numerator / denominator)


def _mean_or_none(values: pd.Series) -> float | None:
    return None if len(values) == 0 else _finite_or_none(float(values.mean()))


def _std_or_none(values: pd.Series) -> float | None:
    return None if len(values) < 2 else _finite_or_none(float(values.std(ddof=1)))


def _finite_or_none(value: float | None) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


def _metric_metadata(
    summary: Mapping[str, MetricValue],
    *,
    per_date: pd.DataFrame,
    quantiles: pd.DataFrame,
    per_date_hash: str,
    quantile_hash: str,
) -> pd.DataFrame:
    valid_ic_dates = int(
        pd.to_numeric(per_date["rank_ic"], errors="coerce").notna().sum()
    )
    complete_quantile_dates = int(quantiles.notna().all(axis=1).sum())
    rows: list[dict[str, object]] = []
    for name, value in sorted(summary.items()):
        quantile_metric = (name.startswith("q") and "_return_" in name) or name in {
            "top_bottom_spread_mean",
            "quantile_monotonicity",
        }
        if name in {"date_count", "valid_ic_date_count"}:
            unit = "count"
        elif "return" in name or "spread" in name:
            unit = "decimal_return"
        elif "turnover" in name:
            unit = "mean_percentile_rank_change"
        elif "coverage" in name or "positive_ratio" in name:
            unit = "ratio"
        else:
            unit = "correlation"
        if "ic" in name and name not in {"valid_ic_date_count"}:
            sample_count = valid_ic_dates
        elif quantile_metric:
            sample_count = complete_quantile_dates
        elif name in {"rank_turnover_mean", "factor_autocorrelation_mean"}:
            sample_count = max(0, len(per_date) - 1)
        else:
            sample_count = len(per_date)
        rows.append(
            {
                "metric": name,
                "sample_count": int(sample_count),
                "unit": unit,
                "status": "ok" if value is not None else "insufficient_data",
                "source_artifact_hash": (
                    quantile_hash if quantile_metric else per_date_hash
                ),
            }
        )
    return pd.DataFrame(rows).set_index("metric")


__all__ = ["FactorEvaluationReport", "FactorEvaluationSuite"]
