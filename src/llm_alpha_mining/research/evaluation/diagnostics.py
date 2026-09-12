"""Pure, descriptive smoke diagnostics for sparse research panels.

This module deliberately does not implement statistical inference, factor
admission, annualisation, portfolio performance, or significance tests.  It is
intended for small end-to-end data/label smoke tests where reporting a formal
IC or an admission decision would overstate the available evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_numeric_dtype,
)

from llm_alpha_mining.research.core.hashing import hash_frame, hash_json, require_sha256


SMOKE_DIAGNOSTICS_SCHEMA = "descriptive-smoke-diagnostics/v2"
_QUANTILE_COUNT = 5
_QUANTILE_ASSIGNMENT_METHOD = "ceil_average_rank_times_quantiles_over_count"
_PER_DATE_COLUMNS = (
    "factor_id",
    "timestamp",
    "intended_domain_count",
    "full_universe_count",
    "finite_count",
    "descriptive_observation_count",
    "descriptive_pearson",
    "descriptive_spearman",
    "q1_mean_return",
    "q2_mean_return",
    "q3_mean_return",
    "q4_mean_return",
    "q5_mean_return",
    "q1_observation_count",
    "q2_observation_count",
    "q3_observation_count",
    "q4_observation_count",
    "q5_observation_count",
    "minimum_quantile_observation_count",
    "q5_minus_q1_return",
    "intended_coverage",
    "full_universe_coverage",
    "label_validity_coverage",
    "dispersion",
    "tie_rate",
)
_PAIRWISE_COLUMNS = (
    "left_factor_id",
    "right_factor_id",
    "timestamp",
    "common_domain_count",
    "common_finite_count",
    "common_domain_coverage",
    "descriptive_pearson",
    "descriptive_spearman",
)


@dataclass(frozen=True, slots=True)
class DescriptiveSmokeDiagnosticsResult:
    """Content-addressed output of descriptive smoke diagnostics.

    DataFrames are defensive copies.  They remain mutable because pandas does
    not provide an immutable frame type; :meth:`verify_content` detects any
    subsequent mutation before the result is consumed or serialized.
    """

    minimum_observations: int
    factor_ids: tuple[str, ...]
    factor_hashes: Mapping[str, str]
    factor_domain_mask_hashes: Mapping[str, str]
    full_universe_mask_hashes: Mapping[str, str]
    labels_hash: str
    label_validity_hash: str
    input_hash: str
    per_date_hash: str
    pairwise_factor_correlation_hash: str
    result_hash: str
    per_date: pd.DataFrame
    pairwise_factor_correlation: pd.DataFrame
    diagnostic_only: bool = True
    formal_inference_permitted: bool = False
    admission_decision_permitted: bool = False
    schema_version: str = SMOKE_DIAGNOSTICS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SMOKE_DIAGNOSTICS_SCHEMA:
            raise ValueError("unsupported descriptive smoke diagnostics schema")
        if (
            self.diagnostic_only is not True
            or self.formal_inference_permitted is not False
            or self.admission_decision_permitted is not False
        ):
            raise ValueError(
                "smoke diagnostics cannot authorize inference or admission"
            )
        if (
            not isinstance(self.minimum_observations, int)
            or isinstance(self.minimum_observations, bool)
            or self.minimum_observations < _QUANTILE_COUNT
        ):
            raise ValueError("smoke minimum_observations must cover all five quantiles")
        factor_ids = tuple(self.factor_ids)
        if (
            not factor_ids
            or factor_ids != tuple(sorted(factor_ids))
            or len(factor_ids) != len(set(factor_ids))
            or any(not item.strip() for item in factor_ids)
        ):
            raise ValueError(
                "smoke factor_ids must be non-empty, unique, and canonical"
            )
        factor_hashes = _validated_hash_mapping(
            self.factor_hashes,
            expected_keys=factor_ids,
            name="factor hashes",
        )
        domain_hashes = _validated_hash_mapping(
            self.factor_domain_mask_hashes,
            expected_keys=factor_ids,
            name="factor-domain mask hashes",
        )
        full_hashes = _validated_hash_mapping(
            self.full_universe_mask_hashes,
            expected_keys=factor_ids,
            name="full-universe mask hashes",
        )
        for value, name in (
            (self.labels_hash, "smoke labels hash"),
            (self.label_validity_hash, "smoke label-validity hash"),
            (self.input_hash, "smoke input hash"),
            (self.per_date_hash, "smoke per-date hash"),
            (
                self.pairwise_factor_correlation_hash,
                "smoke pairwise-factor-correlation hash",
            ),
            (self.result_hash, "smoke result hash"),
        ):
            require_sha256(value, name=name)

        expected_input_hash = _input_hash(
            factor_hashes=factor_hashes,
            factor_domain_mask_hashes=domain_hashes,
            full_universe_mask_hashes=full_hashes,
            labels_hash=self.labels_hash,
            label_validity_hash=self.label_validity_hash,
        )
        if expected_input_hash != self.input_hash:
            raise ValueError("smoke input hash differs from component hashes")

        per_date = pd.DataFrame(self.per_date).copy(deep=True)
        pairwise = pd.DataFrame(self.pairwise_factor_correlation).copy(deep=True)
        if tuple(per_date.columns) != _PER_DATE_COLUMNS:
            raise ValueError("smoke per-date columns differ from schema")
        if tuple(pairwise.columns) != _PAIRWISE_COLUMNS:
            raise ValueError("smoke pairwise columns differ from schema")
        observed_ids = tuple(sorted(set(per_date["factor_id"].astype(str))))
        if observed_ids != factor_ids:
            raise ValueError("smoke per-date factor ids differ")
        if hash_frame(per_date) != self.per_date_hash:
            raise ValueError("smoke per-date hash differs from content")
        if hash_frame(pairwise) != self.pairwise_factor_correlation_hash:
            raise ValueError(
                "smoke pairwise-factor-correlation hash differs from content"
            )
        expected_result_hash = hash_json(
            _result_identity_payload(
                schema_version=self.schema_version,
                minimum_observations=self.minimum_observations,
                factor_ids=factor_ids,
                input_hash=self.input_hash,
                per_date_hash=self.per_date_hash,
                pairwise_hash=self.pairwise_factor_correlation_hash,
                diagnostic_only=self.diagnostic_only,
                formal_inference_permitted=self.formal_inference_permitted,
                admission_decision_permitted=self.admission_decision_permitted,
            )
        )
        if expected_result_hash != self.result_hash:
            raise ValueError("smoke result hash differs from content identity")

        object.__setattr__(self, "factor_ids", factor_ids)
        object.__setattr__(self, "factor_hashes", MappingProxyType(factor_hashes))
        object.__setattr__(
            self,
            "factor_domain_mask_hashes",
            MappingProxyType(domain_hashes),
        )
        object.__setattr__(
            self,
            "full_universe_mask_hashes",
            MappingProxyType(full_hashes),
        )
        object.__setattr__(self, "per_date", per_date)
        object.__setattr__(self, "pairwise_factor_correlation", pairwise)

    @property
    def content_hash(self) -> str:
        self.verify_content()
        return self.result_hash

    def verify_content(self) -> None:
        if hash_frame(self.per_date) != self.per_date_hash:
            raise RuntimeError("smoke per-date content changed after construction")
        if (
            hash_frame(self.pairwise_factor_correlation)
            != self.pairwise_factor_correlation_hash
        ):
            raise RuntimeError("smoke pairwise content changed after construction")
        expected = hash_json(
            _result_identity_payload(
                schema_version=self.schema_version,
                minimum_observations=self.minimum_observations,
                factor_ids=self.factor_ids,
                input_hash=self.input_hash,
                per_date_hash=self.per_date_hash,
                pairwise_hash=self.pairwise_factor_correlation_hash,
                diagnostic_only=self.diagnostic_only,
                formal_inference_permitted=self.formal_inference_permitted,
                admission_decision_permitted=self.admission_decision_permitted,
            )
        )
        if expected != self.result_hash:
            raise RuntimeError("smoke result identity changed after construction")

    def descriptor(self) -> dict[str, object]:
        self.verify_content()
        return {
            "schema_version": self.schema_version,
            "minimum_observations": self.minimum_observations,
            "quantile_count": _QUANTILE_COUNT,
            "quantile_assignment_method": _QUANTILE_ASSIGNMENT_METHOD,
            "factor_ids": list(self.factor_ids),
            "factor_hashes": dict(self.factor_hashes),
            "factor_domain_mask_hashes": dict(self.factor_domain_mask_hashes),
            "full_universe_mask_hashes": dict(self.full_universe_mask_hashes),
            "labels_hash": self.labels_hash,
            "label_validity_hash": self.label_validity_hash,
            "input_hash": self.input_hash,
            "per_date_hash": self.per_date_hash,
            "pairwise_factor_correlation_hash": (self.pairwise_factor_correlation_hash),
            "result_hash": self.result_hash,
            "per_date_row_count": len(self.per_date),
            "pairwise_row_count": len(self.pairwise_factor_correlation),
            "diagnostic_only": self.diagnostic_only,
            "formal_inference_permitted": (self.formal_inference_permitted),
            "admission_decision_permitted": (self.admission_decision_permitted),
        }


def build_descriptive_smoke_diagnostics(
    *,
    factors: Mapping[str, pd.DataFrame],
    labels: pd.DataFrame,
    factor_domain_masks: Mapping[str, pd.DataFrame],
    full_universe_masks: Mapping[str, pd.DataFrame] | pd.DataFrame,
    label_validity: pd.DataFrame,
    minimum_observations: int = _QUANTILE_COUNT,
) -> DescriptiveSmokeDiagnosticsResult:
    """Compute fail-closed descriptive diagnostics on exact common axes.

    ``factor_domain_masks`` define each factor's intended economic/data domain.
    ``full_universe_masks`` may be a factor-specific mapping or one shared mask.
    Every intended domain must be a subset of its corresponding full universe.
    Missing and non-finite factor values remain absent and are never converted
    to zero.
    """

    if (
        not isinstance(minimum_observations, int)
        or isinstance(minimum_observations, bool)
        or minimum_observations < _QUANTILE_COUNT
    ):
        raise ValueError("smoke minimum_observations must cover all five quantiles")
    factor_ids = _factor_ids(factors)
    if set(factor_domain_masks) != set(factor_ids):
        raise ValueError("factor-domain mask ids differ from factor ids")
    full_by_factor = _full_universe_by_factor(
        full_universe_masks,
        factor_ids=factor_ids,
    )

    canonical_labels = _numeric_panel(labels, name="labels")
    canonical_validity = _boolean_panel(
        label_validity,
        name="label validity",
        reference=canonical_labels,
    )
    canonical_factors: dict[str, pd.DataFrame] = {}
    canonical_domains: dict[str, pd.DataFrame] = {}
    canonical_full: dict[str, pd.DataFrame] = {}
    for factor_id in factor_ids:
        factor = _numeric_panel(
            factors[factor_id],
            name=f"factor:{factor_id}",
            reference=canonical_labels,
        )
        domain = _boolean_panel(
            factor_domain_masks[factor_id],
            name=f"factor domain:{factor_id}",
            reference=canonical_labels,
        )
        full = _boolean_panel(
            full_by_factor[factor_id],
            name=f"full universe:{factor_id}",
            reference=canonical_labels,
        )
        outside = domain & ~full
        if bool(outside.to_numpy(dtype=bool).any()):
            raise ValueError(
                f"factor domain is not a subset of full universe:{factor_id}"
            )
        canonical_factors[factor_id] = factor
        canonical_domains[factor_id] = domain
        canonical_full[factor_id] = full

    factor_hashes = {
        factor_id: hash_frame(canonical_factors[factor_id]) for factor_id in factor_ids
    }
    domain_hashes = {
        factor_id: hash_frame(canonical_domains[factor_id]) for factor_id in factor_ids
    }
    full_hashes = {
        factor_id: hash_frame(canonical_full[factor_id]) for factor_id in factor_ids
    }
    labels_hash = hash_frame(canonical_labels)
    validity_hash = hash_frame(canonical_validity)
    input_hash = _input_hash(
        factor_hashes=factor_hashes,
        factor_domain_mask_hashes=domain_hashes,
        full_universe_mask_hashes=full_hashes,
        labels_hash=labels_hash,
        label_validity_hash=validity_hash,
    )

    per_date = _build_per_date(
        factor_ids=factor_ids,
        factors=canonical_factors,
        labels=canonical_labels,
        factor_domains=canonical_domains,
        full_universes=canonical_full,
        label_validity=canonical_validity,
        minimum_observations=minimum_observations,
    )
    pairwise = _build_pairwise(
        factor_ids=factor_ids,
        factors=canonical_factors,
        factor_domains=canonical_domains,
        full_universes=canonical_full,
        minimum_observations=minimum_observations,
    )
    per_date_hash = hash_frame(per_date)
    pairwise_hash = hash_frame(pairwise)
    result_hash = hash_json(
        _result_identity_payload(
            schema_version=SMOKE_DIAGNOSTICS_SCHEMA,
            minimum_observations=minimum_observations,
            factor_ids=factor_ids,
            input_hash=input_hash,
            per_date_hash=per_date_hash,
            pairwise_hash=pairwise_hash,
            diagnostic_only=True,
            formal_inference_permitted=False,
            admission_decision_permitted=False,
        )
    )
    return DescriptiveSmokeDiagnosticsResult(
        minimum_observations=minimum_observations,
        factor_ids=factor_ids,
        factor_hashes=factor_hashes,
        factor_domain_mask_hashes=domain_hashes,
        full_universe_mask_hashes=full_hashes,
        labels_hash=labels_hash,
        label_validity_hash=validity_hash,
        input_hash=input_hash,
        per_date_hash=per_date_hash,
        pairwise_factor_correlation_hash=pairwise_hash,
        result_hash=result_hash,
        per_date=per_date,
        pairwise_factor_correlation=pairwise,
    )


def _build_per_date(
    *,
    factor_ids: tuple[str, ...],
    factors: Mapping[str, pd.DataFrame],
    labels: pd.DataFrame,
    factor_domains: Mapping[str, pd.DataFrame],
    full_universes: Mapping[str, pd.DataFrame],
    label_validity: pd.DataFrame,
    minimum_observations: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    finite_labels = _finite_mask(labels)
    effective_label_validity = label_validity & finite_labels
    for factor_id in factor_ids:
        factor = factors[factor_id]
        domain = factor_domains[factor_id]
        full = full_universes[factor_id]
        finite_factor = _finite_mask(factor)
        for timestamp in factor.index:
            intended = domain.loc[timestamp] & full.loc[timestamp]
            full_row = full.loc[timestamp]
            finite = intended & finite_factor.loc[timestamp]
            descriptive = finite & effective_label_validity.loc[timestamp]
            values = factor.loc[timestamp, finite].astype(float)
            x = factor.loc[timestamp, descriptive].astype(float)
            y = labels.loc[timestamp, descriptive].astype(float)
            quantiles = _quantile_means(
                x,
                y,
                minimum_observations=minimum_observations,
            )
            finite_count = int(finite.sum())
            observation_count = int(descriptive.sum())
            rows.append(
                {
                    "factor_id": factor_id,
                    "timestamp": timestamp,
                    "intended_domain_count": int(intended.sum()),
                    "full_universe_count": int(full_row.sum()),
                    "finite_count": finite_count,
                    "descriptive_observation_count": observation_count,
                    "descriptive_pearson": _descriptive_correlation(
                        x,
                        y,
                        method="pearson",
                        minimum_observations=minimum_observations,
                    ),
                    "descriptive_spearman": _descriptive_correlation(
                        x,
                        y,
                        method="spearman",
                        minimum_observations=minimum_observations,
                    ),
                    **quantiles,
                    "q5_minus_q1_return": _difference(
                        quantiles["q5_mean_return"],
                        quantiles["q1_mean_return"],
                    ),
                    "intended_coverage": _safe_ratio(
                        finite_count,
                        int(intended.sum()),
                    ),
                    "full_universe_coverage": _safe_ratio(
                        finite_count,
                        int(full_row.sum()),
                    ),
                    "label_validity_coverage": _safe_ratio(
                        observation_count,
                        finite_count,
                    ),
                    "dispersion": _dispersion(values),
                    "tie_rate": _tie_rate(values),
                }
            )
    return pd.DataFrame(rows, columns=_PER_DATE_COLUMNS)


def _build_pairwise(
    *,
    factor_ids: tuple[str, ...],
    factors: Mapping[str, pd.DataFrame],
    factor_domains: Mapping[str, pd.DataFrame],
    full_universes: Mapping[str, pd.DataFrame],
    minimum_observations: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left_position, left_id in enumerate(factor_ids):
        for right_id in factor_ids[left_position + 1 :]:
            left = factors[left_id]
            right = factors[right_id]
            common_domain_panel = (
                factor_domains[left_id]
                & factor_domains[right_id]
                & full_universes[left_id]
                & full_universes[right_id]
            )
            left_finite = _finite_mask(left)
            right_finite = _finite_mask(right)
            for timestamp in left.index:
                common_domain = common_domain_panel.loc[timestamp]
                common_finite = (
                    common_domain
                    & left_finite.loc[timestamp]
                    & right_finite.loc[timestamp]
                )
                x = left.loc[timestamp, common_finite].astype(float)
                y = right.loc[timestamp, common_finite].astype(float)
                common_count = int(common_domain.sum())
                finite_count = int(common_finite.sum())
                rows.append(
                    {
                        "left_factor_id": left_id,
                        "right_factor_id": right_id,
                        "timestamp": timestamp,
                        "common_domain_count": common_count,
                        "common_finite_count": finite_count,
                        "common_domain_coverage": _safe_ratio(
                            finite_count,
                            common_count,
                        ),
                        "descriptive_pearson": _descriptive_correlation(
                            x,
                            y,
                            method="pearson",
                            minimum_observations=minimum_observations,
                        ),
                        "descriptive_spearman": _descriptive_correlation(
                            x,
                            y,
                            method="spearman",
                            minimum_observations=minimum_observations,
                        ),
                    }
                )
    return pd.DataFrame(rows, columns=_PAIRWISE_COLUMNS)


def _numeric_panel(
    frame: pd.DataFrame,
    *,
    name: str,
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    values = _canonical_axes(frame, name=name, reference=reference)
    for column in values.columns:
        dtype = values[column].dtype
        if (
            not is_numeric_dtype(dtype)
            or is_bool_dtype(dtype)
            or is_complex_dtype(dtype)
        ):
            raise TypeError(f"{name} must contain only real numeric columns")
    return values.astype(float)


def _boolean_panel(
    frame: pd.DataFrame,
    *,
    name: str,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    values = _canonical_axes(frame, name=name, reference=reference)
    if values.isna().any().any() or any(
        not is_bool_dtype(dtype) for dtype in values.dtypes
    ):
        raise TypeError(f"{name} must be a complete boolean panel")
    return values.astype(bool)


def _canonical_axes(
    frame: pd.DataFrame,
    *,
    name: str,
    reference: pd.DataFrame | None,
) -> pd.DataFrame:
    values = pd.DataFrame(frame).copy(deep=True)
    index = pd.DatetimeIndex(values.index)
    if index.tz is None:
        raise ValueError(f"{name} timestamps must be timezone-aware")
    if not index.is_unique:
        raise ValueError(f"{name} timestamps must be unique")
    columns = [str(item) for item in values.columns]
    if len(columns) != len(set(columns)):
        raise ValueError(f"{name} security columns are ambiguous")
    values.index = index
    values.columns = columns
    values = values.sort_index().sort_index(axis=1)
    if reference is not None:
        reference_index = pd.DatetimeIndex(reference.index)
        values.index = values.index.tz_convert(reference_index.tz)
        if not values.index.equals(reference.index) or not values.columns.equals(
            reference.columns
        ):
            raise ValueError(f"{name} axes differ from labels")
    return values


def _factor_ids(factors: Mapping[str, pd.DataFrame]) -> tuple[str, ...]:
    if not factors:
        raise ValueError("smoke diagnostics require at least one factor")
    if any(not isinstance(key, str) or not key.strip() for key in factors):
        raise ValueError("smoke factor ids must be non-empty strings")
    factor_ids = tuple(sorted(factors))
    if len(factor_ids) != len(set(factor_ids)):
        raise ValueError("smoke factor ids must be unique")
    return factor_ids


def _full_universe_by_factor(
    masks: Mapping[str, pd.DataFrame] | pd.DataFrame,
    *,
    factor_ids: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    if isinstance(masks, pd.DataFrame):
        return {factor_id: masks for factor_id in factor_ids}
    if set(masks) != set(factor_ids):
        raise ValueError("full-universe mask ids differ from factor ids")
    return {factor_id: masks[factor_id] for factor_id in factor_ids}


def _finite_mask(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        np.isfinite(frame.to_numpy(dtype=float)),
        index=frame.index,
        columns=frame.columns,
        dtype=bool,
    )


def _descriptive_correlation(
    left: pd.Series,
    right: pd.Series,
    *,
    method: str,
    minimum_observations: int,
) -> float:
    if len(left) < minimum_observations:
        return float("nan")
    if left.nunique(dropna=True) <= 1 or right.nunique(dropna=True) <= 1:
        return float("nan")
    if method == "spearman":
        left = left.rank(method="average")
        right = right.rank(method="average")
    elif method != "pearson":
        raise ValueError(f"unsupported descriptive correlation:{method}")
    value = left.corr(right, method="pearson")
    return float(value) if value is not None and np.isfinite(value) else float("nan")


def _quantile_means(
    factor: pd.Series,
    labels: pd.Series,
    *,
    minimum_observations: int,
) -> dict[str, float | int]:
    output: dict[str, float | int] = {
        f"q{number}_mean_return": float("nan")
        for number in range(1, _QUANTILE_COUNT + 1)
    }
    output.update(
        {f"q{number}_observation_count": 0 for number in range(1, _QUANTILE_COUNT + 1)}
    )
    output["minimum_quantile_observation_count"] = 0
    if len(factor) < minimum_observations or factor.nunique(dropna=True) <= 1:
        return output
    # Average ranks keep equal factor values in the same bucket.  Dividing the
    # average rank by the complete cross-sectional count creates five equal-
    # frequency target intervals for unique values; unlike min-max scaling, it
    # does not collapse Q5 to the single maximum observation.  Ties that cross
    # a target boundary stay together, so empty or imbalanced buckets are an
    # explicit and order-invariant consequence of the data.
    # Empty buckets are preferable to a result that depends on security-column
    # ordering, especially in sparse infrastructure smoke samples.
    average_ranks = factor.rank(method="average")
    buckets = (
        np.ceil(average_ranks * _QUANTILE_COUNT / float(len(average_ranks)))
        .astype(int)
        .clip(1, _QUANTILE_COUNT)
    )
    counts: list[int] = []
    for number in range(1, _QUANTILE_COUNT + 1):
        selected = labels.loc[buckets == number]
        count = len(selected)
        counts.append(count)
        output[f"q{number}_observation_count"] = count
        if count:
            output[f"q{number}_mean_return"] = float(selected.mean())
    output["minimum_quantile_observation_count"] = min(counts)
    return output


def _dispersion(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(values.std(ddof=0))


def _tie_rate(values: pd.Series) -> float:
    count = len(values)
    if count < 2:
        return float("nan")
    group_sizes = values.value_counts(dropna=False).to_numpy(dtype=float)
    tied_pairs = float(np.sum(group_sizes * (group_sizes - 1.0)))
    return tied_pairs / float(count * (count - 1))


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _difference(left: float, right: float) -> float:
    return (
        float(left - right)
        if np.isfinite(left) and np.isfinite(right)
        else float("nan")
    )


def _validated_hash_mapping(
    values: Mapping[str, str],
    *,
    expected_keys: tuple[str, ...],
    name: str,
) -> dict[str, str]:
    copied = {str(key): str(value) for key, value in values.items()}
    if set(copied) != set(expected_keys):
        raise ValueError(f"{name} ids differ")
    return {
        key: require_sha256(copied[key], name=f"{name}:{key}") for key in expected_keys
    }


def _input_hash(
    *,
    factor_hashes: Mapping[str, str],
    factor_domain_mask_hashes: Mapping[str, str],
    full_universe_mask_hashes: Mapping[str, str],
    labels_hash: str,
    label_validity_hash: str,
) -> str:
    return str(
        hash_json(
            {
                "schema_version": "descriptive-smoke-input/v1",
                "factor_hashes": dict(factor_hashes),
                "factor_domain_mask_hashes": dict(factor_domain_mask_hashes),
                "full_universe_mask_hashes": dict(full_universe_mask_hashes),
                "labels_hash": labels_hash,
                "label_validity_hash": label_validity_hash,
            }
        )
    )


def _result_identity_payload(
    *,
    schema_version: str,
    minimum_observations: int,
    factor_ids: tuple[str, ...],
    input_hash: str,
    per_date_hash: str,
    pairwise_hash: str,
    diagnostic_only: bool,
    formal_inference_permitted: bool,
    admission_decision_permitted: bool,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "minimum_observations": minimum_observations,
        "quantile_count": _QUANTILE_COUNT,
        "quantile_assignment_method": _QUANTILE_ASSIGNMENT_METHOD,
        "factor_ids": list(factor_ids),
        "input_hash": input_hash,
        "per_date_hash": per_date_hash,
        "pairwise_factor_correlation_hash": pairwise_hash,
        "diagnostic_only": diagnostic_only,
        "formal_inference_permitted": formal_inference_permitted,
        "admission_decision_permitted": admission_decision_permitted,
    }


__all__ = [
    "DescriptiveSmokeDiagnosticsResult",
    "SMOKE_DIAGNOSTICS_SCHEMA",
    "build_descriptive_smoke_diagnostics",
]
