from __future__ import annotations

import hmac
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import norm

from alpha_research.core.hashing import hash_frame, hash_json, require_sha256
from alpha_research.research.campaign_protocol import ResearchCampaignProtocol
from alpha_research.robustness.runner import benjamini_hochberg


class SignificanceMethod(str, Enum):
    HAC = "hac"
    BLOCK_BOOTSTRAP = "block_bootstrap"


@dataclass(frozen=True, slots=True)
class SignificanceTestSpec:
    """Pre-registered primary test for a campaign-wide daily IC family."""

    test_id: str
    version: str
    method: SignificanceMethod | str
    minimum_observations: int
    hac_lags: int | None
    bootstrap_trials: int | None
    block_length: int | None
    random_seed: int
    alternative: str = "greater"
    direction_policy: str = "hypothesis_declared_sign_applied_before_test"
    missing_or_failed_p_value: float = 1.0
    multiple_testing_method: str = "benjamini_hochberg"
    multiple_testing_scope: str = "full_tested_family"
    schema_version: str = "significance-test-spec/v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", SignificanceMethod(self.method))
        if self.schema_version != "significance-test-spec/v1":
            raise ValueError("unsupported significance test schema")
        if not self.test_id.strip() or not self.version.strip():
            raise ValueError("significance test id and version are required")
        if (
            not isinstance(self.minimum_observations, int)
            or isinstance(self.minimum_observations, bool)
            or self.minimum_observations < 20
        ):
            raise ValueError(
                "significance test requires at least 20 daily observations"
            )
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise TypeError("significance random seed must be an integer")
        if self.alternative != "greater":
            raise ValueError(
                "primary test must use the predeclared, positive directed alternative"
            )
        if self.direction_policy != "hypothesis_declared_sign_applied_before_test":
            raise ValueError("primary test direction policy differs")
        if (
            not isinstance(self.missing_or_failed_p_value, (int, float))
            or isinstance(self.missing_or_failed_p_value, bool)
            or float(self.missing_or_failed_p_value) != 1.0
        ):
            raise ValueError("missing and failed tests must be assigned p=1")
        object.__setattr__(self, "missing_or_failed_p_value", 1.0)
        if self.multiple_testing_method != "benjamini_hochberg":
            raise ValueError("campaign significance requires Benjamini-Hochberg")
        if self.multiple_testing_scope != "full_tested_family":
            raise ValueError("BH must include the full frozen hypothesis family")
        self._validate_method_parameters()

    def _validate_method_parameters(self) -> None:
        method = self.method
        if not isinstance(method, SignificanceMethod):  # pragma: no cover
            raise RuntimeError("significance method was not normalized")
        if method is SignificanceMethod.HAC:
            if (
                not isinstance(self.hac_lags, int)
                or isinstance(self.hac_lags, bool)
                or self.hac_lags < 0
                or self.hac_lags >= self.minimum_observations
            ):
                raise ValueError("HAC lags must lie in [0, minimum_observations)")
            if self.bootstrap_trials is not None or self.block_length is not None:
                raise ValueError("HAC test cannot declare bootstrap parameters")
        else:
            if self.hac_lags is not None:
                raise ValueError("block bootstrap cannot declare HAC lags")
            if (
                not isinstance(self.bootstrap_trials, int)
                or isinstance(self.bootstrap_trials, bool)
                or self.bootstrap_trials < 2_000
            ):
                raise ValueError("block bootstrap requires at least 2000 trials")
            if (
                not isinstance(self.block_length, int)
                or isinstance(self.block_length, bool)
                or self.block_length < 2
                or self.block_length >= self.minimum_observations
            ):
                raise ValueError("block length must lie in [2, minimum_observations)")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        method = self.method
        if not isinstance(method, SignificanceMethod):  # pragma: no cover
            raise RuntimeError("significance method was not normalized")
        return {
            "schema_version": self.schema_version,
            "test_id": self.test_id,
            "version": self.version,
            "method": method.value,
            "minimum_observations": self.minimum_observations,
            "hac_lags": self.hac_lags,
            "bootstrap_trials": self.bootstrap_trials,
            "block_length": self.block_length,
            "random_seed": self.random_seed,
            "alternative": self.alternative,
            "direction_policy": self.direction_policy,
            "missing_or_failed_p_value": self.missing_or_failed_p_value,
            "multiple_testing_method": self.multiple_testing_method,
            "multiple_testing_scope": self.multiple_testing_scope,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "SignificanceTestSpec":
        expected = {
            "schema_version",
            "test_id",
            "version",
            "method",
            "minimum_observations",
            "hac_lags",
            "bootstrap_trials",
            "block_length",
            "random_seed",
            "alternative",
            "direction_policy",
            "missing_or_failed_p_value",
            "multiple_testing_method",
            "multiple_testing_scope",
        }
        if set(value) != expected:
            raise ValueError("SignificanceTestSpec wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            test_id=_text(value["test_id"], "test_id"),
            version=_text(value["version"], "version"),
            method=_text(value["method"], "method"),
            minimum_observations=_integer(
                value["minimum_observations"], "minimum_observations"
            ),
            hac_lags=_optional_integer(value["hac_lags"], "hac_lags"),
            bootstrap_trials=_optional_integer(
                value["bootstrap_trials"], "bootstrap_trials"
            ),
            block_length=_optional_integer(value["block_length"], "block_length"),
            random_seed=_integer(value["random_seed"], "random_seed"),
            alternative=_text(value["alternative"], "alternative"),
            direction_policy=_text(value["direction_policy"], "direction_policy"),
            missing_or_failed_p_value=_number(
                value["missing_or_failed_p_value"],
                "missing_or_failed_p_value",
            ),
            multiple_testing_method=_text(
                value["multiple_testing_method"], "multiple_testing_method"
            ),
            multiple_testing_scope=_text(
                value["multiple_testing_scope"], "multiple_testing_scope"
            ),
        )


@dataclass(frozen=True, slots=True)
class FrozenSignificanceCandidate:
    """One pre-registered hypothesis, including its direction before testing."""

    candidate_id: str
    factor_spec_hash: str
    hypothesis_hash: str
    declared_sign: int
    schema_version: str = "frozen-significance-candidate/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "frozen-significance-candidate/v1":
            raise ValueError("unsupported frozen significance candidate schema")
        if (
            not isinstance(self.candidate_id, str)
            or not self.candidate_id
            or self.candidate_id != self.candidate_id.strip()
        ):
            raise ValueError("candidate_id must be a canonical non-empty string")
        require_sha256(self.factor_spec_hash, name="candidate factor_spec_hash")
        require_sha256(self.hypothesis_hash, name="candidate hypothesis_hash")
        if (
            not isinstance(self.declared_sign, int)
            or isinstance(self.declared_sign, bool)
            or self.declared_sign not in {-1, 1}
        ):
            raise ValueError("candidate declared_sign must be -1 or 1")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "factor_spec_hash": self.factor_spec_hash,
            "hypothesis_hash": self.hypothesis_hash,
            "declared_sign": self.declared_sign,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> "FrozenSignificanceCandidate":
        expected = {
            "schema_version",
            "candidate_id",
            "factor_spec_hash",
            "hypothesis_hash",
            "declared_sign",
        }
        if set(value) != expected:
            raise ValueError("FrozenSignificanceCandidate wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            candidate_id=_text(value["candidate_id"], "candidate_id"),
            factor_spec_hash=_text(value["factor_spec_hash"], "factor_spec_hash"),
            hypothesis_hash=_text(value["hypothesis_hash"], "hypothesis_hash"),
            declared_sign=_integer(value["declared_sign"], "declared_sign"),
        )


@dataclass(frozen=True, slots=True)
class FrozenSignificanceFamily:
    """Complete candidate family used as the denominator of multiple testing."""

    family_id: str
    version: str
    candidates: tuple[FrozenSignificanceCandidate, ...]
    schema_version: str = "frozen-significance-family/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "frozen-significance-family/v1":
            raise ValueError("unsupported frozen significance family schema")
        if (
            not isinstance(self.family_id, str)
            or not self.family_id.strip()
            or not isinstance(self.version, str)
            or not self.version.strip()
        ):
            raise ValueError("significance family id and version are required")
        candidates = tuple(self.candidates)
        if not candidates or any(
            type(candidate) is not FrozenSignificanceCandidate
            for candidate in candidates
        ):
            raise TypeError("significance family candidates must be exact frozen types")
        ordered = tuple(sorted(candidates, key=lambda item: item.candidate_id))
        if candidates != ordered or len(
            {candidate.candidate_id for candidate in candidates}
        ) != len(candidates):
            raise ValueError("significance family candidates must be unique and sorted")
        object.__setattr__(self, "candidates", candidates)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates)

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "family_id": self.family_id,
            "version": self.version,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> "FrozenSignificanceFamily":
        expected = {"schema_version", "family_id", "version", "candidates"}
        if set(value) != expected:
            raise ValueError("FrozenSignificanceFamily wire fields differ")
        raw = value["candidates"]
        if not isinstance(raw, list) or not all(
            isinstance(candidate, Mapping) for candidate in raw
        ):
            raise TypeError("frozen significance candidates must be objects")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            family_id=_text(value["family_id"], "family_id"),
            version=_text(value["version"], "version"),
            candidates=tuple(
                FrozenSignificanceCandidate.from_mapping(candidate) for candidate in raw
            ),
        )


@dataclass(frozen=True, slots=True)
class FrozenDailyICCalendar:
    """Content-addressed trading-session calendar for the primary daily IC."""

    calendar_id: str
    version: str
    sessions: tuple[str, ...]
    source_calendar_hash: str
    timezone: str = "Asia/Shanghai"
    schema_version: str = "frozen-daily-ic-calendar/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "frozen-daily-ic-calendar/v1":
            raise ValueError("unsupported frozen daily IC calendar schema")
        if (
            not isinstance(self.calendar_id, str)
            or not self.calendar_id.strip()
            or not isinstance(self.version, str)
            or not self.version.strip()
        ):
            raise ValueError("daily IC calendar id and version are required")
        require_sha256(
            self.source_calendar_hash,
            name="daily IC source_calendar_hash",
        )
        sessions = tuple(self.sessions)
        if len(sessions) < 20 or sessions != tuple(sorted(set(sessions))):
            raise ValueError(
                "daily IC calendar sessions must be unique, sorted and at least 20"
            )
        for session in sessions:
            if (
                not isinstance(session, str)
                or len(session) != 8
                or not session.isdigit()
                or pd.Timestamp(session).strftime("%Y%m%d") != session
            ):
                raise ValueError(f"invalid daily IC calendar session:{session}")
        try:
            pd.Timestamp(sessions[0]).tz_localize(self.timezone)
        except (TypeError, ValueError) as error:
            raise ValueError("daily IC calendar timezone is invalid") from error
        object.__setattr__(self, "sessions", sessions)

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(
            [
                pd.Timestamp(session).tz_localize(self.timezone)
                for session in self.sessions
            ]
        )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "calendar_id": self.calendar_id,
            "version": self.version,
            "sessions": list(self.sessions),
            "source_calendar_hash": self.source_calendar_hash,
            "timezone": self.timezone,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> "FrozenDailyICCalendar":
        expected = {
            "schema_version",
            "calendar_id",
            "version",
            "sessions",
            "source_calendar_hash",
            "timezone",
        }
        if set(value) != expected:
            raise ValueError("FrozenDailyICCalendar wire fields differ")
        raw_sessions = value["sessions"]
        if not isinstance(raw_sessions, list) or not all(
            isinstance(session, str) for session in raw_sessions
        ):
            raise TypeError("daily IC calendar sessions must be strings")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            calendar_id=_text(value["calendar_id"], "calendar_id"),
            version=_text(value["version"], "version"),
            sessions=tuple(raw_sessions),
            source_calendar_hash=_text(
                value["source_calendar_hash"], "source_calendar_hash"
            ),
            timezone=_text(value["timezone"], "timezone"),
        )


@dataclass(frozen=True, slots=True)
class CampaignSignificanceReport:
    protocol_hash: str
    significance_spec_hash: str
    frozen_family_hash: str
    calendar_hash: str
    input_series_hashes: Mapping[str, str]
    results_hash: str
    results: pd.DataFrame
    schema_version: str = "campaign-significance-report/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "campaign-significance-report/v1":
            raise ValueError("unsupported campaign significance report schema")
        for name in (
            "protocol_hash",
            "significance_spec_hash",
            "frozen_family_hash",
            "calendar_hash",
            "results_hash",
        ):
            require_sha256(
                str(getattr(self, name)),
                name=f"campaign significance {name}",
            )
        values = pd.DataFrame(self.results).copy(deep=True)
        if hash_frame(values) != self.results_hash:
            raise ValueError("campaign significance results hash differs")
        required = (
            "status",
            "observation_count",
            "directed_mean",
            "standard_error",
            "test_statistic",
            "primary_p_value",
            "bh_q_value",
        )
        if (
            tuple(values.columns) != required
            or values.index.name != "candidate_id"
            or not values.index.is_unique
            or not values.index.is_monotonic_increasing
        ):
            raise ValueError("campaign significance result schema differs")
        input_hashes = {
            str(candidate_id): require_sha256(
                str(digest),
                name=f"daily IC input hash:{candidate_id}",
            )
            for candidate_id, digest in self.input_series_hashes.items()
        }
        if set(input_hashes) != set(values.index.astype(str)):
            raise ValueError("campaign significance input hashes differ from family")
        allowed_statuses = {
            "completed",
            "missing",
            "failed",
            "insufficient_observations",
            "degenerate",
        }
        if not set(values["status"].astype(str)).issubset(allowed_statuses):
            raise ValueError("campaign significance status is invalid")
        for column in ("primary_p_value", "bh_q_value"):
            probabilities = pd.to_numeric(values[column], errors="raise")
            if (
                probabilities.isna().any()
                or not probabilities.between(0.0, 1.0, inclusive="both").all()
            ):
                raise ValueError(
                    f"campaign significance {column} must be finite probabilities"
                )
        incomplete = values["status"].astype(str).ne("completed")
        if (
            values.loc[incomplete, "primary_p_value"].astype(float).ne(1.0).any()
            or values.loc[incomplete, "bh_q_value"].astype(float).ne(1.0).any()
        ):
            raise ValueError("incomplete significance tests must fail closed")
        object.__setattr__(self, "results", values)
        object.__setattr__(
            self,
            "input_series_hashes",
            MappingProxyType(dict(sorted(input_hashes.items()))),
        )

    @property
    def content_hash(self) -> str:
        self.verify_content()
        return cast(
            str,
            hash_json(
                {
                    "schema_version": self.schema_version,
                    "protocol_hash": self.protocol_hash,
                    "significance_spec_hash": self.significance_spec_hash,
                    "frozen_family_hash": self.frozen_family_hash,
                    "calendar_hash": self.calendar_hash,
                    "input_series_hashes": dict(self.input_series_hashes),
                    "results_hash": self.results_hash,
                }
            ),
        )

    def verify_content(self) -> None:
        if hash_frame(self.results) != self.results_hash:
            raise ValueError("campaign significance results changed")


def evaluate_campaign_significance(
    spec: SignificanceTestSpec,
    *,
    protocol: ResearchCampaignProtocol,
    candidate_family: FrozenSignificanceFamily,
    calendar: FrozenDailyICCalendar,
    raw_daily_ic: Mapping[str, pd.Series | None],
) -> CampaignSignificanceReport:
    """Test the protocol-bound complete family without caller-directed signs."""

    if type(protocol) is not ResearchCampaignProtocol:
        raise TypeError("significance requires an exact ResearchCampaignProtocol")
    if type(candidate_family) is not FrozenSignificanceFamily:
        raise TypeError("significance requires an exact frozen candidate family")
    if type(calendar) is not FrozenDailyICCalendar:
        raise TypeError("significance requires an exact frozen daily IC calendar")
    bindings = (
        (
            protocol.candidate_family_hash,
            candidate_family.content_hash,
            "candidate family",
        ),
        (
            protocol.statistical_test_spec_hash,
            spec.content_hash,
            "statistical test spec",
        ),
        (
            protocol.evaluation_calendar_hash,
            calendar.content_hash,
            "evaluation calendar",
        ),
    )
    for expected, observed, label in bindings:
        if not hmac.compare_digest(expected, observed):
            raise ValueError(f"campaign protocol {label} binding differs")
    if len(candidate_family.candidates) > protocol.locked_test_cap:
        raise ValueError("frozen significance family exceeds campaign locked test cap")
    method = spec.method
    if not isinstance(method, SignificanceMethod):  # pragma: no cover
        raise RuntimeError("significance method was not normalized")
    if protocol.p_value_method != method.value:
        raise ValueError("campaign protocol p-value method differs")
    if method is SignificanceMethod.BLOCK_BOOTSTRAP:
        minimum_trials = max(2_000, 20 * len(candidate_family.candidates))
        if spec.bootstrap_trials is None or spec.bootstrap_trials < minimum_trials:
            raise ValueError(
                "block bootstrap trials are insufficient for the frozen family"
            )

    candidate_ids = candidate_family.candidate_ids
    candidate_map = {
        candidate.candidate_id: candidate for candidate in candidate_family.candidates
    }
    unknown = sorted(set(raw_daily_ic).difference(candidate_ids))
    if unknown:
        raise ValueError(
            "significance input contains candidates outside frozen family:"
            + ",".join(unknown)
        )

    rows: list[dict[str, object]] = []
    input_hashes: dict[str, str] = {}
    expected_index = calendar.index
    for candidate_id in candidate_ids:
        candidate = candidate_map[candidate_id]
        if candidate_id not in raw_daily_ic:
            rows.append(_failed_row(candidate_id, "missing"))
            input_hashes[candidate_id] = hash_json(
                {"candidate_id": candidate_id, "status": "missing"}
            )
            continue
        raw = raw_daily_ic[candidate_id]
        if raw is None:
            rows.append(_failed_row(candidate_id, "failed"))
            input_hashes[candidate_id] = hash_json(
                {"candidate_id": candidate_id, "status": "failed"}
            )
            continue
        raw_values = _daily_series(
            raw,
            candidate_id=candidate_id,
            expected_index=expected_index,
        )
        input_hashes[candidate_id] = hash_frame(
            pd.DataFrame(
                {"raw_daily_ic": raw_values.to_numpy(dtype=float)},
                index=raw_values.index,
            )
        )
        values = raw_values * candidate.declared_sign
        if len(values) < spec.minimum_observations:
            rows.append(
                _failed_row(
                    candidate_id,
                    "insufficient_observations",
                    observation_count=len(values),
                    directed_mean=(float(values.mean()) if len(values) else math.nan),
                )
            )
            continue
        if spec.method is SignificanceMethod.HAC:
            standard_error, statistic, p_value = _hac_test(values, spec.hac_lags)
        else:
            standard_error, statistic, p_value = _block_bootstrap_test(
                values,
                trials=spec.bootstrap_trials,
                block_length=spec.block_length,
                random_seed=spec.random_seed,
                candidate_id=candidate_id,
            )
        if not all(
            math.isfinite(value) for value in (standard_error, statistic, p_value)
        ):
            rows.append(
                _failed_row(
                    candidate_id,
                    "degenerate",
                    observation_count=len(values),
                    directed_mean=float(values.mean()),
                )
            )
            continue
        rows.append(
            {
                "candidate_id": candidate_id,
                "status": "completed",
                "observation_count": len(values),
                "directed_mean": float(values.mean()),
                "standard_error": standard_error,
                "test_statistic": statistic,
                "primary_p_value": p_value,
            }
        )

    p_values = tuple(_number(row["primary_p_value"], "primary_p_value") for row in rows)
    q_values = benjamini_hochberg_complete_family(p_values)
    for row, q_value in zip(rows, q_values, strict=True):
        row["bh_q_value"] = q_value
    results = (
        pd.DataFrame(rows)
        .set_index("candidate_id")
        .sort_index(kind="stable")
        .loc[
            :,
            (
                "status",
                "observation_count",
                "directed_mean",
                "standard_error",
                "test_statistic",
                "primary_p_value",
                "bh_q_value",
            ),
        ]
    )
    return CampaignSignificanceReport(
        protocol_hash=protocol.content_hash,
        significance_spec_hash=spec.content_hash,
        frozen_family_hash=candidate_family.content_hash,
        calendar_hash=calendar.content_hash,
        input_series_hashes=input_hashes,
        results_hash=hash_frame(results),
        results=results,
    )


def benjamini_hochberg_complete_family(
    p_values: Sequence[float],
) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in p_values)
    if not normalized or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in normalized
    ):
        raise ValueError("BH requires a non-empty family of finite probabilities")
    names = tuple(f"hypothesis_{offset:08d}" for offset in range(len(normalized)))
    adjusted = benjamini_hochberg(
        dict(zip(names, normalized, strict=True)),
        alpha=0.05,
    )
    return tuple(float(adjusted[name]["qvalue"]) for name in names)


def _daily_series(
    raw: pd.Series,
    *,
    candidate_id: str,
    expected_index: pd.DatetimeIndex,
) -> pd.Series:
    values = pd.Series(raw).copy(deep=True)
    if not isinstance(values.index, pd.DatetimeIndex):
        raise TypeError(f"daily IC index must be DatetimeIndex:{candidate_id}")
    if values.index.tz is None:
        raise ValueError(f"daily IC index must be timezone-aware:{candidate_id}")
    if not values.index.is_unique or not values.index.is_monotonic_increasing:
        raise ValueError(f"daily IC index must be unique and sorted:{candidate_id}")
    if not values.index.equals(expected_index):
        raise ValueError(
            f"daily IC index must exactly match frozen calendar:{candidate_id}"
        )
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"daily IC values must be finite:{candidate_id}")
    return numeric


def _hac_test(
    values: pd.Series,
    lags: int | None,
) -> tuple[float, float, float]:
    if lags is None:  # pragma: no cover - spec invariant
        raise RuntimeError("HAC lags are unavailable")
    array = values.to_numpy(dtype=float)
    count = len(array)
    residual = array - float(array.mean())
    long_run_variance = float(np.dot(residual, residual) / count)
    for lag in range(1, min(lags, count - 1) + 1):
        covariance = float(np.dot(residual[lag:], residual[:-lag]) / count)
        weight = 1.0 - lag / (lags + 1.0)
        long_run_variance += 2.0 * weight * covariance
    if long_run_variance <= 0.0:
        return math.nan, math.nan, math.nan
    standard_error = math.sqrt(long_run_variance / count)
    statistic = float(array.mean()) / standard_error
    return standard_error, statistic, float(norm.sf(statistic))


def _block_bootstrap_test(
    values: pd.Series,
    *,
    trials: int | None,
    block_length: int | None,
    random_seed: int,
    candidate_id: str,
) -> tuple[float, float, float]:
    if trials is None or block_length is None:  # pragma: no cover - spec invariant
        raise RuntimeError("block bootstrap parameters are unavailable")
    array = values.to_numpy(dtype=float)
    count = len(array)
    centered = array - float(array.mean())
    blocks = math.ceil(count / block_length)
    seed = int(
        hash_json({"random_seed": random_seed, "candidate_id": candidate_id})[:16],
        16,
    )
    rng = np.random.default_rng(seed)
    bootstrap_means: npt.NDArray[np.float64] = np.empty(
        trials,
        dtype=float,
    )
    offsets = np.arange(block_length)
    for trial in range(trials):
        starts = rng.integers(0, count, size=blocks)
        indices = ((starts[:, None] + offsets[None, :]) % count).reshape(-1)
        bootstrap_means[trial] = float(centered[indices[:count]].mean())
    standard_error = float(bootstrap_means.std(ddof=1))
    if standard_error <= 0.0:
        return math.nan, math.nan, math.nan
    observed = float(array.mean())
    statistic = observed / standard_error
    p_value = float((1 + np.count_nonzero(bootstrap_means >= observed)) / (trials + 1))
    return standard_error, statistic, p_value


def _failed_row(
    candidate_id: str,
    status: str,
    *,
    observation_count: int = 0,
    directed_mean: float = math.nan,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "status": status,
        "observation_count": observation_count,
        "directed_mean": directed_mean,
        "standard_error": math.nan,
        "test_statistic": math.nan,
        "primary_p_value": 1.0,
    }


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _optional_integer(value: object, name: str) -> int | None:
    return None if value is None else _integer(value, name)


def _number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    return float(value)


__all__ = [
    "CampaignSignificanceReport",
    "FrozenDailyICCalendar",
    "FrozenSignificanceCandidate",
    "FrozenSignificanceFamily",
    "SignificanceMethod",
    "SignificanceTestSpec",
    "benjamini_hochberg_complete_family",
    "evaluate_campaign_significance",
]
