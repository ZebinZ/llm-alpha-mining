from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import math
import numpy as np
from numpy.typing import NDArray

from alpha_research.core.hashing import hash_json, require_sha256
from alpha_research.observability.telemetry import StageTelemetry


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    benchmark_id: str
    version: str
    workload_hash: str
    data_snapshot_hash: str
    code_snapshot_hash: str
    environment_hash: str
    cache_mode: str
    minimum_repeats: int
    maximum_p95_wall_regression_fraction: float
    maximum_peak_memory_regression_fraction: float
    minimum_throughput_ratio: float
    maximum_failure_rate: float
    schema_version: str = "performance-benchmark-spec/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "performance-benchmark-spec/v1":
            raise ValueError("unsupported BenchmarkSpec schema")
        if not self.benchmark_id.strip() or not self.version.strip():
            raise ValueError("benchmark id/version is required")
        for name in (
            "workload_hash",
            "data_snapshot_hash",
            "code_snapshot_hash",
            "environment_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"benchmark {name}")
        if self.cache_mode not in {"cold", "warm"}:
            raise ValueError("benchmark cache_mode must be cold or warm")
        if not isinstance(self.minimum_repeats, int) or self.minimum_repeats < 3:
            raise ValueError("benchmark requires at least three repeats")
        fractions = (
            self.maximum_p95_wall_regression_fraction,
            self.maximum_peak_memory_regression_fraction,
            self.maximum_failure_rate,
        )
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in fractions):
            raise ValueError("benchmark regression/failure fractions must lie in [0,1]")
        if (
            not math.isfinite(self.minimum_throughput_ratio)
            or not 0 < self.minimum_throughput_ratio <= 1
        ):
            raise ValueError("benchmark minimum_throughput_ratio must lie in (0,1]")

    @property
    def comparison_key(self) -> str:
        return hash_json(
            {
                "benchmark_id": self.benchmark_id,
                "version": self.version,
                "workload_hash": self.workload_hash,
                "data_snapshot_hash": self.data_snapshot_hash,
                "environment_hash": self.environment_hash,
                "cache_mode": self.cache_mode,
            }
        )

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "benchmark_id": self.benchmark_id,
            "version": self.version,
            "workload_hash": self.workload_hash,
            "data_snapshot_hash": self.data_snapshot_hash,
            "code_snapshot_hash": self.code_snapshot_hash,
            "environment_hash": self.environment_hash,
            "cache_mode": self.cache_mode,
            "minimum_repeats": self.minimum_repeats,
            "maximum_p95_wall_regression_fraction": self.maximum_p95_wall_regression_fraction,
            "maximum_peak_memory_regression_fraction": self.maximum_peak_memory_regression_fraction,
            "minimum_throughput_ratio": self.minimum_throughput_ratio,
            "maximum_failure_rate": self.maximum_failure_rate,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    benchmark_spec_hash: str
    comparison_key: str
    code_snapshot_hash: str
    sample_hashes: tuple[str, ...]
    metrics: Mapping[str, float | int | None]

    def __post_init__(self) -> None:
        for name in ("benchmark_spec_hash", "comparison_key", "code_snapshot_hash"):
            require_sha256(str(getattr(self, name)), name=f"benchmark result {name}")
        if len(set(self.sample_hashes)) != len(self.sample_hashes):
            raise ValueError("benchmark sample hashes must be unique")
        for digest in self.sample_hashes:
            require_sha256(digest, name="benchmark sample hash")
        metrics = MappingProxyType(dict(sorted(self.metrics.items())))
        object.__setattr__(self, "metrics", metrics)

    @classmethod
    def from_telemetry(
        cls, spec: BenchmarkSpec, samples: Sequence[StageTelemetry]
    ) -> "BenchmarkResult":
        if len(samples) < spec.minimum_repeats:
            raise ValueError("benchmark has insufficient repeats")
        hashes = tuple(item.content_hash for item in samples)
        if len(set(hashes)) != len(hashes):
            raise ValueError("benchmark samples must be distinct observations")
        wall = np.asarray([item.wall_seconds for item in samples], dtype=float)
        cpu = np.asarray([item.cpu_seconds for item in samples], dtype=float)
        throughput = np.asarray(
            [
                np.nan if item.rows_per_second is None else item.rows_per_second
                for item in samples
            ],
            dtype=float,
        )
        cache = np.asarray(
            [
                np.nan if item.cache_hit_rate is None else item.cache_hit_rate
                for item in samples
            ],
            dtype=float,
        )
        metrics: dict[str, float | int | None] = {
            "sample_count": len(samples),
            "failure_count": sum(item.status == "failed" for item in samples),
            "failure_rate": sum(item.status == "failed" for item in samples)
            / len(samples),
            "wall_seconds_p50": float(np.quantile(wall, 0.50)),
            "wall_seconds_p95": float(np.quantile(wall, 0.95)),
            "cpu_seconds_p50": float(np.quantile(cpu, 0.50)),
            "peak_memory_bytes_max": max(item.peak_memory_bytes for item in samples),
            "disk_read_bytes_total": sum(item.disk_read_bytes for item in samples),
            "disk_write_bytes_total": sum(item.disk_write_bytes for item in samples),
            "rows_per_second_p50": _nan_quantile(throughput, 0.50),
            "rows_per_second_p95": _nan_quantile(throughput, 0.95),
            "cache_hit_rate_mean": _nan_mean(cache),
            "llm_tokens_total": sum(item.llm_tokens for item in samples),
            "llm_cost_microusd_total": sum(item.llm_cost_microusd for item in samples),
        }
        return cls(
            benchmark_spec_hash=spec.content_hash,
            comparison_key=spec.comparison_key,
            code_snapshot_hash=spec.code_snapshot_hash,
            sample_hashes=hashes,
            metrics=metrics,
        )

    @property
    def content_hash(self) -> str:
        return hash_json(
            {
                "benchmark_spec_hash": self.benchmark_spec_hash,
                "comparison_key": self.comparison_key,
                "code_snapshot_hash": self.code_snapshot_hash,
                "sample_hashes": list(self.sample_hashes),
                "metrics": dict(self.metrics),
            }
        )


@dataclass(frozen=True, slots=True)
class PerformanceGateReport:
    baseline_result_hash: str
    current_result_hash: str
    checks: Mapping[str, bool]
    observed_ratios: Mapping[str, float | None]
    passed: bool

    def __post_init__(self) -> None:
        require_sha256(self.baseline_result_hash, name="gate baseline hash")
        require_sha256(self.current_result_hash, name="gate current hash")
        checks = MappingProxyType(dict(sorted(self.checks.items())))
        ratios = MappingProxyType(dict(sorted(self.observed_ratios.items())))
        if self.passed != all(checks.values()):
            raise ValueError("performance gate passed flag differs from checks")
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "observed_ratios", ratios)

    @property
    def content_hash(self) -> str:
        return hash_json(
            {
                "baseline_result_hash": self.baseline_result_hash,
                "current_result_hash": self.current_result_hash,
                "checks": dict(self.checks),
                "observed_ratios": dict(self.observed_ratios),
                "passed": self.passed,
            }
        )


def compare_performance(
    spec: BenchmarkSpec,
    baseline: BenchmarkResult,
    current: BenchmarkResult,
) -> PerformanceGateReport:
    if (
        baseline.comparison_key != spec.comparison_key
        or current.comparison_key != spec.comparison_key
    ):
        raise ValueError("performance results are not comparable to BenchmarkSpec")
    base_wall = _number(baseline.metrics, "wall_seconds_p95")
    current_wall = _number(current.metrics, "wall_seconds_p95")
    base_memory = _number(baseline.metrics, "peak_memory_bytes_max")
    current_memory = _number(current.metrics, "peak_memory_bytes_max")
    base_throughput = _number(baseline.metrics, "rows_per_second_p50")
    current_throughput = _number(current.metrics, "rows_per_second_p50")
    failure_rate = _number(current.metrics, "failure_rate")
    wall_ratio = _safe_ratio(current_wall, base_wall)
    memory_ratio = _safe_ratio(current_memory, base_memory)
    throughput_ratio = _safe_ratio(current_throughput, base_throughput)
    checks = {
        "p95_wall": wall_ratio is not None
        and wall_ratio <= 1 + spec.maximum_p95_wall_regression_fraction,
        "peak_memory": memory_ratio is not None
        and memory_ratio <= 1 + spec.maximum_peak_memory_regression_fraction,
        "throughput": throughput_ratio is not None
        and throughput_ratio >= spec.minimum_throughput_ratio,
        "failure_rate": failure_rate is not None
        and failure_rate <= spec.maximum_failure_rate,
    }
    return PerformanceGateReport(
        baseline_result_hash=baseline.content_hash,
        current_result_hash=current.content_hash,
        checks=checks,
        observed_ratios={
            "p95_wall_ratio": wall_ratio,
            "peak_memory_ratio": memory_ratio,
            "throughput_ratio": throughput_ratio,
            "failure_rate": failure_rate,
        },
        passed=all(checks.values()),
    )


def _number(values: Mapping[str, float | int | None], name: str) -> float | None:
    value = values.get(name)
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _nan_quantile(values: NDArray[np.float64], quantile: float) -> float | None:
    finite = values[np.isfinite(values)]
    return None if not len(finite) else float(np.quantile(finite, quantile))


def _nan_mean(values: NDArray[np.float64]) -> float | None:
    finite = values[np.isfinite(values)]
    return None if not len(finite) else float(finite.mean())


__all__ = [
    "BenchmarkResult",
    "BenchmarkSpec",
    "PerformanceGateReport",
    "compare_performance",
]
