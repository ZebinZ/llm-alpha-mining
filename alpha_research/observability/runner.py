from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Mapping

import psutil

from alpha_research.core.hashing import hash_json
from alpha_research.observability.performance import BenchmarkResult, BenchmarkSpec
from alpha_research.observability.telemetry import StageTelemetry


@dataclass(frozen=True, slots=True)
class WorkloadMeasurement:
    input_rows: int
    output_rows: int
    symbols: int
    disk_read_bytes: int = 0
    disk_write_bytes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    worker_count: int = 1
    attributes: Mapping[str, str | int | float | bool | None] | None = None

    def __post_init__(self) -> None:
        for name in (
            "input_rows",
            "output_rows",
            "symbols",
            "disk_read_bytes",
            "disk_write_bytes",
            "cache_hits",
            "cache_misses",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"workload measurement {name} must be non-negative")
        if not isinstance(self.worker_count, int) or self.worker_count <= 0:
            raise ValueError("workload measurement worker_count must be positive")


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    spec: BenchmarkSpec
    samples: tuple[StageTelemetry, ...]
    result: BenchmarkResult

    @property
    def content_hash(self) -> str:
        return hash_json(
            {
                "spec_hash": self.spec.content_hash,
                "sample_hashes": [item.content_hash for item in self.samples],
                "result_hash": self.result.content_hash,
            }
        )


def run_benchmark_workload(
    spec: BenchmarkSpec,
    workload: Callable[[int], WorkloadMeasurement],
    *,
    repeats: int | None = None,
    sampling_interval_seconds: float = 0.005,
) -> BenchmarkRun:
    count = spec.minimum_repeats if repeats is None else repeats
    if not isinstance(count, int) or count < spec.minimum_repeats:
        raise ValueError("benchmark repeats are below the preregistered minimum")
    if not 0 < sampling_interval_seconds <= 1:
        raise ValueError("benchmark sampling interval must lie in (0,1]")
    experiment_hash = hash_json(
        {"kind": "performance_benchmark", "spec_hash": spec.content_hash}
    )
    samples: list[StageTelemetry] = []
    for repeat in range(count):
        started = datetime.now(timezone.utc)
        started_ns = time.time_ns()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        sampler = _MemorySampler(sampling_interval_seconds)
        sampler.start()
        measurement: WorkloadMeasurement | None = None
        failure_code: str | None = None
        try:
            measurement = workload(repeat)
            if not isinstance(measurement, WorkloadMeasurement):
                raise TypeError("workload returned a non-measurement")
        except Exception:
            failure_code = "benchmark_workload_failed"
        finally:
            peak_memory = sampler.stop()
        finished = datetime.now(timezone.utc)
        wall_seconds = time.perf_counter() - wall_start
        cpu_seconds = time.process_time() - cpu_start
        observed = measurement or WorkloadMeasurement(0, 0, 0)
        samples.append(
            StageTelemetry(
                experiment_spec_hash=experiment_hash,
                trace_id=f"benchmark-{spec.content_hash[:16]}",
                span_id=f"repeat-{repeat}-{started_ns}",
                parent_span_id=None,
                attempt_id=repeat + 1,
                stage="performance_benchmark",
                status="failed" if failure_code else "succeeded",
                started_at=started.isoformat(),
                finished_at=finished.isoformat(),
                input_rows=observed.input_rows,
                output_rows=observed.output_rows,
                symbols=observed.symbols,
                wall_seconds=wall_seconds,
                cpu_seconds=cpu_seconds,
                peak_memory_bytes=peak_memory,
                disk_read_bytes=observed.disk_read_bytes,
                disk_write_bytes=observed.disk_write_bytes,
                cache_hits=observed.cache_hits,
                cache_misses=observed.cache_misses,
                worker_count=observed.worker_count,
                llm_calls=0,
                llm_tokens=0,
                llm_cost_microusd=0,
                failure_code=failure_code,
                attributes={
                    "benchmark_id": spec.benchmark_id,
                    "cache_mode": spec.cache_mode,
                    "repeat": repeat,
                    **dict(observed.attributes or {}),
                },
            )
        )
    sample_tuple = tuple(samples)
    result = BenchmarkResult.from_telemetry(spec, sample_tuple)
    return BenchmarkRun(spec=spec, samples=sample_tuple, result=result)


class _MemorySampler:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.process = psutil.Process()
        self.peak = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def start(self) -> None:
        self._observe()
        self.thread.start()

    def stop(self) -> int:
        self.stop_event.set()
        self.thread.join(timeout=max(1.0, self.interval * 4))
        self._observe()
        return self.peak

    def _sample(self) -> None:
        while not self.stop_event.wait(self.interval):
            self._observe()

    def _observe(self) -> None:
        try:
            resident = self.process.memory_info().rss
            resident += sum(
                child.memory_info().rss
                for child in self.process.children(recursive=True)
                if child.is_running()
            )
            self.peak = max(self.peak, resident)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


__all__ = ["BenchmarkRun", "WorkloadMeasurement", "run_benchmark_workload"]
