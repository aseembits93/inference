import time
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

STATISTICS_FORMAT = """
avg: {average_inference_latency_ms}ms\t| exec_avg: {average_execution_time_per_image_ms}ms\t| rps: {requests_per_second}\t| p75: {p75_inference_latency_ms}ms\t| p90: {p90_inference_latency_ms}\t| %err: {error_rate}\t| {error_status_codes}
""".strip()


@dataclass(frozen=True)
class InferenceStatistics:
    inferences_made: int
    images_processed: int
    average_inference_latency_ms: float
    std_inference_latency_ms: float
    average_inference_latency_per_image_ms: float
    average_execution_time_per_image_ms: Optional[float]
    p50_inference_latency_ms: float
    p75_inference_latency_ms: float
    p90_inference_latency_ms: float
    p95_inference_latency_ms: float
    p99_inference_latency_ms: float
    requests_per_second: float
    images_per_second: float
    error_rate: float
    error_status_codes: Dict[str, int]
    avg_remote_execution_time: Optional[float]

    def to_string(self) -> str:
        return STATISTICS_FORMAT.format(
            average_inference_latency_ms=self.average_inference_latency_ms,
            average_execution_time_per_image_ms=self.average_execution_time_per_image_ms
            or "N/A",
            requests_per_second=self.requests_per_second,
            p50_inference_latency_ms=self.p50_inference_latency_ms,
            p75_inference_latency_ms=self.p75_inference_latency_ms,
            p90_inference_latency_ms=self.p90_inference_latency_ms,
            error_rate=self.error_rate,
            error_status_codes=self.error_status_codes,
            avg_remote_execution_time=self.avg_remote_execution_time or "N/A",
        )


class ResultsCollector:

    def __init__(self, preallocate_size: int = 10000):
        self._benchmark_start: Optional[float] = None
        self._preallocate_size = preallocate_size
        self._timestamps = np.zeros(preallocate_size, dtype=np.float64)
        self._batch_sizes = np.zeros(preallocate_size, dtype=np.int32)
        self._durations = np.zeros(preallocate_size, dtype=np.float32)
        self._execution_times = np.full(preallocate_size, np.nan, dtype=np.float32)
        self._remote_execution_times = np.full(preallocate_size, np.nan, dtype=np.float32)
        self._current_index = 0
        self._benchmark_end: Optional[float] = None
        self._errors: List[Tuple[float, int, str]] = []

    def start_benchmark(self) -> None:
        if self._benchmark_start is None:
            self._benchmark_start = time.perf_counter()

    def register_inference_duration(
        self,
        batch_size: int,
        duration: float,
        execution_time: Optional[float] = None,
        remote_execution_time: Optional[float] = None,
    ) -> None:
        if self._current_index >= self._preallocate_size:
            new_size = self._preallocate_size * 2
            self._timestamps = np.resize(self._timestamps, new_size)
            self._batch_sizes = np.resize(self._batch_sizes, new_size)
            self._durations = np.resize(self._durations, new_size)
            self._execution_times = np.resize(self._execution_times, new_size)
            self._remote_execution_times = np.resize(self._remote_execution_times, new_size)
            self._execution_times[self._preallocate_size:] = np.nan
            self._remote_execution_times[self._preallocate_size:] = np.nan
            self._preallocate_size = new_size

        idx = self._current_index
        self._timestamps[idx] = time.perf_counter()
        self._batch_sizes[idx] = batch_size
        self._durations[idx] = duration
        if execution_time is not None:
            self._execution_times[idx] = execution_time
        if remote_execution_time is not None:
            self._remote_execution_times[idx] = remote_execution_time
        self._current_index += 1

    def register_error(self, batch_size: int, status_code: str) -> None:
        self._errors.append((time.perf_counter(), batch_size, status_code))

    def stop_benchmark(self) -> None:
        if self._benchmark_end is None:
            self._benchmark_end = time.perf_counter()

    def has_benchmark_finished(self) -> bool:
        return self._benchmark_end is not None

    def get_statistics(
        self, window: Optional[int] = None
    ) -> Optional[InferenceStatistics]:
        if self._benchmark_start is None or self._current_index < 1:
            return None
        end_time = (
            self._benchmark_end if self._benchmark_end is not None else time.perf_counter()
        )

        n = self._current_index
        if window is not None:
            start_idx = max(0, n - window)
        else:
            start_idx = 0

        timestamps = self._timestamps[start_idx:n]
        batch_sizes = self._batch_sizes[start_idx:n]
        durations = self._durations[start_idx:n]
        execution_times = self._execution_times[start_idx:n]
        remote_execution_times = self._remote_execution_times[start_idx:n]

        inferences_made = n - start_idx
        images_processed = int(np.sum(batch_sizes))

        average_inference_latency_ms = round(float(np.mean(durations)) * 1000, 1)

        valid_exec_times = execution_times[~np.isnan(execution_times)]
        if len(valid_exec_times) > 0:
            average_execution_time_ms = round(float(np.mean(valid_exec_times)) * 1000, 1)
            average_execution_time_per_image_ms = round(
                average_execution_time_ms * inferences_made / images_processed, 2
            )
        else:
            average_execution_time_ms = None
            average_execution_time_per_image_ms = None

        valid_remote_times = remote_execution_times[~np.isnan(remote_execution_times)]
        if len(valid_remote_times) > 0:
            avg_remote_execution_time = float(np.mean(valid_remote_times))
        else:
            avg_remote_execution_time = None

        std_inference_latency_ms = round(float(np.std(durations)) * 1000, 1)
        average_inference_latency_per_image_ms = round(
            average_inference_latency_ms * inferences_made / images_processed, 2
        )
        p50_inference_latency_ms = round(float(np.percentile(durations, 50)) * 1000, 1)
        p75_inference_latency_ms = round(float(np.percentile(durations, 75)) * 1000, 1)
        p90_inference_latency_ms = round(float(np.percentile(durations, 90)) * 1000, 1)
        p95_inference_latency_ms = round(float(np.percentile(durations, 95)) * 1000, 1)
        p99_inference_latency_ms = round(float(np.percentile(durations, 99)) * 1000, 1)

        start_time = (
            self._benchmark_start
            if window is None or inferences_made < window
            else timestamps[0]
        )

        errors = copy(self._errors)
        error_status_codes = defaultdict(int)
        errors_number = 0
        for e in errors:
            if e[0] < start_time:
                continue
            error_status_codes[e[2]] += 1
            errors_number += 1

        error_rate = round(errors_number / inferences_made * 100, 2) if inferences_made > 0 else 0.0
        duration = end_time - start_time
        requests_per_second = round(inferences_made / duration, 1) if duration > 0 else 0.0
        images_per_second = round(images_processed / duration, 1) if duration > 0 else 0.0
        return InferenceStatistics(
            inferences_made=inferences_made,
            images_processed=images_processed,
            average_inference_latency_ms=average_inference_latency_ms,
            std_inference_latency_ms=std_inference_latency_ms,
            average_inference_latency_per_image_ms=average_inference_latency_per_image_ms,
            average_execution_time_per_image_ms=average_execution_time_per_image_ms,
            p50_inference_latency_ms=p50_inference_latency_ms,
            p75_inference_latency_ms=p75_inference_latency_ms,
            p90_inference_latency_ms=p90_inference_latency_ms,
            p95_inference_latency_ms=p95_inference_latency_ms,
            p99_inference_latency_ms=p99_inference_latency_ms,
            requests_per_second=requests_per_second,
            images_per_second=images_per_second,
            error_rate=error_rate,
            error_status_codes=", ".join(
                f"{exc}: {count}" for exc, count in error_status_codes.items()
            ),
            avg_remote_execution_time=avg_remote_execution_time,
        )
