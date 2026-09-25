"""Core data types shared by collectors, rules, and the engine."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Severity(str, Enum):
    """Ordered severity for a Finding."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"info": 0, "warning": 1, "critical": 2}[self.value]


@dataclass
class StepSample:
    """One training-step (or fixed-interval) input-path telemetry snapshot.

    Only ``t`` and ``step`` are required. Distributed identity, storage,
    DataLoader, and checkpoint fields are optional so the same model works for
    single-process jobs, DDP/FSDP jobs, custom loaders, and remote/object-store
    pipelines.
    """

    t: float
    step: int

    # --- Distributed identity ---
    rank: Optional[int] = None
    local_rank: Optional[int] = None
    world_size: Optional[int] = None
    node_id: Optional[str] = None

    # --- Training-loop timing ---
    step_time_s: Optional[float] = None
    data_wait_s: Optional[float] = None
    compute_time_s: Optional[float] = None

    # --- DataLoader worker-pool state ---
    num_workers: Optional[int] = None
    prefetch_factor: Optional[int] = None
    queue_depth: Optional[int] = None
    queue_capacity: Optional[int] = None
    worker_cpu_pct: Optional[float] = None
    worker_restarts: Optional[int] = None

    # --- Disk I/O ---
    disk_name: Optional[str] = None
    disk_read_bytes_per_s: Optional[float] = None
    disk_read_iops: Optional[float] = None
    disk_util_pct: Optional[float] = None
    disk_avg_await_ms: Optional[float] = None

    # --- Storage identity ---
    is_network_fs: Optional[bool] = None
    storage_backend: Optional[str] = None
    filesystem_type: Optional[str] = None
    avg_read_size_bytes: Optional[float] = None

    # --- Generic remote/object-storage telemetry ---
    remote_read_latency_ms: Optional[float] = None
    remote_read_ops_per_s: Optional[float] = None
    remote_read_bytes_per_s: Optional[float] = None
    remote_retries: Optional[int] = None
    remote_errors: Optional[int] = None

    # --- Linux pressure stall information ---
    psi_io_some_avg10: Optional[float] = None
    psi_io_full_avg10: Optional[float] = None

    # --- Shuffle buffer ---
    shuffle_buffer_size: Optional[int] = None
    shuffle_buffer_capacity: Optional[int] = None
    shuffle_refill_event: Optional[bool] = None

    # --- Checkpoint I/O ---
    checkpoint_write_s: Optional[float] = None
    checkpoint_blocking: Optional[bool] = None

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class WindowSummary:
    """Aggregated view over a list of StepSamples."""

    samples: list[StepSample]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("WindowSummary requires at least one sample")
        self.samples = sorted(self.samples, key=lambda s: s.t)

    @property
    def start_t(self) -> float:
        return self.samples[0].t

    @property
    def end_t(self) -> float:
        return self.samples[-1].t

    @property
    def duration_s(self) -> float:
        return max(self.end_t - self.start_t, 1e-9)

    def values(self, field_name: str) -> list[float]:
        out = []
        for sample in self.samples:
            value = getattr(sample, field_name, None)
            if value is not None:
                out.append(float(value))
        return out

    def mean(self, field_name: str) -> Optional[float]:
        vals = self.values(field_name)
        return statistics.fmean(vals) if vals else None

    def max(self, field_name: str) -> Optional[float]:
        vals = self.values(field_name)
        return max(vals) if vals else None

    def fraction_true(self, field_name: str) -> Optional[float]:
        vals = [
            getattr(sample, field_name)
            for sample in self.samples
            if getattr(sample, field_name) is not None
        ]
        return (sum(1 for value in vals if value) / len(vals)) if vals else None

    def count_true(self, field_name: str) -> int:
        return sum(1 for sample in self.samples if getattr(sample, field_name, None))

    def last(self, field_name: str) -> Optional[Any]:
        for sample in reversed(self.samples):
            value = getattr(sample, field_name, None)
            if value is not None:
                return value
        return None


@dataclass
class Finding:
    """A single diagnosis emitted by a rule."""

    rule_id: str
    severity: Severity
    title: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    suggested_fix: str = ""
    window_start: Optional[float] = None
    window_end: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "evidence": self.evidence,
            "suggested_fix": self.suggested_fix,
            "window_start": self.window_start,
            "window_end": self.window_end,
        }


def now() -> float:
    return time.time()
