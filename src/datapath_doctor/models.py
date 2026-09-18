"""Core data types shared by collectors, rules, and the engine.

Design mirrors nccl-doctor: a small set of plain dataclasses that collectors
populate and rules consume, so any collector (real /proc telemetry, a
PyTorch DataLoader hook, or a synthetic generator for tests) can feed the
same rule set.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Severity(str, Enum):
    """Ordered severity for a Finding. String-valued so it serializes
    cleanly to SQLite / JSON without a custom encoder."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"info": 0, "warning": 1, "critical": 2}[self.value]


@dataclass
class StepSample:
    """One training-step (or fixed-interval) snapshot of input-path telemetry.

    All fields are optional except ``t`` and ``step`` because different
    deployments wire up different collectors — a job with no PyTorch
    DataLoader hook (e.g. a custom loader) can still report disk/PSI fields,
    and rules that need a field they don't see simply decline to fire.
    """

    t: float
    step: int

    # --- GPU-wait / step timing (from a DataLoader iterator wrapper) ---
    step_time_s: Optional[float] = None
    data_wait_s: Optional[float] = None  # time GPU/training loop spent blocked on next batch
    compute_time_s: Optional[float] = None

    # --- DataLoader worker pool state ---
    num_workers: Optional[int] = None
    prefetch_factor: Optional[int] = None
    queue_depth: Optional[int] = None  # batches currently buffered, ready to consume
    queue_capacity: Optional[int] = None
    worker_cpu_pct: Optional[float] = None  # mean CPU% across worker processes
    worker_restarts: Optional[int] = None  # cumulative worker process restarts observed so far

    # --- Disk I/O (per collection interval, from /proc/diskstats) ---
    disk_name: Optional[str] = None
    disk_read_bytes_per_s: Optional[float] = None
    disk_read_iops: Optional[float] = None
    disk_util_pct: Optional[float] = None  # % of interval the disk had I/O in flight
    disk_avg_await_ms: Optional[float] = None  # avg time per I/O request, queue + service

    # --- Filesystem hints ---
    is_network_fs: Optional[bool] = None
    avg_read_size_bytes: Optional[float] = None

    # --- PSI (pressure stall information), from /proc/pressure/io ---
    psi_io_some_avg10: Optional[float] = None
    psi_io_full_avg10: Optional[float] = None

    # --- Shuffle buffer (from a DataLoader hook that instruments the shuffle stage) ---
    shuffle_buffer_size: Optional[int] = None
    shuffle_buffer_capacity: Optional[int] = None
    shuffle_refill_event: Optional[bool] = None

    # --- Checkpoint I/O (from the training loop around save/load calls) ---
    checkpoint_write_s: Optional[float] = None
    checkpoint_blocking: Optional[bool] = None  # True if the write held up the next training step

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class WindowSummary:
    """Aggregated view over a list of StepSamples, computed once and handed
    to every rule so rules don't each redo the same statistics."""

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
        for s in self.samples:
            v = getattr(s, field_name, None)
            if v is not None:
                out.append(float(v))
        return out

    def mean(self, field_name: str) -> Optional[float]:
        vals = self.values(field_name)
        return statistics.fmean(vals) if vals else None

    def max(self, field_name: str) -> Optional[float]:
        vals = self.values(field_name)
        return max(vals) if vals else None

    def fraction_true(self, field_name: str) -> Optional[float]:
        """For boolean fields: fraction of samples where the field is True,
        among samples where it was set at all."""
        vals = [getattr(s, field_name) for s in self.samples if getattr(s, field_name) is not None]
        return (sum(1 for v in vals if v) / len(vals)) if vals else None

    def count_true(self, field_name: str) -> int:
        return sum(1 for s in self.samples if getattr(s, field_name, None))

    def last(self, field_name: str) -> Optional[Any]:
        for s in reversed(self.samples):
            v = getattr(s, field_name, None)
            if v is not None:
                return v
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
