"""Consumer-side instrumentation for DataLoader-like iterables."""

from __future__ import annotations

import os
import socket
import time
from typing import Any, Iterable, Iterator, Optional

from datapath_doctor.models import StepSample


def _env_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


class WorkerProcessMonitor:
    """Best-effort CPU and restart tracking for DataLoader workers."""

    def __init__(self, expected_workers: Optional[int] = None) -> None:
        self.expected_workers = expected_workers
        self._known_pids: set[int] = set()
        self._restarts = 0
        self._psutil = None
        try:
            import psutil

            self._psutil = psutil
        except ImportError:
            pass

    def sample(self) -> tuple[Optional[float], int]:
        if self._psutil is None:
            return None, self._restarts

        try:
            me = self._psutil.Process(os.getpid())
            children = me.children(recursive=False)
        except Exception:  # noqa: BLE001
            return None, self._restarts

        current_pids = {child.pid for child in children}
        if self._known_pids and self.expected_workers:
            missing = self._known_pids - current_pids
            new = current_pids - self._known_pids
            if missing and new:
                self._restarts += min(len(missing), len(new))
        self._known_pids = current_pids

        cpu_vals = []
        for child in children:
            try:
                cpu_vals.append(child.cpu_percent(interval=None))
            except Exception:  # noqa: BLE001
                continue

        mean_cpu = (sum(cpu_vals) / len(cpu_vals)) if cpu_vals else None
        return mean_cpu, self._restarts


class DataLoaderProfiler:
    """Wrap a DataLoader-like iterable and record per-step input wait time.

    Rank metadata is optional. If omitted, common torchrun/SLURM-style
    environment variables are used when available.
    """

    def __init__(
        self,
        loader: Iterable[Any],
        num_workers: Optional[int] = None,
        prefetch_factor: Optional[int] = None,
        queue_capacity: Optional[int] = None,
        monitor_workers: bool = True,
        rank: Optional[int] = None,
        local_rank: Optional[int] = None,
        world_size: Optional[int] = None,
        node_id: Optional[str] = None,
    ) -> None:
        self.loader = loader
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.queue_capacity = queue_capacity or (
            num_workers * prefetch_factor if num_workers and prefetch_factor else None
        )

        self.rank = rank if rank is not None else (_env_int("RANK") or _env_int("SLURM_PROCID"))
        self.local_rank = (
            local_rank
            if local_rank is not None
            else (_env_int("LOCAL_RANK") or _env_int("SLURM_LOCALID"))
        )
        self.world_size = (
            world_size
            if world_size is not None
            else (_env_int("WORLD_SIZE") or _env_int("SLURM_NTASKS"))
        )
        self.node_id = node_id or os.getenv("DATAPATH_NODE_ID") or socket.gethostname()

        self.samples: list[StepSample] = []
        self._step = 0
        self._iter: Optional[Iterator[Any]] = None
        self._pending_wait: Optional[float] = None
        self._pending_t0: Optional[float] = None
        self._monitor = WorkerProcessMonitor(expected_workers=num_workers) if monitor_workers else None

    def __iter__(self) -> "DataLoaderProfiler":
        self._iter = iter(self.loader)
        return self

    def __next__(self) -> Any:
        assert self._iter is not None, "call iter(profiler) before next()"
        t0 = time.perf_counter()
        batch = next(self._iter)
        t1 = time.perf_counter()
        self._pending_wait = t1 - t0
        self._pending_t0 = t1
        return batch

    def record_compute_done(self, queue_depth: Optional[int] = None) -> StepSample:
        if self._pending_wait is None or self._pending_t0 is None:
            raise RuntimeError("record_compute_done() called before next(profiler)")

        t2 = time.perf_counter()
        compute_time_s = t2 - self._pending_t0
        data_wait_s = self._pending_wait

        worker_cpu_pct, worker_restarts = (None, None)
        if self._monitor is not None:
            worker_cpu_pct, worker_restarts = self._monitor.sample()

        sample = StepSample(
            t=time.time(),
            step=self._step,
            rank=self.rank,
            local_rank=self.local_rank,
            world_size=self.world_size,
            node_id=self.node_id,
            step_time_s=data_wait_s + compute_time_s,
            data_wait_s=data_wait_s,
            compute_time_s=compute_time_s,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            queue_capacity=self.queue_capacity,
            queue_depth=queue_depth,
            worker_cpu_pct=worker_cpu_pct,
            worker_restarts=worker_restarts,
        )
        self.samples.append(sample)
        self._step += 1
        self._pending_wait = None
        self._pending_t0 = None
        return sample
