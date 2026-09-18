"""Instrumentation for the consumer side of a PyTorch (or PyTorch-shaped)
DataLoader: how long the training loop blocks waiting on the next batch,
how deep the prefetch queue is, and whether worker processes are pegged or
restarting.

This deliberately does not import torch at module scope — it only needs
``next()`` on whatever iterable it's handed, so it works against a real
``torch.utils.data.DataLoader``, a custom loader, or a synthetic iterator in
tests.
"""

from __future__ import annotations

import os
import time
from typing import Any, Iterable, Iterator, Optional

from datapath_doctor.models import StepSample


class WorkerProcessMonitor:
    """Best-effort CPU% and restart tracking for DataLoader worker
    processes, using psutil to inspect the current process's children.

    PyTorch DataLoader workers are child processes of the process that
    created the DataLoader when num_workers > 0, so this only works when
    called from that same process (the common case: the training script).
    """

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
        """Returns (mean_worker_cpu_pct, cumulative_restarts_observed)."""
        if self._psutil is None:
            return None, self._restarts

        try:
            me = self._psutil.Process(os.getpid())
            children = me.children(recursive=False)
        except Exception:  # noqa: BLE001
            return None, self._restarts

        current_pids = {c.pid for c in children}
        if self._known_pids and self.expected_workers:
            missing = self._known_pids - current_pids
            new = current_pids - self._known_pids
            # A restart looks like: a pid we tracked disappeared and a new
            # one showed up in its place, while we're still at/under the
            # expected worker count.
            if missing and new:
                self._restarts += min(len(missing), len(new))
        self._known_pids = current_pids

        cpu_vals = []
        for c in children:
            try:
                cpu_vals.append(c.cpu_percent(interval=None))
            except Exception:  # noqa: BLE001
                continue

        mean_cpu = (sum(cpu_vals) / len(cpu_vals)) if cpu_vals else None
        return mean_cpu, self._restarts


class DataLoaderProfiler:
    """Wraps a DataLoader-like iterable and records per-step timing.

    Usage::

        profiler = DataLoaderProfiler(loader, num_workers=8, prefetch_factor=4)
        for batch in profiler:
            outputs = model(batch)
            loss.backward()
            optimizer.step()
            profiler.record_compute_done()

        samples = profiler.samples  # list[StepSample], feed to RuleEngine.run()
    """

    def __init__(
        self,
        loader: Iterable[Any],
        num_workers: Optional[int] = None,
        prefetch_factor: Optional[int] = None,
        queue_capacity: Optional[int] = None,
        monitor_workers: bool = True,
    ) -> None:
        self.loader = loader
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.queue_capacity = queue_capacity or (
            num_workers * prefetch_factor if num_workers and prefetch_factor else None
        )
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
        batch = next(self._iter)  # propagates StopIteration at epoch end
        t1 = time.perf_counter()
        self._pending_wait = t1 - t0
        self._pending_t0 = t1
        return batch

    def record_compute_done(self, queue_depth: Optional[int] = None) -> StepSample:
        """Call once per step, right after the compute (forward/backward/
        optimizer) portion finishes, to close out timing for that step."""
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
