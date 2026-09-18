"""Synthetic StepSample generators.

Useful for (a) unit-testing rules without a real GPU/disk/DataLoader, and
(b) the `datapath-doctor demo` CLI command that lets someone try the tool
before wiring it into a real job — same role as similar synthetic modes in
stallscope and nccl-doctor.
"""

from __future__ import annotations

import random
from typing import Optional

from datapath_doctor.models import StepSample


def healthy_pipeline(n: int = 200, seed: int = 0) -> list[StepSample]:
    """A well-tuned input pipeline: data_wait is a small fraction of
    step_time, queue stays comfortably full, no restarts."""
    rng = random.Random(seed)
    samples = []
    t = 0.0
    for i in range(n):
        compute = rng.gauss(0.40, 0.02)
        wait = max(0.0, rng.gauss(0.02, 0.005))
        t += compute + wait
        samples.append(
            StepSample(
                t=t,
                step=i,
                step_time_s=compute + wait,
                data_wait_s=wait,
                compute_time_s=compute,
                num_workers=8,
                prefetch_factor=4,
                queue_depth=rng.randint(20, 32),
                queue_capacity=32,
                worker_cpu_pct=rng.gauss(55, 5),
                worker_restarts=0,
                disk_name="nvme0n1",
                disk_read_bytes_per_s=rng.gauss(300e6, 20e6),
                disk_read_iops=rng.gauss(2000, 100),
                disk_util_pct=rng.gauss(30, 5),
                disk_avg_await_ms=rng.gauss(0.3, 0.05),
                avg_read_size_bytes=rng.gauss(150_000, 10_000),
                psi_io_some_avg10=rng.gauss(1.0, 0.3),
                psi_io_full_avg10=rng.gauss(0.1, 0.05),
                shuffle_buffer_size=rng.randint(9000, 10000),
                shuffle_buffer_capacity=10000,
                shuffle_refill_event=False,
                checkpoint_write_s=None,
                checkpoint_blocking=False,
            )
        )
    return samples


def starved_by_slow_network_fs(n: int = 200, seed: int = 1) -> list[StepSample]:
    """GPU mostly idle waiting on data; disk itself isn't saturated but
    per-request latency (await) is high — classic NFS/S3-backed dataset
    latency signature, small read sizes, low queue depth."""
    rng = random.Random(seed)
    samples = []
    t = 0.0
    for i in range(n):
        compute = rng.gauss(0.40, 0.02)
        wait = max(0.0, rng.gauss(0.9, 0.15))
        t += compute + wait
        samples.append(
            StepSample(
                t=t,
                step=i,
                step_time_s=compute + wait,
                data_wait_s=wait,
                compute_time_s=compute,
                num_workers=8,
                prefetch_factor=4,
                queue_depth=rng.randint(0, 2),
                queue_capacity=32,
                worker_cpu_pct=rng.gauss(15, 5),
                worker_restarts=0,
                disk_name="nfs-shared",
                disk_read_bytes_per_s=rng.gauss(20e6, 5e6),
                disk_read_iops=rng.gauss(150, 30),
                disk_util_pct=rng.gauss(35, 8),
                disk_avg_await_ms=rng.gauss(45, 10),
                avg_read_size_bytes=rng.gauss(40_000, 5_000),
                is_network_fs=True,
                psi_io_some_avg10=rng.gauss(20, 4),
                psi_io_full_avg10=rng.gauss(12, 3),
            )
        )
    return samples


def cpu_bound_decode(n: int = 200, seed: int = 2) -> list[StepSample]:
    """Workers pegged at ~100% CPU decoding/tokenizing while disk sits idle
    and GPU waits — the fix is more workers or cheaper decode, not storage."""
    rng = random.Random(seed)
    samples = []
    t = 0.0
    for i in range(n):
        compute = rng.gauss(0.30, 0.02)
        wait = max(0.0, rng.gauss(0.55, 0.08))
        t += compute + wait
        samples.append(
            StepSample(
                t=t,
                step=i,
                step_time_s=compute + wait,
                data_wait_s=wait,
                compute_time_s=compute,
                num_workers=4,
                prefetch_factor=2,
                queue_depth=rng.randint(0, 1),
                queue_capacity=8,
                worker_cpu_pct=rng.gauss(98, 1.5),
                worker_restarts=0,
                disk_name="nvme0n1",
                disk_read_bytes_per_s=rng.gauss(40e6, 5e6),
                disk_read_iops=rng.gauss(300, 40),
                disk_util_pct=rng.gauss(8, 2),
                disk_avg_await_ms=rng.gauss(0.4, 0.1),
                avg_read_size_bytes=rng.gauss(120_000, 8_000),
            )
        )
    return samples


def small_file_storm(n: int = 200, seed: int = 3) -> list[StepSample]:
    """Extremely high IOPS, tiny average read size, high disk utilization —
    dataset stored as millions of small files rather than sharded/tarred."""
    rng = random.Random(seed)
    samples = []
    t = 0.0
    for i in range(n):
        compute = rng.gauss(0.35, 0.02)
        wait = max(0.0, rng.gauss(0.6, 0.1))
        t += compute + wait
        samples.append(
            StepSample(
                t=t,
                step=i,
                step_time_s=compute + wait,
                data_wait_s=wait,
                compute_time_s=compute,
                num_workers=16,
                prefetch_factor=2,
                queue_depth=rng.randint(0, 3),
                queue_capacity=32,
                worker_cpu_pct=rng.gauss(70, 8),
                worker_restarts=0,
                disk_name="nvme0n1",
                disk_read_bytes_per_s=rng.gauss(80e6, 10e6),
                disk_read_iops=rng.gauss(18000, 1500),
                disk_util_pct=rng.gauss(92, 4),
                disk_avg_await_ms=rng.gauss(3.5, 0.8),
                avg_read_size_bytes=rng.gauss(4_500, 500),
            )
        )
    return samples


def checkpoint_stall(n: int = 200, seed: int = 4, stall_every: int = 50) -> list[StepSample]:
    """Otherwise healthy pipeline, but every `stall_every` steps a
    synchronous checkpoint write blocks the training loop for seconds."""
    rng = random.Random(seed)
    samples = []
    t = 0.0
    for i in range(n):
        compute = rng.gauss(0.40, 0.02)
        wait = max(0.0, rng.gauss(0.02, 0.005))
        is_ckpt = (i > 0 and i % stall_every == 0)
        ckpt_write = rng.gauss(8.0, 1.0) if is_ckpt else None
        step_time = compute + wait + (ckpt_write or 0.0)
        t += step_time
        samples.append(
            StepSample(
                t=t,
                step=i,
                step_time_s=step_time,
                data_wait_s=wait,
                compute_time_s=compute,
                num_workers=8,
                prefetch_factor=4,
                queue_depth=rng.randint(20, 32),
                queue_capacity=32,
                worker_cpu_pct=rng.gauss(55, 5),
                worker_restarts=0,
                disk_name="nvme0n1",
                disk_read_bytes_per_s=rng.gauss(300e6, 20e6),
                disk_read_iops=rng.gauss(2000, 100),
                disk_util_pct=rng.gauss(30, 5),
                disk_avg_await_ms=rng.gauss(0.3, 0.05),
                avg_read_size_bytes=rng.gauss(150_000, 10_000),
                checkpoint_write_s=ckpt_write,
                checkpoint_blocking=is_ckpt,
            )
        )
    return samples


def worker_crash_loop(n: int = 200, seed: int = 5) -> list[StepSample]:
    """Worker processes periodically die (OOM, uncaught exception in a
    transform) and get respawned by the DataLoader, each restart costing a
    multi-second stall while the pool refills."""
    rng = random.Random(seed)
    samples = []
    t = 0.0
    restarts = 0
    for i in range(n):
        crashed_this_step = rng.random() < 0.02
        if crashed_this_step:
            restarts += 1
        compute = rng.gauss(0.40, 0.02)
        wait = rng.gauss(3.0, 0.5) if crashed_this_step else max(0.0, rng.gauss(0.03, 0.01))
        t += compute + wait
        samples.append(
            StepSample(
                t=t,
                step=i,
                step_time_s=compute + wait,
                data_wait_s=wait,
                compute_time_s=compute,
                num_workers=8,
                prefetch_factor=4,
                queue_depth=0 if crashed_this_step else rng.randint(15, 32),
                queue_capacity=32,
                worker_cpu_pct=rng.gauss(50, 5),
                worker_restarts=restarts,
                disk_name="nvme0n1",
                disk_read_bytes_per_s=rng.gauss(280e6, 20e6),
                disk_read_iops=rng.gauss(1900, 100),
                disk_util_pct=rng.gauss(28, 5),
                disk_avg_await_ms=rng.gauss(0.3, 0.05),
                avg_read_size_bytes=rng.gauss(150_000, 10_000),
            )
        )
    return samples


SCENARIOS = {
    "healthy": healthy_pipeline,
    "network_fs_latency": starved_by_slow_network_fs,
    "cpu_bound_decode": cpu_bound_decode,
    "small_file_storm": small_file_storm,
    "checkpoint_stall": checkpoint_stall,
    "worker_crash_loop": worker_crash_loop,
}


def generate(scenario: str, n: Optional[int] = None, seed: Optional[int] = None) -> list[StepSample]:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choices: {sorted(SCENARIOS)}")
    kwargs = {}
    if n is not None:
        kwargs["n"] = n
    if seed is not None:
        kwargs["seed"] = seed
    return SCENARIOS[scenario](**kwargs)
