from __future__ import annotations

from datapath_doctor.engine import Rule
from datapath_doctor.models import Finding, Severity, WindowSummary


class WorkerPoolCPUBoundRule(Rule):
    """Fires when worker processes are pegged near 100% CPU while the disk
    underneath is not saturated — the bottleneck is CPU-side work
    (decode, tokenize, augment, collate) rather than storage bandwidth, so
    throwing faster storage at it won't help.
    """

    rule_id = "DPD-005"
    name = "Worker pool CPU-bound (not storage-bound)"
    description = (
        "Flags when mean worker CPU% is very high while disk utilization is "
        "comparatively low, indicating the input pipeline is limited by "
        "CPU-side transform work, not I/O."
    )

    WORKER_CPU_THRESHOLD = 90.0
    DISK_UTIL_CEILING = 50.0  # below this, disk clearly isn't the constraint

    def evaluate(self, window: WindowSummary) -> list[Finding]:
        worker_cpu = window.mean("worker_cpu_pct")
        disk_util = window.mean("disk_util_pct")
        if worker_cpu is None:
            return []
        if worker_cpu < self.WORKER_CPU_THRESHOLD:
            return []
        if disk_util is not None and disk_util >= self.DISK_UTIL_CEILING:
            # Disk is also busy — let DiskThroughputSaturationRule own this one
            # rather than double-diagnosing.
            return []

        num_workers = window.last("num_workers")
        severity = Severity.WARNING if worker_cpu < 98 else Severity.CRITICAL

        return [
            Finding(
                rule_id=self.rule_id,
                severity=severity,
                title="Worker pool CPU-bound",
                message=(
                    f"DataLoader worker processes averaged {worker_cpu:.1f}% CPU"
                    + (f" while disk utilization was only {disk_util:.1f}%" if disk_util is not None else "")
                    + f" (num_workers={num_workers}). The bottleneck is CPU-side per-sample "
                    "work (decode/tokenize/augment/collate), not storage."
                ),
                evidence={
                    "mean_worker_cpu_pct": round(worker_cpu, 1),
                    "mean_disk_util_pct": round(disk_util, 1) if disk_util is not None else None,
                    "num_workers": num_workers,
                },
                suggested_fix=(
                    "Faster storage will not help here. Try: increase num_workers if CPU cores "
                    "are still available on the node; move expensive decode/augment steps to "
                    "GPU (e.g. NVIDIA DALI or GPU-side JPEG decode); pre-tokenize/pre-decode the "
                    "dataset offline so workers only deserialize; or reduce per-sample transform "
                    "cost (cheaper resize/augmentation pipeline)."
                ),
            )
        ]
