from __future__ import annotations

from datapath_doctor.engine import Rule
from datapath_doctor.models import Finding, Severity, WindowSummary


class DiskThroughputSaturationRule(Rule):
    """Fires when the block device backing the dataset is running near its
    I/O ceiling: high utilization % combined with elevated average await
    time indicates the physical disk (or its queue) is the bottleneck, not
    the DataLoader's CPU-side logic.
    """

    rule_id = "DPD-002"
    name = "Disk I/O saturation"
    description = (
        "Flags sustained high disk utilization (% time with I/O in flight) "
        "paired with elevated average await, indicating the storage device "
        "itself is the limiting resource."
    )

    UTIL_WARNING = 80.0
    UTIL_CRITICAL = 95.0
    AWAIT_MS_THRESHOLD = 2.0  # above this, queueing/service time is meaningfully adding latency

    def evaluate(self, window: WindowSummary) -> list[Finding]:
        util = window.mean("disk_util_pct")
        await_ms = window.mean("disk_avg_await_ms")
        throughput = window.mean("disk_read_bytes_per_s")
        disk_name = window.last("disk_name") or "unknown"

        if util is None:
            return []
        if util < self.UTIL_WARNING:
            return []

        severity = Severity.CRITICAL if util >= self.UTIL_CRITICAL else Severity.WARNING
        throughput_mb = (throughput / 1e6) if throughput is not None else None

        message = (
            f"Disk '{disk_name}' averaged {util:.1f}% utilization"
            + (f" at {throughput_mb:.0f} MB/s read throughput" if throughput_mb is not None else "")
            + (f", with {await_ms:.2f} ms average await" if await_ms is not None else "")
            + " — the device is the bottleneck feeding training."
        )

        fix = (
            "The disk is saturated at its current throughput ceiling. Options: move the "
            "dataset to faster storage (local NVMe vs network-attached), shard/compress the "
            "dataset to reduce bytes read per sample, or reduce read amplification (e.g. "
            "sequential shard reads instead of random small reads — see the small-file-I/O "
            "finding if present)."
        )
        if await_ms is not None and await_ms >= self.AWAIT_MS_THRESHOLD:
            fix += (
                f" Average await ({await_ms:.2f} ms) is also elevated, suggesting queue depth "
                "at the device may be too high for its IOPS budget — consider fewer concurrent "
                "readers or a device with better queued-IOPS characteristics."
            )

        return [
            Finding(
                rule_id=self.rule_id,
                severity=severity,
                title="Disk I/O saturation on input path",
                message=message,
                evidence={
                    "disk_name": disk_name,
                    "mean_disk_util_pct": round(util, 2),
                    "mean_read_throughput_mb_s": round(throughput_mb, 2) if throughput_mb is not None else None,
                    "mean_avg_await_ms": round(await_ms, 3) if await_ms is not None else None,
                },
                suggested_fix=fix,
            )
        ]
