from __future__ import annotations

from datapath_doctor.engine import Rule
from datapath_doctor.models import Finding, Severity, WindowSummary


class SmallFileIOOverheadRule(Rule):
    """Fires when average read size is tiny and IOPS are very high relative
    to throughput — the classic "millions of small files" dataset layout
    that trades sequential bandwidth for per-request overhead.
    """

    rule_id = "DPD-006"
    name = "Small-file I/O overhead"
    description = (
        "Flags a read pattern dominated by many small requests (low average "
        "read size, high IOPS relative to throughput), which underutilizes "
        "storage bandwidth and amplifies per-request latency/metadata cost."
    )

    AVG_READ_SIZE_WARNING_BYTES = 64 * 1024  # 64 KiB
    AVG_READ_SIZE_CRITICAL_BYTES = 16 * 1024  # 16 KiB
    MIN_IOPS_TO_CONSIDER = 500  # avoid firing on a nearly-idle disk

    def evaluate(self, window: WindowSummary) -> list[Finding]:
        avg_size = window.mean("avg_read_size_bytes")
        iops = window.mean("disk_read_iops")
        if avg_size is None or iops is None:
            return []
        if iops < self.MIN_IOPS_TO_CONSIDER:
            return []
        if avg_size > self.AVG_READ_SIZE_WARNING_BYTES:
            return []

        severity = (
            Severity.CRITICAL if avg_size <= self.AVG_READ_SIZE_CRITICAL_BYTES else Severity.WARNING
        )
        avg_size_kb = avg_size / 1024

        return [
            Finding(
                rule_id=self.rule_id,
                severity=severity,
                title="Small-file I/O pattern detected",
                message=(
                    f"Average read size is {avg_size_kb:.1f} KiB at {iops:.0f} IOPS — the input "
                    "path is issuing a large number of small reads rather than fewer, larger "
                    "sequential ones."
                ),
                evidence={
                    "mean_avg_read_size_kb": round(avg_size_kb, 2),
                    "mean_read_iops": round(iops, 1),
                },
                suggested_fix=(
                    "Repack the dataset into larger sequential-read shards (WebDataset tar "
                    "shards, TFRecord-style, or Parquet row groups) sized for a few hundred MB "
                    "to a few GB per shard, and read whole shards sequentially per worker "
                    "instead of opening one file per sample. This is usually the single "
                    "highest-leverage fix for a small-file-bound input path."
                ),
            )
        ]
