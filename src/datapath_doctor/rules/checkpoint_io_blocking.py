from __future__ import annotations

from datapath_doctor.engine import Rule
from datapath_doctor.models import Finding, Severity, WindowSummary


class CheckpointIOBlockingRule(Rule):
    """Fires when checkpoint writes are synchronous and long enough to
    materially inflate step time on the steps where they happen — training
    throughput numbers that don't account for this can look fine on
    average while periodically stalling for seconds.
    """

    rule_id = "DPD-009"
    name = "Checkpoint I/O blocking training steps"
    description = (
        "Flags checkpoint writes that block the training loop for long "
        "enough to be a meaningful tax on wall-clock training time, "
        "suggesting async/overlapped checkpointing would help."
    )

    BLOCKING_WRITE_WARNING_S = 2.0
    BLOCKING_WRITE_CRITICAL_S = 8.0

    def evaluate(self, window: WindowSummary) -> list[Finding]:
        blocking_writes = [
            s.checkpoint_write_s
            for s in window.samples
            if s.checkpoint_blocking and s.checkpoint_write_s is not None
        ]
        if not blocking_writes:
            return []

        mean_write = sum(blocking_writes) / len(blocking_writes)
        max_write = max(blocking_writes)
        if mean_write < self.BLOCKING_WRITE_WARNING_S:
            return []

        severity = Severity.CRITICAL if mean_write >= self.BLOCKING_WRITE_CRITICAL_S else Severity.WARNING

        # Estimate wall-clock cost as a fraction of the whole window.
        total_step = sum(window.values("step_time_s")) or None
        total_blocking = sum(blocking_writes)
        overhead_fraction = (total_blocking / total_step) if total_step else None

        message = (
            f"{len(blocking_writes)} checkpoint write(s) blocked the training loop for an "
            f"average of {mean_write:.1f}s (max {max_write:.1f}s) in this window."
        )
        if overhead_fraction is not None:
            message += f" That's {overhead_fraction:.1%} of total wall-clock time in this window."

        return [
            Finding(
                rule_id=self.rule_id,
                severity=severity,
                title="Synchronous checkpoint I/O is stalling training",
                message=message,
                evidence={
                    "num_blocking_checkpoints": len(blocking_writes),
                    "mean_blocking_write_s": round(mean_write, 2),
                    "max_blocking_write_s": round(max_write, 2),
                    "overhead_fraction_of_window": round(overhead_fraction, 4)
                    if overhead_fraction is not None
                    else None,
                },
                suggested_fix=(
                    "Move checkpoint writes off the critical path: write to a staging buffer "
                    "(pinned host memory or local NVMe) and flush to durable/shared storage "
                    "asynchronously in a background thread/process, or use a framework's async "
                    "checkpoint API (e.g. PyTorch Distributed Checkpoint's async save) so the "
                    "next training step can start immediately."
                ),
            )
        ]
