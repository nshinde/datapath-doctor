from __future__ import annotations

from datapath_doctor.engine import Rule
from datapath_doctor.models import Finding, Severity, WindowSummary


class SystemIOPressureRule(Rule):
    """Fires on elevated Linux PSI (pressure stall information) for I/O,
    which captures contention that per-device diskstats can miss — e.g.
    multiple jobs/containers sharing one node competing for the same
    storage, or cgroup-level I/O throttling.
    """

    rule_id = "DPD-003"
    name = "System-wide I/O pressure (PSI)"
    description = (
        "Flags elevated /proc/pressure/io 'full' average, meaning all "
        "runnable tasks on the node were stalled on I/O simultaneously — "
        "a stronger signal of contention than per-device utilization alone."
    )

    FULL_WARNING = 5.0
    FULL_CRITICAL = 20.0

    def evaluate(self, window: WindowSummary) -> list[Finding]:
        full_avg10 = window.mean("psi_io_full_avg10")
        some_avg10 = window.mean("psi_io_some_avg10")
        if full_avg10 is None:
            return []
        if full_avg10 < self.FULL_WARNING:
            return []

        severity = Severity.CRITICAL if full_avg10 >= self.FULL_CRITICAL else Severity.WARNING

        return [
            Finding(
                rule_id=self.rule_id,
                severity=severity,
                title="Node-wide I/O contention (PSI)",
                message=(
                    f"PSI io 'full' avg10 is {full_avg10:.1f}% "
                    + (f"(some avg10 {some_avg10:.1f}%) " if some_avg10 is not None else "")
                    + "— every runnable task on this node was blocked on I/O at once for a "
                    "meaningful fraction of the window. This points to node-level contention, "
                    "not just this job's own read pattern."
                ),
                evidence={
                    "psi_io_full_avg10": round(full_avg10, 2),
                    "psi_io_some_avg10": round(some_avg10, 2) if some_avg10 is not None else None,
                },
                suggested_fix=(
                    "Check for co-located jobs/containers on the same node competing for the "
                    "same disk or network-storage mount, cgroup io.max throttling, or a noisy "
                    "neighbor. If this node is otherwise dedicated to this job, treat it the "
                    "same as the disk-saturation finding."
                ),
            )
        ]
