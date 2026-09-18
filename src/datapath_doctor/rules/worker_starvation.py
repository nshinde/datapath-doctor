from __future__ import annotations

from datapath_doctor.engine import Rule
from datapath_doctor.models import Finding, Severity, WindowSummary


class GpuStarvedByDataPipelineRule(Rule):
    """Fires when the training loop spends a large fraction of step time
    blocked waiting for the next batch — the headline "is this a data
    pipeline problem at all" check that other, more specific rules refine.
    """

    rule_id = "DPD-001"
    name = "GPU starved waiting on data pipeline"
    description = (
        "Flags when mean data_wait_s / step_time_s exceeds a threshold, "
        "meaning the accelerator is idle for a significant fraction of "
        "every step waiting on the input pipeline."
    )

    WARNING_FRACTION = 0.15
    CRITICAL_FRACTION = 0.40

    def evaluate(self, window: WindowSummary) -> list[Finding]:
        waits = window.values("data_wait_s")
        steps = window.values("step_time_s")
        if not waits or not steps or len(waits) != len(steps):
            return []

        total_wait = sum(waits)
        total_step = sum(steps)
        if total_step <= 0:
            return []

        fraction = total_wait / total_step
        if fraction < self.WARNING_FRACTION:
            return []

        severity = Severity.CRITICAL if fraction >= self.CRITICAL_FRACTION else Severity.WARNING
        pct = fraction * 100
        mean_wait_ms = (total_wait / len(waits)) * 1000

        return [
            Finding(
                rule_id=self.rule_id,
                severity=severity,
                title="Data pipeline is the training bottleneck",
                message=(
                    f"The GPU spent {pct:.1f}% of step time waiting on the data "
                    f"pipeline across {len(waits)} steps (mean {mean_wait_ms:.1f} ms/step "
                    f"of pure wait). Compute is starved."
                ),
                evidence={
                    "data_wait_fraction": round(fraction, 4),
                    "mean_data_wait_ms": round(mean_wait_ms, 2),
                    "num_steps": len(waits),
                },
                suggested_fix=(
                    "This step time is dominated by waiting, not computing. Check the more "
                    "specific findings below (disk saturation, network FS latency, CPU-bound "
                    "decode, prefetch underrun) to find the root cause before tuning workers "
                    "blindly."
                ),
            )
        ]
