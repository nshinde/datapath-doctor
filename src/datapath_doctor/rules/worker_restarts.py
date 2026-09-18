from __future__ import annotations

from datapath_doctor.engine import Rule
from datapath_doctor.models import Finding, Severity, WindowSummary


class WorkerCrashRestartLoopRule(Rule):
    """Fires when DataLoader worker processes are restarting during the
    window — each restart typically costs a multi-second stall while the
    pool refills, and repeated restarts point to an underlying instability
    (OOM kills, uncaught exceptions in a transform, flaky remote reads)
    rather than a capacity/tuning problem.
    """

    rule_id = "DPD-010"
    name = "DataLoader worker crash/restart loop"
    description = (
        "Flags any observed increase in worker_restarts across the window, "
        "since worker restarts are never expected in a healthy pipeline."
    )

    def evaluate(self, window: WindowSummary) -> list[Finding]:
        restarts = [s.worker_restarts for s in window.samples if s.worker_restarts is not None]
        if not restarts:
            return []

        delta = max(restarts) - min(restarts)
        if delta <= 0:
            return []

        rate_per_1k_steps = (delta / len(window.samples)) * 1000
        severity = Severity.CRITICAL if delta >= 5 else Severity.WARNING

        return [
            Finding(
                rule_id=self.rule_id,
                severity=severity,
                title="DataLoader workers are crashing and restarting",
                message=(
                    f"Observed {delta} worker restart(s) over {len(window.samples)} samples "
                    f"(~{rate_per_1k_steps:.1f} per 1000 steps). Each restart typically costs a "
                    "multi-second stall while the worker pool refills."
                ),
                evidence={
                    "worker_restarts_in_window": delta,
                    "restarts_per_1000_steps": round(rate_per_1k_steps, 2),
                },
                suggested_fix=(
                    "Check worker dmesg/OOM-killer logs first (a worker holding too much "
                    "per-sample memory, e.g. decoding oversized images, is the most common "
                    "cause), then check for uncaught exceptions in the dataset's __getitem__ "
                    "or transform pipeline swallowed by the worker process boundary. Reducing "
                    "per-worker memory footprint or wrapping transforms with defensive error "
                    "handling usually resolves this faster than tuning worker count."
                ),
            )
        ]
