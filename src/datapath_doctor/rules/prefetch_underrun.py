from __future__ import annotations

from datapath_doctor.engine import Rule
from datapath_doctor.models import Finding, Severity, WindowSummary


class PrefetchBufferUnderrunRule(Rule):
    """Fires when the DataLoader's prefetch queue is frequently empty (or
    near-empty) right when the training loop asks for the next batch —
    the queue isn't staying ahead of consumption, regardless of why.
    """

    rule_id = "DPD-004"
    name = "Prefetch buffer underrun"
    description = (
        "Flags when queue_depth is at/near zero across a large fraction of "
        "samples, meaning producers (worker processes) aren't keeping the "
        "prefetch buffer ahead of the consumer."
    )

    EMPTY_THRESHOLD_FRACTION = 0.30  # fraction of samples with queue_depth <= low_water
    LOW_WATER_FRACTION_OF_CAPACITY = 0.1

    def evaluate(self, window: WindowSummary) -> list[Finding]:
        depths = [
            (s.queue_depth, s.queue_capacity)
            for s in window.samples
            if s.queue_depth is not None and s.queue_capacity is not None and s.queue_capacity > 0
        ]
        if not depths:
            return []

        empty_count = sum(
            1 for depth, cap in depths if depth <= max(1, cap * self.LOW_WATER_FRACTION_OF_CAPACITY)
        )
        fraction_empty = empty_count / len(depths)
        if fraction_empty < self.EMPTY_THRESHOLD_FRACTION:
            return []

        mean_depth = sum(d for d, _ in depths) / len(depths)
        mean_cap = sum(c for _, c in depths) / len(depths)
        severity = Severity.CRITICAL if fraction_empty >= 0.7 else Severity.WARNING

        num_workers = window.last("num_workers")
        prefetch_factor = window.last("prefetch_factor")

        fix = (
            "The prefetch buffer is running dry, meaning workers can't produce batches as "
            "fast as the training loop consumes them. If disk/network telemetry doesn't show "
            "saturation and workers aren't CPU-bound, the fix is usually mechanical: raise "
            "prefetch_factor and/or num_workers so more batches are in flight."
        )
        if num_workers is not None and prefetch_factor is not None:
            fix += f" Current settings: num_workers={num_workers}, prefetch_factor={prefetch_factor}."

        return [
            Finding(
                rule_id=self.rule_id,
                severity=severity,
                title="Prefetch buffer underrun",
                message=(
                    f"Prefetch queue was at or below {self.LOW_WATER_FRACTION_OF_CAPACITY:.0%} of "
                    f"capacity in {fraction_empty:.0%} of samples (mean depth {mean_depth:.1f} / "
                    f"capacity {mean_cap:.0f})."
                ),
                evidence={
                    "fraction_samples_near_empty": round(fraction_empty, 3),
                    "mean_queue_depth": round(mean_depth, 2),
                    "mean_queue_capacity": round(mean_cap, 2),
                    "num_workers": num_workers,
                    "prefetch_factor": prefetch_factor,
                },
                suggested_fix=fix,
            )
        ]
