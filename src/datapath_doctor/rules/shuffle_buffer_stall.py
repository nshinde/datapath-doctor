from __future__ import annotations

from datapath_doctor.engine import Rule
from datapath_doctor.models import Finding, Severity, WindowSummary


class ShuffleBufferStallRule(Rule):
    """Fires when the shuffle buffer is running near-empty and refill
    events are frequent — steps taken during a refill tend to stall since
    the buffer can't serve a shuffled sample until enough new elements have
    arrived.
    """

    rule_id = "DPD-008"
    name = "Shuffle buffer stall"
    description = (
        "Flags a shuffle buffer that spends significant time near-empty "
        "with frequent refill events, undersized for the pipeline's "
        "throughput."
    )

    LOW_FILL_FRACTION_OF_CAPACITY = 0.15
    LOW_FILL_SAMPLE_FRACTION = 0.25
    REFILL_EVENT_FRACTION = 0.10

    def evaluate(self, window: WindowSummary) -> list[Finding]:
        pairs = [
            (s.shuffle_buffer_size, s.shuffle_buffer_capacity)
            for s in window.samples
            if s.shuffle_buffer_size is not None and s.shuffle_buffer_capacity
        ]
        if not pairs:
            return []

        low_fill_count = sum(
            1 for size, cap in pairs if size <= cap * self.LOW_FILL_FRACTION_OF_CAPACITY
        )
        low_fill_fraction = low_fill_count / len(pairs)

        refill_fraction = window.fraction_true("shuffle_refill_event") or 0.0

        if low_fill_fraction < self.LOW_FILL_SAMPLE_FRACTION and refill_fraction < self.REFILL_EVENT_FRACTION:
            return []

        severity = (
            Severity.CRITICAL
            if low_fill_fraction >= 0.5 or refill_fraction >= 0.3
            else Severity.WARNING
        )
        mean_cap = sum(c for _, c in pairs) / len(pairs)

        return [
            Finding(
                rule_id=self.rule_id,
                severity=severity,
                title="Shuffle buffer undersized / stalling",
                message=(
                    f"Shuffle buffer was near-empty (<= {self.LOW_FILL_FRACTION_OF_CAPACITY:.0%} "
                    f"of its {mean_cap:.0f}-element capacity) in {low_fill_fraction:.0%} of "
                    f"samples, with refill events in {refill_fraction:.0%} of samples."
                ),
                evidence={
                    "low_fill_fraction": round(low_fill_fraction, 3),
                    "refill_event_fraction": round(refill_fraction, 3),
                    "mean_capacity": round(mean_cap, 1),
                },
                suggested_fix=(
                    "Increase the shuffle buffer size so it stays ahead of consumption, or "
                    "reduce upstream latency feeding it (see disk/network/CPU findings above) "
                    "so it refills faster. If shuffle quality at a smaller buffer is acceptable, "
                    "an alternative is pre-shuffling shards on disk so the in-memory buffer "
                    "needs to do less work."
                ),
            )
        ]
