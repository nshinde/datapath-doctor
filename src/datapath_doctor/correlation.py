"""Cross-rule correlation for turning independent findings into a concise root-cause hypothesis.

Rules intentionally stay independent and testable.  This module is the layer that
combines compatible signals after evaluation, without making individual rules
depend on each other.
"""

from __future__ import annotations

from dataclasses import dataclass

from datapath_doctor.models import Finding


@dataclass(frozen=True)
class Diagnosis:
    """A best-effort root-cause hypothesis derived from multiple rule findings."""

    root_cause: str
    confidence: str
    chain: tuple[str, ...]
    supporting_rule_ids: tuple[str, ...]


def correlate_findings(findings: list[Finding]) -> Diagnosis | None:
    """Return the strongest supported root-cause hypothesis, if one exists.

    Correlation requires multiple compatible signals where possible.  It does
    not suppress the underlying findings; callers should still render/store
    those independently.
    """
    fired = {f.rule_id for f in findings}

    # Symptom + specific cause + (optionally) queue evidence.
    if {"DPD-001", "DPD-007"} <= fired:
        chain = ["training loop is waiting for input"]
        supporting = ["DPD-001"]
        if "DPD-004" in fired:
            chain.append("prefetch queue is draining/empty")
            supporting.append("DPD-004")
        chain.append("network-backed reads show elevated request latency")
        supporting.append("DPD-007")
        return Diagnosis(
            root_cause="network storage latency is starving the training loop",
            confidence="high" if "DPD-004" in fired else "medium",
            chain=tuple(chain),
            supporting_rule_ids=tuple(supporting),
        )

    if {"DPD-001", "DPD-005"} <= fired:
        return Diagnosis(
            root_cause="CPU-bound input preprocessing is starving the training loop",
            confidence="high",
            chain=(
                "training loop is waiting for input",
                "DataLoader workers are CPU-saturated",
                "storage is not simultaneously saturated",
            ),
            supporting_rule_ids=("DPD-001", "DPD-005"),
        )

    if {"DPD-001", "DPD-002"} <= fired:
        return Diagnosis(
            root_cause="local/block storage saturation is starving the training loop",
            confidence="high",
            chain=(
                "training loop is waiting for input",
                "backing block device is highly utilized",
                "I/O service/queue latency is limiting batch delivery",
            ),
            supporting_rule_ids=("DPD-001", "DPD-002"),
        )

    if {"DPD-001", "DPD-006"} <= fired:
        return Diagnosis(
            root_cause="small-file I/O overhead is starving the training loop",
            confidence="medium",
            chain=(
                "training loop is waiting for input",
                "reads are small and IOPS-heavy",
                "dataset layout is creating excessive per-read overhead",
            ),
            supporting_rule_ids=("DPD-001", "DPD-006"),
        )

    if "DPD-009" in fired:
        return Diagnosis(
            root_cause="synchronous checkpoint I/O is blocking training progress",
            confidence="high",
            chain=(
                "checkpoint write overlaps the slow step",
                "the checkpoint is marked blocking",
                "training cannot advance until the write completes",
            ),
            supporting_rule_ids=("DPD-009",),
        )

    if "DPD-010" in fired:
        return Diagnosis(
            root_cause="DataLoader worker instability is disrupting batch delivery",
            confidence="medium",
            chain=(
                "worker process identities are changing",
                "restarts are observed within the diagnostic window",
                "batch delivery can stall while workers recover",
            ),
            supporting_rule_ids=("DPD-010",),
        )

    return None
