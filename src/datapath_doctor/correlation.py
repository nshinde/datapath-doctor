"""Evidence-aware cross-rule correlation.

Rules remain independent and unit-testable. This layer combines the evidence
they emit into one best-supported diagnosis without hiding the raw findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from datapath_doctor.models import Finding


@dataclass(frozen=True)
class Diagnosis:
    """A best-effort diagnosis derived from multiple pieces of rule evidence."""

    summary: str
    confidence: str
    diagnosis_type: str  # root_cause | contributing_factor | symptom
    chain: tuple[str, ...]
    supporting_rule_ids: tuple[str, ...]
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def root_cause(self) -> str:
        """Backward-compatible alias used by older callers."""
        return self.summary


def _by_id(findings: list[Finding]) -> dict[str, Finding]:
    return {finding.rule_id: finding for finding in findings}


def _evidence_float(finding: Finding | None, key: str) -> float | None:
    if finding is None:
        return None
    value = finding.evidence.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _confidence(score: int) -> str:
    if score >= 5:
        return "high"
    if score >= 3:
        return "medium"
    return "low"


def _fmt_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def correlate_findings(findings: list[Finding]) -> Diagnosis | None:
    """Return the strongest supported diagnosis.

    Correlation uses the measurements inside ``Finding.evidence`` rather than
    treating rule IDs alone as proof. Missing evidence lowers confidence; a
    contradictory finding can downgrade a root-cause claim to a contributing
    factor.
    """
    by_id = _by_id(findings)
    starvation = by_id.get("DPD-001")
    disk = by_id.get("DPD-002")
    prefetch = by_id.get("DPD-004")
    cpu = by_id.get("DPD-005")
    small_files = by_id.get("DPD-006")
    network = by_id.get("DPD-007")
    checkpoint = by_id.get("DPD-009")
    restarts = by_id.get("DPD-010")

    if checkpoint is not None:
        overhead = _evidence_float(checkpoint, "overhead_fraction_of_window")
        mean_write = _evidence_float(checkpoint, "mean_blocking_write_s")
        score = 2
        if overhead is not None and overhead >= 0.10:
            score += 2
        if mean_write is not None and mean_write >= 8.0:
            score += 2

        chain = ["checkpoint write is synchronous/blocking"]
        if mean_write is not None:
            chain.append(f"mean blocking write time is {mean_write:.1f}s")
        if overhead is not None:
            chain.append(f"checkpointing consumes {_fmt_percent(overhead)} of the observed window")

        return Diagnosis(
            summary="synchronous checkpoint I/O is blocking training progress",
            confidence=_confidence(score),
            diagnosis_type="root_cause" if score >= 4 else "contributing_factor",
            chain=tuple(chain),
            supporting_rule_ids=("DPD-009",),
            evidence={
                "mean_blocking_write_s": mean_write,
                "overhead_fraction_of_window": overhead,
                "correlation_score": score,
            },
        )

    if starvation is not None and cpu is not None:
        wait_fraction = _evidence_float(starvation, "data_wait_fraction")
        worker_cpu = _evidence_float(cpu, "mean_worker_cpu_pct")
        disk_util = _evidence_float(cpu, "mean_disk_util_pct")
        score = 0
        if wait_fraction is not None:
            score += 2 if wait_fraction >= 0.40 else 1 if wait_fraction >= 0.15 else 0
        if worker_cpu is not None:
            score += 2 if worker_cpu >= 98.0 else 1 if worker_cpu >= 90.0 else 0
        if disk_util is not None and disk_util < 50.0:
            score += 1
        if network is not None or disk is not None:
            score -= 1

        chain = []
        if wait_fraction is not None:
            chain.append(f"training waits on input for {_fmt_percent(wait_fraction)} of step time")
        if worker_cpu is not None:
            chain.append(f"DataLoader workers average {worker_cpu:.1f}% CPU")
        if disk_util is not None:
            chain.append(f"backing disk averages only {disk_util:.1f}% utilization")

        return Diagnosis(
            summary="CPU-bound input preprocessing is starving the training loop",
            confidence=_confidence(score),
            diagnosis_type="root_cause" if score >= 4 and network is None and disk is None else "contributing_factor",
            chain=tuple(chain),
            supporting_rule_ids=("DPD-001", "DPD-005"),
            evidence={
                "data_wait_fraction": wait_fraction,
                "mean_worker_cpu_pct": worker_cpu,
                "mean_disk_util_pct": disk_util,
                "correlation_score": score,
            },
        )

    if starvation is not None and network is not None:
        wait_fraction = _evidence_float(starvation, "data_wait_fraction")
        queue_empty = _evidence_float(prefetch, "fraction_samples_near_empty")
        remote_latency = _evidence_float(network, "mean_remote_read_latency_ms")
        await_ms = _evidence_float(network, "mean_avg_await_ms")
        latency_ms = remote_latency if remote_latency is not None else await_ms

        score = 0
        if wait_fraction is not None:
            score += 2 if wait_fraction >= 0.40 else 1 if wait_fraction >= 0.15 else 0
        if queue_empty is not None:
            score += 2 if queue_empty >= 0.70 else 1 if queue_empty >= 0.30 else 0
        if latency_ms is not None:
            score += 2 if latency_ms >= 50.0 else 1 if latency_ms >= 15.0 else 0
        if cpu is not None:
            score -= 2

        chain = []
        if wait_fraction is not None:
            chain.append(f"training waits on input for {_fmt_percent(wait_fraction)} of step time")
        if queue_empty is not None:
            chain.append(f"prefetch queue is near-empty in {_fmt_percent(queue_empty)} of samples")
        if latency_ms is not None:
            source = "remote read latency" if remote_latency is not None else "I/O await"
            chain.append(f"network-backed storage {source} averages {latency_ms:.1f} ms")
        if cpu is not None:
            chain.append("worker CPU saturation is also present, so storage may be one contributor rather than the sole cause")

        supporting = ["DPD-001"]
        if prefetch is not None:
            supporting.append("DPD-004")
        supporting.append("DPD-007")

        return Diagnosis(
            summary="remote input-storage latency is starving the training loop",
            confidence=_confidence(score),
            diagnosis_type="root_cause" if score >= 5 and cpu is None else "contributing_factor",
            chain=tuple(chain),
            supporting_rule_ids=tuple(supporting),
            evidence={
                "data_wait_fraction": wait_fraction,
                "fraction_samples_near_empty": queue_empty,
                "remote_latency_ms": latency_ms,
                "correlation_score": score,
            },
        )

    if starvation is not None and small_files is not None:
        wait_fraction = _evidence_float(starvation, "data_wait_fraction")
        read_kb = _evidence_float(small_files, "mean_avg_read_size_kb")
        iops = _evidence_float(small_files, "mean_read_iops")
        score = 0
        if wait_fraction is not None:
            score += 2 if wait_fraction >= 0.40 else 1 if wait_fraction >= 0.15 else 0
        if read_kb is not None:
            score += 2 if read_kb <= 16.0 else 1 if read_kb <= 64.0 else 0
        if iops is not None:
            score += 2 if iops >= 10_000 else 1 if iops >= 500 else 0
        chain = []
        if wait_fraction is not None:
            chain.append(f"training waits on input for {_fmt_percent(wait_fraction)} of step time")
        if read_kb is not None and iops is not None:
            chain.append(f"reads average {read_kb:.1f} KiB at {iops:.0f} IOPS")
        chain.append("dataset layout is creating high per-request/metadata overhead")
        if disk is not None:
            disk_util = _evidence_float(disk, "mean_disk_util_pct")
            if disk_util is not None:
                chain.append(f"disk utilization is also elevated at {disk_util:.1f}%, consistent with downstream I/O pressure")

        supporting = ["DPD-001", "DPD-006"]
        if disk is not None:
            supporting.append("DPD-002")

        return Diagnosis(
            summary="small-file I/O overhead is starving the training loop",
            confidence=_confidence(score),
            diagnosis_type="root_cause" if score >= 5 else "contributing_factor",
            chain=tuple(chain),
            supporting_rule_ids=tuple(supporting),
            evidence={
                "data_wait_fraction": wait_fraction,
                "mean_avg_read_size_kb": read_kb,
                "mean_read_iops": iops,
                "correlation_score": score,
            },
        )

    if starvation is not None and disk is not None:
        wait_fraction = _evidence_float(starvation, "data_wait_fraction")
        disk_util = _evidence_float(disk, "mean_disk_util_pct")
        await_ms = _evidence_float(disk, "mean_avg_await_ms")
        score = 0
        if wait_fraction is not None:
            score += 2 if wait_fraction >= 0.40 else 1 if wait_fraction >= 0.15 else 0
        if disk_util is not None:
            score += 2 if disk_util >= 95.0 else 1 if disk_util >= 80.0 else 0
        if await_ms is not None and await_ms >= 2.0:
            score += 1

        chain = []
        if wait_fraction is not None:
            chain.append(f"training waits on input for {_fmt_percent(wait_fraction)} of step time")
        if disk_util is not None:
            chain.append(f"backing block device averages {disk_util:.1f}% utilization")
        if await_ms is not None:
            chain.append(f"mean I/O await is {await_ms:.1f} ms")

        return Diagnosis(
            summary="block-storage saturation is starving the training loop",
            confidence=_confidence(score),
            diagnosis_type="root_cause" if score >= 4 else "contributing_factor",
            chain=tuple(chain),
            supporting_rule_ids=("DPD-001", "DPD-002"),
            evidence={
                "data_wait_fraction": wait_fraction,
                "mean_disk_util_pct": disk_util,
                "mean_avg_await_ms": await_ms,
                "correlation_score": score,
            },
        )

    if restarts is not None:
        restart_count = _evidence_float(restarts, "worker_restarts_in_window")
        return Diagnosis(
            summary="DataLoader worker instability is disrupting batch delivery",
            confidence="medium",
            diagnosis_type="symptom",
            chain=(
                "worker process identities changed during the diagnostic window",
                "batch delivery can stall while the worker pool recovers",
                "inspect OOM/exception/remote-I/O evidence to determine the actual root cause",
            ),
            supporting_rule_ids=("DPD-010",),
            evidence={"worker_restarts_in_window": restart_count},
        )

    return None
