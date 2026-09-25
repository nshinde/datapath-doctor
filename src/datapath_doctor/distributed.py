"""Cross-rank input-path aggregation for distributed training jobs."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional

from datapath_doctor.engine import RuleEngine
from datapath_doctor.models import Finding, StepSample


@dataclass(frozen=True)
class RankStats:
    rank: int
    node_id: Optional[str]
    num_samples: int
    mean_data_wait_s: float
    p95_data_wait_s: float
    mean_data_wait_fraction: Optional[float]


@dataclass(frozen=True)
class RankSkewDiagnosis:
    straggler_rank: int
    straggler_node_id: Optional[str]
    peer_median_wait_s: float
    straggler_mean_wait_s: float
    slowdown_ratio: float
    confidence: str
    message: str


@dataclass
class DistributedAnalysis:
    rank_stats: list[RankStats]
    per_rank_findings: dict[int, list[Finding]] = field(default_factory=dict)
    straggler: Optional[RankSkewDiagnosis] = None


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def group_by_rank(samples: list[StepSample]) -> dict[int, list[StepSample]]:
    """Group samples by global rank, ignoring samples without rank metadata."""
    grouped: dict[int, list[StepSample]] = {}
    for sample in samples:
        if sample.rank is None:
            continue
        grouped.setdefault(sample.rank, []).append(sample)
    return grouped


def aggregate_rank_stats(samples: list[StepSample]) -> list[RankStats]:
    """Summarize DataLoader wait behavior for every observed rank."""
    out: list[RankStats] = []
    for rank, rank_samples in sorted(group_by_rank(samples).items()):
        waits = [sample.data_wait_s for sample in rank_samples if sample.data_wait_s is not None]
        if not waits:
            continue

        wait_fractions = [
            sample.data_wait_s / sample.step_time_s
            for sample in rank_samples
            if sample.data_wait_s is not None
            and sample.step_time_s is not None
            and sample.step_time_s > 0
        ]
        node_id = next((sample.node_id for sample in rank_samples if sample.node_id), None)
        out.append(
            RankStats(
                rank=rank,
                node_id=node_id,
                num_samples=len(rank_samples),
                mean_data_wait_s=statistics.fmean(waits),
                p95_data_wait_s=_percentile(waits, 0.95),
                mean_data_wait_fraction=(
                    statistics.fmean(wait_fractions) if wait_fractions else None
                ),
            )
        )
    return out


def diagnose_rank_skew(
    samples: list[StepSample],
    *,
    ratio_threshold: float = 2.0,
    absolute_wait_threshold_s: float = 0.05,
) -> Optional[RankSkewDiagnosis]:
    """Identify a rank whose input wait is materially worse than its peers.

    This does not claim the slower rank directly stalls peers at a particular
    collective. It identifies the input-path straggler that can become the
    pacing rank once the distributed step synchronizes.
    """
    stats = aggregate_rank_stats(samples)
    if len(stats) < 2:
        return None

    straggler = max(stats, key=lambda item: item.mean_data_wait_s)
    peer_waits = [item.mean_data_wait_s for item in stats if item.rank != straggler.rank]
    peer_median = statistics.median(peer_waits)
    denominator = max(peer_median, 1e-6)
    ratio = straggler.mean_data_wait_s / denominator

    if straggler.mean_data_wait_s < absolute_wait_threshold_s:
        return None
    if ratio < ratio_threshold:
        return None

    confidence = "high" if ratio >= 4.0 else "medium"
    return RankSkewDiagnosis(
        straggler_rank=straggler.rank,
        straggler_node_id=straggler.node_id,
        peer_median_wait_s=peer_median,
        straggler_mean_wait_s=straggler.mean_data_wait_s,
        slowdown_ratio=ratio,
        confidence=confidence,
        message=(
            f"rank {straggler.rank} waits {straggler.mean_data_wait_s * 1000:.1f} ms "
            f"per step for input versus a {peer_median * 1000:.1f} ms peer median "
            f"({ratio:.1f}x slower)"
        ),
    )


def analyze_distributed(
    samples: list[StepSample],
    engine: Optional[RuleEngine] = None,
) -> DistributedAnalysis:
    """Run ordinary rules per rank and add a cross-rank straggler diagnosis."""
    grouped = group_by_rank(samples)
    rule_engine = engine or RuleEngine.with_default_rules()
    per_rank_findings = {
        rank: rule_engine.run(rank_samples)
        for rank, rank_samples in sorted(grouped.items())
    }
    return DistributedAnalysis(
        rank_stats=aggregate_rank_stats(samples),
        per_rank_findings=per_rank_findings,
        straggler=diagnose_rank_skew(samples),
    )
