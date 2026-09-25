from datapath_doctor.distributed import (
    aggregate_rank_stats,
    analyze_distributed,
    diagnose_rank_skew,
)
from datapath_doctor.models import StepSample


def _rank_samples(rank: int, wait_s: float, steps: int = 20):
    return [
        StepSample(
            t=float(step),
            step=step,
            rank=rank,
            local_rank=rank,
            world_size=4,
            node_id=f"node-{rank // 2}",
            step_time_s=0.4 + wait_s,
            compute_time_s=0.4,
            data_wait_s=wait_s,
        )
        for step in range(steps)
    ]


def test_rank_aggregation_preserves_rank_and_node():
    stats = aggregate_rank_stats(_rank_samples(0, 0.02) + _rank_samples(1, 0.03))
    assert [item.rank for item in stats] == [0, 1]
    assert stats[0].node_id == "node-0"
    assert stats[1].mean_data_wait_s == 0.03


def test_detects_input_path_straggler_rank():
    samples = (
        _rank_samples(0, 0.02)
        + _rank_samples(1, 0.02)
        + _rank_samples(2, 0.20)
        + _rank_samples(3, 0.025)
    )
    diagnosis = diagnose_rank_skew(samples)
    assert diagnosis is not None
    assert diagnosis.straggler_rank == 2
    assert diagnosis.slowdown_ratio > 7.0
    assert diagnosis.confidence == "high"


def test_similar_ranks_do_not_create_false_straggler():
    samples = _rank_samples(0, 0.02) + _rank_samples(1, 0.022)
    assert diagnose_rank_skew(samples) is None


def test_distributed_analysis_runs_rules_per_rank():
    samples = _rank_samples(0, 0.02) + _rank_samples(1, 0.50)
    analysis = analyze_distributed(samples)
    assert analysis.straggler is not None
    assert analysis.straggler.straggler_rank == 1
    assert not any(f.rule_id == "DPD-001" for f in analysis.per_rank_findings[0])
    assert any(f.rule_id == "DPD-001" for f in analysis.per_rank_findings[1])
