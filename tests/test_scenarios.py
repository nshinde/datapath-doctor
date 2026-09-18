"""End-to-end tests: each synthetic scenario should trip the rule it was
designed to demonstrate, and the healthy scenario should trip nothing."""

from datapath_doctor.collectors.synthetic import generate
from datapath_doctor.engine import RuleEngine


def _fired_ids(scenario: str) -> set[str]:
    samples = generate(scenario, n=150, seed=42)
    engine = RuleEngine.with_default_rules()
    findings = engine.run(samples)
    return {f.rule_id for f in findings}


def test_healthy_pipeline_triggers_nothing():
    assert _fired_ids("healthy") == set()


def test_network_fs_latency_scenario_triggers_expected_rules():
    fired = _fired_ids("network_fs_latency")
    assert "DPD-001" in fired  # overall starvation
    assert "DPD-007" in fired  # network FS latency specifically


def test_cpu_bound_decode_scenario_triggers_worker_cpu_rule():
    fired = _fired_ids("cpu_bound_decode")
    assert "DPD-005" in fired
    # disk isn't the problem here, so the disk-saturation rule should stay quiet
    assert "DPD-002" not in fired


def test_small_file_storm_scenario_triggers_small_file_rule():
    fired = _fired_ids("small_file_storm")
    assert "DPD-006" in fired


def test_checkpoint_stall_scenario_triggers_checkpoint_rule():
    fired = _fired_ids("checkpoint_stall")
    assert "DPD-009" in fired


def test_worker_crash_loop_scenario_triggers_restart_rule():
    fired = _fired_ids("worker_crash_loop")
    assert "DPD-010" in fired
