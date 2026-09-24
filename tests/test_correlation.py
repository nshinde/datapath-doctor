from datapath_doctor.correlation import correlate_findings
from datapath_doctor.models import Finding, Severity


def _finding(rule_id: str, **evidence) -> Finding:
    return Finding(
        rule_id=rule_id,
        severity=Severity.WARNING,
        title=rule_id,
        message="test",
        evidence=evidence,
    )


def test_network_storage_high_confidence_uses_measurements():
    diagnosis = correlate_findings(
        [
            _finding("DPD-001", data_wait_fraction=0.55),
            _finding("DPD-004", fraction_samples_near_empty=0.80),
            _finding("DPD-007", mean_remote_read_latency_ms=65.0),
        ]
    )
    assert diagnosis is not None
    assert diagnosis.confidence == "high"
    assert diagnosis.diagnosis_type == "root_cause"
    assert diagnosis.supporting_rule_ids == ("DPD-001", "DPD-004", "DPD-007")
    assert "remote input-storage latency" in diagnosis.summary
    assert diagnosis.evidence["remote_latency_ms"] == 65.0


def test_network_storage_weak_evidence_is_not_overclaimed():
    diagnosis = correlate_findings(
        [
            _finding("DPD-001", data_wait_fraction=0.18),
            _finding("DPD-007", mean_avg_await_ms=18.0),
        ]
    )
    assert diagnosis is not None
    assert diagnosis.confidence == "low"
    assert diagnosis.diagnosis_type == "contributing_factor"


def test_cpu_bound_correlation_uses_worker_and_disk_evidence():
    diagnosis = correlate_findings(
        [
            _finding("DPD-001", data_wait_fraction=0.50),
            _finding("DPD-005", mean_worker_cpu_pct=99.0, mean_disk_util_pct=10.0),
        ]
    )
    assert diagnosis is not None
    assert diagnosis.confidence == "high"
    assert diagnosis.diagnosis_type == "root_cause"
    assert "CPU-bound" in diagnosis.summary


def test_worker_restart_is_a_symptom_not_a_root_cause():
    diagnosis = correlate_findings([_finding("DPD-010", worker_restarts_in_window=3)])
    assert diagnosis is not None
    assert diagnosis.diagnosis_type == "symptom"
    assert diagnosis.summary.startswith("DataLoader worker instability")


def test_unrelated_single_finding_does_not_invent_root_cause():
    assert correlate_findings([_finding("DPD-003")]) is None
