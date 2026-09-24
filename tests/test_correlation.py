from datapath_doctor.correlation import correlate_findings
from datapath_doctor.models import Finding, Severity


def _finding(rule_id: str) -> Finding:
    return Finding(rule_id=rule_id, severity=Severity.WARNING, title=rule_id, message="test")


def test_network_storage_correlation_uses_multiple_signals():
    diagnosis = correlate_findings([
        _finding("DPD-001"),
        _finding("DPD-004"),
        _finding("DPD-007"),
    ])
    assert diagnosis is not None
    assert diagnosis.confidence == "high"
    assert diagnosis.supporting_rule_ids == ("DPD-001", "DPD-004", "DPD-007")
    assert "network storage latency" in diagnosis.root_cause


def test_network_storage_without_prefetch_evidence_is_medium_confidence():
    diagnosis = correlate_findings([_finding("DPD-001"), _finding("DPD-007")])
    assert diagnosis is not None
    assert diagnosis.confidence == "medium"


def test_cpu_bound_correlation():
    diagnosis = correlate_findings([_finding("DPD-001"), _finding("DPD-005")])
    assert diagnosis is not None
    assert diagnosis.confidence == "high"
    assert "CPU-bound" in diagnosis.root_cause


def test_unrelated_single_finding_does_not_invent_root_cause():
    assert correlate_findings([_finding("DPD-003")]) is None
