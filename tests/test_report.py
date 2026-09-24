from datapath_doctor.models import Finding, Severity
from datapath_doctor.report import render_findings


def _finding(rule_id: str, **evidence) -> Finding:
    return Finding(
        rule_id=rule_id,
        severity=Severity.WARNING,
        title=rule_id,
        message="test message",
        evidence=evidence,
    )


def test_report_renders_evidence_aware_root_cause():
    out = render_findings(
        [
            _finding("DPD-001", data_wait_fraction=0.55),
            _finding("DPD-004", fraction_samples_near_empty=0.80),
            _finding("DPD-007", mean_remote_read_latency_ms=65.0),
        ],
        num_samples=100,
    )
    assert "LIKELY ROOT CAUSE" in out
    assert "remote input-storage latency is starving the training loop" in out
    assert "Confidence: high" in out
    assert "65.0 ms" in out
    assert "DPD-001, DPD-004, DPD-007" in out


def test_report_labels_worker_restart_as_signal():
    out = render_findings(
        [_finding("DPD-010", worker_restarts_in_window=3)],
        num_samples=100,
    )
    assert "DIAGNOSTIC SIGNAL" in out
    assert "LIKELY ROOT CAUSE" not in out


def test_report_does_not_invent_root_cause_for_unrelated_signal():
    out = render_findings([_finding("DPD-003")], num_samples=100)
    assert "LIKELY ROOT CAUSE" not in out
