from datapath_doctor.models import Finding, Severity
from datapath_doctor.report import render_findings


def _finding(rule_id: str) -> Finding:
    return Finding(
        rule_id=rule_id,
        severity=Severity.WARNING,
        title=rule_id,
        message="test message",
    )


def test_report_renders_correlated_root_cause():
    out = render_findings(
        [_finding("DPD-001"), _finding("DPD-004"), _finding("DPD-007")],
        num_samples=100,
    )
    assert "LIKELY ROOT CAUSE" in out
    assert "network storage latency is starving the training loop" in out
    assert "Confidence: high" in out
    assert "DPD-001, DPD-004, DPD-007" in out


def test_report_does_not_invent_root_cause_for_unrelated_signal():
    out = render_findings([_finding("DPD-003")], num_samples=100)
    assert "LIKELY ROOT CAUSE" not in out
