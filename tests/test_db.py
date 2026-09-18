from datapath_doctor.db import HistoryDB
from datapath_doctor.models import Finding, Severity


def _finding(rule_id="DPD-001", severity=Severity.WARNING, window_start=0.0, window_end=1.0):
    return Finding(
        rule_id=rule_id,
        severity=severity,
        title="t",
        message="m",
        suggested_fix="f",
        window_start=window_start,
        window_end=window_end,
    )


def test_record_and_retrieve_run(tmp_path):
    db = HistoryDB(tmp_path / "history.db")
    run_id = db.record_run([_finding()], job_name="unit-test", num_samples=100)

    runs = db.recent_runs()
    assert len(runs) == 1
    assert runs[0].run_id == run_id
    assert runs[0].job_name == "unit-test"
    assert runs[0].num_samples == 100

    findings = db.findings_for_run(run_id)
    assert len(findings) == 1
    assert findings[0]["rule_id"] == "DPD-001"


def test_run_with_no_findings_is_still_recorded(tmp_path):
    db = HistoryDB(tmp_path / "history.db")
    run_id = db.record_run([], job_name="healthy-run", num_samples=50)
    runs = db.recent_runs()
    assert len(runs) == 1
    assert db.findings_for_run(run_id) == []


def test_rule_trends_computes_fire_rate(tmp_path):
    db = HistoryDB(tmp_path / "history.db")
    # DPD-001 fires in 2 of 3 runs; DPD-002 fires in 1 of 3.
    db.record_run([_finding(rule_id="DPD-001", severity=Severity.WARNING)], num_samples=10)
    db.record_run([_finding(rule_id="DPD-001", severity=Severity.CRITICAL)], num_samples=10)
    db.record_run([_finding(rule_id="DPD-002", severity=Severity.WARNING)], num_samples=10)

    trends = {t.rule_id: t for t in db.rule_trends()}
    assert trends["DPD-001"].runs_fired == 2
    assert trends["DPD-001"].runs_seen == 3
    assert trends["DPD-001"].fire_rate == 2 / 3
    assert trends["DPD-001"].max_severity == "critical"
    assert trends["DPD-002"].fire_rate == 1 / 3


def test_rule_trends_empty_when_no_runs(tmp_path):
    db = HistoryDB(tmp_path / "history.db")
    assert db.rule_trends() == []
