"""Plain-text report rendering for findings — no external deps so it reads
fine over SSH from a training node's terminal."""

from __future__ import annotations

from datapath_doctor.correlation import correlate_findings
from datapath_doctor.models import Finding, Severity

_SEVERITY_LABEL = {
    Severity.CRITICAL: "[CRITICAL]",
    Severity.WARNING: "[WARNING] ",
    Severity.INFO: "[INFO]    ",
}


def render_findings(findings: list[Finding], num_samples: int) -> str:
    if not findings:
        return f"datapath-doctor: {num_samples} samples analyzed, no issues found. Input path looks healthy."

    lines = [f"datapath-doctor: {num_samples} samples analyzed, {len(findings)} finding(s):", ""]

    diagnosis = correlate_findings(findings)
    if diagnosis is not None:
        lines.append("LIKELY ROOT CAUSE")
        lines.append(f"  {diagnosis.root_cause}")
        lines.append(f"  Confidence: {diagnosis.confidence}")
        lines.append("  Evidence chain:")
        for i, step in enumerate(diagnosis.chain, start=1):
            lines.append(f"    {i}. {step}")
        lines.append(f"  Supporting rules: {', '.join(diagnosis.supporting_rule_ids)}")
        lines.append("")

    for f in findings:
        lines.append(f"{_SEVERITY_LABEL[f.severity]} {f.rule_id}  {f.title}")
        lines.append(f"           {f.message}")
        if f.suggested_fix:
            lines.append(f"           Fix: {f.suggested_fix}")
        lines.append("")
    return "\n".join(lines)


def render_trends(trends) -> str:
    if not trends:
        return "No history yet — run `datapath-doctor demo` or wire the collectors into a job first."

    lines = ["rule_id      fire_rate  runs_fired/seen  max_severity", "-" * 60]
    for t in trends:
        lines.append(
            f"{t.rule_id:<12} {t.fire_rate:>8.0%}  {t.runs_fired}/{t.runs_seen:<13} {t.max_severity}"
        )
    return "\n".join(lines)
