"""SQLite-backed cross-run history, matching nccl-doctor's approach: every
diagnostic run is persisted so the same rule firing repeatedly (or a new
rule firing for the first time) is visible as a trend, not just a one-off
report on stdout.

Default location is ``~/.datapath-doctor/history.db``, overridable for tests
or multi-tenant setups.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datapath_doctor.models import Finding, Severity

DEFAULT_DB_PATH = Path.home() / ".datapath-doctor" / "history.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    job_name TEXT,
    hostname TEXT,
    num_samples INTEGER,
    window_start REAL,
    window_end REAL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    rule_id TEXT NOT NULL,
    severity TEXT NOT NULL,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    suggested_fix TEXT,
    window_start REAL,
    window_end REAL
);

CREATE INDEX IF NOT EXISTS idx_findings_run_id ON findings(run_id);
CREATE INDEX IF NOT EXISTS idx_findings_rule_id ON findings(rule_id);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);
"""


@dataclass
class RunRecord:
    run_id: str
    started_at: float
    job_name: Optional[str]
    hostname: Optional[str]
    num_samples: Optional[int]
    window_start: Optional[float]
    window_end: Optional[float]
    notes: Optional[str]


@dataclass
class RuleTrend:
    rule_id: str
    runs_seen: int
    runs_fired: int
    fire_rate: float
    last_fired_at: Optional[float]
    max_severity: Optional[str]


class HistoryDB:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)

    def record_run(
        self,
        findings: list[Finding],
        *,
        job_name: Optional[str] = None,
        hostname: Optional[str] = None,
        num_samples: Optional[int] = None,
        window_start: Optional[float] = None,
        window_end: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> str:
        run_id = uuid.uuid4().hex
        started_at = time.time()

        if window_start is None and findings:
            window_start = min(f.window_start for f in findings if f.window_start is not None)
        if window_end is None and findings:
            window_end = max(f.window_end for f in findings if f.window_end is not None)

        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO runs (run_id, started_at, job_name, hostname, num_samples, "
                "window_start, window_end, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, started_at, job_name, hostname, num_samples, window_start, window_end, notes),
            )
            conn.executemany(
                "INSERT INTO findings (run_id, rule_id, severity, title, message, "
                "evidence_json, suggested_fix, window_start, window_end) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        run_id,
                        f.rule_id,
                        f.severity.value,
                        f.title,
                        f.message,
                        json.dumps(f.evidence),
                        f.suggested_fix,
                        f.window_start,
                        f.window_end,
                    )
                    for f in findings
                ],
            )
        return run_id

    def recent_runs(self, limit: int = 20) -> list[RunRecord]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            RunRecord(
                run_id=r["run_id"],
                started_at=r["started_at"],
                job_name=r["job_name"],
                hostname=r["hostname"],
                num_samples=r["num_samples"],
                window_start=r["window_start"],
                window_end=r["window_end"],
                notes=r["notes"],
            )
            for r in rows
        ]

    def findings_for_run(self, run_id: str) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM findings WHERE run_id = ? ORDER BY severity DESC", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def rule_trends(self, limit_runs: int = 50) -> list[RuleTrend]:
        """Across the last ``limit_runs`` runs, how often did each rule_id
        that has ever fired, fire — and at what severity."""
        with closing(self._connect()) as conn:
            run_ids = [
                r["run_id"]
                for r in conn.execute(
                    "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT ?", (limit_runs,)
                ).fetchall()
            ]
            if not run_ids:
                return []

            placeholders = ",".join("?" for _ in run_ids)
            rows = conn.execute(
                f"SELECT rule_id, severity, run_id, window_end FROM findings "
                f"WHERE run_id IN ({placeholders})",
                run_ids,
            ).fetchall()

        by_rule: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            by_rule.setdefault(row["rule_id"], []).append(row)

        severity_rank = {"info": 0, "warning": 1, "critical": 2}
        trends = []
        for rule_id, fired_rows in by_rule.items():
            fired_run_ids = {r["run_id"] for r in fired_rows}
            max_sev = max((r["severity"] for r in fired_rows), key=lambda s: severity_rank[s])
            last_fired_at = max((r["window_end"] or 0) for r in fired_rows)
            trends.append(
                RuleTrend(
                    rule_id=rule_id,
                    runs_seen=len(run_ids),
                    runs_fired=len(fired_run_ids),
                    fire_rate=len(fired_run_ids) / len(run_ids),
                    last_fired_at=last_fired_at or None,
                    max_severity=max_sev,
                )
            )

        trends.sort(key=lambda t: t.fire_rate, reverse=True)
        return trends
