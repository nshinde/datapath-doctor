"""Command-line entry point: `datapath-doctor`.

    datapath-doctor demo --scenario network_fs_latency
    datapath-doctor demo --scenario healthy --no-save
    datapath-doctor trends
    datapath-doctor runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from datapath_doctor.collectors.synthetic import SCENARIOS, generate
from datapath_doctor.db import HistoryDB
from datapath_doctor.engine import RuleEngine
from datapath_doctor.report import render_findings, render_trends


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="datapath-doctor", description=__doc__)
    parser.add_argument(
        "--db", type=Path, default=None, help="path to the history SQLite DB (default: ~/.datapath-doctor/history.db)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the rule engine against a synthetic telemetry scenario")
    demo.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default="healthy",
        help="which synthetic input-path scenario to generate",
    )
    demo.add_argument("--steps", type=int, default=None, help="override number of synthetic steps")
    demo.add_argument("--seed", type=int, default=None, help="override RNG seed")
    demo.add_argument("--no-save", action="store_true", help="don't persist this run to history")
    demo.add_argument("--job-name", type=str, default=None)

    sub.add_parser("runs", help="list recent recorded runs")

    trends = sub.add_parser("trends", help="show rule fire-rate trends across recent runs")
    trends.add_argument("--limit", type=int, default=50, help="number of recent runs to consider")

    rules = sub.add_parser("rules", help="list all registered diagnostic rules")

    return parser


def _cmd_demo(args: argparse.Namespace) -> int:
    samples = generate(args.scenario, n=args.steps, seed=args.seed)
    engine = RuleEngine.with_default_rules()
    findings = engine.run(samples)

    print(render_findings(findings, num_samples=len(samples)))

    if not args.no_save:
        db = HistoryDB(args.db)
        run_id = db.record_run(
            findings,
            job_name=args.job_name or f"demo:{args.scenario}",
            num_samples=len(samples),
        )
        print(f"\n(saved as run {run_id} in {db.path})")

    return 0


def _cmd_runs(args: argparse.Namespace) -> int:
    db = HistoryDB(args.db)
    runs = db.recent_runs()
    if not runs:
        print("No runs recorded yet.")
        return 0
    for r in runs:
        print(f"{r.run_id}  job={r.job_name or '-'}  samples={r.num_samples}  started_at={r.started_at:.0f}")
    return 0


def _cmd_trends(args: argparse.Namespace) -> int:
    db = HistoryDB(args.db)
    trends = db.rule_trends(limit_runs=args.limit)
    print(render_trends(trends))
    return 0


def _cmd_rules(args: argparse.Namespace) -> int:
    engine = RuleEngine.with_default_rules()
    for rule in sorted(engine.rules, key=lambda r: r.rule_id):
        print(f"{rule.rule_id}  {rule.name}")
        print(f"           {rule.description}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "demo": _cmd_demo,
        "runs": _cmd_runs,
        "trends": _cmd_trends,
        "rules": _cmd_rules,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
