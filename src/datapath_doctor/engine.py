"""Rule base class and the engine that runs a rule set over a telemetry window.

Same shape as nccl-doctor's engine: rules are small, independent, declare
their own id/name, and are run against one WindowSummary at a time. The
engine's only job is to run every registered rule, catch a misbehaving rule
so it can't take down the whole diagnostic pass, and return findings sorted
by severity.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Iterable, Optional

from datapath_doctor.models import Finding, StepSample, WindowSummary

logger = logging.getLogger("datapath_doctor.engine")


class Rule(ABC):
    """Base class for a single diagnostic rule.

    Subclasses set ``rule_id`` (stable, used as a primary key in the SQLite
    history — never rename once released) and ``name``, and implement
    ``evaluate``. A rule may return zero, one, or multiple Findings (e.g. one
    per disk device).
    """

    rule_id: str = ""
    name: str = ""
    description: str = ""

    def __init__(self) -> None:
        if not self.rule_id:
            raise ValueError(f"{type(self).__name__} must set a non-empty rule_id")

    @abstractmethod
    def evaluate(self, window: WindowSummary) -> list[Finding]:
        """Inspect the window and return any findings. Must not raise for
        missing/None telemetry fields — just decline to fire."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Rule {self.rule_id} ({self.name})>"


class RuleEngine:
    """Holds a registry of rules and runs them all against a window."""

    def __init__(self, rules: Optional[Iterable[Rule]] = None) -> None:
        self._rules: dict[str, Rule] = {}
        if rules:
            for r in rules:
                self.register(r)

    def register(self, rule: Rule) -> None:
        if rule.rule_id in self._rules:
            raise ValueError(f"duplicate rule_id: {rule.rule_id!r}")
        self._rules[rule.rule_id] = rule

    @property
    def rules(self) -> list[Rule]:
        return list(self._rules.values())

    def run(self, samples: list[StepSample]) -> list[Finding]:
        """Evaluate every registered rule against the given samples.

        A single window is used for the whole batch of rules so they all see
        a consistent view and don't re-derive the same aggregates.
        """
        if not samples:
            return []

        window = WindowSummary(samples=samples)
        findings: list[Finding] = []

        for rule in self._rules.values():
            try:
                rule_findings = rule.evaluate(window) or []
            except Exception:  # noqa: BLE001 - one bad rule shouldn't kill the run
                logger.exception("rule %s raised during evaluate(); skipping", rule.rule_id)
                continue

            for f in rule_findings:
                f.window_start = f.window_start if f.window_start is not None else window.start_t
                f.window_end = f.window_end if f.window_end is not None else window.end_t
            findings.extend(rule_findings)

        findings.sort(key=lambda f: f.severity.rank, reverse=True)
        return findings

    @classmethod
    def with_default_rules(cls) -> "RuleEngine":
        """Convenience constructor wiring up the built-in rule set."""
        from datapath_doctor.rules import all_rules

        return cls(rules=[rule_cls() for rule_cls in all_rules()])
