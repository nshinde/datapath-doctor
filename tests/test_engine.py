from datapath_doctor.engine import Rule, RuleEngine
from datapath_doctor.models import Finding, Severity, StepSample


class _AlwaysFiresRule(Rule):
    rule_id = "TEST-ALWAYS"
    name = "always fires"

    def evaluate(self, window):
        return [
            Finding(
                rule_id=self.rule_id,
                severity=Severity.INFO,
                title="t",
                message="m",
            )
        ]


class _NeverFiresRule(Rule):
    rule_id = "TEST-NEVER"
    name = "never fires"

    def evaluate(self, window):
        return []


class _BrokenRule(Rule):
    rule_id = "TEST-BROKEN"
    name = "raises"

    def evaluate(self, window):
        raise RuntimeError("boom")


def _sample(step: int) -> StepSample:
    return StepSample(t=float(step), step=step, step_time_s=0.5, data_wait_s=0.01)


def test_engine_runs_all_registered_rules():
    engine = RuleEngine(rules=[_AlwaysFiresRule(), _NeverFiresRule()])
    findings = engine.run([_sample(0), _sample(1)])
    assert len(findings) == 1
    assert findings[0].rule_id == "TEST-ALWAYS"


def test_engine_survives_a_broken_rule():
    engine = RuleEngine(rules=[_BrokenRule(), _AlwaysFiresRule()])
    findings = engine.run([_sample(0)])
    assert len(findings) == 1
    assert findings[0].rule_id == "TEST-ALWAYS"


def test_engine_empty_samples_returns_no_findings():
    engine = RuleEngine(rules=[_AlwaysFiresRule()])
    assert engine.run([]) == []


def test_duplicate_rule_id_rejected():
    engine = RuleEngine()
    engine.register(_AlwaysFiresRule())
    try:
        engine.register(_AlwaysFiresRule())
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_findings_sorted_by_severity_descending():
    class _Warn(Rule):
        rule_id = "TEST-WARN"
        name = "warn"

        def evaluate(self, window):
            return [Finding(rule_id=self.rule_id, severity=Severity.WARNING, title="w", message="m")]

    class _Crit(Rule):
        rule_id = "TEST-CRIT"
        name = "crit"

        def evaluate(self, window):
            return [Finding(rule_id=self.rule_id, severity=Severity.CRITICAL, title="c", message="m")]

    engine = RuleEngine(rules=[_Warn(), _Crit()])
    findings = engine.run([_sample(0)])
    assert [f.rule_id for f in findings] == ["TEST-CRIT", "TEST-WARN"]


def test_with_default_rules_registers_ten_builtin_rules():
    engine = RuleEngine.with_default_rules()
    assert len(engine.rules) == 10
    ids = {r.rule_id for r in engine.rules}
    assert ids == {f"DPD-{i:03d}" for i in range(1, 11)}
