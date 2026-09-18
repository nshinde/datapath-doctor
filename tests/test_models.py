import pytest

from datapath_doctor.models import StepSample, WindowSummary


def test_window_summary_requires_samples():
    with pytest.raises(ValueError):
        WindowSummary(samples=[])


def test_window_summary_sorts_by_time():
    s1 = StepSample(t=2.0, step=1)
    s2 = StepSample(t=1.0, step=0)
    window = WindowSummary(samples=[s1, s2])
    assert window.samples[0] is s2
    assert window.samples[1] is s1


def test_mean_ignores_none_fields():
    samples = [
        StepSample(t=0, step=0, disk_util_pct=50.0),
        StepSample(t=1, step=1, disk_util_pct=None),
        StepSample(t=2, step=2, disk_util_pct=70.0),
    ]
    window = WindowSummary(samples=samples)
    assert window.mean("disk_util_pct") == 60.0


def test_mean_returns_none_when_no_values():
    samples = [StepSample(t=0, step=0)]
    window = WindowSummary(samples=samples)
    assert window.mean("disk_util_pct") is None


def test_fraction_true():
    samples = [
        StepSample(t=0, step=0, checkpoint_blocking=True),
        StepSample(t=1, step=1, checkpoint_blocking=False),
        StepSample(t=2, step=2, checkpoint_blocking=False),
        StepSample(t=3, step=3, checkpoint_blocking=None),
    ]
    window = WindowSummary(samples=samples)
    assert window.fraction_true("checkpoint_blocking") == pytest.approx(1 / 3)


def test_last_returns_most_recent_non_none():
    samples = [
        StepSample(t=0, step=0, disk_name="a"),
        StepSample(t=1, step=1, disk_name=None),
        StepSample(t=2, step=2, disk_name="b"),
    ]
    window = WindowSummary(samples=samples)
    assert window.last("disk_name") == "b"

    samples2 = [
        StepSample(t=0, step=0, disk_name="a"),
        StepSample(t=1, step=1, disk_name=None),
    ]
    assert WindowSummary(samples=samples2).last("disk_name") == "a"
