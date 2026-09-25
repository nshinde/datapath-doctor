import io

import pytest

from datapath_doctor.collectors.remote import RemoteReadCollector
from datapath_doctor.models import StepSample, WindowSummary
from datapath_doctor.rules import NetworkFilesystemLatencySpikeRule


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds: float):
        self.value += seconds


def test_remote_collector_generates_step_sample_fields_without_block_device():
    clock = FakeClock()
    collector = RemoteReadCollector("s3", filesystem_type="fsspec", clock=clock)
    clock.advance(0.05)
    collector.record_read(0.05, bytes_read=1024, retries=1)
    clock.advance(0.95)

    reading = collector.sample(reset=False)
    assert reading["storage_backend"] == "s3"
    assert reading["filesystem_type"] == "fsspec"
    assert reading["remote_read_latency_ms"] == 50.0
    assert reading["remote_read_ops_per_s"] == pytest.approx(1.0)
    assert reading["remote_read_bytes_per_s"] == pytest.approx(1024.0)
    assert reading["remote_retries"] == 1

    sample = StepSample(t=1.0, step=1, **reading)
    findings = NetworkFilesystemLatencySpikeRule().evaluate(WindowSummary([sample]))
    assert len(findings) == 1
    assert findings[0].evidence["latency_source"] == "remote-read telemetry"


def test_measure_read_infers_bytes():
    clock = FakeClock()
    collector = RemoteReadCollector("http", clock=clock)

    def read():
        clock.advance(0.025)
        return b"x" * 100

    assert collector.measure_read(read) == b"x" * 100
    reading = collector.sample(reset=False)
    assert reading["remote_read_latency_ms"] == 25.0


def test_wrapped_file_measures_reads():
    clock = FakeClock()
    collector = RemoteReadCollector("s3", filesystem_type="fsspec", clock=clock)
    wrapped = collector.wrap_file(io.BytesIO(b"abcdef"))
    clock.advance(0.010)
    assert wrapped.read(3) == b"abc"
    reading = collector.sample(reset=False)
    assert reading["remote_read_latency_ms"] == 10.0
    assert reading["remote_read_bytes_per_s"] > 0


def test_errors_are_recorded_and_reraised():
    clock = FakeClock()
    collector = RemoteReadCollector("s3", clock=clock)

    def fail():
        clock.advance(0.02)
        raise OSError("remote read failed")

    with pytest.raises(OSError):
        collector.measure_read(fail, retries=2)

    reading = collector.sample(reset=False)
    assert reading["remote_errors"] == 1
    assert reading["remote_retries"] == 2
