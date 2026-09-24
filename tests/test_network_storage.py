from datapath_doctor.models import StepSample, WindowSummary
from datapath_doctor.rules import NetworkFilesystemLatencySpikeRule


def _window(**fields):
    samples = []
    for i in range(10):
        kwargs = {name: values[i] for name, values in fields.items()}
        samples.append(StepSample(t=float(i), step=i, **kwargs))
    return WindowSummary(samples=samples)


def test_network_rule_prefers_normalized_remote_latency():
    window = _window(
        storage_backend=["nfs"] * 10,
        filesystem_type=["nfs4"] * 10,
        remote_read_latency_ms=[60.0] * 10,
        disk_avg_await_ms=[2.0] * 10,
        remote_read_bytes_per_s=[20e6] * 10,
    )
    finding = NetworkFilesystemLatencySpikeRule().evaluate(window)[0]
    assert finding.evidence["mean_remote_read_latency_ms"] == 60.0
    assert finding.evidence["latency_source"] == "remote-read telemetry"


def test_network_rule_accepts_backend_identity_without_legacy_boolean():
    window = _window(
        storage_backend=["lustre"] * 10,
        filesystem_type=["lustre"] * 10,
        remote_read_latency_ms=[20.0] * 10,
    )
    assert len(NetworkFilesystemLatencySpikeRule().evaluate(window)) == 1


def test_network_rule_does_not_treat_local_backend_as_remote():
    window = _window(
        storage_backend=["local_nvme"] * 10,
        filesystem_type=["ext4"] * 10,
        disk_avg_await_ms=[80.0] * 10,
    )
    assert NetworkFilesystemLatencySpikeRule().evaluate(window) == []
