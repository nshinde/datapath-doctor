from datapath_doctor.models import Severity, StepSample, WindowSummary
from datapath_doctor.rules import (
    CheckpointIOBlockingRule,
    DiskThroughputSaturationRule,
    GpuStarvedByDataPipelineRule,
    NetworkFilesystemLatencySpikeRule,
    PrefetchBufferUnderrunRule,
    ShuffleBufferStallRule,
    SmallFileIOOverheadRule,
    SystemIOPressureRule,
    WorkerCrashRestartLoopRule,
    WorkerPoolCPUBoundRule,
)


def _window(**kwargs_list):
    """Build a WindowSummary from a dict of field -> list of values (all
    lists must be the same length); t/step are filled in automatically."""
    n = len(next(iter(kwargs_list.values())))
    samples = []
    for i in range(n):
        fields = {k: v[i] for k, v in kwargs_list.items()}
        samples.append(StepSample(t=float(i), step=i, **fields))
    return WindowSummary(samples=samples)


def test_gpu_starved_fires_above_threshold():
    window = _window(step_time_s=[1.0] * 20, data_wait_s=[0.5] * 20)
    findings = GpuStarvedByDataPipelineRule().evaluate(window)
    assert len(findings) == 1
    assert findings[0].severity == Severity.CRITICAL


def test_gpu_starved_silent_when_healthy():
    window = _window(step_time_s=[1.0] * 20, data_wait_s=[0.02] * 20)
    assert GpuStarvedByDataPipelineRule().evaluate(window) == []


def test_disk_saturation_fires_on_high_util():
    window = _window(
        disk_util_pct=[97.0] * 10,
        disk_avg_await_ms=[5.0] * 10,
        disk_read_bytes_per_s=[1e8] * 10,
    )
    findings = DiskThroughputSaturationRule().evaluate(window)
    assert len(findings) == 1
    assert findings[0].severity == Severity.CRITICAL


def test_disk_saturation_silent_when_underutilized():
    window = _window(disk_util_pct=[20.0] * 10)
    assert DiskThroughputSaturationRule().evaluate(window) == []


def test_psi_pressure_fires_on_high_full_avg10():
    window = _window(psi_io_full_avg10=[25.0] * 10, psi_io_some_avg10=[30.0] * 10)
    findings = SystemIOPressureRule().evaluate(window)
    assert len(findings) == 1
    assert findings[0].severity == Severity.CRITICAL


def test_psi_pressure_absent_field_does_not_fire():
    window = _window(step_time_s=[1.0] * 5)
    assert SystemIOPressureRule().evaluate(window) == []


def test_prefetch_underrun_fires_when_queue_empty():
    window = _window(queue_depth=[0] * 20, queue_capacity=[32] * 20)
    findings = PrefetchBufferUnderrunRule().evaluate(window)
    assert len(findings) == 1


def test_prefetch_underrun_silent_when_queue_full():
    window = _window(queue_depth=[30] * 20, queue_capacity=[32] * 20)
    assert PrefetchBufferUnderrunRule().evaluate(window) == []


def test_worker_cpu_bound_fires_when_pegged_and_disk_idle():
    window = _window(worker_cpu_pct=[99.0] * 10, disk_util_pct=[5.0] * 10)
    findings = WorkerPoolCPUBoundRule().evaluate(window)
    assert len(findings) == 1


def test_worker_cpu_bound_defers_to_disk_rule_when_disk_also_busy():
    window = _window(worker_cpu_pct=[99.0] * 10, disk_util_pct=[85.0] * 10)
    assert WorkerPoolCPUBoundRule().evaluate(window) == []


def test_small_file_overhead_fires_on_tiny_reads_high_iops():
    window = _window(avg_read_size_bytes=[4000.0] * 10, disk_read_iops=[10000.0] * 10)
    findings = SmallFileIOOverheadRule().evaluate(window)
    assert len(findings) == 1
    assert findings[0].severity == Severity.CRITICAL


def test_small_file_overhead_silent_on_large_reads():
    window = _window(avg_read_size_bytes=[500_000.0] * 10, disk_read_iops=[10000.0] * 10)
    assert SmallFileIOOverheadRule().evaluate(window) == []


def test_small_file_overhead_silent_on_low_iops_even_if_small_reads():
    window = _window(avg_read_size_bytes=[1000.0] * 10, disk_read_iops=[10.0] * 10)
    assert SmallFileIOOverheadRule().evaluate(window) == []


def test_network_fs_latency_fires_only_when_flagged_network_fs():
    window_net = _window(is_network_fs=[True] * 10, disk_avg_await_ms=[60.0] * 10)
    assert len(NetworkFilesystemLatencySpikeRule().evaluate(window_net)) == 1

    window_local = _window(is_network_fs=[False] * 10, disk_avg_await_ms=[60.0] * 10)
    assert NetworkFilesystemLatencySpikeRule().evaluate(window_local) == []


def test_shuffle_buffer_stall_fires_when_near_empty():
    window = _window(
        shuffle_buffer_size=[100] * 10,
        shuffle_buffer_capacity=[10000] * 10,
        shuffle_refill_event=[True] * 5 + [False] * 5,
    )
    findings = ShuffleBufferStallRule().evaluate(window)
    assert len(findings) == 1


def test_shuffle_buffer_silent_when_well_filled():
    window = _window(
        shuffle_buffer_size=[9500] * 10,
        shuffle_buffer_capacity=[10000] * 10,
        shuffle_refill_event=[False] * 10,
    )
    assert ShuffleBufferStallRule().evaluate(window) == []


def test_checkpoint_blocking_fires_on_long_blocking_writes():
    window = _window(
        checkpoint_blocking=[False] * 9 + [True],
        checkpoint_write_s=[None] * 9 + [10.0],
        step_time_s=[0.5] * 9 + [10.5],
    )
    findings = CheckpointIOBlockingRule().evaluate(window)
    assert len(findings) == 1
    assert findings[0].severity == Severity.CRITICAL


def test_checkpoint_blocking_silent_on_fast_checkpoints():
    window = _window(
        checkpoint_blocking=[False] * 9 + [True],
        checkpoint_write_s=[None] * 9 + [0.2],
        step_time_s=[0.5] * 10,
    )
    assert CheckpointIOBlockingRule().evaluate(window) == []


def test_worker_restarts_fires_on_delta():
    window = _window(worker_restarts=[0] * 5 + [3] * 5)
    findings = WorkerCrashRestartLoopRule().evaluate(window)
    assert len(findings) == 1


def test_worker_restarts_silent_when_flat():
    window = _window(worker_restarts=[2] * 10)
    assert WorkerCrashRestartLoopRule().evaluate(window) == []
