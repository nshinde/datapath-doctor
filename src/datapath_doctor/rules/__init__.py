"""Built-in rule registry.

Ten rules at launch (DPD-001 .. DPD-010), covering the storage/input-path
layer the way nccl-doctor's 17 rules cover the collective-communication
layer. Rule IDs are stable and never reused — new rules append at the next
number rather than reusing a retired one.
"""

from datapath_doctor.rules.checkpoint_io_blocking import CheckpointIOBlockingRule
from datapath_doctor.rules.disk_saturation import DiskThroughputSaturationRule
from datapath_doctor.rules.network_fs_latency import NetworkFilesystemLatencySpikeRule
from datapath_doctor.rules.prefetch_underrun import PrefetchBufferUnderrunRule
from datapath_doctor.rules.psi_pressure import SystemIOPressureRule
from datapath_doctor.rules.shuffle_buffer_stall import ShuffleBufferStallRule
from datapath_doctor.rules.small_file_overhead import SmallFileIOOverheadRule
from datapath_doctor.rules.worker_cpu_bound import WorkerPoolCPUBoundRule
from datapath_doctor.rules.worker_restarts import WorkerCrashRestartLoopRule
from datapath_doctor.rules.worker_starvation import GpuStarvedByDataPipelineRule

_ALL_RULE_CLASSES = [
    GpuStarvedByDataPipelineRule,
    DiskThroughputSaturationRule,
    SystemIOPressureRule,
    PrefetchBufferUnderrunRule,
    WorkerPoolCPUBoundRule,
    SmallFileIOOverheadRule,
    NetworkFilesystemLatencySpikeRule,
    ShuffleBufferStallRule,
    CheckpointIOBlockingRule,
    WorkerCrashRestartLoopRule,
]


def all_rules():
    """Return the list of built-in rule classes (not instances)."""
    return list(_ALL_RULE_CLASSES)


__all__ = [
    "all_rules",
    "GpuStarvedByDataPipelineRule",
    "DiskThroughputSaturationRule",
    "SystemIOPressureRule",
    "PrefetchBufferUnderrunRule",
    "WorkerPoolCPUBoundRule",
    "SmallFileIOOverheadRule",
    "NetworkFilesystemLatencySpikeRule",
    "ShuffleBufferStallRule",
    "CheckpointIOBlockingRule",
    "WorkerCrashRestartLoopRule",
]
