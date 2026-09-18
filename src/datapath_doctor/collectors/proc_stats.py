"""Linux /proc telemetry collectors: block-device I/O and PSI (pressure
stall information). No third-party dependency beyond psutil for the device
list; the counters themselves are read straight from /proc so this works in
minimal containers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# /proc/diskstats field indices (see Documentation/admin-guide/iostats.rst).
# Only the fields we use are named; a 4.18+ kernel has more trailing fields
# but this collector only reads the guaranteed-present prefix.
_DISKSTATS_READS_COMPLETED = 3
_DISKSTATS_SECTORS_READ = 5
_DISKSTATS_READ_TICKS_MS = 6
_DISKSTATS_IO_IN_PROGRESS = 8
_DISKSTATS_IO_TICKS_MS = 9  # time spent doing I/Os (ms) - basis for utilization %
_SECTOR_SIZE_BYTES = 512


@dataclass
class _DiskCounters:
    t: float
    reads_completed: int
    sectors_read: int
    read_ticks_ms: int
    io_ticks_ms: int


class DiskIOCollector:
    """Polls /proc/diskstats for one block device and reports deltas
    (throughput, IOPS, utilization %, average await) between successive
    ``sample()`` calls.

    Usage::

        collector = DiskIOCollector("nvme0n1")
        collector.sample()          # primes the baseline, returns None
        time.sleep(1)
        reading = collector.sample()  # deltas over the last second
    """

    def __init__(self, device: str, diskstats_path: str = "/proc/diskstats") -> None:
        self.device = device
        self._path = Path(diskstats_path)
        self._prev: Optional[_DiskCounters] = None

    def _read_counters(self) -> Optional[_DiskCounters]:
        try:
            text = self._path.read_text()
        except OSError:
            return None

        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 14:
                continue
            if parts[2] != self.device:
                continue
            return _DiskCounters(
                t=time.time(),
                reads_completed=int(parts[_DISKSTATS_READS_COMPLETED]),
                sectors_read=int(parts[_DISKSTATS_SECTORS_READ]),
                read_ticks_ms=int(parts[_DISKSTATS_READ_TICKS_MS]),
                io_ticks_ms=int(parts[_DISKSTATS_IO_TICKS_MS]),
            )
        return None

    def sample(self) -> Optional[dict]:
        cur = self._read_counters()
        if cur is None:
            return None
        prev, self._prev = self._prev, cur
        if prev is None:
            return None

        dt = cur.t - prev.t
        if dt <= 0:
            return None

        d_reads = cur.reads_completed - prev.reads_completed
        d_sectors = cur.sectors_read - prev.sectors_read
        d_read_ticks = cur.read_ticks_ms - prev.read_ticks_ms
        d_io_ticks = cur.io_ticks_ms - prev.io_ticks_ms

        read_bytes_per_s = (d_sectors * _SECTOR_SIZE_BYTES) / dt
        read_iops = d_reads / dt
        util_pct = min(100.0, 100.0 * d_io_ticks / (dt * 1000.0))
        avg_await_ms = (d_read_ticks / d_reads) if d_reads > 0 else 0.0
        avg_read_size_bytes = (d_sectors * _SECTOR_SIZE_BYTES / d_reads) if d_reads > 0 else None

        return {
            "disk_name": self.device,
            "disk_read_bytes_per_s": read_bytes_per_s,
            "disk_read_iops": read_iops,
            "disk_util_pct": util_pct,
            "disk_avg_await_ms": avg_await_ms,
            "avg_read_size_bytes": avg_read_size_bytes,
        }


class PSICollector:
    """Reads /proc/pressure/io (Linux PSI, kernel >= 4.20). Returns None on
    systems without PSI enabled (e.g. some container runtimes, WSL, or a
    kernel built without CONFIG_PSI) rather than raising, since datapath-doctor
    should degrade gracefully wherever it's dropped in.
    """

    def __init__(self, path: str = "/proc/pressure/io") -> None:
        self._path = Path(path)

    def sample(self) -> Optional[dict]:
        try:
            text = self._path.read_text()
        except OSError:
            return None

        result = {}
        for line in text.splitlines():
            # format: "some avg10=0.00 avg60=0.00 avg300=0.00 total=12345"
            parts = line.split()
            if not parts:
                continue
            kind = parts[0]  # "some" or "full"
            fields = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
            avg10 = fields.get("avg10")
            if avg10 is None:
                continue
            result[f"psi_io_{kind}_avg10"] = float(avg10)

        return result or None


def list_block_devices() -> list[str]:
    """Best-effort list of real (non-partition, non-loop) block devices."""
    try:
        import psutil

        return sorted({p.device.rsplit("/", 1)[-1] for p in psutil.disk_partitions(all=False)})
    except Exception:  # noqa: BLE001
        return []
