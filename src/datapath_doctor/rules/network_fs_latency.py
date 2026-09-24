from __future__ import annotations

from datapath_doctor.engine import Rule
from datapath_doctor.models import Finding, Severity, WindowSummary


class NetworkFilesystemLatencySpikeRule(Rule):
    """Flags elevated latency on explicitly network/remote-backed input storage.

    Prefer normalized remote-read latency when a protocol-specific collector
    provides it. Fall back to block-device await for compatibility, but do not
    claim that await alone identifies an NFS/Lustre/FUSE protocol-level cause.
    """

    rule_id = "DPD-007"
    name = "Network filesystem latency spike"
    description = (
        "Flags elevated request latency when the dataset path is identified as "
        "network/remote-backed. Protocol-specific telemetry can strengthen the "
        "signal; block-device await is only a fallback."
    )

    AWAIT_MS_WARNING = 15.0
    AWAIT_MS_CRITICAL = 50.0
    REMOTE_BACKENDS = {
        "nfs",
        "lustre",
        "weka",
        "gpfs",
        "cephfs",
        "fuse",
        "s3_fuse",
        "object_store",
        "network_fs",
    }

    @classmethod
    def _is_remote_storage(
        cls,
        is_network_fs: bool | None,
        storage_backend: str | None,
        filesystem_type: str | None,
    ) -> bool:
        if is_network_fs:
            return True
        backend = (storage_backend or "").lower()
        fs_type = (filesystem_type or "").lower()
        if backend in cls.REMOTE_BACKENDS:
            return True
        return fs_type.startswith(("nfs", "lustre", "fuse", "ceph", "gpfs"))

    def evaluate(self, window: WindowSummary) -> list[Finding]:
        is_network_fs = window.last("is_network_fs")
        storage_backend = window.last("storage_backend")
        filesystem_type = window.last("filesystem_type")

        if not self._is_remote_storage(is_network_fs, storage_backend, filesystem_type):
            return []

        remote_latency_ms = window.mean("remote_read_latency_ms")
        await_ms = window.mean("disk_avg_await_ms")
        latency_ms = remote_latency_ms if remote_latency_ms is not None else await_ms
        if latency_ms is None or latency_ms < self.AWAIT_MS_WARNING:
            return []

        severity = Severity.CRITICAL if latency_ms >= self.AWAIT_MS_CRITICAL else Severity.WARNING
        disk_name = window.last("disk_name") or "remote storage"
        throughput = window.mean("remote_read_bytes_per_s")
        if throughput is None:
            throughput = window.mean("disk_read_bytes_per_s")
        throughput_mb = (throughput / 1e6) if throughput is not None else None
        source = "remote-read telemetry" if remote_latency_ms is not None else "block-device await"

        return [
            Finding(
                rule_id=self.rule_id,
                severity=severity,
                title="Elevated latency on network-backed input storage",
                message=(
                    f"'{disk_name}' is network/remote-backed and averaged {latency_ms:.1f} ms "
                    f"request latency using {source}"
                    + (f" at {throughput_mb:.0f} MB/s" if throughput_mb is not None else "")
                    + ". This is a storage-latency signal; protocol/client telemetry is still "
                    "needed to identify the exact remote-storage cause."
                ),
                evidence={
                    "disk_name": disk_name,
                    "storage_backend": storage_backend,
                    "filesystem_type": filesystem_type,
                    "latency_source": source,
                    "mean_remote_read_latency_ms": round(remote_latency_ms, 2)
                    if remote_latency_ms is not None
                    else None,
                    "mean_avg_await_ms": round(await_ms, 2) if await_ms is not None else None,
                    "mean_read_throughput_mb_s": round(throughput_mb, 2)
                    if throughput_mb is not None
                    else None,
                },
                suggested_fix=(
                    "Confirm the remote-storage cause with protocol/client telemetry "
                    "(for example NFS RPC latency/retransmits, Lustre client stats, or "
                    "FUSE/object-store request latency). Then test more read-ahead/prefetch, "
                    "larger sequential reads, or staging the working set to local NVMe."
                ),
            )
        ]
