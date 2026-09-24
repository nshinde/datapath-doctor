from __future__ import annotations

from datapath_doctor.engine import Rule
from datapath_doctor.models import Finding, Severity, WindowSummary


class NetworkFilesystemLatencySpikeRule(Rule):
    """Fires when telemetry explicitly identifies a network-backed filesystem
    and observed I/O request latency is elevated.

    This rule deliberately reports a network-storage latency *signal*, not a
    protocol-level root cause. Block-device await alone cannot distinguish
    network round trips, metadata-server contention, client queueing, FUSE
    overhead, or server-side throttling.
    """

    rule_id = "DPD-007"
    name = "Network filesystem latency spike"
    description = (
        "Flags elevated I/O request latency when the collector identifies the "
        "dataset path as network-backed. Treat this as a storage-latency signal "
        "that should be correlated with client/protocol telemetry where available."
    )

    AWAIT_MS_WARNING = 15.0
    AWAIT_MS_CRITICAL = 50.0

    def evaluate(self, window: WindowSummary) -> list[Finding]:
        is_net = window.last("is_network_fs")
        if not is_net:
            return []

        await_ms = window.mean("disk_avg_await_ms")
        if await_ms is None or await_ms < self.AWAIT_MS_WARNING:
            return []

        severity = Severity.CRITICAL if await_ms >= self.AWAIT_MS_CRITICAL else Severity.WARNING
        disk_name = window.last("disk_name") or "network mount"
        throughput = window.mean("disk_read_bytes_per_s")
        throughput_mb = (throughput / 1e6) if throughput is not None else None

        return [
            Finding(
                rule_id=self.rule_id,
                severity=severity,
                title="Elevated latency on network-backed input storage",
                message=(
                    f"'{disk_name}' is marked network-backed and averaged {await_ms:.1f} ms "
                    "I/O request latency"
                    + (f" at {throughput_mb:.0f} MB/s" if throughput_mb is not None else "")
                    + ". This confirms a storage-latency signal, but additional NFS/Lustre/FUSE/"
                    "object-store telemetry is needed to identify the protocol-level cause."
                ),
                evidence={
                    "disk_name": disk_name,
                    "mean_avg_await_ms": round(await_ms, 2),
                    "mean_read_throughput_mb_s": round(throughput_mb, 2) if throughput_mb is not None else None,
                },
                suggested_fix=(
                    "First confirm the remote-storage cause with protocol/client telemetry "
                    "(for example NFS RPC latency/retransmits, Lustre client stats, or FUSE/"
                    "object-store request latency). In parallel, test whether more read-ahead/"
                    "prefetch, larger sequential reads, or staging the working set to local "
                    "NVMe reduces DataLoader wait time."
                ),
            )
        ]
