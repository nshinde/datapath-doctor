from __future__ import annotations

from datapath_doctor.engine import Rule
from datapath_doctor.models import Finding, Severity, WindowSummary


class NetworkFilesystemLatencySpikeRule(Rule):
    """Fires when reads are served from a network filesystem (NFS/S3/Lustre/
    etc., as flagged by the collector) and per-request latency is high even
    though the device itself doesn't look "busy" the way a saturated local
    disk would — the signature of network round-trip / metadata-server
    latency rather than bandwidth exhaustion.
    """

    rule_id = "DPD-007"
    name = "Network filesystem latency spike"
    description = (
        "Flags high average await on a network-backed filesystem, which "
        "behaves differently from local disk saturation: latency-bound "
        "rather than bandwidth-bound, and not fixed by more workers alone."
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
                title="Network filesystem latency is bottlenecking reads",
                message=(
                    f"'{disk_name}' is network-backed and averaged {await_ms:.1f} ms per "
                    "request"
                    + (f" at only {throughput_mb:.0f} MB/s" if throughput_mb is not None else "")
                    + " — this looks like round-trip/metadata latency, not a bandwidth ceiling."
                ),
                evidence={
                    "disk_name": disk_name,
                    "mean_avg_await_ms": round(await_ms, 2),
                    "mean_read_throughput_mb_s": round(throughput_mb, 2) if throughput_mb is not None else None,
                },
                suggested_fix=(
                    "Latency-bound network storage is usually fixed by hiding round-trips, not "
                    "by adding bandwidth: increase read-ahead/prefetch depth so more requests "
                    "are in flight concurrently, batch small reads into larger sequential "
                    "range requests, or stage a working set onto local NVMe/tmpfs ahead of "
                    "training (e.g. a warm-cache copy step before the job starts)."
                ),
            )
        ]
