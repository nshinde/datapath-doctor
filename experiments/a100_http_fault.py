#!/usr/bin/env python3
"""Real A100 validation: baseline -> injected HTTP input latency -> recovery.

This is intentionally a small, reproducible first real-GPU experiment. It uses
an HTTP-backed dataset served on localhost so the storage delay can be changed
without needing a second machine. The training work itself runs on CUDA.

The script records real DataLoader wait, remote-read latency, prefetch depth,
throughput, optional nvidia-smi utilization, and datapath-doctor findings.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.request
from dataclasses import asdict
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any


INPUT_SHAPE = (3, 64, 64)
INPUT_BYTES = 3 * 64 * 64


class QuietDelayedHandler(SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        delay_s = float(getattr(self.server, "delay_s", 0.0))
        if delay_s > 0:
            time.sleep(delay_s)
        super().do_GET()


class GpuUtilSampler:
    def __init__(self, gpu_index: int, interval_s: float = 0.5) -> None:
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                raw = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        str(self.gpu_index),
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=2,
                    stderr=subprocess.DEVNULL,
                ).strip()
                if raw:
                    self.samples.append(float(raw.splitlines()[0]))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "GpuUtilSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    @property
    def mean(self) -> float | None:
        return statistics.fmean(self.samples) if self.samples else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run datapath-doctor baseline/fault/recovery validation on one CUDA GPU."
    )
    parser.add_argument("--steps", type=int, default=20, help="Measured steps per phase.")
    parser.add_argument("--warmup-steps", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--fault-latency-ms",
        type=float,
        default=60.0,
        help="Injected per-object HTTP response delay.",
    )
    parser.add_argument(
        "--target-compute-ms",
        type=float,
        default=40.0,
        help="Auto-calibrate repeated training work to approximately this GPU compute time.",
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--files", type=int, default=64)
    parser.add_argument("--file-bytes", type=int, default=65536)
    parser.add_argument("--output", default="a100_validation_results.json")
    return parser


def make_dataset_class(torch):
    class HttpBinaryDataset(torch.utils.data.Dataset):
        def __init__(self, base_url: str, num_files: int) -> None:
            self.base_url = base_url.rstrip("/")
            self.num_files = num_files

        def __len__(self) -> int:
            return 1_000_000

        def __getitem__(self, index: int):
            file_index = index % self.num_files
            url = f"{self.base_url}/sample-{file_index:04d}.bin"
            t0 = time.perf_counter()
            with urllib.request.urlopen(url, timeout=10) as response:
                payload = response.read()
            latency_s = time.perf_counter() - t0

            raw = torch.frombuffer(bytearray(payload[:INPUT_BYTES]), dtype=torch.uint8)
            x = raw.float().reshape(INPUT_SHAPE).div_(255.0)
            label = index % 10
            return x, label, latency_s * 1000.0, len(payload)

    return HttpBinaryDataset


def make_model(torch):
    features = INPUT_BYTES
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(features, 2048),
        torch.nn.GELU(),
        torch.nn.Linear(2048, 512),
        torch.nn.GELU(),
        torch.nn.Linear(512, 10),
    )


def train_microsteps(torch, model, optimizer, criterion, inputs, labels, repeats: int) -> None:
    for _ in range(repeats):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()


def calibrate_repeats(
    torch,
    model,
    optimizer,
    criterion,
    device,
    batch_size: int,
    target_ms: float,
) -> tuple[int, float]:
    inputs = torch.rand((batch_size, *INPUT_SHAPE), device=device)
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)

    for _ in range(3):
        train_microsteps(torch, model, optimizer, criterion, inputs, labels, 1)
    torch.cuda.synchronize()

    timings = []
    for _ in range(5):
        t0 = time.perf_counter()
        train_microsteps(torch, model, optimizer, criterion, inputs, labels, 1)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000.0)

    one_repeat_ms = statistics.median(timings)
    repeats = max(1, min(128, math.ceil(target_ms / max(one_repeat_ms, 0.1))))
    return repeats, one_repeat_ms


def private_prefetch_depth(profiler) -> int | None:
    """Best-effort experiment-only access to PyTorch's private DataLoader queue."""
    iterator = getattr(profiler, "_iter", None)
    queue = getattr(iterator, "_data_queue", None)
    if queue is None:
        return None
    try:
        return int(queue.qsize())
    except Exception:
        return None


def attach_remote_telemetry(remote, sample, latencies_ms, byte_counts) -> None:
    for latency_ms, byte_count in zip(latencies_ms, byte_counts):
        remote.record_read(float(latency_ms) / 1000.0, bytes_read=int(byte_count))
    reading = remote.sample()
    for key, value in reading.items():
        setattr(sample, key, value)


def run_phase(
    *,
    torch,
    base_url: str,
    num_files: int,
    server,
    delay_ms: float,
    phase_name: str,
    model,
    optimizer,
    criterion,
    device,
    repeats: int,
    args,
):
    from datapath_doctor.collectors import DataLoaderProfiler, RemoteReadCollector
    from datapath_doctor.correlation import correlate_findings
    from datapath_doctor.engine import RuleEngine
    from datapath_doctor.report import render_findings

    server.delay_s = delay_ms / 1000.0
    Dataset = make_dataset_class(torch)
    dataset = Dataset(base_url, num_files)

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["multiprocessing_context"] = "spawn"

    loader = torch.utils.data.DataLoader(**loader_kwargs)
    profiler = DataLoaderProfiler(
        loader,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        monitor_workers=True,
    )
    remote = RemoteReadCollector(storage_backend="object_store", filesystem_type="http")

    iterator = iter(profiler)

    def one_step():
        inputs, labels, latencies_ms, byte_counts = next(iterator)
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        train_microsteps(torch, model, optimizer, criterion, inputs, labels, repeats)
        torch.cuda.synchronize()
        queue_depth = private_prefetch_depth(profiler)
        sample = profiler.record_compute_done(queue_depth=queue_depth)
        attach_remote_telemetry(remote, sample, latencies_ms.tolist(), byte_counts.tolist())
        return sample

    for _ in range(args.warmup_steps):
        one_step()
    profiler.samples.clear()

    with GpuUtilSampler(args.gpu_index) as gpu_sampler:
        phase_start = time.perf_counter()
        for _ in range(args.steps):
            one_step()
        duration_s = time.perf_counter() - phase_start

    samples = list(profiler.samples)
    engine = RuleEngine.with_default_rules()
    findings = engine.run(samples)
    diagnosis = correlate_findings(findings)

    total_wait = sum(s.data_wait_s or 0.0 for s in samples)
    total_step = sum(s.step_time_s or 0.0 for s in samples)
    remote_latencies = [
        s.remote_read_latency_ms for s in samples if s.remote_read_latency_ms is not None
    ]
    queue_depths = [s.queue_depth for s in samples if s.queue_depth is not None]

    result = {
        "phase": phase_name,
        "injected_latency_ms": delay_ms,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "duration_s": duration_s,
        "throughput_samples_s": (args.steps * args.batch_size) / duration_s,
        "mean_data_wait_ms": (total_wait / len(samples)) * 1000.0,
        "data_wait_fraction": total_wait / total_step if total_step else None,
        "mean_remote_read_latency_ms": (
            statistics.fmean(remote_latencies) if remote_latencies else None
        ),
        "mean_prefetch_queue_depth": (
            statistics.fmean(queue_depths) if queue_depths else None
        ),
        "mean_gpu_util_pct": gpu_sampler.mean,
        "finding_ids": [finding.rule_id for finding in findings],
        "diagnosis": (
            {
                "summary": diagnosis.summary,
                "confidence": diagnosis.confidence,
                "diagnosis_type": diagnosis.diagnosis_type,
                "chain": list(diagnosis.chain),
                "supporting_rule_ids": list(diagnosis.supporting_rule_ids),
                "evidence": diagnosis.evidence,
            }
            if diagnosis is not None
            else None
        ),
        "report": render_findings(findings, num_samples=len(samples)),
    }

    shutdown = getattr(getattr(profiler, "_iter", None), "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()

    return result


def write_sample_files(directory: Path, count: int, file_bytes: int) -> None:
    payload_size = max(file_bytes, INPUT_BYTES)
    for index in range(count):
        (directory / f"sample-{index:04d}.bin").write_bytes(os.urandom(payload_size))


def fmt(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def print_summary(results: list[dict[str, Any]]) -> None:
    print()
    print("A100 datapath-doctor validation")
    print(
        f"{'phase':<10} {'samples/s':>10} {'wait %':>8} {'wait ms':>9} "
        f"{'remote ms':>10} {'queue':>8} {'GPU %':>7} diagnosis"
    )
    print("-" * 105)
    for result in results:
        diagnosis = result["diagnosis"]
        diagnosis_text = diagnosis["summary"] if diagnosis else "none"
        wait_fraction = result["data_wait_fraction"]
        wait_pct = None if wait_fraction is None else wait_fraction * 100.0
        print(
            f"{result['phase']:<10} "
            f"{fmt(result['throughput_samples_s']):>10} "
            f"{fmt(wait_pct):>8} "
            f"{fmt(result['mean_data_wait_ms']):>9} "
            f"{fmt(result['mean_remote_read_latency_ms']):>10} "
            f"{fmt(result['mean_prefetch_queue_depth']):>8} "
            f"{fmt(result['mean_gpu_util_pct']):>7} "
            f"{diagnosis_text}"
        )


def main() -> None:
    args = build_parser().parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            'PyTorch is required. Install with: pip install "datapath-doctor[torch]"'
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; run this experiment on an NVIDIA GPU instance.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    dist_version = package_version("datapath-doctor")
    print(f"datapath-doctor distribution: {dist_version}")
    print(f"CUDA device: {gpu_name}")
    if "A100" not in gpu_name.upper():
        print("WARNING: this run is not on an A100; results are still valid but label them accordingly.")

    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    torch.backends.cuda.matmul.allow_tf32 = True

    model = make_model(torch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    repeats, one_repeat_ms = calibrate_repeats(
        torch,
        model,
        optimizer,
        criterion,
        device,
        args.batch_size,
        args.target_compute_ms,
    )
    print(
        f"Calibrated GPU work: one microstep={one_repeat_ms:.2f} ms, "
        f"repeats={repeats}, target≈{args.target_compute_ms:.0f} ms"
    )

    with tempfile.TemporaryDirectory(prefix="datapath-a100-") as tmp:
        data_dir = Path(tmp)
        write_sample_files(data_dir, args.files, args.file_bytes)

        handler = partial(QuietDelayedHandler, directory=str(data_dir))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.delay_s = 0.0
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        print(f"HTTP dataset: {base_url} ({args.files} files)")

        try:
            results = [
                run_phase(
                    torch=torch,
                    base_url=base_url,
                    num_files=args.files,
                    server=server,
                    delay_ms=0.0,
                    phase_name="baseline",
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    repeats=repeats,
                    args=args,
                ),
                run_phase(
                    torch=torch,
                    base_url=base_url,
                    num_files=args.files,
                    server=server,
                    delay_ms=args.fault_latency_ms,
                    phase_name="fault",
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    repeats=repeats,
                    args=args,
                ),
                run_phase(
                    torch=torch,
                    base_url=base_url,
                    num_files=args.files,
                    server=server,
                    delay_ms=0.0,
                    phase_name="recovery",
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    repeats=repeats,
                    args=args,
                ),
            ]
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=3)

    payload = {
        "datapath_doctor_version": dist_version,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "compute_repeats": repeats,
        "target_compute_ms": args.target_compute_ms,
        "fault_latency_ms": args.fault_latency_ms,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")

    print_summary(results)
    print()
    print(f"Full reports and raw summary written to: {args.output}")
    print()
    print("Fault-phase report:")
    print(results[1]["report"])


if __name__ == "__main__":
    main()
