#!/usr/bin/env python3
"""Real Lambda filesystem validation on an NVIDIA GPU.

Runs one persistent PyTorch DataLoader/CUDA workload through:
  baseline -> real NFS contention -> recovery

The dataset and contention file both live on the attached Lambda filesystem.
The fault is generated with direct random-read fio load against the same
filesystem, so the measured storage latency is from the real network-backed
path rather than an artificial sleep in the dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import threading
import time
import uuid
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Optional

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        'PyTorch is required. Use the Lambda image PyTorch or install '
        '"datapath-doctor[torch]".'
    ) from exc


INPUT_SHAPE = (3, 64, 64)
INPUT_BYTES = 3 * 64 * 64


class GpuUtilSampler:
    def __init__(self, gpu_index: int, interval_s: float = 0.5) -> None:
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.samples = []  # type: list[float]
        self._stop = threading.Event()
        self._thread = None  # type: Optional[threading.Thread]

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
    def mean(self) -> Optional[float]:
        return statistics.fmean(self.samples) if self.samples else None


class NfsBinaryDataset(torch.utils.data.Dataset):
    def __init__(self, files: list[str]) -> None:
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        t0 = time.perf_counter()
        with open(path, "rb", buffering=0) as handle:
            payload = handle.read()
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                try:
                    os.posix_fadvise(
                        handle.fileno(),
                        0,
                        0,
                        os.POSIX_FADV_DONTNEED,
                    )
                except OSError:
                    pass
        latency_s = time.perf_counter() - t0

        raw = torch.frombuffer(bytearray(payload[:INPUT_BYTES]), dtype=torch.uint8)
        inputs = raw.float().reshape(INPUT_SHAPE).div_(255.0)
        label = index % 10
        return inputs, label, latency_s * 1000.0, len(payload)


class FioContention:
    def __init__(
        self,
        filename: Path,
        numjobs: int,
        block_size: str,
        log_path: Path,
    ) -> None:
        self.filename = filename
        self.numjobs = numjobs
        self.block_size = block_size
        self.log_path = log_path
        self.proc = None  # type: Optional[subprocess.Popen]
        self._log_handle = None

    def start(self) -> None:
        if shutil.which("fio") is None:
            raise RuntimeError("fio is not installed; install it with: sudo apt-get install -y fio")

        cmd = [
            "fio",
            "--name=datapath-nfs-contention",
            f"--filename={self.filename}",
            "--rw=randread",
            f"--bs={self.block_size}",
            "--direct=1",
            "--ioengine=psync",
            f"--numjobs={self.numjobs}",
            "--time_based=1",
            "--runtime=300",
            "--group_reporting=1",
        ]
        self._log_handle = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            cmd,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        time.sleep(2.0)
        if self.proc.poll() is not None:
            self._log_handle.flush()
            raise RuntimeError(
                f"fio contention process exited early; inspect {self.log_path}"
            )

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate datapath-doctor against a real attached Lambda filesystem."
    )
    parser.add_argument("--mount-path", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--target-compute-ms", type=float, default=40.0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--file-bytes", type=int, default=262144)
    parser.add_argument("--fio-numjobs", type=int, default=32)
    parser.add_argument("--fio-bs", default="4k")
    parser.add_argument("--fio-size-mb", type=int, default=512)
    parser.add_argument("--output", default="lambda_nfs_validation_results.json")
    parser.add_argument("--keep-data", action="store_true")
    return parser


def find_mount_info(mount_path: Path) -> dict[str, str]:
    raw = subprocess.check_output(
        [
            "findmnt",
            "-T",
            str(mount_path),
            "-n",
            "-o",
            "TARGET,SOURCE,FSTYPE",
        ],
        text=True,
    ).strip()
    parts = raw.split()
    if len(parts) < 3:
        raise RuntimeError(f"Could not parse findmnt output: {raw!r}")
    target, source, fstype = parts[0], parts[1], parts[2]
    if "nfs" not in fstype.lower():
        raise RuntimeError(
            f"{mount_path} is mounted as {fstype}, not NFS/NFS4. "
            "Use the attached Lambda filesystem mount."
        )
    return {"target": target, "source": source, "fstype": fstype}


def make_model():
    features = INPUT_BYTES
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(features, 2048),
        torch.nn.GELU(),
        torch.nn.Linear(2048, 512),
        torch.nn.GELU(),
        torch.nn.Linear(512, 10),
    )


def train_microsteps(model, optimizer, criterion, inputs, labels, repeats: int) -> None:
    for _ in range(repeats):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()


def calibrate_repeats(
    model,
    optimizer,
    criterion,
    device,
    batch_size: int,
    target_ms: float,
):
    inputs = torch.rand((batch_size, *INPUT_SHAPE), device=device)
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)

    for _ in range(3):
        train_microsteps(model, optimizer, criterion, inputs, labels, 1)
    torch.cuda.synchronize()

    timings = []
    for _ in range(5):
        t0 = time.perf_counter()
        train_microsteps(model, optimizer, criterion, inputs, labels, 1)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000.0)

    one_repeat_ms = statistics.median(timings)
    repeats = max(1, min(128, math.ceil(target_ms / max(one_repeat_ms, 0.1))))
    return repeats, one_repeat_ms


def private_prefetch_depth(profiler) -> Optional[int]:
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


def write_dataset_files(directory: Path, count: int, file_bytes: int) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    payload_size = max(file_bytes, INPUT_BYTES)
    payload = os.urandom(payload_size)
    files = []
    for index in range(count):
        path = directory / f"sample-{index:05d}.bin"
        with open(path, "wb", buffering=0) as handle:
            handle.write(payload)
        files.append(str(path))
    return files


def write_fio_file(path: Path, size_mb: int) -> None:
    chunk = os.urandom(1024 * 1024)
    with open(path, "wb", buffering=0) as handle:
        for _ in range(size_mb):
            handle.write(chunk)
        os.fsync(handle.fileno())


def drop_client_caches() -> None:
    subprocess.run(["sync"], check=True)
    result = subprocess.run(
        ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Could not drop Linux client caches. Ensure passwordless sudo is "
            "available on the Lambda instance. stderr: " + result.stderr.strip()
        )


def measure_phase(
    *,
    phase_name: str,
    profiler,
    iterator,
    remote,
    model,
    optimizer,
    criterion,
    device,
    repeats: int,
    args,
):
    from datapath_doctor.correlation import correlate_findings
    from datapath_doctor.engine import RuleEngine
    from datapath_doctor.report import render_findings

    def one_step():
        inputs, labels, latencies_ms, byte_counts = next(iterator)
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        train_microsteps(model, optimizer, criterion, inputs, labels, repeats)
        torch.cuda.synchronize()

        sample = profiler.record_compute_done(
            queue_depth=private_prefetch_depth(profiler)
        )
        attach_remote_telemetry(
            remote,
            sample,
            latencies_ms.tolist(),
            byte_counts.tolist(),
        )
        return sample

    for _ in range(args.warmup_steps):
        one_step()

    profiler.samples.clear()
    remote.sample(reset=True)

    with GpuUtilSampler(args.gpu_index) as gpu_sampler:
        phase_start = time.perf_counter()
        for _ in range(args.steps):
            one_step()
        duration_s = time.perf_counter() - phase_start

    samples = list(profiler.samples)
    findings = RuleEngine.with_default_rules().run(samples)
    diagnosis = correlate_findings(findings)

    total_wait = sum(sample.data_wait_s or 0.0 for sample in samples)
    total_step = sum(sample.step_time_s or 0.0 for sample in samples)
    remote_latencies = [
        sample.remote_read_latency_ms
        for sample in samples
        if sample.remote_read_latency_ms is not None
    ]
    queue_depths = [
        sample.queue_depth for sample in samples if sample.queue_depth is not None
    ]

    return {
        "phase": phase_name,
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


def fmt(value: Optional[float], digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def print_summary(results) -> None:
    print()
    print("Lambda NFS datapath-doctor validation")
    print(
        f"{'phase':<10} {'samples/s':>10} {'wait %':>8} {'wait ms':>9} "
        f"{'remote ms':>10} {'queue':>8} {'GPU %':>7} diagnosis"
    )
    print("-" * 108)
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

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run on a Lambda NVIDIA GPU instance.")

    mount_path = Path(args.mount_path).resolve()
    if not mount_path.exists():
        raise SystemExit(f"Mount path does not exist: {mount_path}")

    mount_info = find_mount_info(mount_path)
    required_samples = args.batch_size * (args.steps + args.warmup_steps) * 3
    file_count = required_samples + args.batch_size

    device = torch.device(f"cuda:{args.gpu_index}")
    gpu_name = torch.cuda.get_device_name(device)
    dist_version = package_version("datapath-doctor")

    print(f"datapath-doctor distribution: {dist_version}")
    print(f"CUDA device: {gpu_name}")
    print(
        "Lambda filesystem: "
        f"{mount_info['source']} -> {mount_info['target']} ({mount_info['fstype']})"
    )
    print(
        f"Contention fault: fio randread, {args.fio_numjobs} jobs, "
        f"bs={args.fio_bs}, direct=1"
    )

    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    torch.backends.cuda.matmul.allow_tf32 = True

    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    repeats, one_repeat_ms = calibrate_repeats(
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

    run_dir = mount_path / f"datapath-doctor-validation-{uuid.uuid4().hex[:8]}"
    dataset_dir = run_dir / "dataset"
    fio_path = run_dir / "fio-contention.bin"
    fio_log = Path(args.output).with_suffix(".fio.log")

    print(
        f"Preparing {file_count} dataset files "
        f"({args.file_bytes / 1024:.0f} KiB each) on Lambda filesystem..."
    )
    files = write_dataset_files(dataset_dir, file_count, args.file_bytes)
    print(f"Preparing {args.fio_size_mb} MiB fio contention file...")
    write_fio_file(fio_path, args.fio_size_mb)
    drop_client_caches()

    from datapath_doctor.collectors import DataLoaderProfiler, RemoteReadCollector

    dataset = NfsBinaryDataset(files)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": True,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["multiprocessing_context"] = "spawn"
        loader_kwargs["persistent_workers"] = True

    loader = torch.utils.data.DataLoader(**loader_kwargs)
    profiler = DataLoaderProfiler(
        loader,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        monitor_workers=True,
    )
    remote = RemoteReadCollector(
        storage_backend="nfs",
        filesystem_type=mount_info["fstype"],
    )
    iterator = iter(profiler)

    fio = FioContention(
        filename=fio_path,
        numjobs=args.fio_numjobs,
        block_size=args.fio_bs,
        log_path=fio_log,
    )

    results = []
    try:
        print("Running phase: baseline")
        results.append(
            measure_phase(
                phase_name="baseline",
                profiler=profiler,
                iterator=iterator,
                remote=remote,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                repeats=repeats,
                args=args,
            )
        )

        print("Starting real NFS contention with fio...")
        fio.start()
        print("Running phase: fault")
        results.append(
            measure_phase(
                phase_name="fault",
                profiler=profiler,
                iterator=iterator,
                remote=remote,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                repeats=repeats,
                args=args,
            )
        )

        print("Stopping fio contention...")
        fio.stop()
        time.sleep(2.0)

        print("Running phase: recovery")
        results.append(
            measure_phase(
                phase_name="recovery",
                profiler=profiler,
                iterator=iterator,
                remote=remote,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                repeats=repeats,
                args=args,
            )
        )
    finally:
        fio.stop()

    payload = {
        "datapath_doctor_version": dist_version,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "mount_path": str(mount_path),
        "mount_target": mount_info["target"],
        "mount_source": mount_info["source"],
        "filesystem_type": mount_info["fstype"],
        "fault_type": "fio_direct_randread_contention",
        "fio_numjobs": args.fio_numjobs,
        "fio_block_size": args.fio_bs,
        "fio_size_mb": args.fio_size_mb,
        "file_bytes": args.file_bytes,
        "compute_repeats": repeats,
        "target_compute_ms": args.target_compute_ms,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")

    print_summary(results)
    print()
    print(f"Raw result: {args.output}")
    print(f"fio log: {fio_log}")
    print()
    print("Fault-phase report:")
    print(results[1]["report"])

    if args.keep_data:
        print(f"Keeping test data: {run_dir}")
    else:
        shutil.rmtree(run_dir, ignore_errors=True)
        print(f"Removed test data: {run_dir}")


if __name__ == "__main__":
    main()
