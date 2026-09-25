# datapath-doctor

**datapath-doctor** is an evidence-based diagnostic engine for the **storage and
data-loading input path of ML training jobs**.

It answers a question that GPU utilization alone cannot:

> **Why is the training loop waiting for data?**

The tool combines DataLoader timing, worker behavior, prefetch state, Linux I/O
pressure, block-device telemetry, storage identity, checkpoint timing, and
optional remote-storage signals. Ten independent rules detect specific failure
signatures, and a correlation layer combines their evidence into a likely root
cause, contributing factor, or diagnostic symptom.

datapath-doctor is part of a broader open-source ML infrastructure observability
toolkit alongside [torchguard](https://github.com/nshinde/torchguard),
[stallscope](https://github.com/nshinde/stallscope), and
[nccl-doctor](https://github.com/nshinde/nccl-doctor).

Where nccl-doctor asks **"why is collective communication slow?"**,
datapath-doctor focuses one layer earlier:

**"Why is the accelerator waiting for its next batch?"**

## Why this exists

`nvidia-smi` can show GPU utilization dropping between steps, but it cannot tell
you whether the cause is:

- saturated local storage
- latency on network-backed storage
- CPU-bound decode/tokenization/augmentation
- an empty DataLoader prefetch queue
- a small-file-heavy dataset layout
- system-wide I/O contention
- shuffle-buffer stalls
- synchronous checkpoint writes
- crashing/restarting DataLoader workers

Compute-centric profilers see the accelerator side well. datapath-doctor is
built from the opposite direction: instrument the input path, detect independent
signals, then correlate those signals into a more useful diagnosis.

## How it works

```text
Training loop
     |
     v
DataLoaderProfiler + system/storage collectors
     |
     v
StepSample telemetry
     |
     v
WindowSummary
     |
     v
10 independent diagnostic rules
     |
     v
Findings + evidence
     |
     v
Correlation engine
     |
     +--> likely root cause
     +--> contributing factor
     +--> diagnostic symptom
```

The rules intentionally stay independent and unit-testable. Correlation happens
after rule evaluation, using the measurements stored in each finding's
`evidence` rather than treating the presence of a rule ID alone as proof.

For example:

```text
training waits on input for 55% of step time
        |
        v
prefetch queue near-empty in 80% of samples
        |
        v
network-backed storage latency = 65 ms
        |
        v
LIKELY ROOT CAUSE
remote input-storage latency is starving the training loop
Confidence: high
```

The correlation layer also prefers a more specific causal signature over a
generic downstream symptom. For example, tiny reads at very high IOPS can be
reported as **small-file I/O overhead** even when the disk is also highly
utilized, rather than stopping at the less-specific diagnosis of disk
saturation.

## Install

The package is prepared for a PyPI release. Until the first release is
published, install from source:

```bash
git clone https://github.com/nshinde/datapath-doctor.git
cd datapath-doctor
pip install .
```

After the PyPI Trusted Publisher is configured and the first release is
published:

```bash
pip install datapath-doctor
```

For development:

```bash
pip install -e ".[dev]"
```

## Quickstart

No GPU or real training job is required to exercise the diagnostic engine.
Six synthetic fault scenarios are included:

```bash
datapath-doctor demo --scenario healthy
datapath-doctor demo --scenario network_fs_latency
datapath-doctor demo --scenario cpu_bound_decode
datapath-doctor demo --scenario small_file_storm
datapath-doctor demo --scenario checkpoint_stall
datapath-doctor demo --scenario worker_crash_loop

datapath-doctor rules
datapath-doctor runs
datapath-doctor trends
```

## Wiring it into a training job

`DataLoaderProfiler` wraps any DataLoader-like iterable and records how long
the training loop spends waiting for the next batch versus computing.

```python
from datapath_doctor.collectors.dataloader_profiler import DataLoaderProfiler
from datapath_doctor.collectors.proc_stats import DiskIOCollector, PSICollector
from datapath_doctor.engine import RuleEngine
from datapath_doctor.db import HistoryDB

loader = torch.utils.data.DataLoader(
    dataset,
    num_workers=8,
    prefetch_factor=4,
    ...,
)

profiler = DataLoaderProfiler(
    loader,
    num_workers=8,
    prefetch_factor=4,
)

disk = DiskIOCollector("nvme0n1")
psi = PSICollector()

for batch in profiler:
    outputs = model(batch)
    loss.backward()
    optimizer.step()

    sample = profiler.record_compute_done(
        queue_depth=loader_queue_depth(),
    )

    if disk_reading := disk.sample():
        for key, value in disk_reading.items():
            setattr(sample, key, value)

    if psi_reading := psi.sample():
        for key, value in psi_reading.items():
            setattr(sample, key, value)

engine = RuleEngine.with_default_rules()
findings = engine.run(profiler.samples)

HistoryDB().record_run(
    findings,
    job_name="my-training-run",
    num_samples=len(profiler.samples),
)
```

Every field on `StepSample` is optional except timestamp and step number. Rules
that do not have the telemetry they need simply decline to fire.


## Distributed rank awareness

Distributed jobs can fail at the pace of a single slow input rank. `StepSample`
therefore carries optional `rank`, `local_rank`, `world_size`, and `node_id`
fields. `DataLoaderProfiler` accepts them explicitly and can also pick up common
`torchrun` / SLURM environment variables.

`analyze_distributed()` runs the ordinary diagnostic rules independently per
rank and then compares rank-level input wait behavior:

```python
from datapath_doctor.distributed import analyze_distributed

analysis = analyze_distributed(all_rank_samples)

if analysis.straggler:
    print(analysis.straggler.message)
```

A rank is only marked as an input-path straggler when its mean data wait is both
material in absolute terms and significantly slower than the peer median. The
diagnosis intentionally says that the rank *can become the pacing rank* once
distributed steps synchronize; it does not claim to identify the exact later
collective where peers wait.

## Telemetry model

The input-path model currently supports signals in several layers.

### Training / DataLoader

- step time
- data-wait time
- compute time
- worker count
- prefetch factor
- queue depth/capacity
- mean worker CPU utilization
- worker restart count

### Local/block storage

- device name
- read throughput
- read IOPS
- disk utilization
- average I/O await
- average read size

### Storage identity

The model supports explicit storage identity instead of only a boolean
"network filesystem" flag:

- `storage_backend`
- `filesystem_type`
- legacy `is_network_fs` for compatibility

This allows future collectors to distinguish paths such as local NVMe, NFS,
Lustre, Weka, GPFS, CephFS, and FUSE-backed object storage.

### Normalized remote-storage telemetry

Protocol-specific collectors can normalize useful signals into:

- `remote_read_latency_ms`
- `remote_read_ops_per_s`
- `remote_read_bytes_per_s`
- `remote_retries`
- `remote_errors`

Protocol-specific details can remain in `StepSample.extra`.

The current network-storage rule prefers normalized remote-read latency when it
is available and falls back to block-device await for compatibility. That
fallback is treated as a **storage-latency signal**, not proof of an
NFS/Lustre/FUSE protocol-level root cause.


### Direct object-store / streaming readers

Remote loaders do not always have a meaningful block device. For direct S3,
HTTP, fsspec, WebDataset, or custom remote readers, use
`RemoteReadCollector` to instrument the read operation itself:

```python
from datapath_doctor.collectors import RemoteReadCollector

remote = RemoteReadCollector(
    storage_backend="s3",
    filesystem_type="fsspec",
)

with fs.open(path, "rb") as raw:
    f = remote.wrap_file(raw)
    payload = f.read()

reading = remote.sample()
for key, value in reading.items():
    setattr(step_sample, key, value)
```

The collector normalizes request latency, operations/second, bytes/second,
retries, and errors into the existing `StepSample` fields, so DPD-007 can work
without relying on `/proc/diskstats`.

### Linux pressure / pipeline state

- Linux PSI I/O pressure
- shuffle-buffer size/capacity/refill state
- checkpoint duration and blocking state

## Diagnostic rules

| Rule | Detects |
|---|---|
| DPD-001 | Training loop starved waiting on the input pipeline |
| DPD-002 | Block-device I/O saturation |
| DPD-003 | System-wide I/O pressure via Linux PSI |
| DPD-004 | DataLoader prefetch-buffer underrun |
| DPD-005 | CPU-bound DataLoader worker pool |
| DPD-006 | Small-file I/O overhead |
| DPD-007 | Elevated latency on network/remote-backed input storage |
| DPD-008 | Shuffle-buffer stall / undersized buffer |
| DPD-009 | Synchronous checkpoint I/O blocking training |
| DPD-010 | DataLoader worker crash/restart activity |

Rule IDs are stable and never reused.


### Async checkpoint behavior

DPD-009 only evaluates checkpoint writes explicitly marked
`checkpoint_blocking=True`. A long asynchronous flush with
`checkpoint_blocking=False` does not trigger the rule. This behavior has an
explicit regression test to protect against false positives.

## Evidence-aware correlation

A single alert is often insufficient to identify a bottleneck. The correlation
engine combines compatible findings and their evidence.

Examples:

### CPU-bound preprocessing

```text
high data-wait fraction
        +
worker CPU near saturation
        +
disk utilization remains low
        |
        v
CPU-bound input preprocessing
```

### Remote-storage latency

```text
high data-wait fraction
        +
prefetch queue repeatedly drains
        +
elevated remote-storage latency
        |
        v
remote input-storage latency
```

### Small-file dataset layout

```text
high data-wait fraction
        +
very small average read size
        +
high read IOPS
        |
        v
small-file I/O overhead
```

Confidence is derived from the strength of the available measurements. Missing
evidence lowers confidence, and contradictory signals can downgrade a diagnosis
from `root_cause` to `contributing_factor`.

Some findings deliberately remain `symptom` diagnoses. For example, worker
restarts prove worker instability but do not by themselves prove whether the
underlying cause is an OOM kill, an uncaught transform exception, remote-I/O
failure, or another process-level fault.

## Design

- `models.py` — `StepSample`, `WindowSummary`, and `Finding`.
- `engine.py` — rule registry and execution. A broken rule is isolated so it
  cannot terminate the entire diagnostic pass.
- `rules/` — one independent rule per file.
- `correlation.py` — evidence-aware diagnosis and confidence scoring.
- `distributed.py` — per-rank rule evaluation and cross-rank input-straggler analysis.
- `collectors/dataloader_profiler.py` — consumer-side DataLoader timing plus
  worker monitoring.
- `collectors/proc_stats.py` — Linux `/proc/diskstats` and
  `/proc/pressure/io` telemetry.
- `collectors/remote.py` — backend-agnostic remote/object-store read instrumentation.
- `collectors/synthetic.py` — synthetic fault scenarios used by demos/tests.
- `db.py` — SQLite run history and cross-run rule fire-rate tracking.
- `report.py` — terminal-friendly findings and correlated diagnosis output.

## Validation

Current validation includes:

- six synthetic fault-injection scenarios
- targeted positive and negative tests for individual rules
- correlation tests using real evidence values rather than only rule presence
- tests that distinguish root causes, contributing factors, and symptoms
- rank-skew tests covering a single slow DataLoader rank
- direct remote/object-store read instrumentation tests
- an async-checkpoint regression test that prevents false blocking diagnoses
- precedence tests so specific signatures such as small-file I/O outrank
  generic disk saturation

Real-workload validation is still in progress. The next evidence milestone is a
real distributed training run with an injected storage fault (for example,
throttled network-backed input on an A100 node), showing baseline, fault, and
recovery behavior. No real A100/NFS result is claimed in this README until that
experiment has actually been run.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

## License

Apache-2.0
