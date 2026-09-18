# datapath-doctor

A 10-rule diagnostic engine for the **storage / data-loading input path** of ML
training jobs, with SQLite cross-run tracking. Fifth tool in a unified GPU
observability platform, alongside [torchguard](https://github.com/nshinde/torchguard)
(in-process PyTorch profiler + YAML rules engine), [stallscope](https://github.com/nshinde/stallscope)
(node-level GPU/RDMA telemetry agent), and [nccl-doctor](https://github.com/nshinde/nccl-doctor)
(17-rule NCCL diagnostic engine).

Where nccl-doctor answers "why is my collective communication slow," datapath-doctor
answers the question one layer earlier: **why is the GPU waiting at all** — disk
I/O saturation, network filesystem latency, CPU-bound decode/tokenize work in
DataLoader workers, an undersized prefetch buffer, a small-file-heavy dataset
layout, shuffle buffer stalls, synchronous checkpoint I/O, or crashing worker
processes.

## Why a separate tool

`nvidia-smi` shows GPU utilization dropping to zero between steps but not why.
Profilers built for the compute graph (PyTorch Profiler, Nsight) see the GPU
side clearly and the data pipeline as an opaque gap. datapath-doctor is built
the other way around: it instruments the consumer side of the DataLoader, the
block device underneath the dataset, and the training loop's checkpoint calls,
then runs a rule engine over that telemetry to name the specific bottleneck
rather than just reporting "GPU idle X% of the time."

## Install

```bash
pip install -e ".[dev]"        # editable install + test deps
pip install -e ".[torch]"      # if you want to run the DataLoader examples
```

## Quickstart: try it on synthetic data

No GPU, disk telemetry, or training job required — six scripted scenarios
exercise the full rule set:

```bash
datapath-doctor demo --scenario healthy              # no findings
datapath-doctor demo --scenario network_fs_latency    # NFS/S3-style latency-bound reads
datapath-doctor demo --scenario cpu_bound_decode       # workers pegged, disk idle
datapath-doctor demo --scenario small_file_storm       # millions-of-small-files layout
datapath-doctor demo --scenario checkpoint_stall       # sync checkpoint writes blocking steps
datapath-doctor demo --scenario worker_crash_loop      # workers OOMing and restarting

datapath-doctor rules       # list all registered rules
datapath-doctor runs        # list recorded runs (SQLite history)
datapath-doctor trends      # rule fire-rate across recent runs
```

## Wiring it into a real training job

`DataLoaderProfiler` wraps any DataLoader-like iterable and records per-step
timing without changing your training loop's structure:

```python
from datapath_doctor.collectors.dataloader_profiler import DataLoaderProfiler
from datapath_doctor.collectors.proc_stats import DiskIOCollector, PSICollector
from datapath_doctor.engine import RuleEngine
from datapath_doctor.db import HistoryDB

loader = torch.utils.data.DataLoader(dataset, num_workers=8, prefetch_factor=4, ...)
profiler = DataLoaderProfiler(loader, num_workers=8, prefetch_factor=4)
disk = DiskIOCollector("nvme0n1")
psi = PSICollector()

for batch in profiler:
    outputs = model(batch)
    loss.backward()
    optimizer.step()

    sample = profiler.record_compute_done(queue_depth=loader_queue_depth())
    if disk_reading := disk.sample():
        for k, v in disk_reading.items():
            setattr(sample, k, v)
    if psi_reading := psi.sample():
        for k, v in psi_reading.items():
            setattr(sample, k, v)

# Periodically (e.g. every N steps, or at epoch end):
engine = RuleEngine.with_default_rules()
findings = engine.run(profiler.samples)
HistoryDB().record_run(findings, job_name="my-training-run", num_samples=len(profiler.samples))
```

Every field on `StepSample` is optional — a job with no PSI support, no
network filesystem, or no checkpoint instrumentation still gets full value
from the rules that apply to the telemetry it does provide.

## Rule set (v0.1)

| Rule | Fires on |
|---|---|
| DPD-001 | GPU starved waiting on the data pipeline (headline signal) |
| DPD-002 | Disk I/O saturation (utilization % + await) |
| DPD-003 | System-wide I/O pressure via Linux PSI |
| DPD-004 | Prefetch buffer underrun |
| DPD-005 | Worker pool CPU-bound (decode/tokenize, not storage) |
| DPD-006 | Small-file I/O overhead (tiny reads, high IOPS) |
| DPD-007 | Network filesystem latency spike (NFS/S3/Lustre-style) |
| DPD-008 | Shuffle buffer stall / undersized buffer |
| DPD-009 | Checkpoint I/O blocking training steps |
| DPD-010 | DataLoader worker crash/restart loop |

Rule IDs are stable and never reused; new rules append at the next number.

## Design

- `models.py` — `StepSample` (one step's telemetry), `WindowSummary`
  (aggregates a batch of samples once for every rule to share), `Finding`.
- `engine.py` — `Rule` base class and `RuleEngine`, which runs every
  registered rule against a window, isolates a misbehaving rule so it can't
  take down the rest of the pass, and returns findings sorted by severity.
- `collectors/` — telemetry sources: `DataLoaderProfiler` (consumer-side
  timing + worker CPU/restart monitoring via psutil), `proc_stats`
  (`/proc/diskstats`, `/proc/pressure/io`), and `synthetic` (the demo
  scenarios, also used by the test suite).
- `db.py` — SQLite persistence (`HistoryDB`) so findings survive past one
  run and `trends` can show a rule's fire rate over the last N runs, the
  same cross-run tracking pattern as nccl-doctor.
- `rules/` — one file per rule, each independent and unit-testable against
  a hand-built `WindowSummary`.

## Validation

Validated with synthetic fault injection (the six scenarios above) covering
each rule's intended failure signature, plus targeted unit tests per rule
against hand-constructed telemetry windows. Real-workload validation against
GPT-AR / MDLM / Mamba training runs — matching the methodology used for
torchguard, stallscope, and nccl-doctor — is in progress; a second paper
covering storage fault injection results is planned once that data lands.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

## License

Apache-2.0
