# Lambda attached-filesystem A100 validation

This experiment is the production-backend follow-up to the controlled localhost
HTTP validation.

The training dataset is stored directly on a Lambda filesystem mounted at
`/lambda/nfs/<FILESYSTEM_NAME>`. Lambda documents this as networked persistent
storage; the guest-visible filesystem may appear as `virtiofs`, `nfs`, or
`nfs4` depending on the instance/platform path. During the fault phase, `fio`
generates direct random-read contention against that same attached filesystem.
datapath-doctor then measures DataLoader wait, file-read latency, prefetch state,
GPU utilization, and throughput.

The phases are:

```text
real Lambda filesystem baseline
        ↓
direct random-read fio contention on the same filesystem
        ↓
recovery after contention stops
```

This is deliberately different from adding a user-space sleep: the latency and
contention are produced by real I/O against the attached network filesystem.

## Prerequisites

- Lambda GPU instance with a Lambda filesystem attached at launch
- NVIDIA CUDA/PyTorch Lambda image
- `fio`
- passwordless `sudo` so the script can drop the Linux client page cache once
  before the experiment

Install `fio`:

```bash
sudo apt-get update
sudo apt-get install -y fio
```

## Verify the attached filesystem

```bash
df -h | grep /lambda/nfs
findmnt -T /lambda/nfs/<FILESYSTEM_NAME>
```

The path must be the attached Lambda filesystem. The guest-visible filesystem
type may be `virtiofs`, `nfs`, or `nfs4`; the harness records the exact type
instead of assuming a specific transport.

## Run

From the `a100-validation-harness` branch:

```bash
python experiments/a100_lambda_nfs_fault.py \
  --mount-path /lambda/nfs/<FILESYSTEM_NAME> \
  --steps 20 \
  --warmup-steps 6 \
  --target-compute-ms 40 \
  --fio-numjobs 32 \
  --fio-bs 4k \
  --fio-size-mb 512 \
  --output lambda_nfs_validation_results.json
```

The script creates a temporary training dataset and contention file on the
filesystem, drops the client page cache, runs baseline/fault/recovery with one
persistent DataLoader worker pool, writes the JSON result, and removes the
temporary test data.

If 32 fio jobs do not create a meaningful storage-latency change, rerun with:

```bash
--fio-numjobs 64
```

Do not increase load solely to force a desired diagnosis. The JSON should
record the measured behavior even if the first contention level is not enough
to cross a diagnostic threshold.

## Evidence to preserve

Keep both:

```text
lambda_nfs_validation_results.json
lambda_nfs_validation_results.fio.log
```

The strongest validation pattern is:

- baseline: healthy input path
- fault: attached-filesystem read latency rises, prefetch depth falls, data wait rises,
  GPU utilization/throughput decline
- datapath-doctor identifies the storage/input-path cause from the measured
  evidence
- recovery: metrics move back toward baseline

The exact numbers are intentionally not hard-coded.

## Observed A100 result

The completed attached-filesystem run used:

- NVIDIA A100-SXM4-40GB
- datapath-doctor 0.2.0
- PyTorch 2.7.0 / CUDA 12.8
- Lambda persistent filesystem exposed as `virtiofs`
- 32 `fio` direct random-read processes
- 4 KiB reads against a 512 MiB contention file
- 20 measured training steps per phase

| Phase | Throughput | Data wait | Read latency | Prefetch depth | GPU util | Diagnosis |
|---|---:|---:|---:|---:|---:|---|
| baseline | 185.7 samples/s | 0.34% | 3.80 ms | 3.4 | 70.0% | none |
| fault | 67.9 samples/s | 69.85% | 29.17 ms | 0.2 | 28.4% | remote input-storage latency |
| recovery | 186.7 samples/s | 0.38% | 2.56 ms | 3.65 | 69.0% | none |

The fault phase produced DPD-001 + DPD-004 + DPD-007 and the correlation layer
reported:

```text
LIKELY ROOT CAUSE
remote input-storage latency is starving the training loop
Confidence: high
```

The `fio` workload sustained about 5,167 IOPS at 20.2 MiB/s with 6.07 ms mean
completion latency. Training throughput fell by 63.4%, GPU utilization fell by
59.4%, and training-side read latency rose by 7.7x. After contention stopped,
throughput returned to within 0.6% of baseline.

Raw artifacts:

- [result JSON](results/a100_lambda_virtiofs_fio32_2026-09-28.json)
- [fio log](results/a100_lambda_virtiofs_fio32_2026-09-28.fio.log)

