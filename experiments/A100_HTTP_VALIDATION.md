# A100 HTTP input-path validation

This is the first real-GPU validation harness for datapath-doctor. It runs the
same CUDA training-style workload in three phases:

1. **baseline** — local HTTP-backed dataset with no injected delay
2. **fault** — the HTTP storage path adds a configurable delay to every object read
3. **recovery** — the delay is removed

The goal is not to claim that localhost HTTP is equivalent to NFS, Lustre, Weka,
or S3. It is a controlled first experiment that exercises a real A100 training
loop and a real socket-backed input path while producing a reproducible storage
latency fault.

## Run on an A100 instance

Use a clean environment so the test consumes the released PyPI package:

```bash
python -m venv ~/venvs/datapath-a100
source ~/venvs/datapath-a100/bin/activate

python -m pip install --upgrade pip
pip install "datapath-doctor[torch]"

git clone https://github.com/nshinde/datapath-doctor.git
cd datapath-doctor
git checkout main

python experiments/a100_http_fault.py \
  --steps 20 \
  --warmup-steps 6 \
  --fault-latency-ms 60 \
  --target-compute-ms 40 \
  --output a100_validation_results.json
```

The script prints the installed distribution version and GPU model before it
runs. Keep that header with the experiment result.

## Observed A100 result

The first completed run used an **NVIDIA A100-SXM4-40GB**, datapath-doctor
0.2.0, PyTorch 2.7.0, CUDA 12.8, 20 measured steps per phase, and a 60 ms/read
injected storage delay.

| Phase | Throughput | Data wait | Remote latency | Prefetch depth | GPU util | Diagnosis |
|---|---:|---:|---:|---:|---:|---|
| baseline | 189.2 samples/s | 0.36% | 0.92 ms | 4.0 | 59.0% | none |
| fault | 32.5 samples/s | 87.34% | 61.26 ms | 0.0 | 14.4% | remote input-storage latency |
| recovery | 186.1 samples/s | 0.37% | 0.92 ms | 4.0 | 59.5% | none |

The fault phase produced DPD-001 + DPD-004 + DPD-007 and the correlation layer
reported:

```text
LIKELY ROOT CAUSE
remote input-storage latency is starving the training loop
Confidence: high
```

The fault reduced throughput by 82.8% and GPU utilization by 75.6%. Removing
the fault restored throughput to within 1.7% of baseline.

Raw result:
[`results/a100_http_60ms_2026-09-28.json`](results/a100_http_60ms_2026-09-28.json)


## Measurements

The harness records:

- real `DataLoaderProfiler` batch wait time
- real CUDA training-style compute time
- normalized `RemoteReadCollector` latency/throughput
- best-effort PyTorch prefetch queue depth
- samples/second
- best-effort GPU utilization from `nvidia-smi`
- individual datapath-doctor findings
- evidence-aware correlated diagnosis

Results are written as JSON so they can later be turned into a README table or
paper figure.

## Caveat: prefetch queue depth

The experiment reads PyTorch's private iterator `_data_queue.qsize()` when it
is available. PyTorch does not provide a stable public queue-depth API. The
metric is therefore experiment-only and the harness safely falls back to
`None` if that implementation detail is unavailable.

## Attached-filesystem follow-up

The stronger follow-up has now been completed on the same A100 class using a
real Lambda attached persistent filesystem exposed as `virtiofs`, with direct
random-read `fio` contention against the same storage path used by the training
dataset.

See [`A100_LAMBDA_NFS_VALIDATION.md`](A100_LAMBDA_NFS_VALIDATION.md) for the
measured baseline/fault/recovery result and committed raw artifacts. Future
validation can extend the same methodology to S3/fsspec, Lustre, Weka, or other
production storage backends.
