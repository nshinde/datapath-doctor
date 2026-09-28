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
git checkout a100-validation-harness

python experiments/a100_http_fault.py \
  --steps 20 \
  --warmup-steps 6 \
  --fault-latency-ms 60 \
  --target-compute-ms 40 \
  --output a100_validation_results.json
```

The script prints the installed distribution version and GPU model before it
runs. Keep that header with the experiment result.

## What a useful result looks like

Do **not** hard-code expected numbers. On a successful validation we expect the
directional behavior to look like this:

| Phase | Throughput | Data wait | Remote latency | Expected diagnosis |
|---|---|---|---|---|
| baseline | higher | low | low | healthy / no input bottleneck |
| fault | lower | sharply higher | near injected delay | remote input-storage latency |
| recovery | returns near baseline | low again | low again | healthy / no input bottleneck |

The fault phase is strongest evidence if it produces DPD-001 + DPD-007 and,
when the private PyTorch prefetch queue metric is available, DPD-004 as well.

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

## Stronger follow-up

After the localhost HTTP experiment is reproducible, repeat the same
baseline/fault/recovery methodology with a real network storage backend:

- NFS with controlled network delay/bandwidth
- S3/fsspec with request-level telemetry
- Lustre/Weka/another cluster filesystem if available

That second experiment can support stronger claims about a specific storage
backend.
