# Real storage fault-injection validation

datapath-doctor now has two completed NVIDIA A100 baseline/fault/recovery
validations. Raw results are committed so the README claims are reproducible.

## 1. Controlled HTTP/socket-backed input path

Hardware and software:

- NVIDIA A100-SXM4-40GB
- datapath-doctor 0.2.0
- PyTorch 2.7.0
- CUDA 12.8
- 20 measured steps per phase
- 60 ms/read controlled HTTP delay during the fault phase

| Phase | Throughput | Data wait | Remote read latency | Prefetch depth | GPU util | Diagnosis |
|---|---:|---:|---:|---:|---:|---|
| Baseline | 189.2 samples/s | 0.36% | 0.92 ms | 4.0 | 59.0% | none |
| Fault | 32.5 samples/s | 87.34% | 61.26 ms | 0.0 | 14.4% | remote input-storage latency |
| Recovery | 186.1 samples/s | 0.37% | 0.92 ms | 4.0 | 59.5% | none |

The fault phase produced DPD-001, DPD-004, and DPD-007 with a high-confidence
root-cause diagnosis:

```text
remote input-storage latency is starving the training loop
```

Raw result:
[`results/a100_http_60ms_2026-09-28.json`](results/a100_http_60ms_2026-09-28.json)

This run validates the diagnosis on a real A100 training loop, but it is not
presented as production network-filesystem validation because the fault source
was a controlled localhost HTTP/socket-backed path.

## 2. Lambda attached persistent filesystem (virtiofs)

A second A100 run placed the training dataset on a real Lambda persistent
filesystem exposed to the guest as `virtiofs`. The fault phase generated
direct 4 KiB random-read contention with 32 `fio` processes against the same
attached filesystem.

| Phase | Throughput | Data wait | Read latency | Prefetch depth | GPU util | Diagnosis |
|---|---:|---:|---:|---:|---:|---|
| Baseline | 185.7 samples/s | 0.34% | 3.80 ms | 3.4 | 70.0% | none |
| Fault | 67.9 samples/s | 69.85% | 29.17 ms | 0.2 | 28.4% | remote input-storage latency |
| Recovery | 186.7 samples/s | 0.38% | 2.56 ms | 3.65 | 69.0% | none |

The contention workload sustained about 5.2K random-read IOPS at 20.2 MiB/s
with 6.07 ms mean completion latency. During contention:

- throughput fell 63.4%
- GPU utilization fell 59.4%
- training-side read latency increased 7.7x
- the prefetch queue nearly drained
- DPD-001, DPD-004, and DPD-007 produced the same high-confidence root-cause
  diagnosis

After contention stopped, throughput returned to within 0.6% of baseline and
the input path was reported healthy again.

Raw artifacts:

- [result JSON](results/a100_lambda_virtiofs_fio32_2026-09-28.json)
- [fio log](results/a100_lambda_virtiofs_fio32_2026-09-28.fio.log)

The guest-visible filesystem in this run was `virtiofs`, so the result is
described as Lambda attached-filesystem validation rather than making an NFS
transport claim.

## Next validation milestone

The next useful expansion is distributed multi-rank validation where one rank's
input path is selectively degraded and rank identity is retained end-to-end.
That is future work; the completed single-A100 results above are the current
real-workload evidence.
