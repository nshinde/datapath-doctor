# Real storage fault-injection validation

The repository currently does **not** claim a real A100/network-storage result.

A useful first validation run should compare three phases on the same training
workload:

1. baseline storage/input path
2. injected remote-storage latency or bandwidth throttling
3. recovery after removing the fault

Collect per-rank datapath-doctor samples in all phases. Report at minimum:

| Phase | Throughput | Mean data wait | Slowest rank | Remote latency | Diagnosis |
|---|---:|---:|---:|---:|---|
| Baseline | TBD | TBD | TBD | TBD | TBD |
| Fault injected | TBD | TBD | TBD | TBD | TBD |
| Recovery | TBD | TBD | TBD | TBD | TBD |

For distributed training, retain rank identity so the run can demonstrate both
the local storage diagnosis and whether one rank became the input-path
straggler. Do not replace the TBD values with synthetic numbers.
