"""datapath-doctor: diagnostics for the storage / data-loading input path of ML training jobs.

Fifth tool in the unified GPU observability platform (torchguard, stallscope,
nccl-doctor, datapath-doctor). Where nccl-doctor diagnoses the collective
communication path, datapath-doctor diagnoses the path that feeds the GPUs
in the first place: disk I/O, the PyTorch DataLoader pipeline, prefetching,
decode/tokenize CPU work, and checkpoint I/O.
"""

from datapath_doctor.models import Finding, Severity, StepSample, WindowSummary
from datapath_doctor.engine import Rule, RuleEngine

__version__ = "0.1.0"

__all__ = [
    "Finding",
    "Severity",
    "StepSample",
    "WindowSummary",
    "Rule",
    "RuleEngine",
    "__version__",
]
