# Repository layout

The open artifact is organized by pipeline role rather than under a test
directory:

- `runners/baseline/`: single-workload profiling entrypoints.
- `runners/shared/`: colocation profiling entrypoints. Original runner names,
  including the `*_fixedpower.sh` suffixes, are preserved.
- `runners/training/`: training and prediction entrypoints.
- `configs/profiling/shared/`: configurations sourced by shared runners.
- `scripts/parsing/`: baseline and colocation log parsers.
- `scripts/datasets/`: manifest, fold, and dataset preparation utilities.
- `scripts/baselines/`: MuDi, GSlice, and MuxFlow implementations.
- `scripts/evaluation/`: paper-result aggregation.
- `data/experiment_inputs/colocations/`: workload manifests to execute.
- `data/baseline_metrics/`: parsed single-workload profiles.
- `data/colocations/`: parsed colocated measurements formerly called stage 2.
- `data/model_datasets/`: feature-and-label datasets consumed by `main.py`.
- `artifacts/predictions/`: retained model prediction outputs.
- `artifacts/baseline_evaluations/`: retained baseline allocation outputs.
- `results/paper/`: final reported summary CSVs.

Shell entrypoints derive `REPO_ROOT` from `runners/lib/paths.sh`, so they may be
invoked from any working directory. Python evaluation scripts derive paths
from their own location rather than a developer-specific home directory.
