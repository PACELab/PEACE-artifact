# PEACE artifact

This repository contains the open artifact for **PEACE: Power and Performance
Aware Colocation for Efficient GPU Spatial Partitioning** (SoCC 2026).

The open-access branch is intentionally limited to the code, processed data,
prediction inputs, baseline allocations, and reported summaries needed to audit
or reproduce the paper results. Historical experiment logs, intermediate plots,
checkpoints, and unrelated development integrations are not included.

## Artifact map

See [`docs/repository-layout.md`](docs/repository-layout.md) for the complete
directory map and data-flow boundaries.

- `main.py`, `utils/`: feature construction, model training, and prediction.
- `runners/`: baseline, shared-workload, training, and evaluation entrypoints.
- `scripts/profiling/`: profiling monitors and helper utilities.
- `data/colocations/`: colocation profiling aggregation.
- `data/model_datasets/`: processed training and oracle datasets.
- `artifacts/predictions/`: only the Extra Trees prediction CSVs read by
  the final result parsers. These files were copied into this artifact from the
  `mps_thread` development branch; no second repository is required.
- `scripts/baselines/`: Mudi, GSlice, and MuxFlow implementations;
  retained allocation summaries are under `artifacts/baseline_evaluations/`.
- `scripts/evaluation/get_pred_data_mean_multifreqselect.py`:
  final multi-frequency policy analysis for Figures 4--6.
- `scripts/evaluation/get_pred_data_mean_crossvalidate.py`:
  single-frequency cross-validation analysis.
- `results/paper/09152025_*`: checked-in reported result
  tables. These are retained as the audit reference.

## Environment

The experiments were developed with Python 3.10 and NVIDIA MPS on V100 GPUs.
GPU profiling additionally requires the NVIDIA tools invoked by the scripts
(`nvidia-smi`, Nsight Systems, and Nsight Compute).

```bash
conda create -n peace python=3.10
conda activate peace
pip install -r requirements.txt
```

WANDB is used by the training workflow. Set credentials in the environment;
do not place them in this repository.

```bash
export WANDB_API_KEY='...'
export WANDB_SILENT=true
```

## Reproduce the reported two-workload summaries

The parser is independent of the current working directory. It accepts the
power cap and whether the checked-in unseen-workload or cross-validation
predictions should be analyzed. The paper uses fold size 7, corresponding to
the approximately 30% training split.

```bash
# Unseen-workload results (repeat for 60, 100, and 200 W).
python scripts/evaluation/get_pred_data_mean_multifreqselect.py \
  --power-limit 60 \
  --output-dir output/reproduced/unseen-powercap60

# Cross-validation results (repeat for 60, 100, and 200 W).
python scripts/evaluation/get_pred_data_mean_multifreqselect.py \
  --power-limit 60 \
  --cross-validation \
  --fold-size 7 \
  --output-dir output/reproduced/crossvalid-powercap60
```

The output summaries can be compared with the matching `09152025_*` directory
under `results/paper/`. The parser defaults to a power
epsilon of 10% of the selected cap (6, 10, and 20 W respectively), matching the
reported runs.

The three-workload reported summaries are also retained in the `09152025_*`
directories. Their processed oracle and training datasets are under
`data/model_datasets/09152025_*comb3*`, and the required prediction
CSVs are under matching directories in `artifacts/predictions/`. Invoke
the parser with `--combinations 3` to reproduce their summaries.

## Profiling and model workflow

Detailed, command-by-command instructions are in
[`docs/profile.md`](docs/profile.md). The guide covers single-workload profiling,
baseline parsing, colocated profiling with and without DVFS, `stage2.py`
aggregation, dataset construction, unseen-workload training, and the paper's
fold-based cross-validation workflow.

The end-to-end data flow is:

```text
single-workload profiling
  -> data/baseline_metrics
colocation and DVFS profiling
  -> data/colocations
processed labels and train/test data
  -> data/model_datasets
main.py training and prediction
  -> artifacts/predictions
final policy analysis
  -> results/paper
```

Common model commands are:

```bash
python main.py --train_file <train.csv> --test_file <test.csv> \
  -t 100 --train

python main.py --train_file <train.csv> --test_file <test.csv> \
  -comb 2 -mt threadclass --predAllacc
```

Profiling scripts encode machine-specific GPU IDs, clock settings, and workload
paths. Review their `config*.sh` files before running them on a new host.
