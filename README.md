# PEACE artifact

This repository contains the open artifact for **PEACE: Power and Performance
Aware Colocation for Efficient GPU Spatial Partitioning** (SoCC 2026).

The open-access branch is intentionally limited to the code, processed data,
prediction inputs, baseline allocations, and reported summaries needed to audit
or reproduce the paper results. Historical experiment logs, intermediate plots,
checkpoints, and unrelated development integrations are not included.

## Artifact map

- `main.py`, `utils/`: feature construction, model training, and prediction.
- `tests/mps/`: MPS and DVFS profiling scripts.
- `tests/mps/analysis/kernel_profiles/`: single-workload kernel profiling and
  parsing scripts.
- `tests/mps/analysis/stage2/`: colocation profiling aggregation.
- `tests/mps/freq_scaling/dataset/`: processed training and oracle datasets.
- `tests/mps/freq_scaling/output/`: only the Extra Trees prediction CSVs read by
  the final result parsers. These files were copied into this artifact from the
  `mps_thread` development branch; no second repository is required.
- `tests/mps/eval_baselines/`: Mudi, GSlice, and MuxFlow implementations and
  allocation summaries.
- `tests/mps/analysis/plot_analysis/get_pred_data_mean_multifreqselect.py`:
  final multi-frequency policy analysis for Figures 4--6.
- `tests/mps/analysis/plot_analysis/get_pred_data_mean_crossvalidate.py`:
  single-frequency cross-validation analysis.
- `tests/mps/analysis/plot_analysis/09152025_*`: checked-in reported result
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
python tests/mps/analysis/plot_analysis/get_pred_data_mean_multifreqselect.py \
  --power-limit 60 \
  --output-dir output/reproduced/unseen-powercap60

# Cross-validation results (repeat for 60, 100, and 200 W).
python tests/mps/analysis/plot_analysis/get_pred_data_mean_multifreqselect.py \
  --power-limit 60 \
  --cross-validation \
  --fold-size 7 \
  --output-dir output/reproduced/crossvalid-powercap60
```

The output summaries can be compared with the matching `09152025_*` directory
under `tests/mps/analysis/plot_analysis/`. The parser defaults to a power
epsilon of 10% of the selected cap (6, 10, and 20 W respectively), matching the
reported runs.

The three-workload reported summaries are also retained in the `09152025_*`
directories. Their processed oracle and training datasets are under
`tests/mps/freq_scaling/dataset/09152025_*comb3*`; regenerate their prediction
CSVs with `main.py` before invoking the parser with `--combinations 3`.

## Profiling and model workflow

The end-to-end data flow is:

```text
single-workload profiling
  -> tests/mps/analysis/kernel_profiles
colocation and DVFS profiling
  -> tests/mps/analysis/stage2
processed labels and train/test data
  -> tests/mps/freq_scaling/dataset
main.py training and prediction
  -> tests/mps/freq_scaling/output
final policy analysis
  -> tests/mps/analysis/plot_analysis
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
