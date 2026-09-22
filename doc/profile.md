# Profiling and dataset-generation guide

This guide documents the profiling pipeline used to produce the processed data
and prediction inputs in the PEACE artifact. It is for regenerating data on an
NVIDIA GPU host; reproducing the paper summaries from the checked-in prediction
CSVs does not require rerunning GPU profiling.

Run commands from the repository root unless a command begins with `cd`. The
examples assume the checkout is available as `$PEACE_ROOT`:

```bash
export PEACE_ROOT=/path/to/PEACE-artifact
cd "$PEACE_ROOT"
```

## Prerequisites and machine-specific settings

The profiling scripts expect Linux, Docker with NVIDIA Container Toolkit,
NVIDIA MPS, `nvidia-smi`, and DCGM (`dcgmi`). The workload images and model
caches must be available locally. Several scripts use `sudo` to set clocks,
enable persistence mode, stop monitors, or clean up profiling processes.

Before each experiment, inspect the script and its sourced configuration file.
The checked-in configurations preserve the experiment host's paths and are not
portable without editing. In particular, verify:

- `DEVICE_NUM` or `NVIDIA_VISIBLE_DEVICES` in the baseline scripts;
- `CSV_FILE`, `N_COMBINATIONS`, `VALID_FREQS`, `RUNS`, `TIMEOUT`, and
  `SLEEP_TIMES` in `tests/mps/freq_scaling/config*.sh`;
- `GPU_POWERCAP_VALUE` in `config_gpumonitor.sh` or
  `config_nonDL_gpumonitor.sh` for fixed-power runs;
- the Docker image names, repository bind mount, workload/model maps, and MPS
  percentage list;
- that the input workload CSV uses the same workload names and combination
  count as the selected configuration.

The frequency scripts source configuration files by relative name, so invoke
them from `tests/mps/freq_scaling`. The baseline scripts use paths relative to
`tests/mps`, so invoke them from that directory.

## Pipeline overview

```text
single-workload execution logs
  -> baseline_metrics/*.csv
colocated execution logs
  -> analysis/stage2/*.csv
baseline features + stage2 labels
  -> freq_scaling/dataset/*_total_labels_comb{2,3}.csv
training and prediction
  -> freq_scaling/output/.../predicts_by_workloads/*.csv
paper result parser
  -> summary CSVs
```

Single-workload profiling here means measuring complete workload executions at
fixed GPU clocks and MPS allocations. It is distinct from kernel profiling with
Nsight Compute or Nsight Systems.

## 1. Profile single workloads

### 1.1 Deep-learning workloads

`baseline_freqscale.sh` profiles the configured training or inference workload
set. Its positional arguments are:

```text
baseline_freqscale.sh LOG_DIR ARRIVAL_FILE LS_MODE BE_MODE
```

`LS_MODE` and `BE_MODE` must each be `train` or `inference`. The current script
uses `VALID_FREQS`, `DEVICE_NUM`/GPU 1, workload arrays, batch sizes, MPS
percentages, and timeout values defined in the script. `ARRIVAL_FILE` is kept
only as a positional placeholder and is not consumed by this workflow. It does
not need to name a real file; pass any arbitrary string such as `unused`.

Example training and inference runs:

```bash
cd "$PEACE_ROOT/tests/mps"

# Edit VALID_FREQS and DEVICE_NUM in baseline_freqscale.sh first.
bash baseline_freqscale.sh \
  ccv100_logs/baseline_DL_train \
  unused \
  train train

bash baseline_freqscale.sh \
  ccv100_logs/baseline_DL_inf \
  unused \
  inference inference
```

The script writes a hierarchy such as
`LOG_DIR/FREQ1530/RUN1/LS0/.../BE_<model>_batch<batch>/`. Each leaf contains the
workload log plus GPU memory, process-monitor, and DCGM metric files.

### 1.2 CUDA samples and custom workloads

Use `baseline_freqscale_customscript.sh` for CUDA samples. It has the same four
positional arguments, and its second argument is likewise unused: pass any
arbitrary placeholder string. Select the samples in `train_candidates`, set
`VALID_FREQS` and `DEVICE_NUM`, and use `train` as `BE_MODE`; the current custom
script launches CUDA samples through that branch.

```bash
cd "$PEACE_ROOT/tests/mps"

bash baseline_freqscale_customscript.sh \
  ccv100_logs/baseline_nonDL \
  unused \
  inference train
```

The checked-in script currently includes `sortingNetworks`, `transpose`,
`reductionMultiBlockCG`, `fastWalshTransform`, and `cudaTensorCoreGemm`.

### 1.3 Parse baseline metrics

The parser expects a root containing workload-type subdirectories named
`train`, `inf`, `autoregressive`, and/or `cuda_samples`, with `FREQ<number>`
below them. It scans the frequencies in `idle_power_dict` and skips missing
inputs. Run it from `baseline_metrics` because its imports are relative:

```bash
cd "$PEACE_ROOT/tests/mps/freq_scaling/baseline_metrics"

python parse_baseline_sysmetrics_freqscale.py \
  ../../ccv100_logs/baseline_nonDL/ \
  0505_cudasample
```

This produces `0505_cudasample_baseline_metrics.csv` in the current directory.
The older README showed a trailing `2`; the parser only reads the first two
arguments, so the combination count is not required here. Checked-in examples
include `0506_FREQ300_baseline_metrics.csv`,
`0502_FREQ900_baseline_metrics.csv`, and
`0206_FREQ1530_baseline_metrics.csv`.

Inspect the resulting CSV before continuing:

```bash
python - <<'PY'
import pandas as pd

path = "0505_cudasample_baseline_metrics.csv"
df = pd.read_csv(path)
print(df.shape)
print(df.columns.tolist())
print(df.head())
PY
```

## 2. Profile colocated workloads

All four scripts below read workload combinations from `CSV_FILE` in their
sourced configuration and create per-run/per-allocation log trees under the
single `LOG_DIR` positional argument.

### 2.1 Prepare multiple-workload colocation data

Before running an `auto_multiinstance` script, generate the workload
combination CSV that the script will iterate over. Use
`tests/mps/multiinstance/get_all_combinations.py` with the single-workload
baseline metrics produced in Section 1:

```text
python get_all_combinations.py \
  --baseline_file BASELINE_METRICS.csv \
  --num_combinations N \
  --output_prefix OUTPUT_PREFIX
```

Run the command from `tests/mps/multiinstance`, because the generated CSV is
written to the current directory. For example, given a 300 MHz baseline file
containing the 11 batch-2 DL workloads:

```bash
cd "$PEACE_ROOT/tests/mps/multiinstance"

python get_all_combinations.py \
  --baseline_file ../freq_scaling/baseline_metrics/0506_FREQ300_DL_baseline_metrics.csv \
  --num_combinations 2 \
  --output_prefix 05052025_freq300_DL
```

The output filename is assembled automatically from the prefix, combination
count, and selected batch sizes:

```text
05052025_freq300_DL_baseline_labels_comb2_batches2.csv
```

The corresponding checked-in example is
`tests/mps/multiinstance/05052025_freq300_DL_baseline_labels_comb2_batches2.csv`.
It contains 66 rows: all combinations with replacement of the 11 DL workloads.
By default, repeated workloads such as `(resnet-50, resnet-50)` are included.
Add `--nonrepetitive` to use combinations without replacement and exclude such
self-pairs.

`get_all_combinations.py` selects batch size 2 by default. The batch-size
argument is not needed for the paper workflow. It does not filter by workload
family. Therefore:

- for `auto_multiinstance_freqscale*.sh`, pass a baseline CSV containing only
  the DL workloads intended for the experiment;
- for `auto_multiinstance_nonDL_freqscale*.sh`, pass a baseline CSV containing
  the desired DL and CUDA-sample workloads;
- for three-way colocations, use `--num_combinations 3` and set
  `N_COMBINATIONS=3` in the matching profiling configuration.

The illustrative `0506_FREQ300_DL_baseline_metrics.csv` input name above refers
to the DL-only result of Section 1. If DL and CUDA rows were parsed into the
same baseline file—such as the checked-in
`0506_FREQ300_baseline_metrics.csv`—either use that combined file for the
non-DL experiment or first create a DL-only baseline CSV. Do not give an
all-workload baseline file a `*_DL_*` output prefix without filtering it.

After generation, set `CSV_FILE` in the relevant configuration to the new
absolute or repository-relative path and make its `N_COMBINATIONS` match:

```bash
# Example settings in tests/mps/freq_scaling/config.sh
CSV_FILE="$PEACE_ROOT/tests/mps/multiinstance/05052025_freq300_DL_baseline_labels_comb2_batches2.csv"
N_COMBINATIONS=2
```

Check the workload columns before launching a long profiling run:

```bash
python - <<'PY'
import pandas as pd

path = "05052025_freq300_DL_baseline_labels_comb2_batches2.csv"
df = pd.read_csv(path)
print("rows:", len(df))
print(df[["workload1", "workload2"]].head(10))
PY
```

### 2.2 Without DVFS power-cap monitoring

Use the DL script when every member of a colocation is a deep-learning
workload:

```bash
cd "$PEACE_ROOT/tests/mps/freq_scaling"

# Edit config.sh: CSV_FILE, N_COMBINATIONS, VALID_FREQS, and GPU settings.
bash auto_multiinstance_freqscale.sh \
  ../ccv100_logs/colocated_DL_nodvfs
```

Use the non-DL script for DL + CUDA-sample combinations. Its optional second
argument overrides `VALID_FREQS`; it accepts a comma- or space-separated list.

```bash
cd "$PEACE_ROOT/tests/mps/freq_scaling"

# Edit config_nonDL.sh first.
bash auto_multiinstance_nonDL_freqscale.sh \
  ../ccv100_logs/colocated_nonDL_nodvfs \
  300,900,1530
```

These are called “without DVFS” in the artifact because they collect a fixed
clock sweep without the fixed-power GPU-monitor policy. Each frequency still
comes from `VALID_FREQS` or the command-line override.

### 2.3 With DVFS under a fixed power cap

The fixed-power scripts source the `*_gpumonitor.sh` configurations. Set
`GPU_POWERCAP_VALUE` to the desired cap (for example 60, 100, or 200 W), select
the frequencies in `VALID_FREQS`, and verify the input combination CSV.

For all-DL colocations:

```bash
cd "$PEACE_ROOT/tests/mps/freq_scaling"

# Edit config_gpumonitor.sh, especially GPU_POWERCAP_VALUE.
bash auto_multiinstance_freqscale_gpumonitor_fixedpower.sh \
  ../ccv100_logs/colocated_DL_powercap60_dvfs
```

For DL + CUDA-sample colocations:

```bash
cd "$PEACE_ROOT/tests/mps/freq_scaling"

# Edit config_nonDL_gpumonitor.sh, especially GPU_POWERCAP_VALUE.
bash auto_multiinstance_nonDL_freqscale_gpumonitor_fixedpower.sh \
  ../ccv100_logs/colocated_nonDL_powercap60_dvfs
```

Do not run two profiling scripts against the same GPU simultaneously. After a
run, verify that each expected workload/allocation leaf contains workload logs
and GPU-monitor output before parsing it.

### 2.4 Parse colocation logs with `stage2.py`

`stage2.py` accepts:

```text
python stage2.py INPUT_LOG_DIR OUTPUT_PREFIX NUM_COMBINATIONS [METRIC_TYPE]
```

`METRIC_TYPE` is `throughput` by default and may be set to `latency`. The script
opens `baseline_steps_stage2.csv` relative to the working directory, so run it
from `tests/mps/analysis/stage2`. It writes its CSVs to that same directory.

Two-workload, no-DVFS example:

```bash
cd "$PEACE_ROOT/tests/mps/analysis/stage2"

python stage2.py \
  ../../ccv100_logs/colocated_DL_nodvfs \
  05052025_freq300 \
  2 \
  throughput
```

Two-workload, fixed-power DVFS example:

```bash
python stage2.py \
  ../../ccv100_logs/colocated_DL_powercap60_dvfs \
  05052025_powercap60_DL_dvfs \
  2 \
  throughput
```

Three-workload example:

```bash
python stage2.py \
  ../../ccv100_logs/colocated_DL_comb3_freq300 \
  09152025_freq300_DL_comb3 \
  3 \
  throughput
```

For prefix `PREFIX` and combination count `N`, the important outputs are:

```text
PREFIX_share_combN_freqscale_throughput_individual_avg.csv
PREFIX_share_combN_freqscale_power_avg_stage2.csv
PREFIX_share_combN_freqscale_duration_avg_stage2.csv
PREFIX_share_combN_freqscale_energy_avg_stage2.csv
```

The parser also produces sum, standard-deviation, and provenance files. If
latency labels are needed, repeat with `latency` and retain the corresponding
individual-average output.

## 3. Build model datasets

`main.py -pd2` joins single-workload features and baseline measurements to the
colocated throughput, power, duration, and energy labels produced above.

### 3.1 Select the correct single-profile feature file

Before running `main.py -pd2`, open `getstage2Data()` in `main.py` and set its
`kernelData` argument to exactly one feature file matching the experiment's
frequency and combination count. The repository retains commented examples for
300, 900, and 1530 MHz and combination 2 or 3. This selection is currently a
source-code setting rather than a command-line flag.

Also ensure that:

- `--baseline_file` uses the same frequency as `kernelData`;
- all four stage-2 inputs share the same prefix and combination count;
- `-comb` matches the workload count in the stage-2 files;
- `--label_policy separate_throughputpower_regression` is used for the paper's
  throughput and power models.

### 3.2 Two-workload fixed-frequency example

This example mirrors the argument records under
`tests/mps/freq_scaling/dataset/05052025_FREQ300_*`:

```bash
cd "$PEACE_ROOT"

python main.py -pd2 \
  -sd tests/mps/analysis/stage2/05052025_nonDL_freq300_nodvfs_share_comb2_freqscale_throughput_individual_avg.csv \
  -power tests/mps/analysis/stage2/05052025_nonDL_freq300_nodvfs_share_comb2_freqscale_power_avg_stage2.csv \
  -duration tests/mps/analysis/stage2/05052025_nonDL_freq300_nodvfs_share_comb2_freqscale_duration_avg_stage2.csv \
  -energy tests/mps/analysis/stage2/05052025_nonDL_freq300_nodvfs_share_comb2_freqscale_energy_avg_stage2.csv \
  -comb 2 \
  -t 100 \
  -mt threadclass \
  --output_prefix 0505_FREQ300_nodvfs \
  --output_dir tests/mps/freq_scaling/dataset/05052025_FREQ300_nonDL_nodvfs \
  --label_policy separate_throughputpower_regression \
  --baseline_file tests/mps/freq_scaling/baseline_metrics/0506_FREQ300_baseline_metrics.csv
```

### 3.3 Two-workload fixed-power DVFS example

```bash
python main.py -pd2 \
  -sd tests/mps/analysis/stage2/05052025_powercap60_DL_dvfs_share_comb2_freqscale_throughput_individual_avg.csv \
  -power tests/mps/analysis/stage2/05052025_powercap60_DL_dvfs_share_comb2_freqscale_power_avg_stage2.csv \
  -duration tests/mps/analysis/stage2/05052025_powercap60_DL_dvfs_share_comb2_freqscale_duration_avg_stage2.csv \
  -energy tests/mps/analysis/stage2/05052025_powercap60_DL_dvfs_share_comb2_freqscale_energy_avg_stage2.csv \
  -comb 2 \
  -t 100 \
  -mt threadclass \
  --output_prefix 0505_FREQ300_dvfs \
  --output_dir tests/mps/freq_scaling/dataset/05052025_FREQ300_DL_powercap60_dvfs \
  --label_policy separate_throughputpower_regression \
  --baseline_file tests/mps/freq_scaling/baseline_metrics/0506_FREQ300_baseline_metrics.csv
```

### 3.4 Three-workload example

Select the 300 MHz combination-3 `kernelData` line in `main.py`, then run:

```bash
python main.py -pd2 \
  -sd tests/mps/analysis/stage2/09152025_freq300_DL_comb3_share_comb3_freqscale_throughput_individual_avg.csv \
  -power tests/mps/analysis/stage2/09152025_freq300_DL_comb3_share_comb3_freqscale_power_avg_stage2.csv \
  -duration tests/mps/analysis/stage2/09152025_freq300_DL_comb3_share_comb3_freqscale_duration_avg_stage2.csv \
  -energy tests/mps/analysis/stage2/09152025_freq300_DL_comb3_share_comb3_freqscale_energy_avg_stage2.csv \
  -comb 3 \
  -t 100 \
  -mt threadclass \
  --output_prefix 09152025_freq300_DL_comb3 \
  --output_dir tests/mps/freq_scaling/dataset/09152025_freq300_DL_comb3 \
  --label_policy separate_throughputpower_regression \
  --baseline_file tests/mps/freq_scaling/baseline_metrics/0506_FREQ300_baseline_metrics.csv
```

Each run writes both a full feature-and-label CSV and a `*_labels.csv` file,
plus `*_args.txt` recording the parsed arguments. Compare column names and row
counts with the checked-in examples before training:

```bash
python - <<'PY'
import pandas as pd

path = "tests/mps/freq_scaling/dataset/09152025_freq300_DL_comb3/09152025_freq300_DL_comb3_throughput_total_labels_comb3.csv"
df = pd.read_csv(path)
print("rows, columns:", df.shape)
print("workloads:", [c for c in df if c.startswith("workload")])
print("labels present:", {"weight_Throughput_Sum", "Power"}.issubset(df.columns))
print(df[["workload1", "workload2", "workload3", "weight_Throughput_Sum", "Power"]].head())
PY
```

Some paper datasets combine DL and CUDA-sample rows collected on different
dates. `tests/mps/freq_scaling/dataset/merge_cudaDL.py` contains that historical
merge workflow. Verify its input paths and output schema before using it; do not
merge datasets with different feature columns or frequencies.

## 4. Train models and generate predictions

The paper result parser reads Extra Trees prediction files named:

```text
throughput/.../predicts_by_workloads/pred_separate_throughputpower_regression.csv
power/.../predicts_by_workloads/pred_power_regression.csv
```

The wrapper scripts below partition data, train the models, and write this
layout. Both wrappers contain experiment-specific path variables. Update those
variables before running them; the checked-in defaults include historical
absolute paths.

### 4.1 Unseen-workload partition

Edit `tests/mps/multiinstance/run_unseen_partitions.sh`:

- set `n_combination` to `2` or `3`;
- set `result_output_dir` to the matching frequency output root;
- in `unseen_partitions.sh`, verify the input dataset path used for the run;
- retain `extratrees`, `train`, and `predict` for the paper output format.

Then run from `tests/mps/multiinstance`, because the wrapper invokes
`unseen_partitions.sh` by relative path:

```bash
cd "$PEACE_ROOT/tests/mps/multiinstance"
bash run_unseen_partitions.sh
```

For combination 3, run once per fixed-frequency dataset with output roots:

```text
../freq_scaling/output/09152025_freq300_DL_comb3
../freq_scaling/output/09152025_freq900_DL_comb3
../freq_scaling/output/09152025_freq1530_DL_comb3
```

The final paper parser consumes the resulting `unseen_partition/throughput`
and `unseen_partition/power` trees.

### 4.2 Seen-workload cross-validation

Edit `run_train_splits_mpsthreads_crossvalidate.sh` so that `datapath`,
`total_workload_path`, and `output_result_dir` all refer to the same frequency
dataset. The wrapper creates workload folds first, then trains throughput and
power models for each selected rotation.

The paper result artifacts use Extra Trees, random seed 10, and seven test
folds out of ten. To reproduce only that configuration, set:

```bash
NUM_TEST_FOLD=(7)
```

and leave only the two active `extratrees` calls—one for `throughput`, one for
`power`. Then run:

```bash
cd "$PEACE_ROOT/tests/mps/multiinstance"
bash run_train_splits_mpsthreads_crossvalidate.sh
```

Repeat after selecting the 300, 900, and 1530 MHz dataset/output pairs. The
expected prediction leaves are under paths of this form:

```text
tests/mps/freq_scaling/output/<experiment>/seen_partition/crossvalid/
  throughput/trainratio_/rand10/fold_7_test_<rotation>/extratrees/
  power/trainratio_/rand10/fold_7_test_<rotation>/extratrees/
```

### 4.3 Validate prediction coverage

For unseen combination-3 results, each retained paper frequency contains 11
throughput and 11 power prediction CSVs:

```bash
for freq in 300 900 1530; do
  root="tests/mps/freq_scaling/output/09152025_freq${freq}_DL_comb3"
  find "$root/unseen_partition/throughput" \
    -name pred_separate_throughputpower_regression.csv | wc -l
  find "$root/unseen_partition/power" \
    -name pred_power_regression.csv | wc -l
done
```

For the retained combination-2 cross-validation data, each frequency/metric
has ten `fold_7_test_*` rotations.

## 5. Generate final summaries

Unseen combination-2 example:

```bash
cd "$PEACE_ROOT"
python tests/mps/analysis/plot_analysis/get_pred_data_mean_multifreqselect.py \
  --combinations 2 \
  --power-limit 60 \
  --output-dir output/reproduced/comb2-unseen-powercap60
```

Seen combination-2, seven-fold example:

```bash
python tests/mps/analysis/plot_analysis/get_pred_data_mean_multifreqselect.py \
  --combinations 2 \
  --cross-validation \
  --fold-size 7 \
  --power-limit 60 \
  --output-dir output/reproduced/comb2-crossvalid-powercap60
```

Unseen combination-3 example:

```bash
python tests/mps/analysis/plot_analysis/get_pred_data_mean_multifreqselect.py \
  --combinations 3 \
  --power-limit 60 \
  --output-dir output/reproduced/comb3-unseen-powercap60
```

Compare generated summaries with the matching `09152025_*` result directories
under `tests/mps/analysis/plot_analysis`.

## Troubleshooting checklist

- A missing `config.sh` error usually means a frequency-scaling script was run
  outside `tests/mps/freq_scaling`.
- A missing `baseline_steps_stage2.csv` error means `stage2.py` was run outside
  `tests/mps/analysis/stage2`.
- Empty stage-2 output usually means the log hierarchy, workload spelling,
  frequency, or combination count differs from the parser's expectations.
- Missing feature rows during dataset construction usually indicate that
  `kernelData`, `--baseline_file`, and the stage-2 workload names do not match.
- Missing prediction leaves usually indicate that only throughput or only
  power training was run, or that the wrapper output root does not match the
  final parser's configured experiment directory.
