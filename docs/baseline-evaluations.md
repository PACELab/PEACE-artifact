# Baseline evaluation CSVs

This directory contains the baseline allocations consumed by
`scripts/evaluation/get_pred_data_mean_multifreqselect.py`.
The files are generated from the single-workload baseline metrics and the
colocated throughput grids produced by the profiling workflow in
[`docs/profile.md`](../docs/profile.md).

Run all commands below from the repository root:

```bash
cd "$PEACE_ROOT"
conda activate ml
pip install -r requirements.txt
```

The scripts require `pandas`, `numpy`, and `matplotlib`. The profiling and
stage-2 parsing steps must be completed before running these baselines.

## Inputs that must exist first

The checked-in comb2 results use these single-workload files:

```text
data/baseline_metrics/0506_FREQ300_baseline_metrics.csv
data/baseline_metrics/0502_FREQ900_baseline_metrics.csv
data/baseline_metrics/0206_FREQ1530_baseline_metrics.csv
```

They must contain `Type`, `freq`, `Exclusive10` through `Exclusive100`, and
`SMACT%100`. Generate them by following Section 1 of `docs/profile.md`.

The fixed-frequency comb2 baselines also require the DL and CUDA-sample
stage-2 throughput grids below. These are produced by the colocation profiling
and `stage2.py` parsing steps in Sections 2 and 3 of `docs/profile.md`.

| Frequency | Required stage-2 throughput grids |
| --- | --- |
| 300 | `05052025_freq300_share_comb2_freqscale_throughput_individual_avg.csv`, `05052025_nonDL_freq300_nodvfs_share_comb2_freqscale_throughput_individual_avg.csv` |
| 900 | `05052025_freq900_share_comb2_freqscale_throughput_individual_avg.csv`, `05052025_nonDL_freq900_nodvfs_share_comb2_freqscale_throughput_individual_avg.csv` |
| 1530 | `09152025_freq1530_nodvfs_share_comb2_freqscale_throughput_individual_avg.csv`, `0311_freq1530_nonDL_nodvfs_share_comb2_freqscale_throughput_individual_avg.csv` |

All paths in the table are relative to `data/colocations/`.

## MuDi comb2 CSVs

The six retained files are:

```text
mudi/curvefit/summary/cutoff_freq300_optimal_allocation.csv
mudi/curvefit/summary/cutoff_freq900_optimal_allocation.csv
mudi/curvefit/summary/cutoff_freq1530_optimal_allocation.csv
mudi/curvefit/summary/cutoff_freq300_optimal_allocation_latency.csv
mudi/curvefit/summary/cutoff_freq900_optimal_allocation_latency.csv
mudi/curvefit/summary/cutoff_freq1530_optimal_allocation_latency.csv
```

MuDi is a two-step pipeline. First, `kneedle_curvefit.py` fits one piecewise
curve per colocated workload. For example, the 300 MHz throughput fit is:

```bash
python scripts/baselines/mudi/curvefit/kneedle_curvefit.py \
  --input_csv \
    data/colocations/05052025_freq300_share_comb2_freqscale_throughput_individual_avg.csv \
    data/colocations/05052025_nonDL_freq300_nodvfs_share_comb2_freqscale_throughput_individual_avg.csv \
  --output_dir colocated_freq300 \
  --summary_name colocated_freq300.csv \
  --n_comb 2
```

Repeat at 900 and 1530 MHz, substituting the two input files in the table and
changing both occurrences of the frequency in the output names. The summaries
are always written under `mudi/curvefit/summary/`; `--output_dir` matters only
when `--plot` is added.

For the `_latency.csv` variants, run the same fitter on the corresponding
stage-2 latency grids and use `colocated_freq<FREQ>_latency.csv` as the summary
name. Those raw fixed-frequency latency grids are profiling products and are
not part of the minimal retained artifact; regenerate them with `stage2.py`
before attempting to recreate the latency variants.

Second, run the allocation calculation:

```bash
python scripts/baselines/mudi/curvefit/get_optimal_allocation.py --cutoff
```

Important: `get_optimal_allocation.py` currently keeps `freq`, `n_comb`, and
the input filename as configuration constants near the bottom of the script.
Before each run set:

```python
n_comb = 2
freq = 300  # then 900 and 1530
csv_path = base_dir / "summary" / f"colocated_freq{freq}_comb{n_comb}_latency.csv"
```

Set `csv_path` to the actual fitter output (`colocated_freq<FREQ>.csv` or
`colocated_freq<FREQ>_latency.csv`) and set `summary_csv_path` to the matching
checked-in destination. `--cutoff` produces the retained cutoff allocation;
`--xput_sum` is an alternative experiment and does not produce the paper input.

## GSlice comb2 CSVs

The retained outputs are:

```text
gslice/summary/colocated_freq300.csv
gslice/summary/colocated_freq900.csv
gslice/summary/colocated_freq1530.csv
```

GSlice needs three kinds of input: the colocated throughput grids, a
single-workload knee/cap file containing `Type` and `delta0`, and the baseline
metrics file containing `Type` and `Exclusive100`. Generate each cap file from
the appropriate single-workload baseline file:

```bash
python scripts/baselines/mudi/curvefit/kneedle_curvefit.py \
  --input_csv data/baseline_metrics/0506_FREQ300_baseline_metrics.csv \
  --summary_name single_freq300.csv \
  --n_comb 1
```

Then generate the 300 MHz result:

```bash
python scripts/baselines/gslice/gslice_allocator.py \
  --throughput_csv \
    data/colocations/05052025_freq300_share_comb2_freqscale_throughput_individual_avg.csv \
    data/colocations/05052025_nonDL_freq300_nodvfs_share_comb2_freqscale_throughput_individual_avg.csv \
  --caps_csv artifacts/baseline_evaluations/mudi/curvefit/summary/single_freq300.csv \
  --arrival_csv data/baseline_metrics/0506_FREQ300_baseline_metrics.csv \
  --n_comb 2 \
  --summary_name colocated_freq300.csv \
  --fig_dir colocated_freq300 \
  --log_level WARNING
```

Repeat for 900 and 1530 MHz with the corresponding baseline and stage-2 files.
The output is written automatically to `gslice/summary/`. Diagnostic PNGs are
written beneath `gslice/fig/` and do not need to be retained.

## MuxFlow comb2 CSVs

The retained outputs are:

```text
muxflow/final-powercap60_result_optimal_percentages.csv
muxflow/final-powercap100_result_optimal_percentages.csv
muxflow/final-powercap200_result_optimal_percentages.csv
```

`scripts/baselines/muxflow/muxflow.py` combines four prerequisite data sources:

1. the three single-workload baseline metric CSVs listed above;
2. the 1530 MHz DL and CUDA-sample stage-2 throughput grids, used to classify
   online and offline workloads;
3. the measured power-cap label CSV for the requested cap under
   `data/model_datasets/`;
4. the PEACE selection summary named
   `summary_xput_under_powercap_all_pairs.csv`, generated first with
   `get_pred_data_mean_multifreqselect.py` for the same cap and epsilon.

For V100, use these measured label files:

| Cap | Measured label CSV |
| --- | --- |
| 60 W | `05052025_mergecudaDL_powercap60_dvfs/0505_nonDL_powercap60_dvfs_throughput_total_labels_comb2_labels.csv` |
| 100 W | `05052025_nonDL_09152025_DL_mergecudaDL_powercap100_dvfs/merged_labels.csv` |
| 200 W | `09102025_mergecudaDL_powercap200_dvfs/merged_labels.csv` |

The paths are relative to `data/model_datasets/`.

The script is configuration-driven rather than argument-driven. Its active
block currently targets RTX A6000. To reproduce the retained V100 files,
enable the V100 block at the top of `muxflow.py`, disable the RTX A6000 block,
and set `powercap_limit` and `power_epsilon` for each run. Repository inputs
are resolved through `REPO_ROOT`. Run it from its directory because its
intermediate filenames are relative:

```bash
cd "$PEACE_ROOT"/artifacts/baseline_evaluations/muxflow
python muxflow.py
```

The script first writes `powercap<CAP>_result_optimal_percentages.csv` and then
writes the paper input under
`v100/final-powercap<CAP>_result_optimal_percentages.csv`. Copy that final file
to the retained top-level `muxflow/` filename shown above. Debug intermediates
such as `updated_target_df.csv` and `final_target_df.csv` are not required.

## MuDi comb3 CSVs

The retained power-cap outputs are:

```text
mudi/mudi_comb3_final_powercap60_v2.csv
mudi/mudi_comb3_final_powercap100.csv
mudi/mudi_comb3_final_powercap200.csv
```

Their pipeline is:

1. Run `kneedle_curvefit.py --n_comb 3` on each fixed-frequency comb3
   throughput grid (`09152025_freq300_DL_comb3...`,
   `09152025_freq900_DL_comb3...`, and
   `09152025_freq1530_DL_comb3...`).
2. Run `get_optimal_allocation.py --cutoff` with `n_comb = 3` and each
   frequency to obtain an optimal split per workload triple.
3. Run `find_missing_comb3_threads.py` to join the selected frequency, optimal
   split, measured throughput, and power for each cap.

Example first-stage command:

```bash
python scripts/baselines/mudi/curvefit/kneedle_curvefit.py \
  --input_csv data/colocations/09152025_freq300_DL_comb3_share_comb3_freqscale_throughput_individual_avg.csv \
  --output_dir colocated_comb3_freq300 \
  --summary_name colocated_freq300_comb3.csv \
  --n_comb 3
```

Before the final join, edit the configuration constants at the top of
`find_missing_comb3_threads.py`:

- set `baseline = "mudi"`;
- point `FREQ_FILE_PATHS` at the three generated MuDi allocation summaries;
- point each `SELECTION_CONFIGS[*]["path"]` at the corresponding comb3
  `summary_xput_under_powercap_all_pairs.csv` from
  `get_pred_data_mean_multifreqselect.py`;
- verify each `metrics_path` points to the retained 60, 100, and 200 W comb3
  dataset CSVs.

Then run:

```bash
python scripts/baselines/find_missing_comb3_threads.py
```

The script's configured `output` names should be changed to the three retained
`mudi_comb3_final_powercap*.csv` names above. The `_v2` suffix is part of the
retained 60 W filename only.

## Validate the retained inputs

The paper parser should be run after regenerating a baseline CSV because it
checks the joins against the retained workload datasets:

```bash
python scripts/evaluation/get_pred_data_mean_multifreqselect.py \
  --combinations 2 --power-limit 100

python scripts/evaluation/get_pred_data_mean_multifreqselect.py \
  --combinations 3 --power-limit 100
```

Use the parser's `--help` output for the supported cap, epsilon, and evaluation
mode options. Generated figures, debug CSVs, curve-fit intermediates, and ratio
reports are not required inputs to the final paper parser.
