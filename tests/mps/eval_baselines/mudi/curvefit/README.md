# mudi Curvefit Baselines

This directory packages the utilities that build the MuDi curve-fitting baselines. Use
`kneedle_curvefit.py` to estimate knee points for colocated throughput traces and
`analyze_throughput_sum.py` to summarize the resulting aggregate throughput curves.

## Running the curve-fit script

1. Activate the project environment (see repo root README for details).
2. Execute the kneedle fitter against the colocated throughput averages:

```bash
python kneedle_curvefit.py \
  --input_csv \
    ~/mlProfiler/tests/mps/analysis/stage2/05052025_freq300_share_comb2_freqscale_throughput_individual_avg.csv \
    ~/mlProfiler/tests/mps/analysis/stage2/05052025_nonDL_freq300_nodvfs_share_comb2_freqscale_throughput_individual_avg.csv \
  --output_dir colocated_freq300 \
  --summary_name 05052025_freq300_kneedle_params.csv \
  --n_comb 2 \
  --plot
```

The run writes fitted parameters and diagnostic plots into the `colocated_freq300/` output folder.

## Summarizing throughput totals

After generating the kneedle parameters, aggregate the throughput signals with:

```bash
python analyze_throughput_sum.py
```

Always double-check the arguments expected by `analyze_throughput_sum.py` (e.g., via
`python analyze_throughput_sum.py --help`) so the inputs and output paths align with the dataset you
just produced.
