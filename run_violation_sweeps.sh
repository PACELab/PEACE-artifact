#!/usr/bin/env bash
set -euo pipefail

POWERcaps=(101 102 104 106 108 110)
WINDOWS=(2 3 4 5 6 7 8 9 10)

LOG_DIR1="/home/cc/mlProfiler/tests/mps/ccv100_logs/sharenonDL_powercap100_dvfs/"
LOG_DIR2="/home/cc/mlProfiler/tests/mps/ccv100_logs/shareDL_powercap100_dvfs/"

OUTPUT_ROOT="violation_sweeps_powercap100"
mkdir -p "$OUTPUT_ROOT"

for cap in "${POWERcaps[@]}"; do
  for window in "${WINDOWS[@]}"; do
    OUTPUT_DIR="$OUTPUT_ROOT/powercap_${cap}_window${window}"
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_FILE="$OUTPUT_DIR/summary.csv"

    python ./calculate_violation.py \
      "$LOG_DIR1" "$LOG_DIR2" \
      --powercap "$cap" \
      --start_delay 60 \
      --window_seconds "$window" \
      --output "$OUTPUT_FILE"
  done
done
