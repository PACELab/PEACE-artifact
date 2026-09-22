#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
base_powercap=$1
#update POWERcaps to be relative to base_freq
#cap_increment=(1 2 4 6 8 10 12 14 16 18 20)
cap_increment=(0)
#POWERcaps = base_freq + cap_increment
POWERcaps=()
for inc in "${cap_increment[@]}"; do
  POWERcaps+=($((base_powercap + inc)))
done
#echo "${POWERcaps[@]}"

#POWERcaps=(101 102 104 106 108 110)
WINDOWS=(2 3 4 5 6 7 8 9 10)

LOG_DIR1="$REPO_ROOT/ccv100_logs/sharenonDL_powercap${base_powercap}_dvfs/"
LOG_DIR2="$REPO_ROOT/ccv100_logs/shareDL_powercap${base_powercap}_dvfs/"

OUTPUT_ROOT="powercap${base_powercap}_dvfs"
mkdir -p "$OUTPUT_ROOT"

for cap in "${POWERcaps[@]}"; do
  for window in "${WINDOWS[@]}"; do
    OUTPUT_DIR="$OUTPUT_ROOT/powercap_${cap}_window${window}"
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_FILE="$OUTPUT_DIR/summary.csv"

    python "$SCRIPT_DIR/calculate_violation.py" \
      "$LOG_DIR1" "$LOG_DIR2" \
      --powercap "$cap" \
      --start_delay 60 \
      --window_seconds "$window" \
      --output "$OUTPUT_FILE"
  done
done
