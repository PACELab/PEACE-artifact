#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/paths.sh"
cd "$REPO_ROOT"
# SoCC'26 Rebuttal: Re-run missing thread combinations for Triton-kernel workloads
#
# Usage:
#   bash run_rebuttal_customkernel_missingthreads.sh <log_dir> [frequency]
#
# Examples:
#   bash run_rebuttal_customkernel_missingthreads.sh ../../ccv100_logs/triton_missing
#   bash run_rebuttal_customkernel_missingthreads.sh ../ccv100_logs/triton_missing 900
#
# Reads missing thread combinations from the CSV specified in
# config_missingthreads_triton.sh and runs Triton-kernel workloads for those
# combinations only (via auto_multiinstance_freqscale_missingthreads.sh).
#
# If [frequency] is given, overrides VALID_FREQUENCY with that single value.
# Otherwise uses VALID_FREQUENCY from config_missingthreads_triton.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <log_dir> [frequency]"
    echo "  e.g.: $0 ../../ccv100_logs/triton_missing"
    echo "  e.g.: $0 ../ccv100_logs/triton_missing 900"
    exit 1
fi

LOG_DIR="$1"
FREQ_OVERRIDE="${2:-}"
CONFIG_DIR="$REPO_ROOT/configs/profiling/shared"

# Backup original config_missingthreads.sh
if [ -f "$CONFIG_DIR/config_missingthreads.sh" ]; then
    cp "$CONFIG_DIR/config_missingthreads.sh" "$CONFIG_DIR/config_missingthreads.sh.bak"
    echo "Backed up config_missingthreads.sh -> config_missingthreads.sh.bak"
fi

# Swap in triton missing-threads config
cp "$CONFIG_DIR/config_rebuttal_customkernel_missingthreads.sh" "$CONFIG_DIR/config_missingthreads.sh"
echo "Using config_rebuttal_customkernel_missingthreads.sh"

# Override VALID_FREQUENCY if a frequency was specified
if [ -n "$FREQ_OVERRIDE" ]; then
    sed -i "s/^VALID_FREQUENCY=.*/VALID_FREQUENCY=$FREQ_OVERRIDE/" "$CONFIG_DIR/config_missingthreads.sh"
    echo "Overriding VALID_FREQUENCY with: $FREQ_OVERRIDE"
fi

# Display what we're about to run
echo "=== Starting Triton-kernel missing-threads experiment ==="
echo "Log directory: $LOG_DIR"
if [ -n "$FREQ_OVERRIDE" ]; then
    echo "Frequency: $FREQ_OVERRIDE"
else
    echo "Frequency: using config default (1530)"
fi
echo "CSV: $(grep '^CSV_FILE=' "$CONFIG_DIR/config_missingthreads.sh" | cut -d= -f2-)"
echo ""

bash "$SCRIPT_DIR/auto_multiinstance_freqscale_missingthreads.sh" "$LOG_DIR"

# Restore original config
if [ -f "$CONFIG_DIR/config_missingthreads.sh.bak" ]; then
    mv "$CONFIG_DIR/config_missingthreads.sh.bak" "$CONFIG_DIR/config_missingthreads.sh"
    echo "Restored original config_missingthreads.sh"
fi

echo "=== Missing-threads experiment complete ==="
