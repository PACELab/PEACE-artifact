#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/paths.sh"
cd "$REPO_ROOT"
# SoCC'26 Rebuttal: Run custom Triton-kernel workloads for MPS sensitivity
#
# Usage:
#   bash run_rebuttal_customkernel.sh <log_dir> [frequency]
#
# Examples:
#   bash run_rebuttal_customkernel.sh ../../ccv100_logs/rebuttal_triton_kernels
#   bash run_rebuttal_customkernel.sh ../ccv100_logs/triton 900
#
# If [frequency] is given, overrides VALID_FREQS with that single value.
# Otherwise uses VALID_FREQS from config_rebuttal_customkernel.sh.
#
# This script swaps in the rebuttal config, runs the experiment, and restores
# the original config.sh afterward.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <log_dir> [frequency]"
    echo "  e.g.: $0 ../../ccv100_logs/rebuttal_triton_kernels"
    echo "  e.g.: $0 ../ccv100_logs/triton 900"
    exit 1
fi

LOG_DIR="$1"
FREQ_OVERRIDE="${2:-}"
CONFIG_DIR="$REPO_ROOT/configs/profiling/shared"

# Backup original config.sh
if [ -f "$CONFIG_DIR/config.sh" ]; then
    cp "$CONFIG_DIR/config.sh" "$CONFIG_DIR/config.sh.bak"
    echo "Backed up config.sh -> config.sh.bak"
fi

# Swap in rebuttal config
cp "$CONFIG_DIR/config_rebuttal_customkernel.sh" "$CONFIG_DIR/config.sh"
echo "Using config_rebuttal_customkernel.sh"

# Override VALID_FREQS if a frequency was specified
if [ -n "$FREQ_OVERRIDE" ]; then
    sed -i "s/^VALID_FREQS=(.*/VALID_FREQS=($FREQ_OVERRIDE)/" "$CONFIG_DIR/config.sh"
    echo "Overriding VALID_FREQS with: ($FREQ_OVERRIDE)"
fi

# Run the experiment
echo "=== Starting custom-kernel MPS sensitivity experiment ==="
echo "Log directory: $LOG_DIR"
if [ -n "$FREQ_OVERRIDE" ]; then
    echo "Frequency: $FREQ_OVERRIDE"
else
    echo "Frequencies: using config defaults"
fi
echo "Workloads: albert-base-v2-triton (train) + mobilenet-triton (inf), resnet-50-triton (train) + wav2vec2-base-960h-triton (inf)"
echo ""

bash "$SCRIPT_DIR/auto_multiinstance_freqscale.sh" "$LOG_DIR"

# Restore original config
if [ -f "$CONFIG_DIR/config.sh.bak" ]; then
    mv "$CONFIG_DIR/config.sh.bak" "$CONFIG_DIR/config.sh"
    echo "Restored original config.sh"
fi

echo "=== Experiment complete ==="
