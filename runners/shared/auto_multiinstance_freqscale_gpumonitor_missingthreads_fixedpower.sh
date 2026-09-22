#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/paths.sh"
cd "$REPO_ROOT"

# Source the configuration file
source "$REPO_ROOT/configs/profiling/shared/config_gpumonitor_missingthreads.sh"
echo "timeouts: $TIMEOUT"
LOG_DIR=$1
sudo nvidia-smi -pm 1

# ------------------------------------------------------------------------------
# 1) FUNCTIONS FOR MPS COMBINATIONS
#skip for this script since it is provided in the csv file
# ------------------------------------------------------------------------------




wait_any_with_timeout() {
    local timeout=$1
    shift
    local pids=("$@")
    local start_time=$(date +%s)
    local end_time=$((start_time + timeout))

    while (( $(date +%s) < end_time )); do
        for pid in "${pids[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "Process $pid has finished."
                return 0
            fi
        done
        sleep 1
    done
    echo "Timeout of $timeout seconds reached."
    return 1
}



# ------------------------------------------------------------------------------
# 3) HELPER TO PARSE A WORKLOAD
# ------------------------------------------------------------------------------
parse_workload() {
    local workload=$1
    local workload_name=""
    local batchsize=""
    local mode=""
    local prefix=""
    local task=""

    if [[ $workload =~ -(train|inf)$ ]]; then
        mode=${BASH_REMATCH[1]}
    fi
    if [[ $workload =~ _batch([0-9]+) ]]; then
        batchsize=${BASH_REMATCH[1]}
    fi
    workload_name=${workload/_batch${batchsize}-${mode}/}
    prefix=${workloadprefix[$workload_name]}
    task=${task_model[$workload_name]}

    echo "$workload_name,$batchsize,$mode,$prefix,$task"
}

# ------------------------------------------------------------------------------
# 4) MAIN LOOP OVER CSV ROWS (header-driven, N_COMBINATIONS-aware, fixed power)
# ------------------------------------------------------------------------------

# Build header-based column indices so we can read w{i}_percentage directly and
# avoid any dependence on the Thread_combination column.

FIELD_SEPARATOR=$'\x1f'  # Non-whitespace delimiter to preserve empty fields

# Parse header fields (tab-separated) using CSV-aware awk tokenizer
IFS="$FIELD_SEPARATOR" read -r -a header_columns < <(
    awk -v FIELD_SEP="$FIELD_SEPARATOR" '
         BEGIN { FPAT = "([^,]*)|(\"[^\"]*\")"; OFS = FIELD_SEP }
         NR==1 {
             for (i = 1; i <= NF; i++) { gsub(/^\"|\"$/, "", $i) }
             for (i = 1; i <= NF; i++) { printf "%s%s", $i, (i<NF ? FIELD_SEP : ORS) }
         }' "$CSV_FILE"
)

# Helper: find the index of a column by its exact header name
find_col_index() {
    local name="$1"
    local idx
    for idx in "${!header_columns[@]}"; do
        if [[ "${header_columns[$idx]}" == "$name" ]]; then
            echo "$idx"
            return 0
        fi
    done
    echo -1
    return 1
}

# Helper: treat blank, placeholder, or NA-like strings as missing values
is_missing_field() {
    local value="$1"
    # Trim leading/trailing whitespace
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    local normalized="${value^^}"
    if [[ -z "$value" || "$normalized" == "NA" || "$normalized" == "N/A" || \
          "$normalized" == "NULL" || "$normalized" == "NONE" || "$normalized" == "NAN" ]]; then
        return 0
    fi
    return 1
}

# Resolve indices for workload and percentage columns by header names
workload_col_indices=()
percent_col_indices=()
for (( i=1; i<=N_COMBINATIONS; i++ )); do
    widx=$(find_col_index "Workload${i}")
    pidx=$(find_col_index "w${i}_percentage")
    if (( widx < 0 || pidx < 0 )); then
        echo "Missing expected header(s) for Workload${i} or w${i}_percentage in $CSV_FILE"
        exit 1
    fi
    workload_col_indices+=("$widx")
    percent_col_indices+=("$pidx")
done

weight_throughput_index=$(find_col_index "weight_Throughput_sum")
power_index=$(find_col_index "Power")

# CSV contains a quoted field (Thread_combination) with commas; we ignore it
# and read percentages from w{i}_percentage columns. Use awk to tokenize safely
# and re-emit rows as tab-separated fields for the loop below.
while IFS="$FIELD_SEPARATOR" read -r -a columns; do
    if [[ "${columns[0]}" == "Workload1" || "${columns[0]}" == "Workload2" ]]; then
        continue  # Skip header or irrelevant rows
    fi

    skip_row=false
    weight_val=""
    power_val=""
    if (( weight_throughput_index >= 0 )); then
        weight_val="${columns[$weight_throughput_index]}"
        if ! is_missing_field "$weight_val"; then
            skip_row=true
        fi
    fi
    if (( power_index >= 0 )); then
        power_val="${columns[$power_index]}"
        if ! is_missing_field "$power_val"; then
            skip_row=true
        fi
    fi
    if [[ "$skip_row" == true ]]; then
        echo "Skipping row with existing metrics: weight_Throughput_sum=$weight_val, Power=$power_val"
        continue
    fi

    # Power cap is fixed from config
    powercap_val=$GPU_POWERCAP_VALUE
    if [[ -z "$powercap_val" ]] || ! [[ "$powercap_val" =~ ^[0-9]*\.?[0-9]+$ ]]; then
        echo "Row skipped: GPU_POWERCAP_VALUE is empty or invalid => [$powercap_val]"
        continue
    fi
    powercap_val_thres=$powercap_val
    echo "Fixed PowerCap used=$powercap_val_thres"

    # Read dynamic percentages by header indices
    percentages=()
    for (( i=0; i<$N_COMBINATIONS; i++ )); do
        percentage_col_index=${percent_col_indices[$i]}
        percentage="${columns[$percentage_col_index]}"
        if [[ -z "$percentage" || ! "$percentage" =~ ^[0-9]+$ ]]; then
            echo "Invalid percentage at position w$((i+1)): $percentage. Skipping row."
            continue 2
        fi
        percentages+=("$percentage")
    done

    # Display combination for logging
    percentage_display=""
    for (( i=0; i<$N_COMBINATIONS; i++ )); do
        percentage_display+="w$((i+1))=${percentages[$i]}%"
        if (( i < N_COMBINATIONS - 1 )); then
            percentage_display+=", "
        fi
    done
    echo "Parsed Thread Combination: $percentage_display"

    # Collect workloads in this row
    workload_names=()
    batchsizes=()
    modes=()
    prefixes=()
    tasks=()
    for (( i=0; i<$N_COMBINATIONS; i++ )); do
        workload_col_index=${workload_col_indices[$i]}
        workload=${columns[$workload_col_index]}
        if [[ -z "$workload" ]]; then
            echo "Workload $((i+1)) missing. Skipping row."
            continue 2
        fi
        parsed=$(parse_workload "$workload")
        workload_name=$(echo "$parsed" | cut -d, -f1)
        batchsize=$(echo "$parsed" | cut -d, -f2)
        mode=$(echo "$parsed" | cut -d, -f3)
        prefix=$(echo "$parsed" | cut -d, -f4)
        task=$(echo "$parsed" | cut -d, -f5)
        if [[ "$mode" == "inf" ]]; then
            mode="inference"
        fi
        workload_names+=("$workload_name")
        batchsizes+=("$batchsize")
        modes+=("$mode")
        prefixes+=("$prefix")
        tasks+=("$task")
    done

    echo "Starting Docker instances for colocation: ${workload_names[*]}"

    # ----------------------------------------------------------------------------
    # 5) LOOPS OVER RUNS AND MPS COMBINATIONS
    # ----------------------------------------------------------------------------
    for RUN in $(seq 1 $RUNS); do
        echo "RUN$RUN..."
        CURR_LOG_DIR="$LOG_DIR/RUN$RUN"
        for (( i=0; i<${#workload_names[@]}; i++ )); do
            CURR_LOG_DIR+="/${workload_names[$i]}_batch${batchsizes[$i]}-${modes[$i]}"
        done
        mkdir -p "$CURR_LOG_DIR"
        echo "Log directory: $CURR_LOG_DIR"

        # Build MPS combination string with underscores
        mps_combination="${percentages[*]}"
        MPS_LOG_DIR="${CURR_LOG_DIR}/$(echo ${mps_combination// /_})"
        mkdir -p "$MPS_LOG_DIR"
        echo quit | sudo nvidia-cuda-mps-control
        sleep 5
        sudo nvidia-cuda-mps-control -d

        # ------------------------------------------------------------------
        # LAUNCH DOCKER WORKLOADS
        # ------------------------------------------------------------------
        for SLEEP_TIME in "${SLEEP_TIMES[@]}"; do
            echo "Sleep time between jobs: $SLEEP_TIME seconds"
            # Remove monitor log if exists (helps with restarts)
            mps_combination_underscore=$(echo ${mps_combination// /_})
            monitor_log="$MPS_LOG_DIR/gpu_monitor_FREQ0_${mps_combination_underscore}_SLEEP${SLEEP_TIME}.log"
            if [ -f "$monitor_log" ]; then
                rm "$monitor_log"
                echo "Deleted monitor log: $monitor_log"
            fi

            # 1. Start GPU Power Monitor in the background
            python "$REPO_ROOT/scripts/profiling/GPUpowerMonitor.py" \
                --logfile "$monitor_log" \
                --gpu_index "$GPUMON_GPU_INDEX" \
                --power_cap "$powercap_val_thres" \
                --start_freq "$GPUMON_START_FREQ" \
                --runtime "$GPUMON_RUNTIME" \
                --monitor_interval_ms "$GPUMON_MONITOR_INTERVAL_MS" \
                --consecutive_exceed "$GPUMON_CONSECUTIVE_EXCEED" \
                --max_freq "$GPUMON_MAX_FREQ" \
                --min_freq "$GPUMON_MIN_FREQ" \
                --adjust_step "$GPUMON_ADJUST_STEP" \
                & monitor_pid=$!
            echo "Started GPU monitor (PID=$monitor_pid). Sleeping 2s to let it settle..."

            # Start each workload container
            pids=()  # fresh PIDs workload array
            for (( i=0; i<${#workload_names[@]}; i++ )); do
                MPS_PERCENTAGE=${percentages[$i]}
                echo "Starting workload ${workload_names[$i]} with task=${tasks[$i]} batch size ${batchsizes[$i]}, mode=(${modes[$i]}), MPS=$MPS_PERCENTAGE"
                docker run --rm --name "w$((i+1))" \
                    --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MPS_PERCENTAGE \
                    --env NVIDIA_VISIBLE_DEVICES=$GPUMON_GPU_INDEX  --gpus device=$GPUMON_GPU_INDEX \
                    -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                    -v ~/.cache/huggingface:/root/.cache/huggingface \
                    -v ~/.cache/torch:/root/.cache/torch \
                    -v "$REPO_ROOT:/root/mlprofiler" --ipc=host \
                    --cap-add=SYS_ADMIN -v /opt/nvidia/nsight-systems/2023.3.3:/nsys \
                    nba556677/ml_tasks:latest \
                    /bin/sh -c "cd /root/mlprofiler/workloads/${modes[$i]}; python ${tasks[$i]}-${modes[$i]}.py --model_name ${prefixes[$i]}${workload_names[$i]} \
                    --batch_size ${batchsizes[$i]} --log_dir ../../$MPS_LOG_DIR --profile_nstep 999999999" \
                > "$MPS_LOG_DIR/w$((i+1))_${workload_names[$i]}_batch${batchsizes[$i]}-${modes[$i]}_FREQ0_MPS${MPS_PERCENTAGE}_SLEEP${SLEEP_TIME}.log" 2>&1 &

                pids+=($!)
                if [ "$SLEEP_TIME" -gt 0 ]; then
                    sleep "$SLEEP_TIME"
                fi
            done

            # Additional GPU/perf monitors
            nvidia-smi pmon -o DT -i "$GPUMON_GPU_INDEX" \
                -f "$MPS_LOG_DIR/gpu_info_FREQ0_${mps_combination_underscore}_SLEEP${SLEEP_TIME}.csv" &
            nvidia_smi_pid=$!

            nvidia-smi -i "$GPUMON_GPU_INDEX" --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,memory.total,memory.free,memory.used,power.draw,power.limit,clocks.gr,clocks.sm,clocks.mem --format=csv --loop-ms=500 \
                > "$MPS_LOG_DIR/gpu_mem_FREQ0_${mps_combination_underscore}_SLEEP${SLEEP_TIME}.csv" &
            nvidia_smi_pid2=$!

            dcgmi dmon -e 1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012 -i "$GPUMON_GPU_INDEX" \
                > "$MPS_LOG_DIR/dcgm_info_FREQ0_${mps_combination_underscore}_SLEEP${SLEEP_TIME}.csv" &
            dcgmi_pid=$!

            # ------------------------------------------------------------------
            # WAIT FOR ANY DOCKER PROCESS TO END OR TIMEOUT
            # ------------------------------------------------------------------
            wait_any_with_timeout $TIMEOUT "${pids[@]}"
            echo "A container has finished or timed out. Cleaning up..."

            # KILL THE GPU MONITOR (since we only need it during this colocation)
            echo "Killing GPU Power Monitor (PID $monitor_pid)..."
            kill -9 "$monitor_pid" 2>/dev/null

            # CLEAN UP MONITORING PROCESSES
            kill -9 "$nvidia_smi_pid" "$nvidia_smi_pid2" "$dcgmi_pid" "$monitor_pid" 2>/dev/null

            # TERMINATE DOCKER CONTAINERS IF STILL RUNNING
            for (( i=0; i<${#workload_names[@]}; i++ )); do
                docker kill "w$((i+1))" 2>/dev/null
            done

            bash "$REPO_ROOT/runners/lib/terminate_client.sh"
            sudo kill -9 $(pgrep -f "docker run") 2>/dev/null
            wait "${pids[@]}" 2>/dev/null
            sleep 5
        done # SLEEP_TIMES
    done # RUNS
    echo "-------"
done < <(
    awk -v FIELD_SEP="$FIELD_SEPARATOR" '
         BEGIN { FPAT = "([^,]*)|(\"[^\"]*\")"; OFS = FIELD_SEP }
         NR>1 {
             for (i = 1; i <= NF; i++) { gsub(/^\"|\"$/, "", $i) }
             for (i = 1; i <= NF; i++) { printf "%s%s", $i, (i<NF ? FIELD_SEP : ORS) }
         }' "$CSV_FILE"
)
