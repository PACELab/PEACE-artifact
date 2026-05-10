#!/bin/bash

# Source the configuration file
source config_missingthreads.sh
echo "timeouts: $TIMEOUT"
LOG_DIR=$1
sudo nvidia-smi -pm 1
#set to one freq only
sudo nvidia-smi -lgc $VALID_FREQUENCY,$VALID_FREQUENCY

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
# 4) MAIN LOOP OVER CSV ROWS
# ------------------------------------------------------------------------------
# Build header-based column indices so we can read w{i}_percentage directly and
# avoid any dependence on the Thread_combination column.

# Parse header fields (tab-separated) using CSV-aware awk tokenizer
IFS=$'\t' read -r -a header_columns < <(
    awk 'BEGIN { FPAT = "([^,]*)|(\"[^\"]*\")"; OFS = "\t" }
         NR==1 {
             for (i = 1; i <= NF; i++) { gsub(/^\"|\"$/, "", $i) }
             for (i = 1; i <= NF; i++) { printf "%s%s", $i, (i<NF ? OFS : ORS) }
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
powercap_col_index=$(find_col_index "PowerCap")
if (( powercap_col_index < 0 )); then
    echo "No PowerCap header in $CSV_FILE (optional, continuing)"
fi

# CSV contains a quoted field (Thread_combination) with commas; we ignore it
# and read percentages from w{i}_percentage columns. Use awk to tokenize safely
# and re-emit rows as tab-separated fields for the loop below.
while IFS=$'\t' read -r -a columns; do
    if [[ "${columns[0]}" == "Workload1" || "${columns[0]}" == "Workload2" ]]; then
        continue  # Skip header or irrelevant rows
    fi
    # Read PowerCap if column exists
    powercap_val=""
    if (( powercap_col_index >= 0 )); then
        powercap_val="${columns[$powercap_col_index]}"
    fi

    # Dynamically read percentages based on N_COMBINATIONS
    # Percentages start after N_COMBINATIONS workload columns and Thread_combination column
    # Column layout: Workload1, Workload2, ..., WorkloadN, Thread_combination, w1_percentage, w2_percentage, ..., wN_percentage, ...
    percentages=()
    for (( i=0; i<$N_COMBINATIONS; i++ )); do
        percentage_col_index=${percent_col_indices[$i]}
        percentage="${columns[$percentage_col_index]}"

        # Check if percentage is valid
        if [[ -z "$percentage" || ! "$percentage" =~ ^[0-9]+$ ]]; then
            echo "Invalid percentage at position w$((i+1)): $percentage. Skipping row."
            continue 2
        fi

        percentages+=("$percentage")
    done

    # Build percentage display string
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

    # We assume each row can have up to N_COMBINATIONS workloads
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

        # Build MPS combination string with underscores (matching auto_multiinstance_freqscale.sh format)
        mps_combination="${percentages[*]}"
        MPS_LOG_DIR="${CURR_LOG_DIR}/$(echo ${mps_combination// /_})"
        mkdir -p "$MPS_LOG_DIR"
        # enforce mps control
        #echo quit | sudo nvidia-cuda-mps-control
        #sleep 5
        #sudo nvidia-cuda-mps-control -d
        # ------------------------------------------------------------------
        # LAUNCH DOCKER WORKLOADS
        # ------------------------------------------------------------------
        for SLEEP_TIME in "${SLEEP_TIMES[@]}"; do
            echo "Sleep time between jobs: $SLEEP_TIME seconds"
            
            # Start each workload container
            pids=()  # fresh PIDs workload array

            for (( i=0; i<${#workload_names[@]}; i++ )); do
                MPS_PERCENTAGE=${percentages[$i]}
                echo "Starting workload ${workload_names[$i]} with task=${tasks[$i]} batch size ${batchsizes[$i]}, mode=(${modes[$i]}), MPS=$MPS_PERCENTAGE"
                docker run --rm --name "w$((i+1))" \
                    --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MPS_PERCENTAGE \
                    --env NVIDIA_VISIBLE_DEVICES=1 --gpus device=1 \
                    -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                    -v ~/.cache/huggingface:/root/.cache/huggingface \
                    -v ~/.cache/torch:/root/.cache/torch \
                    -v ~/mlProfiler:/root/mlprofiler --ipc=host \
                    --cap-add=SYS_ADMIN -v /opt/nvidia/nsight-systems/2023.3.3:/nsys \
                    nba556677/ml_tasks:latest \
                    /bin/sh -c "cd /root/mlprofiler/workloads/${modes[$i]}; python ${tasks[$i]}-${modes[$i]}.py --model_name ${prefixes[$i]}${workload_names[$i]} \
                    --batch_size ${batchsizes[$i]} --log_dir ../../tests/mps/multiinstance/$MPS_LOG_DIR --profile_nstep 4000" \
                > "$MPS_LOG_DIR/w$((i+1))_${workload_names[$i]}_batch${batchsizes[$i]}-${modes[$i]}_FREQ${VALID_FREQUENCY}_MPS${MPS_PERCENTAGE}_SLEEP${SLEEP_TIME}.log" 2>&1 &

                pids+=($!)
                if [ "$SLEEP_TIME" -gt 0 ]; then
                    sleep "$SLEEP_TIME"
                fi
            done

            # Additional GPU/perf monitors
            mps_combination_underscore=$(echo ${mps_combination// /_})
            nvidia-smi pmon -o DT -i "$GPUMON_GPU_INDEX" \
                -f "$MPS_LOG_DIR/gpu_info_FREQ${VALID_FREQUENCY}_${mps_combination_underscore}_SLEEP${SLEEP_TIME}.csv" &
            nvidia_smi_pid=$!

            nvidia-smi -i "$GPUMON_GPU_INDEX" --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,memory.total,memory.free,memory.used,power.draw,power.limit,clocks.gr,clocks.sm,clocks.mem --format=csv --loop-ms=500 \
                > "$MPS_LOG_DIR/gpu_mem_FREQ${VALID_FREQUENCY}_${mps_combination_underscore}_SLEEP${SLEEP_TIME}.csv" &
            nvidia_smi_pid2=$!

            dcgmi dmon -e 1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012 -i "$GPUMON_GPU_INDEX" \
                > "$MPS_LOG_DIR/dcgm_info_FREQ${VALID_FREQUENCY}_${mps_combination_underscore}_SLEEP${SLEEP_TIME}.csv" &
            dcgmi_pid=$!

            # ------------------------------------------------------------------
            # WAIT FOR ANY DOCKER PROCESS TO END OR TIMEOUT
            # ------------------------------------------------------------------
            wait_any_with_timeout $TIMEOUT "${pids[@]}"
            echo "A container has finished or timed out. Cleaning up..."

            # CLEAN UP MONITORING PROCESSES
            kill -9 "$nvidia_smi_pid" "$nvidia_smi_pid2" "$dcgmi_pid" 2>/dev/null

            # TERMINATE DOCKER CONTAINERS IF STILL RUNNING
            for (( i=0; i<${#workload_names[@]}; i++ )); do
                docker kill "w$((i+1))" 2>/dev/null
            done

            bash ../terminate_client.sh
            sudo kill -9 $(pgrep -f "docker run") 2>/dev/null
            wait "${pids[@]}" 2>/dev/null
            sleep 5
        done # SLEEP_TIMES
    done # RUNS
    echo "-------"
done < <(
    awk 'BEGIN { FPAT = "([^,]*)|(\"[^\"]*\")"; OFS = "\t" }
         NR>1 {
             for (i = 1; i <= NF; i++) { gsub(/^\"|\"$/, "", $i) }
             for (i = 1; i <= NF; i++) { printf "%s%s", $i, (i<NF ? OFS : ORS) }
         }' "$CSV_FILE"
)
