#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/paths.sh"
cd "$REPO_ROOT"

# Source the configuration file
source "$REPO_ROOT/configs/profiling/shared/config_gpumonitor.sh"
echo "timeouts: $TIMEOUT"
LOG_DIR=$1
sudo nvidia-smi -pm 1

# ------------------------------------------------------------------------------
# 1) FUNCTIONS FOR MPS COMBINATIONS
# ------------------------------------------------------------------------------
SAMPLE_STRIDE="${SAMPLE_STRIDE:-1}"
if ! [[ "$SAMPLE_STRIDE" =~ ^[0-9]+$ ]]; then
    SAMPLE_STRIDE=1
fi
# Function to generate all valid MPS percentage combinations, including the all-100s case
generate_mps_combinations() {
    local n=$1
    local total=100

    combinations=()

    # Always include the special all-100s combination for n workloads
    local all100=""
    for ((i=1; i<=n; i++)); do
        all100+="${all100:+ }100"
    done
    combinations+=("$all100")

    _mps_generate_combo "" "$n" "$total"
}

_mps_build_values() {
    local vals=()
    local step="${STEP:-10}"
    local even_only="${EVEN_ONLY:-0}"

    if ! [[ "$step" =~ ^[0-9]+$ ]] || (( step <= 0 )); then
        step=10
    fi
    if ! [[ "$even_only" =~ ^[0-9]+$ ]]; then
        even_only=0
    fi

    if [[ -n "${VALUES:-}" ]]; then
        read -r -a vals <<< "${VALUES}"
    else
        if (( even_only == 1 )); then
            vals=(20 40 60 80)
        else
            local v=10
            while (( v <= 90 )); do
                vals+=("$v")
                (( v += step ))
            done
        fi
    fi

    echo "${vals[*]}"
}

_mps_generate_combo() {
    local prefix="$1"
    local remaining="$2"
    local total="$3"
    local percentages

    read -r -a percentages <<< "$(_mps_build_values)"

    if (( remaining == 1 )); then
        if (( total >= 10 && total <= 100 )); then
            combinations+=("${prefix:+$prefix }$total")
        fi
        return
    fi

    for pct in "${percentages[@]}"; do
        [[ "$pct" =~ ^-?[0-9]+$ ]] || continue
        (( pct > 0 && pct <= total )) || continue
        _mps_generate_combo "${prefix:+$prefix }$pct" $((remaining - 1)) $((total - pct))
    done
}

apply_sample_stride() {
    local stride="$1"
    (( stride > 1 )) || return

    local filtered=()
    for idx in "${!combinations[@]}"; do
        (( idx % stride == 0 )) && filtered+=("${combinations[$idx]}")
    done
    combinations=("${filtered[@]}")
}

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
# 2) MPS COMBINATIONS
# ------------------------------------------------------------------------------
# Check if custom combinations are provided
auto_generated=0
if [ ${#custom_combinations[@]} -gt 0 ]; then
    echo "Using custom combinations specified in config.sh"
    combinations=("${custom_combinations[@]}")
    
else
    echo "Generating MPS combinations automatically"
    generate_mps_combinations $N_COMBINATIONS
    auto_generated=1
fi
if (( auto_generated == 1 )); then
    apply_sample_stride "$SAMPLE_STRIDE"
fi
echo "Combinations: ${combinations[@]}"

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
while IFS=, read -ra columns; do
    if [[ "${columns[0]}" == "workload1" ]]; then
        continue
    fi
    # Grab the last column => powercap100_100
    powercap_val=$GPU_POWERCAP_VALUE

    # 1. Check if it's empty or non-numeric
    if [[ -z "$powercap_val" ]] || ! [[ "$powercap_val" =~ ^[0-9]*\.?[0-9]+$ ]]; then
        echo "Row skipped: powercap100_100 is empty or invalid => [$powercap_val]"
        continue  # Skip to the next line
    fi
    # 2. Multiply by threshold
    powercap_val_thres=$powercap_val
    # 2. Multiply by threshold
    echo "powercap read=$powercap_val, under thres = $powercap_val_thres"
    # Collect workloads in this row
    workload_names=()
    batchsizes=()
    modes=()
    prefixes=()
    tasks=()
    pids=()

    # We assume each row can have up to N_COMBINATIONS workloads
    for (( i=0; i<$N_COMBINATIONS; i++ )); do
        workload_col_index=$((i*2))
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

        for mps_combination in "${combinations[@]}"; do
            echo "MPS Combination: $mps_combination"
            IFS=' ' read -ra mps_percentages <<< "$mps_combination"

            if [ ${#mps_percentages[@]} -ne ${#workload_names[@]} ]; then
                echo "Mismatch in # of workloads vs. MPS percentages."
                continue
            fi

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

                # 1. Start GPU Power Monitor in the background
                python "$REPO_ROOT/scripts/profiling/GPUpowerMonitor.py" \
                    --logfile "$MPS_LOG_DIR/gpu_monitor_FREQ0_${mps_combination// /_}_SLEEP${SLEEP_TIME}.log" \
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
                # 2. Launch your Docker workloads (backgrounded)

                pids=()  # fresh PIDs workload array

                # Start each workload container
                for (( i=0; i<${#workload_names[@]}; i++ )); do
                    MPS_PERCENTAGE=${mps_percentages[$i]}
                    docker run --rm --name "w$((i+1))" \
                        --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MPS_PERCENTAGE \
                        --env NVIDIA_VISIBLE_DEVICES=$GPUMON_GPU_INDEX --gpus device=$GPUMON_GPU_INDEX \
                        -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                        -v ~/.cache/huggingface:/root/.cache/huggingface \
                        -v ~/.cache/torch:/root/.cache/torch \
                        -v "$REPO_ROOT:/root/mlprofiler" --ipc=host \
                        --cap-add=SYS_ADMIN -v /opt/nvidia/nsight-systems/2023.3.3:/nsys \
                        nba556677/ml_tasks:latest \
                        /bin/sh -c "cd /root/mlprofiler/workloads/${modes[$i]}; python ${tasks[$i]}-${modes[$i]}.py --model_name ${prefixes[$i]}${workload_names[$i]} \
                        --batch_size ${batchsizes[$i]} --log_dir ../../$MPS_LOG_DIR --profile_nstep 99999999" \
                    > "$MPS_LOG_DIR/w$((i+1))_${workload_names[$i]}_batch${batchsizes[$i]}-${modes[$i]}_FREQ0_MPS${MPS_PERCENTAGE}_SLEEP${SLEEP_TIME}.log" 2>&1 &

                    pids+=($!)
                    if [ "$SLEEP_TIME" -gt 0 ]; then
                        sleep "$SLEEP_TIME"
                    fi
                done

                # Additional GPU/perf monitors
                nvidia-smi pmon -o DT -i "$GPUMON_GPU_INDEX" \
                    -f "$MPS_LOG_DIR/gpu_info_FREQ0_${mps_combination// /_}_SLEEP${SLEEP_TIME}.csv" &
                nvidia_smi_pid=$!

                nvidia-smi -i "$GPUMON_GPU_INDEX" --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,memory.total,memory.free,memory.used,power.draw,power.limit,clocks.gr,clocks.sm,clocks.mem --format=csv --loop-ms=500 \
                    > "$MPS_LOG_DIR/gpu_mem_FREQ0_${mps_combination// /_}_SLEEP${SLEEP_TIME}.csv" &
                nvidia_smi_pid2=$!

                dcgmi dmon -e 1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012 -i "$GPUMON_GPU_INDEX" \
                    > "$MPS_LOG_DIR/dcgm_info_FREQ0_${mps_combination// /_}_SLEEP${SLEEP_TIME}.csv" &
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
        done  # MPS combinations
    done # RUNS
    echo "-------"
done < <(tail -n +2 "$CSV_FILE")
