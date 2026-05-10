#!/bin/bash

# Source the configuration file
source config_nonDL_gpumonitor.sh
echo "timeouts: $TIMEOUT"
LOG_DIR=$1
sudo nvidia-smi -pm 1

# ------------------------------------------------------------------------------
# 1) FUNCTIONS FOR MPS COMBINATIONS
# ------------------------------------------------------------------------------
generate_mps_combinations() {
    local n=$1
    local percentages=(10 20 30 40 50 60 70 80 90 100)
    local total=100

    if [ $n -eq 1 ]; then
        combinations=( "${percentages[@]}" )
    else
        combinations=( "$(printf "100 %.0s" $(seq 1 $n))" )
        generate_combinations_recursive "" $n $total
    fi
}

generate_combinations_recursive() {
    local prefix=$1
    local n=$2
    local total=$3
    local percentages=(10 20 30 40 50 60 70 80 90)

    if [ $n -eq 1 ]; then
        if (( total % 10 == 0 )) && (( total >= 10 )) && (( total <= 100 )); then
            combinations+=( "${prefix}${total}" )
        fi
    else
        for pct in "${percentages[@]}"; do
            if (( pct <= total )); then
                generate_combinations_recursive "${prefix}${pct} " $((n-1)) $((total - pct))
            fi
        done
    fi
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
if [ ${#custom_combinations[@]} -gt 0 ]; then
    echo "Using custom combinations specified in config.sh"
    combinations=("${custom_combinations[@]}")
else
    echo "Generating MPS combinations automatically"
    generate_mps_combinations $N_COMBINATIONS
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

    if [[ $workload =~ -(train|inf|samples)$ ]]; then
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
        #correct mode samples to cuda_samples
        if [[ "$mode" == "samples" ]]; then
            mode="cuda_samples"
        fi

        workload_names+=("$workload_name")
        batchsizes+=("$batchsize")
        modes+=("$mode")
        prefixes+=("$prefix")
        tasks+=("$task")
        echo "Workload $((i+1)): $workload_name, batchsize=$batchsize, mode=$mode, prefix=$prefix, task=$task"

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


            # ------------------------------------------------------------------
            # LAUNCH DOCKER WORKLOADS
            # ------------------------------------------------------------------
            for SLEEP_TIME in "${SLEEP_TIMES[@]}"; do
                echo "Sleep time between jobs: $SLEEP_TIME seconds"
                
                # 1. Start GPU Power Monitor in the background
                python GPUpowerMonitor.py \
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
                    #if modes is cuda_samples, then run the samples with cd /root/mlprofiler/workloads/cuda_samples/benchmarks/${MODEL}; ./${MODEL}
                    if [[ "${modes[$i]}" == "cuda_samples" ]]; then
                        stdbuf -oL \
                        docker run --rm --name "w$((i+1))" \
                        --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MPS_PERCENTAGE \
                        --env NVIDIA_VISIBLE_DEVICES=$GPUMON_GPU_INDEX --gpus device=$GPUMON_GPU_INDEX \
                        -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                        -v ~/.cache/huggingface:/root/.cache/huggingface \
                        -v ~/.cache/torch:/root/.cache/torch \
                        -v ~/mlProfiler:/root/mlprofiler --ipc=host \
                        --cap-add=SYS_ADMIN -v /opt/nvidia/nsight-systems/2023.3.3:/nsys \
                        nba556677/ml_tasks:cuda_samples \
                        /bin/sh -c "cd /root/mlprofiler/workloads/cuda_samples/benchmarks/${workload_names[$i]}; ./${workload_names[$i]}" \
                        | stdbuf -oL tee "$MPS_LOG_DIR/w$((i+1))_${workload_names[$i]}_batch${batchsizes[$i]}-${modes[$i]}_FREQ0_MPS${MPS_PERCENTAGE}_SLEEP${SLEEP_TIME}.log" 2>&1 &
                        pids+=($!)  # Store PID

                    else    
                        docker run --rm --name "w$((i+1))" \
                            --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MPS_PERCENTAGE \
                            --env NVIDIA_VISIBLE_DEVICES=$GPUMON_GPU_INDEX --gpus device=$GPUMON_GPU_INDEX \
                            -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                            -v ~/.cache/huggingface:/root/.cache/huggingface \
                            -v ~/.cache/torch:/root/.cache/torch \
                            -v ~/mlProfiler:/root/mlprofiler --ipc=host \
                            --cap-add=SYS_ADMIN -v /opt/nvidia/nsight-systems/2023.3.3:/nsys \
                            nba556677/ml_tasks:cuda_samples \
                            /bin/sh -c "cd /root/mlprofiler/workloads/${modes[$i]}; python ${tasks[$i]}-${modes[$i]}.py --model_name ${prefixes[$i]}${workload_names[$i]} \
                            --batch_size ${batchsizes[$i]} --log_dir ../../tests/mps/multiinstance/$MPS_LOG_DIR --profile_nstep 999999999999" \
                        > "$MPS_LOG_DIR/w$((i+1))_${workload_names[$i]}_batch${batchsizes[$i]}-${modes[$i]}_FREQ0_MPS${MPS_PERCENTAGE}_SLEEP${SLEEP_TIME}.log" 2>&1 &
                        pids+=($!)  # Store PID
                    fi
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

                bash ../terminate_client.sh
                sudo kill -9 $(pgrep -f "docker run") 2>/dev/null
                wait "${pids[@]}" 2>/dev/null
                sleep 5
            done # SLEEP_TIMES
        done  # MPS combinations
    done # RUNS
    echo "-------"
done < <(tail -n +2 "$CSV_FILE")
