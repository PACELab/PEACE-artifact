#!/bin/bash

# Source the configuration file
source config_nonDL_missingthreads.sh
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
# ------------------------------------------------------------------------------
# 4) MAIN LOOP OVER CSV ROWS
# ------------------------------------------------------------------------------
while IFS=, read -ra columns; do
    if [[ "${columns[0]}" == "Workload1" || "${columns[0]}" == "Workload2" ]]; then
        continue  # Skip header or irrelevant rows
    fi
    #echo columns: ${columns[5]}
    # Extract necessary columns from missing.csv
    powercap_val="${columns[${#columns[@]}-1]}"  # PowerCap is last column

    # Read w1_percentage and w2_percentage directly from the CSV
    w1_percentage="${columns[4]}"  # Assuming w1_percentage is in the 4th column
    w2_percentage="${columns[5]}"  # Assuming w2_percentage is in the 5th column

    # Check if percentages are valid
    if [[ -z "$w1_percentage" || -z "$w2_percentage" || ! "$w1_percentage" =~ ^[0-9]+$ || ! "$w2_percentage" =~ ^[0-9]+$ ]]; then
        echo "Invalid percentages: w1_percentage=$w1_percentage, w2_percentage=$w2_percentage. Skipping row."
        continue  # Skip if percentages are invalid
    fi

    echo "Parsed Thread Combination: w1=${w1_percentage}%, w2=${w2_percentage}%"

    # Collect workloads in this row
    workload_names=()
    batchsizes=()
    modes=()
    prefixes=()
    tasks=()

    # We assume each row can have up to N_COMBINATIONS workloads
    for (( i=0; i<$N_COMBINATIONS; i++ )); do
        workload_col_index=$((i))
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

        # Use w1_percentage and w2_percentage directly from CSV
        MPS_LOG_DIR="${CURR_LOG_DIR}/${w1_percentage}_${w2_percentage}"
        mkdir -p "$MPS_LOG_DIR"

        # ------------------------------------------------------------------
        # LAUNCH DOCKER WORKLOADS
        # ------------------------------------------------------------------
        for SLEEP_TIME in "${SLEEP_TIMES[@]}"; do
            echo "Sleep time between jobs: $SLEEP_TIME seconds"
            
            # Start each workload container
            pids=()  # fresh PIDs workload array

            for (( i=0; i<${#workload_names[@]}; i++ )); do
                if [ $i -eq 0 ]; then
                    MPS_PERCENTAGE=$w1_percentage
                else
                    MPS_PERCENTAGE=$w2_percentage
                fi
                echo "Starting workload ${workload_names[$i]} with task=${tasks[$i]} batch size ${batchsizes[$i]}, mode=(${modes[$i]}), MPS=$MPS_PERCENTAGE"
                #if modes is cuda_samples, then run the samples with cd /root/mlprofiler/workloads/cuda_samples/benchmarks/${MODEL}; ./${MODEL}
                if [[ "${modes[$i]}" == "cuda_samples" ]]; then
                    stdbuf -oL \
                    docker run --rm --name "w$((i+1))" \
                    --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MPS_PERCENTAGE \
                    --env NVIDIA_VISIBLE_DEVICES=1 --gpus device=1 \
                    -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                    -v ~/.cache/huggingface:/root/.cache/huggingface \
                    -v ~/.cache/torch:/root/.cache/torch \
                    -v ~/mlProfiler:/root/mlprofiler --ipc=host \
                    --cap-add=SYS_ADMIN -v /opt/nvidia/nsight-systems/2023.3.3:/nsys \
                    nba556677/ml_tasks:cuda_samples \
                    /bin/sh -c "cd /root/mlprofiler/workloads/cuda_samples/benchmarks/${workload_names[$i]}; ./${workload_names[$i]}" \
                    | stdbuf -oL tee "$MPS_LOG_DIR/w$((i+1))_${workload_names[$i]}_batch${batchsizes[$i]}-${modes[$i]}_FREQ${VALID_FREQUENCY}_MPS${MPS_PERCENTAGE}_SLEEP${SLEEP_TIME}.log" 2>&1 &
                    pids+=($!)  # Store PID

                else    
                    docker run --rm --name "w$((i+1))" \
                        --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MPS_PERCENTAGE \
                        --env NVIDIA_VISIBLE_DEVICES=1 --gpus device=1 \
                        -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                        -v ~/.cache/huggingface:/root/.cache/huggingface \
                        -v ~/.cache/torch:/root/.cache/torch \
                        -v ~/mlProfiler:/root/mlprofiler --ipc=host \
                        --cap-add=SYS_ADMIN -v /opt/nvidia/nsight-systems/2023.3.3:/nsys \
                        nba556677/ml_tasks:cuda_samples \
                        /bin/sh -c "cd /root/mlprofiler/workloads/${modes[$i]}; python ${tasks[$i]}-${modes[$i]}.py --model_name ${prefixes[$i]}${workload_names[$i]} \
                        --batch_size ${batchsizes[$i]} --log_dir ../../tests/mps/multiinstance/$MPS_LOG_DIR --profile_nstep 999999999999" \
                    > "$MPS_LOG_DIR/w$((i+1))_${workload_names[$i]}_batch${batchsizes[$i]}-${modes[$i]}_FREQ${VALID_FREQUENCY}_MPS${MPS_PERCENTAGE}_SLEEP${SLEEP_TIME}.log" 2>&1 &
                    pids+=($!)  # Store PID
                fi
                if [ "$SLEEP_TIME" -gt 0 ]; then
                    sleep "$SLEEP_TIME"
                fi
            done

            # Additional GPU/perf monitors
            nvidia-smi pmon -o DT -i "$GPUMON_GPU_INDEX" \
                -f "$MPS_LOG_DIR/gpu_info_FREQ${VALID_FREQUENCY}_${w1_percentage}_${w2_percentage}_SLEEP${SLEEP_TIME}.csv" &
            nvidia_smi_pid=$!

            nvidia-smi -i "$GPUMON_GPU_INDEX" --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,memory.total,memory.free,memory.used,power.draw,power.limit,clocks.gr,clocks.sm,clocks.mem --format=csv --loop-ms=500 \
                > "$MPS_LOG_DIR/gpu_mem_FREQ${VALID_FREQUENCY}_${w1_percentage}_${w2_percentage}_SLEEP${SLEEP_TIME}.csv" &
            nvidia_smi_pid2=$!

            dcgmi dmon -e 1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012 -i "$GPUMON_GPU_INDEX" \
                > "$MPS_LOG_DIR/dcgm_info_FREQ${VALID_FREQUENCY}_${w1_percentage}_${w2_percentage}_SLEEP${SLEEP_TIME}.csv" &
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
done < <(tail -n +2 "$CSV_FILE")
