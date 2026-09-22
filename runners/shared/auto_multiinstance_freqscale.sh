#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/paths.sh"
cd "$REPO_ROOT"

# Source the configuration file
source "$REPO_ROOT/configs/profiling/shared/config.sh"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-1}"
if ! [[ "$SAMPLE_STRIDE" =~ ^[0-9]+$ ]]; then
    SAMPLE_STRIDE=1
fi
echo "timeouts: $TIMEOUT"
LOG_DIR=$1
sudo nvidia-smi -pm 1
#VALID_FREQS=(952 1147 1335 1530)  # Add valid frequency scaling factors

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

# Function to wait until any of the processes finish or timeout occurs
wait_any_with_timeout() {
    local timeout=$1
    shift
    local pids=("$@")
    local start_time=$(date +%s)
    local end_time=$((start_time + timeout))

    while (( $(date +%s) < end_time )); do
        for pid in "${pids[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                # Process has exited
                echo "Process $pid has finished."
                return 0
            fi
        done
        sleep 1
    done
    # Timeout reached
    echo "Timeout of $timeout seconds reached."
    return 1
}

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
#exit 0

# Function to parse a workload string and extract details
parse_workload() {
    local workload=$1
    local workload_name=""
    local batchsize=""
    local mode=""
    local prefix=""
    local task=""

    # Extract mode (either '-train' or '-inf')
    if [[ $workload =~ -(train|inf)$ ]]; then
        mode=${BASH_REMATCH[1]}
    fi

    # Extract batch size
    if [[ $workload =~ _batch([0-9]+) ]]; then
        batchsize=${BASH_REMATCH[1]}
    fi

    # Extract workload name by removing the batch size and mode parts
    workload_name=${workload/_batch${batchsize}-${mode}/}

    # Get prefix and task
    prefix=${workloadprefix[$workload_name]}
    task=${task_model[$workload_name]}

    echo "$workload_name,$batchsize,$mode,$prefix,$task"
}

# Read the CSV file and process each row
while IFS=, read -ra columns; do
    # Skip the header
    if [[ "${columns[0]}" == "workload1" ]]; then
        continue
    fi

    echo "Processing row:"

    # Arrays to store workload details
    workload_names=()
    batchsizes=()
    modes=()
    prefixes=()
    tasks=()
    pids=()

    # Loop over the number of combinations
    for (( i=0; i<$N_COMBINATIONS; i++ )); do
        workload_col_index=$((i*2))  # Adjust index if workloads are in consecutive columns
        workload=${columns[$workload_col_index]}

        # Check if the workload exists
        if [[ -z "$workload" ]]; then
            echo "Workload $((i+1)) is missing. Skipping this row."
            continue 2  # Skip to the next row
        fi

        echo "workload$((i+1)): $workload"

        parsed=$(parse_workload "$workload")
        workload_name=$(echo "$parsed" | cut -d, -f1)
        batchsize=$(echo "$parsed" | cut -d, -f2)
        mode=$(echo "$parsed" | cut -d, -f3)
        prefix=$(echo "$parsed" | cut -d, -f4)
        task=$(echo "$parsed" | cut -d, -f5)

        # Correct mode if necessary
        if [[ "$mode" == "inf" ]]; then
            mode="inference"
        fi

        # Append details to arrays
        workload_names+=("$workload_name")
        batchsizes+=("$batchsize")
        modes+=("$mode")
        prefixes+=("$prefix")
        tasks+=("$task")
    done

    echo "Starting Docker instances for colocation: ${workload_names[*]}"

    # Run the workloads with varying MPS percentages
    for RUN in $(seq 1 $RUNS); do
        echo "RUN$RUN..."
        # Create a unique log directory for this run
        CURR_LOG_DIR="$LOG_DIR/RUN$RUN"
        for (( i=0; i<${#workload_names[@]}; i++ )); do
            CURR_LOG_DIR+="/${workload_names[$i]}_batch${batchsizes[$i]}-${modes[$i]}"
        done
        mkdir -p "$CURR_LOG_DIR"
        echo "Log directory: $CURR_LOG_DIR"

        for mps_combination in "${combinations[@]}"; do
            echo "MPS Combination: $mps_combination"

            IFS=' ' read -ra mps_percentages <<< "$mps_combination"

            # Check if the number of percentages matches the number of workloads
            if [ ${#mps_percentages[@]} -ne ${#workload_names[@]} ]; then
                echo "Mismatch in number of workloads and MPS percentages."
                continue
            fi

            # Create subdirectory for each thread combination
            MPS_LOG_DIR="${CURR_LOG_DIR}/$(echo ${mps_combination// /_})"
            mkdir -p "$MPS_LOG_DIR"
            echo "Created directory for MPS combination: $MPS_LOG_DIR"

            # Start Docker instances for each workload
            pids=()  # Reset PIDs array for each percentage combination

            for FREQ_SCALING in "${VALID_FREQS[@]}"; do
                echo "Running with frequency scaling: $FREQ_SCALING MHz"

                # Set frequency scaling
                sudo nvidia-smi -lgc $FREQ_SCALING,$FREQ_SCALING
                #echo quit | sudo nvidia-cuda-mps-control
                #sleep 3
                #sudo nvidia-cuda-mps-control -d
                sleep 2
                # Loop over each sleep time for the current job combination
                for SLEEP_TIME in "${SLEEP_TIMES[@]}"; do
                    echo "Setting sleep time to $SLEEP_TIME seconds between jobs."
                    for (( i=0; i<${#workload_names[@]}; i++ )); do
                        MPS_PERCENTAGE=${mps_percentages[$i]}
                        #add if for gpt2-xl in worklaod names
                        if [[ "${workload_names[$i]}" == "gpt2-xl" ]]; then
                            docker run --rm --name "w$((i+1))" --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MPS_PERCENTAGE \
                            --env NVIDIA_VISIBLE_DEVICES=1 --gpus device=1 \
                            -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                            -v ~/.cache/huggingface:/root/.cache/huggingface \
                            -v ~/.cache/torch:/root/.cache/torch \
                            -v "$REPO_ROOT:/root/mlprofiler" --ipc=host \
                            --cap-add=SYS_ADMIN -v /opt/nvidia/nsight-systems/2023.3.3:/nsys \
                            nba556677/ml_tasks:latest \
                            /bin/sh -c "cd /root/mlprofiler/workloads/${modes[$i]}; python ${tasks[$i]}-${modes[$i]}.py --model_name ${prefixes[$i]}${workload_names[$i]} \
                            --batch_size 1 --log_dir ../../$MPS_LOG_DIR --output_length ${batchsizes[$i]} --num_requests 1000000000 --profile_nstep 4000"  \
                            > "$MPS_LOG_DIR/w$((i+1))_${workload_names[$i]}_batch${batchsizes[$i]}-${modes[$i]}_FREQ${FREQ_SCALING}_MPS${MPS_PERCENTAGE}_SLEEP${SLEEP_TIME}.log" 2>&1 & pids+=($!)


                        else
                            docker run --rm --name "w$((i+1))" \
                            --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MPS_PERCENTAGE \
                            --env NVIDIA_VISIBLE_DEVICES=1 --gpus device=1 \
                            -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                            -v ~/.cache/huggingface:/root/.cache/huggingface \
                            -v ~/.cache/torch:/root/.cache/torch \
                            -v "$REPO_ROOT:/root/mlprofiler" --ipc=host \
                            --cap-add=SYS_ADMIN -v /opt/nvidia/nsight-systems/2023.3.3:/nsys \
                            nba556677/ml_tasks:latest \
                            /bin/sh -c "cd /root/mlprofiler/workloads/${modes[$i]}; python ${tasks[$i]}-${modes[$i]}.py --model_name ${prefixes[$i]}${workload_names[$i]} \
                            --batch_size ${batchsizes[$i]} --log_dir ../../$MPS_LOG_DIR --profile_nstep 4000" \
                            > "$MPS_LOG_DIR/w$((i+1))_${workload_names[$i]}_batch${batchsizes[$i]}-${modes[$i]}_FREQ${FREQ_SCALING}_MPS${MPS_PERCENTAGE}_SLEEP${SLEEP_TIME}.log" 2>&1 &

                            pids+=($!)  # Store PID
                        fi

                        # Sleep between each job within the current configuration
                        if [ "$SLEEP_TIME" -gt 0 ]; then
                            echo "Sleeping for $SLEEP_TIME seconds between jobs."
                            sleep "$SLEEP_TIME"
                        fi
                    done

                    # Start monitoring GPU usage
                    # Start monitoring GPU usage
                    nvidia-smi pmon -o DT -i 1 -f "$MPS_LOG_DIR/gpu_info_FREQ${FREQ_SCALING}_${mps_combination// /_}_SLEEP${SLEEP_TIME}.csv" &
                    nvidia-smi -i 1 --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,memory.total,memory.free,memory.used,power.draw,power.limit --format=csv --loop-ms=500 \
                    > "$MPS_LOG_DIR/gpu_mem_FREQ${FREQ_SCALING}_${mps_combination// /_}_SLEEP${SLEEP_TIME}.csv" &
                    dcgmi dmon -e 1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012  -i 1  > "$MPS_LOG_DIR/dcgm_info_FREQ${FREQ_SCALING}_${mps_combination// /_}_SLEEP${SLEEP_TIME}.csv" &

                    # Wait for any Docker instance to finish
                    #wait -n "${pids[@]}"
                    #echo "Processes have finished. Cleaning up..."
                    #sleep 1

                    # Wait until any process finishes or timeout occurs
                    wait_any_with_timeout $TIMEOUT "${pids[@]}"
                    echo "A process has finished or timeout occurred. Cleaning up..."
                    sleep 1


                    # Terminate Docker instances
                    for (( i=0; i<${#workload_names[@]}; i++ )); do
                        docker kill "w$((i+1))" 2>/dev/null
                    done
                    bash "$REPO_ROOT/runners/lib/terminate_client.sh"
                    sudo kill -9 $(pgrep -f "docker run") 2>/dev/null
                    sudo kill -9 $(pgrep -f "nvidia-smi pmon") 2>/dev/null
                    sudo kill -9 $(pgrep -f "nvidia-smi") 2>/dev/null
                    sudo kill -9 $(pgrep -f "dcgmi dmon") 2>/dev/null
                    wait "${pids[@]}" 2>/dev/null
                    sleep 5
                done # End of sleep times loop
            done # End of frequency scaling loop
        done  # End of MPS combinations loop
    done # End of RUNS loop
    echo "-------"
done < <(tail -n +2 "$CSV_FILE")
