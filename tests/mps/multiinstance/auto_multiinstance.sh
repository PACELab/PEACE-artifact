#!/bin/bash

# Source the configuration file
source config.sh

LOG_DIR=$1
RUNS=1

# Function to generate all valid MPS percentage combinations, including 100 for all combinations
generate_mps_combinations() {
    local n=$1
    local percentages=(10 20 30 40 50 60 70 80 90 100)
    local total=100

     # Initialize combinations based on n
    if [ $n -eq 1 ]; then
        combinations=( "${percentages[@]}" )
    else
        combinations=( "$(printf "100 %.0s" $(seq 1 $n))" )
        generate_combinations_recursive "" $n $total
    fi
}

# Recursive helper function to generate combinations
generate_combinations_recursive() {
    local prefix=$1
    local n=$2
    local total=$3

    if [ $n -eq 1 ]; then
        if (( total % 10 == 0 )) && (( total >= 10 )) && (( total <= 100 )); then
            combinations+=( "${prefix}${total}" )
        fi
    else
        for pct in 10 30 50 70 90; do
            if (( pct <= total )); then
                generate_combinations_recursive "${prefix}${pct} " $((n-1)) $((total - pct))
            fi
        done
    fi
}

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

    # Create log directory
    CURR_LOG_DIR="$LOG_DIR"
    for (( i=0; i<${#workload_names[@]}; i++ )); do
        CURR_LOG_DIR+="/${workload_names[$i]}_batch${batchsizes[$i]}-${modes[$i]}"
    done
    mkdir -p "$CURR_LOG_DIR"

    # Generate MPS percentage combinations
    generate_mps_combinations $N_COMBINATIONS

    # Run the workloads with varying MPS percentages
    for RUN in $(seq 1 $RUNS); do
        echo "RUN$RUN..."

        for mps_combination in "${combinations[@]}"; do
            echo "MPS Combination: $mps_combination"

            IFS=' ' read -ra mps_percentages <<< "$mps_combination"

            # Check if the number of percentages matches the number of workloads
            if [ ${#mps_percentages[@]} -ne ${#workload_names[@]} ]; then
                echo "Mismatch in number of workloads and MPS percentages."
                continue
            fi

            # Start Docker instances for each workload
            pids=()  # Reset PIDs array for each percentage combination

            for (( i=0; i<${#workload_names[@]}; i++ )); do
                MPS_PERCENTAGE=${mps_percentages[$i]}

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
                --batch_size ${batchsizes[$i]} --log_dir ../../tests/mps/multiinstance/$CURR_LOG_DIR --profile_nstep 2000" \
                > "$CURR_LOG_DIR/w$((i+1))_${workload_names[$i]}_batch${batchsizes[$i]}-${modes[$i]}_MPS${MPS_PERCENTAGE}.log" 2>&1 &

                pids+=($!)  # Store PID
            done

            # Start monitoring GPU usage
            nvidia-smi pmon -o DT -i 1 -f "$CURR_LOG_DIR/gpu_info_${mps_combination// /_}.csv" &
            nvidia-smi -i 1 --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,memory.total,memory.free,memory.used,power.draw,power.limit --format=csv --loop-ms=500 \
            > "$CURR_LOG_DIR/gpu_mem_${mps_combination// /_}.csv" &

            # Wait for any Docker instance to finish
            wait -n "${pids[@]}"
            echo "Processes have finished. Cleaning up..."
            sleep 1

            # Terminate Docker instances
            for (( i=0; i<${#workload_names[@]}; i++ )); do
                docker kill "w$((i+1))" 2>/dev/null
            done
            bash ../terminate_client.sh
            sudo kill -9 $(pgrep -f "docker run") 2>/dev/null
            sudo kill -9 $(pgrep -f "nvidia-smi pmon") 2>/dev/null
            sudo kill -9 $(pgrep -f "nvidia-smi") 2>/dev/null
            wait "${pids[@]}" 2>/dev/null
            sleep 5

        done  # End of MPS combinations loop
    done
    echo "-------"
done < <(tail -n +2 "$CSV_FILE")
