#bash baseline_freqscale_triton.sh <log_dir> <arrival_file> <LS_MODE> <BE_MODE> <GPU_DEVICE>
# SoCC'26 Rebuttal: Baseline runs for Triton-kernel workloads (solo, no colocation)
# Same output format as baseline_freqscale.sh but with Triton-compiled custom kernels.
#
# Usage:
#   bash baseline_freqscale_triton.sh <log_dir> <arrival_file> <LS_MODE> <BE_MODE> [GPU_DEVICE]
#
# Examples:
#   bash baseline_freqscale_triton.sh ../../ccv100_logs/baseline_triton arrival.txt train train 0
#   bash baseline_freqscale_triton.sh ../../ccv100_logs/baseline_triton arrival.txt inference inference 1

LOG_DIR=$1
arrival_file=$2
LS_MODE=$3
BE_MODE=$4
GPU_DEVICE=${5:-1}
# Valid frequency scaling factors
VALID_FREQS=(1530)
TIMEOUT=60

if [ "$LS_MODE" != "train" ] && [ "$LS_MODE" != "inference" ]; then
    echo "LS_MODE should be train or inference only"
    exit 1
fi

#BE_MODE should be train or inference only
if [ "$BE_MODE" != "train" ] && [ "$BE_MODE" != "inference" ]; then
    echo "BE_MODE should be train or inference only"
    exit 1
fi
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


#task:model:epoch:batch_size

#INFERENCE
#task:model:epoch:batch_size
LS_MODELS=("speech-recognition-triton:facebook/wav2vec2-base-960h-triton:0:1")

TASKS_MODELS=()
batch_sizes=(2)
train_candidates=(
    #"recommend-triton:albert-base-v2-triton"
    #"imgclassification-triton:resnet-50-triton"
    #"recommend-triton:bert-base-cased-triton"
    "imgclassification-triton:mobilenet-triton"
    #"imgclassification-triton:vit_h_14-triton"
)

inference_candidates=(
    "speech-recognition-triton:facebook/wav2vec2-base-960h-triton"
    "recommend-triton:bert-base-cased-triton"
    "speech-recognition-triton:openai/whisper-large-v2-triton"
    "imgclassification-triton:google/mobilenet_v2_1.0_224-triton"
    "imgclassification-triton:google/vit-base-patch16-224-triton"
    "imgclassification-triton:microsoft/resnet-50-triton"
)

if [ "$BE_MODE" == "train" ]; then
    epoch=2
    # Loop over tasks and models strings
    for task_model in "${train_candidates[@]}"; do
        # Split the string into task and model using IFS (Internal Field Separator)
        IFS=":" read -r task model <<< "$task_model"
        for batch_size in "${batch_sizes[@]}"; do
            # Append the task:model:epoch:batch_size to the LS_MODELS array
            TASKS_MODELS+=("${task}:${model}:${epoch}:${batch_size}")
        done
    done
fi

if [ "$BE_MODE" == "inference" ]; then
    epoch=0
    # Loop over tasks and models strings
    for task_model in "${inference_candidates[@]}"; do
        # Split the string into task and model using IFS (Internal Field Separator)
        IFS=":" read -r task model <<< "$task_model"
        for batch_size in "${batch_sizes[@]}"; do
            # Append the task:model:epoch:batch_size to the LS_MODELS array
            TASKS_MODELS+=("${task}:${model}:${epoch}:${batch_size}")
        done
    done
fi

#LS_PERCENTAGES: 0 means no colocation (baseline solo run)
#LS_PERCENTAGES=(0 0 0 0 0 0 0 0 0 0)
#BE_PERCENTAGES=(10 20 30 40 50 60 70 80 90 100)
LS_PERCENTAGES=(0)
BE_PERCENTAGES=(20)
RUNS=1
mkdir -p $LOG_DIR
for RUN in $(seq 1 $RUNS); do
    echo RUN$RUN...;
    # Loop through each LS_PERCENTAGE and BE_PERCENTAGE pair
    for LS_MODEL in "${LS_MODELS[@]}"; do
        IFS=':' read -r LS_TASK LS_MODEL LS_EPOCH LS_BATCH <<< "$LS_MODEL"
        echo "LS_TASK=$LS_TASK, LS_MODEL=$LS_MODEL LS_EPOCH=$LS_EPOCH LS_BATCH=$LS_BATCH"
        for ((i=0; i<${#LS_PERCENTAGES[@]}; i++)); do
            LS_PERCENTAGE=${LS_PERCENTAGES[$i]}
            BE_PERCENTAGE=${BE_PERCENTAGES[$i]}
            echo "LS percent=$LS_PERCENTAGE, BE percent=$BE_PERCENTAGE"

            for TASK_MODEL in "${TASKS_MODELS[@]}"; do
                IFS=':' read -r TASK MODEL BE_EPOCH BE_BATCH <<< "$TASK_MODEL"
                echo "TASK=$TASK, MODEL=$MODEL BE_EPOCH=$BE_EPOCH BATCH=$BE_BATCH"
                for FREQ_SCALING in "${VALID_FREQS[@]}"; do
                    echo "Running with frequency scaling: $FREQ_SCALING MHz"
                    CURR_LOG_DIR="$LOG_DIR/FREQ${FREQ_SCALING}/RUN${RUN}/LS${LS_PERCENTAGE}/${LS_TASK}/${LS_MODEL}_batch${LS_BATCH}/BE_${MODEL}_batch${BE_BATCH}"
                    echo "CURR_LOG_DIR=$CURR_LOG_DIR"
                    mkdir -p "$CURR_LOG_DIR"

                    # Set frequency scaling
                    echo "Setting frequency scaling to $FREQ_SCALING MHz..."
                    sudo nvidia-smi -lgc $FREQ_SCALING,$FREQ_SCALING
                    #wait 2 seconds to take effect
                    sleep 2
                    pids=()
                    # Start LS server only when LS_PERCENTAGE is not 0
                    if [ $LS_PERCENTAGE -eq 0 ]; then
                        echo "LS_PERCENTAGE is 0. Skipping LS server..."
                    else
                        # Start LS server
                        echo "Starting LS server..."
                        if [ "$LS_MODE" == "train" ]; then
                            docker run --rm --name LS --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$LS_PERCENTAGE \
                            --env NVIDIA_VISIBLE_DEVICES=$GPU_DEVICE --gpus device=$GPU_DEVICE \
                            -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                            -v ~/.cache/huggingface:/root/.cache/huggingface \
                            -v ~/.cache/torch:/root/.cache/torch \
                            -v ~/mlProfiler:/root/mlprofiler --ipc=host \
                            --cap-add=SYS_ADMIN -v /opt/nvidia/nsight-systems/2023.3.3:/nsys \
                            nba556677/ml_tasks:latest \
                            /bin/sh -c "cd /root/mlprofiler/workloads/train; python ${LS_TASK}-train.py --model_name ${LS_MODEL} --n_epoch ${LS_EPOCH}\
                            --batch_size ${LS_BATCH} --log_dir ../../tests/mps/$CURR_LOG_DIR"  \
                            > $CURR_LOG_DIR/LS_${LS_TASK}_FREQ${FREQ_SCALING}_MPS${LS_PERCENTAGE}.log 2>&1 & pids+=($!)
                        else
                            docker run --rm --name LS --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$LS_PERCENTAGE \
                            --env NVIDIA_VISIBLE_DEVICES=$GPU_DEVICE --gpus device=$GPU_DEVICE \
                            -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                            -v ~/.cache/huggingface:/root/.cache/huggingface \
                            -v ~/.cache/torch:/root/.cache/torch \
                            -v ~/mlProfiler:/root/mlprofiler --ipc=host \
                            --cap-add=SYS_ADMIN -v /opt/nvidia/nsight-systems/2023.3.3:/nsys \
                            nba556677/ml_tasks:latest \
                            /bin/sh -c "cd /root/mlprofiler/workloads/inference; python ${LS_TASK}-inference.py --model_name ${LS_MODEL} \
                            --batch_size ${LS_BATCH} --log_dir ../../tests/mps/$CURR_LOG_DIR --profile_nstep 200"  \
                            > $CURR_LOG_DIR/LS_${LS_TASK}_FREQ${FREQ_SCALING}_MPS${LS_PERCENTAGE}.log 2>&1 & pids+=($!)
                        fi
                    fi

                    #BEMODE = train
                    if [ "$BE_MODE" == "train" ]; then
                        echo "BE_MODE=train"
                        # Start BE job
                        docker run --rm --name BE --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$BE_PERCENTAGE \
                        --env NVIDIA_VISIBLE_DEVICES=$GPU_DEVICE --gpus device=$GPU_DEVICE \
                        -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                        -v ~/.cache/huggingface:/root/.cache/huggingface \
                        -v ~/.cache/torch:/root/.cache/torch \
                        -v ~/mlProfiler:/root/mlprofiler --ipc=host \
                        --cap-add=SYS_ADMIN -v /opt/nvidia/nsight-systems/2023.3.3:/nsys \
                        nba556677/ml_tasks:latest\
                        /bin/sh -c "cd /root/mlprofiler/workloads/train; python ${TASK}-train.py --n_epoch ${BE_EPOCH} \
                        --model_name ${MODEL} --batch_size ${BE_BATCH} --profile_nstep 500" \
                        > $CURR_LOG_DIR/BE_${TASK}_FREQ${FREQ_SCALING}_MPS${BE_PERCENTAGE}.log 2>&1  & pids+=($!)
                    else
                        echo "BE_MODE=inference"
                        # Start BE job
                        docker run --rm --name BE --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$BE_PERCENTAGE \
                        --env NVIDIA_VISIBLE_DEVICES=$GPU_DEVICE --gpus device=$GPU_DEVICE \
                        -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                        -v ~/.cache/huggingface:/root/.cache/huggingface \
                        -v ~/.cache/torch:/root/.cache/torch \
                        -v ~/mlProfiler:/root/mlprofiler --ipc=host \
                        --cap-add=SYS_ADMIN -v /opt/nvidia/nsight-systems/2023.3.3:/nsys \
                        nba556677/ml_tasks:latest \
                        /bin/sh -c "cd /root/mlprofiler/workloads/inference; python ${TASK}-inference.py --model_name ${MODEL} \
                        --batch_size ${BE_BATCH} --log_dir ../../tests/mps/$CURR_LOG_DIR --profile_nstep 200"  \
                        > $CURR_LOG_DIR/BE_${TASK}_FREQ${FREQ_SCALING}_MPS${BE_PERCENTAGE}.log 2>&1 & pids+=($!)
                    fi

                    nvidia-smi pmon -o DT -i $GPU_DEVICE > $CURR_LOG_DIR/BE_${TASK}_FREQ${FREQ_SCALING}_MPS${BE_PERCENTAGE}_gpu_info.csv &
                    nvidia-smi -i $GPU_DEVICE --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,memory.total,memory.free,memory.used,power.draw,power.limit --format=csv --loop-ms=500  > $CURR_LOG_DIR/BE_${TASK}_FREQ${FREQ_SCALING}_MPS${BE_PERCENTAGE}_gpu_mem.csv &
                    dcgmi dmon -e 1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012  -i $GPU_DEVICE | awk '{print strftime("%Y-%m-%d %H:%M:%S"), $0}' > $CURR_LOG_DIR/BE_${TASK}_FREQ${FREQ_SCALING}_MPS${BE_PERCENTAGE}_dcgm_info.csv &

                    # Wait for any process to finish and kill the other
                    wait_any_with_timeout $TIMEOUT "${pids[@]}"
                    echo "One process finished with exit code: $wait_exit_code"
                    echo "BE process has finished. Cleaning up..."
                    sleep 1

                    # Terminate client
                    docker kill LS BE
                    #bash terminate_client.sh
                    sudo kill -9 $(pgrep -f "docker run")
                    sudo kill -9 $(pgrep -f "nvidia-smi pmon")
                    sudo kill -9 $(pgrep -f "nvidia-smi")
                    sudo kill -9 $(pgrep -f "dcgmi dmon")

                    wait "${pids[@]}" 2>/dev/null
                    sleep 5
                done
            done
        done
    done

done
