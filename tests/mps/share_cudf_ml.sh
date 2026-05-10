

#bash mps_percentage_test.sh rtx6000_logs/sharetest IAT/0301/0301_100reqs_lambd2.0.json 2>rtx6000_logs/sharetest/err.log 1>rtx6000_logs/sharetest/stdout.log
LOG_DIR=$1
arrival_file=$2
# add $3 as one of two modes as train or inference. store in BE_MODE add help message as well
BE_MODE=$3
#BE_MODE should be train or inference only
if [ "$BE_MODE" != "train" ] && [ "$BE_MODE" != "inference" ]; then
    echo "BE_MODE should be train or inference only"
    exit 1
fi

TASKS=("recommend" "imgclassification")
MODELS=("bert-base-cased" "microsoft/resnet-50")
#task:model:epoch:batch_size

#INFERENCE
#task:model:epoch:batch_size
#LS_MODELS=("speech-recognition:facebook/wav2vec2-base-960h:0:1" "speech-recognition:openai/whisper-large-v2:0:1" "imgclassification:google/vit-base-patch16-224:0:1")
LS_MODELS=("LS_cudf")
#TASKS_MODELS=("imgclassification:microsoft/resnet-50:0:32" "speech-recognition:facebook/wav2vec2-base-960h:0:1" "imgclassification:google/mobilenet_v2_1.0_224:0:64")
TASKS_MODELS=("recommend:bert-base-cased:2:8")
#BE_EPOCHS=(7 7) 
LS_PERCENTAGES=(30 50 70 90)
BE_PERCENTAGES=(70 50 30 10)
#LS_PERCENTAGES=(0)
#BE_PERCENTAGES=(100)
RUNS=1


mkdir -p $LOG_DIR
for RUN in $(seq 1 $RUNS); do 
    echo RUN$RUN...;
    # Loop through each LS_PERCENTAGE and BE_PERCENTAGE pair
    
    for ((i=0; i<${#LS_PERCENTAGES[@]}; i++)); do
        LS_PERCENTAGE=${LS_PERCENTAGES[$i]}
        BE_PERCENTAGE=${BE_PERCENTAGES[$i]}
        echo "LS percent=$LS_PERCENTAGE, BE percent=$BE_PERCENTAGE"
        
        # Loop through each TASK
        #loop through LS_MODELS as outer loop
        LS_MODEL=${LS_MODELS[0]}
        for TASK_MODEL in "${TASKS_MODELS[@]}"; do
            IFS=':' read -r TASK MODEL BE_EPOCH BE_BATCH <<< "$TASK_MODEL"
            echo "TASK=$TASK, MODEL=$MODEL BE_EPOCH=$BE_EPOCH BATCH=$BE_BATCH"
            CURR_LOG_DIR="$LOG_DIR/RUN${RUN}/LS${LS_PERCENTAGE}/${LS_TASK}/${LS_MODEL}/BE_${MODEL}"
            echo "CURR_LOG_DIR=$CURR_LOG_DIR"
            mkdir -p "$CURR_LOG_DIR"
            
            # Start LS server only when LS_PERCENTAGE is not 0
            if [ $LS_PERCENTAGE -eq 0 ]; then
                echo "LS_PERCENTAGE is 0. Skipping LS server..."
            else
                # Start LS server
                echo "Starting LS server for cudf"
                cd /home/cc/db-benchmark
                export NVIDIA_VISIBLE_DEVICES=1 
                CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${LS_PERCENTAGE} bash runcudf.sh > /home/cc/mlProfiler/tests/mps/$CURR_LOG_DIR/cudf_read_csv_MPS${LS_PERCENTAGE}.log &
            fi

            cd /home/cc/mlProfiler/tests/mps

            
            #BEMODE = train
            if [ "$BE_MODE" == "train" ]; then
                echo "BE_MODE=train"
                # Start BE job
                docker run --rm --name BE --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$BE_PERCENTAGE \
                --env NVIDIA_VISIBLE_DEVICES=1 --gpus device=1 \
                -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                -v ~/.cache/huggingface:/root/.cache/huggingface \
                -v ~/.cache/torch:/root/.cache/torch \
                -v ~/mlProfiler:/root/mlprofiler --ipc=host \
                --cap-add=SYS_ADMIN -v /opt/nvidia/nsight-systems/2023.3.3:/nsys \
                nba556677/ml_tasks:latest\
                /bin/sh -c "cd /root/mlprofiler/workloads/training; python ${TASK}-train.py --n_epoch ${BE_EPOCH} \
                --model_name ${MODEL} --batch_size ${BE_BATCH}" \
                > $CURR_LOG_DIR/BE_${TASK}_MPS${BE_PERCENTAGE}.log 2>&1
            else
                echo "BE_MODE=inference"
                # Start BE job
                docker run --rm --name BE --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$BE_PERCENTAGE \
                --env NVIDIA_VISIBLE_DEVICES=1 --gpus device=1 \
                -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                -v ~/.cache/huggingface:/root/.cache/huggingface \
                -v ~/.cache/torch:/root/.cache/torch \
                -v ~/mlProfiler:/root/mlprofiler --ipc=host \
                --cap-add=SYS_ADMIN -v /opt/nvidia/nsight-systems/2023.3.3:/nsys \
                nba556677/ml_tasks:latest \
                /bin/sh -c "cd /root/mlprofiler/workloads/inference; python ${TASK}-inference.py --model_name ${MODEL} \
                --batch_size ${BE_BATCH} --log_dir ../../tests/mps/$CURR_LOG_DIR"  \
                > $CURR_LOG_DIR/BE_${TASK}_MPS${BE_PERCENTAGE}.log 2>&1
            fi
            
            
            echo "BE job finishes. cleaning up poisson..."
            sleep 1
            # Terminate client
            # Terminate client
            bash terminate_client.sh
            sudo kill -9 $(pgrep -f "docker run")
            sudo kill -9 $(pgrep -f "bash runcudf.sh")
            sudo kill -9 $(pgrep -f "python3 -m")
            sleep 5
        done
    done
    
done