#share 100% MPS active thread percentage LS server with CPU job. control knob - set cores to 50% of total cores


#bash mps_percentage_test.sh rtx6000_logs/sharetest IAT/0301/0301_100reqs_lambd2.0.json 2>rtx6000_logs/sharetest/err.log 1>rtx6000_logs/sharetest/stdout.log
LOG_DIR=$1
arrival_file=$2
DEVICE="cuda"
RUNS=1
# add $3 as one of two modes as train or inference. store in BE_MODE add help message as well
BE_MODE=$3

#add usage to arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 LOG_DIR ARRIVAL_FILE BE_MODE"
    exit 1
fi
#BE_MODE should be train or inference only
if [ "$BE_MODE" != "train" ] && [ "$BE_MODE" != "inference" ]; then
    echo "BE_MODE should be train or inference only"
    exit 1
fi
#get total num of cores -2 for the system
TOTAL_CORES=$(($(nproc) - 2))

LS_PERCENTAGES=(10 20 30 40 50 60 70 80 90)
BE_PERCENTAGES=(90 80 70 60 50 40 30 20 10)
#LS_PERCENTAGES=(100)
#BE_PERCENTAGES=(0)

#TRAINING
#TASKS_MODELS=("imgclassification:microsoft/resnet-50:1:8" "recommend:bert-base-cased:7:8" "imgclassification:mobilenet:1:8")
#inference
TASKS_MODELS=("imgclassification:google/mobilenet_v2_1.0_224:0:64" "imgclassification:google/vit-base-patch16-224:0:64" "speech-recognition:facebook/wav2vec2-base-960h:0:1" "imgclassification:microsoft/resnet-50:0:32")
#set cores to LS_PERCENTAGE of total cores 

echo "total_cores=$TOTAL_CORES"

#create a ml inference job that constantly runs on CPU. the job =  {docker run --rm --name BE -it -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/.cache/torch:/root/.cache/torch --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 --env NVIDIA_VISIBLE_DEVICES=0 --gpus device=0 -v /tmp/nvidia-mps:/tmp/nvidia-mps -v ~/mlProfiler:/root/mlprofiler --ipc=host nba556677/ml_tasks:latest /bin/sh -c "cd /root/mlprofiler/workloads/inference; python imgclassification-inference.py --model_name microsoft/resnet-50 --batch_size 8 --log_dir ../../tests/mps/tmp/cpu --device cpu}
#repeateadly submit the job once it finishes, but still allow main script to continue
#docker run --rm --name BE --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 --env NVIDIA_VISIBLE_DEVICES=0 --gpus device=0 -v /tmp/nvidia-mps:/tmp/nvidia-mps -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/.cache/torch:/root/.cache/torch -v ~/mlProfiler:/root/mlprofiler --ipc=host nba556677/ml_tasks:latest /bin/sh -c "cd /root/mlprofiler/workloads/inference; python imgclassification-inference.py --model_name microsoft/resnet-50 --batch_size 8 --log_dir ../../tests/mps/tmp/cpu --device cpu"
#docker run --rm --name BE --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 --env NVIDIA_VISIBLE_DEVICES=0 --gpus device=0 -v /tmp/nvidia-mps:/tmp/nvidia-mps -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/.cache/torch:/root/.cache/torch -v ~/mlProfiler:/root/mlprofiler --ipc=host nba556677/ml_tasks:latest /bin/sh -c "cd /root/mlprofiler/workloads/inference; python imgclassification-inference.py --model_name microsoft/resnet-50 --batch_size 8 --log_dir ../../tests/mps/tmp/cpu --device cpu"
long_run_CPU_inference() {
    i1=0
    while true; do
        # Run the inference job on CPU
        docker run --rm --name BE_CPU -v ~/.cache/huggingface:/root/.cache/huggingface \
                -v ~/.cache/torch:/root/.cache/torch -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                -v ~/mlProfiler:/root/mlprofiler --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 \
                --env NVIDIA_VISIBLE_DEVICES=0 --gpus device=0 --ipc=host \
                nba556677/ml_tasks:latest /bin/sh -c "cd /root/mlprofiler/workloads/inference; \
                python imgclassification-inference.py --model_name microsoft/resnet-50 \
                --batch_size 8 --log_dir ../../tests/mps/$1 --device cpu"  \
                > $1/CPUBE_$i1.log 2>&1

        
        echo "cpu inference counter $i1" 
        
        #increment counter
        i1=$((i1+1))
        # Wait for 1 seconds before retrying CPU inference
        sleep 1
    done
}



#set MPS active thread percentage to 100%
#export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100

#start long running CPU inference
echo "start long run CPU inference"
mkdir -p $LOG_DIR/longrun_cpu && long_run_CPU_inference $LOG_DIR/longrun_cpu &
run_CPU_inference_pid=$!

mkdir -p $LOG_DIR
for RUN in $(seq 1 $RUNS); do 
    echo RUN$RUN...;
    


    # Loop through each LS_PERCENTAGE and BE_PERCENTAGE pair
    for ((i=0; i<${#LS_PERCENTAGES[@]}; i++)); do
        LS_PERCENTAGE=${LS_PERCENTAGES[$i]}
        BE_PERCENTAGE=${BE_PERCENTAGES[$i]}
        #set num of cores of LS_CORES and BE_CORES based on percentage
        LS_CORES=$(($TOTAL_CORES * $LS_PERCENTAGE / 100))
        BE_CORES=$(($TOTAL_CORES * $BE_PERCENTAGE / 100))
        echo "LS percent=$LS_PERCENTAGE, BE percent=$BE_PERCENTAGE"
        echo "LS_CORES=$LS_CORES, BE_CORES=$BE_CORES"
        LS_LAST=$(($LS_CORES + 1))
        BE_START=$(($LS_CORES + 2))
        BE_END=$(($LS_CORES + $BE_CORES + 1))
        echo "LS_LAST=$LS_LAST, BE_START=$BE_START, BE_END=$BE_END"
        
        # Loop through each TASK
        for TASK_MODEL in "${TASKS_MODELS[@]}"; do
            IFS=':' read -r TASK MODEL BE_EPOCH BE_BATCH <<< "$TASK_MODEL"
            echo "TASK=$TASK, MODEL=$MODEL BE_EPOCH=$BE_EPOCH BATCH=$BE_BATCH"
            CURR_LOG_DIR="$LOG_DIR/RUN${RUN}/LS${LS_PERCENTAGE}/${TASK}/${MODEL}"
            echo "CURR_LOG_DIR=$CURR_LOG_DIR"
            mkdir -p "$CURR_LOG_DIR"
            
            # Start LS server
            #use cpu-set-cpus=2-LS_LAST to set the number of cores for LS
            #use cpu-set-cpus=BE_START-BE_END to set the number of cores for BE
            #--cpuset-cpus=2-$LS_LAST --cpuset-cpus=$BE_START-$BE_END \

            docker run --rm --name LS --runtime nvidia --gpus all \
            -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/mlProfiler:/root/mlprofiler -v /tmp/nvidia-mps:/tmp/nvidia-mps \
            --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$LS_PERCENTAGE \
            --env NVIDIA_VISIBLE_DEVICES=0 --gpus device=0 \
            --env "HUGGING_FACE_HUB_TOKEN=<access_token>" \
            --cap-add=SYS_ADMIN --ipc=host -p 8000:8000 \
            vllm/vllm-openai:latest --model mistralai/Mistral-7B-Instruct-v0.2 \
            --gpu-memory-utilization 0.65 --max-model-len 2000 --dtype=half &
            
            # Wait for LS server to warm up
            echo "Warming up LS server..."
            sleep 60
            
            # Start sending LS requests
            echo "Start sending LS requests from $arrival_file..."
            #log should be flushed realtime
            stdbuf -oL python poisson_arrival.py --ip 127.0.0.1 -l 2 -n 500 -f $arrival_file \
            > "$CURR_LOG_DIR/arrival_${TASK}_LSMPS${LS_PERCENTAGE}.log" 2>&1 &
            
            # Log metrics for LS server
            python vllm_logging.py $CURR_LOG_DIR > "$CURR_LOG_DIR/LSmetric_${TASK}_MPS${LS_PERCENTAGE}.log" 2>&1 &
            
            
            #BEMODE = train
            if [ "$BE_MODE" == "train" ]; then
                echo "BE_MODE=train"
                # Start BE job
                docker run --rm --name BE1 --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$BE_PERCENTAGE \
                --env NVIDIA_VISIBLE_DEVICES=0 --gpus device=0 \
                -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                -v ~/.cache/huggingface:/root/.cache/huggingface \
                -v ~/.cache/torch:/root/.cache/torch \
                -v ~/mlProfiler:/root/mlprofiler --ipc=host \
                nba556677/ml_tasks:latest \
                /bin/sh -c "cd /root/mlprofiler/workloads/training; python ${TASK}-train.py --n_epoch ${BE_EPOCH} \
                --model_name ${MODEL} --batch_size ${BE_BATCH} --log_dir ../../tests/mps/$CURR_LOG_DIR/BE1 --device ${DEVICE}" \
                > $CURR_LOG_DIR/BE1_${TASK}_MPS${BE_PERCENTAGE}.log 2>&1
            else
                echo "BE_MODE=inference"
                # Start BE job
                docker run --rm --name BE1 --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$BE_PERCENTAGE \
                --env NVIDIA_VISIBLE_DEVICES=0 --gpus device=0 \
                -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                -v ~/.cache/huggingface:/root/.cache/huggingface \
                -v ~/.cache/torch:/root/.cache/torch \
                -v ~/mlProfiler:/root/mlprofiler --ipc=host \
                nba556677/ml_tasks:latest\
                /bin/sh -c "cd /root/mlprofiler/workloads/inference; python ${TASK}-inference.py --model_name ${MODEL} \
                --batch_size ${BE_BATCH} --log_dir ../../tests/mps/$CURR_LOG_DIR/BE1 --device ${DEVICE}"  \
                > $CURR_LOG_DIR/BE1_${TASK}_MPS${BE_PERCENTAGE}.log 2>&1 


            fi
            
            
            echo "BE job finishes. cleaning up poisson..."
            sudo kill -9 $(pgrep -f "python poisson")
            sudo kill -15 $(pgrep -f "python vllm_logging.py")
            sleep 1
            # Terminate client
            bash terminate_client.sh
            #kill LS job
            docker kill LS
            #sudo kill -9 $(pgrep -f "docker run")
            sleep 5
        done
    done
done

echo "kill long run CPU inference"
kill -9 $run_CPU_inference_pid && docker kill BE_CPU