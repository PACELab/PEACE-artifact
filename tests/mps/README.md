# run chatbot docker 

```
docker run  --rm -it --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 --env NVIDIA_VISIBLE_DEVICES=0 --gpus device=0 -v /tmp/nvidia-mps:/tmp/nvidia-mps -v ~/.cache/huggingface:/root/.cache/huggingface --ipc=host --cap-add=SYS_ADMIN -p 8000:8000  nba556677/chatbot:vllm bash
```

# run vllm server
```
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=75 python -m vllm.entrypoints.openai.api_server     --model mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.65 --max-model-len 2000 --dtype=half


docker run --rm  --name LS --runtime nvidia -v /usr/local/cuda:/nsys -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/mlProfiler:/root/mlprofiler --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100  --env NVIDIA_VISIBLE_DEVICES=0 --gpus device=0 --env "HUGGING_FACE_HUB_TOKEN=<access_token>" --cap-add=SYS_ADMIN --ipc=host -p 8000:8000  /nsys/bin/nsys profile -t cuda,nvtx vllm/vllm-openai:latest --model mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.65 --max-model-len 2000  --dtype=half
```
# run Best effort docker 

```
docker run --rm --name BE -it -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/.cache/torch:/root/.cache/torch --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 --env NVIDIA_VISIBLE_DEVICES=0 --gpus device=0 -v /tmp/nvidia-mps:/tmp/nvidia-mps -v ~/mlProfiler:/root/mlprofiler --ipc=host nba556677/ml_tasks:latest bash
```
# entrypoint command
```
# inference
cd /root/mlprofiler/workloads/inference/ && python imgclassification-inference.py --model_name microsoft/resnet-50 --batch_size 64  --log_dir $log_dir

cd /root/mlprofiler/workloads/inference/ &&  python speech-recognition-inference.py --model_name facebook/wav2vec2-base-960h --log_dir ../../tests/mps/tmp --batch_size 2
```

# run Best Effort job
```
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 python imgclassification-train.py --model_name microsoft/resnet-50 --batch_size 8 --n_epoch 10 > img_25_PERCENTAGE.log

```


# running NSYS

```
docker run --rm --name BE -it -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/.cache/torch:/root/.cache/torch --cap-add=SYS_ADMIN -v /home/cc/nsight-systems-2024.2.1:/nsys --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 --env NVIDIA_VISIBLE_DEVICES=0 --gpus device=0 -v /tmp/nvidia-mps:/tmp/nvidia-mps -v ~/mlProfiler:/root/mlprofiler --ipc=host nba556677/ml_tasks:latest bash
cd ~/mlprofiler/workloads/inference
```
# nsys
```
/nsys/bin/nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --cudabacktrace=true -x true -o ../../tests/mps/analysis/kernel_profiles/source/nsys/whisper-large-v2_batch8-inf  python speech-recognition-inference.py --model_name openai/whisper-large-v2 --batch_size 8   --profile_nstep 12  --profile
/nsys/bin/nsys stats -r cuda_gpu_sum -o train_vit_h_14.csv  report1.nsys-rep
```

# ncu mem thrpughput in docker >0 -> profile with host
## Please refer to tests/mps/kernel/profile for more details
```
#PREREQUISITE: install pytorch in host conda environment first
sudo /usr/local/cuda/bin/ncu  --nvtx --nvtx-include "steps10/"  --call-stack --target-processes all --verbose  --set full --csv -o ../../tests/mps/analysis/kernel_profiles/source/ncu/wav2vec2-base-960h_batch2-inf /home/cc/miniconda3/bin/python speech-recognition-inference.py --model_name facebook/wav2vec2-base-960h --batch_size 2 --profile_nstep 12 --profile
```
# export ncu
```
ncu -i [ncu-rep file] --csv   --print-details  all > vit_h_14-train_ncu.csv
```
```
ncu -i albert-train.ncu-rep  --csv --print-summary per-kernel > albert_train_ncu.csv

#ncu -i  albert-train.ncu-rep --csv --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.#pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,launch__grid_size,launch__registers_per_thread,launch__block_size,#launch__waves_per_multiprocessor,launch__shared_mem_per_block_static,sm__warps_active.avg.per_cycle_active,sm__instruction_throughput.avg.#pct_of_peak_sustained_active > albert_train__ncu.csv
```