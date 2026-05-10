FROM nvcr.io/nvidia/pytorch:23.10-py3
WORKDIR /app
RUN pip install vllm
RUN pip install openai==0.28
#COPY ./aladdin-main .
SHELL ["/bin/bash", "-c"]


#run docker - docker run  --rm -it --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 --env NVIDIA_VISIBLE_DEVICES=0 --gpus device=0 --ipc=host -p 8000:8000   nba556677/chatbot:vllm bash
#entrypointncommand - 
#python -m vllm.entrypoints.openai.api_server     --model mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.9 --max-model-len 2000
