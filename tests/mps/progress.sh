#!/usr/bin/bash
sudo /usr/local/cuda/bin/nsys profile  -o vitinf_nsysprofile -s process-tree  -t cuda,nvtx      /home/cc/miniconda3/bin/python /home/cc/mlProfiler/workloads/inference/imgclassification-inference.py  --model_name google/vit-base-patch16-224  --batch_size 1 --profile_1step
sudo /usr/local/cuda/bin/nsys profile  -o wav2vec_nsysprofile -s process-tree  -t cuda,nvtx      /home/cc/miniconda3/bin/python /home/cc/mlProfiler/workloads/inference/speech-recognition-inference.py  --model_name facebook/wav2vec2-base-960h  --batch_size 1 --profile_1step
sudo /usr/local/cuda/bin/nsys profile  -o res50inf_nsysprofile -s process-tree  -t cuda,nvtx     /home/cc/miniconda3/bin/python /home/cc/mlProfiler/workloads/inference/imgclassification-inference.py  --model_name microsoft/resnet-50 --batch_size 32 --profile_1step
sudo /usr/local/cuda/bin/nsys profile  -o whisper_nsysprofile -s process-tree  -t cuda,nvtx     /home/cc/miniconda3/bin/python /home/cc/mlProfiler/workloads/inference/speech-recognition-inference.py  --model_name openai/whisper-large-v2  --batch_size 1 --profile_1step

