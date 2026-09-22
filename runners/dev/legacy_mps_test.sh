#!/bin/bash

# Define an array to hold the models
LS_MODELS=()

# Define tasks and their corresponding models as strings in an array
train_candidates=(
    "recommend:bert-base-cased"
    "imgclassification:mobilenet"
    "imgclassification:vit_h_14"
    "recommend:albert/albert-base-v2"
    "imgclassification:resnet-50"
)

# Epoch is a fixed value, you can change it as needed
epoch=200000

# Array of batch sizes
batch_sizes=(4 32 64)

# Loop over tasks and models strings
for task_model in "${train_candidates[@]}"; do
    # Split the string into task and model using IFS (Internal Field Separator)
    IFS=":" read -r task model <<< "$task_model"
    for batch_size in "${batch_sizes[@]}"; do
        # Append the task:model:epoch:batch_size to the LS_MODELS array
        LS_MODELS+=("${task}:${model}:${epoch}:${batch_size}")
    done
done


#construct TASK_MODELS for inference
#INFERENCE
#task:model:epoch:batch_size
#inference
TASK_MODELS=()
inference_candidates=(
    "speech-recognition:openai/whisper-large-v2"
    "imgclassification:google/mobilenet_v2_1.0_224"
    "recommend:bert-base-cased"
    "imgclassification:google/vit-base-patch16-224"
    "imgclassification:microsoft/resnet-50"
    "speech-recognition:facebook/wav2vec2-base-960h"
)
epoch=0

# Loop over tasks and models strings
for task_model in "${inference_candidates[@]}"; do
    # Split the string into task and model using IFS (Internal Field Separator)
    IFS=":" read -r task model <<< "$task_model"
    for batch_size in "${batch_sizes[@]}"; do
        # Append the task:model:epoch:batch_size to the LS_MODELS array
        TASK_MODELS+=("${task}:${model}:${epoch}:${batch_size}")
    done
done
# Print the array to verify
echo "LS_MODELS array contents:"
for entry in "${LS_MODELS[@]}"; do
    echo "$entry"
done

echo "TASK_MODELS array contents:"
for entry in "${TASK_MODELS[@]}"; do
    echo "$entry"
done
