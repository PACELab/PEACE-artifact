# config_missingthreads_triton.sh
# SoCC'26 Rebuttal: Re-run missing thread combinations for Triton-kernel workloads
#
# Used with auto_multiinstance_freqscale_missingthreads.sh to fill in
# gaps identified during the triton DL baseline collection.

# Number of workloads per combination
N_COMBINATIONS=2
RUNS=1

# Path to the missing combinations CSV file
CSV_FILE="/home/cc/mlProfiler/tests/mps/freq_scaling/dataset/triton_DL_socc26_basemetric_nontriton/missing_entries/all_missing_combinations_with_powercap.csv"

# Single frequency (missingthreads script uses VALID_FREQUENCY, not VALID_FREQS)
VALID_FREQUENCY=1530
TIMEOUT=110

# Sleep times between jobs
SLEEP_TIMES=(0)

# Workload arrays – Triton-kernel variants
workload_array=('albert-base-v2-triton' 'mobilenet-triton' 'resnet-50-triton' 'wav2vec2-base-960h-triton' 'bert-base-cased-triton' 'vit_h_14-triton' 'whisper-large-v2-triton' 'mobilenet_v2_1.0_224-triton' 'vit-base-patch16-224-triton')

# Mapping of workload names to HuggingFace/model prefixes
declare -A workloadprefix
workloadprefix['albert-base-v2-triton']=''
workloadprefix['mobilenet-triton']=''
workloadprefix['resnet-50-triton']=''
workloadprefix['wav2vec2-base-960h-triton']='facebook/'
workloadprefix['bert-base-cased-triton']=''
workloadprefix['vit_h_14-triton']=''
workloadprefix['whisper-large-v2-triton']='openai/'
workloadprefix['mobilenet_v2_1.0_224-triton']='google/'
workloadprefix['vit-base-patch16-224-triton']='google/'
# Original workloads (needed if CSV references non-triton names)
workloadprefix['albert-base-v2']=''
workloadprefix['mobilenet']=''
workloadprefix['mobilenet_v2_1.0_224']='google/'
workloadprefix['resnet-50']=''
workloadprefix['wav2vec2-base-960h']='facebook/'
workloadprefix['whisper-large-v2']='openai/'
workloadprefix['vit_h_14']=''
workloadprefix['vit-base-patch16-224']='google/'
workloadprefix['bert-base-cased']=''

# Mapping of workload names to task scripts
# Triton variants: {task}-triton-{mode}.py
declare -A task_model
task_model['albert-base-v2-triton']='recommend-triton'
task_model['mobilenet-triton']='imgclassification-triton'
task_model['resnet-50-triton']='imgclassification-triton'
task_model['wav2vec2-base-960h-triton']='speech-recognition-triton'
task_model['bert-base-cased-triton']='recommend-triton'
task_model['vit_h_14-triton']='imgclassification-triton'
task_model['whisper-large-v2-triton']='speech-recognition-triton'
task_model['mobilenet_v2_1.0_224-triton']='imgclassification-triton'
task_model['vit-base-patch16-224-triton']='imgclassification-triton'
# Original workloads (for reference/comparison runs)
task_model['albert-base-v2']='recommend'
task_model['mobilenet']='imgclassification'
task_model['mobilenet_v2_1.0_224']='imgclassification'
task_model['resnet-50']='imgclassification'
task_model['wav2vec2-base-960h']='speech-recognition'
task_model['whisper-large-v2']='speech-recognition'
task_model['vit_h_14']='imgclassification'
task_model['vit-base-patch16-224']='imgclassification'
task_model['bert-base-cased']='recommend'

# GPU monitor arguments (PowerCap comes from the CSV)
GPUMON_DEBUG=""
GPUMON_GPU_INDEX=1
GPUMON_START_FREQ=1305
GPUMON_RUNTIME=9999999999999
GPUMON_MONITOR_INTERVAL_MS=500
GPUMON_CONSECUTIVE_EXCEED=3
GPUMON_MAX_FREQ=1530
GPUMON_MIN_FREQ=300
GPUMON_ADJUST_STEP=100
GPU_POWERCAP_THRES=0.75
