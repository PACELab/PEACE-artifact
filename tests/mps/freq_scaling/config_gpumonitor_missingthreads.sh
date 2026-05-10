# Number of workloads per combination
N_COMBINATIONS=3
RUNS=1

# Path to the missing combinations CSV file
CSV_FILE="/home/cc/mlProfiler/tests/mps/eval_baselines/gslice/gslice_missing_thread_powercap60.csv"  # Replace with actual path

# Add valid frequency scaling factors (kept for potential future use)
VALID_FREQS=(1530)  
TIMEOUT=330

# Array of sleep times between each job (in seconds)
SLEEP_TIMES=(0)

# Array of workload names
workload_array=('wav2vec2-base-960h' 'bert-base-cased' 'mobilenet' 'mobilenet_v2_1.0_224' 'vit-base-patch16-224' 'vit_h_14' 'whisper-large-v2' 'albert-base-v2' 'gpt2-xl' 'resnet-50')

# Mapping of workload names to prefixes
declare -A workloadprefix
workloadprefix['wav2vec2-base-960h']='facebook/'
workloadprefix['bert-base-cased']=''
workloadprefix['mobilenet']=''
workloadprefix['mobilenet_v2_1.0_224']='google/'
workloadprefix['vit-base-patch16-224']='google/'
workloadprefix['vit_h_14']=''
workloadprefix['whisper-large-v2']='openai/'
workloadprefix['albert-base-v2']=''
workloadprefix['gpt2-xl']=''
workloadprefix['resnet-50']='microsoft/'

# Mapping of workload names to tasks
declare -A task_model
task_model['wav2vec2-base-960h']='speech-recognition'
task_model['bert-base-cased']='recommend'
task_model['mobilenet']='imgclassification'
task_model['mobilenet_v2_1.0_224']='imgclassification'
task_model['vit-base-patch16-224']='imgclassification'
task_model['resnet-50']='imgclassification'
task_model['vit_h_14']='imgclassification'
task_model['whisper-large-v2']='speech-recognition'
task_model['albert-base-v2']='recommend'
task_model['gpt2-xl']='autoregressive'

# GPU monitor arguments (PowerCap comes from MISSING_CSV)
GPUMON_DEBUG=""
GPUMON_GPU_INDEX=1
GPUMON_START_FREQ=1305
GPUMON_RUNTIME=9999999999999
GPUMON_MONITOR_INTERVAL_MS=500
GPUMON_CONSECUTIVE_EXCEED=3
GPUMON_MAX_FREQ=1530
GPUMON_MIN_FREQ=100
GPUMON_ADJUST_STEP=100
GPU_POWERCAP_THRES=0.75  # Still used to adjust PowerCap from CSV
GPU_POWERCAP_VALUE=60 #for fixedpower