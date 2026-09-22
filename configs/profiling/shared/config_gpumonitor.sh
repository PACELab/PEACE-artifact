# config.sh

# Number of workloads per combination
N_COMBINATIONS=2
RUNS=1
# Path to the CSV file containing workload combinations
#CSV_FILE='$REPO_ROOT/data/experiment_inputs/colocations/debug/1017_debugtest_comb2_batch2.csv'
#CSV_FILE="$REPO_ROOT/data/experiment_inputs/colocations/debug/staticfreq_debugtest_comb2_batch2.csv"
#all DL data
CSV_FILE="$REPO_ROOT/data/experiment_inputs/colocations/comb2/02062025_powercap100_DL_baseline_labels_comb2_batches2.csv"
#0915 not run data
#CSV_FILE="$REPO_ROOT/data/experiment_inputs/colocations/comb2/02062025_powercap100_DL_baseline_labels_comb2_batches2_filtered.csv"
#comb3
#CSV_FILE="$REPO_ROOT/data/experiment_inputs/colocations/comb3/09132025_freq1530_DL_baseline_labels_comb3_batches2.csv"
#comb3_debug
#CSV_FILE="$REPO_ROOT/data/experiment_inputs/colocations/debug/09132025_freq1530_DL_baseline_labels_comb3_batches2_debug.csv"
# Add valid frequency scaling factors
VALID_FREQS=(1530)  
#VALID_FREQS=(952 1147 1335 1530)
#SRIDE FOR combination skipping to reduce total number of combinations for faster profiling
SAMPLE_STRIDE=1
#timeout for colocation if not finished
TIMEOUT=300

# Array of sleep times between each job (in seconds)
SLEEP_TIMES=(0)

# Custom combinations: If specified, this will override the automatic generation of combinations.
# Format: Each array entry corresponds to a thread combination for all workloads in the form of space-separated percentages.
#custom_combinations=(
#  "100 100"
#  "10 90"
#  "30 70"
#  "50 50"
#  "70 30"
#  "90 10"
#  "20 80"
#  "40 60"
#  "60 40"
#  "80 20"
#)
# Array of workload names
workload_array=('wav2vec2-base-960h' 'bert-base-cased' 'mobilenet' 'mobilenet_v2_1.0_224' 'vit-base-patch16-224' 'vit_h_14' 'whisper-large-v2' 'vit-base-patch16-224' 'albert-base-v2' 'gpt2-xl')

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

# ------------------------------------------------------------------------------
# NEW: GPU monitor arguments in variables
# ------------------------------------------------------------------------------
GPUMON_DEBUG=""            # or "" if you want to disable debug
GPUMON_GPU_INDEX=1
GPUMON_POWER_CAP=-1
GPUMON_START_FREQ=1305
GPUMON_RUNTIME=9999999999999
GPUMON_MONITOR_INTERVAL_MS=500
GPUMON_CONSECUTIVE_EXCEED=3
GPUMON_MAX_FREQ=1530
GPUMON_MIN_FREQ=100
GPUMON_ADJUST_STEP=100
GPU_POWERCAP_THRES=0.75 #powercap threshold
GPU_POWERCAP_VALUE=250 #for fixedpower
