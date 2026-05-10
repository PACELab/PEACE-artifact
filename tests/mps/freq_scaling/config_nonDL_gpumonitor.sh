# config.sh

# Number of workloads per combination
N_COMBINATIONS=2
RUNS=1
# Path to the CSV file containing workload combinations
#misc-running samples without powercap set in the 02062025 run
CSV_FILE="/home/cc/mlProfiler/tests/mps/multiinstance/03042025_powercap100_nobothcudasamples_nonDL_baseline_labels_comb2_batches2.csv"
#data includes both cuda samples as pairs (those wont run)
#CSV_FILE="/home/cc/mlProfiler/tests/mps/multiinstance/02062025_powercap100_nonDL_baseline_labels_comb2_batches2.csv"
#data includes 1 cuda sample and 1 DL model. all powercap set with 100,100
#CSV_FILE="/home/cc/mlProfiler/tests/mps/multiinstance/03042025_powercap100_nobothcudasamples_nonDL_baseline_labels_comb2_batches2.csv"
#DEBUG - 09152025-finish temporal ppowercap200
#CSV_FILE="/home/cc/mlProfiler/tests/mps/multiinstance/09152025_powercap200_nonDL_dvfs_temporal_missingthread.csv"

# Add valid frequency scaling factors
VALID_FREQS=(900)  
#VALID_FREQS=(952 1147 1335 1530)
#timeout for colocation if not finished
TIMEOUT=99999999

# Array of sleep times between each job (in seconds)
SLEEP_TIMES=(0)

# Custom combinations: If specified, this will override the automatic generation of combinations.
# Format: Each array entry corresponds to a thread combination for all workloads in the form of space-separated percentages.
custom_combinations=(
  "100 100"
  "10 90"
  "30 70"
  "50 50"
  "70 30"
  "90 10"
  "20 80"
  "40 60"
  "60 40"
  "80 20"
)
# Array of workload names
workload_array=('cudaTensorCoreGemm' 'fastWalshTransform' 'reductionMultiBlockCG' 'transpose' 'BlackScholes' 'sortingNetworks', 'wav2vec2-base-960h' 'bert-base-cased' 'mobilenet' 'mobilenet_v2_1.0_224' 'vit-base-patch16-224' 'vit_h_14' 'whisper-large-v2' 'vit-base-patch16-224' 'albert-base-v2' 'gpt2-xl')

# Mapping of workload names to prefixes
declare -A workloadprefix
workloadprefix['cudaTensorCoreGemm']=''
workloadprefix['fastWalshTransform']=''
workloadprefix['reductionMultiBlockCG']=''
workloadprefix['transpose']=''
workloadprefix['sortingNetworks']=''
workloadprefix['BlackScholes']=''
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
task_model['sortingNetworks']='cuda_samples'
task_model['BlackScholes']='cuda_samples'
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
