# config_rebuttal_customkernel.sh
# SoCC'26 Rebuttal: Custom Triton-kernel workloads for MPS sensitivity
# Reproduces Figure 1 motivation (albert-train + mobilenet-inf) with
# Triton-compiled custom GPU kernels to address reviewer concern on
# non-linear performance cliffs with custom kernels.

# Number of workloads per combination
N_COMBINATIONS=2
RUNS=1

# CSV file with workload pairs (Triton-kernel variants of Fig 1 workloads)
CSV_FILE="$REPO_ROOT/data/experiment_inputs/colocations/comb2/rebuttal_customkernel_comb2.csv"
#43/65 DL pairs to be run
#CSV_FILE="$REPO_ROOT/data/experiment_inputs/colocations/comb2/02062025_powercap100_DL_baseline_labels_comb2_batches2_triton.csv"

# Frequency scaling: highest first per reviewer request
# Uncomment additional frequencies for full sweep
VALID_FREQS=(1530)
#VALID_FREQS=(1530 1335 1147 952 300)

# Timeout for colocation
TIMEOUT=60

# Sleep times between jobs
SLEEP_TIMES=(0)

# MPS percentage combinations matching Figure 1 experimental setup
custom_combinations=(
  "100 100"
  #"10 90"
  #"20 80"
  #"30 70"
  #"40 60"
  #"50 50"
  #"60 40"
  #"70 30"
  #"80 20"
  #"90 10"
)

# Workload arrays – includes both original and Triton-kernel variants
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
# Original workloads (for reference/comparison runs)
workloadprefix['albert-base-v2']=''
workloadprefix['mobilenet']=''
workloadprefix['mobilenet_v2_1.0_224']='google/'
workloadprefix['resnet-50']=''
workloadprefix['wav2vec2-base-960h']='facebook/'
workloadprefix['whisper-large-v2']='openai/'
workloadprefix['vit_h_14']=''
workloadprefix['vit-base-patch16-224']='google/'

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
