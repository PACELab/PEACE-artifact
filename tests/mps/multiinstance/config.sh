# config.sh

# Number of workloads per combination
N_COMBINATIONS=2

# Path to the CSV file containing workload combinations
CSV_FILE='/home/cc/mlProfiler/tests/mps/multiinstance/1009_addsmallmodels_kernel_labels_comb2_batches2.csv'
#CSV_FILE="/home/cc/mlProfiler/tests/mps/multiinstance/1010_batch2_kernel_labels_comb1_batches2.csv"

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

# Mapping of workload names to tasks
declare -A task_model
task_model['wav2vec2-base-960h']='speech-recognition'
task_model['bert-base-cased']='recommend'
task_model['mobilenet']='imgclassification'
task_model['mobilenet_v2_1.0_224']='imgclassification'
task_model['vit-base-patch16-224']='imgclassification'
task_model['vit_h_14']='imgclassification'
task_model['whisper-large-v2']='speech-recognition'
task_model['albert-base-v2']='recommend'
task_model['gpt2-xl']='autoregressive'
