#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/paths.sh"
cd "$REPO_ROOT"

#
# Mimic cross validations, but on splitting 10 folds.
rm_100partitions="True"
n_combinations="2"
rdseed="10"
# Dataset path
#freq1530
datapath="$REPO_ROOT/data/model_datasets/09152025DL_0311nonDL_FREQ1530_mergecudaDL_nodvfs_remerge/merged_total_labels_comb2.csv"
#freq900
#datapath="$REPO_ROOT/data/model_datasets/05052025_FREQ900_mergecudaDL_nodvfs_remerge/merged_total_labels_comb2.csv"
#freq300
#datapath="$REPO_ROOT/data/model_datasets/05052025_FREQ300_mergecudaDL_nodvfs_remerge/merged_total_labels_comb2.csv"
#
########################################################
#[SOCC'26 Rebuttal ONLY - triton kernel data]
#tritonkernel still use original 1530 metrics. not triton baseline metrics.
#datapath="$REPO_ROOT/data/model_datasets/04132026_triton_DL_socc26_basemetric_nontriton/triton_DL_throughput_total_labels_comb2.csv"
#triton kernle with triton baseline metrics
#datapath="$REPO_ROOT/data/model_datasets/04132026_triton_DL_socc26_basemetric_TRITON_1/triton_DL_throughput_total_labels_comb2.csv"
########################################################
# Root output of partitioned dataset
#root_dataset_output_dir="$REPO_ROOT/data/model_datasets/03112025_DL0207_0307_nonDL0311_nodvfs/seen_partition/10foldvalidation"
#root_dataset_output_dir="$REPO_ROOT/data/model_datasets/05052025_FREQ900_nodvfs/seen_partition/10foldvalidation"
root_dataset_output_dir="$(dirname "$datapath")/seen_partition/10foldvalidation"

#
#freq1530
output_result_dir="$REPO_ROOT/artifacts/predictions/09152025DL_0311nonDL_FREQ1530_mergecudaDL_nodvfs_remerge"
#freq900
#output_result_dir="$REPO_ROOT/artifacts/predictions/05052025_FREQ900_mergecudaDL_nodvfs_remerge"
#freq300
#output_result_dir="$REPO_ROOT/artifacts/predictions/05052025_FREQ300_mergecudaDL_nodvfs_remerge"

########################################################
#[SOCC'26 Rebuttal ONLY - triton kernel data]
#tritonkernel still use original 1530 metrics. not triton baseline metrics.
#output_result_dir="$REPO_ROOT/artifacts/predictions/04132026_triton_FREQ1530_DL_nodvfs_BASELINE_NOTRITON"
#triton kernle with triton baseline metrics
#output_result_dir="$REPO_ROOT/artifacts/predictions/04132026_triton_FREQ1530_DL_nodvfs_basemetric_TRITON_1"
########################################################
# Define TEST_FOLDS_PER_NUM for 10-fold cross-validation
# Each key (NUM_TEST_FOLD) has 10 different test fold combinations
TEST_FOLDS_PER_NUM=(
  "1:1" "1:2" "1:3" "1:4" "1:5" "1:6" "1:7" "1:8" "1:9" "1:10"
  "2:1-2" "2:2-3" "2:3-4" "2:4-5" "2:5-6" "2:6-7" "2:7-8" "2:8-9" "2:9-10" "2:10-1"
  "3:1-3" "3:2-4" "3:3-5" "3:4-6" "3:5-7" "3:6-8" "3:7-9" "3:8-10" "3:9-1" "3:10-2"
  "4:1-4" "4:2-5" "4:3-6" "4:4-7" "4:5-8" "4:6-9" "4:7-10" "4:8-1" "4:9-2" "4:10-3"
  "5:1-5" "5:2-6" "5:3-7" "5:4-8" "5:5-9" "5:6-10" "5:7-1" "5:8-2" "5:9-3" "5:10-4"
  "6:1-6" "6:2-7" "6:3-8" "6:4-9" "6:5-10" "6:6-1" "6:7-2" "6:8-3" "6:9-4" "6:10-5"
  "7:1-7" "7:2-8" "7:3-9" "7:4-10" "7:5-1" "7:6-2" "7:7-3" "7:8-4" "7:9-5" "7:10-6"
  "8:1-8" "8:2-9" "8:3-10" "8:4-1" "8:5-2" "8:6-3" "8:7-4" "8:8-5" "8:9-6" "8:10-7"
  "9:1-9" "9:2-10" "9:3-1" "9:4-2" "9:5-3" "9:6-4" "9:7-5" "9:8-6" "9:9-7" "9:10-8"
)
#use this to define folds
#
total_workload_path="$REPO_ROOT/data/model_datasets/09152025DL_0311nonDL_FREQ1530_mergecudaDL_nodvfs_remerge/merged_total_labels_comb2.csv"
fold_workloads_output_path="$REPO_ROOT/data/model_datasets/comb$n_combinations/10foldvalidation"
########################################################
#[SOCC'26 Rebuttal ONLY - triton kernel data]
#tritonkernel still use original 1530 metrics. not triton baseline metrics.
#total_workload_path="$REPO_ROOT/data/model_datasets/04132026_triton_DL_socc26_basemetric_TRITON_1/triton_DL_throughput_total_labels_comb2.csv"
#triton kernle with triton baseline metrics
#total_workload_path="$REPO_ROOT/data/model_datasets/04132026_triton_DL_socc26_basemetric_TRITON_1/triton_DL_throughput_total_labels_comb2.csv"
#fold_workloads_output_path="$REPO_ROOT/data/model_datasets/comb$n_combinations/triton_socc26/10foldvalidation"

########################################################
#create workload_folds before actual runs
python create_folds_by_workload.py --workload_list_path $total_workload_path --fold_workloads_output_path $fold_workloads_output_path --n_combination $n_combinations --rdseed $rdseed

#TEST_FOLDS_PER_NUM=(
#    "7:1-7" "7:2-8" "7:3-9" "7:4-10" "7:5-1" "7:6-2" "7:7-3" "7:8-4" "7:9-5" "7:10-6"
#)



##TESTING Regularization
max_depth="None"
min_samples_split=2

# Define NUM_TEST_FOLD options
NUM_TEST_FOLD=(1 2 3 4 5 6 7 8 9)
#NUM_TEST_FOLD=(7)

echo "starting test"
#tuning parameters - only listed default values
MAX_DEPTH_VALUES="None"
#MIN_SAMPLES_SPLIT_VALUES=(2 5 10 20)
min_samples_split=2
max_features="None"
n_estimators=100
bootstrap="False"
min_samples_leaf=1

n_occur=0

# Iterate over each NUM_TEST_FOLD
for num_test_fold in "${NUM_TEST_FOLD[@]}"; do
    # Filter TEST_FOLDS_PER_NUM for the current num_test_fold
    matching_folds=()
    for entry in "${TEST_FOLDS_PER_NUM[@]}"; do
        key="${entry%%:*}"
        if [ "$key" == "$num_test_fold" ]; then
            value="${entry#*:}"
            matching_folds+=("$value")
        fi
    done
    if [ "$max_depth" == "None" ]; then
        max_depth_arg="-1"
    else
        max_depth_arg="$max_depth"
    fi
    # Print the matching folds for debugging
    echo "Matching folds for num_test_fold=$num_test_fold: ${matching_folds[@]}"
    # Iterate over the 10 test fold combinations for this num_test_fold
    for test_folds in "${matching_folds[@]}"; do
        #echo "Processing num_test_fold=$num_test_fold, test_folds=$test_folds"
        echo "Processing num_test_fold=$num_test_fold, test_folds=$test_folds, max_depth=$max_depth, min_samples_split=$min_samples_split, min_samples_leaf=$min_samples_leaf, max_features=$max_features, n_estimators=$n_estimators, bootstrap=$bootstrap"

        # Pass test_folds to the script
        #bash "$SCRIPT_DIR/train_splits_mpsthreads_crossvalidate.sh" linear 0 "$output_result_dir" train predict_all_throughput "$datapath" "$root_dataset_output_dir" "$rm_100partitions" "$n_combinations" power "$num_test_fold" "$test_folds" "$max_depth_arg" "$min_samples_split" "$min_samples_leaf" "$max_features" "$n_estimators" "$bootstrap" "$fold_workloads_output_path"
        #bash "$SCRIPT_DIR/train_splits_mpsthreads_crossvalidate.sh" extratrees 0 "$output_result_dir" train predict_all_throughput "$datapath" "$root_dataset_output_dir" "$rm_100partitions" "$n_combinations" throughput "$num_test_fold" "$test_folds" "$max_depth_arg" "$min_samples_split" "$min_samples_leaf" "$max_features" "$n_estimators" "$bootstrap" "$fold_workloads_output_path"
        #bash "$SCRIPT_DIR/train_splits_mpsthreads_crossvalidate.sh" RF 0 "$output_result_dir" train predict_all_throughput "$datapath" "$root_dataset_output_dir" "$rm_100partitions" "$n_combinations" power "$num_test_fold" "$test_folds" "$max_depth_arg" "$min_samples_split" "$min_samples_leaf" "$max_features" "$n_estimators" "$bootstrap" "$fold_workloads_output_path"
        #bash "$SCRIPT_DIR/train_splits_mpsthreads_crossvalidate.sh" AutoML 0 "$output_result_dir" fef predict_all_throughput "$datapath" "$root_dataset_output_dir" "$rm_100partitions" "$n_combinations" throughput "$num_test_fold" "$test_folds" "$max_depth_arg" "$min_samples_split" "$min_samples_leaf" "$max_features" "$n_estimators" "$bootstrap" "$fold_workloads_output_path"

        bash "$SCRIPT_DIR/train_splits_mpsthreads_crossvalidate.sh" extratrees 0 "$output_result_dir" train predict_all_throughput "$datapath" "$root_dataset_output_dir" "$rm_100partitions" "$n_combinations" throughput "$num_test_fold" "$test_folds" "$max_depth_arg" "$min_samples_split" "$min_samples_leaf" "$max_features" "$n_estimators" "$bootstrap" "$fold_workloads_output_path"
        bash "$SCRIPT_DIR/train_splits_mpsthreads_crossvalidate.sh" extratrees 0 "$output_result_dir" train predict_all_throughput "$datapath" "$root_dataset_output_dir" "$rm_100partitions" "$n_combinations" power "$num_test_fold" "$test_folds" "$max_depth_arg" "$min_samples_split" "$min_samples_leaf" "$max_features" "$n_estimators" "$bootstrap" "$fold_workloads_output_path"
        bash "$SCRIPT_DIR/train_splits_mpsthreads_crossvalidate.sh" RF 0 "$output_result_dir" train predict_all_throughput "$datapath" "$root_dataset_output_dir" "$rm_100partitions" "$n_combinations" throughput "$num_test_fold" "$test_folds" "$max_depth_arg" "$min_samples_split" "$min_samples_leaf" "$max_features" "$n_estimators" "$bootstrap" "$fold_workloads_output_path"
        bash "$SCRIPT_DIR/train_splits_mpsthreads_crossvalidate.sh" RF 0 "$output_result_dir" train predict_all_throughput "$datapath" "$root_dataset_output_dir" "$rm_100partitions" "$n_combinations" power "$num_test_fold" "$test_folds" "$max_depth_arg" "$min_samples_split" "$min_samples_leaf" "$max_features" "$n_estimators" "$bootstrap" "$fold_workloads_output_path"
        bash "$SCRIPT_DIR/train_splits_mpsthreads_crossvalidate.sh" AutoML 0 "$output_result_dir" train predict_all_throughput "$datapath" "$root_dataset_output_dir" "$rm_100partitions" "$n_combinations" throughput "$num_test_fold" "$test_folds" "$max_depth_arg" "$min_samples_split" "$min_samples_leaf" "$max_features" "$n_estimators" "$bootstrap" "$fold_workloads_output_path"
        bash "$SCRIPT_DIR/train_splits_mpsthreads_crossvalidate.sh" AutoML 0 "$output_result_dir" train predict_all_throughput "$datapath" "$root_dataset_output_dir" "$rm_100partitions" "$n_combinations" power "$num_test_fold" "$test_folds" "$max_depth_arg" "$min_samples_split" "$min_samples_leaf" "$max_features" "$n_estimators" "$bootstrap" "$fold_workloads_output_path"

    done
done
