#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/paths.sh"
cd "$REPO_ROOT"

#
# Mimic cross validations, but on splitting 10 folds.
rm_100partitions="True"
n_combinations="2"
# Dataset path
datapath="$REPO_ROOT/data/model_datasets/03112025_DL0207_0307_nonDL0311_nodvfs/mergecudaDL_nodvfs_throughput_total_labels_comb2.csv"
# Root output of partitioned dataset
root_dataset_output_dir="$REPO_ROOT/data/model_datasets/03112025_DL0207_0307_nonDL0311_nodvfs/seen_partition/10foldvalidation"
output_result_dir="output/test_regularization_extratrees2"
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
#FOR TESTING PURPOSES
TEST_FOLDS_PER_NUM=(
  "1:3"
)
# Define regularization parameters to test
#MAX_DEPTH_VALUES=(5 10 15 20 "None")
MAX_DEPTH_VALUES=("None")
#MIN_SAMPLES_SPLIT_VALUES=(2 5 10 20)
MIN_SAMPLES_SPLIT_VALUES=(2)
MAX_FEATURES_VALUES=("sqrt" "None")
N_ESTIMATORS_VALUES=(100)
BOOTSTRAP_VALUES=("True" "False")
MIN_SAMPLES_LEAF_VALUES=(50 100 200)
# Define NUM_TEST_FOLD options
#NUM_TEST_FOLD=(1 2 3 4 5 6 7 8 9)
NUM_TEST_FOLD=(1)



n_occur=0

# Define regularization parameters to test
#MAX_DEPTH_VALUES=(5 10 15 20 "None")
MAX_DEPTH_VALUES=("None")
#MIN_SAMPLES_SPLIT_VALUES=(2 5 10 20)
MIN_SAMPLES_SPLIT_VALUES=(2)
MAX_FEATURES_VALUES=("sqrt" "None")
N_ESTIMATORS_VALUES=(50 100 200)
BOOTSTRAP_VALUES=("True" "False")
MIN_SAMPLES_LEAF_VALUES=(50 100 200)
# Define NUM_TEST_FOLD options
#NUM_TEST_FOLD=(1 2 3 4 5 6 7 8 9)
NUM_TEST_FOLD=(1)

n_occur=0

# Iterate over max_depth values
for max_depth in "${MAX_DEPTH_VALUES[@]}"; do
    # Convert "None" to -1 for max_depth
    if [ "$max_depth" == "None" ]; then
        max_depth_arg="-1"
    else
        max_depth_arg="$max_depth"
    fi
    # Iterate over min_samples_split values
    for min_samples_split in "${MIN_SAMPLES_SPLIT_VALUES[@]}"; do
        # Iterate over min_samples_leaf values
        for min_samples_leaf in "${MIN_SAMPLES_LEAF_VALUES[@]}"; do
            # Iterate over max_features values
            for max_features in "${MAX_FEATURES_VALUES[@]}"; do
                # Iterate over n_estimators values
                for n_estimators in "${N_ESTIMATORS_VALUES[@]}"; do
                    # Iterate over bootstrap values
                    for bootstrap in "${BOOTSTRAP_VALUES[@]}"; do
                        # Set the output directory for this combination
                        current_output_dir="$output_result_dir/max_depth_${max_depth}/min_samples_split_${min_samples_split}/min_samples_leaf_${min_samples_leaf}/max_features_${max_features}/n_estimators_${n_estimators}/bootstrap_${bootstrap}"
                        echo "Creating output directory: $current_output_dir"
                        mkdir -p "$current_output_dir"

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

                            # Iterate over the test fold combinations for this num_test_fold
                            for test_folds in "${matching_folds[@]}"; do
                                echo "Processing num_test_fold=$num_test_fold, test_folds=$test_folds, max_depth=$max_depth, min_samples_split=$min_samples_split, min_samples_leaf=$min_samples_leaf, max_features=$max_features, n_estimators=$n_estimators, bootstrap=$bootstrap"
                                # Run the training script with the current regularization parameters
                                bash "$SCRIPT_DIR/train_splits_mpsthreads_crossvalidate.sh" extratrees 0 "$current_output_dir" train predict_all_throughput "$datapath" "$root_dataset_output_dir" "$rm_100partitions" "$n_combinations" throughput "$num_test_fold" "$test_folds" "$max_depth_arg" "$min_samples_split" "$min_samples_leaf" "$max_features" "$n_estimators" "$bootstrap"
                            done
                        done
                    done
                done
            done
        done
    done
done