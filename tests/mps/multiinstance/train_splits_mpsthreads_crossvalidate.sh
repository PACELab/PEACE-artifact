#!/bin/bash

modeltype=$1
correlation=$2
outdir=$3
istrain=$4
ispredict=$5
datapath=$6
root_dataset_output_dir=$7
rm_100partitions=$8
n_combination=${9}
throughput_or_power_task=${10}
num_test_fold=${11}  # New argument: number of test folds
test_folds=${12}     # New argument: specific test folds (e.g., "1-3")
# Regularization parameters
max_depth=${13}      # New argument: max_depth for RF/ET
min_samples_split=${14}  # New argument: min_samples_split for RF/ET
min_samples_leaf=${15}  # New argument: min_samples_leaf for RF/ET
max_features=${16}   # New argument: max_features for RF/ET
n_estimators=${17}   # New argument: n_estimators for RF/ET
bootstrap=${18}      # New argument: bootstrap for RF/ET
fold_workloads_base=${19}  # Optional: directory with precomputed workload folds
seeds=(10)  # Single seed for consistency; adjust as needed

# Print arguments
echo "modeltype: $modeltype"
echo "correlation: $correlation"
echo "n_occur: $n_occur"
echo "outdir: $outdir"
echo "istrain: $istrain"
echo "ispredict: $ispredict"
echo "datapath: $datapath"
echo "root_dataset_output_dir: $root_dataset_output_dir"
echo "rm_100partitions: $rm_100partitions"
echo "n_combination: $n_combination"
echo "throughput_or_power_task: $throughput_or_power_task"
echo "num_test_fold: $num_test_fold"
echo "test_folds: $test_folds"
echo "max_depth: $max_depth"
echo "min_samples_split: $min_samples_split"
echo "min_samples_leaf: $min_samples_leaf"
echo "max_features: $max_features"
echo "n_estimators: $n_estimators"
echo "bootstrap: $bootstrap"
echo "fold_workloads_base: $fold_workloads_base"

# Check if model and correlation are provided
if [ -z "$correlation" ]; then
    echo "Correlation is not provided"
    exit 1
fi
if [ -z "$modeltype" ]; then
    echo "Model is not provided"
    exit 1
fi

if [ "$throughput_or_power_task" == "throughput" ]; then
    label_policy="separate_throughputpower_regression"
else
    label_policy="power_regression"
fi

EXCOL=""
root_dataset_dir="${root_dataset_output_dir}/trainratio_${n_occur}"
if [ "$modeltype" == "hotcloud" ]; then
    echo "hotcloud model not implemented"
    exit 1
    full_data_path="/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/baselines/hotcloud/hotcloud_combined_noexclu_throughput_targetMPS100.csv"
else
    full_data_path=$datapath
fi

test_file="testing_set.csv"
train_file="training_set.csv"
targetMPS=100

# Parse test_folds range (e.g., "7-5")
IFS='-' read -r -a test_fold_range <<< "$test_folds"
start_fold=${test_fold_range[0]}
end_fold=${test_fold_range[1]}

# Define all possible folds
all_folds=(1 2 3 4 5 6 7 8 9 10)

# Function to generate range of folds, handling wrap-around
generate_fold_range() {
    local start=$1
    local end=$2
    local max_fold=10
    local folds=()

    if [ "$start" -le "$end" ]; then
        # Normal range (e.g., 1-3)
        for ((i=start; i<=end; i++)); do
            folds+=("$i")
        done
    else
        # Wrap-around range (e.g., 7-5: 7 to 10, then 1 to 5)
        for ((i=start; i<=max_fold; i++)); do
            folds+=("$i")
        done
        for ((i=1; i<=end; i++)); do
            folds+=("$i")
        done
    fi
    echo "${folds[@]}"
}

# Parse test folds based on num_test_fold
if [ "$num_test_fold" -eq 1 ]; then
    # Single fold case: directly use test_folds as the fold number
    test_fold_array=("$test_folds")
    echo "Test folds array: ${test_fold_array[@]}"
else
    # Range case: parse test_folds as "x-y"
    IFS='-' read -r -a test_fold_range <<< "$test_folds"
    if [ ${#test_fold_range[@]} -ne 2 ]; then
        echo "Error: test_folds must be in 'x-y' format (e.g., '1-3') when num_test_fold > 1"
        exit 1
    fi
    start_fold=${test_fold_range[0]}
    end_fold=${test_fold_range[1]}
    test_fold_array=($(generate_fold_range "$start_fold" "$end_fold"))
    echo "Test folds array: ${test_fold_array[@]}"
fi

# Validate test_fold_array length matches num_test_fold
test_fold_count=${#test_fold_array[@]}
if [ "$test_fold_count" -ne "$num_test_fold" ]; then
    echo "Error: Number of test folds ($test_fold_count) does not match num_test_fold ($num_test_fold)"
    exit 1
fi

# Calculate train folds (complement of test folds in 1-10)
train_fold_array=()
for fold in "${all_folds[@]}"; do
    skip=false
    for test_fold in "${test_fold_array[@]}"; do
        if [ "$fold" -eq "$test_fold" ]; then
            skip=true
            break
        fi
    done
    if [ "$skip" == "false" ]; then
        train_fold_array+=("$fold")
    fi
done
echo "Train folds array: ${train_fold_array[@]}"

# Convert fold arrays to comma-separated strings for splitData.py
test_folds_str=$(IFS=,; echo "${test_fold_array[*]}")
train_folds_str=$(IFS=,; echo "${train_fold_array[*]}")

# Optionally, verify the number of test folds matches num_test_fold
test_fold_count=${#test_fold_array[@]}
if [ "$test_fold_count" -ne "$num_test_fold" ]; then
    echo "Warning: Number of test folds ($test_fold_count) does not match num_test_fold ($num_test_fold)"
fi

echo "generate random splits with 10-fold cross-validation for test folds: $test_folds_str"


if [ "$istrain" == "train" ]; then
    for seed in "${seeds[@]}"; do
        echo "seed $seed"
        dataset_dir="$root_dataset_dir/rand${seed}/fold_${num_test_fold}_test_${test_folds}"
        outdir_rand="$outdir/seen_partition/crossvalid/${throughput_or_power_task}/trainratio_${n_occur}/rand${seed}/fold_${num_test_fold}_test_${test_folds}/$modeltype"
        mkdir -p "$dataset_dir"
        mkdir -p "$outdir_rand"
        #check test_folds_str value
        echo "test_folds_str: $test_folds_str"
        echo "train_folds_str: $train_folds_str"
        if [ ! -f "$dataset_dir/$test_file" ] || [ ! -f "$dataset_dir/$train_file" ]; then
            echo "generate data for $dataset_dir..."
            split_args=(splitData.py -d "$full_data_path" \
                --train_file "$dataset_dir/$train_file" \
                --test_file "$dataset_dir/$test_file" \
                -rd $seed \
                --test_folds "$test_folds_str" \
                --train_folds "$train_folds_str" \
                --rm_100partitions "$rm_100partitions" \
                --split_workload_with_all_threads \
                --cross_validation \
                --n_combination "$n_combination")
            if [ -n "$fold_workloads_base" ]; then
                split_args+=(--readbyworkloadlist "$fold_workloads_base")
            fi
            python "${split_args[@]}"

            echo "data saved in $dataset_dir"
        fi
        
        echo "train model with $dataset_dir/$train_file..."
        python ../../../main.py --test_file "$dataset_dir/$test_file" \
            --train_file "$dataset_dir/$train_file" \
            -t $targetMPS -rd $seed --train -corr $correlation \
            --output_dir "$outdir_rand" --modeltype $modeltype -comb $n_combination \
            --label_policy $label_policy --debug \
            --max_depth "$max_depth" --min_samples_split "$min_samples_split" \
            --max_features "$max_features" --n_estimators "$n_estimators" \
            --bootstrap "$bootstrap" --min_samples_leaf "$min_samples_leaf"
    done
fi

if [ "$ispredict" == "predict_all_throughput" ]; then
    echo "start predicting throughput for different workloads"
    for seed in "${seeds[@]}"; do
        echo "seed $seed"
        dataset_dir="$root_dataset_dir/rand${seed}/fold_${num_test_fold}_test_${test_folds}"
        model_dir="$outdir/seen_partition/crossvalid/${throughput_or_power_task}/trainratio_${n_occur}/rand${seed}/fold_${num_test_fold}_test_${test_folds}/$modeltype"
        outdir_rand="$outdir/seen_partition/crossvalid/${throughput_or_power_task}/trainratio_${n_occur}/rand${seed}/fold_${num_test_fold}_test_${test_folds}/$modeltype"
        outdir_pred="$outdir_rand/predictAllacc"
        mkdir -p "$outdir_pred"
        
        echo "search for model in $model_dir..."
        if [ "$modeltype" != "AutoML" ]; then
            model_file=$(find "$model_dir" -name "*$modeltype*.pkl")
        else
            model_file=$(find "$model_dir" -type f -name "*AutoML*" ! -name "*.txt")
        fi
        
        if [ -z "$model_file" ]; then
            echo "Model $modeltype not found in $model_dir"
            exit 1
        fi
        if [ $(echo $model_file | wc -l) -gt 1 ]; then
            echo "Multiple models found in $model_dir"
            exit 1
        fi

        echo "using model $model_file to predict..."
        python ../../../main.py --test_file "$dataset_dir/$test_file" \
            --train_file "$dataset_dir/$train_file" \
            --train_postprocess_file "$dataset_dir/X_train_postprocess_$label_policy.csv" \
            --HPworkload "" --targetMPS 100 \
            --model "$model_file" -corr $correlation \
            -rd $seed --output_dir "$outdir_pred" \
            --rulebase --modeltype $modeltype \
            --predAllacc -comb $n_combination --label_policy $label_policy --debug
        predict_status=$?

        echo testfile=$dataset_dir/$test_file > "$outdir_pred/config.txt"
        echo trainfile=$dataset_dir/$train_file >> "$outdir_pred/config.txt"
        echo model=$model_file >> "$outdir_pred/config.txt"
        echo correlation=$correlation >> "$outdir_pred/config.txt"
        echo train_n_occur=$n_occur >> "$outdir_pred/config.txt"
        echo test_folds=$test_folds >> "$outdir_pred/config.txt"

        if [ $predict_status -ne 0 ]; then
            echo "An error occurred while running main.py. Exiting..."
            echo "current output directory is $outdir_pred"
            exit 1
        fi

        if command -v zip >/dev/null 2>&1; then
            archive="${model_file}.zip"
            echo "Compressing model to save disk space: $model_file -> $archive"
            rm -f "$archive" 2>/dev/null || true
            if zip -jq "$archive" "$model_file" >/dev/null; then
                rm -f "$model_file"
                echo "Model archived and original removed."
            else
                echo "Warning: failed to archive $model_file; original retained."
            fi
        else
            echo "Warning: zip command not found; skipping model compression."
        fi
    done
fi
