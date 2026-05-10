#!/usr/bin/env bash

# List of workloads
workloads=(
  "whisper-large-v2_batch2-inf"
  #"mobilenet_batch2-train"
  #"bert-base-cased_batch2-train"
)

# Example model type(s); you could vary them or set them externally
modeltype_throughput="AutoML"
modeltype_power="AutoML"

# Loop over workloads
for workload in "${workloads[@]}"; do
  echo "Running for workload: ${workload}"

  # Define directories where we expect to find the throughput and power models
  throughput_model_dir="tests/mps/multiinstance/output/run02112025_comb2_batch2/unseen_partition/throughput/rand10/${modeltype_throughput}/${workload}"
  power_model_dir="tests/mps/multiinstance/output/run02112025_comb2_batch2/unseen_partition/power/rand10/${modeltype_power}/${workload}"

  # --- Find THROUGHPUT model ---
  echo "Searching for THROUGHPUT model in: ${throughput_model_dir}"
  if [ "${modeltype_throughput}" != "AutoML" ]; then
      throughput_model_file=$(find "${throughput_model_dir}" -name "*${modeltype_throughput}*.pkl")
  else
      # If "AutoML", find any file containing "AutoML" but exclude .txt logs
      throughput_model_file=$(find "${throughput_model_dir}" -type f -name "*AutoML*" ! -name "*.txt")
  fi

  # Check if throughput model was found
  if [ -z "${throughput_model_file}" ]; then
    echo "ERROR: Throughput model '${modeltype_throughput}' not found in ${throughput_model_dir}"
    exit 1
  fi
  # Ensure exactly one match
  if [ "$(echo "${throughput_model_file}" | wc -l)" -gt 1 ]; then
    echo "ERROR: Multiple throughput models found in ${throughput_model_dir}:"
    echo "${throughput_model_file}"
    exit 1
  fi
  echo "Found throughput model: ${throughput_model_file}"

  # --- Find POWER model ---
  echo "Searching for POWER model in: ${power_model_dir}"
  if [ "${modeltype_power}" != "AutoML" ]; then
      power_model_file=$(find "${power_model_dir}" -name "*${modeltype_power}*.pkl")
  else
      power_model_file=$(find "${power_model_dir}" -type f -name "*AutoML*" ! -name "*.txt")
  fi

  # Check if power model was found
  if [ -z "${power_model_file}" ]; then
    echo "ERROR: Power model '${modeltype_power}' not found in ${power_model_dir}"
    exit 1
  fi
  # Ensure exactly one match
  if [ "$(echo "${power_model_file}" | wc -l)" -gt 1 ]; then
    echo "ERROR: Multiple power models found in ${power_model_dir}:"
    echo "${power_model_file}"
    exit 1
  fi
  echo "Found power model: ${power_model_file}"

  # --- Run main.py with the discovered model paths ---
  python main.py \
    --train_file "tests/mps/multiinstance/dataset/02062025_batch2_all/02062025_batch2_all/unseen_partition/power/rand10/${modeltype_power}/${workload}/training_set.csv" \
    --test_file "tests/mps/multiinstance/dataset/02062025_batch2_all/02062025_batch2_all/unseen_partition/power/rand10/${modeltype_power}/${workload}/testing_set.csv" \
    -comb 2 \
    -t 100 \
    --output_dir "tests/mps/multiinstance/output/run02112025_comb2_batch2/unseen_partition/e2e/${modeltype_power}/${workload}" \
    --label_policy power_regression \
    -mt ${modeltype_power} \
    --throughput_model "${throughput_model_file}" \
    --power_model "${power_model_file}" \
    --end_to_end \
    --debug

  echo "Finished workload: ${workload}"
  echo "----------------------------------------"
done

