#!/bin/bash

# Directory containing the CSV files
input_dir="../../kernel_profiles/source/ncu"
output_dir="kernel_hotcloud"  # Replace with your actual output directory

# Ensure the output and temporary directories exist
mkdir -p "$output_dir"
mkdir -p tmp

# Find all CSV files in the input directory
for file in "$input_dir"/*.csv; do
  # Extract the filename without the path and extension
  echo "Processing $file"
  filename=$(basename "$file")
  job_type="${filename%.csv}"
  job_type="${job_type%_ncu}"
  job_type="${job_type%_steps}"
  # Execute the Python command
  python hotcloud_ncu_orion_cloud.py -l 200 --input_file "$file" --results_dir "$output_dir" --job_type "$job_type" > "tmp/${job_type}_ncu.log"
done
