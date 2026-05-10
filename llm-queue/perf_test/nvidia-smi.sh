#!/bin/bash

# CSV file to store GPU information
csv_dir="csv"
csv_file="gpu_info.csv"
# File to store the output of the Python script
workload_output_file="python_output_datawrangle.txt"
outputdir="1datawrangle"
mkdir -p logs/${outputdir}
mkdir -p csv
# Python script command
python_script_command="python /home/bing/llm-queue/build/FlexGen-main/flexgen/apps/completion.py --model facebook/opt-6.7b --percent 100 0 100 0 100 0"
data_wrangle_script_command="bash /home/bing/llm-queue/build/FlexGen-main/flexgen/apps/data_wrangle/test_batch_query_case1-3_opt6.7b.sh 100 0 100 0 100 0 ${outputdir}"
# Add header to the CSV file if it doesn't exist
if [ ! -e "$csv_file" ]; then
    echo "timestamp,power.draw,utilization.gpu,memory.used,memory.total,fan.speed,temperature.gpu" > "${csv_dir}/${csv_file}"
fi

#start inference code
# Cleanup function to be executed on script termination
cleanup() {
    # Kill the Python script process
    pkill -f "$python_script_command"
    pkill -f "$data_wrangle_script_command"
    exit
}
# Trap Ctrl+C to execute cleanup function
trap cleanup INT

# Get current timestamp
timestamp=$(date "+%Y-%m-%d %H:%M:%S")
echo "$timestamp: Starting workload script..."

# Run the Python script
#eval "$python_script_command" > "$workload_output_file" 2>&1 &
eval "$data_wrangle_script_command" > "logs/${outputdir}/${workload_output_file}" 2>&1 &
# Infinite loop to record GPU information and run the Python script
while true; do
    # Get current timestamp
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    

    
    # Run nvidia-smi, prepend timestamp, and append the result to the CSV file
    echo -n "$timestamp," >> "${csv_dir}/${csv_file}"
    nvidia-smi --format=csv --query-gpu=power.draw,utilization.gpu,memory.used,memory.total,fan.speed,temperature.gpu | tail -n 1 >> "${csv_dir}/${csv_file}"
    
    # Sleep for a desired interval (e.g., 1 second)
    sleep 1
done
