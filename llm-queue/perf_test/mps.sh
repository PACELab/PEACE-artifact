#!/bin/bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
#echo get_active_thread_percentage 14439 | sudo nvidia-cuda-mps-control
#echo set_active_thread_percentage 14439 20 | sudo nvidia-cuda-mps-control
#echo get_active_thread_percentage 14439 | sudo nvidia-cuda-mps-control
# Function to set active thread percentage
set_active_thread_percentage() {
    local gpu_pid="$1"
    local percentage="$2"

    if [ -z "$percentage" ]; then
        echo "Error: No percentage value provided."
        exit 1
    fi

    echo "set_active_thread_percentage $gpu_pid $percentage" | sudo nvidia-cuda-mps-control
}


# CSV file to store GPU information
csv_dir="csv"
csv_file="gpu_info.csv"
# File to store the output of the Python script
chatbot_output_file="python_output_chatbot.txt"
data_wrangle_output_file="python_output_datawrangle.txt"
outputdir="outputs/mps_chatbot_1datawrangle/90_10"
mkdir -p logs/${outputdir}
mkdir -p csv
mkdir -p ${outputdir}/logs
mkdir -p ${outputdir}/${csv_dir}

# Python script command
chatbot_script_command="python /home/bing/mlProfiler/llm-queue/build/FlexGen-main/flexgen/apps/completion.py --model facebook/opt-6.7b --percent 100 0 100 0 100 0"
data_wrangle_script_command1="bash /home/bing/mlProfiler/llm-queue/build/FlexGen-main/flexgen/apps/data_wrangle/test_batch_query_case1-3_opt6.7b.sh 0 100 0 100 0 100 ${outputdir}/job1"
#data_wrangle_script_command2="bash /home/bing/llm-queue/build/FlexGen-main/flexgen/apps/data_wrangle/test_batch_query_case1-3_opt6.7b.sh 0 100 0 100 0 100 ${outputdir}/job2"
#data_wrangle_script_command3="bash /home/bing/llm-queue/build/FlexGen-main/flexgen/apps/data_wrangle/test_batch_query_case1-3_opt6.7b.sh 0 100 0 100 0 100 ${outputdir}/job3"
# Add header to the CSV file if it doesn't exist
if [ ! -e "$csv_file" ]; then
    echo "timestamp,power.draw,utilization.gpu,memory.used,memory.total,fan.speed,temperature.gpu" > "${outputdir}/${csv_dir}/${csv_file}"
fi

#start inference code
# Cleanup function to be executed on script termination
cleanup() {
    # Kill the Python script process
    pkill -f "$chatbot_script_command"
    #pkill -f "$data_wrangle_script_command1"
    exit
}
# Trap Ctrl+C to execute cleanup function
trap cleanup INT

# Get current timestamp
timestamp=$(date "+%Y-%m-%d %H:%M:%S")
echo "$timestamp: Starting workload script..."

# Run the Python script
#eval "$data_wrangle_script_command1" > "${outputdir}/logs/${data_wrangle_output_file}_job1" 2>&1 &
echo set_active_thread_percentage 14439 90 | sudo nvidia-cuda-mps-control
sleep 5
eval "$chatbot_script_command" > "${outputdir}/logs/${chatbot_output_file}" 2>&1 &
#eval "$data_wrangle_script_command2" > "${outputdir}/logs/${data_wrangle_output_file}_job2" 2>&1 &
#eval "$data_wrangle_script_command3" > "${outputdir}/logs/${data_wrangle_output_file}_job3" 2>&1 &

# Infinite loop to record GPU information and run the Python script
while true; do
    # Get current timestamp
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    

    
    # Run nvidia-smi, prepend timestamp, and append the result to the CSV file
    echo -n "$timestamp," >> "${outputdir}/${csv_dir}/${csv_file}"
    nvidia-smi --format=csv --query-gpu=power.draw,utilization.gpu,memory.used,memory.total,fan.speed,temperature.gpu | tail -n 1 >> "${outputdir}/${csv_dir}/${csv_file}"
    
    # Sleep for a desired interval (e.g., 1 second)
    sleep 1
done
