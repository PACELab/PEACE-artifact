#!/bin/bash

# Check if the IP address is provided as an argument
if [ $# -ne 3 ]; then
    echo "Usage: $0 [ipaddress]"
    exit 1
fi

filename="$1"

# Set the IP address
ipaddress="$2"

logdir="$3"
mkdir -p $logdir
# Log file path
logfile="$logdir/request_$ipaddress.log"
if [ -f "$logfile" ]; then
    # Remove the file
    rm "$logfile"
    echo "File '$logfile' removed."
else
    touch "$logfile"
    echo "File '$logfile' does not exist."
fi
# Initialize request number
request_num=0
#write time stamp
start_timestamp=$(date -u "+%Y-%m-%d %H:%M:%S")
echo "$start_timestamp: Starting workload script..." >> "$logfile"

# Define trap to handle SIGTERM signal
sigterm_handler() {
    end_timestamp=$(date -u "+%Y-%m-%d %H:%M:%S")
    duration=$(($(date -d "$end_timestamp" "+%s") - $(date -d "$start_timestamp" "+%s")))
    echo "$end_timestamp: Terminating workload script... Duration: $duration seconds" >> "$logfile"
    exit 0
}
trap 'sigterm_handler' SIGTERM SIGINT


# Loop indefinitely
while true; do
    # Check if request_num is equal to 120
    if [ "$request_num" -eq 120 ]; then
        echo "Reached request number 120. Exiting loop."
        break
    fi
    # Increment request number
    ((request_num++))

    timestamp=$(date -u "+%Y-%m-%d %H:%M:%S")
    echo "$timestamp: request $request_num" >> "$logfile"
    # Execute the Python script with the provided IP address and append the output to the log file
    python "$filename" "$ipaddress" >> "$logfile"
    
    # Sleep for 10 seconds
    sleep 1
done
timestamp=$(date -u "+%Y-%m-%d %H:%M:%S")
echo "$timestamp: Finished workload script..." >> "$logfile"
