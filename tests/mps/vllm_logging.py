import requests
import time
from collections import defaultdict
import numpy as np
import sys, json
import http.client



# URL of the metrics endpoint
url = 'http://localhost:8000/metrics'
log_dir = sys.argv[1]
# Initialize dictionary of lists to store metric data for each second
metrics_data = defaultdict(list)
# Function to parse the response and extract metric data
def parse_metrics(response_text):
    metrics = {}
    for line in response_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        #print(parts)
        if len(parts) < 2:
            continue
        metric_name = parts[0]
        metric_value = float(parts[1])
        metrics[metric_name] = metric_value
    return metrics


def handleException():
    metrics_data_copy = metrics_data.copy()
    #print(metrics_data_copy)
    for metric_name, metric_value in metrics_data.items():
        #print(np.mean(metric_value))
        if "vllm:avg_prompt_throughput_toks_per_s" in metric_name:
            print(f"Average prefill throughput in tokens/s: {np.mean(metric_value)}")
            metrics_data_copy["Average_prefill_throughput"] = np.mean(metric_value)
        if "vllm:avg_generation_throughput_toks_per_s" in metric_name:
            print(f"average generation throughput in tokens/s: {np.mean(metric_value)}")
            metrics_data_copy["Average_generation_throughput"] = np.mean(metric_value)
    print("\ndumping metrics data...")
    

    import csv
    # Write defaultdict to CSV file
    with open(f"{log_dir}/LSmetrics.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric Name', 'Metric Values'])
        for key, value in metrics_data_copy.items():
            writer.writerow([key, value])

    print("\nExiting...")
    exit(0)
    



import signal
# Define the SIGTERM handler
def sigterm_handler(signum, frame):
    print("Received SIGTERM. Handling exception...")
    handleException()

# Register the SIGTERM handler
signal.signal(signal.SIGTERM, sigterm_handler)
#signal.signal(signal.SIGTERM, handleException)

    # Infinite loop to fetch metrics every one second
while True:
    try:
    # Fetch metrics from the endpoint
        time.sleep(1)
        response = requests.get(url)
        #print(response.status_code)
        if response.status_code == 200:
            # Parse the response and extract metric data
            metrics = parse_metrics(response.text)

            # Append metric data to the dictionary of lists
            for metric_name, metric_value in metrics.items():
                metrics_data[metric_name].append(metric_value)

            # Log the metric data
            print(f"Metrics fetched at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value}")
            print("----------------------------------------")
        else:
            handleException()
    except Exception as e:
        handleException()
    except KeyboardInterrupt as e:
        handleException()

        
    


    

    
