import requests
import time
from collections import defaultdict
# Function to parse the response and extract metric data
def parse_metrics(response_text):
    metrics = {}
    for line in response_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        metric_name = parts[0]
        metric_value = float(parts[1])
        metrics[metric_name] = metric_value
    return metrics

# URL of the metrics endpoint
url = 'http://localhost:8000/metrics'

# Initialize dictionary of lists to store metric data for each second
metrics_data = defaultdict(list)

try:
    # Infinite loop to fetch metrics every one second
    while True:
        # Fetch metrics from the endpoint
        response = requests.get(url)
        print(response.status_code)
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

        # Wait for one second before fetching metrics again
        time.sleep(1)

except KeyboardInterrupt:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}End recordings. metrics_data: {metrics_data}")
    for k,v in metrics_data.items():
        if "throughput"in k:
            print(f"mean of {k}: {sum(v)/len(v)}")
    # Handle Ctrl+C gracefully
    print("\nExiting...")
