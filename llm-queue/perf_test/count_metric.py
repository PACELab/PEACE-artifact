import os, sys
import json

def process_metrics_file(file_path):
    with open(file_path, 'r') as file:
        metrics_data = json.load(file)
        return metrics_data.get("total_time", 0.0)[0], metrics_data.get("output_throughput_avg", 0.0)

def calculate_metrics(directory):
    total_time_sum = 0.0
    output_throughput_avg_sum = 0.0
    num_occurrences = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("metrics.json"):
                num_occurrences += 1
                file_path = os.path.join(root, file)
                total_time, output_throughput_avg = process_metrics_file(file_path)
                #print(total_time, output_throughput_avg)
                total_time_sum += total_time
                output_throughput_avg_sum += output_throughput_avg
    print(num_occurrences)
    return total_time_sum , output_throughput_avg_sum / num_occurrences if num_occurrences > 0 else 0.0

if __name__ == "__main__":
    directory_name = sys.argv[1]
    total_time_sum, avg_output_throughput = calculate_metrics(directory_name)

    print(f"Average of sum of 'total_time': {total_time_sum}")
    print(f"Average of the sum of 'output_throughput_avg': {avg_output_throughput}")
