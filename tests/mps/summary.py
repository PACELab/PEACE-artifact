import re
import sys
import numpy as np
from collections import defaultdict

def parse_train_log_file(log_file_path):
    # Regular expression to match lines containing epoch information
    epoch_time_pattern = r'Epoch (\d+), Loss: [\d.]+, Time: ([\d.]+) seconds'
    epoch_times = {}

    with open(log_file_path, 'r') as file:
        for line in file:
            match = re.match(epoch_time_pattern, line)
            if match:
                epoch_number = int(match.group(1))
                epoch_time = float(match.group(2))
                epoch_times[epoch_number] = epoch_time
        print(f"Average Training Time: {np.mean(list(epoch_times.values())):.2f} seconds")
    return epoch_times
throughput_dict = defaultdict(list)
LS_percents =  [10 * i for i in range(1, 10)]
for percent in LS_percents:
    throughput_dict[f"LS{percent}"] = []

# Example usage:
#list all subfile within the log dir as sys.argv[1]. If any subdir == LS10, LS20, LS30, LS40, LS50, LS60, LS70, LS80, LS90, then parse the average training time to the throughput_dict
#log_dir, task, model = sys.argv[1], sys.argv[2], sys.argv[3]
#epoch_times = parse_train_log_file(log_file_path)
#get total avrage time per epoch from epoch times



log_dir, task, model = sys.argv[1], sys.argv[2], sys.argv[3]
epoch_times = parse_train_log_file(log_file_path)
#get total avrage time per epoch from epoch times

print(f"Average Training Time: {np.mean(list(epoch_times.values())):.2f} seconds")
for epoch, time in epoch_times.items():
    print(f'Epoch {epoch}: Average time = {time} seconds')
