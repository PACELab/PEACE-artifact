import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
from analysis.parse_util import parse_inf_log_avgStep_file, parse_train_log_avgStep_file

# Path to the folder containing the CSV files
csv_folder = "/home/cc/mlProfiler/tests/mps/ccv100_logs/comb2_stage2power_batch2/bert-base-cased_batch2-train/whisper-large-v2_batch2-inference"
w1 = "bert-base-cased_batch2-train"
w2 = "whisper-large-v2_batch2-inference"
# List of GPU memory CSV files
csv_files = [
    "gpu_mem_10_90.csv",
    "gpu_mem_30_70.csv",
    "gpu_mem_50_50.csv",
    "gpu_mem_70_30.csv",
    "gpu_mem_90_10.csv"
]
#idle gpu power that should be excluded
idle_power = 50
# Function to parse and calculate the average power.draw from each CSV file
# Function to parse and calculate the average power.draw from each CSV file, adjusting for units and column names
def calculate_average_power_adjusted(csv_files, idle_power):
    avg_power_list = []
    for file in csv_files:
        try:
            print(file)
            # Read the CSV file with full path
            df = pd.read_csv(os.path.join(csv_folder, file))
            #df = pd.read_csv(file)
            
            # Strip leading/trailing spaces from column names
            df.columns = df.columns.str.strip()
            
            # Remove the ' W' unit and convert 'power.draw [W]' to float
            df['power.draw [W]'] = df['power.draw [W]'].str.replace(' W', '').astype(float)
            
            #print(df['power.draw [W]'].tolist())
            #drop power below idle power
            df = df[df['power.draw [W]'] > idle_power]
            # Calculate the average of 'power.draw [W]'
            avg_power = df['power.draw [W]'].mean()
            avg_power_list.append(avg_power)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue
    return avg_power_list

def combine_w1_w2_logs(log_folder):
    thread_percentages = ["10", "30", "50", "70", "90"]
    combined_results = {}

    for percentage in thread_percentages:
        w1_file = f"{log_folder}/w1_{w1}_MPS{percentage}.log"
        w2_file = f"{log_folder}/w2_{w2}_MPS{100-int(percentage)}.log"
        print(f"parseing {w1_file} and {w2_file}")
        print("w1 is train" if "train" in w1 else "w1 is inf")
        print("w2 is train" if "train" in w2 else "w2 is inf")
        w1_avg_step_time = parse_train_log_avgStep_file(w1_file) if "train" in w1 else parse_inf_log_avgStep_file(w1_file)
        w2_avg_step_time = parse_train_log_avgStep_file(w2_file) if "train" in w2 else parse_inf_log_avgStep_file(w2_file)
            
        #w1_avg_step_time = parse_inf_log_avgStep_file(w1_file)
        #w2_avg_step_time = parse_inf_log_avgStep_file(w2_file)

        if w1_avg_step_time is not None and w2_avg_step_time is not None:
            combined_results[f"{percentage}%"] = {
                "W1": w1_avg_step_time,
                "W2": w2_avg_step_time
            }

    # Print the combined results
    for percentage, result in combined_results.items():
        print(f"\nThread Percentage: {percentage}")
        print(f"W1 Avg Step Time: {result['W1']:.4f} steps/second")
        print(f"W2 Avg Step Time: {result['W2']:.4f} steps/second")
    return combined_results
combined_results = combine_w1_w2_logs(csv_folder)
print(combined_results)
#plot combined results
#labels
labels = [os.path.basename(file).replace("gpu_mem_", "").replace(".csv", "") for file in csv_files]
print(labels)
#delete '10_90' from label
labels = labels[1:]


plt.figure(figsize=(8, 6))
plt.plot(labels, [result['W1'] for result in combined_results.values()], marker='o', linestyle='-', color='b', label='W1')
plt.plot(labels, [result['W2'] for result in combined_results.values()], marker='o', linestyle='-', color='r', label='W2')
#plot sum of W1 and W2
plt.plot(labels, [result['W1'] + result['W2'] for result in combined_results.values()], marker='o', linestyle='-', color='g', label='W1 + W2')
plt.title(f'Throughput for {w1} + {w2}')
plt.xlabel('Thread Combination')
plt.ylabel('Throughput (steps/second)')
plt.legend()
plt.grid(True)
plt.savefig(f'average_step_time{w1} + {w2}.png')
# Calculate the average power draw for each file with adjusted column names and values
avg_power_list_adjusted = calculate_average_power_adjusted(csv_files, idle_power)
if len(labels) < 5:
    avg_power_list_adjusted = avg_power_list_adjusted[1:]

# Plot the average power draw with adjusted data
plt.figure(figsize=(8, 6))
plt.plot(labels, avg_power_list_adjusted, marker='o', linestyle='-', color='b')
plt.title(f'Average Power Draw for {w1} + {w2}')
plt.xlabel('Thread Combination')
plt.ylabel('Average Power Draw (W)')
plt.grid(True)
plt.savefig(f'average_power_{w1} + {w2}draw.png')
