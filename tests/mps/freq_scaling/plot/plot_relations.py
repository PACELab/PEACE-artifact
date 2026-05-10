import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load your data
file_steps_path = '/home/cc/mlProfiler/tests/mps/analysis/stage2/1015_smallset_share_comb2_freqscale_stage2.csv'
file_power_path = '/home/cc/mlProfiler/tests/mps/analysis/stage2/1015_smallset_share_comb2_freqscale_stage2_power.csv'

# Load the datasets
file_steps_df = pd.read_csv(file_steps_path)
file_power_df = pd.read_csv(file_power_path)

# Merge the two dataframes based on common columns (workload1, freq1, workload2, freq2)
merged_df = pd.merge(file_steps_df, file_power_df, on=['workload1', 'workload2', 'freq1','freq2'], suffixes=('_throughput', '_power'))

# Filter out rows where throughput data is missing or "None"
merged_df = merged_df[merged_df['(w1_100, w2_100)_throughput'].apply(lambda x: x != "(None, None)")].copy()

# Convert throughput tuples to numeric values and calculate throughput_sum
merged_df['throughput_sum'] = merged_df['(w1_100, w2_100)_throughput'].apply(lambda x: float(x.split(',')[0][1:])) + \
                              merged_df['(w1_100, w2_100)_throughput'].apply(lambda x: float(x.split(',')[1][:-1]))

# Unique workload pairs
unique_workload_pairs = merged_df[['workload1', 'workload2']].drop_duplicates()

# Create 3D plots for each unique workload pair with different thread partitions
fig = plt.figure(figsize=(18, 12))

# Define thread partition combinations we're interested in
thread_partitions = ['(w1_100, w2_100)', '(w1_90, w2_10)', '(w1_10, w2_90)', '(w1_50, w2_50)']
colors = ['r', 'g', 'b', 'c']  # Colors for different thread partitions

for idx, (workload1, workload2) in enumerate(unique_workload_pairs.values):
    ax = fig.add_subplot(2, 3, idx + 1, projection='3d')  # Create a 2x3 grid for six graphs
    
    # Filter data for the current workload pair
    workload_pair_data = merged_df[(merged_df['workload1'] == workload1) & (merged_df['workload2'] == workload2)]
    
    # Get the axes data for the current workload pair
    throughput_sum = workload_pair_data['throughput_sum']
    frequency = workload_pair_data['freq1'].astype(float)

    # Plot the data for each thread partition
    for partition, color in zip(thread_partitions, colors):
        if f'{partition}_power' in workload_pair_data.columns:
            partition_power = workload_pair_data[f'{partition}_power']
            ax.scatter(throughput_sum, frequency, partition_power, label=partition, color=color)
    
    # Labels and titles for each graph
    ax.set_xlabel('Throughput Sum')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_zlabel('Power (W)')
    ax.set_title(f'3D Plot: {workload1} vs {workload2}')
    
    # Add a legend to indicate different thread partitions
    ax.legend()

# Adjust layout and display the plots
plt.tight_layout()
plt.savefig('3d_plots.png')
