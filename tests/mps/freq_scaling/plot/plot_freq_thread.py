import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load your data
file_steps_path = '/home/cc/mlProfiler/tests/mps/analysis/stage2/1015_smallset_share_comb2_freqscale_stage2.csv'
file_power_path = '/home/cc/mlProfiler/tests/mps/analysis/stage2/1015_smallset_share_comb2_freqscale_stage2_power.csv'

# Load the datasets
file_steps_df = pd.read_csv(file_steps_path)
file_power_df = pd.read_csv(file_power_path)

# Merge the two dataframes based on common columns (workload1, freq1, workload2, freq2)
merged_df = pd.merge(file_steps_df, file_power_df, on=['workload1', 'freq1', 'workload2', 'freq2'], suffixes=('_throughput', '_power'))

# Filter out rows where throughput data is missing or "None"
merged_df = merged_df[merged_df['(w1_100, w2_100)_throughput'].apply(lambda x: x != "(None, None)")].copy()

# Unique workload pairs
unique_workload_pairs = merged_df[['workload1', 'workload2']].drop_duplicates()

# Define thread partition combinations we're interested in
thread_partitions = ['(w1_100, w2_100)', '(w1_90, w2_10)', '(w1_10, w2_90)', '(w1_50, w2_50)', '(w1_70, w2_30)', '(w1_30, w2_70)']

# Assigning each partition an index for y-axis representation
partition_indices = {partition: idx for idx, partition in enumerate(thread_partitions)}

# Function to calculate throughput sum for each partition
def calculate_throughput_sum(throughput_col):
    try:
        return float(throughput_col.split(',')[0][1:]) + float(throughput_col.split(',')[1][:-1])
    except:
        return None

# Plot 1: Throughput Sum
for idx, (workload1, workload2) in enumerate(unique_workload_pairs.values):
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection='3d')  # Separate figure for throughput sum

    # Filter data for the current workload pair
    workload_pair_data = merged_df[(merged_df['workload1'] == workload1) & (merged_df['workload2'] == workload2)]
    
    print(f"Workload pair: {workload1} vs {workload2}")
    
    # Plot throughput sum for each thread partition
    for partition in thread_partitions:
        throughput_col = f'{partition}_throughput'
        
        if throughput_col in workload_pair_data.columns:
            # Calculate throughput sum
            workload_pair_data[f'{partition}_throughput_sum'] = workload_pair_data[throughput_col].apply(calculate_throughput_sum)
            
            # Filter out rows where throughput sum is None
            valid_data = workload_pair_data[workload_pair_data[f'{partition}_throughput_sum'].notnull()]
            
            throughput_sum = valid_data[f'{partition}_throughput_sum']
            frequency = valid_data['freq1'].astype(float)
            partition_index = np.full_like(frequency, partition_indices[partition])  # Constant partition index
            
            # Plot the valid points
            ax1.scatter(frequency, partition_index, throughput_sum, label=partition)
    
    # Labels and title for throughput sum plot
    ax1.set_xlabel('Frequency (MHz)')
    ax1.set_ylabel('Thread Partition')
    ax1.set_zlabel('Throughput Sum')
    ax1.set_title(f'Throughput Sum: {workload1} vs {workload2}')
    ax1.legend()
    
    # Save the figure for throughput sum
    plt.tight_layout()
    plt.savefig(f'figs/throughput_sum_{workload1}_vs_{workload2}.png')


# Plot 2: Power
for idx, (workload1, workload2) in enumerate(unique_workload_pairs.values):
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection='3d')  # Separate figure for power
    
    # Filter data for the current workload pair
    workload_pair_data = merged_df[(merged_df['workload1'] == workload1) & (merged_df['workload2'] == workload2)]
    
    # Plot power for each thread partition
    for partition in thread_partitions:
        power_col = f'{partition}_power'
        
        if power_col in workload_pair_data.columns:
            # Filter out rows where power is None
            valid_data = workload_pair_data[workload_pair_data[power_col].notnull()]
            
            power = valid_data[power_col]
            frequency = valid_data['freq1'].astype(float)
            partition_index = np.full_like(frequency, partition_indices[partition])  # Constant partition index
            
            # Plot the valid points
            ax2.scatter(frequency, partition_index, power, label=partition)
    
    # Labels and title for power plot
    ax2.set_xlabel('Frequency (MHz)')
    ax2.set_ylabel('Thread Partition')
    ax2.set_zlabel('Power (W)')
    ax2.set_title(f'Power: {workload1} vs {workload2}')
    ax2.legend()
    
    # Save the figure for power
    plt.tight_layout()
    plt.savefig(f'figs/power_{workload1}_vs_{workload2}.png')

