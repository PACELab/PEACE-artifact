import pandas as pd
import os
"""
CONFIG VARIABLES
"""
# Declare reusable variables;
"""V100"""
"""
DEBUG = True
powercap_limit = 60
power_epsilon = 6
powercap_file_dict = {
    100: "/home/bihhan/mlProfiler/tests/mps/freq_scaling/dataset/05052025_nonDL_09152025_DL_mergecudaDL_powercap100_dvfs/merged_labels.csv",
    60:  "/home/bihhan/mlProfiler/tests/mps/freq_scaling/dataset/05052025_mergecudaDL_powercap60_dvfs/0505_nonDL_powercap60_dvfs_throughput_total_labels_comb2_labels.csv",
    200: "/home/bihhan/mlProfiler/tests/mps/freq_scaling/dataset/09102025_mergecudaDL_powercap200_dvfs/0910_powercap200_mergecudaDL_dvfs_throughput_total_labels_comb2_labels.csv"
}
powercapfile = powercap_file_dict[powercap_limit]
#powercapfile = "/home/bihhan/mlProfiler/tests/mps/freq_scaling/dataset/05052025_nonDL_09152025_DL_mergecudaDL_powercap100_dvfs/merged_labels.csv"
powercap_selected_freq_file = f"/home/bihhan/mlProfiler/tests/mps/analysis/plot_analysis/09152025_xput_under_powercap_mergecudaDL_unseen_multi_freq_powercap{powercap_limit}_epsilon{power_epsilon}_pred_vs_baselines_dvfs/summary_xput_under_powercap_all_pairs.csv"
powercap_result_file = f"powercap{powercap_limit}_result_optimal_percentages.csv"
freq300_path = "/home/bihhan/mlProfiler/tests/mps/freq_scaling/baseline_metrics/0506_FREQ300_baseline_metrics.csv"
freq900_path = "/home/bihhan/mlProfiler/tests/mps/freq_scaling/baseline_metrics/0502_FREQ900_baseline_metrics.csv"
freq1530_path = "/home/bihhan/mlProfiler/tests/mps/freq_scaling/baseline_metrics/0206_FREQ1530_baseline_metrics.csv"
#selected_freqs = ['300', '900', '1530']  # iterate through 300, 900, 1530
#read three freq files
freq300_df = pd.read_csv(freq300_path)
freq900_df = pd.read_csv(freq900_path)
freq1530_df = pd.read_csv(freq1530_path)
#available_freqs = ['300', '900', '1530']
output_dir = "v100"#relative output dir

# Create target dataset columns
column_names = ["workload_online", "workload_offline"]
target_df = pd.DataFrame(columns=column_names)

# Load the individual workload metrics file
# freq1530 this is for workload pair selection. NO other freq of shared throughput data should be used for this.
shared_xput_root_dir = "/home/bihhan/mlProfiler/tests/mps/analysis/stage2"

shared_xput_dict = {
    1530: [f"{shared_xput_root_dir}/09152025_freq1530_nodvfs_share_comb2_freqscale_throughput_individual_avg.csv", 
    f"{shared_xput_root_dir}/0311_freq1530_nonDL_nodvfs_share_comb2_freqscale_throughput_individual_avg.csv"]
}
#freq to determine on and offline workloads
freq_determine_on_and_offline_workloads = 1530
"""

"""RTX A6000"""

DEBUG = True
powercap_limit = 100
power_epsilon = 10
powercap_file_dict = {
    100: "/home/bihhan/mlProfiler/tests/mps/freq_scaling/dataset/rtxa6000/01312026_DL_powercap100_dvfs/merged_01312026_DL_powercap100_dvfs_throughput_total_labels_comb2_labels.csv"
}
powercapfile = powercap_file_dict[powercap_limit]
#powercapfile = "/home/bihhan/mlProfiler/tests/mps/freq_scaling/dataset/05052025_nonDL_09152025_DL_mergecudaDL_powercap100_dvfs/merged_labels.csv"
#"/home/bihhan/mlProfiler/tests/mps/analysis/plot_analysis/rtxa6000/02062026_xput_under_powercap_comb2_mergecudaDL_unseen_multi_freq_powercap100_epsilon10_pred_vs_baselines_dvfs/summary_xput_under_powercap_all_pairs.csv"
powercap_selected_freq_file = f"/home/bihhan/mlProfiler/tests/mps/analysis/plot_analysis/rtxa6000/02062026_xput_under_powercap_comb2_mergecudaDL_unseen_multi_freq_powercap{powercap_limit}_epsilon{power_epsilon}_pred_vs_baselines_dvfs/summary_xput_under_powercap_all_pairs.csv"
powercap_result_file = f"powercap{powercap_limit}_result_optimal_percentages.csv"
available_freqs = ['300', '1200', '2100']
freq_path_dict = {}

for freq in available_freqs:
    freq_path_dict[f"freq{freq}_path"] = f"/home/bihhan/mlProfiler/tests/mps/freq_scaling/baseline_metrics/rtxa6000/freq_{freq}.csv"

# unpack variables for compatibility
for varname, path in freq_path_dict.items():
    globals()[varname] = path

# Load the DataFrames using the dynamic variable names set above
for freq in available_freqs:
    df_varname = f"freq{freq}_df"
    path_varname = f"freq{freq}_path"
    globals()[df_varname] = pd.read_csv(globals()[path_varname])
#selected_freqs = ['300', '900', '1530']  # iterate through 300, 900, 1530
output_dir = "rtxa6000"#relative output dir

# Create target dataset columns
column_names = ["workload_online", "workload_offline"]
target_df = pd.DataFrame(columns=column_names)

# Load the individual workload metrics file
# freq1530 this is for workload pair selection. NO other freq of shared throughput data should be used for this.
shared_xput_root_dir = "/home/bihhan/mlProfiler/tests/mps/analysis/stage2/rtxa6000"

shared_xput_dict = {
    2100: [f"{shared_xput_root_dir}/01312025_DL_freq2100_share_comb2_freqscale_throughput_individual_avg.csv"
    ]
}

#freq to determine on and offline workloads
freq_determine_on_and_offline_workloads = 2100





# Concatenate df when given freq_determine_on_and_offline_workloads
indivdual_xput_when_shared_df = pd.concat([pd.read_csv(f) for f in shared_xput_dict[freq_determine_on_and_offline_workloads]])

print(f"pairs of the read shared throughput data under freq{freq_determine_on_and_offline_workloads}: {len(indivdual_xput_when_shared_df)}")

# Get the list of columns in the indivdual_xput_when_shared_df
print("Columns in indivdual_xput_when_shared_df:")
print(indivdual_xput_when_shared_df.columns.tolist())

"""
END CONFIG
"""
#config done
# Iterate through each row and extract workload names
print("\nIterating through rows and extracting workload names:")
for index, row in indivdual_xput_when_shared_df.iterrows():
    workload1 = row['workload1']
    workload2 = row['workload2']
    print(f"Row {index}: workload1='{workload1}', workload2='{workload2}'")

# Create a new DataFrame for the updated classification
updated_target_df = pd.DataFrame(columns=['workload_online', 'workload_offline'])

for index, row in indivdual_xput_when_shared_df.iterrows():
    workload1 = row['workload1']
    workload2 = row['workload2']

    # Rule 1: If one workload is inference ('inf' in name) and the other is not, 'inf' workload is online.
    if 'inf' in workload1 and 'inf' not in workload2:
        online_workload = workload1
        offline_workload = workload2
    elif 'inf' not in workload1 and 'inf' in workload2:
        online_workload = workload2
        offline_workload = workload1
    # Rule 2: In all other cases (both inf, both non-inf), consider both as online candidates.
    # For this updated logic, we will just list the pair as is, and the subsequent steps will handle which is online or offline.
    # We will represent this by creating two rows for each pair in this case.
    else:
        # Consider workload1 as online and workload2 as offline
        updated_target_df = pd.concat([updated_target_df, pd.DataFrame([{'workload_online': workload1, 'workload_offline': workload2}])], ignore_index=True)
        # Consider workload2 as online and workload1 as offline
        updated_target_df = pd.concat([updated_target_df, pd.DataFrame([{'workload_online': workload2, 'workload_offline': workload1}])], ignore_index=True)
        continue  # Skip the rest of the loop for this pair as we've added two rows

    # Append the classification to the updated_target_df for cases handled by Rule 1
    updated_target_df = pd.concat([updated_target_df, pd.DataFrame([{'workload_online': online_workload, 'workload_offline': offline_workload}])], ignore_index=True)
if DEBUG:
    updated_target_df.to_csv('updated_target_df.csv', index=False)

# Load powercap_selected_freq_file
powercap_selected_freq = pd.read_csv(powercap_selected_freq_file)

# Load powercap data
powercap_df = pd.read_csv(powercapfile)

# Create the new DataFrame
final_target_df = pd.DataFrame(columns=['Workload1', 'Workload2', 'Thread_combination', 'weight_Throughput_sum',
                                        'Power', 'Duration', 'Energy',
                                        'workload_online', 'workload_offline'])

# Iterate through each row of updated_target_df
for index, row_updated in updated_target_df.iterrows():
    online_workload = row_updated['workload_online']
    offline_workload = row_updated['workload_offline']

    # Determine if the pair is inference/non-inference or other
    is_inference_non_inference = ('inf' in online_workload and 'inf' not in offline_workload) or \
                                 ('inf' not in online_workload and 'inf' in offline_workload)

    if is_inference_non_inference:
        # Case 1: Exactly one workload is inference
        # Check if the pair is present as is in powercap_df
        matching_rows_powercap = powercap_df[
            (powercap_df['Workload1'] == online_workload) & (powercap_df['Workload2'] == offline_workload) |
            ((powercap_df['Workload1'] == offline_workload) & (powercap_df['Workload2'] == online_workload))
        ]

        if not matching_rows_powercap.empty:
            # Add all 10 rows corresponding to the partition-pairs
            for _, row_powercap in matching_rows_powercap.iterrows():
                new_row = row_powercap.to_dict()
                new_row['workload_online'] = online_workload
                new_row['workload_offline'] = offline_workload
                final_target_df = pd.concat([final_target_df, pd.DataFrame([new_row])], ignore_index=True)

    else:
        # Case 2: Both inf or both non-inf
        # Check if the pair is present as is or in reverse in powercap_df
        matching_rows_powercap = powercap_df[
            ((powercap_df['Workload1'] == online_workload) & (powercap_df['Workload2'] == offline_workload)) |
            ((powercap_df['Workload1'] == offline_workload) & (powercap_df['Workload2'] == online_workload))
        ]

        if not matching_rows_powercap.empty:
            # Add all 10 rows corresponding to the partition-pairs
            for _, row_powercap in matching_rows_powercap.iterrows():
                new_row = row_powercap.to_dict()
                new_row['workload_online'] = online_workload
                new_row['workload_offline'] = offline_workload
                final_target_df = pd.concat([final_target_df, pd.DataFrame([new_row])], ignore_index=True)


if DEBUG:
    target_df.to_csv('final_target_df.csv', index=False)

# Rename columns in powercap_selected_freq
powercap_selected_freq = powercap_selected_freq.rename(columns={'workload1': 'Workload1', 'workload2': 'Workload2'})

# Merge target_df with powercap_selected_freq
if 'selected_model_freq' in final_target_df.columns:
    final_target_df = final_target_df.drop(columns=['selected_model_freq'])

final_target_df = final_target_df.merge(powercap_selected_freq[['Workload1', 'Workload2', 'selected_model_freq']],
                                        on=['Workload1', 'Workload2'],
                                        how='left')



# Create a new column for online_smact if it doesn't exist
if 'online_smact' not in final_target_df.columns:
    final_target_df['online_smact'] = None

# Iterate through each row of final_target_df
#for selected_freq in selected_freqs:
#print(f"Processing frequency: {selected_freq}")
for index, row in final_target_df.iterrows():
    online_workload = row['workload_online']
    selected_freq = row['selected_model_freq']

    smact_value = None
    freq_type = None
    
    # Dynamically get the dataframe using the configurable variable names
    freq_type = selected_freq.replace('freq', '')
    df_varname = f"{selected_freq}_df"
    
    if df_varname in globals():
        smact_df = globals()[df_varname]
    else:
        raise ValueError(f"Unknown frequency '{selected_freq}' at index {index}. Available frequencies: {available_freqs}")
        
    # Find the SMACT%100 value for the online workload in the selected frequency DataFrame
    smact_row = smact_df[smact_df['Type'] == online_workload]

    if not smact_row.empty:
        smact_value = smact_row['SMACT%100'].iloc[0]
    else:
        print(f"Warning: Online workload '{online_workload}' not found in {selected_freq}_df at index {index}")

    # Assign the SMACT value to the online_smact column
    final_target_df.at[index, 'online_smact'] = smact_value

print(smact_value)

import numpy as np

if 'estimated_offline_partition' in final_target_df.columns:
    final_target_df = final_target_df.drop(columns=['estimated_offline_partition'])
if 'estimated_offline_smact' in final_target_df.columns:
    final_target_df = final_target_df.drop(columns=['estimated_offline_smact'])

# Calculate estimated offline SMACT and partition
final_target_df['estimated_offline_smact'] = 100 - final_target_df['online_smact']
final_target_df['estimated_offline_smact'] = pd.to_numeric(final_target_df['estimated_offline_smact'])

# Round to nearest multiple of 10, then clip between 10 and 90
final_target_df['estimated_offline_partition'] = (
    (np.round(final_target_df['estimated_offline_smact'] / 10) * 10)
    .clip(lower=10, upper=90)
    .astype(int)
)

print(f"final_target_df shape: {final_target_df.shape}")

# Reorder columns
new_column_order = ['Workload1', 'Workload2', 'workload_online', 'workload_offline', 'Thread_combination', 'weight_Throughput_sum',
                    'Power', 'Duration', 'Energy', 'selected_model_freq',
                    'online_smact', 'estimated_offline_smact', 'estimated_offline_partition']
final_target_df = final_target_df[new_column_order]

from typing_extensions import final

# Create new columns to store reordered thread combinations and weighted throughput
final_target_df['ThreadCombination_online_offline'] = None
final_target_df['Weighted_Thruput_Online_Offline'] = None

# Define a mapping from original column names to online/offline structure
partition_map = {
    '(w1_10, w2_90)': (10, 90),
    '(w1_20, w2_80)': (20, 80),
    '(w1_30, w2_70)': (30, 70),
    '(w1_40, w2_60)': (40, 60),
    '(w1_50, w2_50)': (50, 50),
    '(w1_60, w2_40)': (60, 40),
    '(w1_70, w2_30)': (70, 30),
    '(w1_80, w2_20)': (80, 20),
    '(w1_90, w2_10)': (90, 10),
    '(w1_100, w2_100)': (100, 100)
}

# Iterate through each row and reorder based on online/offline classification
for index, row in final_target_df.iterrows():
    thread_combination = row['Thread_combination']
    weighted_throughput = row['weight_Throughput_sum']

    online_workload = row['workload_online']
    workload1_pc = row['Workload1']

    # Determine the online and offline percentages based on the thread combination string
    # Assuming the format is consistent like '(w1_XX, w2_YY)'
    try:
        # Extract percentages from the string
        parts = thread_combination.replace('(', '').replace(')', '').split(', ')
        w1_percent = int(parts[0].split('_')[1])
        w2_percent = int(parts[1].split('_')[1])

        # Reorder based on which workload is online
        if online_workload == workload1_pc:
            online_percent = w1_percent
            offline_percent = w2_percent
            online_throughput = weighted_throughput  # Assuming weighted_throughput is for the pair
            offline_throughput = None  # We don't have individual throughput here
        else:  # workload_online == Workload2
            online_percent = w2_percent
            offline_percent = w1_percent
            online_throughput = weighted_throughput  # Assuming weighted_throughput is for the pair
            offline_throughput = None  # We don't have individual throughput here

        # Store the reordered thread combination and weighted throughput
        final_target_df.at[index, 'ThreadCombination_online_offline'] = f'(online_{online_percent}, offline_{offline_percent})'
        final_target_df.at[index, 'Weighted_Thruput_Online_Offline'] = weighted_throughput  # Store the pair's weighted throughput
    except:
        # Handle cases where the thread_combination string format is unexpected
        print(f"Could not parse thread_combination string: {thread_combination} at index {index}")
        final_target_df.at[index, 'ThreadCombination_online_offline'] = None
        final_target_df.at[index, 'Weighted_Thruput_Online_Offline'] = None
if DEBUG:
    final_target_df.to_csv('final_target_df.csv', index=False)

# Create a list to store the data with the calculated ratios
processed_chunks_data = {"estimated_best_partition_throughput": [], "actual_best_partition_throughput": [], "estimated_to_actual_ratio": []}

# Iterate through final_target_df in chunks of 10
chunk_size = 10
for i in range(0, len(final_target_df), chunk_size):
    chunk = final_target_df.iloc[i:i + chunk_size].copy()

    # Filter out (w1_100, w2_100) rows before calculating max throughput
    if DEBUG:
        #print chunk thread combination
        print(chunk['Thread_combination'].tolist())
        #print()
    chunk_filtered = chunk[chunk['Thread_combination'] != '(w1_100, w2_100)']
    max_value = chunk_filtered['Weighted_Thruput_Online_Offline'].max()

    # Get the estimated offline partition for this chunk (assuming it's the same for all rows in the chunk)
    estimated_partition = chunk['estimated_offline_partition'].iloc[0]
    current_pair = (final_target_df["workload_online"].iloc[i], final_target_df["workload_offline"].iloc[i])

    # Construct the estimated thread combination string
    estimated_online_percent = 100 - estimated_partition
    estimated_combination_str = f'(online_{int(estimated_online_percent)}, offline_{int(estimated_partition)})'

    estimated_throughput_value = chunk[chunk['ThreadCombination_online_offline'] == estimated_combination_str]['Weighted_Thruput_Online_Offline'].iloc[0]

    processed_chunks_data["estimated_best_partition_throughput"].extend([estimated_throughput_value] * chunk_size)
    processed_chunks_data["actual_best_partition_throughput"].extend([max_value] * chunk_size)
    processed_chunks_data["estimated_to_actual_ratio"].extend([estimated_throughput_value/max_value] * chunk_size)

df_new_columns = pd.DataFrame(processed_chunks_data)

final_target_df = pd.concat([final_target_df, df_new_columns], axis=1)

if DEBUG:
    final_target_df.to_csv(powercap_result_file, index=False)

# Create a new DataFrame to store the filtered results
filtered_final_target_df = pd.DataFrame(columns=final_target_df.columns)

# Iterate through final_target_df in chunks of 10
chunk_size = 10
i = 0
while i < len(final_target_df):
    chunk1 = final_target_df.iloc[i : i + chunk_size].copy()
    workload1_chunk1 = chunk1['workload_online'].iloc[0]
    workload2_chunk1 = chunk1['workload_offline'].iloc[0]

    # Check if the pair is both inf or both non-inf
    is_both_inf = ('inf' in workload1_chunk1 and 'inf' in workload2_chunk1)
    is_both_non_inf = ('inf' not in workload1_chunk1 and 'inf' not in workload2_chunk1)

    if is_both_inf or is_both_non_inf:
        # Get the next chunk which should be the reversed pair
        chunk2 = final_target_df.iloc[i + chunk_size : i + (2 * chunk_size)].copy()
        workload1_chunk2 = chunk2['workload_online'].iloc[0]
        workload2_chunk2 = chunk2['workload_offline'].iloc[0]

        # Verify that chunk2 is the reversed pair of chunk1
        if (workload1_chunk1 == workload2_chunk2) and (workload2_chunk1 == workload1_chunk2):
            estimated_throughput_chunk1 = chunk1['estimated_best_partition_throughput'].iloc[0]
            estimated_throughput_chunk2 = chunk2['estimated_best_partition_throughput'].iloc[0]

            # Compare estimated throughput and keep the chunk with the higher value
            if estimated_throughput_chunk1 >= estimated_throughput_chunk2:
                filtered_final_target_df = pd.concat([filtered_final_target_df, chunk1], ignore_index=True)
            else:
                filtered_final_target_df = pd.concat([filtered_final_target_df, chunk2], ignore_index=True)

            i += 2 * chunk_size  # Move to the next pair of chunks
        else:
            # If the next chunk is not the reversed pair, something is unexpected.
            # Append chunk1 and move to the next chunk
            print(f"Warning: Expected reversed pair at index {i + chunk_size} but found different pair. Appending chunk at index {i}.")
            filtered_final_target_df = pd.concat([filtered_final_target_df, chunk1], ignore_index=True)
            i += chunk_size  # Move to the next chunk
    else:
        # If the pair is inf/non-inf, keep the chunk as is and move to the next chunk
        filtered_final_target_df = pd.concat([filtered_final_target_df, chunk1], ignore_index=True)
        i += chunk_size  # Move to the next chunk

print(f"Filtered DataFrame shape: {filtered_final_target_df.shape}")

# Create new columns for optimal percentages
filtered_final_target_df['w1_optimal_percentage'] = None
filtered_final_target_df['w2_optimal_percentage'] = None

# Iterate through the filtered DataFrame and calculate optimal percentages
for index, row in filtered_final_target_df.iterrows():
    workload1 = row['Workload1']
    workload2 = row['Workload2']
    workload_online = row['workload_online']
    estimated_partition = row['estimated_offline_partition']

    if workload1 == workload_online:
        filtered_final_target_df.at[index, 'w1_optimal_percentage'] = 100 - estimated_partition
        filtered_final_target_df.at[index, 'w2_optimal_percentage'] = estimated_partition
    else:  # Workload2 must be workload_online
        filtered_final_target_df.at[index, 'w1_optimal_percentage'] = estimated_partition
        filtered_final_target_df.at[index, 'w2_optimal_percentage'] = 100 - estimated_partition

# Find average of the estimated_to_actual_ratio column
average_ratio = filtered_final_target_df['estimated_to_actual_ratio'].mean()
print("Average estimated_to_actual_ratio:", average_ratio)

# Before saving final results, for each workload1, workload2 pair, only keep the row where Weighted_Thruput_Online_Offline == estimated_best_partition_throughput
filtered_final_target_df = filtered_final_target_df[
    filtered_final_target_df['Weighted_Thruput_Online_Offline'] == filtered_final_target_df['estimated_best_partition_throughput']
]
print(f"Final filtered DataFrame shape (keeping only optimal partitions): {filtered_final_target_df.shape}")

# Save final results
#create output dir if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
filtered_final_target_df.to_csv(f'{output_dir}/final-' + powercap_result_file, index=False)
print(f"saved to final" + powercap_result_file)
