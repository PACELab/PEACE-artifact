import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Font sizes
axis_font, label_font = 14, 16

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot colocated performance data')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--powercap', action='store_true', help='Plot powercap data')
group.add_argument('--freq', action='store_true', help='Plot frequency data')
args = parser.parse_args()

# File paths organized by powercap
powercap_files_throughput = {
    60: [
        f"{REPO_ROOT}/data/colocations/05052025_nonDL_powercap60_dvfs_share_comb2_freqscale_throughput_individual_avg.csv",
        f"{REPO_ROOT}/data/colocations/05052025_powercap60_DL_dvfs_share_comb2_freqscale_throughput_individual_avg.csv"
    ],
    100: [
        f"{REPO_ROOT}/data/colocations/05052025_powercap100_DL_dvfs_share_comb2_freqscale_throughput_individual_avg.csv",
        f"{REPO_ROOT}/data/colocations/05052025_powercap100_nonDL_dvfs_share_comb2_freqscale_throughput_individual_avg.csv"
    ],
    200:[
        f"{REPO_ROOT}/data/colocations/09102025_powercap200_nonDL_dvfs_share_comb2_freqscale_throughput_individual_avg.csv"
    ]
}

powercap_files_power = {
    60: [
        f"{REPO_ROOT}/data/colocations/05052025_nonDL_powercap60_dvfs_share_comb2_freqscale_power_avg_stage2.csv",
        f"{REPO_ROOT}/data/colocations/05052025_powercap60_DL_dvfs_share_comb2_freqscale_power_avg_stage2.csv"
    ],
    100: [
        f"{REPO_ROOT}/data/colocations/05052025_powercap100_DL_dvfs_share_comb2_freqscale_power_avg_stage2.csv",
        f"{REPO_ROOT}/data/colocations/05052025_powercap100_nonDL_dvfs_share_comb2_freqscale_power_avg_stage2.csv"
    ],
    200:[
        f"{REPO_ROOT}/data/colocations/09102025_powercap200_nonDL_dvfs_share_comb2_freqscale_power_avg_stage2.csv"
    ]
}

# File paths organized by frequency - these files contain both throughput and power data
freq_files = {
    1500: f"{REPO_ROOT}/data/model_datasets/03112025_DL0207_0307_nonDL0311_nodvfs/mergecudaDL_nodvfs_throughput_total_labels_comb2_labels.csv",
    300: f"{REPO_ROOT}/data/model_datasets/05052025_FREQ300_mergecudaDL_nodvfs/0505_FREQ300_nodvfs_throughput_total_labels_comb2_labels.csv",
    900: f"{REPO_ROOT}/data/model_datasets/05052025_FREQ900_mergecudaDL_nodvfs/0505_FREQ900_nodvfs_throughput_total_labels_comb2_labels.csv"
}

# Set output directory and mode based on arguments
if args.powercap:
    output_dir = "powercap_dvfs"
    mode = "powercap"
else:
    output_dir = "freq_nodvfs"
    mode = "freq"
base_powercap = 100
base_freq = 900  # Base frequency for normalization in freq mode
base_percent = 50

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load and combine CSV files by powercap
def load_powercap_data(powercap_files_dict):
    combined_data = {}
    for powercap, file_list in powercap_files_dict.items():
        combined_df = pd.DataFrame()
        for file_path in file_list:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['powercap'] = powercap  # Add powercap column
                combined_df = pd.concat([combined_df, df], ignore_index=True)
        combined_data[powercap] = combined_df
    return combined_data

# Load frequency data - these files contain both throughput and power data
def load_freq_data(freq_files_dict):
    combined_data = {}
    for freq, file_path in freq_files_dict.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['freq'] = freq  # Add frequency column
            # Rename columns to match expected format
            if 'Workload1' in df.columns:
                df = df.rename(columns={'Workload1': 'workload1', 'Workload2': 'workload2'})
            combined_data[freq] = df
    return combined_data

# Load data based on mode
if mode == "powercap":
    data_power_dict = load_powercap_data(powercap_files_power)
    data_throughput_dict = load_powercap_data(powercap_files_throughput)
else:  # freq mode
    # Load frequency data - these files contain both throughput and power data
    freq_data_dict = load_freq_data(freq_files)
    # For freq mode, we use the same unified data for both power and throughput
    data_power_dict = freq_data_dict
    data_throughput_dict = freq_data_dict

# Function to create customized subfigures
def create_custom_subfigs(selected_axes):
    axes_map = {
        1: ('Power vs Thread Partition', 'Power (W)'),
        2: ('1st Throughput vs Thread Partition', 'Throughput1'),
        3: ('2nd Throughput vs Thread Partition', 'Throughput2'),
        4: ('Throughput Sum vs Thread Partition', 'Throughput Sum'),
        6: ('Throughput per Watt vs Thread Partition', 'Throughput per Watt')
    }
    
    n_plots = len(selected_axes)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    return fig, axes, {i: axes_map[i] for i in selected_axes}

# Get unique workload pairs from the first available dataset
first_key = list(data_power_dict.keys())[0]
if mode == "powercap":
    workload_pairs = data_power_dict[first_key][['workload1', 'workload2']].drop_duplicates()
else:  # freq mode
    workload_pairs = data_power_dict[first_key][['workload1', 'workload2']].drop_duplicates()

# Specify which axes to plot based on mode
if mode == "powercap":
    selected_axes = [1, 2, 3, 4, 6]  # All plots for powercap
else:  # freq mode
    selected_axes = [1, 4, 6]  # Only power, throughput sum, and throughput per watt

# Loop through each workload pair
for _, row in workload_pairs.iterrows():
    w1 = row['workload1']
    w2 = row['workload2']
    
    # Create custom subfigures
    fig, axes, axes_info = create_custom_subfigs(selected_axes)
    
    # Get thread partitions from the first available dataset
    first_data = data_power_dict[first_key]
    subset_first = first_data[(first_data['workload1'] == w1) & (first_data['workload2'] == w2)]
    if subset_first.empty:
        print(f"No data found for workload pair {w1} & {w2}")
        continue
    
    if mode == "powercap":
        thread_partitions = [col for col in subset_first.columns if 'w1' in col and "100" not in col]
        base_partition = [col for col in subset_first.columns if f"{base_percent}" in col][0]
        print(f"Base partition: {base_partition}")
    else:  # freq mode
        # For freq mode, parse thread partitions from Thread_combination column
        unique_combinations = subset_first['Thread_combination'].unique()
        thread_partitions = []
        for combo in unique_combinations:
            # Extract w1 and w2 percentages from "(w1_X, w2_Y)" format
            match = re.search(r'w1_(\d+), w2_(\d+)', combo)
            if match:
                w1_pct = int(match.group(1))
                w2_pct = int(match.group(2))
                # Exclude 100_100 case and only include valid partitions
                if w1_pct + w2_pct == 100 and w1_pct != 100:
                    thread_partitions.append(f"{w1_pct}_{w2_pct}")
        # Sort by w1 percentage
        thread_partitions = sorted(thread_partitions, key=lambda x: int(x.split('_')[0]))
        base_partition = f"{base_percent}_{100-base_percent}"  # Use full partition format
        print(f"Thread partitions: {thread_partitions}")
        print(f"Base partition: {base_partition}")
    
    # Store legend handles and labels for global legend
    legend_handles = []
    legend_labels = []
    
    # Define unique colors with good contrast for different powercaps
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Get base values
    if mode == "powercap":
        # Get base values from base_powercap
        base_data_throughput = data_throughput_dict[base_powercap]
        base_subset_throughput = base_data_throughput[(base_data_throughput['workload1'] == w1) & (base_data_throughput['workload2'] == w2)]
        
        if not base_subset_throughput.empty:
            base_throughput = base_subset_throughput[base_partition].iloc[0]
            if pd.notna(base_throughput) and "None" not in str(base_throughput):
                base_throughput1 = float(str(base_throughput).split(",")[0].replace("(", "").strip())
                base_throughput2 = float(str(base_throughput).split(",")[1].replace(")", "").strip())
                base_throughput_sum = base_throughput1 + base_throughput2
            else:
                base_throughput1 = None
                base_throughput2 = None
                base_throughput_sum = None
        else:
            base_throughput1 = None
            base_throughput2 = None
            base_throughput_sum = None

        # Get base power value
        base_data_power = data_power_dict[base_powercap]
        base_subset_power = base_data_power[(base_data_power['workload1'] == w1) & (base_data_power['workload2'] == w2)]
        
        if not base_subset_power.empty:
            base_power = float(base_subset_power[base_partition].iloc[0])
        else:
            base_power = None
    else:  # freq mode
        # Get base values from base_freq and base_partition, but fallback to first available if not found
        base_data = None
        base_subset = pd.DataFrame()
        
        # Try base_freq first
        if base_freq in data_power_dict:
            base_data = data_power_dict[base_freq]
            base_subset = base_data[(base_data['workload1'] == w1) & (base_data['workload2'] == w2)]
            # Filter by base partition using full format
            if not base_subset.empty:
                base_w1_pct = base_partition.split('_')[0]
                base_w2_pct = base_partition.split('_')[1]
                base_subset = base_subset[base_subset['Thread_combination'].str.contains(f'w1_{base_w1_pct}, w2_{base_w2_pct}')]
        
        # If not found in base_freq with base_partition, try other frequencies
        if base_subset.empty:
            for freq in sorted(data_power_dict.keys()):
                base_data = data_power_dict[freq]
                temp_subset = base_data[(base_data['workload1'] == w1) & (base_data['workload2'] == w2)]
                if not temp_subset.empty:
                    # Try to find base partition first
                    base_w1_pct = base_partition.split('_')[0]
                    base_w2_pct = base_partition.split('_')[1]
                    base_subset = temp_subset[temp_subset['Thread_combination'].str.contains(f'w1_{base_w1_pct}, w2_{base_w2_pct}')]
                    if not base_subset.empty:
                        print(f"Using frequency {freq}MHz as base instead of {base_freq}MHz for workload pair {w1} & {w2}")
                        break
                    # If base partition not found, use first available partition as fallback
                    elif base_subset.empty and temp_subset.iloc[0]['Thread_combination'] is not None:
                        base_subset = temp_subset.iloc[[0]]
                        print(f"Using frequency {freq}MHz and partition {temp_subset.iloc[0]['Thread_combination']} as base for workload pair {w1} & {w2}")
                        break
        
        if not base_subset.empty:
            base_throughput_sum = float(base_subset['weight_Throughput_sum'].iloc[0]) if 'weight_Throughput_sum' in base_subset.columns else None
            base_power = float(base_subset['Power'].iloc[0]) if 'Power' in base_subset.columns else None
        else:
            base_throughput_sum = None
            base_power = None
        
        base_throughput1 = None  # Not used in freq mode
        base_throughput2 = None  # Not used in freq mode

    print(f"Base throughput for {w1} & {w2} at powercap {base_powercap}W: {base_throughput_sum}")
    print(f"Base power for {w1} & {w2} at powercap {base_powercap}W: {base_power}")
    
    if base_throughput_sum is None or base_power is None:
        print(f"Skipping {w1} & {w2} due to missing base values.")
        continue

    power_values_allpowercap = {}
    power_values_allpowercap_raw = {}  # Store raw power values
    
    # Plot data for each selected axis
    for idx, ax_num in enumerate(selected_axes):
        ax = axes[idx]
        
        # Plot Power vs Thread Partition/Frequency (ax1)
        if ax_num == 1:
            if mode == "powercap":
                for powercap_idx, powercap in enumerate(sorted(data_power_dict.keys())):
                    data_power = data_power_dict[powercap]
                    subset_power = data_power[(data_power['workload1'] == w1) & (data_power['workload2'] == w2)]
                    
                    if subset_power.empty:
                        continue
                        
                    power_values = [
                        float(subset_power[col].iloc[0]) if pd.notna(subset_power[col].iloc[0]) else None
                        for col in thread_partitions
                    ]
                    # Store raw power values
                    power_values_allpowercap_raw[powercap] = power_values.copy()
                    # Normalize power values for plotting
                    power_values_norm = [x / base_power if x is not None else None for x in power_values]
                    power_values_allpowercap[powercap] = power_values_norm
                    color = colors[powercap_idx % len(colors)]
                    line = ax.plot(thread_partitions, power_values_norm, label=f'PowerCap {powercap}W', marker='o', linestyle='-', linewidth=2, color=color)
                    # Store legend info only once from the first plot
                    if idx == 0 and f'PowerCap {powercap}W' not in legend_labels:
                        legend_handles.append(line[0])
                        legend_labels.append(f'PowerCap {powercap}W')
            else:  # freq mode
                for freq_idx, freq in enumerate(sorted(data_power_dict.keys())):
                    data_freq = data_power_dict[freq]
                    subset_freq = data_freq[(data_freq['workload1'] == w1) & (data_freq['workload2'] == w2)]
                    
                    if subset_freq.empty:
                        continue
                    
                    power_values = []
                    power_values_raw = []
                    
                    for partition in thread_partitions:
                        w1_pct = partition.split('_')[0]
                        w2_pct = partition.split('_')[1]
                        # Filter by thread partition
                        partition_subset = subset_freq[subset_freq['Thread_combination'].str.contains(f'w1_{w1_pct}, w2_{w2_pct}')]
                        if not partition_subset.empty and 'Power' in partition_subset.columns:
                            power_val = float(partition_subset['Power'].iloc[0])
                            power_values_raw.append(power_val)
                            power_values.append(power_val / base_power if base_power else None)
                        else:
                            power_values_raw.append(None)
                            power_values.append(None)
                    
                    # Store raw power values for this frequency
                    power_values_allpowercap_raw[freq] = power_values_raw
                    power_values_allpowercap[freq] = power_values
                    color = colors[freq_idx % len(colors)]
                    line = ax.plot(thread_partitions, power_values, label=f'Freq {freq}MHz', marker='o', linestyle='-', linewidth=2, color=color)
                    # Store legend info only once from the first plot
                    if idx == 0 and f'Freq {freq}MHz' not in legend_labels:
                        legend_handles.append(line[0])
                        legend_labels.append(f'Freq {freq}MHz')
        
        # Plot Throughput-related plots for ax2, ax3, ax4, ax6
        if ax_num in [2, 3, 4, 6]:
            if mode == "powercap":
                for powercap_idx, powercap in enumerate(sorted(data_throughput_dict.keys())):
                    data_throughput = data_throughput_dict[powercap]
                    subset_throughput = data_throughput[(data_throughput['workload1'] == w1) & (data_throughput['workload2'] == w2)]
                    
                    if subset_throughput.empty:
                        continue
                        
                    first_throughput_values = []
                    second_throughput_values = []
                    throughput_sum_values = []
                    throughput_sum_raw = []  # Store raw throughput sum
                    
                    for col in thread_partitions:
                        val = subset_throughput[col].iloc[0]
                        if pd.notna(val) and val != 'None':
                            try:
                                t1 = float(str(val).split(",")[0].replace("(", "").strip())
                                t2 = float(str(val).split(",")[1].replace(")", "").strip())
                                # Sum the raw throughputs
                                t_sum = t1 + t2
                                # Store raw sum
                                throughput_sum_raw.append(t_sum)
                                # Normalize the sum for throughput sum plot
                                t_sum_normalized = t_sum / base_throughput_sum if base_throughput_sum else None
                                # Store individual throughputs if needed
                                first_throughput_values.append(t1 / base_throughput1 if base_throughput1 else None)
                                second_throughput_values.append(t2 / base_throughput2 if base_throughput2 else None)
                                throughput_sum_values.append(t_sum_normalized)
                            except (ValueError, IndexError):
                                first_throughput_values.append(None)
                                second_throughput_values.append(None)
                                throughput_sum_values.append(None)
                                throughput_sum_raw.append(None)
                        else:
                            first_throughput_values.append(None)
                            second_throughput_values.append(None)
                            throughput_sum_values.append(None)
                            throughput_sum_raw.append(None)
                    
                    # Ensure throughput_sum_values length matches power_values
                    if powercap in power_values_allpowercap:
                        assert len(throughput_sum_values) == len(power_values_allpowercap[powercap])
                    
                    # Calculate throughput per watt using original values
                    throughput_per_watt = []
                    base_throughput_per_watt = base_throughput_sum / base_power if base_power else None
                    for i, xput in enumerate(throughput_sum_raw):
                        if xput is not None and powercap in power_values_allpowercap_raw and power_values_allpowercap_raw[powercap][i] is not None:
                            # Use original throughput and power values
                            t_per_w = xput / power_values_allpowercap_raw[powercap][i]
                            # Normalize by base throughput per watt for plotting
                            if base_throughput_per_watt:
                                t_per_w_normalized = t_per_w / base_throughput_per_watt
                                throughput_per_watt.append(t_per_w_normalized)
                            else:
                                throughput_per_watt.append(None)
                        else:
                            throughput_per_watt.append(None)
                    
                    # Plot based on axis number with consistent colors
                    color = colors[powercap_idx % len(colors)]
                    if ax_num == 2:
                        ax.plot(thread_partitions, first_throughput_values, label=f'PowerCap {powercap}W', marker='o', linestyle='-', linewidth=2, color=color)
                    elif ax_num == 3:
                        ax.plot(thread_partitions, second_throughput_values, label=f'PowerCap {powercap}W', marker='o', linestyle='--', linewidth=2, color=color)
                    elif ax_num == 4:
                        ax.plot(thread_partitions, throughput_sum_values, label=f'PowerCap {powercap}W', marker='o', linestyle=':', linewidth=2, color=color)
                    elif ax_num == 6:
                        ax.plot(thread_partitions, throughput_per_watt, label=f'PowerCap {powercap}W', marker='o', linestyle=':', linewidth=2, color=color)
            else:  # freq mode
                for freq_idx, freq in enumerate(sorted(data_throughput_dict.keys())):
                    data_freq = data_throughput_dict[freq]
                    subset_freq = data_freq[(data_freq['workload1'] == w1) & (data_freq['workload2'] == w2)]
                    
                    if subset_freq.empty:
                        continue
                    
                    throughput_sum_values = []
                    throughput_sum_raw = []
                    
                    for partition in thread_partitions:
                        w1_pct = partition.split('_')[0]
                        w2_pct = partition.split('_')[1]
                        # Filter by thread partition
                        partition_subset = subset_freq[subset_freq['Thread_combination'].str.contains(f'w1_{w1_pct}, w2_{w2_pct}')]
                        if not partition_subset.empty and 'weight_Throughput_sum' in partition_subset.columns:
                            throughput_val = float(partition_subset['weight_Throughput_sum'].iloc[0])
                            throughput_sum_raw.append(throughput_val)
                            throughput_sum_values.append(throughput_val / base_throughput_sum if base_throughput_sum else None)
                        else:
                            throughput_sum_raw.append(None)
                            throughput_sum_values.append(None)
                    
                    # Calculate throughput per watt
                    throughput_per_watt = []
                    base_throughput_per_watt = base_throughput_sum / base_power if base_power and base_throughput_sum else None
                    for i, (xput, power) in enumerate(zip(throughput_sum_raw, power_values_allpowercap_raw.get(freq, []))):
                        if xput is not None and power is not None:
                            t_per_w = xput / power
                            if base_throughput_per_watt:
                                throughput_per_watt.append(t_per_w / base_throughput_per_watt)
                            else:
                                throughput_per_watt.append(None)
                        else:
                            throughput_per_watt.append(None)
                    
                    # Plot based on axis number with consistent colors
                    color = colors[freq_idx % len(colors)]
                    if ax_num == 4:  # Throughput sum
                        ax.plot(thread_partitions, throughput_sum_values, label=f'Freq {freq}MHz', marker='o', linestyle=':', linewidth=2, color=color)
                    elif ax_num == 6:  # Throughput per watt
                        ax.plot(thread_partitions, throughput_per_watt, label=f'Freq {freq}MHz', marker='o', linestyle=':', linewidth=2, color=color)
        
        # Set title, labels, and formatting
        fig.suptitle(f'{w1} & {w2}', fontsize=label_font+3)
        if mode == "powercap":
            ax.set_xlabel('Thread Partition', fontsize=axis_font)
            ax.set_xticklabels(thread_partitions, rotation=45)
        else:  # freq mode
            ax.set_xlabel('Thread Partition', fontsize=axis_font)
            ax.set_xticklabels(thread_partitions, rotation=45)
        ax.set_ylabel(axes_info[ax_num][1], fontsize=label_font)
        ax.grid(True)
    
    # Add global legend outside the plots
    if legend_handles:
        legend_title = "PowerCaps" if mode == "powercap" else "Frequencies"
        fig.legend(legend_handles, legend_labels, title=legend_title, 
                  bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=axis_font-1)
    
    # Adjust layout and save the figure
    plt.tight_layout()
    filename_suffix = "dvfs" if mode == "powercap" else "freq"
    plt.savefig(f'./{output_dir}/{w1}_{w2}_custom_power_throughput_plots_{filename_suffix}.png', bbox_inches='tight')
    plt.close(fig)
