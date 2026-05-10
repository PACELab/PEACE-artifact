import pandas as pd
import matplotlib.pyplot as plt
import os

# Font sizes
axis_font, label_font, legend_font = 22, 22, 16
# Output controls
include_plot_legend = True  # Set True to retain legend inside the main figure
save_legend_only = False  # Set False to skip separate legend image

# File paths
#file_path_power = '~/Downloads/04252022_3freq_nonDL_share_comb2_freqscale_power_avg_stage2.csv'
#file_path_power = "~/Downloads/04252025_3freq_share_comb2_freqscale_power_avg_stage2.csv" 
#file_path_power = "~/Downloads/04222025_3freq_share_comb2_freqscale_power_avg_stage2.csv" 
file_path_power = "/home/cc/mlProfiler/tests/mps/analysis/stage2/freq_motivation/09112025_share_comb2_freqscale_power_avg_stage2.csv"
file_path_power = "/home/cc/mlProfiler/tests/mps/analysis/stage2/v100_socc26/triton_share_comb2_freqscale_power_avg_stage2.csv"

#file_path_throughput = '~/Downloads/04252022_3freq_nonDL_share_comb2_freqscale_throughput_individual_avg.csv'
#file_path_throughput = "~/Downloads/04252025_3freq_share_comb2_freqscale_throughput_individual_avg.csv"
#file_path_throughput = "~/Downloads/04222025_3freq_share_comb2_freqscale_throughput_individual_avg.csv"
#file_path_throughput = "/home/cc/mlProfiler/tests/mps/analysis/stage2/freq_motivation/09112025_share_comb2_freqscale_throughput_individual_avg.csv"
file_path_throughput = "/home/cc/mlProfiler/tests/mps/analysis/stage2/v100_socc26/triton_share_comb2_freqscale_throughput_individual_avg.csv"
output_dir = "v100_rebutaal_socc26"
base_freq = 900  # Will auto-select closest available freq if no exact match
base_percent = 50

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the CSV files
data_power = pd.read_csv(file_path_power)
data_throughput = pd.read_csv(file_path_throughput)

# Function to create customized subfigures
def create_custom_subfigs(selected_axes):
    axes_map = {
        1: ('Power vs SM partition', 'Power (W)'),
        2: ('1st Throughput vs SM partition', 'Throughput1'),
        3: ('2nd Throughput vs SM partition', 'Throughput2'),
        4: ('Throughput Sum vs SM partition', 'Normalized throughput sum'),
        6: ('Throughput per Watt vs SM partition', 'Throughput per Watt')
    }
    
    n_plots = len(selected_axes)
    fig, axes = plt.subplots(1, n_plots, figsize=(9 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    return fig, axes, {i: axes_map[i] for i in selected_axes}

# Get unique workload pairs
workload_pairs = data_power[['workload1', 'workload2']].drop_duplicates()

# Specify which axes to plot
selected_axes = [1, 2,3, 4, 6]
selected_axes = [4]

# Loop through each workload pair
for _, row in workload_pairs.iterrows():
    w1 = row['workload1']
    w2 = row['workload2']
    
    # Filter the data for the specific workload pair
    subset_vit_whisper_power = data_power[(data_power['workload1'] == w1) & (data_power['workload2'] == w2)]
    subset_vit_whisper_throughput = data_throughput[(data_throughput['workload1'] == w1) & (data_throughput['workload2'] == w2)]
    
    # Create custom subfigures
    fig, axes, axes_info = create_custom_subfigs(selected_axes)
    
    # Extract SM partition columns
    thread_partitions = [col for col in subset_vit_whisper_power.columns if 'w1' in col and "100" not in col]
    base_partition = [col for col in subset_vit_whisper_power.columns if f"{base_percent}" in col][0]
    print(f"Base partition: {base_partition}")

    # Auto-select closest available frequency if base_freq is not in the data
    available_freqs = sorted(subset_vit_whisper_power['freq1'].unique())
    if base_freq not in available_freqs:
        effective_base_freq = min(available_freqs, key=lambda f: abs(f - base_freq))
        print(f"Warning: base_freq {base_freq} not found, using closest available: {effective_base_freq}")
    else:
        effective_base_freq = base_freq
    
    # Store legend handles and labels for global legend
    legend_handles = []
    legend_labels = []
    
    # Define unique colors with good contrast for different frequencies
    colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9A6324']

    # Get base throughput values
    base_throughput = subset_vit_whisper_throughput[subset_vit_whisper_throughput["freq1"] == effective_base_freq][base_partition]
    if not base_throughput.empty:
        base_throughput = base_throughput.values[0]
        if "None" in base_throughput:
            base_throughput = None
            base_throughput1 = None
            base_throughput2 = None
            base_throughput_sum = None
        else:
            base_throughput1 = float(base_throughput.split(",")[0].replace("(", "").strip())
            base_throughput2 = float(base_throughput.split(",")[1].replace(")", "").strip())
            base_throughput_sum = base_throughput1 + base_throughput2
    else:
        base_throughput = None
        base_throughput1 = None
        base_throughput2 = None
        base_throughput_sum = None

    # Get base power value
    base_power = subset_vit_whisper_power[subset_vit_whisper_power["freq1"] == effective_base_freq][base_partition]
    if not base_power.empty:
        base_power = float(base_power.values[0])
    else:
        base_power = None

    print(f"Base throughput for {w1} & {w2} at {effective_base_freq} MHz: {base_throughput}")
    print(f"Base throughput sum for {w1} & {w2} at {effective_base_freq} MHz: {base_throughput_sum}")
    print(f"Base power for {w1} & {w2} at {effective_base_freq} MHz: {base_power}")
    if base_throughput_sum is None or base_power is None:
        print(f"Skipping {w1} & {w2} due to missing base values.")
        continue

    power_values_allfreq = {}
    power_values_allfreq_raw = {}  # Store raw power values

    # Always process power data first (needed for throughput calculations)
    for freq_idx, freq in enumerate(sorted(subset_vit_whisper_power['freq1'].unique())):
        subsubset_vit_whisper_power = subset_vit_whisper_power[subset_vit_whisper_power['freq1'] == freq]
        power_values = [
            float(subsubset_vit_whisper_power[col].iloc[0]) if pd.notna(subsubset_vit_whisper_power[col].iloc[0]) else None
            for col in thread_partitions
        ]
        # Store raw power values
        power_values_allfreq_raw[freq] = power_values.copy()
        # Normalize power values
        power_values_norm = [x / base_power if x is not None else None for x in power_values]
        power_values_allfreq[freq] = power_values_norm

    # Plot data for each selected axis
    for idx, ax_num in enumerate(selected_axes):
        ax = axes[idx]

        # Plot Power vs SM partition (ax1)
        if ax_num == 1:
            for freq_idx, freq in enumerate(sorted(subset_vit_whisper_power['freq1'].unique())):
                # Use already-processed power values
                power_values_norm = power_values_allfreq[freq]
                color = colors[freq_idx % len(colors)]
                line = ax.plot(
                    thread_partitions,
                    power_values_norm,
                    label=f'{freq}MHz',
                    linestyle='-',
                    linewidth=4,
                    color=color,
                    zorder=2,
                )
                ax.plot(
                    thread_partitions,
                    power_values_norm,
                    linestyle='None',
                    marker='o',
                    markersize=18,
                    color=color,
                    markeredgecolor='white',
                    markeredgewidth=1.0,
                    zorder=3,
                )
                # Store legend info from the first plot
                if idx == 0:
                    legend_handles.append(line[0])
                    legend_labels.append(f'{freq}MHz')
        
        # Plot Throughput-related plots for ax2, ax3, ax4, ax6
        if ax_num in [2, 3, 4, 6]:
            print(subset_vit_whisper_throughput['freq1'].unique())
            for freq_idx, freq in enumerate(sorted(subset_vit_whisper_throughput['freq1'].unique())):

                subsubset_vit_whisper_throughput = subset_vit_whisper_throughput[subset_vit_whisper_throughput['freq1'] == freq]
                first_throughput_values = []
                second_throughput_values = []
                throughput_sum_values = []
                throughput_sum_raw = []  # Store raw throughput sum
                for col in thread_partitions:
                    val = subsubset_vit_whisper_throughput[col].iloc[0]
                    if pd.notna(val) and val != 'None':
                        try:
                            t1 = float(val.split(",")[0].replace("(", "").strip())
                            t2 = float(val.split(",")[1].replace(")", "").strip())
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
                assert len(throughput_sum_values) == len(power_values_allfreq[freq])
                
                # Calculate throughput per watt using original values
                throughput_per_watt = []
                base_throughput_per_watt = base_throughput_sum / base_power if base_power else None
                for i, xput in enumerate(throughput_sum_raw):
                    if xput is not None and power_values_allfreq_raw[freq][i] is not None:
                        # Use original throughput and power values
                        t_per_w = xput / power_values_allfreq_raw[freq][i]
                        # Normalize by base throughput per watt for plotting
                        if base_throughput_per_watt:
                            t_per_w_normalized = t_per_w / base_throughput_per_watt
                            throughput_per_watt.append(t_per_w_normalized)
                        else:
                            throughput_per_watt.append(None)
                    else:
                        throughput_per_watt.append(None)
                
                # Plot based on axis number with consistent colors
                color = colors[freq_idx % len(colors)]
                if ax_num == 2:
                    line = ax.plot(
                        thread_partitions,
                        first_throughput_values,
                        label=f'{freq}MHz',
                        linestyle='-',
                        linewidth=4,
                        color=color,
                        zorder=2,
                    )
                    ax.plot(
                        thread_partitions,
                        first_throughput_values,
                        linestyle='None',
                        marker='o',
                        markersize=18,
                        color=color,
                        markeredgecolor='white',
                        markeredgewidth=1.0,
                        zorder=3,
                    )
                    # Store legend info from the first plot
                    if idx == 0:
                        legend_handles.append(line[0])
                        legend_labels.append(f'{freq}MHz')
                elif ax_num == 3:
                    line = ax.plot(
                        thread_partitions,
                        second_throughput_values,
                        label=f'{freq}MHz',
                        linestyle='--',
                        linewidth=4,
                        color=color,
                        zorder=2,
                    )
                    ax.plot(
                        thread_partitions,
                        second_throughput_values,
                        linestyle='None',
                        marker='o',
                        markersize=18,
                        color=color,
                        markeredgecolor='white',
                        markeredgewidth=1.2,
                        zorder=3,
                    )
                    # Store legend info from the first plot
                    if idx == 0:
                        legend_handles.append(line[0])
                        legend_labels.append(f'{freq}MHz')
                elif ax_num == 4:
                    # First plot the line without markers
                    line = ax.plot(
                        thread_partitions,
                        throughput_sum_values,
                        label=f'{freq}MHz',
                        linestyle=':',
                        linewidth=4,
                        color=color,
                        zorder=2,
                    )
                    # Store legend info from the first plot
                    if idx == 0:
                        legend_handles.append(line[0])
                        legend_labels.append(f'{freq}MHz')
                    # Find the index of maximum throughput sum
                    valid_values = [(i, val) for i, val in enumerate(throughput_sum_values) if val is not None]
                    if valid_values:
                        max_idx, max_val = max(valid_values, key=lambda x: x[1])
                        # Plot only the maximum point with marker
                        ax.plot(
                            [thread_partitions[max_idx]],
                            [max_val],
                            linestyle='None',
                            marker='o',
                            markersize=18,
                            color=color,
                            markeredgecolor='white',
                            markeredgewidth=1.2,
                            zorder=4,
                        )
                elif ax_num == 6:
                    line = ax.plot(
                        thread_partitions,
                        throughput_per_watt,
                        label=f'{freq}MHz',
                        linestyle='-',
                        linewidth=4,
                        color=color,
                        zorder=2,
                    )
                    ax.plot(
                        thread_partitions,
                        throughput_per_watt,
                        linestyle='None',
                        marker='o',
                        markersize=18,
                        color=color,
                        markeredgecolor='white',
                        markeredgewidth=1.0,
                        zorder=3,
                    )
                    # Store legend info from the first plot
                    if idx == 0:
                        legend_handles.append(line[0])
                        legend_labels.append(f'{freq}MHz')
        
        # Set title, labels, and formatting
        #fig.suptitle(f'{w1} & {w2}', fontsize=label_font+3)
        #ax.set_xlabel('SM partition', fontsize=axis_font-3)
        ax.set_ylabel(axes_info[ax_num][1], fontsize=label_font-3)
        ax.grid(True)
        # Create clean x-axis labels by removing w1_, w2_ prefixes and keeping numeric values
        clean_labels = []
        for label in thread_partitions:
            # Extract numbers from labels like 'w1_10_w2_90' -> '(10,90)'
            import re
            # Use regex to find numbers that come after w1_ and w2_
            w1_match = re.search(r'w1_(\d+)', label)
            w2_match = re.search(r'w2_(\d+)', label)
            if w1_match and w2_match:
                w1_val = w1_match.group(1)
                w2_val = w2_match.group(1)
                clean_labels.append(f"({w1_val}-{w2_val})")
            else:
                clean_labels.append(label)  # fallback to original if pattern doesn't match
        ax.set_xticks(range(len(thread_partitions)))
        ax.set_xticklabels(clean_labels, rotation=45, fontsize=axis_font-3)
        ax.tick_params(axis='y', labelsize=axis_font-3)
        #print(clean_labels)
    # Add legend at the top of the plot
    if legend_handles:
        if include_plot_legend:
            fig.legend(
                legend_handles,
                legend_labels,
                bbox_to_anchor=(0.98, 0.6),
                loc='center left',
                ncol=1,
                fontsize=legend_font,
                columnspacing=0.6,
                handletextpad=0.5,
                handlelength=1.0,
                borderaxespad=0.0
            )
        if save_legend_only:
            legend_height = max(1.2, 0.28 * len(legend_labels))
            legend_fig, legend_ax = plt.subplots(figsize=(2.4, legend_height))
            legend_ax.axis('off')
            legend_ax.legend(
                legend_handles,
                legend_labels,
                loc='center left',
                ncol=1,
                fontsize=legend_font,
                frameon=False,
                columnspacing=0.6,
                handletextpad=0.6,
                handlelength=1.0
            )
            legend_fig.savefig(
                f'./{output_dir}/{w1}_{w2}_frequency_legend.png',
                bbox_inches='tight'
            )
            plt.close(legend_fig)
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f'./{output_dir}/{w1}_{w2}_custom_power_throughput_plots.png', bbox_inches='tight')
    plt.close(fig)
