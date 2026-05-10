import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys 
import os
input_prefix = "../analysis/stage2/freq_scaling"
csv_prefix = sys.argv[1]
# Load the CSV files
throughput_avg_file = f'{input_prefix}/1022_share_comb2_freqscale_throughput_individual_avg.csv'
throughput_std_file = f'{input_prefix}/1022_share_comb2_freqscale_throughput_individual_std.csv'
baseline_file = "../analysis/system_metrics/baseline_metrics.csv"
power_avg_file = f'{input_prefix}/1022_share_comb2_freqscale_power_avg_stage2.csv'
power_std_file = f'{input_prefix}/1022_share_comb2_freqscale_power_std_stage2.csv'

# Read CSV files into pandas DataFrames
throughput_avg_df = pd.read_csv(throughput_avg_file)
throughput_std_df = pd.read_csv(throughput_std_file)
power_avg_df = pd.read_csv(power_avg_file)
power_std_df = pd.read_csv(power_std_file)
baseline_df = pd.read_csv(baseline_file)

# Calculate harmonic mean throughput using formula: 2 * w1 * w2 / (w1 + w2)
def harmonic_mean(throughput_tuple, base_tuple):
    throughput_tuple = eval(throughput_tuple)  # Convert string to tuple
    
    # Check if both w1 and w2 are not None
    if all(value is not None for value in throughput_tuple):
        try:
            print(throughput_tuple)
            print(f"base_tuple: {base_tuple}")
            #throughput_tuple = tuple(throughput_tuple[i]/base_tuple[i] for i in range(len(throughput_tuple)))
            #print("normalized: ", throughput_tuple)
            return len(throughput_tuple) * np.prod(throughput_tuple) / sum(throughput_tuple)
        except ZeroDivisionError:
            return np.nan
    else:
        return np.nan  # Return None if any value in the tuple is None

def plot_power_vs_throughput_fscore(workload1, workload2, avg_df, std_df, power_avg_df, power_std_df):
    # Extracting relevant data from the dataframes
    workload_data = avg_df[(avg_df['workload1'] == workload1) & (avg_df['workload2'] == workload2)]
    power_data = power_avg_df[(power_avg_df['workload1'] == workload1) & (power_avg_df['workload2'] == workload2)]
    workload_std_data = std_df[(std_df['workload1'] == workload1) & (std_df['workload2'] == workload2)]
    power_std_data = power_std_df[(power_std_df['workload1'] == workload1) & (power_std_df['workload2'] == workload2)]
    #get baseline data
    base_steps = (float(baseline_df[baseline_df["Type"].str.contains(workload1)]["Exclusive100"]), float(baseline_df[baseline_df["Type"].str.contains(workload2)]["Exclusive100"]))

    frequencies = workload_data[['freq1']].drop_duplicates().values.flatten()
    thread_columns = [col for col in workload_data.columns if 'w1_' in col]

    plt.figure(figsize=(11, 6))

    color_map = plt.cm.get_cmap('tab10', len(frequencies))  # Assign distinct colors for each frequency
    pareto_x, pareto_y = [], []  # Store pareto front points

    for idx, freq in enumerate(frequencies):
        # Filtering data for the specific frequency
        workload_freq_data = workload_data[workload_data['freq1'] == freq]
        power_freq_data = power_data[power_data['freq1'] == freq]
        workload_std_freq_data = workload_std_data[workload_std_data['freq1'] == freq]
        power_std_freq_data = power_std_data[power_std_data['freq1'] == freq]
        
        # Prepare points for the plot
        x_vals = []
        y_vals = []
        x_err = []
        y_err = []
        thread_comb = []
        
        for col in thread_columns:
            # Calculate harmonic mean throughput
            throughput_avg = harmonic_mean(workload_freq_data[col].values[0], base_steps)
            #eval th
            
            power_avg = power_freq_data[col].values[0]
            #throughput_std = workload_std_freq_data[col].values[0]
            power_std = power_std_freq_data[col].values[0]

            
            #text of col
            col_text = col.replace("\"", "").replace("(", "").replace(")", "").replace(" ","").split(",")
            #print(col_texts)
            #remove w1_ and w2_
            col_text = ",".join([c.split("_")[1] for c in col_text])
            
            #col_text = [c.split("_")[1] for c in col].join(",")
            
            if not np.isnan(throughput_avg) and not np.isnan(power_avg):
                if col == '(w1_100, w2_100)':
                    # Special marker for (w1_100, w2_100) point - triangle and bigger size
                    plt.errorbar(power_avg, throughput_avg, xerr=power_std, label=f'Freq {freq} MHz (w1_100, w2_100)', fmt='^', capsize=5, markersize=16, markerfacecolor='none', color=color_map(idx))
                    pareto_x.append(power_avg)
                    pareto_y.append(throughput_avg)  # Collect Pareto front points
                    #plot col
                    plt.text(power_avg, throughput_avg, col_text, fontsize=9)
                else:
                    x_vals.append(power_avg)
                    y_vals.append(throughput_avg)
                    x_err.append(power_std)
                    thread_comb.append(col_text)
                    #plot text at the position
                    #plt.text(power_avg, throughput_avg, col_text, fontsize=9)
                    #y_err.append(throughput_std)
        
        # Plotting the other points with error bars
        if x_vals and y_vals:
            plt.errorbar(x_vals, y_vals, xerr=x_err, label=f'Freq {freq} MHz', fmt='o-', capsize=5, color=color_map(idx))



        sorted_val_indices = sorted(range(len(x_vals)), key=lambda k: x_vals[k])
        # Plot text for each point outside pareto fronyt
        x_vals = [x_vals[i] for i in sorted_val_indices]
        y_vals = [y_vals[i] for i in sorted_val_indices]
        sorted_thread_comb = [thread_comb[i] for i in sorted_val_indices]
        delta_y = 0.01  # Small delta to adjust y-coordinate
        x_margin = 5  # Margin to check if two points are close
        y_margin = 0.01  # Margin to check if two points are close
        for i, (x, y) in enumerate(zip(x_vals, y_vals)):
            #print(f"x, y: {x}, {y}")
            if i > 0 and abs(x_vals[i] - x_vals[i-1]) < x_margin and abs(y_vals[i] - y_vals[i-1]) < y_margin:
                print(f"adjust y for {sorted_thread_comb[i]}, {x}, {y}")
                y += delta_y
            plt.text(x, y, sorted_thread_comb[i], fontsize=9)
    # Sort pareto points by power (x-values) and draw Pareto front curve
    sorted_pareto_indices = sorted(range(len(pareto_x)), key=lambda k: pareto_x[k])
    pareto_x = [pareto_x[i] for i in sorted_pareto_indices]
    pareto_y = [pareto_y[i] for i in sorted_pareto_indices]
    plt.plot(pareto_x, pareto_y, 'k--', label='100-100 curve', linewidth=3)

    # Shade the area under the Pareto front
    #plt.fill_between(pareto_x, pareto_y, np.min(pareto_y), color='black', alpha=0.1)

    plt.title(f'Throughput f-score vs Power with Pareto Front for {workload1} + {workload2}')
    plt.xlabel('Power (Watts)')
    plt.ylabel('Throughput f-score')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5), borderaxespad=0., fontsize='small')
    plt.grid(True)
    #only set x upper limit
    #plt.xlim(0, max(max(x_vals), max(pareto_x))+10)    #plt.xlim(50, 300)
    plt.xlim(min(min(x_vals), min(pareto_x))-20, max(max(x_vals), max(pareto_x))+10)    #plt.xlim(50, 300)
    # Adjust layout to fit the legend
    plt.tight_layout(rect=[0, 0, 0.9, 1])


    # Save the figure
    os.makedirs(f"figs/throughput_fscore_vs_power/{csv_prefix}", exist_ok=True)
    plt.savefig(f"figs/throughput_fscore_vs_power/{csv_prefix}/{csv_prefix}_{workload1}_{workload2}_notnorm_power_vs_throughput.png")
    plt.close()

# Workload list (pair of workload1 and workload2) to create separate figures
workload_pairs = throughput_avg_df[['workload1', 'workload2']].drop_duplicates()

# Plot for each workload pair with flipped axes and same color for the same frequency
for _, row in workload_pairs.iterrows():
    plot_power_vs_throughput_fscore(row['workload1'], row['workload2'],throughput_avg_df, throughput_std_df, power_avg_df, power_std_df)