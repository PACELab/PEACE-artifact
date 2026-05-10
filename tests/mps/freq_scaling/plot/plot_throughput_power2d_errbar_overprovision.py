import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

csv_prefix = sys.argv[1]
#len of sys.argv is 2
#print args instructions if len(sys.argv) < 2
if len(sys.argv) < 2:
    print("please provide the input_shared_log_dir_prefix")
    print("python plot_throughput_power2d.py prefix")
    exit(1)
# Load the CSV files1025_share_comb2_freqscale_throughput_sum_avg_stage2
avg_stage2 = pd.read_csv(f'../analysis/stage2/freq_scaling/{csv_prefix}_share_comb2_freqscale_throughput_sum_avg_stage2.csv')
std_stage2 = pd.read_csv(f'../analysis/stage2/freq_scaling/{csv_prefix}_share_comb2_freqscale_throughput_sum_std_stage2.csv')
power_avg_stage2 = pd.read_csv(f'../analysis/stage2/freq_scaling/{csv_prefix}_share_comb2_freqscale_power_avg_stage2.csv')
power_std_stage2 = pd.read_csv(f'../analysis/stage2/freq_scaling/{csv_prefix}_share_comb2_freqscale_power_std_stage2.csv')

def get_thread_columns(workload_data, filter = False):
    all_threads  = [col for col in workload_data.columns if 'w1_' in col]
    if not filter:
        #default - get all thread columns
        return all_threads
    else:
        #custom filter
        unique_combinations = {}
        for col in all_threads:
            #strip  col.replace("\"", "").replace("(", "").replace(")", "").replace(" ","").split(",") in all_threads
            col_s = col.replace("\"", "").replace("(", "").replace(")", "").replace(" ","").split(",")
            col_s = [c.split("_")[1] for c in col_s]

            
            thread_tuple = tuple(col_s)

            #print(thread_tuple)
            # Use the sorted tuple as a key to ensure uniqueness
            if thread_tuple not in unique_combinations:
                unique_combinations[thread_tuple] = col
            

        #keep keys in unique_combinations if it is in sum_100_combs or over_100_combs
        custom_unique_comb = {}
        #add 100,100
        custom_unique_comb[(str(100), str(100))] = unique_combinations[(str(100), str(100))]

        for k in range(0, 101, 10):
            
            if (str(k), str(100-k)) in unique_combinations and (str(k), str(100)) in unique_combinations:
                custom_unique_comb[(str(k), str(100-int(k)))] = unique_combinations[(str(k), str(100-int(k)))]
                custom_unique_comb[(str(k), str(100))] = unique_combinations[(str(k), str(100))]

        #unique_combinations = {k: v for k, v in unique_combinations.items() if k in sum_100_combs or k in over_100_combs}
        #print(unique_combinations.keys())
        #print(custom_unique_comb.keys())
        #exit(1)
        #return list(unique_combinations.values())
        return list(custom_unique_comb.values())
def plot_power_vs_throughput_pareto(workload1, workload2, avg_df, std_df, power_avg_df, power_std_df):
    # Extracting relevant data from the dataframes
    workload_data = avg_df[(avg_df['workload1'] == workload1) & (avg_df['workload2'] == workload2)]
    power_data = power_avg_df[(power_avg_df['workload1'] == workload1) & (power_avg_df['workload2'] == workload2)]
    workload_std_data = std_df[(std_df['workload1'] == workload1) & (std_df['workload2'] == workload2)]
    power_std_data = power_std_df[(power_std_df['workload1'] == workload1) & (power_std_df['workload2'] == workload2)]
    
    frequencies = workload_data[['freq1']].drop_duplicates().values.flatten()
    #thread_columns = [col for col in workload_data.columns if 'w1_' in col]
    thread_columns = get_thread_columns(workload_data, filter=True)
    plt.figure(figsize=(10, 6))

    color_map = plt.cm.get_cmap('tab10', len(frequencies))  # Assign distinct colors for each frequency
    pareto_x, pareto_y = [], []  # Store pareto front points

    for idx, freq in enumerate(frequencies):
        # Filtering data for the specific frequency
        workload_freq_data = workload_data[workload_data['freq1'] == freq]
        power_freq_data = power_data[power_data['freq1'] == freq]
        workload_std_freq_data = workload_std_data[workload_std_data['freq1'] == freq]
        power_std_freq_data = power_std_data[power_std_data['freq1'] == freq]
        
        # Prepare points for the plot
        complement_x_vals, overprovision_x_vals = [], []
        complement_y_vals, overprovision_y_vals = [], []
        complement_x_err, overprovision_x_err = [], []
        complement_y_err, overprovision_y_err = [], []
        x_vals = []
        y_vals = []
        x_err = []
        y_err = []
        thread_comb = []
        
        for col in thread_columns:
            throughput_avg = workload_freq_data[col].values[0]
            power_avg = power_freq_data[col].values[0]
            throughput_std = workload_std_freq_data[col].values[0]
            power_std = power_std_freq_data[col].values[0]
            #text of col
            col_texts = col.replace("\"", "").replace("(", "").replace(")", "").replace(" ","").split(",")
            #print(col_texts)
            #remove w1_ and w2_
            col_text = ",".join([c.split("_")[1] for c in col_texts])
            sum_threads = sum([int(c.split("_")[1]) for c in col_texts])
            if not np.isnan(throughput_avg) and not np.isnan(power_avg):
                if col == '(w1_100, w2_100)':
                    # Special marker for (w1_100, w2_100) point - triangle and bigger size
                    
                    plt.errorbar(power_avg/power_avg * 100, throughput_avg/throughput_avg * 100, xerr=power_std, yerr=throughput_std, label=f'Freq {freq} MHz (w1_100, w2_100)', fmt='^', capsize=5, markersize=16, markerfacecolor='none', color=color_map(idx))
                    pareto_x.append(power_avg)
                    pareto_y.append(throughput_avg)  # Collect Pareto front points
                    plt.text(power_avg/power_avg * 100, throughput_avg/throughput_avg * 100, col_text, fontsize=9)
                else:
                    x_vals.append(power_avg)
                    y_vals.append(throughput_avg)
                    if sum_threads == 100:
                       complement_x_vals.append(power_avg)
                       complement_y_vals.append(throughput_avg)
                       complement_x_err.append(power_std)
                       complement_y_err.append(throughput_std)
                    elif sum_threads > 100:
                        overprovision_x_vals.append(power_avg)
                        overprovision_y_vals.append(throughput_avg)
                        overprovision_x_err.append(power_std)
                        overprovision_y_err.append(throughput_std)
                    x_err.append(power_std)
                    y_err.append(throughput_std)
                    thread_comb.append(col_text)
        
        if len(pareto_x) > 1:
            raise ValueError("Pareto front should have only one point. multiple frequencies are not considered for normalization")
        # Plotting the other points with error bars
        #normalized x_vals, y_vals, 
        x_vals = x_vals / pareto_x[0] * 100
        x_err = x_err / pareto_x[0] * 100
        y_vals = y_vals / pareto_y[0] * 100
        y_err = y_err / pareto_y[0] * 100
        if complement_x_vals and complement_y_vals:
            complement_x_vals = complement_x_vals / pareto_x[0] * 100
            complement_x_err = complement_x_err / pareto_x[0] * 100
            complement_y_vals = complement_y_vals / pareto_y[0] * 100
            complement_y_err = complement_y_err / pareto_y[0] * 100
            plt.errorbar(complement_x_vals, complement_y_vals, xerr=complement_x_err, yerr=complement_y_err, label=f'sum thread - 100%', fmt='o', capsize=5, color="blue")
        if overprovision_x_vals and overprovision_y_vals:
            overprovision_x_vals = overprovision_x_vals / pareto_x[0] * 100
            overprovision_x_err = overprovision_x_err / pareto_x[0] * 100
            overprovision_y_vals = overprovision_y_vals / pareto_y[0] * 100
            overprovision_y_err = overprovision_y_err / pareto_y[0] * 100
            plt.errorbar(overprovision_x_vals, overprovision_y_vals, xerr=overprovision_x_err, yerr=overprovision_y_err, label=f'sum thread > 100%', fmt='o', capsize=5, color="red")
    
    

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
    pareto_x, pareto_y = zip(*sorted(zip(pareto_x, pareto_y)))
    plt.plot(pareto_x, pareto_y/pareto_y[0], 'k--', label='100-100 curve', linewidth=3)
    
    # Shade the area under the Pareto front
    #plt.fill_between(pareto_x, pareto_y, np.min(pareto_y), color='black', alpha=0.1)

    plt.title(f'Throughput sum (divided by 100-100) vs Power for {workload1} + {workload2}')
    plt.xlabel('Power cp 100-100 power (%)')
    plt.ylabel('Throughput sum cp. 100-100 throughput sum (%)')
    #set x limit span to be min-50 and max+50 of power_avg
    plt.xlim(min(min(x_vals), min(pareto_x))-10, max(max(x_vals), max(pareto_x))+10)    #plt.xlim(50, 300)
    plt.legend()
    plt.grid(True)

    # Save the figure
    import os
    #create dir if not exist
    os.makedirs(f"figs/throughputsum_vs_power/{csv_prefix}", exist_ok=True)
    plt.savefig(f"figs/throughputsum_vs_power/{csv_prefix}/{csv_prefix}_{workload1}_{workload2}_power_vs_throughput.png")
    plt.close()

# Workload list (pair of workload1 and workload2) to create separate figures
workload_pairs = avg_stage2[['workload1', 'workload2']].drop_duplicates()

# Plot for each workload pair with flipped axes and same color for the same frequency
for _, row in workload_pairs.iterrows():
    plot_power_vs_throughput_pareto(row['workload1'], row['workload2'], avg_stage2, std_stage2, power_avg_stage2, power_std_stage2)

