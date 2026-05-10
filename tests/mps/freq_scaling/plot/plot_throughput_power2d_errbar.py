import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.patches import FancyBboxPatch

csv_prefix = sys.argv[1]
n_combination = int(sys.argv[2])
# len of sys.argv is 2
# print args instructions if len(sys.argv) < 2
if len(sys.argv) < 2:
    print("please provide the input_shared_log_dir_prefix n_combination")
    print("python plot_throughput_power2d.py prefix n_combination")
    exit(1)
#len of sys.argv is 2
#print args instructions if len(sys.argv) < 2
if len(sys.argv) < 2:
    print("please provide the input_shared_log_dir_prefix")
    print("python plot_throughput_power2d.py prefix")
    exit(1)
# Load the CSV files1025_share_comb2_freqscale_throughput_sum_avg_stage2
avg_stage2 = pd.read_csv(f'../analysis/stage2/freq_scaling/{csv_prefix}_share_comb{n_combination}_freqscale_throughput_sum_avg_stage2.csv')
std_stage2 = pd.read_csv(f'../analysis/stage2/freq_scaling/{csv_prefix}_share_comb{n_combination}_freqscale_throughput_sum_std_stage2.csv')
power_avg_stage2 = pd.read_csv(f'../analysis/stage2/freq_scaling/{csv_prefix}_share_comb{n_combination}_freqscale_power_avg_stage2.csv')
power_std_stage2 = pd.read_csv(f'../analysis/stage2/freq_scaling/{csv_prefix}_share_comb{n_combination}_freqscale_power_std_stage2.csv')

def add_annotate(x_pos, y_pos, width, height ,text):
    """
    Adds an annotation to the plot with a custom bounding box, where width and height are set relative to the plot range.

    Parameters:
    - x_pos: float, relative x position (0.0 - 1.0 in Axes coordinates)
    - y_pos: float, relative y position (0.0 - 1.0 in Axes coordinates)
    - text: str, annotation text content
    - width_ratio: float, width of the bbox as a fraction of the x-axis range
    - height_ratio: float, height of the bbox as a fraction of the y-axis range
    """
    #put text outside of the plot, below
    # Adjust the plot boundaries to make space for the annotation
    plt.subplots_adjust(bottom=0.2)

    plt.text(
        x_pos, y_pos, text, 
        transform=plt.gca().transAxes,  # Use relative coordinates
        fontsize=12,
        ha='center',  # Horizontal alignment
        va='center',  # Vertical alignment
        bbox=dict(
            boxstyle=f"round,pad=0.3",  # Setting width and height of box
            fc="0.85",  # Fill color
            lw=0.5  # Line width
        )
    )
    

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

            #for 
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

    

def plot_power_vs_throughput_pareto(workloads, avg_df, std_df, power_avg_df, power_std_df):
    # Extracting relevant data from the dataframes
    #workload_data = avg_df[(avg_df['workload1'] == workload1) & (avg_df['workload2'] == workload2)]
    #power_data = power_avg_df[(power_avg_df['workload1'] == workload1) & (power_avg_df['workload2'] == workload2)]
    #workload_std_data = std_df[(std_df['workload1'] == workload1) & (std_df['workload2'] == workload2)]
    #power_std_data = power_std_df[(power_std_df['workload1'] == workload1) & (power_std_df['workload2'] == workload2)]
    
    condition = True
    for i in range(1, n_combination+1):
        condition &= (avg_df[f'workload{i}'] == workloads[f'workload{i}'])
    
    # Apply the filter condition to select relevant rows for all dataframes
    workload_data = avg_df[condition]
    power_data = power_avg_df[condition]
    workload_std_data = std_df[condition]
    power_std_data = power_std_df[condition]
    
    frequencies = workload_data[['freq1']].drop_duplicates().values.flatten()
    thread_columns = get_thread_columns(workload_data, filter = True)
    print(f"thread_columns used: {thread_columns}")

    plt.figure(figsize=(10, 6))

    #color_map = plt.cm.get_cmap('tab10', len(frequencies))  # Assign distinct colors for each frequency


    
    # Get unique w1 percentages and assign each a color
    unique_w1_percentages = sorted(set(col.split(",")[0].split("_")[1] for col in thread_columns))
    color_map = plt.cm.get_cmap('tab10', len(unique_w1_percentages))
    color_dict = {int(w1): color_map(i) for i, w1 in enumerate(unique_w1_percentages)}
    #print(unique_w1_percentages)
    w1_color_mapping = {}  # Dictionary to hold color for each w1 percentage
    color_w1_mapping = {}  # Dictionary
    # For legend entries, keep track of which labels have been added
    
    pareto_x, pareto_y = [], []  # Store pareto front points
    added_legend_labels = set()
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
        w1_labels = []
        thread_comb = []
        colors = []
        
        for col in thread_columns:
            throughput_avg = workload_freq_data[col].values[0]
            power_avg = power_freq_data[col].values[0]
            throughput_std = workload_std_freq_data[col].values[0]
            power_std = power_std_freq_data[col].values[0]
            #text of col
            col_text = col.replace("\"", "").replace("(", "").replace(")", "").replace(" ","").split(",")
            #print(col_texts)
            #remove w1_ and w2_
            col_text = ",".join([c.split("_")[1] for c in col_text])
            w1_percentage = int(col_text.split(",")[0])  # Extract the w1 percentage
            
            
            if not np.isnan(throughput_avg) and not np.isnan(power_avg):
                

                base_percentage = f"({', '.join([f'w{i}_100' for i in range(1, int(n_combination)+1)])})"
                #print(f"base_percentage: {base_percentage}")
                if col == base_percentage:
                    # Special marker for (w1_100, w2_100) point - triangle and bigger size

                    pareto_x.append(power_avg)
                    pareto_y.append(throughput_avg)  # Collect Pareto front points
                    #normalize to 100
                    power_avg = 100
                    throughput_avg = 100
                    plt.errorbar(power_avg, throughput_avg, xerr=power_std, yerr=throughput_std, label=f'Freq {freq} MHz all 100%', fmt='^', capsize=5, markersize=16, markerfacecolor='none', color='black')
                    plt.text(power_avg, throughput_avg, col_text, fontsize=9)
                else:
                    #if w1_percentage == 20:
                    x_vals.append(power_avg)
                    y_vals.append(throughput_avg)
                    x_err.append(power_std)
                    y_err.append(throughput_std)
                    thread_comb.append(col_text)
                    # Extract w1 percentage and assign color
        
                    w1_labels.append(w1_percentage)
                    
        
        # Plotting the other points with error bars
        print(f"pareto_x: {pareto_x}")
        if x_vals and y_vals:
            
            x_vals = np.array(x_vals) / pareto_x[0] * 100
            y_vals = np.array(y_vals) / pareto_y[0] * 100
            x_err = np.array(x_err) / pareto_x[0] * 100
            y_err = np.array(y_err) / pareto_y[0] * 100
            for i, (x, y, xe, ye, label) in enumerate(zip(x_vals, y_vals, x_err, y_err, w1_labels)):
                if label in added_legend_labels:
                    #print(f"added legend label: {added_legend_labels}")
                    plt.errorbar(x, y, xerr=xe, yerr=ye, fmt='o', capsize=5, color=color_dict[label])
                else:
                    added_legend_labels.add(label)
                    plt.errorbar(x, y, xerr=xe, yerr=ye, fmt='o', capsize=5, color=color_dict[label], label=f'w1 {label}%')
            #plt.errorbar(x_vals, y_vals, xerr=normalized_x_err, yerr=normalized_y_err, fmt='o', color=color, capsize=5)
            
            # Add text labels for each thread combination
            #for x, y, txt in zip(normalized_x_vals, normalized_y_vals, thread_comb):
            #    plt.text(x, y, txt, fontsize=9)
            #plt.errorbar(x_vals, y_vals, xerr=x_err, yerr=y_err, label=f'Freq {freq} MHz', fmt='o', capsize=5, color=color_map(idx))
        #use % instead of absolute value for x and y axis for annotate
        


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
            
            plt.text(x, y, sorted_thread_comb[i], fontsize=9, rotation=30)
    # Sort pareto points by power (x-values) and draw Pareto front curve
    pareto_x, pareto_y = zip(*sorted(zip(pareto_x, pareto_y)))
    pareto_x = pareto_x / pareto_x[0] * 100
    pareto_y = pareto_y / pareto_y[0] * 100
    plt.plot(pareto_x, pareto_y, 'k--', label='100-100 curve', linewidth=3)


    #add_annotate(x_pos=0.5, y_pos=-0.2,
    #            width=0.2,height=0.8,
    #            text="Abnormal - (70,20,10) and (70,10,20) should be the same")

    # Shade the area under the Pareto front
    #plt.fill_between(pareto_x, pareto_y, np.min(pareto_y), color='black', alpha=0.1)

    plt.title(f'Throughput sum vs Power with Pareto Front for \n{" + ".join(list(workloads))}')
    plt.xlabel('Power cp 100-100 power (%)')
    plt.ylabel('Throughput sum cp. 100-100 throughput sum (%)')
    #set x limit span to be min-50 and max+50 of power_avg
    plt.xlim(min(min(x_vals), min(pareto_x))-10, max(max(x_vals), max(pareto_x))+10)
    #set ylim=80 if min(y_vals) > 80 
    if min(y_vals) > 90:
        plt.ylim(80, max(max(y_vals), max(pareto_y))+10)
    #plt.xlim(50, 300)
    #add legend for the color_dict key-values

    plt.legend()
    plt.grid(True)

    # Save the figure
    import os
    #create dir if not exist
    os.makedirs(f"figs/throughputsum_vs_power/{csv_prefix}", exist_ok=True)
    plt.savefig(f"figs/throughputsum_vs_power/{csv_prefix}/{csv_prefix}_{"_".join(list(workloads))}_power_vs_throughput.png")
    plt.close()

# Workload list (pair of workload1 and workload2) to create separate figures
workload_pairs = avg_stage2[[f'workload{i}' for i in range(1, n_combination+1)]].drop_duplicates()
print(workload_pairs)

# Plot for each workload pair with flipped axes and same color for the same frequency
for _, workloads in workload_pairs.iterrows():
    plot_power_vs_throughput_pareto(workloads, avg_stage2, std_stage2, power_avg_stage2, power_std_stage2)

