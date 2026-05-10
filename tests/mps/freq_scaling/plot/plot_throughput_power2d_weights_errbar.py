import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.patches import FancyBboxPatch

csv_prefix = sys.argv[1]
n_combination = int(sys.argv[2])
baseline_csv = sys.argv[3]
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
avg_stage2 = pd.read_csv(f'../analysis/stage2/freq_scaling/{csv_prefix}_share_comb{n_combination}_freqscale_throughput_individual_avg.csv')
std_stage2 = pd.read_csv(f'../analysis/stage2/freq_scaling/{csv_prefix}_share_comb{n_combination}_freqscale_throughput_individual_std.csv')
power_avg_stage2 = pd.read_csv(f'../analysis/stage2/freq_scaling/{csv_prefix}_share_comb{n_combination}_freqscale_power_avg_stage2.csv')
power_std_stage2 = pd.read_csv(f'../analysis/stage2/freq_scaling/{csv_prefix}_share_comb{n_combination}_freqscale_power_std_stage2.csv')

baseline_df = pd.read_csv(baseline_csv)

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

def calculate_weights(throughput_tuple, weight):
    throughput_tuple = eval(throughput_tuple)  # Convert string to tuple
    
    # Check if both w1 and w2 are not None
    if all(value is not None for value in throughput_tuple):
        try:
            #print(throughput_tuple)
            #caculate weighted throughput as first element of the tuple * weight + rest if the elements of the tuple wuthout weight
            print("weight", weight)
            weighted_throughput = float(throughput_tuple[0]) * weight + sum(throughput_tuple[i] for i in range(1, len(throughput_tuple)))
            #print(weighted_throughput)
            return weighted_throughput

        except ZeroDivisionError:
            return np.nan
    else:
        print(f"has none value in  {throughput_tuple}")
        return None
    


def plot_power_vs_throughput_pareto(workloads, avg_df, std_df, power_avg_df, power_std_df, weights, unique_color = False):
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
    thread_columns = get_thread_columns(workload_data, filter = False)
    print(f"thread_columns used: {thread_columns}")

    plt.figure(figsize=(10, 6))

    #color_map = plt.cm.get_cmap('tab10', len(frequencies))  # Assign distinct colors for each frequency


    
    # Get unique w1 percentages and assign each a color
    unique_w1_percentages = sorted(set(col.split(",")[0].split("_")[1] for col in thread_columns))
    if unique_color:
        color_map = plt.cm.get_cmap('tab10', len(weights))
        weight_colors = {weight: color_map(i) for i, weight in enumerate(weights)}
        #modify weight=1
        weight_colors[1] = 'black'
 
    else:
        color_map = {'default': 'blue'}
        color_dict = {int(w1): 'blue' for i, w1 in enumerate(unique_w1_percentages)}
    
    #print(unique_w1_percentages)
    w1_color_mapping = {}  # Dictionary to hold color for each w1 percentage
    color_w1_mapping = {}  # Dictionary
    # For legend entries, keep track of which labels have been added
    
    pareto_x, pareto_y = [], []  # Store pareto front points
    pareto_x_err, pareto_y_err = [], []
    added_legend_labels = set()
    
    for idx, freq in enumerate(frequencies):
        # Filtering data for the specific frequency
        workload_freq_data = workload_data[workload_data['freq1'] == freq]
        power_freq_data = power_data[power_data['freq1'] == freq]
        workload_std_freq_data = workload_std_data[workload_std_data['freq1'] == freq]
        power_std_freq_data = power_std_data[power_std_data['freq1'] == freq]
        
        # Prepare points for the plot
    # Dictionaries to store values by weight
        pareto_data = {weight: {'x': [], 'y': [], 'x_err': [], 'y_err': []} for weight in weights}
        other_points = {weight: {'x': [], 'y': [], 'x_err': [], 'y_err': [], 'labels': [], 'threads': []} for weight in weights}
        #individual data as individual run
        individual_data = {weight: {'x': [], 'y': [], 'x_err': [], 'y_err': []} for weight in weights}

        w1_labels = []
        thread_comb = []
        colors = []
        base_power_avg, base_throughput_avg = None, None
        all_x_vals, all_y_vals = [], []  # Track all x and y values to calculate limits
        for weight in weights:
            for col in thread_columns:
                # Calculate harmonic mean throughput
                throughput_avg = calculate_weights(workload_freq_data[col].values[0], weight)
                power_avg = power_freq_data[col].values[0]
                throughput_std = calculate_weights(workload_std_freq_data[col].values[0], weight)
                if throughput_std is None or throughput_avg is None:
                    continue
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
                        
                        pareto_data[weight]['x'].append(power_avg)
                        pareto_data[weight]['y'].append(throughput_avg)
                        pareto_data[weight]['x_err'].append(power_std)
                        pareto_data[weight]['y_err'].append(throughput_std)
                        #normalize to 100
                        if weight == 1 or len(weights) == 1:
                            base_power_avg = power_avg
                            base_throughput_avg = throughput_avg
                            
                          
                    else:
                        #if w1_percentage == 20:
                        other_points[weight]['x'].append(power_avg)
                        other_points[weight]['y'].append(throughput_avg)
                        other_points[weight]['x_err'].append(power_std)
                        other_points[weight]['y_err'].append(throughput_std)
                        other_points[weight]['labels'].append(w1_percentage)
                        other_points[weight]['threads'].append(col_text)
                        thread_comb.append(col_text)
                        # Extract w1 percentage and assign color
            
                        w1_labels.append(w1_percentage)

            #add individual data from baseline_df 
            for i, workload in enumerate(workloads):
                #match baselinedf with workload
                baseline_data = baseline_df[baseline_df['Type'] == workload ]
                #catch value error if baseline_data is empty
                if baseline_data.empty:
                    raise ValueError(f"Baseline data not found for workload {workload}")
                    continue
                    
                #get x as power column
                power = baseline_data['power'].values[0]
                throughput = baseline_data['Exclusive100'].values[0]
                print(f"power: {power}, throughput: {throughput}")
                if  i == 0:
                    #power *= weight#not calculate weight for power
                    throughput *= weight
                #get y as throughput column
                #parse into individual data
               
                individual_data[weight]['x'].append(power)
                individual_data[weight]['y'].append(throughput)
                individual_data[weight]['x_err'].append(0)
                individual_data[weight]['y_err'].append(0)
                #

            
            
        #print("xvals", x_vals)
        #print("yvals", y_vals)
        # Plotting the other points with error bars
        #print(f"pareto_x: {pareto_x}")



        #plot individual data
        max_individual_power, max_individual_throughput = -1, -1
        for weight, data in individual_data.items():
            if base_power_avg and base_throughput_avg:
                #plt text before normalization
                #print(data)
                for i in range(len(data['x'])):
                    plt.text(data['x'][i]/ base_power_avg * 100+2, data['y'][i]/ base_throughput_avg * 100, f'({data['x'][i]:.2f}, {data['y'][i]:.2f})', fontsize=9, bbox=dict(facecolor='0.85', alpha=0.5), color='green')
                #plt.text(data['x'], data['y'], f'({data['x']}, {data['y']})', fontsize=9, bbox=dict(facecolor='0.85', alpha=0.5))
                #get max
                max_individual_power = max(data['x'])
                max_individual_throughput = max(data['y'])
                data['x'] = np.array(data['x']) / base_power_avg * 100
                data['y'] = np.array(data['y']) / base_throughput_avg * 100
                data['x_err'] = np.array(data['x_err']) / base_power_avg * 100
                data['y_err'] = np.array(data['y_err']) / base_throughput_avg * 100
                all_x_vals.extend(data['x'])
                all_y_vals.extend(data['y'])
                for i in range(len(data['x'])):            
                    plt.errorbar(data['x'][i], data['y'][i], xerr=data['x_err'][i], yerr=data['y_err'][i],
                            label=f'exclusive{i+1}', fmt='*', capsize=5, markersize=16, 
                            markerfacecolor='none', color='green')
                    #plot text with bbox of the x y value
                    #plt.text(data['x'][i], data['y'][i], f'({data['x'][i]}, {data['y'][i]})', fontsize=9, bbox=dict(facecolor='0.85', alpha=0.5))
        
        # Normalize and plot Pareto points
        for weight, data in pareto_data.items():
            if base_power_avg and base_throughput_avg:
                for i in range(len(data['x'])):
                    #text to float 2 decimal places

                    plt.text(data['x'][i]/ base_power_avg * 100+2, data['y'][i]/ base_throughput_avg * 100, f'({data['x'][i]:.2f}, {data['y'][i]:.2f})\n power/xput= {data['x'][i]/max_individual_power*100:.2f}%/{data['y'][i]/max_individual_throughput*100:.2f}%', fontsize=9, bbox=dict(facecolor='0.85', alpha=0.5), color= "blue")
                data['x'] = np.array(data['x']) / base_power_avg * 100
                data['y'] = np.array(data['y']) / base_throughput_avg * 100
                data['x_err'] = np.array(data['x_err']) / base_power_avg * 100
                data['y_err'] = np.array(data['y_err']) / base_throughput_avg * 100
                all_x_vals.extend(data['x'])
                all_y_vals.extend(data['y'])
                plt.errorbar(data['x'], data['y'], xerr=data['x_err'], yerr=data['y_err'],
                            label=f'100-100% (Weight {weight})', fmt='^', capsize=5, markersize=16, 
                            markerfacecolor='none', color=weight_colors[weight])
                #plt text with bbox of the x y value
                #plt.text(data['x'], data['y'], f'({data["x"]}, {data["y"]})', fontsize=9, bbox=dict(facecolor='0.85', alpha=0.5))

        
        # Normalize and plot other points
        for weight, data in other_points.items():
            if base_power_avg and base_throughput_avg:
                max_throughput_idx = -1
                #find index of max data[y]
                for i, y in enumerate(data['y']):
                    if y == max(data['y']):
                        max_throughput_idx = i
                        break
                #plot text with max data[y]
                if max_throughput_idx != -1:
                    #text plot with bbox
                    plt.text(data['x'][max_throughput_idx]/ base_power_avg * 100+2, data['y'][max_throughput_idx]/ base_throughput_avg * 100, f'({data["x"][max_throughput_idx]:.2f}, {data["y"][max_throughput_idx]:.2f})\n power/xput= {(data['x'][i]/max_individual_power*100):.2f}%/{(data['y'][i]/max_individual_throughput*100):.2f}%', fontsize=9, bbox=dict(facecolor='0.85', alpha=0.5), color="red")

                data['x'] = np.array(data['x']) / base_power_avg * 100
                data['y'] = np.array(data['y']) / base_throughput_avg * 100
                data['x_err'] = np.array(data['x_err']) / base_power_avg * 100
                data['y_err'] = np.array(data['y_err']) / base_throughput_avg * 100
                all_x_vals.extend(data['x'])
                all_y_vals.extend(data['y'])

                # Sort points by thread combination
                sorted_indices = np.argsort(data['threads'])
                data['x'] = data['x'][sorted_indices]
                data['y'] = data['y'][sorted_indices]
                data['x_err'] = data['x_err'][sorted_indices]
                data['y_err'] = data['y_err'][sorted_indices]
                sorted_threads = [data['threads'][i] for i in sorted_indices]

                # Plot lines connecting the points
                plt.plot(data['x'], data['y'], linestyle='-', color=weight_colors[weight])
                
                for i, (x, y, xe, ye, label, thread) in enumerate(zip(data['x'], data['y'], data['x_err'], data['y_err'], data['labels'], data['threads'])):
                    plt.errorbar(x, y, xerr=xe, yerr=ye, fmt='o', capsize=5, color=weight_colors[weight],
                                label=f'Weight {weight}' if i == 0 else None)
                    plt.text(x, y, thread, fontsize=9, rotation=30)
                    


            #plt.errorbar(x_vals, y_vals, xerr=normalized_x_err, yerr=normalized_y_err, fmt='o', color=color, capsize=5)
            
            # Add text labels for each thread combination
            #for x, y, txt in zip(normalized_x_vals, normalized_y_vals, thread_comb):
            #    plt.text(x, y, txt, fontsize=9)
            #plt.errorbar(x_vals, y_vals, xerr=x_err, yerr=y_err, label=f'Freq {freq} MHz', fmt='o', capsize=5, color=color_map(idx))
        #use % instead of absolute value for x and y axis for annotate
        

        """
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

        """



    #add_annotate(x_pos=0.5, y_pos=-0.2,
    #            width=0.2,height=0.8,
    #            text="Abnormal - (70,20,10) and (70,10,20) should be the same")

    # Shade the area under the Pareto front
    #plt.fill_between(pareto_x, pareto_y, np.min(pareto_y), color='black', alpha=0.1)
    # Dynamically set xlim and ylim
    min_x, max_x = min(40, min(all_x_vals, default=40)-2), max(110, max(all_x_vals, default=110)+2)
    min_y, max_y = min(40, min(all_y_vals, default=40)-2), max(110, max(all_y_vals, default=110)+2)
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    plt.title(f'weighted Throughput sum vs Power for \n{" + ".join(list(workloads))}')
    plt.xlabel('Power cp weight=1  100-100 power (%)')
    plt.ylabel('weighted Throughput sum cp. weight=1 (%)')
    #set x limit span to be min-50 and max+50 of power_avg

    #plt.xlim(50, 300)
    #add legend for the color_dict key-values

    plt.legend()
    plt.grid(True)

    # Save the figure
    import os
    #create dir if not exist

    os.makedirs(f"figs/weight_throughputsum_vs_power/{csv_prefix}/{"_".join(list(workloads))}", exist_ok=True)
    weights_str = "_".join([str(w) for w in weights])
    #print(f"figs/weight_throughputsum_vs_power/{csv_prefix}/{"_".join(list(workloads))}/{csv_prefix}_{"_".join(list(workloads))}_weights_{weights_str}_power_vs_throughput.png")
    plt.savefig(f"figs/weight_throughputsum_vs_power/{csv_prefix}/{"_".join(list(workloads))}/{csv_prefix}_{"_".join(list(workloads))}_weights_{weights_str}_power_vs_throughput.png")
    plt.close()


# Workload list (pair of workload1 and workload2) to create separate figures
workload_pairs = avg_stage2[[f'workload{i}' for i in range(1, n_combination+1)]].drop_duplicates()
print(workload_pairs)

# Define weight pairs
#weight_pairs = [[1], [0.1, 1], [0.2, 1], [0.5, 1], [5,1], [10, 1], [100,1]]
weight_pairs = [[1],[0.1], [0.2], [0.5], [5], [10],[30],[100]]
#weight_pairs = [[30]]
# Plot for each workload pair with flipped axes and same color for the same frequency
for weight_pair in weight_pairs:
    for _, workloads in workload_pairs.iterrows():
        #plot_power_vs_throughput_pareto(workloads, avg_stage2, std_stage2, power_avg_stage2, power_std_stage2, weights=[0.1, 0.2, 1, 5, 10] ,unique_color= True)
        #plot weights = [0.1, 1]
        plot_power_vs_throughput_pareto(workloads, avg_stage2, std_stage2, power_avg_stage2, power_std_stage2, weight_pair ,unique_color= True)
