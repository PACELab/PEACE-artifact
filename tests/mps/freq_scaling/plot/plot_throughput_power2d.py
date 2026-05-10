import pandas as pd
import matplotlib.pyplot as plt

# Define a dictionary to assign fixed colors to each frequency
frequency_colors = {
    1147: 'blue',
    1335: 'green',
    1530: 'orange',
    952: 'red'
}

def shorten_label(label):
    filename_type_dict = {'wav2vec2-base-960h' : 'Wav2vec2', 'bert-base-cased': 'BERT',
                      'mobilenet': 'mobile', 'mobilenet_v2_1.0_224' : 'mobile','vit' : 'ViT', 
                      'vit_h_14' : 'ViT', 'whisper-large-v2': 'Whisper', 'vit-base-patch16-224' : "ViT",
                      'albert-base-v2' : 'ALBERT'}
    for key, value in filename_type_dict.items():
        if key in label:
            batch_type = label.rsplit('_', 1)[1]
            batch_type = batch_type.replace('batch', 'b')
            return value+ '_' + batch_type
    return label
file_steps_path = '/home/cc/mlProfiler/tests/mps/analysis/stage2/1015_smallset_share_comb2_freqscale_stage2.csv'
file_power_path = '/home/cc/mlProfiler/tests/mps/analysis/stage2/1015_smallset_share_comb2_freqscale_stage2_power.csv'

# Load the throughput and power datasets
data = pd.read_csv(file_steps_path)
power_data = pd.read_csv(file_power_path)

# Function to clean and extract throughput values for plotting
def extract_throughput(row):
    # Keep the order of percentages fixed
    percentages = ['(w1_100, w2_100)', '(w1_10, w2_90)', '(w1_20, w2_80)', '(w1_30, w2_70)', 
                   '(w1_40, w2_60)', '(w1_50, w2_50)', '(w1_60, w2_40)', '(w1_70, w2_30)', 
                   '(w1_80, w2_20)', '(w1_90, w2_10)']
    
    w1_throughput = []
    w2_throughput = []
    valid_percentages = []
    
    # Extract throughput values in the fixed order
    for pct in percentages:
        value = row[pct]
        if value != '(None, None)':
            w1, w2 = eval(value)
            w1_throughput.append(w1)
            w2_throughput.append(w2)
            valid_percentages.append(pct)
    
    return valid_percentages, w1_throughput, w2_throughput

# Function to extract both throughput sum and average power for each colocation pair
def extract_throughput_and_power(index):
    throughput_row = data.iloc[index]
    power_row = power_data.iloc[index]

    percentages, w1_throughput, w2_throughput = extract_throughput(throughput_row)
    sum_throughput = [w1 + w2 for w1, w2 in zip(w1_throughput, w2_throughput)]
    freq = data.iloc[index]['freq1']
    power_freq = power_data.iloc[index]['freq1']
    assert freq == power_freq, f"Frequency mismatch: {freq} != {power_freq}"
    
    w1_power = []
    for pct in percentages:
        value = power_row[pct]
        if not pd.isna(value):
            w1_power.append(value)
    
    avg_power = [p for p in w1_power]
    
    return percentages, sum_throughput, avg_power, freq

# Function to normalize values by dividing them by the 100,100 setup values
def normalize_values(sum_throughput, avg_power):
    if sum_throughput[0] != 0:
        norm_throughput = [val / sum_throughput[0] for val in sum_throughput]
    else:
        norm_throughput = sum_throughput
    
    if avg_power[0] != 0:
        norm_power = [val / avg_power[0] for val in avg_power]
    else:
        norm_power = avg_power
    
    return norm_throughput, norm_power

# Group the data by workload pairs, then plot different frequencies in one graph
workload_pairs = data.groupby(['workload1', 'workload2'])

for (workload1, workload2), group in workload_pairs:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    for index, row in group.iterrows():
        shorten_w1 = shorten_label(workload1)
        shorten_w2 = shorten_label(workload2)
        percentages, sum_throughput, avg_power, freq = extract_throughput_and_power(index)
        
        norm_throughput, norm_power = normalize_values(sum_throughput, avg_power)
        
        # Get color based on frequency
        color = frequency_colors.get(freq, 'black')  # Default to black if freq is not found
        
        # Plotting the throughput (separate point for 100-100, rest connected)
        ax1.scatter(percentages[0], norm_throughput[0], color=color, marker='o')  # No label for (w1_100, w2_100)
        ax1.plot(percentages[1:], norm_throughput[1:], label=f'Throughput FREQ{freq}', marker='o', color=color)

        # Plotting the power (separate point for 100-100, rest connected)
        ax2.scatter(percentages[0], norm_power[0], color=color, marker='s')  # No label for (w1_100, w2_100)
        ax2.plot(percentages[1:], norm_power[1:], label=f'Power FREQ{freq}', linestyle='--', marker='s', color=color)
        
    # Set Y-axis lower limit for throughput (to 0.5)
    if min(norm_throughput) > 0.5:
        ax1.set_ylim(bottom=0.5)

    # Set Y-axis lower limit for power (to 0.7)
    if min(norm_power) > 0.7:
        ax2.set_ylim(bottom=0.7)
    # Set titles and labels for each subplot
    ax1.set_title(f"Normalized Throughput for {shorten_w1} and {shorten_w2}")
    ax1.set_xlabel("Percentage Allocation")
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_xticks(range(len(percentages)))
    ax1.set_xticklabels(percentages, rotation=45, ha='right')
    #set ylow limit=0.5
   
    ax1.grid()

    ax2.set_title(f"Normalized Power Consumption for {shorten_w1} and {shorten_w2}")
    ax2.set_xlabel("Percentage Allocation")
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set_xticks(range(len(percentages)))
    ax2.set_xticklabels(percentages, rotation=45, ha='right')
    ax2.grid()

    plt.tight_layout()
    plt.savefig(f'2dnormalized_throughput_power_{shorten_w1}_{shorten_w2}.png')
