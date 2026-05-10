import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def shorten_label(label):
    filename_type_dict = {'wav2vec2-base-960h' : 'Wav2vec2', 'bert-base-cased': 'BERT',
                      'mobilenet': 'mobile', 'mobilenet_v2_1.0_224' : 'mobile','vit' : 'ViT', 
                      'vit_h_14' : 'ViT', 'whisper-large-v2': 'Whisper', 'vit-base-patch16-224' : "ViT",
                      'albert-base-v2' : 'ALBERT'}
    for key, value in filename_type_dict.items():
        if key in label:
            #split the label by '-' in last occurence
            batch_type = label.rsplit('_', 1)[1]
            #print(label.rsplit('_', 1)[0])
            batch_type = batch_type.replace('batch', 'b')
            #print(label.rsplit('_', 1))
            #capitalize the first letter
            #value = value.capitalize()
            return value+ '_' + batch_type
    return label

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
#apply shorten label to merged_df  
merged_df["workload1"] = merged_df["workload1"].apply(shorten_label)
merged_df["workload2"] = merged_df["workload2"].apply(shorten_label)
# Unique workload pairs
unique_workload_pairs = merged_df[['workload1', 'workload2']].drop_duplicates()


# Define thread partition combinations we're interested in
thread_partitions = ['(w1_10, w2_90)', '(w1_30, w2_70)', '(w1_50, w2_50)', '(w1_70, w2_30)', '(w1_90, w2_10)', '(w1_100, w2_100)']
colors = ['r', 'g', 'b', 'c', 'm', 'y']  # Added enough colors for each partition

# Function to calculate throughput sum for each partition
def calculate_throughput_sum(throughput_col):
    try:
        return float(throughput_col.split(',')[0][1:]) + float(throughput_col.split(',')[1][:-1])
    except:
        return None

# Create 3D plots for each unique workload pair with different thread partitions
fig = plt.figure(figsize=(20, 14))

for idx, (workload1, workload2) in enumerate(unique_workload_pairs.values):
    ax = fig.add_subplot(2, 3, idx + 1, projection='3d')  # Create a 2x3 grid for six graphs
    
    # Filter data for the current workload pair
    workload_pair_data = merged_df[(merged_df['workload1'] == workload1) & (merged_df['workload2'] == workload2)]
    
    print("workload_pair", workload1, workload2)
    
    # Plot the data for each thread partition
    for partition, color in zip(thread_partitions, colors):
        throughput_col = f'{partition}_throughput'
        power_col = f'{partition}_power'
        
        if throughput_col in workload_pair_data.columns and power_col in workload_pair_data.columns:
            # Calculate throughput sum
            workload_pair_data[f'{partition}_throughput_sum'] = workload_pair_data[throughput_col].apply(calculate_throughput_sum)
            # Filter out rows where throughput sum or power is None
            valid_data = workload_pair_data[(workload_pair_data[f'{partition}_throughput_sum'].notnull()) & 
                                            (workload_pair_data[power_col].notnull())]
            

            throughput_sum = valid_data[f'{partition}_throughput_sum']
            power = valid_data[power_col]
            frequency = valid_data['freq1'].astype(float)  # Filter frequency to match valid rows
            
            # Plot the valid points
            ax.scatter(frequency, power, throughput_sum, label=partition, color=color)
            
            # Connect the points with lines
            ax.plot(frequency, power, throughput_sum, color=color)
    
    # Labels and titles for each graph
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Power (W)')
    ax.set_zlabel('Throughput Sum')
    
    ax.set_title(f'{workload1} vs {workload2}')
    
    # Add a legend to indicate different thread partitions
    ax.legend()
# Adjust the layout, including the space between the plots
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
# Adjust layout and display the plots
plt.tight_layout()
plt.savefig('3d_plots_with_lines.png')
