import pandas as pd
import matplotlib.pyplot as plt

# Load the three CSV files for different frequencies
file_path_300 = '~/Downloads/0422_freq300_baseline_metrics.csv'
file_path_900 = '~/Downloads/0422_freq900_baseline_metrics.csv'
file_path_1530 = '~/Downloads/0422_freq1530_baseline_metrics.csv'

data_300 = pd.read_csv(file_path_300)
data_900 = pd.read_csv(file_path_900)
data_1530 = pd.read_csv(file_path_1530)

# Add a 'Frequency' column to each dataset
data_300['Frequency'] = 300
data_900['Frequency'] = 900
data_1530['Frequency'] = 1530

# Combine all the datasets into a single dataframe
data_combined = pd.concat([data_300, data_900, data_1530])

# List of thread percentages (10%, 20%, ..., 100%)
thread_percentages_numeric = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# List of power columns (power_Exclusive10, power_Exclusive20, ..., power_Exclusive100)
power_columns = [f'power_Exclusive{i}' for i in thread_percentages_numeric]
throughput_columns = [f'Exclusive{i}' for i in thread_percentages_numeric]

# Get unique workload types
workload_types = data_combined['Type'].unique()

# Plotting power vs thread percentage and throughput vs thread percentage for each workload
for workload in workload_types:
    # Filter data for the specific workload
    subset = data_combined[data_combined['Type'] == workload]

    # Create subplots (one for power and one for throughput)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Power vs Thread Percentage for each frequency (300, 900, 1530)
    for freq in [300, 900, 1530]:
        subset_freq = subset[subset['Frequency'] == freq]
        
        
        # Check if any data exists for the frequency
        if subset_freq.empty:
            continue
        
        # Extract power values for the specific frequency and thread percentages
        power_values = subset_freq[power_columns].values.flatten()
        throughput_values = subset_freq[throughput_columns].values.flatten()
        #find if any None in throughput_values
        #normalize - if Exclusive 100 not None, divide all by that value
        if throughput_values[-1] is not None:
            throughput_values = [x / throughput_values[-1] for x in throughput_values if x is not None]
        if power_values[-1] is not None:
            power_values = [x / power_values[-1] for x in power_values if x is not None]
        
        #record index, change power_values to None for that index
        index = [i for i, x in enumerate(throughput_values) if pd.isna(x)]
        for i in index:
            power_values[i] = None
        
        # Only plot if there are valid data points
        if len(power_values) > 0:
            ax1.plot(thread_percentages_numeric, power_values, label=f'Frequency {freq} MHz', marker='o', linestyle='-', linewidth=2)
        if len(throughput_values) > 0:
            ax2.plot(thread_percentages_numeric, throughput_values, label=f'Frequency {freq} MHz', marker='o', linestyle='-', linewidth=2)

    ax1.set_title(f'Power vs Thread Percentage ({workload})')
    ax1.set_xlabel('Thread Percentage (%)')
    ax1.set_ylabel('Power (W)')
    ax1.legend(title="Frequencies")
    ax1.grid(True)


    ax2.set_title(f'Throughput vs Thread Percentage ({workload})')
    ax2.set_xlabel('Thread Percentage (%)')
    ax2.set_ylabel('Throughput (Units)')
    ax2.legend(title="Frequencies")
    ax2.grid(True)

    # Adjust layout and save the figure for each workload
    plt.tight_layout()
    plt.savefig(f'{workload}_power_throughput_plots.png')  # Save each figure with the workload name
    plt.close(fig)  # Close the figure to avoid display in the loop

print("Figures have been saved for each workload.")
