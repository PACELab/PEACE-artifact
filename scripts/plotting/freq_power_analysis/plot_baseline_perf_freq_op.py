import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the dataset
file_path = '~/Downloads/0429_baseline_metrics.csv'
data = pd.read_csv(file_path)

output_dir = "0429_base_freq"
axis_font, label_font = 14, 16

is_norm = True
#basecolumn to normalize
base_percent = 50
base_freq = 1300
# Filter the dataset to get unique model types
model_types = data['Type'].unique()

# Initialize an empty list to store the figures
figures = []

# Loop through each model type
for model in model_types:
    # Filter data for the current model type
    model_data = data[data['Type'] == model]
    #get base throughput and power
    base_throughput = model_data[data["freq"] == base_freq][f'Exclusive{base_percent}'].values[0]
    base_power = model_data[data["freq"] == base_freq][f'power_Exclusive{base_percent}'].values[0]
    print(f"Base throughput for {model} at {base_freq} MHz: {base_throughput}")
    print(f"Base power for {model} at {base_freq} MHz: {base_power}")
    assert(base_throughput != None)
    assert(base_power != None)
    # Prepare the subfigures (3 subplots: throughput, power, and throughput per watt)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Metrics for {model}, normaled with 50%, maxfreq', fontsize=label_font+3)
    
    # Plot throughput vs frequency (Exclusive)
    axes[0].set_title('Exclusive Throughput vs Frequency', fontsize=label_font)
    axes[0].set_xlabel('Frequency' , fontsize=label_font)
    axes[0].set_ylabel('Throughput (Exclusive)', fontsize=label_font)
    
    # Plot power vs frequency (Power Exclusive)
    axes[1].set_title('Power Exclusive vs Frequency', fontsize=label_font)
    axes[1].set_xlabel('Frequency', fontsize=label_font)
    axes[1].set_ylabel('Power (Power Exclusive)', fontsize=label_font)
    
    # Plot throughput per watt vs frequency
    axes[2].set_title('Throughput per Watt vs Frequency', fontsize=label_font)
    axes[2].set_xlabel('Frequency', fontsize=label_font)
    axes[2].set_ylabel('Throughput per Watt', fontsize=label_font)
    

    # Loop through each partition percentage and plot
    for partition in [i for i in range(10, 101, 10)]:
        # Create the column names dynamically for throughput and power
        throughput_column = f'Exclusive{partition}'
        power_column = f'power_Exclusive{partition}'
        
        if throughput_column in model_data.columns or power_column in model_data.columns:
            # Extract relevant data for throughput and power
            if model_data[throughput_column].isnull().all() and model_data[power_column].isnull().all():
                continue
            #create throughput per watt column


            # Plot throughput on the first subfigure (Exclusive)
            # Normalize the throughput and power values based on the base values
           #if base_throughput != None and base_power != None:
                #model_data[throughput_column] = model_data[throughput_column] 
                #model_data[power_column] = model_data[power_column] / base_power
                # Normalize the throughput per watt
                
            throughput_per_watt = model_data[throughput_column] / model_data[power_column]
            if is_norm:
                
                axes[0].plot(model_data['freq'], model_data[throughput_column] / base_throughput, label=f'{partition}%', marker='o')
            
                # Plot power on the second subfigure (Power Exclusive)
                axes[1].plot(model_data['freq'], model_data[power_column] / base_power, label=f'{partition}%', marker='o')
                throughput_per_watt = throughput_per_watt / (base_throughput / base_power)  # Normalize the throughput per watt
                
                axes[2].plot(model_data['freq'], throughput_per_watt, label=f'{partition}%', marker='o')
            else:
                axes[0].plot(model_data['freq'], model_data[throughput_column], label=f'{partition}%', marker='o')
                # Plot power on the second subfigure (Power Exclusive)
                axes[1].plot(model_data['freq'], model_data[power_column], label=f'{partition}%', marker='o')
                axes[2].plot(model_data['freq'], throughput_per_watt, label=f'{partition}%', marker='o')

    
    # Add legends to all subfigures
    axes[0].legend(title="Partition Percentage")
    axes[1].legend(title="Partition Percentage")
    axes[2].legend(title="Partition Percentage")
    #add grid
    for ax in axes:
        ax.grid(True)
        #set xaxis size
        ax.tick_params(axis='x', labelsize=axis_font)
        #set yaxis size
        ax.tick_params(axis='y', labelsize=axis_font)
        y_min = min(ax.get_ylim())
        #set y min to be min(min(minimun of all axis, 0.8))
        ax.set_ylim(bottom=min(0.7, y_min))
    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the figure for the model type
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    figure_filename = f'./{output_dir}/{model}_metrics.png'
    plt.savefig(figure_filename)
    
    # Close the current figure to avoid memory overload
    plt.close(fig)

# Return the file paths for the saved figures
figures
