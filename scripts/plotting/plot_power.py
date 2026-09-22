import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr  

# 1. Read your CSV into a DataFrame
df = pd.read_csv("../analysis/0207_baseline_metrics.csv")


#print correlation of Power_Exclusive100 with all other columns

# 2. Prepare the x-values that map to the columns
x_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# 3. Identify unique workloads (types)
workload_types = df['Type'].unique()

#get 100 plots

fig = plt.figure()

# If there's only one unique type, axes will not be an array. Ensure it’s always iterable.
if len(workload_types) == 1:
    axes = [axes]
#scatterplot of sm%100 vs power_exclusive100
for percent in [10, 50, 100]:
    for feat in ["SMACT%", "SMOCC%", "TENSO%", "DRAMA%", "FP32A%"]:
        plt.figure()
        #print correlation of Power_Exclusive100 with all other columns
        #print(f"feature={feat}, corr={df.corr()[f'power_Exclusive{percent}'][f'{feat}{percent}']}")
        plt.scatter(df[f'{feat}100'], df[f'power_Exclusive{percent}'])
        #label each scatter point with the workload type
        #for idx, row in df.iterrows():
        #    plt.text(row[f'{feat}{percent}'], row[f'power_Exclusive{percent}'], row['Type'])
        plt.xlabel(feat)
        plt.ylabel(f"Power_Exclusive{percent}")
        plt.title(f"{feat}{percent} vs Power_Exclusive{percent}")
        
       
        #plt.text(row[f'{feat}{percent}'], row[f'power_Exclusive{percent}'], row['Type'])
        #plt add text of correlation in plot
        plt.text(0.5, 0.5, f"corr={df.corr()[f'power_Exclusive{percent}'][f'{feat}{percent}']}", horizontalalignment='center', verticalalignment='center',  bbox=dict(facecolor='0.85', alpha=0.5))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"scatter_plot_{percent}%power_{feat}.png")


# 4. Set up subplots
fig, axes = plt.subplots(
    nrows=len(workload_types),
    ncols=1,
    figsize=(7, 4 * len(workload_types)),  # Adjust figsize as needed
    sharex=False,  # or True if you want a common X-axis scale
    sharey=False   # or True if you want a common Y-axis scale
)


# 5. Plot each workload on a separate subplot
for ax, workload_type in zip(axes, workload_types):
    # Filter rows that belong to the current workload
    subset = df[df['Type'] == workload_type]
    
    # Plot each row for this workload
    for idx, row in subset.iterrows():
        sm_values = [row[f"SMACT%{x}"] for x in x_values]
        power_values = [row[f"power_Exclusive{x}"] for x in x_values]
        #print(sm_values)
        ax.scatter(sm_values, power_values, label=f"Row {idx}")
        #add scatter text as x for each scatter point
        #for idx, row in subset.iterrows():
        #    ax.text(row[f"sm%{x}"], row[f"power_Exclusive{x}"], row['Type'])
        # Add text labels for each scatter point
        for sm, power, x_val in zip(sm_values, power_values, x_values):
            ax.text(sm, power, str(x_val), fontsize=8, ha='left', va='bottom', color='black')

    ax.set_title(f"Workload: {workload_type}")
    ax.set_xlabel("SMACT%")
    ax.set_ylabel("Power_Exclusive")
    #plot correlation of all points in ax sm values vs power values
    #for idx, row in subset.iterrows():
    #    ax.text(row[f"sm%{x}"], row[f"power_Exclusive{x}"], row['Type'])
    #place a text box in upper middle of ax coords

    ax.text(1, 1.05, f"corr={pearsonr(sm_values, power_values)[0]}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,  bbox=dict(facecolor='0.85', alpha=0.5))

    ax.grid(True)
    ax.legend()
    #cap power from 50 to 250
    ax.set_ylim(50, 250)

plt.tight_layout()
plt.savefig("scatter_plot_subplots_SMACT.png")




#feat 
for feat in ["SMACT%", "SMOCC%", "TENSO%", "DRAMA%", "FP32A%", "PCITX", "PCIRX"]:
# 4. Set up subplots
    fig, axes = plt.subplots(
        nrows=len(workload_types),
        ncols=1,
        figsize=(7, 4 * len(workload_types)),  # Adjust figsize as needed
        sharex=False,  # or True if you want a common X-axis scale
        sharey=False   # or True if you want a common Y-axis scale
    )

    # If there's only one unique type, axes will not be an array. Ensure it’s always iterable.
    if len(workload_types) == 1:
        axes = [axes]

    for ax, workload_type in zip(axes, workload_types):
        # Filter rows that belong to the current workload
        subset = df[df['Type'] == workload_type]
        
        # Plot each row for this workload
        for idx, row in subset.iterrows():
            feat_values = [row[f"{feat}{x}"] for x in x_values]
            power_values = [row[f"power_Exclusive{x}"] for x in x_values]
            ax.scatter(feat_values, power_values, label=f"Row {idx}")
            # Add text labels for each scatter point
            for x_feat, power, x_val in zip(feat_values, power_values, x_values):
                ax.text(x_feat, power, str(x_val), fontsize=8, ha='left', va='bottom', color='black')
            #add corrleation text to ax
            ax.text(1, 1.05, f"corr={pearsonr(feat_values, power_values)[0]}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,  bbox=dict(facecolor='0.85', alpha=0.5))
        ax.set_title(f"Workload: {workload_type}")
        ax.set_xlabel(f"{feat}")
        ax.set_ylabel("Power_Exclusive")
        ax.grid(True)
        ax.legend()
        #cap power from 50 to 250
        ax.set_ylim(50, 250)

    plt.tight_layout()
    plt.savefig(f"scatter_plot_subplots_{feat}.png")








#plot_power("/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/data/model_datasets/0203_batch2/unseen_partition/power/rand10/AutoML/whisper-large-v2_batch2-inf/training_set.csv")