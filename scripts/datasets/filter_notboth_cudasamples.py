import pandas as pd
import sys
import os
import argparse

# Define the cudasamples array
cudasamples_array = [
    'cudaTensorCoreGemm',
    'fastWalshTransform',
    'reductionMultiBlockCG',
    'transpose',
    'BlackScholes',
    'sortingNetworks'
]

# Set up argument parser
parser = argparse.ArgumentParser(description='Filter out rows with multiple cudasample workloads')
parser.add_argument('input_csv', help='Input CSV file path')
parser.add_argument('--num-combinations', '-n', type=int, default=None,
                    help='Number of workload combinations (auto-detected if not specified)')
args = parser.parse_args()

# Input and output file paths
input_csv = args.input_csv
output_csv = input_csv.replace('.csv', '_removebothcudasamples.csv')

# Read the CSV file into a DataFrame
df = pd.read_csv(input_csv)

# Auto-detect number of workload combinations if not specified
if args.num_combinations is None:
    workload_cols = [col for col in df.columns if col.startswith('workload') and col[8:].isdigit()]
    num_combinations = len(workload_cols)
    print(f"Auto-detected {num_combinations} workload combinations")
else:
    num_combinations = args.num_combinations
    print(f"Using {num_combinations} workload combinations as specified")

# Function to check if a workload contains any substring from cudasamples_array
def contains_cudasample(workload):
    if pd.isna(workload):
        return False
    return any(sample in str(workload) for sample in cudasamples_array)

# Create list of workload column names
workload_columns = [f'workload{i+1}' for i in range(num_combinations)]

# Filter rows: remove rows where MORE THAN ONE workload contains cudasamples
def has_multiple_cudasamples(row):
    cudasample_count = sum(contains_cudasample(row[col]) for col in workload_columns if col in row)
    return cudasample_count > 1

# Apply the filter: keep rows that do NOT have multiple cudasamples
filtered_df = df[~df.apply(has_multiple_cudasamples, axis=1)]

# Now separate the filtered data into nonDL (with cuda samples) and DL only
def has_any_cudasample(row):
    cudasample_count = sum(contains_cudasample(row[col]) for col in workload_columns if col in row)
    return cudasample_count > 0

def has_no_cudasamples(row):
    cudasample_count = sum(contains_cudasample(row[col]) for col in workload_columns if col in row)
    return cudasample_count == 0

# Create nonDL CSV (rows with at least one cudasample workload - but not multiple since we already filtered those out)
nonDL_df = filtered_df[filtered_df.apply(has_any_cudasample, axis=1)]

# Create DL only CSV (rows where no workloads are cuda samples)
dl_only_df = filtered_df[filtered_df.apply(has_no_cudasamples, axis=1)]

# Generate output file names
nonDL_output_csv = input_csv.replace('mergecudaDL', 'nonDL')
dl_output_csv = input_csv.replace('mergecudaDL', 'DL')

# Print some statistics
original_rows = len(df)
filtered_rows = len(filtered_df)
nonDL_rows = len(nonDL_df)
dl_only_rows = len(dl_only_df)

print(f"Original number of rows: {original_rows}")
print(f"Filtered number of rows (removed multiple cuda samples): {filtered_rows}")
print(f"Rows removed: {original_rows - filtered_rows}")
print(f"NonDL rows (with exactly one cudasample): {nonDL_rows}")
print(f"DL only rows (no cudasamples): {dl_only_rows}")
print(f"Total check (nonDL + DL should equal filtered): {nonDL_rows + dl_only_rows} == {filtered_rows}")

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv(output_csv, index=False)
print(f"\nFiltered data saved to {output_csv}")

# Save nonDL CSV (rows with exactly one cudasample)
nonDL_df.to_csv(nonDL_output_csv, index=False)
print(f"NonDL data (with cudasamples) saved to {nonDL_output_csv}")

# Save DL only CSV
dl_only_df.to_csv(dl_output_csv, index=False)
print(f"DL only data saved to {dl_output_csv}")

# Optional: Display the first few rows of each DataFrame
print("\nFirst few rows of the filtered DataFrame:")
print(filtered_df.head())
print(f"\nFirst few rows of nonDL DataFrame ({len(nonDL_df)} rows):")
print(nonDL_df.head())
print(f"\nFirst few rows of DL only DataFrame ({len(dl_only_df)} rows):")
print(dl_only_df.head())