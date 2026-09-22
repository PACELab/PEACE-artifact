import pandas as pd
from functools import reduce

# List all 10 CSV file paths
file_paths = [
    "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/data/experiment_inputs/colocations/test/fold_1.csv",
    "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/data/experiment_inputs/colocations/test/fold_2.csv",
    "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/data/experiment_inputs/colocations/test/fold_3.csv",
    "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/data/experiment_inputs/colocations/test/fold_4.csv",
    "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/data/experiment_inputs/colocations/test/fold_5.csv",
    "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/data/experiment_inputs/colocations/test/fold_6.csv",
    "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/data/experiment_inputs/colocations/test/fold_7.csv",
    "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/data/experiment_inputs/colocations/test/fold_8.csv",
    "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/data/experiment_inputs/colocations/test/fold_9.csv",
    "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/data/experiment_inputs/colocations/test/fold_10.csv"
]

# 1. Read all CSVs into a list of DataFrames
dfs = [pd.read_csv(fp) for fp in file_paths]

# 2. Concatenate them into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# 3. Check for duplicates across *all* columns
duplicate_rows = combined_df[combined_df.duplicated(keep=False)]

if not duplicate_rows.empty:
    # Raise an error if duplicates are found
    raise ValueError(f"Duplicates found in the concatenated data:\n{duplicate_rows}")


# 4. Save the merged DataFrame to a single CSV
combined_df.to_csv("merged_10files.csv", index=False)
print("Merged CSV saved to /mnt/data/merged_10files.csv without duplicates.")
