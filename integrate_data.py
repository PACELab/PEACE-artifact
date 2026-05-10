#!/usr/bin/env python3

import pandas as pd
import sys

def main():
    # File paths
    source_file = "/home/cc/mlProfiler/tests/mps/freq_scaling/dataset/09152025_freq1530_nodvfs_missing/09152025_freq1530_nodvfs_missing_throughput_total_labels_comb2.csv"
    target_file = "/home/cc/mlProfiler/tests/mps/multiinstance/dataset/03112025_DL0207_0307_nonDL0311_nodvfs/0307_data1229_DL_throughput_total_labels_comb2.csv"

    print("Reading source file (missing data)...")
    source_df = pd.read_csv(source_file)
    print(f"Source file loaded: {len(source_df)} rows")

    print("Reading target file (main dataset)...")
    target_df = pd.read_csv(target_file)
    print(f"Target file loaded: {len(target_df)} rows")

    # Verify columns match
    source_cols = set(source_df.columns)
    target_cols = set(target_df.columns)

    if source_cols != target_cols:
        missing_in_source = target_cols - source_cols
        missing_in_target = source_cols - target_cols

        if missing_in_source:
            print(f"ERROR: Columns missing in source file: {missing_in_source}")
        if missing_in_target:
            print(f"ERROR: Columns missing in target file: {missing_in_target}")

        sys.exit(1)

    print("Column names match between files ✓")

    # Reorder source columns to match target
    source_df = source_df[target_df.columns]

    # Create a lookup for sensitivity values from target dataset
    print("Creating sensitivity lookup from main dataset...")
    sensitivity_lookup = {}
    for _, row in target_df.iterrows():
        key = (row['workload1'], row['workload2'])
        sensitivity_lookup[key] = (row['w1_sensitivity'], row['w2_sensitivity'])

        # Also store the reverse key for bidirectional lookup
        reverse_key = (row['workload2'], row['workload1'])
        sensitivity_lookup[reverse_key] = (row['w2_sensitivity'], row['w1_sensitivity'])

    print(f"Sensitivity lookup created with {len(sensitivity_lookup)} entries")

    # Update sensitivity values in source data
    print("Updating sensitivity values in source data...")
    updated_rows = 0
    for idx, row in source_df.iterrows():
        workload_pair = (row['workload1'], row['workload2'])
        if workload_pair in sensitivity_lookup:
            new_w1_sens, new_w2_sens = sensitivity_lookup[workload_pair]
            source_df.at[idx, 'w1_sensitivity'] = new_w1_sens
            source_df.at[idx, 'w2_sensitivity'] = new_w2_sens
            updated_rows += 1
        else:
            print(f"WARNING: No sensitivity data found for workload pair: {workload_pair}")

    print(f"Updated sensitivity values for {updated_rows} rows")

    # Check for data conflicts before merging
    print("Checking for duplicate entries...")

    # Create composite keys for both datasets
    target_df['composite_key'] = target_df['workload1'] + '_' + target_df['workload2'] + '_' + target_df['idx1'].astype(str) + '_' + target_df['idx2'].astype(str) + '_' + target_df['w1_Threadpercent'].astype(str) + '_' + target_df['w2_Threadpercent'].astype(str)
    source_df['composite_key'] = source_df['workload1'] + '_' + source_df['workload2'] + '_' + source_df['idx1'].astype(str) + '_' + source_df['idx2'].astype(str) + '_' + source_df['w1_Threadpercent'].astype(str) + '_' + source_df['w2_Threadpercent'].astype(str)

    # Check for overlapping keys
    target_keys = set(target_df['composite_key'])
    source_keys = set(source_df['composite_key'])
    overlap = target_keys & source_keys

    if overlap:
        print(f"ERROR: Found {len(overlap)} duplicate entries between datasets:")
        for key in list(overlap)[:5]:  # Show first 5 duplicates
            print(f"  - {key}")
        if len(overlap) > 5:
            print(f"  ... and {len(overlap) - 5} more")
        sys.exit(1)

    print("No duplicate entries found ✓")

    # Remove composite_key columns before merging
    target_df = target_df.drop('composite_key', axis=1)
    source_df = source_df.drop('composite_key', axis=1)

    # Merge the datasets
    print("Merging datasets...")
    merged_df = pd.concat([target_df, source_df], ignore_index=True)

    print(f"Merged dataset: {len(merged_df)} rows (original: {len(target_df)}, added: {len(source_df)})")

    # Sort by workload1, workload2, idx1, idx2, w1_Threadpercent, w2_Threadpercent for consistent ordering
    merged_df = merged_df.sort_values(['workload1', 'workload2', 'idx1', 'idx2', 'w1_Threadpercent', 'w2_Threadpercent'])

    # Backup original file
    backup_file = target_file + ".backup"
    print(f"Creating backup: {backup_file}")
    target_df.to_csv(backup_file, index=False)

    # Write merged result
    print(f"Writing merged data to: {target_file}")
    merged_df.to_csv(target_file, index=False)

    print("✓ Integration completed successfully!")
    print(f"Final dataset contains {len(merged_df)} rows")

if __name__ == "__main__":
    main()