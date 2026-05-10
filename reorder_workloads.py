#!/usr/bin/env python3

import pandas as pd
import re

def parse_thread_combination(thread_str):
    """Parse thread combination string like '(w1_90, w2_10)' to extract percentages"""
    match = re.search(r'\(w1_(\d+), w2_(\d+)\)', thread_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def create_ordering_from_reference(reference_file):
    """Create ordering mapping from reference file"""
    print("Reading reference file...")
    ref_df = pd.read_csv(reference_file)

    ordering = []
    for idx, row in ref_df.iterrows():
        w1_pct, w2_pct = parse_thread_combination(row['Thread_combination'])
        if w1_pct is not None and w2_pct is not None:
            key = (row['Workload1'], row['Workload2'], w1_pct, w2_pct)
            ordering.append(key)

    print(f"Found {len(ordering)} unique workload combinations in reference file")
    return ordering

def reorder_target_file(target_file, reference_ordering):
    """Reorder target file to match reference ordering"""
    print("Reading target file...")
    target_df = pd.read_csv(target_file)

    # For this type of file, ordering is just based on workload1, workload2 pairs
    # Create composite key for target data (no thread percentages in this format)
    target_df['sort_key'] = target_df.apply(lambda row: (row['workload1'], row['workload2']), axis=1)

    # Create ordering dictionary from reference (just workload pairs, ignore thread percentages)
    ref_pairs_seen = set()
    ref_pair_order = {}
    for idx, (workload1, workload2, w1_pct, w2_pct) in enumerate(reference_ordering):
        pair_key = (workload1, workload2)
        if pair_key not in ref_pairs_seen:
            ref_pair_order[pair_key] = len(ref_pair_order)
            ref_pairs_seen.add(pair_key)

    print(f"Found {len(ref_pair_order)} unique workload pairs in reference")

    # Function to get sort order
    def get_sort_order(row_key):
        workload1, workload2 = row_key

        # Try exact match first
        if row_key in ref_pair_order:
            return ref_pair_order[row_key]

        # Try reverse match (w1 and w2 swapped)
        reverse_key = (workload2, workload1)
        if reverse_key in ref_pair_order:
            return ref_pair_order[reverse_key]

        # If not found, put at end
        return len(ref_pair_order) + 1000

    # Apply sort ordering
    target_df['sort_order'] = target_df['sort_key'].apply(get_sort_order)

    # Sort by the ordering
    target_df_sorted = target_df.sort_values('sort_order')

    # Remove helper columns
    target_df_sorted = target_df_sorted.drop(['sort_key', 'sort_order'], axis=1)

    print(f"Reordered {len(target_df_sorted)} rows")

    return target_df_sorted

def reorder_target_file_with_thread_pct(target_file, reference_ordering):
    """Reorder target file to match reference ordering (for files with thread percentages)"""
    print("Reading target file...")
    target_df = pd.read_csv(target_file)

    # Create composite key for target data
    target_df['sort_key'] = target_df.apply(lambda row: (row['workload1'], row['workload2'],
                                                        row['w1_Threadpercent'], row['w2_Threadpercent']), axis=1)

    # Create ordering dictionary from reference
    order_dict = {key: idx for idx, key in enumerate(reference_ordering)}

    # Function to get sort order
    def get_sort_order(row_key):
        workload1, workload2, w1_pct, w2_pct = row_key

        # Try exact match first
        if row_key in order_dict:
            return order_dict[row_key]

        # Try reverse match (w1 and w2 swapped)
        reverse_key = (workload2, workload1, w2_pct, w1_pct)
        if reverse_key in order_dict:
            return order_dict[reverse_key]

        # If not found, put at end
        return len(reference_ordering) + 1000

    # Apply sort ordering
    target_df['sort_order'] = target_df['sort_key'].apply(get_sort_order)

    # Sort by the ordering
    target_df_sorted = target_df.sort_values('sort_order')

    # Remove helper columns
    target_df_sorted = target_df_sorted.drop(['sort_key', 'sort_order'], axis=1)

    print(f"Reordered {len(target_df_sorted)} rows")

    return target_df_sorted

def main():
    reference_file = "/home/cc/mlProfiler/tests/mps/multiinstance/dataset/03112025_DL0207_0307_nonDL0311_nodvfs/0307_data1229_DL_throughput_total_labels_comb2_labels.csv"

    # Choose target file - uncomment the one you want to reorder
    # target_file = "/home/cc/mlProfiler/tests/mps/multiinstance/02062025_powercap100_DL_baseline_labels_comb2_batches2.csv"  # For workload pairs only
    target_file = "/home/cc/mlProfiler/tests/mps/multiinstance/dataset/03112025_DL0207_0307_nonDL0311_nodvfs/0307_data1229_DL_throughput_total_labels_comb2.csv"  # For full workload+thread combinations

    # Create ordering from reference
    reference_ordering = create_ordering_from_reference(reference_file)

    # Choose reordering function based on target file type
    if "02062025_powercap100_DL_baseline_labels_comb2_batches2.csv" in target_file:
        # For files without explicit thread percentages
        reordered_df = reorder_target_file(target_file, reference_ordering)
    else:
        # For files with explicit thread percentages
        reordered_df = reorder_target_file_with_thread_pct(target_file, reference_ordering)

    # Create backup
    backup_file = target_file + ".backup"
    print(f"Creating backup: {backup_file}")

    # Read original for backup
    original_df = pd.read_csv(target_file)
    original_df.to_csv(backup_file, index=False)

    # Write reordered file
    print(f"Writing reordered data to: {target_file}")
    reordered_df.to_csv(target_file, index=False)

    print("✓ Reordering completed successfully!")

    # Verify first few entries match
    print("\nFirst 5 workload combinations in reordered file:")
    if "02062025_powercap100_DL_baseline_labels_comb2_batches2.csv" in target_file:
        for idx, row in reordered_df.head(5).iterrows():
            print(f"  {row['workload1']} + {row['workload2']} (idx1={row['idx1']}, idx2={row['idx2']})")
    else:
        for idx, row in reordered_df.head(5).iterrows():
            print(f"  {row['workload1']} + {row['workload2']}: w1_{row['w1_Threadpercent']}%, w2_{row['w2_Threadpercent']}%")

if __name__ == "__main__":
    main()