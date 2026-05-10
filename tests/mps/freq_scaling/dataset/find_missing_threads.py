import pandas as pd
import os
import sys
import argparse

# Define all possible thread combinations
ALL_COMBINATIONS = { 
    "comb2" : [
    "(w1_10, w2_90)", "(w1_20, w2_80)", "(w1_30, w2_70)",
    "(w1_40, w2_60)", "(w1_50, w2_50)", "(w1_60, w2_40)",
    "(w1_70, w2_30)", "(w1_80, w2_20)", "(w1_90, w2_10)",
    "(w1_100, w2_100)"
],
    "comb3_sampled": ["(w1_100, w2_100, w3_100)",
    "(w1_10, w2_30, w3_60)",
    "(w1_10, w2_60, w3_30)",
    "(w1_20, w2_10, w3_70)",
    "(w1_20, w2_40, w3_40)",
    "(w1_20, w2_70, w3_10)",
    "(w1_30, w2_30, w3_40)",
    "(w1_30, w2_60, w3_10)",
    "(w1_40, w2_30, w3_30)",
    "(w1_50, w2_10, w3_40)",
    "(w1_50, w2_40, w3_10)",
    "(w1_60, w2_30, w3_10)",
    "(w1_80, w2_10, w3_10)"
    ],
    "comb3": [
        "(w1_100, w2_100, w3_100)",
        "(w1_10, w2_10, w3_80)",
        "(w1_10, w2_20, w3_70)",
        "(w1_10, w2_30, w3_60)",
        "(w1_10, w2_40, w3_50)",
        "(w1_10, w2_50, w3_40)",
        "(w1_10, w2_60, w3_30)",
        "(w1_10, w2_70, w3_20)",
        "(w1_10, w2_80, w3_10)",
        "(w1_20, w2_10, w3_70)",
        "(w1_20, w2_20, w3_60)",
        "(w1_20, w2_30, w3_50)",
        "(w1_20, w2_40, w3_40)",
        "(w1_20, w2_50, w3_30)",
        "(w1_20, w2_60, w3_20)",
        "(w1_20, w2_70, w3_10)",
        "(w1_30, w2_10, w3_60)",
        "(w1_30, w2_20, w3_50)",
        "(w1_30, w2_30, w3_40)",
        "(w1_30, w2_40, w3_30)",
        "(w1_30, w2_50, w3_20)",
        "(w1_30, w2_60, w3_10)",
        "(w1_40, w2_10, w3_50)",
        "(w1_40, w2_20, w3_40)",
        "(w1_40, w2_30, w3_30)",
        "(w1_40, w2_40, w3_20)",
        "(w1_40, w2_50, w3_10)",
        "(w1_50, w2_10, w3_40)",
        "(w1_50, w2_20, w3_30)",
        "(w1_50, w2_30, w3_20)",
        "(w1_50, w2_40, w3_10)",
        "(w1_60, w2_10, w3_30)",
        "(w1_60, w2_20, w3_20)",
        "(w1_60, w2_30, w3_10)",
        "(w1_70, w2_10, w3_20)",
        "(w1_70, w2_20, w3_10)",
        "(w1_80, w2_10, w3_10)"
    ]
}


def find_missing_combinations(df, combinations_list):
    # Group by workload pairs
    #for Workload1', 'Workload2' columns, replace cuda_samples substring with samples
    df['Workload1'] = df['Workload1'].str.replace('cuda_samples', 'samples')
    df['Workload2'] = df['Workload2'].str.replace('cuda_samples', 'samples')

    # Determine workload columns based on number of combinations
    workload_cols = ['Workload1', 'Workload2']
    if 'Workload3' in df.columns:
        workload_cols.append('Workload3')
        df['Workload3'] = df['Workload3'].str.replace('cuda_samples', 'samples')

    grouped = df.groupby(workload_cols)

    missing_entries = []

    # For each workload pair/tuple
    for workloads, group in grouped:
        # Get existing thread combinations for this pair
        existing_combinations = set(group['Thread_combination'])
        # Find missing combinations
        missing_combinations = set(combinations_list) - existing_combinations

        # Create entries for missing combinations
        for combo in missing_combinations:
            entry = {col: val for col, val in zip(workload_cols, workloads if isinstance(workloads, tuple) else [workloads])}
            entry.update({
                'Thread_combination': combo,
                'weight_Throughput_sum': None,
                'Power': None
            })
            missing_entries.append(entry)

    return pd.DataFrame(missing_entries)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Find missing thread combinations in profiling data')
    parser.add_argument('baseline_csv', help='Baseline CSV file with powercap info')
    parser.add_argument('original_csv', help='Original CSV with throughput label data')
    parser.add_argument('--n_comb', type=int, required=True, choices=[2, 3],
                        help='Number of workload combinations (2 or 3)')
    parser.add_argument('--sample', action='store_true',
                        help='Use sampled combinations for comb3 (comb3_sampled)')

    args = parser.parse_args()

    baseline_csv_path = args.baseline_csv
    original_csv_path = args.original_csv
    n_combinations = args.n_comb
    use_sampled = args.sample

    # Select combinations based on n_comb and sample flag
    if n_combinations == 2:
        combinations_list = ALL_COMBINATIONS['comb2']
    elif n_combinations == 3:
        if use_sampled:
            combinations_list = ALL_COMBINATIONS['comb3_sampled']
        else:
            combinations_list = ALL_COMBINATIONS['comb3']
    else:
        print(f"Error: Unsupported number of combinations: {n_combinations}")
        sys.exit(1)
    
    # Check if files exist
    for path in [baseline_csv_path, original_csv_path]:
        if not os.path.exists(path):
            print(f"Error: File not found at {path}")
            sys.exit(1)
    
    try:
        # Read the CSV files

        baseline_df = pd.read_csv(baseline_csv_path)
        original_df = pd.read_csv(original_csv_path)
        print('df read')
        # Verify required columns in original CSV
        required_original_cols = {'Workload1', 'Workload2', 'Thread_combination'}
        if not required_original_cols.issubset(original_df.columns):
            missing_cols = required_original_cols - set(original_df.columns)
            print(f"Error: Original CSV file is missing required columns: {missing_cols}")
            sys.exit(1)
        
        # Check for powercap column in baseline CSV (optional)
        powercap_column = 'powercap50_50'  # Adjust this if your column name differs
        has_powercap = powercap_column in baseline_df.columns

        if not has_powercap:
            print(f"Warning: Baseline CSV file is missing powercap column: {powercap_column}. Skipping powercap merge.")

        # Ensure baseline_df has consistent workload naming
        baseline_df = baseline_df.rename(columns={'workload1': 'Workload1', 'workload2': 'Workload2', 'workload3': 'Workload3'})

        # Find missing combinations using original_df
        missing_df = find_missing_combinations(original_df, combinations_list)

        # Optionally merge with baseline_df to get powercap values if column exists
        if has_powercap:
            merge_cols = ['Workload1', 'Workload2']
            if n_combinations == 3 and 'Workload3' in missing_df.columns:
                merge_cols.append('Workload3')

            missing_df = missing_df.merge(
                baseline_df[merge_cols + [powercap_column]],
                on=merge_cols,
                how='left'
            ).rename(columns={powercap_column: 'PowerCap'})
        else:
            # Add PowerCap column with None values
            missing_df['PowerCap'] = None

        # Extract percentages from Thread_combination based on n_combinations
        if n_combinations == 2:
            missing_df[['w1_percentage', 'w2_percentage']] = missing_df['Thread_combination'].str.extract(r'\(w1_(\d+),\s*w2_(\d+)\)')
            missing_df['w1_percentage'] = missing_df['w1_percentage'].astype(int)
            missing_df['w2_percentage'] = missing_df['w2_percentage'].astype(int)

            # Select only the desired columns, including the new percentage columns
            missing_df = missing_df[['Workload1', 'Workload2', 'Thread_combination',
                                     'w1_percentage', 'w2_percentage',
                                     'weight_Throughput_sum', 'Power', 'PowerCap']]
        else:  # n_combinations == 3
            missing_df[['w1_percentage', 'w2_percentage', 'w3_percentage']] = missing_df['Thread_combination'].str.extract(r'\(w1_(\d+),\s*w2_(\d+),\s*w3_(\d+)\)')
            missing_df['w1_percentage'] = missing_df['w1_percentage'].astype(int)
            missing_df['w2_percentage'] = missing_df['w2_percentage'].astype(int)
            missing_df['w3_percentage'] = missing_df['w3_percentage'].astype(int)

            # Select only the desired columns, including the new percentage columns
            missing_df = missing_df[['Workload1', 'Workload2', 'Workload3', 'Thread_combination',
                                     'w1_percentage', 'w2_percentage', 'w3_percentage',
                                     'weight_Throughput_sum', 'Power', 'PowerCap']]
        
        # Create output directory if it doesn't exist (in same directory as original file)
        output_dir = os.path.join(os.path.dirname(original_csv_path), 'missing_entries')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all missing entries to a single CSV file
        output_file = os.path.join(output_dir, 'all_missing_combinations_with_powercap.csv')
        missing_df.to_csv(output_file, index=False)
        
        print(f"Saved {len(missing_df)} missing combinations to {output_file}")
        
        # Print summary
        total_missing = len(missing_df)
        print(f"\nFound {total_missing} missing combinations across all workload pairs")
        
    except Exception as e:
        if missing_df.empty:
            print("No missing combinations found")
            output_dir = os.path.join(os.path.dirname(original_csv_path), 'missing_entries')
            output_file = os.path.join(output_dir, 'all_missing_combinations_with_powercap.csv')
            missing_df.to_csv(output_file, index=False)
            print(f"Saved {len(missing_df)} missing combinations to {output_file}")
            sys.exit(0)
        else:
            print(f"Error processing files: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    main()