import pandas as pd
import sys

def compare_csvs(file1_path, file2_path):
    # Load CSV files
    df_file1 = pd.read_csv(file1_path)
    df_file2 = pd.read_csv(file2_path)

    # Ensure columns exist
    required_columns = {"workload1", "workload2"}
    if not required_columns.issubset(df_file1.columns) or not required_columns.issubset(df_file2.columns):
        raise ValueError("Both CSV files must contain 'workload1' and 'workload2' columns.")

    # Extract workload pairs as sets of tuples
    pairs_file1 = set(df_file1[['workload1', 'workload2']].itertuples(index=False, name=None))
    pairs_file2 = set(df_file2[['workload1', 'workload2']].itertuples(index=False, name=None))

    # Determine which file has more pairs to assign as csv1 (reference)
    if len(pairs_file1) >= len(pairs_file2):
        csv1_path, csv2_path = file1_path, file2_path
        df1, df2 = df_file1, df_file2
        pairs1, pairs2 = pairs_file1, pairs_file2
    else:
        csv1_path, csv2_path = file2_path, file1_path
        df1, df2 = df_file2, df_file1
        pairs1, pairs2 = pairs_file2, pairs_file1

    print(f"Using {csv1_path} as csv1 (reference) with {len(pairs1)} pairs.")
    print(f"Using {csv2_path} as csv2 with {len(pairs2)} pairs.")

    # Find pairs in csv1 that are truly missing (neither pair nor its reverse in csv2)
    truly_missing_in_csv2 = set()
    for pair in pairs1:
        reversed_pair = (pair[1], pair[0])
        if pair not in pairs2 and reversed_pair not in pairs2:
            truly_missing_in_csv2.add(pair)
        elif reversed_pair in pairs2:
            print(f"Pair {pair} from {csv1_path} matches reversed pair {reversed_pair} in {csv2_path}")

    # Filter df1 for rows with truly missing pairs
    result_df = df1[df1[['workload1', 'workload2']].apply(tuple, axis=1).isin(truly_missing_in_csv2)]

    # Save results to CSV
    result_df.to_csv("missing_workloads.csv", index=False)
    print(f"\nFound {len(truly_missing_in_csv2)} pairs from {csv1_path} truly missing in {csv2_path} (no direct or reversed match).")
    print("Rows with truly missing pairs saved to 'missing_workloads.csv'.")

# Example usage with command-line arguments
if len(sys.argv) != 3:
    print("Usage: python script.py <file1.csv> <file2.csv>")
    sys.exit(1)

file1_path = sys.argv[1]
file2_path = sys.argv[2]
compare_csvs(file1_path, file2_path)