import pandas as pd
import os

def are_csv_files_equal(file1_path, file2_path, ignore_columns=None, precision=6):
    """
    Check if two CSV files contain the same data, ignoring row order.
    
    Parameters:
    -----------
    file1_path : str
        Path to the first CSV file.
    file2_path : str
        Path to the second CSV file.
    ignore_columns : list, optional
        List of column names to ignore during comparison (e.g., ['index']).
    precision : int, optional
        Number of decimal places to round floats to (default=6).
    
    Returns:
    --------
    bool : True if the files contain the same data, False otherwise.
    """
    # Check if files exist
    if not os.path.isfile(file1_path):
        print(f"Error: {file1_path} does not exist.")
        return False
    if not os.path.isfile(file2_path):
        print(f"Error: {file2_path} does not exist.")
        return False

    # Read the CSV files into DataFrames
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    print(f"File 1 rows: {len(df1)}, File 2 rows: {len(df2)}")
    print(f"File 1 columns: {list(df1.columns)}")
    print(f"File 2 columns: {list(df2.columns)}")

    # Check if the number of rows is different
    if len(df1) != len(df2):
        print(f"Files have different row counts: {len(df1)} vs {len(df2)}")
        return False

    # Check if the column names are the same
    if set(df1.columns) != set(df2.columns):
        print(f"Files have different columns: {set(df1.columns)} vs {set(df2.columns)}")
        return False

    # Drop ignored columns if specified
    if ignore_columns:
        df1 = df1.drop(columns=[col for col in ignore_columns if col in df1.columns])
        df2 = df2.drop(columns=[col for col in ignore_columns if col in df2.columns])

    # Round floating-point columns to specified precision
    numeric_cols = df1.select_dtypes(include=['float64']).columns
    df1[numeric_cols] = df1[numeric_cols].round(precision)
    df2[numeric_cols] = df2[numeric_cols].round(precision)

    # Convert DataFrames to sets of tuples for order-independent comparison
    columns = sorted(df1.columns)
    set1 = set(tuple(row) for row in df1[columns].itertuples(index=False))
    set2 = set(tuple(row) for row in df2[columns].itertuples(index=False))

    # Compare the sets
    if set1 == set2:
        print("The files contain the same data (row order may differ).")
        return True
    else:
        # Find differences for debugging
        only_in_file1 = set1 - set2
        only_in_file2 = set2 - set1
        print(f"Files differ. Rows unique to {file1_path}: {len(only_in_file1)}")
        print(f"Rows unique to {file2_path}: {len(only_in_file2)}")
        
        # Print differences with full precision
        if only_in_file1:
            print("Rows unique to file1 (first 5):")
            for row in list(only_in_file1)[:5]:
                print(row)
        if only_in_file2:
            print("Rows unique to file2 (first 5):")
            for row in list(only_in_file2)[:5]:
                print(row)
        
        # Additional check: compare a sample row directly
        if only_in_file1 and only_in_file2:
            print("\nDirect comparison of a differing row:")
            row1 = list(only_in_file1)[0]
            row2 = list(only_in_file2)[0]
            for col, val1, val2 in zip(columns, row1, row2):
                if val1 != val2:
                    print(f"Column '{col}': {val1} (file1) vs {val2} (file2)")
        
        return False
# Example usage
if __name__ == "__main__":
    file1 = "./merged_10files.csv"  # Replace with your first file path
    file2 = "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/data/model_datasets/03112025_DL0207_0307_nonDL0311_nodvfs/mergecudaDL_nodvfs_throughput_total_labels_comb2.csv"  # Replace with your second file path
    
    # Optionally, specify columns to ignore (e.g., index columns)
    ignore_cols = None  # e.g., ['index'] if you want to ignore an index column
    
    result = are_csv_files_equal(file1, file2, ignore_columns=ignore_cols)
    print(f"Are the files equal? {result}")