# workload_pairs_check_oneway.py
import sys
import pandas as pd

def load_workload_pairs(file_path):
    """
    Load workload pairs from a CSV file.
    Returns a set of tuples containing (workload1, workload2).
    """
    try:
        df = pd.read_csv(file_path)
        # Extract workload1 and workload2 columns
        pairs = set(zip(df['workload1'], df['workload2']))
        #print len(pairs)
        print(f"len of pairs of csv: {len(pairs)}")
        return pairs
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return set()
    except KeyError as e:
        print(f"Error: Missing expected column {e} in '{file_path}'.")
        return set()

def find_csv2_in_csv1(pairs1, pairs2):
    """
    Check if any pairs in pairs2 exist in pairs1, considering both (A, B) and (B, A).
    Returns a set of pairs from pairs2 that are found in pairs1.
    """
    common_pairs = set()
    for pair2 in pairs2:
        # Check if pair2 or its reverse exists in pairs1
        if pair2 in pairs1 or (pair2[1], pair2[0]) in pairs1:
            if (pair2[1], pair2[0]) in pairs1:
                print(f"{pairs1}should not  have reveresed pair!")
            common_pairs.add(pair2)
    return common_pairs

def main():
    # File paths for the two CSV files (replace with your actual file paths)
    csv_file1 = sys.argv[1]
    csv_file2 = sys.argv[2]

    # Load workload pairs from both files
    print(f"Loading pairs from '{csv_file1}'...")
    pairs1 = load_workload_pairs(csv_file1)
    print(f"Loading pairs from '{csv_file2}'...")
    pairs2 = load_workload_pairs(csv_file2)

    # Check if any pairs were loaded
    if not pairs1 or not pairs2:
        print("One or both files could not be loaded. Exiting.")
        return

    # Find pairs from csv2 that exist in csv1
    common_pairs = find_csv2_in_csv1(pairs1, pairs2)

    # Report results
    if common_pairs:
        print(f"\nFound the following workload pairs from '{csv_file2}' that exist in '{csv_file1}':")
        for pair in common_pairs:
            print(f" - {pair[0]}, {pair[1]}")
        print(f"Total common pairs: {len(common_pairs)}")
    else:
        print(f"\nNo workload pairs from '{csv_file2}' were found in '{csv_file1}'.")

if __name__ == "__main__":
    main()