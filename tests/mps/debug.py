#!/usr/bin/env python3

import sys
import pandas as pd

def filter_resnet50(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Filter condition:
    # Keep rows where workload1 == "resnet-50_batch2-inf" OR workload2 == "resnet-50_batch2-inf"
    mask = (df['workload1'] == 'resnet-50_batch2-inf') | (df['workload2'] == 'resnet-50_batch2-inf')
    filtered_df = df[mask]
    
    # Write the filtered DataFrame to a new CSV
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered CSV written to {output_csv}")

if __name__ == "__main__":
    # Usage: python filter_script.py input.csv output.csv
    if len(sys.argv) < 3:
        print("Usage: python filter_script.py <input_csv> <output_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    filter_resnet50(input_csv, output_csv)


