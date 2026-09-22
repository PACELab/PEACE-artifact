import pandas as pd
import sys
import argparse
import os

def merge_csv_files(original_csv_path, missing_csv_path, output_csv_path):
    try:
        # Read both CSV files
        df_original = pd.read_csv(original_csv_path)
        df_missing = pd.read_csv(missing_csv_path)

        # Detect which format we're dealing with based on available columns
        format1_keys = ['Workload1', 'Workload2', 'Thread_combination']
        format2_keys = ['workload1', 'workload2', 'w1_Threadpercent', 'w2_Threadpercent']
        
        # Determine key columns based on what's present in original CSV
        if all(col in df_original.columns for col in format1_keys):
            key_columns = format1_keys
            column_order = df_original.columns.tolist()  # Preserve original column order
            format_name = "Format 1 - labels "
        elif all(col in df_original.columns for col in format2_keys):
            key_columns = format2_keys
            column_order = df_original.columns.tolist()  # Preserve original column order
            format_name = "Format 2 - train features"
        else:
            raise ValueError("Unable to determine CSV format - missing required key columns")

        # Verify required columns exist in both files
        for df, name in [(df_original, 'original'), (df_missing, 'missing')]:
            missing_cols = [col for col in key_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in {name} CSV for {format_name}: {missing_cols}")

        # Merge the dataframes
        df_original_indexed = df_original.set_index(key_columns)
        df_missing_indexed = df_missing.set_index(key_columns)
        
        # Update original with missing entries (missing takes priority)
        df_merged = df_missing_indexed.combine_first(df_original_indexed)
        
        # Reset index to turn key columns back into regular columns
        df_merged = df_merged.reset_index()
        
        # Ensure all expected columns exist and are in correct order
        df_merged = df_merged[column_order]
        
        # Save to new CSV file
        df_merged.to_csv(output_csv_path, index=False)
        print(f"Merged CSV saved to: {output_csv_path}")
        print(f"Format detected: {format_name}")
        print(f"Total rows in merged file: {len(df_merged)}")
        print(f"Rows from original: {len(df_original)}")
        print(f"Rows from missing: {len(df_missing)}")
        duplicate_count = len(df_original.merge(df_missing, on=key_columns, how='inner'))
        print(f"Number of duplicates found: {duplicate_count}")
        
    except pd.errors.EmptyDataError:
        raise ValueError("One or both input CSV files are empty")
    except pd.errors.ParserError:
        raise ValueError("Error parsing CSV files - check file format and contents")
    except KeyError as e:
        raise ValueError(f"Column error: {str(e)}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Merge two CSV files, prioritizing missing entries for duplicates"
    )
    parser.add_argument(
        "--original_csv",
        help="Path to the original CSV file"
    )
    parser.add_argument(
        "--missing_csv",
        help="Path to the missing entries CSV file"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="merged_output.csv",
        help="Path for the output merged CSV file (default: merged_output.csv)"
    )

    try:
        # Parse arguments
        args = parser.parse_args()
        
        # Verify input files exist
        if not os.path.isfile(args.original_csv):
            raise FileNotFoundError(f"Original CSV file not found: {args.original_csv}")
        if not os.path.isfile(args.missing_csv):
            raise FileNotFoundError(f"Missing entries CSV file not found: {args.missing_csv}")
        
        # Verify files have .csv extension
        if not args.original_csv.endswith('.csv') or not args.missing_csv.endswith('.csv'):
            raise ValueError("Input files must have .csv extension")
        
        # Call merge function
        merge_csv_files(args.original_csv, args.missing_csv, args.output)
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()