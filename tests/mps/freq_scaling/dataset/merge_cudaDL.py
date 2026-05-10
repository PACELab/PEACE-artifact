#!/usr/bin/env python3

import os
import shutil
import pandas as pd
import sys
from pathlib import Path

def create_output_folder(output_name):
    """Create output folder if it doesn't exist"""
    if os.path.exists(output_name):
        print(f"Warning: Output folder '{output_name}' already exists. Contents may be overwritten.")
    else:
        os.makedirs(output_name)
    return output_name

def copy_txt_files(folder1, folder2, output_folder):
    """Copy all .txt files from both folders to output folder"""
    print("Copying .txt files...")

    for folder in [folder1, folder2]:
        for file in os.listdir(folder):
            if file.endswith('.txt'):
                src_path = os.path.join(folder, file)
                dst_path = os.path.join(output_folder, file)
                shutil.copy2(src_path, dst_path)
                print(f"  Copied: {file}")

def copy_other_files(folder1, folder2, output_folder):
    """Copy non-CSV, non-TXT files (like folders)"""
    print("Copying other files/folders...")

    for folder in [folder1, folder2]:
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            if os.path.isdir(item_path):
                dst_path = os.path.join(output_folder, item)
                if not os.path.exists(dst_path):
                    shutil.copytree(item_path, dst_path)
                    print(f"  Copied folder: {item}")

def merge_labels_csv(folder1, folder2, output_folder):
    """Merge CSV files ending with 'labels.csv'"""
    print("Merging *labels.csv files...")

    labels_files = []
    for folder in [folder1, folder2]:
        for file in os.listdir(folder):
            if file.endswith('labels.csv'):
                labels_files.append((folder, file))

    if len(labels_files) != 2:
        print(f"Warning: Expected 2 *labels.csv files, found {len(labels_files)}")
        return

    dfs = []
    for folder, file in labels_files:
        file_path = os.path.join(folder, file)
        df = pd.read_csv(file_path)
        dfs.append(df)
        print(f"  Loaded: {file} ({len(df)} rows)")

    merged_df = pd.concat(dfs, ignore_index=True)
    output_file = os.path.join(output_folder, "merged_labels.csv")
    merged_df.to_csv(output_file, index=False)
    print(f"  Merged labels.csv saved: {output_file} ({len(merged_df)} rows)")

def merge_total_labels_comb2_csv(folder1, folder2, output_folder):
    """Merge CSV files ending with 'total_labels_comb2.csv' with column validation"""
    print("Merging *total_labels_comb2.csv files...")

    comb2_files = []
    for folder in [folder1, folder2]:
        for file in os.listdir(folder):
            if file.endswith('total_labels_comb2.csv'):
                comb2_files.append((folder, file))

    if len(comb2_files) != 2:
        print(f"Warning: Expected 2 *total_labels_comb2.csv files, found {len(comb2_files)}")
        return

    dfs = []
    column_sets = []

    for folder, file in comb2_files:
        file_path = os.path.join(folder, file)
        df = pd.read_csv(file_path)
        dfs.append(df)
        column_sets.append(set(df.columns))
        print(f"  Loaded: {file} ({len(df)} rows, {len(df.columns)} columns)")

    # Check for column name mismatches
    if column_sets[0] != column_sets[1]:
        missing_in_first = column_sets[1] - column_sets[0]
        missing_in_second = column_sets[0] - column_sets[1]

        error_msg = "Column name mismatch detected:\n"
        if missing_in_first:
            error_msg += f"  Columns in second file but not first: {sorted(missing_in_first)}\n"
        if missing_in_second:
            error_msg += f"  Columns in first file but not second: {sorted(missing_in_second)}\n"

        raise ValueError(error_msg)

    # Reorder columns to match the first dataframe
    first_df = dfs[0]
    second_df = dfs[1][first_df.columns]

    merged_df = pd.concat([first_df, second_df], ignore_index=True)
    output_file = os.path.join(output_folder, "merged_total_labels_comb2.csv")
    merged_df.to_csv(output_file, index=False)
    print(f"  Merged total_labels_comb2.csv saved: {output_file} ({len(merged_df)} rows)")

def copy_other_csv_files(folder1, folder2, output_folder):
    """Copy other CSV files that don't match the special patterns"""
    print("Copying other CSV files...")

    for folder in [folder1, folder2]:
        for file in os.listdir(folder):
            if (file.endswith('.csv') and
                not file.endswith('labels.csv') and
                not file.endswith('total_labels_comb2.csv')):
                src_path = os.path.join(folder, file)
                dst_path = os.path.join(output_folder, file)
                shutil.copy2(src_path, dst_path)
                print(f"  Copied: {file}")

def main():
    folder1 = "10032025_sharenonDL_powercap250_dvfs"
    folder2 = "10032025_shareDL_powercap250_dvfs"
    output_folder = "10032025_mergecudaDL_powercap250_dvfs"

    # Validate input folders exist
    if not os.path.exists(folder1):
        print(f"Error: Folder '{folder1}' does not exist")
        sys.exit(1)
    if not os.path.exists(folder2):
        print(f"Error: Folder '{folder2}' does not exist")
        sys.exit(1)

    print(f"Merging data from:")
    print(f"  Folder 1: {folder1}")
    print(f"  Folder 2: {folder2}")
    print(f"  Output:   {output_folder}")
    print()

    # Create output folder
    create_output_folder(output_folder)

    try:
        # 1. Copy txt files
        copy_txt_files(folder1, folder2, output_folder)

        # 2. Merge labels.csv files
        merge_labels_csv(folder1, folder2, output_folder)

        # 3. Merge total_labels_comb2.csv files with validation
        merge_total_labels_comb2_csv(folder1, folder2, output_folder)

        # 4. Copy other CSV files
        copy_other_csv_files(folder1, folder2, output_folder)

        # 5. Copy other files/folders
        copy_other_files(folder1, folder2, output_folder)

        print(f"\nMerge completed successfully! Output saved to: {output_folder}")

    except Exception as e:
        print(f"\nError during merge: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()