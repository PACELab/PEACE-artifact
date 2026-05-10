#!/usr/bin/env python3
"""
Filter CSV files to keep only specified MPS percentage combinations.
This script filters both feature and label files simultaneously to maintain alignment.
"""

import pandas as pd
import argparse
import os
from pathlib import Path


def parse_thread_combination(thread_comb_str):
    """
    Parse thread combination string to extract percentages.
    Example: "(w1_10, w2_30, w3_60)" -> (10, 30, 60)
    """
    # Remove parentheses and split by comma
    parts = thread_comb_str.strip("()").split(",")
    percentages = []
    for part in parts:
        # Extract the number after the underscore
        percentage = int(part.strip().split("_")[1])
        percentages.append(percentage)
    return tuple(percentages)


def filter_by_sampled_combinations(features_file, labels_file, output_dir=None):
    """
    Filter both CSV files to keep only the sampled combinations.

    Args:
        features_file: Path to features CSV file
        labels_file: Path to labels CSV file
        output_dir: Output directory (defaults to same as input files)
    """
    # Define sampled combinations
    comb3_sampled = [
        "(w1_100, w2_100, w3_100)",
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
    ]

    # Parse sampled combinations to tuples for easier matching
    sampled_tuples = [parse_thread_combination(comb) for comb in comb3_sampled]
    print(f"Sampled combinations: {sampled_tuples}")

    # Read both CSV files
    print(f"Reading {features_file}...")
    df_features = pd.read_csv(features_file)
    print(f"Features file shape: {df_features.shape}")

    print(f"Reading {labels_file}...")
    df_labels = pd.read_csv(labels_file)
    print(f"Labels file shape: {df_labels.shape}")

    # Verify both files have the same number of rows
    if len(df_features) != len(df_labels):
        raise ValueError(f"Files have different number of rows: {len(df_features)} vs {len(df_labels)}")

    # Create boolean mask based on thread percentage columns
    # Check if the files have w1_Threadpercent, w2_Threadpercent, w3_Threadpercent columns
    if 'w1_Threadpercent' in df_features.columns:
        print("Using w1_Threadpercent, w2_Threadpercent, w3_Threadpercent columns for filtering...")
        mask = df_features.apply(
            lambda row: (
                int(row['w1_Threadpercent']),
                int(row['w2_Threadpercent']),
                int(row['w3_Threadpercent'])
            ) in sampled_tuples,
            axis=1
        )
    # Otherwise, try to parse Thread_combination column from labels file
    elif 'Thread_combination' in df_labels.columns:
        print("Using Thread_combination column from labels file for filtering...")
        mask = df_labels['Thread_combination'].apply(
            lambda x: parse_thread_combination(x) in sampled_tuples
        )
    else:
        raise ValueError("Cannot find thread percentage columns or Thread_combination column")

    # Apply mask to both dataframes
    df_features_filtered = df_features[mask].reset_index(drop=True)
    df_labels_filtered = df_labels[mask].reset_index(drop=True)

    print(f"\nFiltered features shape: {df_features_filtered.shape}")
    print(f"Filtered labels shape: {df_labels_filtered.shape}")
    print(f"Kept {len(df_features_filtered)} out of {len(df_features)} rows ({len(df_features_filtered)/len(df_features)*100:.2f}%)")

    # Verify alignment
    if len(df_features_filtered) != len(df_labels_filtered):
        raise ValueError("Filtered dataframes have different lengths!")

    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(features_file)
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Generate output filenames
    features_basename = os.path.basename(features_file)
    labels_basename = os.path.basename(labels_file)

    # Add "_sampled" before the extension
    features_name, features_ext = os.path.splitext(features_basename)
    labels_name, labels_ext = os.path.splitext(labels_basename)

    output_features = os.path.join(output_dir, f"{features_name}_sampled{features_ext}")
    output_labels = os.path.join(output_dir, f"{labels_name}_sampled{labels_ext}")

    # Save filtered dataframes
    print(f"\nSaving filtered features to {output_features}...")
    df_features_filtered.to_csv(output_features, index=False)

    print(f"Saving filtered labels to {output_labels}...")
    df_labels_filtered.to_csv(output_labels, index=False)

    print("\nFiltering complete!")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    if 'Thread_combination' in df_labels_filtered.columns:
        print("\nCombination counts in filtered data:")
        print(df_labels_filtered['Thread_combination'].value_counts().sort_index())

    return output_features, output_labels


def main():
    parser = argparse.ArgumentParser(
        description="Filter CSV files to keep only sampled MPS percentage combinations"
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Path to features CSV file"
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Path to labels CSV file"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (defaults to same as input files)"
    )

    args = parser.parse_args()

    filter_by_sampled_combinations(args.features, args.labels, args.output_dir)


if __name__ == "__main__":
    main()
