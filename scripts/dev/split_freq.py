#!/usr/bin/env python3
"""
split_csv_by_freq.py

A Python script that splits CSV files based on the 'freq1' column.
- Inputs: Directory path and file prefix to match CSVs.
- Outputs: Separate CSVs for each unique freq1 value, named with 'FREQXX_' prefix.
"""

import os
import glob
import pandas as pd
import argparse
import logging

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('split_csv_by_freq.log')
        ]
    )
    return logging.getLogger(__name__)

def find_csv_files(input_dir, prefix):
    """
    Find all CSV files in input_dir that match the given prefix.
    
    :param input_dir: Directory containing input CSV files
    :param prefix: Prefix to match CSV filenames
    :return: List of matching CSV file paths
    """
    pattern = os.path.join(input_dir, f"{prefix}*.csv")
    return glob.glob(pattern)

def split_csv_by_freq(input_csv, output_dir, logger):
    """
    Split a CSV file into separate CSVs based on the 'freq1' column.
    
    :param input_csv: Path to input CSV file
    :param output_dir: Directory to save output CSVs
    :param logger: Logger instance
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_csv)
        logger.info(f"Processing file: {input_csv}")
        
        # Check if 'freq1' column exists
        if 'freq1' not in df.columns:
            logger.error(f"'freq1' column not found in {input_csv}")
            return
        
        # Get unique freq1 values
        freq_values = df['freq1'].unique()
        logger.info(f"Found freq1 values: {freq_values}")
        
        # Get the base filename without path and extension
        base_filename = os.path.splitext(os.path.basename(input_csv))[0]
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Split and save CSVs for each freq1 value
        for freq in freq_values:
            if pd.isna(freq):
                logger.warning(f"Skipping NaN freq1 value in {input_csv}")
                continue
            # Filter rows for this freq1 value
            freq_df = df[df['freq1'] == freq]
            
            # Create output filename: FREQXX_originalname.csv
            output_filename = f"FREQ{freq}_{base_filename}.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save to CSV
            freq_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(freq_df)} rows to {output_path}")
            
    except Exception as e:
        logger.error(f"Error processing {input_csv}: {e}")

def main(input_dir, prefix, output_dir):
    """
    Main function to process all matching CSVs and split by freq1.
    
    :param input_dir: Directory containing input CSV files
    :param prefix: Prefix to match CSV filenames
    :param output_dir: Directory to save output CSVs
    """
    logger = setup_logging()
    logger.info(f"Starting CSV splitting with input_dir={input_dir}, prefix={prefix}, output_dir={output_dir}")
    
    # Find matching CSV files
    csv_files = find_csv_files(input_dir, prefix)
    if not csv_files:
        logger.warning(f"No CSV files found with prefix '{prefix}' in {input_dir}")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files: {csv_files}")
    
    # Process each CSV file
    for csv_file in csv_files:
        split_csv_by_freq(csv_file, output_dir, logger)
    
    logger.info("Finished processing all CSV files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split CSV files based on freq1 column")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input CSV files")
    parser.add_argument("--prefix", type=str, required=True,
                        help="Prefix to match CSV filenames")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output CSV files (default: output_freq_split)")
    
    args = parser.parse_args()
    
    # Run the main function
    main(args.input_dir, args.prefix, args.output_dir)