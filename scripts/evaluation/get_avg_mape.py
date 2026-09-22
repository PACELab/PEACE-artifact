import os
import re
import argparse

def extract_mape_from_file_content(file_content):
    """
    Extracts the MAPE value from the content of a metrics file.
    Example line: "MAPE: 1.44%"
    """
    for line in file_content.splitlines():
        if line.startswith("MAPE:"):
            try:
                # Extract the numeric part and remove the '%' sign
                mape_str = line.split(":")[1].strip().rstrip('%')
                return float(mape_str)
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse MAPE from line: '{line}'. Error: {e}")
                return None
    return None # MAPE line not found

def analyze_metrics_from_directory(base_directory_path):
    """
    Analyzes a directory structure to find metric files, read their MAPE values,
    and calculate average MAPE for different categories.
    """
    results = {
        "seen_power": {"mapes": [], "count": 0, "paths": []},
        "seen_throughput": {"mapes": [], "count": 0, "paths": []},
        "unseen_power": {"mapes": [], "count": 0, "paths": []},
        "unseen_throughput": {"mapes": [], "count": 0, "paths": []},
    }

    # Walk through the directory tree
    for root, dirs, files in os.walk(base_directory_path):
        for filename in files:
            full_file_path = os.path.join(root, filename)
            
            fill_file_path_splits = full_file_path.split(os.sep)
            #print(f"fill_file_path_splits: {fill_file_path_splits}")
            # Normalize path separators for consistent checking
            normalized_path = full_file_path.replace("\\", "/")
            
            is_seen_partition = "seen_partition" in fill_file_path_splits
            is_unseen_partition = "unseen_partition" in fill_file_path_splits
            
            # Determine if it's a power or throughput metric file and context
            # The subdirectories "power" or "throughput" are key context.
            is_power_context = "/power/" in normalized_path
            is_throughput_context = "/throughput/" in normalized_path

            mape_value = None
            category_key = None

            if filename == "pred_metrics_power_regression.txt" and "extratrees" in fill_file_path_splits:
                if not is_power_context:
                    # print(f"Warning: '{filename}' found outside a 'power' directory: {full_file_path}")
                    # Decide if this should be skipped or categorized differently. For now, we require context.
                    continue 
                try:
                    with open(full_file_path, 'r') as f:
                        content = f.read()
                    mape_value = extract_mape_from_file_content(content)
                except Exception as e:
                    print(f"Error reading or parsing file {full_file_path}: {e}")
                    continue

                if mape_value is not None:
                    if is_seen_partition:
                        category_key = "seen_power"
                        print(f"root: {root} filename: {filename} seen_power")
                    elif is_unseen_partition:
                        category_key = "unseen_power"
            
            elif filename == "pred_metrics_separate_throughputpower_regression.txt" and "extratrees" in fill_file_path_splits:
                if not is_throughput_context:
                    # print(f"Warning: '{filename}' found outside a 'throughput' directory: {full_file_path}")
                    # Decide if this should be skipped or categorized differently. For now, we require context.
                    continue
                try:
                    with open(full_file_path, 'r') as f:
                        content = f.read()
                    mape_value = extract_mape_from_file_content(content)
                except Exception as e:
                    print(f"Error reading or parsing file {full_file_path}: {e}")
                    continue
                
                if mape_value is not None:
                    if is_seen_partition:
                        category_key = "seen_throughput"
                    elif is_unseen_partition:
                        category_key = "unseen_throughput"

            # Store results if a category was determined and MAPE was found
            if category_key and mape_value is not None:
                results[category_key]["mapes"].append(mape_value)
                results[category_key]["count"] += 1
                results[category_key]["paths"].append(full_file_path)
            elif (filename == "pred_metrics_power_regression.txt" or \
                  filename == "pred_metrics_separate_throughputpower_regression.txt") and \
                 mape_value is None:
                print(f"Warning: MAPE value not extracted from {full_file_path}")
    #print counts of each category
    for category, data in results.items():
        print(f"Category '{category}' has {data['count']} files processed.")

    # Calculate averages
    averages = {}
    for category, data in results.items():
        if data["mapes"]:
            averages[category] = sum(data["mapes"]) / len(data["mapes"])
        else:
            averages[category] = 0.0  # Or float('nan') if you prefer for no data
            
    return averages, results

# --- Main execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate average MAPE from metric files in a directory structure."
    )
    parser.add_argument(
        "--input_dir", 
        type=str, 
        help="The root directory containing the experiment results (e.g., 05052025_FREQ900_nodvfs)."
    )
    
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Directory not found: {args.input_dir}")
        exit(1)

    print(f"Analyzing directory: {args.input_dir}\n")
    average_mapes, detailed_results = analyze_metrics_from_directory(args.input_dir)

    print("Analysis Results:\n")
    total_files_processed = 0
    for category, avg_mape in average_mapes.items():
        count = detailed_results[category]["count"]
        total_files_processed += count
        print(f"Category: {category}")
        print(f"  Files Found & Processed: {count}")
        if count > 0:
            print(f"  Average MAPE: {avg_mape:.2f}%")
        else:
            print(f"  Average MAPE: N/A (no files found or processed for this category)")
        # Uncomment to see the paths of the files found for each category
        # if detailed_results[category]["paths"]:
        #     print(f"  Sample Paths Found:")
        #     for p in detailed_results[category]["paths"][:min(3, len(detailed_results[category]["paths"]))]: # Print first 3 paths
        #         print(f"    {p}")
        print("-" * 40)
    
    print(f"Total metric files processed: {total_files_processed}")

