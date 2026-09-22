# generate_sorted_workload_pairs_to_csv.py

import csv

# List of all workloads from the baseline file
workloads = [
    "mobilenet_batch2-train",
    "resnet-50_batch2-train",
    "vit_h_14_batch2-train",
    "albert-base-v2_batch2-train",
    "bert-base-cased_batch2-train",
    "whisper-large-v2_batch2-inf",
    "resnet-50_batch2-inf",
    "mobilenet_v2_1.0_224_batch2-inf",
    "vit-base-patch16-224_batch2-inf",
    "wav2vec2-base-960h_batch2-inf",
    "bert-base-cased_batch2-inf",
    "BlackScholes_batch2-cuda_samples",
    "sortingNetworks_batch2-cuda_samples",
    "transpose_batch2-cuda_samples",
    "cudaTensorCoreGemm_batch2-cuda_samples",
    "fastWalshTransform_batch2-cuda_samples",
    "reductionMultiBlockCG_batch2-cuda_samples"
]

# Workloads to exclude when both are in this list (non-DL workloads minus BlackScholes)
exclude_both_list = [
    "sortingNetworks_batch2-cuda_samples",
    "transpose_batch2-cuda_samples",
    "cudaTensorCoreGemm_batch2-cuda_samples",
    "fastWalshTransform_batch2-cuda_samples",
    "reductionMultiBlockCG_batch2-cuda_samples"
]

# Define DL workloads (first 11 in the list)
dl_workloads = workloads[:11]

def get_filtered_combinations(workloads, exclude_workload, exclude_both_list, dl_only=False):
    """
    Generate all unordered pairs, with different filtering based on dl_only flag:
    - If dl_only=False: Exclude pairs with exclude_workload and where both are in exclude_both_list.
    - If dl_only=True: Only include pairs where both workloads are in dl_workloads.
    Returns a list of tuples (workload1, workload2).
    """
    # Filter out the excluded workload
    filtered_workloads = [w for w in workloads if w != exclude_workload]
    pairs = []
    
    for i in range(len(filtered_workloads)):
        for j in range(i, len(filtered_workloads)):  # Start from i for unordered pairs
            w1 = filtered_workloads[i]
            w2 = filtered_workloads[j]
            if dl_only:
                # For DL-only, both workloads must be in dl_workloads
                if w1 in dl_workloads and w2 in dl_workloads:
                    pairs.append((w1, w2))
            else:
                # Original logic: exclude if both are in exclude_both_list
                if not (w1 in exclude_both_list and w2 in exclude_both_list):
                    pairs.append((w1, w2))
    
    return pairs

def save_to_csv(pairs, filename):
    """
    Save the sorted list of pairs to a CSV file with workload1 and workload2 columns.
    """
    # Sort pairs alphabetically by workload1, then workload2
    sorted_pairs = sorted(pairs, key=lambda x: (x[0], x[1]))
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["workload1", "workload2"])
        # Write each pair
        for w1, w2 in sorted_pairs:
            writer.writerow([w1, w2])
    print(f"Results saved to '{filename}' with {len(sorted_pairs)} pairs")

def main():
    # Generate combinations for original file (exclude BlackScholes and both in exclude_both_list)
    original_pairs = get_filtered_combinations(workloads, "BlackScholes_batch2-cuda_samples", exclude_both_list, dl_only=False)
    
    # Generate combinations for DL file (only pairs where both are DL workloads)
    dl_pairs = get_filtered_combinations(workloads, "BlackScholes_batch2-cuda_samples", exclude_both_list, dl_only=True)
    
    # Print total counts
    print(f"Total number of valid combinations (original): {len(original_pairs)}")
    print(f"Total number of valid DL-only combinations: {len(dl_pairs)}")
    
    # Save to original CSV file
    save_to_csv(original_pairs, "workload_pairs.csv")
    
    # Save to DL-specific CSV file
    save_to_csv(dl_pairs, "DL_all_possible_workload_pairs.csv")
    
    # Optional: Print the sorted pairs to console (commented out)
    # print("\nOriginal workload pairs (sorted):")
    # for idx, (w1, w2) in enumerate(sorted(original_pairs, key=lambda x: (x[0], x[1])), 1):
    #     print(f"{idx}. ({w1}, {w2})")
    # print("\nDL-only workload pairs (sorted):")
    # for idx, (w1, w2) in enumerate(sorted(dl_pairs, key=lambda x: (x
if __name__ == "__main__":
    main()