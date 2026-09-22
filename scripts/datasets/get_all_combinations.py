import os
import pandas as pd
import argparse
from itertools import combinations_with_replacement, combinations

# Define the main function for processing kernels
def get_all_kernels(KERNEL_DIR, workloadlist, byworkloadlist, selected_batch_sizes):
    df_features = pd.DataFrame()
    for file in os.listdir(KERNEL_DIR):
        if file.endswith('.csv'):
            if byworkloadlist:
                # Extract the workload name from the filename
                if any(workload in file for workload in workloadlist):
                    df_temp = pd.read_csv(f'{KERNEL_DIR}/{file}')
                    df_features = pd.concat([df_features, df_temp], ignore_index=True)
            else:
                parts = file.split('_')
                batch_part = next((part for part in parts if part.startswith('batch')), None)
                if batch_part:
                    try:
                        batch_size = int(''.join(filter(str.isdigit, batch_part)))
                    except ValueError:
                        continue
                    if batch_size not in selected_batch_sizes:
                        continue
                    df_temp = pd.read_csv(f'{KERNEL_DIR}/{file}')
                    df_features = pd.concat([df_features, df_temp], ignore_index=True)
    return df_features

# Main function
def main(args):
    # Workload list (can be passed via arguments or can be set as default)
    WORKLOADLIST = [
        'whisper-large-v2_batch16-inf', 'whisper-large-v2_batch8-inf', 'whisper-large-v2_batch2-inf', 
        'bert-base-cased_batch16-inf', 'bert-base-cased_batch8-inf', 'vit-base-patch16-224_batch8-inf',
        'vit-base-patch16-224_batch2-inf', 'vit-base-patch16-224_batch16-inf', 'wav2vec2-base-960h_batch2-inf', 
        'wav2vec2-base-960h_batch16-inf', 'wav2vec2-base-960h_batch8-inf', 'bert-base-cased_batch2-inf', 
        'vit_h_14_batch8-train', 'vit_h_14_batch16-train', 'bert-base-cased_batch16-train', 'bert-base-cased_batch8-train', 
        'vit_h_14_batch2-train', 'albert-base-v2_batch2-train', 'albert-base-v2_batch8-train', 'albert-base-v2_batch16-train', 
        'gpt2-xl_batch20-inf', 'gpt2-xl_batch214-inf', 'gpt2-xl_batch100-inf', "mobilenet_batch2-train", 
        "mobilenet_batch8-train", "mobilenet_batch16-train", "mobilenet_v2_1.0_224_batch2-inf", "mobilenet_v2_1.0_224_batch8-inf", 
        "mobilenet_v2_1.0_224_batch16-inf", "resnet-50_batch2-train", "resnet-50_batch8-train", "resnet-50_batch16-train",
        "resnet-50_batch2-inf", "resnet-50_batch8-inf", "resnet-50_batch16-inf", "cudaTensorCoreGemm_batch2-cuda_samples", 
        "fastWalshTransform_batch2-cuda_samples", "BlackScholes_batch2-cuda_samples", "sortingNetworks_batch2-cuda_samples", 
        "reductionMultiBlockCG_batch2-cuda_samples", "transpose_batch2-cuda_samples", 
        "gpt2-xl_batch20-inf"
    ]

    # Read the batch sizes from the argparse argument
    SELECTED_BATCH_SIZE = [int(batch) for batch in args.batchsize]

    # Load baseline file and kernel profiles
    baseline_file = args.baseline_file
    filter_by_workloadlist = args.byworkloadlist

    # Read baseline file if provided
    if baseline_file:
        df_features = pd.read_csv(baseline_file)
        #should filter df_features by selected batch sizes
        df_features = df_features[df_features['Type'].str.contains('|'.join([f'batch{b}' for b in SELECTED_BATCH_SIZE]))]
    else:
        # Load kernels based on selected arguments
        df_features = get_all_kernels(args.kernel_dir, WORKLOADLIST, filter_by_workloadlist, SELECTED_BATCH_SIZE)

    # Print the 'Type' column
    print(df_features['Type'])

    # Label the 'Type' column in df_feature with labels based on index
    labels = {label: i for i, label in enumerate(df_features['Type'].unique())}
    print(labels)

    # Generate combinations of the dictionary keys
    if args.nonrepetitive:
        key_combinations = combinations(labels.keys(), args.num_combinations)
    else:
        key_combinations = combinations_with_replacement(labels.keys(), args.num_combinations)
    #sort the key combinations to make it easier to compare
    key_combinations = [sorted(comb) for comb in key_combinations]

    # Prepare the combined data
    data = []
    for w_combinations in key_combinations:
        combined_data = []
        combined_values = {}

        for idx, w in enumerate(w_combinations):
            value = labels[w]
            df_w = df_features[df_features['Type'] == w]

            # Drop 'Type' column and add prefix
            df_w = df_w.drop(['Type'], axis=1)
            df_w = df_w.add_prefix(f"w{idx+1}_")

            # Convert to dictionary and add to combined data
            combined_data.append(df_w.to_dict(orient='records')[0])
            combined_values[f"workload{idx+1}"] = w
            combined_values[f"idx{idx+1}"] = value

        # Combine all the dictionaries into one
        final_data = {**combined_values}
        for d in combined_data:
            final_data.update(d)

        data.append(final_data)

    # Create a DataFrame from the list of combined data
    df = pd.DataFrame(data)
    

    # Print the DataFrame (first few rows)
    print(df.head())
    #print df with na rows
    print(f"rows with NA values: {df.isna().any(axis=1).sum()}")
    print(f"miss value rows: {df[df.isna().any(axis=1)]}")

    # Write the DataFrame to CSV
    str_batches = "-".join(map(str, SELECTED_BATCH_SIZE))
    if args.baseline_file:
        df.to_csv(f'{args.output_prefix}_baseline_labels_comb{args.num_combinations}_batches{str_batches}.csv', header=True, index=False)
    elif args.kernel_dir:
        df.to_csv(f'{args.output_prefix}_kernel_labels_comb{args.num_combinations}_batches{str_batches}.csv', header=True, index=False)
    else:
        print("No baseline or kernel file specified to write to CSV.")


# Argument parsing with argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process kernel profiling CSVs and generate combinations.")
    parser.add_argument('--baseline_file', type=str, nargs='?', default=None, help="Optional path to the baseline CSV file.")
    parser.add_argument('--kernel_dir', type=str, help="[KACE only] Path to the directory containing kernel profile CSV files.")
    parser.add_argument('--num_combinations', "-comb",type=int, help="Number of combinations of workloads.")
    parser.add_argument('--output_prefix', type=str, help="Prefix for the output CSV file.")
    parser.add_argument('--batchsize', type=int, nargs='+', default=[2], help="List of selected batch sizes to filter by.")
    parser.add_argument('--byworkloadlist', action='store_true', help="[KACE only] Flag to filter by workload list (default is False).")
    parser.add_argument('--nonrepetitive', action='store_true', help="Flag to generate combinations without repetition (no duplicate workloads in same combination).")

    args = parser.parse_args()
    main(args)
