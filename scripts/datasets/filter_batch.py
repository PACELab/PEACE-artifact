import pandas as pd
import sys
# Reading the CSV file
#csv_file = "02062025_freq1530_baseline_labels_comb2_batches2-8-16.csv"
csv_file = sys.argv[1]
df = pd.read_csv(csv_file)

# Filtering rows where either workload1 or workload2 contain "cuda_samples"
cuda_samples_df = df[(df['workload1'].str.contains('cuda_samples', case=False)) | 
                    (df['workload2'].str.contains('cuda_samples', case=False))]
#exclude rows with cuda_samples in both rows
cuda_samples_df = cuda_samples_df[~((cuda_samples_df['workload1'].str.contains('cuda_samples', case=False)) & 
                                     (cuda_samples_df['workload2'].str.contains('cuda_samples', case=False)))]
# Filtering out rows where both workload1 and workload2 contain "batch2"
df = df[~((df['workload1'].str.contains('batch2', case=False)) & 
          (df['workload2'].str.contains('batch2', case=False)))]

# Filtering out rows where both workload1 and workload2 contain "batch16"
df = df[~((df['workload1'].str.contains('batch16', case=False)) & 
          (df['workload2'].str.contains('batch16', case=False)))]

# Filtering out rows where either workload1 or workload2 contain "cuda_samples" from main output
df = df[~((df['workload1'].str.contains('cuda_samples', case=False)) | 
          (df['workload2'].str.contains('cuda_samples', case=False)))]

# Saving the filtered data to a new CSV file
#output_file is input file name with "_filtered" appended
output_file = csv_file.replace('.csv', '_filtered_0515.csv')
df.to_csv(output_file, index=False)

# Saving the cuda_samples data to a separate CSV file
#output_file is input file name with "_cuda_samples" appended
cuda_output_file = csv_file.replace('.csv', '_cuda_samples_0515.csv')
cuda_samples_df.to_csv(cuda_output_file, index=False)

print(f"Filtered data saved to {output_file}")
print(f"CUDA samples data saved to {cuda_output_file}")
print(f"Original number of rows: {len(pd.read_csv(csv_file))}")
print(f"Filtered number of rows: {len(df)}")
print(f"CUDA samples number of rows: {len(cuda_samples_df)}")