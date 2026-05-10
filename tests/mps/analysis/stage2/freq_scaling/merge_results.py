import pandas as pd
import sys
# Load the primary (1025) and secondary (1022) CSV files
#primary_file = '1025_share_comb2_freqscale_throughput_sum_avg_stage2.csv'
#secondary_file = '1022_share_comb2_freqscale_throughput_sum_avg_stage2.csv'
primary_file = sys.argv[1]
secondary_file = sys.argv[2]

df_primary = pd.read_csv(primary_file)
df_secondary = pd.read_csv(secondary_file)

# Filter the rows where frequency is 1530 in both datasets
df_primary_filtered = df_primary[df_primary['freq1'] == 1530]
df_secondary_filtered = df_secondary[df_secondary['freq1'] == 1530]

# Merge rows based on 'workload1', 'workload2', 'freq1', 'freq2'
merged_df = pd.merge(df_primary_filtered, df_secondary_filtered , on=['workload1', 'workload2', 'freq1', 'freq2'], how='outer')
# Prioritize columns from the primary file when columns overlap by renaming columns from secondary
for col in df_secondary.columns:
    if "workload" in col or "freq" in col:
        continue
    if col in df_primary.columns:  # Skip (w1_100, w2_100) from primary
        #df_merged[col] = df_merged[col + '_x'].combine_first(df_merged[col + '_y'])
        merged_df.drop([col + '_y'], axis=1, inplace=True)
        merged_df.rename(columns={col + '_x': col}, inplace=True)
 
# Save the merged result to a new CSV file
output_file = f'merged_primary{primary_file.split("/")[-1].split("_")[0]}_secondary{secondary_file.split("/")[-1].split("_")[0]}_freq1530_{primary_file.split("/")[-1]}'
print(f"Saving merged data to {output_file}")
merged_df.to_csv(output_file, index=False)
