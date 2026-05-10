import pandas as pd
import sys
# Load the two files into DataFrames
#args explaination
#sys.argv[0] is the name of the script

primary = sys.argv[1]
secondary = sys.argv[2]
output_path = sys.argv[3]
#error handle
if primary is None or secondary is None:
    raise ValueError("Please provide paths to the primary and secondary files.")


primary_data = pd.read_csv(primary)
secondary_data = pd.read_csv(secondary)


# Ensure workload1 and workload2 columns exist in both files
p_col = [i.lower() for i in primary_data.columns]
s_col = [i.lower() for i in secondary_data.columns]
if "workload1" not in s_col or "workload2" not in s_col:
    raise ValueError("1229 file is missing 'workload1' or 'workload2' columns.")

if "workload1" not in p_col or "workload2" not in s_col:
    raise ValueError("1222 file is missing 'workload1' or 'workload2' columns.")
if "workload1" in primary_data.columns:
    workload_prefix = "workload"
else:
    workload_prefix = "Workload"
# Convert (workload1, workload2) into keys for merging
secondary_data['key'] = list(zip(secondary_data[f'{workload_prefix}1'], secondary_data[f'{workload_prefix}2']))
primary_data['key'] = list(zip(primary_data[f'{workload_prefix}1'], primary_data[f'{workload_prefix}2']))

# Find keys that are in 1229 but not in 1222
keys_1222 = set(primary_data['key'])
new_entries = secondary_data[~secondary_data['key'].isin(keys_1222)]
print(f"Found {len(new_entries)} new entries.")
print(f"Keys in 1229 but not in 1222: {new_entries['key']}")

# Append new entries to 1222 data
merged_data = pd.concat([primary_data, new_entries], ignore_index=True)

# Drop the temporary 'key' column
merged_data = merged_data.drop(columns=['key'])


#deduplicate sort by workload1 and workload2


# Generate a canonical sorted key for each entry to identify duplicates regardless of order
merged_data['canonical_key'] = merged_data.apply(
    lambda row: tuple(sorted((row[f'{workload_prefix}1'], row[f'{workload_prefix}2']))), axis=1
)
print(f"orifginal length: {len(merged_data)}")
# Remove duplicates based on the canonical key, keeping the first occurrence
deduplicated_data = merged_data.drop_duplicates(subset='canonical_key', keep='first')
print(f"deduplicated length: {len(deduplicated_data)}")
# Drop the temporary 'canonical_key' column
deduplicated_data = deduplicated_data.drop(columns=['canonical_key'])

# Save the deduplicated data

deduplicated_data.to_csv(output_path, index=False)


