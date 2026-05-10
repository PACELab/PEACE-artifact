# Creating a Python script to process the CSV as requested


import pandas as pd
import sys
def filter_csv(input_csv, output_csv):
    # Load the CSV file
    df = pd.read_csv(input_csv)
    
    # Filter out rows where 'powercap50_50' is not null
    filtered_df = df[df['powercap50_50'].isnull()]
    
    # Save the filtered DataFrame to a new CSV
    filtered_df.to_csv(output_csv, index=False)
    
    print(f"Filtered data saved to {output_csv}")


def fill_powercap(nopower_data, label_data, output_csv):
    nopower_df = pd.read_csv((nopower_data))
    label_df = pd.read_csv((label_data))
    
    # Filter label_df for (w1_100, w2_100) entries (this is the source of Power values)
    w100_df = label_df[label_df['Thread_combination'] == "(w1_100, w2_100)"]
    #replace cuda_samples to samples in Workload1 and Workload2
    w100_df['Workload1'] = w100_df['Workload1'].str.replace('cuda_samples', 'samples')
    w100_df['Workload2'] = w100_df['Workload2'].str.replace('cuda_samples', 'samples')
    # Create a dictionary from label_df for workload pairs and Power values
    power_dict = {}
    for _, row in w100_df.iterrows():
        key = (row['Workload1'], row['Workload2'])
        power_dict[key] = row['Power']
    print(power_dict)
    # Function to update powercap50_50 column in nopower_df
    def update_powercap(row):
        key = (row['workload1'], row['workload2'])
        if key in power_dict:
            return power_dict[key]  # Replace with Power from label_df if available
        return row.get('powercap50_50', None)  # Keep original powercap50_50 if no match (or None if missing)

    # Apply the update to nopower_df
    nopower_df['powercap50_50'] = nopower_df.apply(update_powercap, axis=1)

    # Save the updated DataFrame to a new CSV file
    #outputname = filledpower + filename of nopower_data
    import os
    nopower_df.to_csv("filledpower"+os.path.basename(nopower_data), index=False)

    # Print the updated rows for verification (showing workload pairs and powercap50_50)
    print("Rows in nopower_df with updated powercap50_50:")
    print(nopower_df[['workload1', 'workload2', 'powercap50_50']])

def find_matched_rows(target_file, all_data_file, output_csv):
    #get all workload pairs from target_file with Workload1 and Workload2
    target_df = pd.read_csv(target_file)
    all_data_df = pd.read_csv(all_data_file)
    #target_df replace cuda_samples to samples in Workload1 and Workload2
    target_df['Workload1'] = target_df['Workload1'].str.replace('cuda_samples', 'samples')
    target_df['Workload2'] = target_df['Workload2'].str.replace('cuda_samples', 'samples')
    #get all workload pairs from target_file with Workload1 and Workload2
    target_pairs = set(target_df[['Workload1', 'Workload2']].apply(tuple, axis=1))
    # Find rows in all_data_df with matching Workload1 and Workload2
    matched_rows = all_data_df[
        all_data_df.apply(lambda row: (row['workload1'], row['workload2']) in target_pairs, axis=1)
    ]
    #save the matched rows to a new CSV file
    matched_rows.to_csv(output_csv, index=False)
    return matched_rows


if __name__ == "__main__":
    # Provide the input CSV file path as an argument
    if len(sys.argv) < 2:
        print("Please provide the input CSV file path")
        print("Usage: python filter_csv.py input_csv output_csv")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = 'nopower_nonDL_data.csv'  # Replace with your desired output CSV file path

    #filter_csv(input_csv, output_csv)
    nopower_data = sys.argv[1]
    label_data = sys.argv[2]
    fill_powercap(nopower_data, label_data, output_csv)

    #target_file = sys.argv[1]
    #all_data_file = sys.argv[2]
    #output_csv = 'matched_rows_nopowercap_nonDL.csv'  # Replace with your desired output CSV file path
    #find_matched_rows(target_file, all_data_file, output_csv)



