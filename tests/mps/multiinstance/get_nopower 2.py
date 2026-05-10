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


def fill_powercap(nopower_data, label_data):
    nopower_df = pd.read_csv((nopower_data))
    label_df = pd.read_csv((label_data))

    # Filter nopower_data for (w1_100, w2_100) entries
    w100_df = label_df[label_df['Thread_combination'] == '"(w1_100, w2_100)"']

    # Create a dictionary from label_df for workload pairs and powercap50_50
    # Note: Since powercap50_50 is missing in some rows, we'll handle it with a fallback
    label_powercap = {}
    for _, row in nopower_df.iterrows():
        key = (row['workload1'], row['workload2'])
        # Check if powercap50_50 exists and is numeric; otherwise, set to None
        powercap = row.get('powercap50_50', None)
        label_powercap[key] = float(powercap) if pd.notna(powercap) and str(powercap).replace('.', '').isdigit() else None

    # Function to update Power column
    def update_power(row):
        key = (row['Workload1'], row['Workload2'])
        if key in label_powercap and label_powercap[key] is not None:
            return label_powercap[key]  # Replace with powercap50_50 if available
        return row['Power']  # Keep original Power if no match or powercap is None

    # Apply the update to the filtered DataFrame
    w100_df['Power'] = w100_df.apply(update_power, axis=1)

    # Merge back with the original DataFrame
    updated_df = label_df.copy()
    updated_df.loc[updated_df['Thread_combination'] == '"(w1_100, w2_100)"', 'Power'] = w100_df['Power']

    # Save the updated DataFrame to a new CSV file
    updated_df.to_csv('updated_nopower_data.csv', index=False)

    # Print the rows with (w1_100, w2_100) for verification
    print("Rows with (w1_100, w2_100) after update:")
    print(updated_df[updated_df['Thread_combination'] == '"(w1_100, w2_100)"'])

if __name__ == "__main__":
    # Provide the input CSV file path as an argument
    if len(sys.argv) < 2:
        print("Please provide the input CSV file path")
        print("Usage: python filter_csv.py input_csv output_csv")
        sys.exit(1)
    
    #input_csv = sys.argv[1]
    output_csv = 'nopower_nonDL_data.csv'  # Replace with your desired output CSV file path

    #filter_csv(input_csv, output_csv)
    nopower_data = sys.argv[1]
    label_data = sys.argv[2]
    fill_powercap(nopower_data, label_data)



