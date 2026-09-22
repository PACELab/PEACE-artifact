import csv
import sys
def modify_header(input_csv, output_csv):
    with open(input_csv, mode='r') as infile, open(output_csv, mode='w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Get the first row (header)
        header = next(reader)
        
        # Modify the header
        new_header = []
        for col in header:
            if "LS" in col and "BE" in col:
                # Extract percentages from the current header
                ls_percent = col.split(",")[0].replace("(", "w1_").replace("LS", "").strip()
                be_percent = col.split(",")[1].replace(")", "").replace("BE", "w2_").strip()
                new_header.append(f"({ls_percent}, {be_percent})")
            else:
                # Keep workload1, workload2 unchanged
                new_header.append(col)
        
        # Write new header
        
        #move "(w1_100, w2_100)" to the third column
        new_header = new_header[:2] + [new_header[-1]] + new_header[2:-1]
        writer.writerow(new_header)
        # Write the rest of the data rows
        for row in reader:
            #reorder the row to match the new header
            row = row[:2] + [row[-1]] + row[2:-1]
            writer.writerow(row)

# Replace the file paths with the actual paths to your input/output CSV files
input_csv = sys.argv[1]
output_csv = sys.argv[2]

modify_header(input_csv, output_csv)
