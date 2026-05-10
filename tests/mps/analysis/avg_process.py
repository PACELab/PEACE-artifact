import re
import sys
# Open the log file
with open(sys.argv[1], "r") as file:
    lines = file.readlines()

# Regular expression pattern to extract process times
pattern = r"request processing time: ([0-9.]+) seconds"

# Initialize variables to store total time and count of lines
total_time = 0
count = 0

# Iterate through the lines in the log file
for line in lines:
    match = re.search(pattern, line)  # Search for process time in the line
    if match:
        process_time = float(match.group(1))  # Extract process time
        total_time += process_time  # Add process time to total time
        count += 1  # Increment the count

# Calculate the average process time
if count > 0:
    avg_process_time = total_time / count
    print(f"Average process time: {avg_process_time:.6f} seconds")
else:
    print("No process times found in the log file.")
