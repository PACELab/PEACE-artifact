import re
import os
import glob
from datetime import datetime
import csv # Import csv module for potential header handling

PROFILE_FILENAME = "profile.txt"
TARGET_PREFIX = "BE_"
OUTPUT_DIR_PREFIX = "SM"
OUTPUT_DIR_SUFFIX = "percent"

def parse_timestamp(line):
    """
    Attempts to parse a timestamp from a log line using common formats.

    Args:
        line (str): The log line string.

    Returns:
        datetime | None: The parsed datetime object or None if no valid timestamp found.
    """
    # Regex to find potential timestamp patterns (YYYY-MM-DD HH:MM:SS,ms or YYYY-MM-DD HH:MM:SS)
    # It looks for the pattern commonly found at the start of log lines from various tools.
    # Adjust this regex if timestamps appear elsewhere or in different base formats.
    # Match YYYY-MM-DD HH:MM:SS potentially followed by ,ms or .ms
    match = re.search(r"(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}(?:[.,]\d{3,6})?)", line)
    if not match:
         # Try matching just YYYY-MM-DD HH:MM:SS if the above fails
         match = re.search(r"(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})", line)
         if not match:
              return None # No recognizable timestamp pattern found

    timestamp_str = match.group(1)

    # Try parsing with milliseconds/microseconds first
    for fmt in ["%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]:
        try:
            # Handle potential extra digits in microseconds if present
            if '.' in timestamp_str or ',' in timestamp_str:
                 ts_part, ms_part = timestamp_str.replace(',', '.').split('.')
                 timestamp_str_trimmed = f"{ts_part}.{ms_part[:6]}" # Keep up to 6 digits for microseconds
                 if fmt.endswith("%f"): # Only attempt formats with microseconds if they exist
                      return datetime.strptime(timestamp_str_trimmed, fmt)
            elif not fmt.endswith("%f"): # Only attempt format without microseconds if none exist
                 return datetime.strptime(timestamp_str, fmt)

        except ValueError:
            continue # Try the next format

    print(f"Warning: Could not parse timestamp from string: {timestamp_str} in line: {line.strip()}")
    return None # Return None if all formats fail


def parse_profile_log(profile_filename):
    """
    Parses the profile log to extract start and end times for each SM percentage.

    Args:
        profile_filename (str): The path to the profile.txt log file.

    Returns:
        dict: A dictionary where keys are percentages (int) and values are
              dicts containing 'start' and 'end' datetime objects.
              Returns None if the file cannot be read or parsing fails.
    """
    time_ranges = {}
    start_regex = re.compile(r"\[.*\] ([\d\- :]+) - INFO - Iteration: (\d+)% SMs")
    # Made end regex more flexible to match "completed" or "finished" lines
    end_regex = re.compile(r"\[.*\] ([\d\- :]+) - INFO - \s*(?:Kernel execution completed|Kernel finished for (\d+)%)")

    current_percent = None
    temp_start_time = None

    try:
        with open(profile_filename, 'r') as f:
            for line in f:
                start_match = start_regex.search(line)
                end_match = end_regex.search(line)

                if start_match:
                    timestamp_str = start_match.group(1)
                    percent = int(start_match.group(2))
                    try:
                        temp_start_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                        current_percent = percent
                        # Initialize dict for this percentage if not seen before
                        if current_percent not in time_ranges:
                             time_ranges[current_percent] = {'start': None, 'end': None}
                        time_ranges[current_percent]['start'] = temp_start_time
                        # print(f"Debug: Found start for {current_percent}% at {temp_start_time}") # Debug print
                    except ValueError:
                        print(f"Warning: Could not parse start timestamp '{timestamp_str}' for {percent}%")
                        temp_start_time = None
                        current_percent = None

                elif end_match and current_percent is not None:
                     timestamp_str = end_match.group(1)
                     # Check if the percentage was captured in the end_match (group 2)
                     # If not, assume it corresponds to the current_percent from the start line
                     end_percent_str = end_match.group(2)
                     end_percent = int(end_percent_str) if end_percent_str else current_percent

                     if end_percent == current_percent: # Ensure end line matches the current iteration
                         try:
                             end_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                             if current_percent in time_ranges and time_ranges[current_percent]['start'] is not None:
                                 time_ranges[current_percent]['end'] = end_time
                                 # print(f"Debug: Found end for {current_percent}% at {end_time}") # Debug print
                                 # Reset for next iteration
                                 current_percent = None
                                 temp_start_time = None
                             else:
                                  print(f"Warning: Found end time for {end_percent}% but no corresponding start time was stored.")
                         except ValueError:
                             print(f"Warning: Could not parse end timestamp '{timestamp_str}' for {end_percent}%")


    except FileNotFoundError:
        print(f"Error: Profile log file '{profile_filename}' not found.")
        return None
    except Exception as e:
        print(f"Error reading or parsing profile log '{profile_filename}': {e}")
        return None

    # Validation: Check if all percentages have start and end times
    all_found = True
    for percent in range(10, 101, 10):
        if percent not in time_ranges or not time_ranges[percent].get('start') or not time_ranges[percent].get('end'):
            print(f"Warning: Missing start or end time for {percent}% SMs iteration in {profile_filename}")
            all_found = False
            # Optionally remove incomplete entries
            if percent in time_ranges:
                 del time_ranges[percent]


    if not time_ranges:
         print("Error: No valid time ranges could be extracted from profile log.")
         return None

    print(f"Successfully parsed {len(time_ranges)} time ranges from {profile_filename}.")
    return time_ranges


def process_log_file(source_filename, time_ranges):
    """
    Splits a source log file based on the provided time ranges.

    Args:
        source_filename (str): Path to the source log file (e.g., BE_*.csv/txt).
        time_ranges (dict): Dictionary of time ranges parsed from profile.txt.
    """
    print(f"\nProcessing file: {source_filename}...")
    output_files = {}
    output_writers = {} # For CSV files
    header = None
    is_csv = source_filename.lower().endswith(".csv")

    try:
        # --- Prepare output files and directories ---
        base_name, ext = os.path.splitext(source_filename)
        for percent, times in time_ranges.items():
            if not times.get('start') or not times.get('end'):
                print(f"Skipping {percent}% due to missing start/end time.")
                continue

            output_dir = f"{OUTPUT_DIR_PREFIX}_{percent}_{OUTPUT_DIR_SUFFIX}"
            os.makedirs(output_dir, exist_ok=True) # Create subdir if it doesn't exist

            output_filename = os.path.join(output_dir, f"{base_name}_{percent}pct{ext}")
            try:
                # Open file handle and store it
                output_files[percent] = open(output_filename, 'w', newline='' if is_csv else None, encoding='utf-8')
                print(f"  Created output file: {output_filename}")
                # If CSV, create a writer object
                if is_csv:
                    output_writers[percent] = csv.writer(output_files[percent])
            except IOError as e:
                print(f"Error: Could not open output file {output_filename}: {e}")
                # Clean up already opened files for this source_filename if one fails
                for f in output_files.values():
                    f.close()
                return # Stop processing this source file

        # --- Read source file and write to splits ---
        line_count = 0
        processed_count = 0
        with open(source_filename, 'r', encoding='utf-8') as infile:
            if is_csv:
                # Read header and write to all output CSVs
                try:
                    csv_reader = csv.reader(infile)
                    header = next(csv_reader)
                    for writer in output_writers.values():
                        writer.writerow(header)
                    line_count = 1 # Start counting after header
                except StopIteration: # Handle empty CSV
                     print(f"  Warning: Input file {source_filename} is empty or has no header.")
                     header = [] # Mark as handled
                except Exception as e:
                     print(f"  Error reading CSV header from {source_filename}: {e}")
                     # Clean up and exit processing this file
                     for f in output_files.values(): f.close()
                     return

            # Process remaining lines (or all lines if TXT)
            for line in infile:
                line_count += 1
                line_time = parse_timestamp(line)

                if line_time:
                    written = False
                    for percent, times in time_ranges.items():
                         # Ensure we have valid times for comparison
                         if not times.get('start') or not times.get('end'):
                              continue

                         if times['start'] <= line_time <= times['end']:
                            try:
                                if is_csv:
                                     # For CSV, we assume the reader already split the line
                                     # Re-read the line for splitting? No, just write raw line for simplicity now.
                                     # If proper CSV splitting is needed per line, need csv.reader here.
                                     # Writing raw line is safer if format varies slightly.
                                     output_files[percent].write(line)
                                else:
                                     output_files[percent].write(line)
                                written = True
                                break # Line belongs to this time range
                            except Exception as e:
                                 print(f"Error writing line {line_count} to {output_files[percent].name}: {e}")
                                 # Consider whether to stop or continue
                    if written:
                         processed_count += 1
                # else: # Optional: Handle lines without timestamps if needed
                #    print(f"  Skipping line {line_count} (no timestamp): {line.strip()}")


    except FileNotFoundError:
        print(f"Error: Source log file '{source_filename}' not found.")
    except Exception as e:
        print(f"Error processing file '{source_filename}': {e}")
    finally:
        # --- Close all output files ---
        closed_count = 0
        for f in output_files.values():
            try:
                f.close()
                closed_count += 1
            except Exception as e:
                print(f"Error closing file {f.name}: {e}")
        print(f"  Finished processing {source_filename}. Read {line_count} lines, wrote {processed_count} lines to {closed_count} split files.")


def main():
    """
    Main function to orchestrate the log splitting process.
    """
    print("Starting log splitting process...")

    # 1. Parse profile log for time ranges
    time_ranges = parse_profile_log(PROFILE_FILENAME)
    if not time_ranges:
        print("Exiting due to errors parsing profile log.")
        return

    # Print parsed ranges for verification
    #for percent, times in sorted(time_ranges.items()):
    #    print(f"  {percent}%: Start={times.get('start')}, End={times.get('end')}")

    # 2. Find target log files
    target_files = glob.glob(f"{TARGET_PREFIX}*")
    if not target_files:
        print(f"No target files found matching prefix '{TARGET_PREFIX}' in the current directory.")
        return

    print(f"\nFound {len(target_files)} target files to process:")
    for fname in target_files:
        print(f"  - {fname}")

    # 3. Process each target file
    for filename in target_files:
        process_log_file(filename, time_ranges)

    print("\nLog splitting process complete.")

if __name__ == "__main__":
    main()
