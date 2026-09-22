import pandas as pd
import sys
import os
from collections import defaultdict
import re
from datetime import datetime
import itertools
import numpy as np
def generate_thread_combinations(n_combinations):
    # Define the set of possible values for thread percentages
    possible_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Create a list to store the valid combinations of thread percentages for n_combination workloads
    thread_combinations = [tuple([100] * n_combinations)]
    for combination in itertools.product(possible_values, repeat=n_combinations):
        if sum(combination) == 100:
            thread_combinations.append(combination)
    return thread_combinations

def calculate_average_power_duration_adjusted(file, idle_power):
    
    try:
        print(file)
        # Read the CSV file with full path
        df = pd.read_csv(file)
        #df = pd.read_csv(file)
        
        # Strip leading/trailing spaces from column names
        df.columns = df.columns.str.strip()
        
        # Remove the ' W' unit and convert 'power.draw [W]' to float
        df['power.draw [W]'] = df['power.draw [W]'].str.replace(' W', '').astype(float)
        
        #print(df['power.draw [W]'].tolist())
        #drop power below idle power
        df = df[df['power.draw [W]'] > idle_power]
        if len(df) == 0:
            print(f"no power data above idle power {idle_power}")
            avg_power = None
            total_duration = None
            total_energy = None
        else:
            # Calculate the average of 'power.draw [W]'
            avg_power = df['power.draw [W]'].mean()

            # Convert the timestamp column to datetime format
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M:%S.%f')

            # Get the total time duration by subtracting the first timestamp from the last timestamp
            start_time = df['timestamp'].iloc[0]
            end_time = df['timestamp'].iloc[-1]
            total_duration = (end_time - start_time).total_seconds()

            # Calculate time intervals in seconds
            df['time_interval'] = df['timestamp'].diff().dt.total_seconds()
            
            # Calculate energy for each row: energy = power * time_interval
            df['energy [J]'] = df['power.draw [W]'] * df['time_interval']

            # Calculate total energy (sum of energy values)
            total_energy = df['energy [J]'].sum()

            print(f"Total energy: {total_energy} Joules")
            print(f"avg power * duration= {avg_power * total_duration}")
            print(f"Total duration: {total_duration} seconds")
       
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return avg_power, total_duration, total_energy

def calculate_average_sm_memory(filename):
    # Read the log file into a DataFrame, skipping the header and footer
    df = pd.read_csv(filename, delim_whitespace=True)
    #drop rows with df['sm'] == '-' or df['mem'] == '%'
    #drop rows with df['sm']  not a number
    #drop first 15 rows if total rows > 15
    #drop last 15 rows if total rows > 15
    if len(df) > 12:
        df = df[10:]
        #delete last 1 row
        df = df[:-2]
    print(df['sm'].unique())
    for col in ['sm', 'mem']:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df.dropna(subset=[col], inplace=True)
    print("sm", df['sm'].unique())
    
    # Calculate the average of 'sm' and 'mem' columns
    avg_sm = df['sm'].astype(float).sum() / len(df)
    avg_mem = df['mem'].astype(float).sum() / len(df)
    return avg_sm, avg_mem

def read_average_sm_memory(filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)
    
    # Get the average sm and memory values
    avg_sm = df['average_sm%'].values[0]
    avg_mem = df['average_memory%'].values[0]
    
    return avg_sm, avg_mem
def save_to_csv(avg_sm, avg_mem, filename):
    # Create a DataFrame with the averages
    df_avg = pd.DataFrame({'sm%': [avg_sm], 'mem%': [avg_mem]})

    # Save the DataFrame to a new CSV file
    df_avg.to_csv(filename[:-4] + '_smi.csv', index=False)
    print(f"Average SM% and memory% saved to {filename[:-4] + '_smi.csv'}")

def calculate_average_gcgm_usage(file_name, columns):
    # Load the file

    # Read the file while handling headers and formatting issues
    with open(file_name, 'r') as file:
        lines = file.readlines()

    # Identify and remove redundant headers

    metric_names = ["SMACT%", "SMOCC%", "TENSO%", "DRAMA%", "FP64A%", "FP32A%", "FP16A%", 
        "PCITX", "PCIRX", "NVLTX", "NVLRX"]
    metric_dict = {metric: [] for metric in metric_names}
    idx = 1
    leading_zeros = True
    for line in lines:
        if "#Entity" in line or "ID" in line:
            idx += 1
            continue  # Skip redundant headers

        # Split line into values, skipping the first four columns (timestamp GPU ID and instance)
        feats = line.split()[4:]
        if "N/A" in feats:
            continue  # Skip lines with N/A values
        #print(feats)

        # Use SMACT as skipping indicator for leading zeros
        if float(feats[0]) == 0 and float(feats[1]) == 0 and leading_zeros:
            print(f"skip row {idx}")
            idx += 1
            continue 

        leading_zeros = False

        # Ensure number of metrics matches expected columns
        if len(metric_names) != len(feats):
            raise ValueError(f"Number of metrics ({len(feats)}) does not match the number of metric names ({len(metric_names)})")
        #print(f"feats: {feats}")
        #print(f"idx: {idx}, SMACT: {feats[0]}, SMOCC: {feats[1]}, TENSO: {feats[2]}, DRAMA: {feats[3]}, FP64A: {feats[4]}, FP32A: {feats[5]}, FP16A: {feats[6]}")

        # Store values in metric dictionary
        for i, feat in enumerate(feats):
            if i < 7:
                metric_dict[metric_names[i]].append(float(feat)*100)
            else:
                metric_dict[metric_names[i]].append(float(feat))
        
        idx += 1

    # Identify trailing zeros in SMACT and SMOCC and remove corresponding entries
    trim_idx = len(metric_dict["SMACT%"])
    for i in range(len(metric_dict["SMACT%"]) - 1, -1, -1):  # Iterate backward
        if metric_dict["SMACT%"][i] == 0 and metric_dict["SMOCC%"][i] == 0 :
            trim_idx = i  # Update index to trim at
        else:
            break  # Stop once a non-zero is encountered
    #print(f"trim_idx: {trim_idx}")
    

    # Trim all metrics in metric_dict to remove trailing zero rows
    for key in metric_dict.keys():
        metric_dict[key] = metric_dict[key][:trim_idx]
    print(f"after trim: {metric_dict}")
    # Convert the cleaned dictionary to a DataFrame
    df_mean = pd.DataFrame(metric_dict).mean()
    #print(f"mean: {df_mean}")
    #return df_mean with the mentioned columns
    print(f"selected returns: {df_mean[columns]}")
    return df_mean[columns]
    

    #remove trailing zeros by checking how many indexs in SMACT are 0 counting from back
    






def convert_mib_to_gb(memory_used_mib):
    # Convert MiB to GB and return it as float with 1 decimal precision
    return round(float(memory_used_mib.replace(' MiB', '')) / 1024, 1)

def parse_memory_used_to_gb(filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)
    
    # Assuming "memory.used [MiB]" is the column to convert
    column_name = [col for col in df.columns if 'memory.used' in col.lower()][0]
    
    # Convert MiB to GB and return the maximum value
    gb_values = df[column_name].apply(convert_mib_to_gb)    
    # Return the list of memory used in GB
    return gb_values.max()

def parse_gpu_mem_filename(file, freqscale=False):
    """
    Parses a GPU memory filename to extract frequency and MPS percentages.
    
    If `freqscale` is True, it also extracts the frequency from the filename.
    
    Args:
        file (str): The GPU memory filename (e.g., "gpu_mem_FREQ1147_10_90.csv").
        freqscale (bool): If True, extracts the frequency from the filename.
    
    Returns:
        tuple: (freq, percentages) if `freqscale` is True.
               (percentages) if `freqscale` is False.
    """
    import re
    freq = None
    percentages = []

    # Remove '.csv' suffix
    file = file.replace('.csv', '')

    # Pattern for extracting frequency and percentages when freqscale is True
    if freqscale:
        pattern = r"gpu_mem_FREQ(\d+)_([\d_]+)"
    else:
        pattern = r"gpu_mem_([\d_]+)"
    
    match = re.match(pattern, file)
    
    if match:
        if freqscale:
            freq = match.group(1)
            percentages = [int(p) for p in match.group(2).split('_') if p]
            return freq, percentages
        else:
            percentages = [int(p) for p in match.group(1).split('_') if p]
            return percentages
    else:
        print(f"Filename does not match pattern: {file}")
        return None, None if freqscale else None

# Define the log file path

def parse_train_log_avgStep_file(filename):
    # Lists to store timestamps and average step times
    timestamps = []
    average_step_times = []

    # Define a regex pattern to match the relevant lines in the log file
    pattern = re.compile(r'\[logger.py:\d+\] (\d+-\d+-\d+ \d+:\d+:\d+,\d+) - INFO - average step time: (\d+\.\d+) seconds')

    # Read the log file and extract the required information
    with open(filename, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                timestamp_str, avg_step_time_str = match.groups()
                #print(f"timestamp_str: {timestamp_str}, avg_step_time_str: {avg_step_time_str}")
                timestamps.append(datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f'))
                average_step_times.append(float(avg_step_time_str))

    if len(average_step_times) <= 1  or len(timestamps) <= 1:
            return None
    # exclude first 10 steps in timesteps and average_step_times to exclude slow start
    if len(timestamps) > 10:
        timestamps = timestamps[9:]
    if len(average_step_times) > 10:
         average_step_times = average_step_times[9:]
    # Calculate the total time elapsed
    if timestamps:
        #print(f"start time: {timestamps[0]}")
        #print(f"end time: {timestamps[-1]}")

        time_elapsed = (timestamps[-1] - timestamps[0]).total_seconds()
        
    # Calculate the average step time
    if average_step_times:
        number_of_steps = len(average_step_times)-1#intervals
        average_step_time = number_of_steps / time_elapsed
        print(f'Number of steps: {number_of_steps}')
        print(f'Time elapsed: {time_elapsed} seconds')
        print(f'Average step time: {average_step_time:.4f} steps/second')
        print(f"average time for each step: {1/average_step_time:.4f} seconds/step")
        

    # Save the timestamps and average step times to lists (if needed)
    #print(f'Timestamps: {timestamps}')
    #print(f'Average Step Times: {average_step_times}')
    #print(f"entries {len(timestamps)}, {len(average_step_times)}")
    return  (average_step_time, number_of_steps)

def parse_inf_log_avgStep_file(filename):
    # Lists to store timestamps and average step times
    timestamps = []
    average_step_times = []

    # Define a regex pattern to match the relevant lines in the log file
    pattern = re.compile(r'\[logger.py:\d+\] (\d+-\d+-\d+ \d+:\d+:\d+,\d+) - INFO - Average processing time: (\d+\.\d+) seconds')

    # Read the log file and extract the required information
    with open(filename, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                timestamp_str, avg_step_time_str = match.groups()
                timestamps.append(datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f'))
                average_step_times.append(float(avg_step_time_str))

    # exclude first 10 steps in timesteps and average_step_times to exclude slow start
    if len(average_step_times) <= 1  or len(timestamps) <= 1:
        return None
    
        # exclude first 10 steps in timesteps and average_step_times to exclude slow start
    if len(timestamps) > 10:
        timestamps = timestamps[9:]
    if len(average_step_times) > 10:
         average_step_times = average_step_times[9:]

    # Calculate the total time elapsed
    if timestamps:
        time_elapsed = (timestamps[-1] - timestamps[0]).total_seconds()

    # Calculate the average step time
    if average_step_times:
        number_of_steps = len(average_step_times)-1
        average_step_time = number_of_steps / time_elapsed
        #print(f'Number of steps: {number_of_steps}')
        #print(f'Time elapsed: {time_elapsed} seconds')
        print(f'Average step time: {average_step_time:.4f} steps/second')
        print(f"average time for each step: {1/average_step_time:.4f} seconds/step")

    # Save the timestamps and average step times to lists (if needed)
    #print(f'Timestamps: {timestamps}')
    #print(f'Average Step Times: {average_step_times}')
    print(f"entries {len(timestamps)}, {len(average_step_times)}")
    print(f"time elapsed: {time_elapsed}")
    return  (average_step_time, number_of_steps)

def parse_cuda_samples_log_avgStep_file(filename, workload):
    print(f"parsing {filename}")
    print(workload)
    if workload in ["sortingNetworks", "BlackScholes", "cudaTensorCoreGemm", "fastWalshTransform", "reductionMultiBlockCG", "transpose"]:
        # Lists to store timestamps and average step times
        iteration_pattern = r"\((\d+)(?: identical)? iterations\)"
        datetime_pattern = r"(\d{4}-\d{1,2}-\d{1,2} \d{1,2}:\d{1,2}:\d{1,2})"
        gops_pattern = r"GOP/s: ([\d\.]+)"  # Example pattern for GOP/s
        
        #bandwdith patttern matching Bandwidth:    416.586850 GB/s
        bandwidth_pattern = r"Bandwidth:    ([\d\.]+) GB/s"
        throughput_pattern = r"Throughput\s*=\s*([\d\.]+)\s*GB/s"
        
        # Patterns for parsing element counts
        elements_pattern = r"^(\d+) elements$"  # reductionMultiBlockCG: "33554432 elements"
        transpose_size_pattern = r"Size\s*=\s*(\d+)\s+fp32 elements"  # transpose: "Size = 1048576 fp32 elements"
        data_length_pattern = r"Data length:\s*(\d+)"  # fastWalshTransform: "Data length: 8388608"
        
        #throughput = gops if workload is fastWalshTransform 
        gops = None
        bandwidth = None
        throughput = None
        iterations = None
        elements = None
        start_time = None
        shutdown_time = None

        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Check for iterations
                if "iterations" in line:
                    match = re.search(iteration_pattern, line)
                    if match:
                        iterations = int(match.group(1))
                
                # Check for element counts for specific workloads
                elements_match = re.search(elements_pattern, line)
                if elements_match and workload == "reductionMultiBlockCG":
                    elements = int(elements_match.group(1))
                
                transpose_size_match = re.search(transpose_size_pattern, line)
                if transpose_size_match and workload == "transpose":
                    elements = int(transpose_size_match.group(1))
                
                data_length_match = re.search(data_length_pattern, line)
                if data_length_match and workload == "fastWalshTransform":
                    elements = int(data_length_match.group(1))
                
                 # Check for GOP/s
                gops_match = re.search(gops_pattern, line)
                if gops_match:
                    gops = float(gops_match.group(1))

                bandwidth_match = re.search(bandwidth_pattern, line)
                if bandwidth_match:
                    bandwidth = float(bandwidth_match.group(1))

                throughput_match = re.search(throughput_pattern, line)
                if throughput_match:
                    print("match throughput")
                    throughput = float(throughput_match.group(1))
                # Check for timestamps
                # We assume the first valid timestamp encountered is start_time
                # and the last valid timestamp is shutdown_time.
                dt_match = re.search(datetime_pattern, line)
                if dt_match:
                    current_dt = datetime.strptime(dt_match.group(1), "%Y-%m-%d %H:%M:%S")
                    # If start_time not set, this must be start_time
                    if start_time is None:
                       
                        start_time = current_dt
                    else:
                        # Keep updating shutdown_time as we go; last one found will be used
                        shutdown_time = current_dt
        # Calculate throughput
        if start_time is None or shutdown_time is None:
            print("Start or shutdown time not found in log file")
            return None
        elapsed = (shutdown_time - start_time).total_seconds()
        if elapsed <= 0:
            return None
        
        print(f"start_time={start_time}, shutdown_time={shutdown_time}, elapsed={elapsed}")
        #Fixed handle - cuda TensorCoreGemm
        if workload == "cudaTensorCoreGemm":
            iterations = 50000
        if workload == "reductionMultiBlockCG":
            throughput = bandwidth
            if throughput is None:
                print("Throughput could not be calculated")
                return None
            print(f"Throughput: {throughput}, Elements: {elements}")
            assert(start_time is not None and shutdown_time is not None)
            # reductionMultiBlockCG logs bandwidth (GB/s), and element count
            return (throughput, elements)
         #setthrouhgput == gops if workload is fastWalshTransform
        if workload == "fastWalshTransform":
            throughput = gops
            #print throughput is None 
            if throughput is None:
                print("Throughput could not be calculated")
                return None
            print(f"Throughput: {throughput}, Elements: {elements}")
            assert(start_time is not None and shutdown_time is not None)
            # fastWalshTransform logs GOP/s (giga operations per second), and data length
            return (throughput, elements)
        if workload == "transpose":
            throughput = throughput
            if throughput is None:
                print("Throughput could not be calculated")
                return None
            print(f"Throughput: {throughput}, Elements: {elements}")
            assert(start_time is not None and shutdown_time is not None)
            # transpose logs throughput (GB/s), and element count
            return (throughput, elements)
        # Validate that we found the required data
        if iterations is None:
            print("Iterations not found in log file")
            return None
        

        
        
        throughput = iterations / elapsed
        print(f"iterations: {iterations}, elapsed: {elapsed}, throughput: {throughput}")
        return (throughput, iterations)
    return None

def parse_train_log_avgStep_file_latency(filename):
    """
    Wrapper function that returns latency (seconds/step) instead of throughput (steps/second).
    """
    result = parse_train_log_avgStep_file(filename)
    if result is None:
        return None
    throughput, number_of_steps = result
    if throughput is None or throughput == 0:
        return None
    return (1.0 / throughput, number_of_steps)

def parse_inf_log_avgStep_file_latency(filename):
    """
    Wrapper function that returns latency (seconds/step) instead of throughput (steps/second).
    """
    result = parse_inf_log_avgStep_file(filename)
    if result is None:
        return None
    throughput, number_of_steps = result
    if throughput is None or throughput == 0:
        return None
    return (1.0 / throughput, number_of_steps)

def parse_cuda_samples_log_avgStep_file_latency(filename, workload):
    """
    Wrapper function that returns latency (seconds/iteration) instead of throughput (iterations/second).
    """
    result = parse_cuda_samples_log_avgStep_file(filename, workload)
    if result is None:
        return None
    throughput, iterations = result
    if throughput is None or throughput == 0:
        return None
    return (1.0 / throughput, iterations)

from collections import defaultdict
def get_all_base_avgStep(directory, BEtype):
    THREAD_PERCENTAGES = [i for i in  range(10, 101, 10)]
    BASELINE_THREADS = ["MPS" + str(i) for i in  THREAD_PERCENTAGES]
    LS_dir = os.path.basename(directory)
    file_steps = defaultdict(list)   # List to store the paths of all files
    
    for root, dirs, files in os.walk(directory):

        for file in files:
            if file.startswith('BE'):
                file_path = os.path.abspath(os.path.join(root, file))
                BE_dir = os.path.basename(root)
                #remove "BE" from BE+dir if it exists
                BE_dir = BE_dir.replace("BE_", "")
                print(file)
            #split file with "_" and find if BASELINE_THREADS is in the split
                for MPSpercent in BASELINE_THREADS:

                    if MPSpercent+".log" in file.split("_") and file.startswith('BE') :
                        file_path = os.path.abspath(os.path.join(root, file))
                        batch_size = int(BE_dir.split("_")[-1][5:])
                        print(file_path)
                        print(f"{BE_dir} {MPSpercent}")
                        print(f"batch_size: {batch_size}")
                        #get batch size from BE_dir
                        
                        result = parse_train_log_avgStep_file(file_path)
                        if BEtype == "train" and result is not None:
                            steps, _ = result
                            file_steps[f"{BE_dir}-train"].append((int(f"{MPSpercent[3:]}"),  (steps * batch_size)))
                        else:
                            result = parse_inf_log_avgStep_file(file_path)
                            if result is not None:
                                steps, _ = result
                                file_steps[f"{BE_dir}-inf"].append((int(f"{MPSpercent[3:]}"),  (steps * batch_size)))
                        break

                #file_paths.append(file_path)
    return file_steps

def get_share_avgStep(directory, LStype, BEtype, all_models):
    THREAD_PERCENTAGES = [i for i in  range(10, 101, 10)]
    BASELINE_THREADS = ["MPS" + str(i) for i in  THREAD_PERCENTAGES]
    LS_dir = os.path.basename(directory)
    file_steps = defaultdict(dict)   # List to store the paths of all files
    
    
    for root, dirs, files in os.walk(directory):

        for file in files:
            for MPSpercent in BASELINE_THREADS:
                if MPSpercent+".log" in file.split("_") and (file.startswith('BE') or file.startswith('LS')):
                    file_path = os.path.abspath(os.path.join(root, file))
                    #split file_path
                    file_path_split = file_path.split("/")
                    
                    print(file)
                    print(file_path_split)
                    LS_model = None
                    #get LS model name by matching file_path_split with all_models
                    for segments in file_path_split:
                        if segments+"-"+LStype in all_models:
                            LS_model = segments
                            break
                    if LS_model is None:
                        print(f"LS model not found in all models, returning...")
                        return
                    print(LS_model)
                    
                    LS_batch_size = int(LS_model.split("_")[-1][5:])
                    BE_dir = os.path.basename(root)
                    #remove "BE" from BE+dir if it exists
                    BE_dir = BE_dir.replace("BE_", "")
                    print(f"LS BE: {LS_model} {BE_dir}")
                #split file with "_" and find if BASELINE_THREADS is in the split
                

                    
                    BE_batch_size = int(BE_dir.split("_")[-1][5:])
                    print(file_path)
                    print(f"{BE_dir} {MPSpercent}")
                    print(f"batch_size: {BE_batch_size}")
                    #get batch size from BE_dir
                    #get LS logs
                    
                    if  file.startswith('LS') :
                        if LStype == "train":
                            result = parse_train_log_avgStep_file(file_path)
                            if result is not None:
                                steps, _ = result
                                LS_steps = steps * LS_batch_size
                                file_steps[(f"{LS_model}-{LStype}", f"{BE_dir}-{BEtype}")][f"LS{MPSpercent[3:]}_{LS_model}-{LStype}"] = LS_steps
                        elif LStype == "inf":
                            result = parse_inf_log_avgStep_file(file_path)
                            if result is not None:
                                steps, _ = result
                                LS_steps = steps * LS_batch_size
                                file_steps[(f"{LS_model}-{LStype}", f"{BE_dir}-{BEtype}")][f"LS{MPSpercent[3:]}_{LS_model}-{LStype}"] = LS_steps
                    
                    elif file.startswith('BE'):
                        if BEtype == "train":
                            result = parse_train_log_avgStep_file(file_path)
                            if result is not None:
                                steps, _ = result
                                BE_steps = steps * BE_batch_size
                                file_steps[(f"{LS_model}-{LStype}", f"{BE_dir}-{BEtype}")][f"BE{MPSpercent[3:]}_{BE_dir}-{BEtype}"] = BE_steps
                        elif BEtype == "inf":
                            result = parse_inf_log_avgStep_file(file_path)
                            if result is not None:
                                steps, _ = result
                                BE_steps = steps * BE_batch_size
                                file_steps[(f"{LS_model}-{LStype}", f"{BE_dir}-{BEtype}")][f"BE{MPSpercent[3:]}_{BE_dir}-{BEtype}"] = BE_steps
           
                            
                    break
    print(file_steps)
    #reshape file_steps to a dictionary with key=({LS_model}-{LStype}, {BE_dir}-{BEtype}) and value = (LS_steps, BE_steps)


                        

                #file_paths.append(file_path)
    return file_steps

def parse_workload_dir(dirname):
    # Example dirname: 'gpt2-xl_batch100-inf%vit_h_14_batch16-train'
    workloads = dirname.split('%')  # Split by '%'
    
    parsed_workloads = defaultdict(dict)  # To store parsed workloads as tuples of (idx, workload_name, batchsize, mode)
    
    for i, workload in enumerate(workloads):
        idx, workload_name, batchsize, mode = "", "", "", ""
        
        # Split the workload by '_batch' to separate the name from the batchsize and mode
        parts = workload.split('_batch')
        workload_name = parts[0]  # Get the workload name (e.g., 'gpt2-xl')

        if len(parts) > 1:
            # Further split the batch and mode part (e.g., '100-inf')
            batch_mode = parts[1].split('-')
            batchsize = batch_mode[0]  # Extract the batchsize (e.g., '100')
            mode = batch_mode[1]  # Extract the mode (e.g., 'inf' or 'train')

            if mode not in ["inf", "train", "cuda_samples"]:
                raise ValueError(f"Invalid mode: {mode}")
        
        # Add the parsed workload as a dict

        parsed_workloads[f'w{i+1}']["workload_name"] = workload_name
        parsed_workloads[f'w{i+1}']["batchsize"] = batchsize
        parsed_workloads[f'w{i+1}']["mode"] = mode
        #parsed_workloads.append((f'w{i+1}', workload_name, batchsize, mode))
    
    return parsed_workloads

def parse_workload(log_file, freqscale=False, valid_modes=["inference", "train", "cuda_samples"]):
    """
    Parses a log filename to extract workload details including index, workload name, batch size, mode, and MPS percentage.
    
    If `freqscale` is True, it also extracts the frequency.
    
    Args:
        log_file (str): The log filename (e.g., "w2_vit_h_14_batch2-inference_MPS100.log").
        freqscale (bool): If True, extracts frequency from the filename.
    
    Returns:
        tuple: (idx, workload_name, batchsize, mode, MPSpercent, freq) if `freqscale` is True.
               (idx, workload_name, batchsize, mode, MPSpercent) if `freqscale` is False.
    """
    # Remove '.log' suffix from the filename
    log_file = log_file.replace('.log', '')
    idx, workload_name, batchsize, mode, MPSpercent, freq = "", "", "", "", "", None

    # Splitting the string to extract parts
    parts = log_file.split('_')
    #special case for cuda_samples. if samples in parts, connect with previous part
    for i, part in enumerate(parts):
        if part == "samples":
            parts[i-1] = parts[i-1] + "_samples"
            parts.pop(i)
            break

    
    # Process the workload log file depending on whether freqscale is True or False
    for part in parts:
        if part.startswith('batch'):
            print(part)
            batchsize, mode = part.split("-")[0].replace('batch', ''), part.split("-")[1]
            print(f"batchsize: {batchsize}, mode: {mode}")
            if mode not in valid_modes:
                raise ValueError(f"Invalid mode: {mode}")
            if mode == "inference":
                mode = "inf"
        elif part.startswith('MPS'):
            MPSpercent = part.replace('MPS', '')
 
        # If freqscale is True, extract the frequency from the log filename
        if freqscale and part.startswith('FREQ'):
            freq = part.replace('FREQ', '')

    # Extract workload_name and index from the filename
    idx, workload_name = log_file.split('_batch')[0].split("_", 1)

    # Return based on whether freqscale is True or False
    if freqscale:
        #print all extracted values
        print(f"idx: {idx}, workload_name: {workload_name}, batchsize: {batchsize}, mode: {mode}, MPSpercent: {MPSpercent}, freq: {freq}")
        return idx, workload_name, batchsize, mode, freq, MPSpercent
    else:
        return idx, workload_name, batchsize, mode, MPSpercent


def shorten_mode(workload):
    #if inference in workload, shorten to inf
    if "inference" in workload:
        workload = workload.replace("inference", "inf")
    return workload

def get_multiworkloads_share_avgStep_with_power(directory, n_combinations, idle_power, autorergressive_models=["gpt2-xl"]):
    file_steps = defaultdict(dict)
    power_dict = defaultdict(dict)
    thread_percentages = [i for i in range(10, 101, 10)]

    for root, dirs, files in os.walk(directory):
        current_depth = root[len(directory):].count(os.sep)
        if current_depth == n_combinations-1:
            
            #skip basename== cuda
            if os.path.basename(root) == "cuda":
                raise ValueError(f"should not be cuda directory (one level below the desired depth)")
            #print(root)
            #print(files)
            #print(os.path.basename(root))
            workloads = [shorten_mode(w) for w in root.split(os.sep)[-n_combinations:]]
            
            if len(workloads) != n_combinations:
                print(f"Expected {n_combinations} workloads, but found {len(workloads)} workloads")
                continue
            print(workloads)
            for file in files:
                if ".log" in file:
                    #for idx, workload_name, batchsize, mode in parsed_workloads:
   
                    file_path = os.path.abspath(os.path.join(root, file))
                    #print(f"file split: {file.split("_")}")
                    idx, workload_name, batchsize, mode, MPSpercent = parse_workload(file)
                    if  workload_name in autorergressive_models:
                        print(f"{idx} is autorergressive model, use subdir batch as outputlength field...")
                        #SPECIAL CASE - use the workload_list to get batchsize and mode
                        batchsize = workload_list[idx]["batchsize"]
                        mode = workload_list[idx]["mode"]
                    print(f"idx: {idx}, workload_name: {workload_name}, batchsize: {batchsize}, mode: {mode}, MPSpercent: {MPSpercent}")

                        
                    #print(idx, workload_name, batchsize, mode, MPSpercent)

                    if mode == "train":
                        result = parse_train_log_avgStep_file(file_path)
                        if result is not None:
                            steps, _ = result
                            train_steps = steps * int(batchsize) if workload_name not in autorergressive_models else steps * int(batchsize) 
                            file_steps[tuple(workloads)][f"{idx}_{workload_name}_batch{batchsize}-{mode}_MPS{MPSpercent}"] = train_steps
                    elif mode == "inf":
                        result = parse_inf_log_avgStep_file(file_path)
                        if result is not None:
                            steps, _ = result
                            inf_steps = steps * int(batchsize) if workload_name not in autorergressive_models else steps * int(batchsize) 
                            file_steps[tuple(workloads)][f"{idx}_{workload_name}_batch{batchsize}-{mode}_MPS{MPSpercent}"] = inf_steps
                        #file_steps[(f"{LS_model}-{mode}", f"{BE_dir}-{mode}")][f"LS{MPSpercent[3:]}_{LS_model}-{mode}"] = LS_steps
                if "gpu_mem" in file:
                    #parse power 
                    power_file = os.path.abspath(os.path.join(root, file))
                    power = calculate_average_power_adjusted(power_file, idle_power = idle_power)
                    #strip csv extension
                    file = file.replace(".csv", "")
                    #strip _ on the right if any
                    file = file.rstrip("_")
                    #print(file.split("_")[-n_combinations:])
                    #get percentatge from file name - gpu_mem_10_90.csv
                    percentage = [int(i) for i in file.split("_")[-n_combinations:]]
                    power_dict[tuple(workloads)][(tuple(percentage))] = power
    print("file steps", file_steps)
    print("power dict", power_dict)
    return file_steps, power_dict

def get_multiworkloads_share_avgStep(directory, n_combinations, all_models, autorergressive_models=["gpt2-xl"]):
    THREAD_PERCENTAGES = [i for i in range(10, 101, 10)]
    BASELINE_THREADS = ["MPS" + str(i) for i in THREAD_PERCENTAGES]
    file_steps = defaultdict(dict)  # Dictionary to store the paths of all files

    for root, dirs, files in os.walk(directory):
        for subdir in dirs:
            if "%" in subdir:
                #use parse_workload_dir to parse subdir
                
                workloads = subdir.split('%')
                if len(workloads) != n_combinations:
                    raise ValueError(f"Expected {n_combinations} workloads, but found {len(workloads)} workloads")
                workload_list = parse_workload_dir(subdir)
                print("subdir parse", workload_list)
                #parsed_workloads = [parse_workload(i+1, workload) for i, workload in enumerate(workloads)]
                #print(parsed_workloads)
                subdir_path = os.path.abspath(os.path.join(root, subdir))
                        #continue
                #print(f"subdir contain: {os.listdir(subdir_path)}")
                for file in os.listdir(subdir_path):
                    if ".log" in file:
                    #for idx, workload_name, batchsize, mode in parsed_workloads:
   
                        file_path = os.path.abspath(os.path.join(root,subdir, file))
                        #print(f"file split: {file.split("_")}")
                        idx, workload_name, batchsize, mode, MPSpercent = parse_workload(file)
                        if  workload_name in autorergressive_models:
                            print(f"{idx} is autorergressive model, use subdir batch as outputlength field...")
                            #SPECIAL CASE - use the workload_list to get batchsize and mode
                            batchsize = workload_list[idx]["batchsize"]
                            mode = workload_list[idx]["mode"]
                        print(f"idx: {idx}, workload_name: {workload_name}, batchsize: {batchsize}, mode: {mode}, MPSpercent: {MPSpercent}")

                            
                        #print(idx, workload_name, batchsize, mode, MPSpercent)

                        if mode == "train" and parse_train_log_avgStep_file(file_path) is not None:
                            train_steps = parse_train_log_avgStep_file(file_path) * int(batchsize) if workload_name not in autorergressive_models else parse_train_log_avgStep_file(file_path)* int(batchsize) 
                            file_steps[tuple(workloads)][f"{idx}_{workload_name}_batch{batchsize}-{mode}_MPS{MPSpercent}"] = train_steps
                        elif mode == "inf" and parse_inf_log_avgStep_file(file_path) is not None:
                            inf_steps = parse_inf_log_avgStep_file(file_path) * int(batchsize) if workload_name not in autorergressive_models else parse_inf_log_avgStep_file(file_path)* int(batchsize) 
                            file_steps[tuple(workloads)][f"{idx}_{workload_name}_batch{batchsize}-{mode}_MPS{MPSpercent}"] = inf_steps
                            #file_steps[(f"{LS_model}-{mode}", f"{BE_dir}-{mode}")][f"LS{MPSpercent[3:]}_{LS_model}-{mode}"] = LS_steps
                    
                    
                        
                        
    print(file_steps)
    #reshape file_steps to a dictionary with key=({LS_model}-{LStype}, {BE_dir}-{BEtype}) and value = (LS_steps, BE_steps)
    return file_steps

#get multiworkloads share avgStep with frequency and power
def get_multiworkloads_share_avgStep_with_freqscale(directory, n_combinations, idle_power, autorergressive_models=["gpt2-xl"]):
    file_steps = defaultdict(lambda: defaultdict(dict))
    power_dict = defaultdict(lambda: defaultdict(dict))
    thread_percentages = [i for i in range(10, 101, 10)]

    for root, dirs, files in os.walk(directory):
        current_depth = root[len(directory):].count(os.sep)
        if current_depth == n_combinations - 1:
            if os.path.basename(root) == "cuda":
                continue
            workloads = [shorten_mode(w) for w in root.split(os.sep)[-n_combinations:]]
            if len(workloads) != n_combinations:
                print(f"Expected {n_combinations} workloads, but found {len(workloads)} workloads")
                continue
            print(f"Processing workloads: {workloads}")
            for file in files:
                file_path = os.path.abspath(os.path.join(root, file))
                if ".log" in file:
                    idx, workload_name, batchsize, mode, freq, MPSpercent = parse_workload(file, freqscale=True)
                    if None in [idx, workload_name, batchsize, mode, freq, MPSpercent]:
                        continue
                    print(f"Parsed log file: idx={idx}, workload_name={workload_name}, batchsize={batchsize}, mode={mode}, freq={freq}, MPSpercent={MPSpercent}")
                    if workload_name in autorergressive_models:
                        batchsize = workload_list[idx]["batchsize"]
                        mode = workload_list[idx]["mode"]
                    if mode == "train":
                        steps = parse_train_log_avgStep_file(file_path)
                    elif mode == "inf":
                        steps = parse_inf_log_avgStep_file(file_path)
                    else:
                        steps = None
                    if steps is not None:
                        adjusted_steps = steps * int(batchsize)
                        file_steps[tuple(workloads)][freq][f"{idx}_{workload_name}_batch{batchsize}-{mode}_MPS{MPSpercent}"] = adjusted_steps
                elif "gpu_mem" in file:
                    freq, percentages = parse_gpu_mem_filename(file, freqscale=True)
                    if None in [freq, percentages]:
                        continue
                    print(f"Parsed GPU mem file: freq={freq}, percentages={percentages}")
                    power = calculate_average_power_adjusted(file_path, idle_power=idle_power)
                    power_dict[tuple(workloads)][freq][tuple(percentages)] = power

    print("file steps", file_steps)
    print("power dict", power_dict)
    return file_steps, power_dict



def aggregate_runs(directory, n_combinations, idle_power, normalize_by_baseline_csv = None, autorergressive_models=["gpt2-xl"], cost_savings = False):
    """
    Function to aggregate throughput and power data across multiple runs.
    Excludes entries where any workload's throughput is None.
    """
    # Initialize dictionaries to store aggregated data for multiple runs
    aggregated_file_steps = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    aggregated_power_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    thread_percentages = [i for i in range(10, 101, 10)]
    thread_combinations = generate_thread_combinations(n_combinations)  # Generate all thread combinations for workloads

    # Walk through the run directories (e.g., RUN1, RUN2, etc.)
    for run_dir in os.listdir(directory):
        run_path = os.path.join(directory, run_dir)
        if not os.path.isdir(run_path):
            continue

        for root, dirs, files in os.walk(run_path):
            current_depth = root[len(run_path):].count(os.sep)

            if current_depth == n_combinations:
                if os.path.basename(root) == "cuda":
                    continue

                # Get workload names
                workloads = [shorten_mode(w) for w in root.split(os.sep)[-n_combinations:]]
                if len(workloads) != n_combinations:
                    print(f"Expected {n_combinations} workloads, but found {len(workloads)} workloads")
                    continue

                # Iterate through all files in the directory
                for file in files:
                    file_path = os.path.abspath(os.path.join(root, file))

                    # Parse throughput log files
                    if ".log" in file:
                        idx, workload_name, batchsize, mode, freq, MPSpercent = parse_workload(file, freqscale=True)
                        if None in [idx, workload_name, batchsize, mode, freq, MPSpercent]:
                            continue
                        print(f"Parsed log file: idx={idx}, workload_name={workload_name}, batchsize={batchsize}, mode={mode}, freq={freq}, MPSpercent={MPSpercent}")
                        if workload_name in autorergressive_models:
                            batchsize = workload_list[idx]["batchsize"]
                            mode = workload_list[idx]["mode"]
                        if mode == "train":
                            result = parse_train_log_avgStep_file(file_path)
                        elif mode == "inf":
                            result = parse_inf_log_avgStep_file(file_path)
                        else:
                            result = None
                        
                        if result is not None:
                            steps, _ = result
                            adjusted_steps = steps * int(batchsize)
                        else:
                            adjusted_steps = None
                        aggregated_file_steps[tuple(workloads)][freq][f"{idx}_{workload_name}_batch{batchsize}-{mode}_MPS{MPSpercent}"].append(adjusted_steps)

                    # Parse GPU memory (power) files
                    elif "gpu_mem" in file:
                        freq, percentages = parse_gpu_mem_filename(file, freqscale=True)
                        if None in [freq, percentages]:
                            raise ValueError(f"Invalid frequency or percentages: {freq}, {percentages}")

                        power = calculate_average_power_adjusted(file_path, idle_power=idle_power)
                        #if not np.isnan(power):
                        aggregated_power_dict[tuple(workloads)][freq][tuple(percentages)].append(power)
    #print("Aggregated file steps:", aggregated_file_steps)
    #print("Aggregated power dict: ", aggregated_power_dict)
    # Remove invalid runs where throughput is None
    aggregate_file_steps_thread_combination = remove_invalid_runs(aggregated_file_steps, aggregated_power_dict, thread_combinations)


    print("Aggregated file steps after invalid run removal:", aggregate_file_steps_thread_combination)
    print("Aggregated power dict after invalid run removal", aggregated_power_dict)
    # Compute the average and standard deviation for each entry
    avg_file_steps, std_file_steps =  compute_steps_sum_avg_std(aggregate_file_steps_thread_combination, normalize_by_baseline_csv = normalize_by_baseline_csv, cost_savings = cost_savings)
    avg_power_dict, std_power_dict = compute_power_sum_avg_std(aggregated_power_dict)
    print("Average file steps:", avg_file_steps)
    print("Standard deviation file steps:", std_file_steps)
    

    
    
    return avg_file_steps, std_file_steps, avg_power_dict, std_power_dict
    

import os
import numpy as np
from collections import defaultdict

def aggregate_runs_threaddir(directory, n_combinations, output_prefix, idle_power_dict, normalize_by_baseline_csv=None, autorergressive_models=["gpt2-xl"], cost_savings=False, metric_type="throughput"):
    """
    Function to aggregate throughput and power data across multiple runs.
    Excludes entries where any workload's throughput is None.
    Now uses thread combination as a subdirectory for each workload pair.

    Args:
        metric_type (str): Type of metric to collect. Either "throughput" (steps/second)
                          or "latency" (seconds/step). Default is "throughput".
    """
    # Initialize dictionaries to store aggregated data for multiple runs
    #file steps is a dictionary with key = (workload1, workload2) and value = (freq, thread_combination, idx_workload_name_batchsize-mode_MPSpercent)
    aggregated_file_steps = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    aggregated_power_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    aggregated_duration_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    aggregated_energy_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    aggregated_steps_count_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    thread_percentages = [i for i in range(10, 101, 10)]
    thread_combinations = generate_thread_combinations(n_combinations)  # Generate all thread combinations for workloads

    # Walk through the run directories (e.g., RUN1, RUN2, etc.)
    for run_dir in os.listdir(directory):
        run_path = os.path.join(directory, run_dir)
        if not os.path.isdir(run_path):
            continue

        for root, dirs, files in os.walk(run_path):
            current_depth = root[len(run_path):].count(os.sep)
            
            # Adjust depth to account for the subdirectory containing thread combinations (e.g., "40_100", "60_140")
            if current_depth == n_combinations + 1:  # +1 because of thread combination directory
                if os.path.basename(root) == "cuda":
                    continue

                # Get workload names
                workloads = [shorten_mode(w) for w in root.split(os.sep)[-n_combinations-1:-1]]
                thread_combination = tuple([int(thread_comb) for thread_comb in os.path.basename(root).rstrip("_").split("_")])  # Use the thread combination directory name as the key
                print(f"thread_combination, {thread_combination}")
                if len(workloads) != n_combinations:
                    print(f"Expected {n_combinations} workloads, but found {len(workloads)} workloads")
                    continue

                # Iterate through all files in the directory
                for file in files:
                    file_path = os.path.abspath(os.path.join(root, file))

                    # Parse throughput log files
                    if ".log" in file and "gpu_monitor" not in file:
                        print(f"parsing log file {file}")
                        idx, workload_name, batchsize, mode, freq, MPSpercent = parse_workload(file, freqscale=True)
                        
                        if None in [idx, workload_name, batchsize, mode, freq, MPSpercent]:
                            continue
                        print(f"Parsed log file: idx={idx}, workload_name={workload_name}, batchsize={batchsize}, mode={mode}, freq={freq}, MPSpercent={MPSpercent}")
                       
                        if workload_name in autorergressive_models:
                            batchsize = workload_list[idx]["batchsize"]
                            mode = workload_list[idx]["mode"]
                        if mode == "train":
                            if metric_type == "latency":
                                result = parse_train_log_avgStep_file_latency(file_path)
                            else:
                                result = parse_train_log_avgStep_file(file_path)
                        elif mode == "inf":
                            if metric_type == "latency":
                                result = parse_inf_log_avgStep_file_latency(file_path)
                            else:
                                result = parse_inf_log_avgStep_file(file_path)
                        elif mode == "cuda_samples":
                            if metric_type == "latency":
                                result = parse_cuda_samples_log_avgStep_file_latency(file_path, workload=workload_name)
                            else:
                                result = parse_cuda_samples_log_avgStep_file(file_path, workload=workload_name)
                        
                        # Unpack the result tuple
                        if result is not None:
                            steps, num_steps = result
                            if mode == "train" or mode == "inf":
                                if metric_type == "latency":
                                    adjusted_steps = steps / int(batchsize)
                                else:  # throughput
                                    adjusted_steps = steps * int(batchsize)
                            else:
                                adjusted_steps = steps
                        else:
                            adjusted_steps = None
                            num_steps = None
                        
                        aggregated_file_steps[tuple(workloads)][freq][thread_combination][f"{idx}_{workload_name}_batch{batchsize}-{mode}_MPS{MPSpercent}"].append(adjusted_steps)
                        aggregated_steps_count_dict[tuple(workloads)][freq][thread_combination][f"{idx}_{workload_name}_batch{batchsize}-{mode}_MPS{MPSpercent}"].append(num_steps)

                    # Parse GPU memory (power) files
                    elif "gpu_mem" in file:
                        freq, percentages = parse_gpu_mem_filename(file, freqscale=True)
                        if None in [freq, percentages]:
                            raise ValueError(f"Invalid frequency or percentages: {freq}, {percentages}")
                        assert(tuple(percentages) == thread_combination)
                        #assert freq in idel_power_dict
                        if int(freq) not in idle_power_dict:
                            #print(f"type of freq: {type(freq)}")
                            print(f"current gpu mem file: {os.path.join(root,file_path)}")
                            raise ValueError(f"frequency not in idle power dict: {freq}")
                        #assert(freq in idle_power_dict)
                        idle_power = idle_power_dict[int(freq)]
                        #print(f"idle power: {idle_power}")
                        power, duration, energy = calculate_average_power_duration_adjusted(file_path, idle_power=idle_power)
                        aggregated_power_dict[tuple(workloads)][freq][tuple(percentages)].append(power)
                        aggregated_duration_dict[tuple(workloads)][freq][tuple(percentages)].append(duration)
                        aggregated_energy_dict[tuple(workloads)][freq][tuple(percentages)].append(energy)
    print("Aggregated file steps:", aggregated_file_steps)
    print("Aggregated power dict: ", aggregated_power_dict)
    # #region agent log
    import json
    try:
        os.makedirs(os.path.dirname('/tmp/peace_parse_debug.log'), exist_ok=True)
    except: pass
    with open('/tmp/peace_parse_debug.log', 'a') as f:
        try:
            sample_key = list(aggregated_steps_count_dict.keys())[0] if aggregated_steps_count_dict else None
            if sample_key:
                sample_freq = list(aggregated_steps_count_dict[sample_key].keys())[0]
                sample_thread = list(aggregated_steps_count_dict[sample_key][sample_freq].keys())[0]
                sample_workload = list(aggregated_steps_count_dict[sample_key][sample_freq][sample_thread].keys())[0]
                sample_list = aggregated_steps_count_dict[sample_key][sample_freq][sample_thread][sample_workload]
                f.write(json.dumps({"sessionId":"debug-session","runId":"pre-fix","hypothesisId":"B,D","location":"parse_util.py:1030","message":"aggregated_steps_count_dict raw data","data":{"sample_list_type":str(type(sample_list)),"is_list":isinstance(sample_list, list),"list_len":len(sample_list) if isinstance(sample_list, list) else "N/A"},"timestamp":__import__('time').time()*1000})+'\n')
        except Exception as e:
            f.write(json.dumps({"sessionId":"debug-session","runId":"pre-fix","hypothesisId":"B,D","location":"parse_util.py:1030","message":"ERROR sampling aggregated_steps_count_dict","data":{"error":str(e)},"timestamp":__import__('time').time()*1000})+'\n')
    # #endregion
    # Remove invalid runs in power, energy, duration, steps_count where throughput is None
    aggregate_file_steps_thread_combination, aggregate_steps_count_thread_combination = remove_invalid_runs_threaddir(aggregated_file_steps, aggregated_power_dict, aggregated_duration_dict, aggregated_energy_dict, aggregated_steps_count_dict)
    #save aggregated_file_steps_thread_combination to csv
    save_with_freqscale_individual_avg_std(aggregate_file_steps_thread_combination, f"{output_prefix}_share_comb{n_combinations}_freqscale_{metric_type}_individual_avg.csv", f"{output_prefix}_share_comb{n_combinations}_freqscale_{metric_type}_individual_std.csv" , n_combinations)
    #save aggregated_steps_count (transformed to list format) to csv
    save_with_freqscale_individual_avg_std(aggregate_steps_count_thread_combination, f"{output_prefix}_share_comb{n_combinations}_freqscale_steps_count_individual_avg.csv", f"{output_prefix}_share_comb{n_combinations}_freqscale_steps_count_individual_std.csv" , n_combinations)
    #save_with_freqscale_individual_avg_std(aggregate_file_duration_thread_combination, f"{output_prefix}_share_comb{n_combinations}_freqscale_duration_individual_avg.csv", f"{output_prefix}_share_comb{n_combinations}_freqscale_duration_individual_std.csv" , n_combinations)

    print("Aggregated file steps after invalid run removal:", aggregate_file_steps_thread_combination)
    print("Aggregated power dict after invalid run removal", aggregated_power_dict)
    # #region agent log
    import json
    with open('/tmp/peace_parse_debug.log', 'a') as f:
        try:
            sample_key = list(aggregate_file_steps_thread_combination.keys())[0] if aggregate_file_steps_thread_combination else None
            if sample_key:
                sample_freq = list(aggregate_file_steps_thread_combination[sample_key].keys())[0]
                sample_thread = list(aggregate_file_steps_thread_combination[sample_key][sample_freq].keys())[0]
                sample_value = aggregate_file_steps_thread_combination[sample_key][sample_freq][sample_thread]
                f.write(json.dumps({"sessionId":"debug-session","runId":"post-fix","hypothesisId":"A,C","location":"parse_util.py:1047","message":"aggregate_file_steps_thread_combination sample","data":{"sample_value_type":str(type(sample_value)),"is_list":isinstance(sample_value, list),"list_len":len(sample_value) if isinstance(sample_value, list) else "not_list"},"timestamp":__import__('time').time()*1000})+'\n')
        except Exception as e:
            f.write(json.dumps({"sessionId":"debug-session","runId":"post-fix","hypothesisId":"A,C","location":"parse_util.py:1047","message":"ERROR sampling aggregate_file_steps","data":{"error":str(e)},"timestamp":__import__('time').time()*1000})+'\n')
    # #endregion
    # Compute the average and standard deviation for each entry
    avg_file_steps, std_file_steps = compute_steps_sum_avg_std(aggregate_file_steps_thread_combination, normalize_by_baseline_csv=normalize_by_baseline_csv, cost_savings=cost_savings)
    avg_power_dict, std_power_dict = compute_power_sum_avg_std(aggregated_power_dict)
    avg_duration_dict, std_duration_dict = compute_power_sum_avg_std(aggregated_duration_dict)
    avg_energy_dict, std_energy_dict = compute_power_sum_avg_std(aggregated_energy_dict)
    #compute_steps_sum_avg_std(aggregate_file_duration_thread_combination, normalize_by_baseline_csv=None, cost_savings=False, getmin=True)
    
    print("Average file steps:", avg_file_steps)
    print("Standard deviation file steps:", std_file_steps)

    return avg_file_steps, std_file_steps, avg_power_dict, std_power_dict, avg_duration_dict, std_duration_dict, avg_energy_dict, std_energy_dict

        
def compute_power_sum_avg_std(aggregated_power_dict):
    """
    Compute the sum, average, and standard deviation of the power sums across different thread combinations.
    """
    # Iterate through all workloads, frequencies, and thread combinations
    avg_power_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    std_power_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for workloads, freq_dict in aggregated_power_dict.items():
        for freq, thread_dict in freq_dict.items():
            for threads, power_list in thread_dict.items():
                # Calculate the average and standard deviation of the power sums
                power_list = [power for power in power_list if power is not None]
                if power_list:
                    #remove None values from power_list
                    avg_power_dict[workloads][freq][threads] = float(np.mean(power_list))
                    std_power_dict[workloads][freq][threads] = float(np.std(power_list))
                else:
                    # If no valid power values were found, leave as zero
                    avg_power_dict[workloads][freq][threads] = None
                    std_power_dict[workloads][freq][threads] = None

    return avg_power_dict, std_power_dict

def compute_steps_count_avg_std(aggregated_steps_count_dict):
    """
    Compute the average and standard deviation of number_of_steps.
    """
    avg_steps_count_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    std_steps_count_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))

    for workloads, freq_dict in aggregated_steps_count_dict.items():
        for freq, thread_dict in freq_dict.items():
            for threads, workload_dict in thread_dict.items():
                for workload_key, steps_list in workload_dict.items():
                    steps_list = [s for s in steps_list if s is not None]
                    if steps_list:
                        avg_steps_count_dict[workloads][freq][threads][workload_key] = float(np.mean(steps_list))
                        std_steps_count_dict[workloads][freq][threads][workload_key] = float(np.std(steps_list))
                    else:
                        avg_steps_count_dict[workloads][freq][threads][workload_key] = None
                        std_steps_count_dict[workloads][freq][threads][workload_key] = None

    return avg_steps_count_dict, std_steps_count_dict

def compute_steps_sum_avg_std(aggregate_file_steps_thread_combination, normalize_by_baseline_csv = None, cost_savings = False, getmin= False):
    """
    Compute the sum, average, and standard deviation of the throughput sums across different thread combinations.
    """
    #iterate through all workloads, frequencies and thread combinations and sum up file steps
    avg_file_steps = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    std_file_steps = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    if getmin:
        pass
        """
        # Iterate through workloads, frequencies, and thread combinations
        for workloads, freq_dict in aggregate_file_steps_thread_combination.items():
            for freq, thread_dict in freq_dict.items():
                for threads, file_steps_list in thread_dict.items():

                    # Initialize a list to store the sum of throughput for each run
                    sum_throughput_per_run = []
                    
                    # Sum the throughput across all workloads for each run index
                    num_runs = len(file_steps_list[0])  # Assuming all runs have the same length
                    for run_idx in range(num_runs):
                        run_min = None
                        valid_run = True  # Track if we encounter None

                        for workload_steps in file_steps_list:
                            if workload_steps[run_idx] is None:
                                valid_run = False
                                break  # If any value is None, skip this run

                            run_min = min(run_min, workload_steps[run_idx])

                        if valid_run and run_min is not None:
                            sum_throughput_per_run.append(run_sum)

                    # Calculate the average and standard deviation of the throughput sums
                    if sum_throughput_per_run:
                        avg_file_steps[workloads][freq][threads] = float(np.mean(sum_throughput_per_run))
                        std_file_steps[workloads][freq][threads] = float(np.std(sum_throughput_per_run))
                    else:
                        # If no valid runs were found, leave as zero
                        avg_file_steps[workloads][freq][threads] = None
                        std_file_steps[workloads][freq][threads] = None
                    print(f"workloads: {workloads}, file_steps_list: {file_steps_list}")
                print(avg_file_steps)
                print(f"std_file_steps: {std_file_steps}")
                raise ValueError("stop here")
        """
    else:
        # Normalize the throughput by the baseline if a CSV file is provided
        if normalize_by_baseline_csv and not cost_savings:
            print("normalize throughput by baseline...")
            baseline_df = pd.read_csv(normalize_by_baseline_csv)
            #normalized the throughput by the baseline
            for workloads, freq_dict in aggregate_file_steps_thread_combination.items():   
                for freq, thread_dict in freq_dict.items():
                    for threads, file_steps_list in thread_dict.items():
                        for workload_idx, workload_steps in enumerate(file_steps_list):
                            base_steps = float(baseline_df[baseline_df["Type"].str.contains(workloads[workload_idx])]["Exclusive100"])
                            
                            if base_steps is None:
                                raise ValueError(f"Baseline throughput not found for workload: {workloads[workload_idx]}")
                            #print(f"{workloads[workload_idx]} base_steps: {base_steps} ")
                            #print(f"workload_steps: {workload_steps}")
                            workload_steps = [step/base_steps for step in workload_steps if step is not None]
                            #print(f"normalized workload_steps: {workload_steps}")
                            #saveback the normalized values
                            file_steps_list[workload_idx] = workload_steps
        
                        
        # Iterate through workloads, frequencies, and thread combinations
        for workloads, freq_dict in aggregate_file_steps_thread_combination.items():
            for freq, thread_dict in freq_dict.items():
                for threads, file_steps_list in thread_dict.items():

                    # Initialize a list to store the sum of throughput for each run
                    sum_throughput_per_run = []
                    
                        
                        
                    # Sum the throughput across all workloads for each run index
                    num_runs = len(file_steps_list[0])  # Assuming all runs have the same length
                    for run_idx in range(num_runs):
                        run_sum = 0
                        valid_run = True  # Track if we encounter None

                        for workload_steps in file_steps_list:
                            if workload_steps[run_idx] is None:
                                valid_run = False
                                break  # If any value is None, skip this run

                            run_sum += workload_steps[run_idx]

                        if valid_run:
                            sum_throughput_per_run.append(run_sum)

                    # Calculate the average and standard deviation of the throughput sums
                    if sum_throughput_per_run:
                        if cost_savings and normalize_by_baseline_csv:
                            print("use costsavings and normalize by baseline")
                            baseline_df = pd.read_csv(normalize_by_baseline_csv)
                            #get exclusive throughput from baseline_df
                            base_steps = [float(baseline_df[baseline_df["Type"].str.contains(workloads[i])]["Exclusive100"]) for i in range(len(workloads))]
                            print(workloads, base_steps)
                            sum_base_steps = sum(base_steps)
                            #2* elements in sum_throughput_per_run and divide by sum_base_steps
                            sum_throughput_per_run = [2*step/sum_base_steps for step in sum_throughput_per_run]
                        avg_file_steps[workloads][freq][threads] = float(np.mean(sum_throughput_per_run))
                        std_file_steps[workloads][freq][threads] = float(np.std(sum_throughput_per_run))
                    else:
                        # If no valid runs were found, leave as zero
                        avg_file_steps[workloads][freq][threads] = None
                        std_file_steps[workloads][freq][threads] = None

    return avg_file_steps, std_file_steps

def remove_invalid_runs(aggregated_file_steps, aggregated_power_dict, thread_combinations):
    """
    Remove runs where any workload throughput is None. Also, remove corresponding power data. create a new aggreagate with dictionary
    """
    
    ### helper function to clear values if None

    def clear_values_if_none(steps_dict):
        # Get the length of the first list
        lengths = [len(lst) for lst in steps_dict.values()]
        
        # Check if all lists have the same length
        if len(set(lengths)) != 1:
            raise ValueError("All lists in the dictionary must have the same length.")
        
        # Get the length of the lists (all are the same length)
        list_length = lengths[0]
        marked_invalid_indices = []  # List to store indices that were cleared

        # Iterate through each index in the list
        for i in range(list_length):
            # Check if any value at index i is None
            if any(steps_dict[key][i] is None for key in steps_dict):
                # If any value is None, mark the index and clear the value at index i for all keys
                marked_invalid_indices.append(i)
                # If any value is None, clear the value at index i for all keys
                for key in steps_dict:
                    steps_dict[key][i] = None

        return marked_invalid_indices


    aggregate_file_steps_percent_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for workloads, freq_dict in aggregated_file_steps.items():
        for freq, workload_steps_dict in freq_dict.items():
            for threads in thread_combinations:
                throughput_per_combination = defaultdict(list)
                #print(f"processing: Workloads: {workloads}, Frequency: {freq}, Threads: {threads}")
                # Collect throughput values for the current combination of workloads and threads
                for idx, workload in enumerate(workloads):
                    key = f'w{idx+1}_{workload}_MPS{threads[idx]}'
                    value = workload_steps_dict.get(key, None)
                    throughput_per_combination[key] = value
                #print(throughput_per_combination)
                #if all values are None, skip the current combination
                if all(value is None for value in throughput_per_combination.values()):
                    continue
                marked_invalid_runs = clear_values_if_none(throughput_per_combination)
                
                # Check if any throughput in the combination is None, and remove the corresponding run
                #delete the invalid runs from power dict using deleted_runs
                 # Also remove corresponding power data
                if tuple(threads) in aggregated_power_dict[workloads][freq]:
                    for idx in marked_invalid_runs:
                        aggregated_power_dict[workloads][freq][tuple(threads)][idx] = None
                        #start here
                    #print(f"power dict after removing invalid runs: {aggregated_power_dict[workloads][freq][tuple(threads)]}")
            
                #save the valid thorughput per combination into aggregate_file_steps_percent_dict
                aggregate_file_steps_percent_dict[workloads][freq][tuple(threads)] = [throughput_per_combination[key] for key in throughput_per_combination]
           # print(f"aggregate_file_steps_percent_dict at frequency{freq}: {aggregate_file_steps_percent_dict[workloads][freq]}")

    #print(f"aggregate_file_steps_percent_dict: {aggregate_file_steps_percent_dict}")
    return aggregate_file_steps_percent_dict
    
def remove_invalid_runs_threaddir(aggregated_file_steps, aggregated_power_dict, aggregate_duration_dict=None, aggregate_energy_dict=None, aggregate_steps_count_dict=None):
    """
    Remove runs where any workload throughput is None. Also, remove corresponding power, duration, energy, and steps_count data.
    """

    # Helper function to clear values if None
    def clear_values_if_none(steps_dict):
        # Get the length of the first list
        lengths = [len(lst) for lst in steps_dict.values()]
        print(f"current steps_dict before cleaned : {steps_dict}")
        # Check if all lists have the same length
        if len(set(lengths)) != 1:
            raise ValueError("All lists in the dictionary must have the same length.")
        
        # Get the length of the lists (all are the same length)
        list_length = lengths[0]
        marked_invalid_indices = []  # List to store indices that were cleared

        # Iterate through each index in the list
        for i in range(list_length):
            # Check if any value at index i is None
            if any(steps_dict[key][i] is None for key in steps_dict):
                # If any value is None, mark the index and clear the value at index i for all keys
                marked_invalid_indices.append(i)
                # Clear the value at index i for all keys
                for key in steps_dict:
                    steps_dict[key][i] = None
        print(f"marked_invalid_indices: {marked_invalid_indices}")
        return marked_invalid_indices

    # Dictionary to store valid steps after removing invalid runs
    aggregate_file_steps_percent_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # Dictionary to store valid steps_count after removing invalid runs (transformed to list format)
    aggregate_steps_count_percent_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Iterate through aggregated_file_steps
    for workloads, freq_dict in aggregated_file_steps.items():
        for freq, workload_steps_dict in freq_dict.items():
            for threads, throughput_per_combination in workload_steps_dict.items():
                # Collect throughput values for the current combination of workloads and threads
                retrieved_throughput_per_combination = defaultdict(list)
                # Also collect steps_count values for the current combination
                retrieved_steps_count_per_combination = defaultdict(list)
                
                for idx, workload in enumerate(workloads):
                    key = f'w{idx+1}_{workload}_MPS{threads[idx]}'
                    value = throughput_per_combination.get(key, None)
                    retrieved_throughput_per_combination[key] = value
                    # Also retrieve steps_count for this workload
                    if aggregate_steps_count_dict and workloads in aggregate_steps_count_dict and freq in aggregate_steps_count_dict[workloads] and threads in aggregate_steps_count_dict[workloads][freq]:
                        steps_count_value = aggregate_steps_count_dict[workloads][freq][threads].get(key, None)
                        retrieved_steps_count_per_combination[key] = steps_count_value
                
                # If all throughput values are None, skip the current combination
                if all(value is None for value in retrieved_throughput_per_combination.values()):
                    continue
                
                # Clear invalid values and mark invalid indices
                marked_invalid_runs = clear_values_if_none(retrieved_throughput_per_combination)
                
                # Remove invalid power data corresponding to the cleared indices
                if threads in aggregated_power_dict[workloads][freq]:
                    for idx in marked_invalid_runs:
                        aggregated_power_dict[workloads][freq][threads][idx] = None
                #remove invalid duration data corresponding to the cleared indices
                if aggregate_duration_dict and threads in aggregate_duration_dict[workloads][freq]:
                    for idx in marked_invalid_runs:
                        aggregate_duration_dict[workloads][freq][threads][idx] = None
                #remove invalid energy data corresponding to the cleared indices
                if aggregate_energy_dict and threads in aggregate_energy_dict[workloads][freq]:
                    for idx in marked_invalid_runs:
                        aggregate_energy_dict[workloads][freq][threads][idx] = None
                #remove invalid steps_count data corresponding to the cleared indices
                if aggregate_steps_count_dict and workloads in aggregate_steps_count_dict and freq in aggregate_steps_count_dict[workloads] and threads in aggregate_steps_count_dict[workloads][freq]:
                    for workload_key in aggregate_steps_count_dict[workloads][freq][threads]:
                        for idx in marked_invalid_runs:
                            if idx < len(aggregate_steps_count_dict[workloads][freq][threads][workload_key]):
                                aggregate_steps_count_dict[workloads][freq][threads][workload_key][idx] = None
                # Also clear invalid indices in retrieved_steps_count_per_combination
                for workload_key in retrieved_steps_count_per_combination:
                    if retrieved_steps_count_per_combination[workload_key] is not None:
                        for idx in marked_invalid_runs:
                            if idx < len(retrieved_steps_count_per_combination[workload_key]):
                                retrieved_steps_count_per_combination[workload_key][idx] = None
            
                # Save the valid throughput per combination into aggregate_file_steps_percent_dict
                aggregate_file_steps_percent_dict[workloads][freq][threads] = [retrieved_throughput_per_combination[key] for key in retrieved_throughput_per_combination]
                # Save the valid steps_count per combination into aggregate_steps_count_percent_dict (transformed to list format)
                aggregate_steps_count_percent_dict[workloads][freq][threads] = [retrieved_steps_count_per_combination[key] for key in retrieved_steps_count_per_combination]

    return aggregate_file_steps_percent_dict, aggregate_steps_count_percent_dict


    

def save_multiinstance_share_file_steps(share_file_steps,filename, n_combination):
    import csv
    import itertools
    # Define the set of possible values for x, y, and z
    possible_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Create a list to store the valid combinations
    thread_combinations = [tuple([100] * n_combination)]

    # Iterate through all possible combinations of x, y, and z
    # Generate all possible combinations of thread percentages for n_combination workloads
    for combination in itertools.product(possible_values, repeat=n_combination):
        if sum(combination) == 100:
            thread_combinations.append(combination)

    
    header = []
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for i in range(n_combination):
            header += [f'workload{i+1}']
        #header += [f'(w1_{x}, w2_{y}, w3_{z})' for x,y,z in thread_combinations]
        # Generalize the header creation for thread combinations based on n_combination
        comb_labels = [f"({', '.join([f'w{i+1}_{thread}' for i, thread in enumerate(threads)])})"
                       for threads in thread_combinations]
        
        header += comb_labels
        csvwriter.writerow(header)

        for workloads, throughput_dict in share_file_steps.items():
            row = list(workloads)
            for threads in thread_combinations:
                throughput_per_combination = []
                for idx in range(n_combination):
                    thread = threads[idx]
                    workload = workloads[idx]
                    key = f'w{idx+1}_{workload}_MPS{thread}'
                    value = throughput_dict.get(key, None)
                    throughput_per_combination.append(value)
                row.append(tuple(throughput_per_combination))
            csvwriter.writerow(row)


import csv

def save_multiinstance_share_file_steps_threads(share_file_steps, filename, n_combination):
    import csv
    import itertools
    # Define the set of possible values for x, y, and z
    possible_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Create a list to store the valid combinations
    thread_combinations = [tuple([100] * n_combination)]

    # Iterate through all possible combinations of thread percentages for n_combination workloads
    for combination in itertools.product(possible_values, repeat=n_combination):
        if sum(combination) == 100:
            thread_combinations.append(combination)

    header = []
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for i in range(n_combination):
            header += [f'workload{i+1}']
        
        # Generalize the header creation for thread combinations based on n_combination
        comb_labels = [f"({', '.join([f'w{i+1}_{thread}' for i, thread in enumerate(threads)])})"
                       for threads in thread_combinations]
        
        header += comb_labels
        csvwriter.writerow(header)

        for workloads, throughput_dict in share_file_steps.items():
            row = list(workloads)
            for threads in thread_combinations:
                throughput_per_combination = []
                for idx in range(n_combination):
                    thread = threads[idx]
                    workload = workloads[idx]
                    key = f'w{idx+1}_{workload}_MPS{thread}'
                    value = throughput_dict.get(key, None)
                    throughput_per_combination.append(value)
                row.append(tuple(throughput_per_combination))
            csvwriter.writerow(row)

def save_multiinstance_avg_power(power_dict, filename, n_combination):
    import csv
    import itertools
    # Define the set of possible values for x, y, and z
    possible_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Create a list to store the valid combinations
    thread_combinations = [tuple([100] * n_combination)]

    # Iterate through all possible combinations of thread percentages for n_combination workloads
    for combination in itertools.product(possible_values, repeat=n_combination):
        if sum(combination) == 100:
            thread_combinations.append(combination)

    header = []
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for i in range(n_combination):
            header += [f'workload{i+1}']
        
        # Generalize the header creation for thread combinations based on n_combination
        comb_labels = [f"({', '.join([f'w{i+1}_{thread}' for i, thread in enumerate(threads)])})"
                       for threads in thread_combinations]
        
        header += comb_labels
        csvwriter.writerow(header)

        for workloads, power_dict in power_dict.items():
            row = list(workloads)
            for threads in thread_combinations:    
                value = power_dict.get(threads, None)
                   
                row.append(float(value)if value else None)
            csvwriter.writerow(row)

def save_share_file_steps(share_file_steps,filename):
    # Define the thread combinations
    thread_combinations = [(10, 90), (20, 80), (30, 70), (40, 60), (50, 50), (60, 40), (70, 30), (80, 20), (90, 10), (100, 100)]

    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
       
        header = ['workload1', 'workload2'] + [f'(LS{ls}, BE{be})' for ls, be in thread_combinations]
        csvwriter.writerow(header)
        
        # Write data
        for (workload1, workload2), throughput_dict in share_file_steps.items():
            row = [workload1, workload2]
            for ls, be in thread_combinations:
                ls_key = f'LS{ls}_{workload1}'
                be_key = f'BE{be}_{workload2}'
                ls_value = throughput_dict.get(ls_key, None)
                be_value = throughput_dict.get(be_key, None)
                row.append((ls_value, be_value))
            csvwriter.writerow(row)

def save_sum_std_freqscale(aggregated_share_file_steps):
    import csv
    import itertools
    
    # Define the set of possible values for thread percentages
    possible_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Create a list to store the valid combinations of thread percentages for n_combination workloads
    thread_combinations = [tuple([100] * n_combination)]
    for combination in itertools.product(possible_values, repeat=n_combination):
        if sum(combination) == 100:
            thread_combinations.append(combination)

    header = []
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Prepare the header
        for i in range(n_combination):
            header += [f'workload{i+1}']
        
        # Generalize the header creation for thread combinations based on n_combination
        comb_labels = [f"({', '.join([f'w{i+1}_{thread}' for i, thread in enumerate(threads)])})"
                       for threads in thread_combinations]
        
        header += comb_labels
        csvwriter.writerow(header)

        # Write rows based on shared file steps and power data
        for workloads, throughput_freq_dict in aggregated_share_file_steps.items():
            for freq, throughput_dict in throughput_freq_dict.items():
                #throughput dict key value - (thread, list of throughputs)
                row = list(workloads) + [freq] * n_combination  # Adding workloads and frequencies
                for threads in thread_combinations:
                    sum_throughput_per_combination = 0
                    for idx in range(n_combination):
                        thread = threads[idx]
                        workload = workloads[idx]
                        key = f'w{idx+1}_{workload}_MPS{thread}'
                        value = throughput_dict.get(key, None)
                        if len(value) > 0:
                            print(value)
                            std_throughput_per_combination = float(np.std(value))
                        else:
                            std_throughput_per_combination = None
                        
                    row.append(std_throughput_per_combination)


                csvwriter.writerow(row)

#write aggreagated sum and std of throughput data to csv file
def save_with_freqscale_sum_avg_std(share_file_steps, filename, n_combination):
    import csv
    
    #thread_combinations = generate_thread_combinations(n_combination)
    #get thread combinations from share file steps
    unique_thread_combinations = set()
    #unique_frequencies = set()
    #unique_workloads = set()

    # Iterate through aggregated file steps to collect unique thread combinations, frequencies, and workloads
    for workloads, freq_dict in share_file_steps.items(): 
        for freq, workload_steps_dict in freq_dict.items():
            for thread_combination in workload_steps_dict.keys():
                unique_thread_combinations.add(thread_combination)  # Thread combinations

    # Convert sets to sorted lists for consistency
    thread_combinations = sorted(unique_thread_combinations)
    print(f"thread_combinations {thread_combinations}")

    header = []
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Prepare the header
        for i in range(n_combination):
            header += [f'workload{i+1}']
        for i in range(n_combination):
            header += [f'freq{i+1}']
        
        # Generalize the header creation for thread combinations based on n_combination
        comb_labels = [f"({', '.join([f'w{i+1}_{thread}' for i, thread in enumerate(threads)])})"
                       for threads in thread_combinations]
        
        header += comb_labels
        csvwriter.writerow(header)

        # Write rows based on shared file steps and power data
        for workloads, throughput_freq_dict in share_file_steps.items():
            for freq, throughput_dict in throughput_freq_dict.items():
                row = list(workloads) + [freq] * n_combination  # Adding workloads and frequencies
                for threads in thread_combinations:
                    value = throughput_dict.get(tuple(threads), None)
                    row.append(value)
                


                csvwriter.writerow(row)
    #save power data by throughput keys
    #save_power_with_freqscale(share_file_steps, power_dict, f"{filename[:-4]}_power.csv", n_combination)

import csv
from statistics import mean, stdev

def save_with_freqscale_individual_avg_std(share_file_steps, avg_filename, std_filename, n_combination):
    """
    Write the  indiidual workload's average, and standard deviation of throughput data to a CSV file
    """
    # #region agent log
    import json
    with open('/tmp/peace_parse_debug.log', 'a') as f:
        sample_key = list(share_file_steps.keys())[0] if share_file_steps else None
        if sample_key:
            sample_freq = list(share_file_steps[sample_key].keys())[0]
            sample_thread = list(share_file_steps[sample_key][sample_freq].keys())[0]
            sample_value = share_file_steps[sample_key][sample_freq][sample_thread]
            f.write(json.dumps({"sessionId":"debug-session","runId":"pre-fix","hypothesisId":"A,C,E","location":"parse_util.py:1573","message":"save_with_freqscale_individual_avg_std input","data":{"sample_value_type":str(type(sample_value)),"sample_value_preview":str(sample_value)[:200],"is_dict":isinstance(sample_value, dict),"is_list":isinstance(sample_value, list)},"timestamp":__import__('time').time()*1000})+'\n')
    # #endregion
    # Get unique thread combinations from the share_file_steps
    unique_thread_combinations = set()

    # Collect all thread combinations that exist in the share_file_steps dict
    for workloads, freq_dict in share_file_steps.items():
        for freq, workload_steps_dict in freq_dict.items():
            for thread_combination in workload_steps_dict.keys():
                unique_thread_combinations.add(thread_combination)  # Collect thread combinations

    # Sort thread combinations for consistent ordering in the CSV
    thread_combinations = sorted(unique_thread_combinations)

    print(f"Thread combinations: {thread_combinations}")

    # Prepare the header
    header = []
    for i in range(n_combination):
        header.append(f'workload{i + 1}')
    for i in range(n_combination):
        header.append(f'freq{i + 1}')
    
    # Create headers for each thread combination
    comb_labels = [f"({', '.join([f'w{i + 1}_{thread}' for i, thread in enumerate(threads)])})"
                    for threads in thread_combinations]
    header += comb_labels


    # Open the output avg CSV file
    with open(avg_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)


        # Write the header to the CSV file
        csvwriter.writerow(header)

        # Write the data rows based on shared file steps and power data
        for workloads, throughput_freq_dict in share_file_steps.items():
            for freq, throughput_dict in throughput_freq_dict.items():
                row = list(workloads) + [freq] * n_combination  # Adding workloads and frequencies
                
                for threads in thread_combinations:
                    throughput_values = throughput_dict.get(threads, None)
                    print(f"workload={workloads}, threads={threads}, throughput={throughput_values}")
                    # #region agent log
                    import json
                    with open('/tmp/peace_parse_debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"pre-fix","hypothesisId":"A,E","location":"parse_util.py:1619","message":"throughput_values structure","data":{"type":str(type(throughput_values)),"is_dict":isinstance(throughput_values, dict),"is_list":isinstance(throughput_values, list),"preview":str(throughput_values)[:200] if throughput_values else None},"timestamp":__import__('time').time()*1000})+'\n')
                    # #endregion
                    # If there are throughput values, calculate mean and standard deviation
                    if throughput_values and all(v is not None for v in throughput_values):
                        avg_values = []
                        for i in range(n_combination):
                            valid_values = [v for v in throughput_values[i] if v is not None]
                            if valid_values:
                                avg_values.append(mean(valid_values))
                            else:
                                avg_values.append(None)
                        row.append(f"({', '.join([f'{avg:.3f}' if avg is not None else 'None' for avg in avg_values])})")
                    else:
                        #append (None)*combination using join

                        row.append(f"({', '.join(['None']*n_combination)})")
                
                # Write the row to the CSV
                csvwriter.writerow(row)

    print(f"avg File saved successfully as: {avg_filename}")

    #open the std file
    import numpy as np
    with open(std_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Write the header to the CSV file
        csvwriter.writerow(header)

        # Write the data rows based on shared file steps and power data
        for workloads, throughput_freq_dict in share_file_steps.items():
            for freq, throughput_dict in throughput_freq_dict.items():
                row = list(workloads) + [freq] * n_combination  # Adding workloads and frequencies
                
                for threads in thread_combinations:
                    throughput_values = throughput_dict.get(threads, None)
                    #print(throughput_values)
                    # If there are throughput values, calculate mean and standard deviation
                    if throughput_values and all(v is not None for v in throughput_values):
                        std_values = []
                        for i in range(n_combination):
                            valid_values = [v for v in throughput_values[i] if v is not None]
                            if valid_values:
                                std_values.append(float(np.std(valid_values)))
                            else:
                                std_values.append(None)
                        row.append(f"({', '.join([f'{std:.3f}' if std is not None else 'None' for std in std_values])})")
                    else:
                        row.append("(None, None)")
                
                # Write the row to the CSV
                csvwriter.writerow(row)

    print(f"std File saved successfully as: {std_filename}")





def save_with_freqscale(share_file_steps, power_dict, filename, n_combination):
    import csv
    import itertools
    
    # Define the set of possible values for thread percentages
    possible_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Create a list to store the valid combinations of thread percentages for n_combination workloads
    thread_combinations = [tuple([100] * n_combination)]
    for combination in itertools.product(possible_values, repeat=n_combination):
        if sum(combination) == 100:
            thread_combinations.append(combination)

    header = []
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Prepare the header
        for i in range(n_combination):
            header += [f'workload{i+1}']
        for i in range(n_combination):
            header += [f'freq{i+1}']
        
        # Generalize the header creation for thread combinations based on n_combination
        comb_labels = [f"({', '.join([f'w{i+1}_{thread}' for i, thread in enumerate(threads)])})"
                       for threads in thread_combinations]
        
        header += comb_labels
        csvwriter.writerow(header)

        # Write rows based on shared file steps and power data
        for workloads, throughput_freq_dict in share_file_steps.items():
            for freq, throughput_dict in throughput_freq_dict.items():
                row = list(workloads) + [freq] * n_combination  # Adding workloads and frequencies
                for threads in thread_combinations:
                    throughput_per_combination = []
                    for idx in range(n_combination):
                        thread = threads[idx]
                        workload = workloads[idx]
                        key = f'w{idx+1}_{workload}_MPS{thread}'
                        value = throughput_dict.get(key, None)
                        throughput_per_combination.append(value)
                    row.append(tuple(throughput_per_combination))


                csvwriter.writerow(row)
    #save power data by throughput keys
    save_power_with_freqscale(share_file_steps, power_dict, f"{filename[:-4]}_power.csv", n_combination)
    

def save_power_with_freqscale(share_file_steps, power_dict, filename, n_combination):
    import csv
    import itertools
    
    # Define the set of possible values for thread percentages
    possible_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Create a list to store the valid combinations of thread percentages for n_combination workloads
    thread_combinations = [tuple([100] * n_combination)]
    for combination in itertools.product(possible_values, repeat=n_combination):
        if sum(combination) == 100:
            thread_combinations.append(combination)

    # Open a new CSV file to write the power data
    header = []
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Prepare the header
        for i in range(n_combination):
            header += [f'workload{i+1}']
        for i in range(n_combination):
            header += [f'freq{i+1}']
        
        # Generalize the header creation for thread combinations based on n_combination
        comb_labels = [f"({', '.join([f'w{i+1}_{thread}' for i, thread in enumerate(threads)])})"
                       for threads in thread_combinations]
        
        header += comb_labels
        csvwriter.writerow(header)

        # Iterate over share_file_steps to find corresponding power values
        for workloads, throughput_freq_dict in share_file_steps.items():
            for freq in throughput_freq_dict.keys():  # Only iterate through frequencies in file_steps
                row = list(workloads) + [freq] * n_combination  # Add workloads and frequencies to the row
                
                # Add power values corresponding to the thread combinations
                for threads in thread_combinations:
                    power_value = None
                    if freq in power_dict.get(tuple(workloads), {}):
                        power_value = power_dict[tuple(workloads)][freq].get(threads, None)
                        #check if throughput freq dict has none in value
                        for idx in range(n_combination):
                            thread = threads[idx]
                            workload = workloads[idx]
                            key = f'w{idx+1}_{workload}_MPS{thread}'
                            value = throughput_freq_dict[freq].get(key, None)
                            if value is None:
                                #one of output is None - row append none. power is not used
                                power_value = None
                                #print(f"workloadkey: {key} has None value in throughput_freq_dict.")
                                #print(f"thread combination: {threads} should be none")
                                break
                            
                            
                    
                    
                    # Append the power value to the row (or None if missing)
                    row.append(float(power_value) if power_value is not None else None)
                
                # Write the row to the CSV
                csvwriter.writerow(row)

# Function to save the collected data as CSV
def get_all_base_results(directory, BEtype, dcgm_columns, idle_power=48, autoregressive_1batch=False, is_freqscale=False):
    
    THREAD_PERCENTAGES = [i for i in range(10, 101, 10)]
    #THREAD_PERCENTAGES = [100]
    print(f"current dir={directory}")
    BASELINE_THREADS = ["MPS" + str(i) for i in THREAD_PERCENTAGES]
    results = defaultdict(dict)
    print(f"auroregressive_1batch: {autoregressive_1batch}")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('BE'):
                BE_dir = os.path.basename(root).replace("BE_", "")
                print(f"BE_dir: {BE_dir}")
                batch_size = int(BE_dir.split("_")[-1][5:])
                file_path = os.path.abspath(os.path.join(root, file))
                print(f"Processing file: {file_path}, Batch size: {batch_size}")
                model_type = BE_dir+"-"+BEtype

                for MPSpercent in BASELINE_THREADS:
                    if MPSpercent + ".log" in file.split("_"):
                        if BEtype == "train":
                            result = parse_train_log_avgStep_file(file_path)
                            if result is not None:
                                steps, _ = result
                                throughput = steps * batch_size if not autoregressive_1batch else steps * batch_size
                                results[model_type][f'Exclusive{MPSpercent[3:]}'] = throughput
                        elif BEtype == "inf":
                            result = parse_inf_log_avgStep_file(file_path)
                            if result is not None:
                                steps, _ = result
                                throughput = steps * batch_size if not autoregressive_1batch else steps * batch_size
                                results[model_type][f'Exclusive{MPSpercent[3:]}'] = throughput
                        
                        elif BEtype == "cuda_samples":
                            workload = BE_dir.split("_")[0]
                            result = parse_cuda_samples_log_avgStep_file(file_path, workload=workload)
                            if result is not None:
                                throughput, _ = result
                                results[model_type][f'Exclusive{MPSpercent[3:]}'] = throughput
                        break
                        
                        

                            

                    # If gpu_info.csv file, calculate sm% and mem% for MPS100 only
                    if MPSpercent in file.split("_") and file.endswith("_gpu_info.csv"):
                        filepath = os.path.join(root, file)
                        avg_sm, avg_mem = calculate_average_sm_memory(filepath)
                        results[model_type][f'sm%{MPSpercent[3:]}'] = avg_sm
                        results[model_type][f'mem%{MPSpercent[3:]}'] = avg_mem

                    # If gpu_mem.csv file, calculate memory cap (GB)
                    if MPSpercent in file.split("_") and file.endswith("_gpu_mem.csv"):
                        filepath = os.path.join(root, file)
                        gb_max = parse_memory_used_to_gb(filepath)
                        power, _, _ = calculate_average_power_duration_adjusted(filepath, idle_power=idle_power)
                        results[model_type][f'memcap{MPSpercent[3:]}'] = gb_max
                        results[model_type][f'power_Exclusive{MPSpercent[3:]}'] = power

                    if MPSpercent in file.split("_") and file.endswith("_dcgm_info.csv"):
                        filepath = os.path.join(root, file)
                        
                        features = calculate_average_gcgm_usage(filepath, columns = dcgm_columns)
                        for feature, value in features.items():
                            results[model_type][f'{feature}{MPSpercent[3:]}'] = value
                        
   

    # Save the results to CSV
    print(results)
    return results
    #save results in csv, with results first key as rows, second ket as columns


def save_baseline_to_csv(results, dcgm_columns ,output_file, is_freqscale=False):
    # Define the header
    
    header = ['Type']
    if is_freqscale:
        header += ['freq']
    exclusive_header = [f'sm%{i}' for i in range(10, 101, 10)]
    exclusive_header += [f'mem%{i}' for i in range(10, 101, 10)]
    exclusive_header += [f'memcap{i}' for i in range(10, 101, 10)]
    exclusive_header += [f"Exclusive{i}" for i in range(10, 101, 10)]
    exclusive_header += [f"power_Exclusive{i}" for i in range(10, 101, 10)]
    #add dcgm columns
    exclusive_header += [f"{col}{i}" for col in dcgm_columns for i in range(10, 101, 10)]
    header += exclusive_header

    # Open the output CSV file in write mode
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(header)
        if is_freqscale:
            # Write each row from result_dict with freq: model_type : metrics
            for BEtype, results_freq2metrics in results.items():
                for freq in results_freq2metrics.keys():
                    #print (f"results with {freq}: {results[freq]}")
                    for model_type, metrics in results_freq2metrics[freq].items():
                        #skip type
                        
                        data = [metrics.get(header, '') for header in header[2:]]
                        #check if any data is empty string
                        if '' in data:
                            #print metric of feat type with missing data
                            for  i, key in enumerate(metrics.keys()):
                                if data[i] == '':
                                    print(f" {model_type} has missing data: {key}")
                        
                        #+= model_type
                        data = [model_type, freq] + data
                        writer.writerow(data)
            
        else:
            # Write each row from result_dict
            for model_type, metrics in results.items():
                #skip type
                data = [metrics.get(header, '') for header in header[1:]]
                #check if any data is empty string
                if '' in data:
                    #print metric of feat type with missing data
                    for  i, key in enumerate(metrics.keys()):
                        if data[i] == '':
                            print(f" {model_type} has missing data: {key}")
                    

                #+= model_type
                data = [model_type] + data
                writer.writerow(data)



if __name__ == "__main__":
    # Provide the filename as an argument
    filename = sys.argv[1]
    
    #parse_inf_log_avgStep_file("/Users/bing/Downloads/BE_openai/whisper-large-v2_batch2/BE_speech-recognition_MPS100.log")

    #get_multiworkloads_share_avgStep(filename,3, )

    # Calculate average sm and memory
    #avg_sm, avg_mem = calculate_average_sm_memory(filename)
    # Example usage
    """
    csv_file = '/home/cc/mlProfiler/tests/mps/ccv100/baseline/batch4_mem/train/RUN1/LS0/speech-recognition/openai/whisper-large-v2_batch1/BE_bert-base-cased_batch4/BE_recommend_MPS100_gpu_mem.csv'  # Replace with your actual file path
    gb_values = parse_memory_used_to_gb(csv_file)
    print(gb_values)
    output_file = f"baseline_metrics.csv"
    # Save the average sm and memory to a new CSV file
    dirname = sys.argv[1]
    
    BEtypes = ["train", "inf"]  # or "train"
    result_dict = defaultdict(dict)
    for BEtype in BEtypes:
        
    #save_to_csv(avg_sm, avg_mem, filename)
        result = get_all_base_results(f"{dirname}/{BEtype}", BEtype, autoregressive_1batch=True)
        #update result_dict with result
        result_dict.update(result)
    save_baseline_to_csv(result_dict, f"baseline_metrics.csv")
    """
    #test: parse cuda samples
    filename = "../../../../ccvlogs/ccv100_02042025/mlProfiler/tests/mps/ccv100_logs/baseline_dcgm/train/RUN1/LS0/speech-recognition/openai/whisper-large-v2_batch1/BE_mobilenet_batch2/BE_imgclassification_MPS10_dcgm_info.csv"
    calculate_average_gcgm_usage(filename, 
                    columns = ["SMACT%", "SMOCC%", "TENSO%", "DRAMA%", "FP64A%", "FP32A%", "FP16A%"])
    #parse_cuda_samples_log_avgStep_file(filename, "BlackScholes")
    #print(calculate_average_power_adjusted(filename, idle_power=48))

        
