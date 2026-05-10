import pandas as pd
import sys
import os
from collections import defaultdict
#append parent dir
sys.path.append('../..')
from analysis.parse_util import *

if __name__ == "__main__":
    # Provide the filename as an argument
    #filename = sys.argv[1]
    #parse_inf_log_avgStep_file("/Users/bing/Downloads/BE_openai/whisper-large-v2_batch2/BE_speech-recognition_MPS100.log")

    #get_multiworkloads_share_avgStep(filename,3, )

    # Calculate average sm and memory
    #avg_sm, avg_mem = calculate_average_sm_memory(filename)
    # Example usage
    #csv_file = '/home/cc/mlProfiler/tests/mps/ccv100/baseline/batch4_mem/train/RUN1/LS0/speech-recognition/openai/whisper-large-v2_batch1/BE_bert-base-cased_batch4/BE_recommend_MPS100_gpu_mem.csv'  # Replace with your actual file path
    #gb_values = parse_memory_used_to_gb(csv_file)
    #args instructions: python parse_baseline_sysmetrics.py /path/to/dir output_prefix
    #print args instructions if len(sys.argv) < 3

    idle_power_dict = {
        300 : 42,
        500: 44, 
        700: 45, 
        900: 46,
        1100: 49,
        1300: 56.2 ,
        1530: 68
    }
    
    if len(sys.argv) < 3:
        print("please provide the input_shared_log_dir  output prefix")
        print("python parse_baseline_sysmetrics.py inputfile output_predix")
        exit(1)
    output_file = f"baseline_metrics.csv"
    # Save the average sm and memory to a new CSV file
    dirname = sys.argv[1]
    output_prefix = sys.argv[2]
    is_freqscale = True

    dcgm_columns = ["SMACT%", "SMOCC%", "TENSO%", "DRAMA%", "FP32A%", "PCITX", "PCIRX" ]
    BEtypes = ["train", "inf", "autoregressive", "cuda_samples"]  # or "train"
    #BEtypes = ["train"]  # or "train"
    #result_dict = defaultdict(lambda: defaultdict(dict))
    result_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))  # Nested defaultdict

    for BEtype in BEtypes:
        autoregressive_1batch=True if BEtype == "autoregressive" else False
    #save_to_csv(avg_sm, avg_mem, filename)
        for freq in idle_power_dict.keys():
            result = get_all_base_results(f"{dirname}/{BEtype}/FREQ{freq}", BEtype, dcgm_columns=dcgm_columns, idle_power=idle_power_dict[freq], autoregressive_1batch=autoregressive_1batch)
            #update result_dict with result
            #if result is not empty:
            if result: 
                #raise ValueError("stop here")
                #result_with_freq = defaultdict(dict)
                #result_with_freq.update({freq: result})
                #print(f"freq={freq}")
                #print(f"results with freq:\n{result}")
                result_dict[BEtype][freq].update(result)
                #result_dict.update(result_with_freq)
                #if BEtype == "train": 
                #    print(f"train result: {result}")
                print(f"result_dict for {BEtype}:\n{result_dict[BEtype]}")
                #raise ValueError("stop here")
                    
            #raise ValueError("stop here")

    print(f"all results: {result_dict}")
    save_baseline_to_csv(result_dict, dcgm_columns=dcgm_columns, output_file=f"{output_prefix}_{output_file}", is_freqscale=is_freqscale)