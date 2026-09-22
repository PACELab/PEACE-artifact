import pandas as pd
import sys
import os
from collections import defaultdict
from parse_util import *

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
    if len(sys.argv) < 3:
        print("please provide the input_shared_log_dir  output prefix")
        print("python parse_baseline_sysmetrics.py inputfile output_predix")
        exit(1)
    output_file = f"baseline_metrics.csv"
    # Save the average sm and memory to a new CSV file
    dirname = sys.argv[1]
    output_prefix = sys.argv[2]
    

    dcgm_columns = ["SMACT%", "SMOCC%", "TENSO%", "DRAMA%", "FP32A%", "PCITX", "PCIRX" ]
    BEtypes = ["train", "inf", "autoregressive", "cuda_samples"]  # or "train"
    result_dict = defaultdict(dict)
    for BEtype in BEtypes:
        autoregressive_1batch=True if BEtype == "autoregressive" else False
    #save_to_csv(avg_sm, avg_mem, filename)
        result = get_all_base_results(f"{dirname}/{BEtype}", BEtype, dcgm_columns=dcgm_columns, idle_power=41, autoregressive_1batch=autoregressive_1batch)
        #update result_dict with result
        result_dict.update(result)
    save_baseline_to_csv(result_dict, dcgm_columns=dcgm_columns, output_file=f"{output_prefix}_{output_file}")