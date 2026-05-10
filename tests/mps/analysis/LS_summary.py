
# read and calculate mean of IAT file
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
#IAT is a json file with a list of IAT. read the file and calculate the mean, median, stddev, max, min, 95th, 99th, 99.9th percentile of IAT
#usage: python3 read_IAT.py <file>
#example: python3 read_IAT.py IAT.json

def read_IAT(filename):
    if not os.path.exists(filename):
        print("File {} does not exist".format(filename))
        sys.exit(1)
    with open(filename, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data, columns=["iat"])
    print("Mean IAT: {:.2f}".format(df["iat"].mean()))
    print("Median IAT: {:.2f}".format(df["iat"].median()))
    print("Stddev IAT: {:.2f}".format(df["iat"].std()))
    print("Max IAT: {:.2f}".format(df["iat"].max()))
    print("Min IAT: {:.2f}".format(df["iat"].min()))
    print("IAT 95th percentile: {:.2f}".format(df["iat"].quantile(0.95)))
    print("IAT 99th percentile: {:.2f}".format(df["iat"].quantile(0.99)))
    print("IAT 99.9th percentile: {:.2f}".format(df["iat"].quantile(0.999)))

    plt.hist(df["iat"], bins=100)
    plt.show()

#read_IAT("/home/cc/mlProfiler/tests/mps/IAT/0301/0301_500reqs_lambd0.5.json")


#BASELINE
#read LS metric from baseline and clean data
#plot the LS metric for each baseline and clean data
import csv
import pandas as pd
#read csv from "/home/cc/mlProfiler/tests/mps/rtx6000_logs/baseline/LS/500req_lambd0.5/LSmetrics.csv" abd store in pandas dataframe. get the mean of key= vllm:avg_generation_throughput_toks_per_s
#csv format - Metric Name,Metric Values
#vllm:avg_generation_throughput_toks_per_s,0.000000
all_file_metrics = {}
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
def parse_LS_metrics(filename):
    df = pd.read_csv(filename)
    df["Metric Values"] = df["Metric Values"].tolist()

    # get avg_generation_throughput_toks_per_s
    generation_throughput = df[df["Metric Name"] == f"vllm:avg_generation_throughput_toks_per_s{{model_name=\"{model_name}\"}}"]["Metric Values"]
    #eval convert str of list to list
    #print(np.array(generation_throughput)[0])
    generation_throughput = eval(np.array(generation_throughput)[0])
    #print(generation_throughput)
    #trim zeros in generation_throughput
    generation_throughput = np.trim_zeros(generation_throughput, 'b')

    all_file_metrics["total_recording_sec"] = len(generation_throughput)
    #get average prompt throughput token/s
    prompt_throughput =  df[df["Metric Name"].str.contains("avg_prompt_throughput_toks_per_s")]["Metric Values"]
    prompt_throughput = eval(np.array(prompt_throughput)[0])
    prompt_throughput = np.trim_zeros(prompt_throughput, 'b')
    #print(f"prompt_throughput: {prompt_throughput}")

    #print(generation_throughput)
    generation_throughput_mean = np.mean(generation_throughput)
    prompt_throughput_mean = np.mean(prompt_throughput)
    #print(f"mean of generation throughput: {generation_throughput_mean}")
    #print(f"mean of prompt throughput: {prompt_throughput_mean}")
    all_file_metrics["generation_throughput"] = generation_throughput_mean
    all_file_metrics["prompt_throughput"] = prompt_throughput_mean
    

    # get total generation token using generation_tokens_total
    generation_tokens_total = df[df["Metric Name"].str.contains("generation_tokens_total")]["Metric Values"]
    generation_tokens_total = eval(np.array(generation_tokens_total)[0])
    all_file_metrics["generation_tokens_total"] = max(generation_tokens_total)

    # get gpu usage
    #get gpu_cache_usage_perc as kvcache usage and plot
    gpu_cache_usage_perc = df[df["Metric Name"].str.contains("gpu_cache_usage_perc")]["Metric Values"]
    gpu_cache_usage_perc = eval(np.array(gpu_cache_usage_perc)[0])
    gpu_cache_usage_perc = np.trim_zeros(gpu_cache_usage_perc, 'b')
    all_file_metrics["gpu_cache_usage_perc"] = np.mean(gpu_cache_usage_perc)

    #get request metrics
    request_metrics = ["num_requests_running", "num_requests_waiting", "requests_total_counter", "status_codes_counter"]
    values = []
    for field in request_metrics:
        value = df[df["Metric Name"].str.contains(field)]["Metric Values"]
        #print(f"field={field}, value={value}")
        if np.array(value).size == 0:
            all_file_metrics[field] = 0
            continue
        value = eval(np.array(value)[0])
        all_file_metrics[field] = max(value)

    #time per output token
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, "+inf"]
    time_per_output_token_buckets = []
    df_all_buckets = df[df["Metric Name"].str.contains("time_per_output_token_seconds_bucket")]["Metric Values"]
    df_all_buckets = np.array(df_all_buckets)
    #print(df_all_buckets)
    #only append last element in each bucket
    all_file_metrics["time_per_output_token_seconds_bucket"] = {}
    for i in range(len(df_all_buckets)):
        all_file_metrics["time_per_output_token_seconds_bucket"][buckets[i]] = eval(df_all_buckets[i])[-1]
    


    #time to first token
    buckets=[0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, "+inf"]
    time_to_first_token_buckets = []
    df_all_buckets = df[df["Metric Name"].str.contains("time_to_first_token_seconds_bucket")]["Metric Values"]
    df_all_buckets = np.array(df_all_buckets)
    #print(df_all_buckets)
    #only append last element in each bucket
    all_file_metrics["time_to_first_token_seconds_bucket"] = {}
    for i in range(len(df_all_buckets)):
        all_file_metrics["time_to_first_token_seconds_bucket"][buckets[i]] = eval(df_all_buckets[i])[-1]
    #print(all_file_metrics["time_to_first_token_seconds_bucket"])

    #time to first token sum
    time_to_first_token_seconds_sum = df[df["Metric Name"].str.contains("time_to_first_token_seconds_sum")]["Metric Values"]
    time_to_first_token_seconds_sum = eval(np.array(time_to_first_token_seconds_sum)[0])
    all_file_metrics["time_to_first_token_seconds_sum"] = max(time_to_first_token_seconds_sum)
    #print(all_file_metrics["time_to_first_token_seconds_sum"])
    return all_file_metrics


import re

def parse_waiting_processing_time(log_file_path):
    with open(log_file_path, 'r') as file:
        log_data = file.read()

    # Define the regex pattern to match waiting+processing time
    pattern = r'waiting\+processtime:\s*(\d+\.\d+)s'

    # Find all matches in the log data
    matches = re.findall(pattern, log_data)
    #print(len(matches))
    # Convert the matched strings to float and calculate the total time
    total_time = sum(float(match) for match in matches)

    return total_time



if __name__ == "__main__":
    #read_IAT("/home/cc/mlProfiler/tests/mps/IAT/0301/0301_500reqs_lambd0.5.json")
    parse_LS_metrics("/home/cc/mlProfiler/tests/mps/rtx6000_logs/baseline/LS/500req_lambd0.5/LSmetrics.csv")
    # Example usage:
    log_file_path = '/home/cc/mlProfiler/tests/mps/rtx6000_logs/sharetest/100req_lambd0.05/1LS_1GPUBE/RUN1/LS10/imgclassification/microsoft/resnet-50/arrival_imgclassification_LSMPS10.log'
    total_time = parse_waiting_processing_time(log_file_path)
    print("Total waiting+processing time:", total_time, "seconds")



