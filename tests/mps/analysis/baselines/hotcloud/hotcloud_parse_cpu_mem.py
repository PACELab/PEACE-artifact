#parse avg cpu, avg mem from each file to hotcloud_baseline.labels.csv
from hotcloud_util import calculate_averages
import os
import pandas as pd
import sys
from collections import defaultdict
import csv



def parse_files(directory, type):
    #get all files with docker_stats.log
    utils = defaultdict(dict)   # List to store the paths of all files
   
        #add avg_cpu,avg_mem to header column

    #f.write("avg_cpu,avg_mem\n")
    for root, dirs, files in os.walk(directory):

        for file in files:
            if file.startswith('docker_stats'):
                file_path = os.path.abspath(os.path.join(root, file))
                BE_dir = os.path.basename(root)
                #remove "BE" from BE+dir if it exists
                BE_dir = BE_dir.replace("BE_", "")
                print(BE_dir)
            #split file with "_" and find if BASELINE_THREADS is in the split
                avgcpu, avgmem = calculate_averages(file_path)
                utils[f"{BE_dir}-{type}"] = {"avg_cpu": avgcpu, "avg_mem": avgmem}
                
                
                #file_paths.append(file_path)
    return utils

def update_base_csv(utils, filename):
    rows = []
    #update the csv file
    #filename = "hotcloud_baseline.labels.csv"
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames + ['AvgCPU', 'AvgMem']
        for row in reader:
            print(row["Type"])
            if row["Type"] in utils:  
                row['AvgCPU'] = utils[row["Type"]]['avg_cpu']
                row['AvgMem'] = utils[row["Type"]]["avg_mem"]
            else:
                row['AvgCPU'] = 0
                row['AvgMem'] = 0
            rows.append(row)
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def get_all_kernels(directory):
        df_features = pd.DataFrame()
        for file in os.listdir(directory):
            if file.endswith('.csv'):
                df_temp = pd.read_csv(f'{directory}/{file}')
                df_features = pd.concat([df_features, df_temp])
        return df_features

def generate_kernel_features():
    #read all csv files in kernel_profiles/kernel_hotcloud
    df_features = get_all_kernels('kernel_hotcloud')
    #only take columns 
    #label the Type column  in df_feature in labels based on index
    labels = {}
    for i, label in enumerate(df_features['Type'].unique()):
        labels[label] = i
    #df_features['Type'] = df_features.index.map(labels)
    print(labels)
    # Generate combinations of the dictionary keys
    from itertools import combinations_with_replacement
    key_combinations = combinations_with_replacement(labels.keys(), 2)
    types, idx = [], []
    data = []
    # Print each combination and their corresponding values
    for (w1, w2) in key_combinations:
        
        value1 = labels[w1]
        value2 = labels[w2]
        types.append((w1, w2))
        idx.append((value1, value2))

        df_w1 = df_features[df_features['Type'] == w1]
        df_w2 = df_features[df_features['Type'] == w2]
        #get the sum of all columns in df_w1 and df_w2. store it in another array
        df_w1 = df_w1.drop(['Type'], axis=1)
        df_w2 = df_w2.drop(['Type'], axis=1)
        #add w1_ prefix to all df_w1 columns
        df_w1 = df_w1.add_prefix(f"w1_")
        df_w2 = df_w2.add_prefix(f"w2_")
        #merge df_w1 and df_w2 with  all columns are placed side by side
        df_sum =  pd.concat([df_w1, df_w2], axis=1).to_dict(orient='records')[0]


        #use multiplication
        #df_sum = (df_w1 * df_w2).to_dict(orient='records')[0]
        #print(df_sum)
        
        row = df_sum.update({"workload1": w1, "workload2": w2, "idx1": value1, "idx2": value2})
        #print(df_sum)
        data.append(df_sum)
        
        #print(f"({key1}, {key2}) -> ({value1}, {value2})")

    print(len(types),types)
    print(len(idx),idx)



    #create a new dataframe that has columns = [workload1, workload2]
    # rows = types[0] types[1]...
    # Create a DataFrame from the list
    df = pd.DataFrame(data)

    # Print the DataFrame
    df.head()
    #wrte df to csv
    df.to_csv('hotcloud_kernel_labels.csv', header=True, index=False)
if __name__ == "__main__":
    
    utils = defaultdict(dict)
    #utils.update(parse_files("/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/cclogs/ccv100_0624_hotcloud/cpu_util/train", "train"))
    utils.update(parse_files("/home/cc/mlProfiler/tests/mps/analysis/baselines/hotcloud/ccv100/baseline", "inf"))
    print(utils)

    #update_base_csv(utils, "hotcloud_baseline_labels.csv")
    
    
    #generate_kernel_features()