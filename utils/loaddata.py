import nbformat
from nbconvert import PythonExporter
import importlib.util
import sys
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from utils.util_dataloader import get_thread_columns
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler
import re
import joblib



import logging


def getcloud_stage1trainData(baselineData,shareThroughputData, kernelData,outname,targetMPS, selected_feats):
    base_df = pd.read_csv(baselineData)
    #drop row with nan values
    base_df.dropna(inplace=True)
    #store sm in a dict wuth key as Type from base_df, base_df["sm%"] as value
    base_sm_dict = base_df.set_index('Type')['sm%'].to_dict()
    base_mem_dict = base_df.set_index('Type')['mem%'].to_dict()
    #base_memcap_dict = base_df.set_index('Type')['memcap'].to_dict()
    base_throughput_dict  = base_df.set_index('Type')[f'Exclusive{targetMPS}'].to_dict()
    base_CPU_dict = base_df.set_index('Type')['AvgCPU'].to_dict()
    base_MainMem_dict = base_df.set_index('Type')['AvgMem'].to_dict()


    #print(base_sm_dict)

    # Step 1: Read the first CSV file
    shared_throughput = pd.read_csv(shareThroughputData )
    if targetMPS != 100 :
        targetLS, targetBE = targetMPS, 100-targetMPS 
    else:
        targetLS, targetBE = 100,100
        
    # Step 2: Read the second CSV file
    kernel_file = pd.read_csv(kernelData)
    #only keep columns with selected feats
    #add w1_ and w2_ prefix to all columns in kernel_file
    selected_feats = ['w1_'+ col for col in selected_feats] + ['w2_'+ col for col in selected_feats]
    #add wrorkload1 and workload2 to selected_feats
    selected_feats += ['workload1', 'workload2', 'idx1', 'idx2'] 

    kernel_file = kernel_file[selected_feats]

    
    columns = shared_throughput.columns
    #get column pos if  targetMPS in column
    targetMPS_pos = [i for i, col in enumerate(columns) if f'LS{targetLS}, BE{targetBE}' in col][0]
    #print(targetMPS_pos)
    target_throughput = [eval(y) for y in shared_throughput.iloc[:, targetMPS_pos].to_list()]
    #get the entrue row of targetMPS_pos
    shared_throughput[[f'LS{targetLS}', f'BE{targetBE}']] = pd.DataFrame(target_throughput, index=shared_throughput.index)
    #print(shared_throughput[[f'LS{targetMPS}', f'BE{100-targetMPS}']])
    # Step 3: Extract the (LS100, BE100) pairs and create a DataFrame
    
    #shared_throughput = shared_throughput.drop(columns=[f'LS{targetMPS}, BE{100-targetMPS}'])

    # Step 4: Create a mapping for both regular and reversed (workload1, workload2) pairs
    mapping = {}
    for _, row in shared_throughput.iterrows():
        w1, w2 = row['workload1'], row['workload2']
        ls, be = row[f'LS{targetLS}'], row[ f'BE{targetBE}']
        mapping[(w1, w2)] = (ls, be)
        mapping[(w2, w1)] = (be, ls)

    # Step 5: Merge the DataFrames based on workload1 and workload2
    merged_rows = []
    no_data = []
    for _, row in kernel_file.iterrows():
        workload1 = row['workload1']
        workload2 = row['workload2']
        ls100_be100 = mapping.get((workload1, workload2), (None, None))
        if ls100_be100[0] is None or ls100_be100[1] is None:
            no_data.append((workload1, workload2))
            #continue  # Skip if any element of the tuple is None
        merged_row = row.tolist() + [ls100_be100[0], ls100_be100[1]]
        merged_rows.append(merged_row)

    # Step 6: Create a DataFrame from the merged rows
    merged_df = pd.DataFrame(merged_rows, columns=list(kernel_file.columns) + ['w1throughput', 'w2throughput'])
    #drop nan values
    merged_df.dropna(inplace=True)

    print(f"No MPS{targetMPS} throughput  for the following pairs: {no_data}")
    merged_df ['w1exclusive_throughput'] = merged_df['workload1'].map(base_throughput_dict)
    merged_df ['w2exclusive_throughput'] = merged_df['workload2'].map(base_throughput_dict)

    #apply sm and mem% to csv_data using base_sm_dict and base_mem_dict
    merged_df ['w1sm%'] = merged_df ['workload1'].map(base_sm_dict)
    merged_df ['w2sm%'] = merged_df ['workload2'].map(base_sm_dict)
    merged_df ['w1mem%'] = merged_df ['workload1'].map(base_mem_dict)
    merged_df ['w2mem%'] = merged_df ['workload2'].map(base_mem_dict)
    #merged_df ['w1memcap'] = merged_df ['workload1'].map(base_memcap_dict)
    #merged_df ['w2memcap'] = merged_df ['workload2'].map(base_memcap_dict)
    merged_df ['w1CPU%'] = merged_df ['workload1'].map(base_CPU_dict)
    merged_df ['w2CPU%'] = merged_df ['workload2'].map(base_CPU_dict)
    merged_df ['w1MainMem%'] = merged_df ['workload1'].map(base_MainMem_dict)
    merged_df ['w2MainMem%'] = merged_df ['workload2'].map(base_MainMem_dict)
    merged_df.dropna(inplace=True)

    #kernel_file.dropna(inplace=True)
    merged_df.to_csv(f"{outname}_targetMPS{targetMPS}.csv", index=False)
    #print noData
   
    print("stage1  training data saved to: ", f"{outname}_targetMPS{targetMPS}.csv")
    return outname
    





def train_test_split_with_counts(data,  train_outname, test_outname, n_workload_counts, random_seed, standard_split):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(data)
    if standard_split:
        #split randomly by selecting 20% as test, 80% as train
        train_set, testing_set = np.split(df.sample(frac=1, random_state=random_seed), [int(.8*len(df))])
        #create dir if not exist
        os.makedirs(os.path.dirname(train_outname), exist_ok=True)
        os.makedirs(os.path.dirname(test_outname), exist_ok=True)
        train_set.to_csv(train_outname, index=False)
        testing_set.to_csv(test_outname, index=False)
        

        print(f"standard training set save to {os.path.abspath(train_outname)}")
        print(f"standard testing set save to {os.path.abspath(test_outname)}")
        return train_outname, test_outname
    # Create a dictionary to count the appearances of each workload
    #shuffle df
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    workload_count = defaultdict(int)

    # Create a list to store the indices of rows to include in the training set
    training_indices = []

    # Iterate over each row in the DataFrame
    #ensure that each workload appears at most n_workload_counts times
    while (all(value < n_workload_counts for value in workload_count.values())):
        for idx, row in df.iterrows():
            workload1 = row['workload1']
            workload2 = row['workload2']
            if workload1 == workload2:
                continue
            # Check if adding this row will exceed the count of 2 for either workload
            if workload_count[workload1] < n_workload_counts or workload_count[workload2] < n_workload_counts:
                training_indices.append(idx)
                workload_count[workload1] += 1
                workload_count[workload2] += 1

    # Create the training set DataFrame
    training_set = df.loc[training_indices]
    testing_set = df.drop(training_indices)
    #get train_outname absoulte path
    #train_outname = os.path.abspath(train_outname)
    print(f"training set save to {os.path.abspath(train_outname)}")
    print(f"testing set save to {os.path.abspath(test_outname)}")
    # Save the training set to a new CSV file
    training_set.to_csv(train_outname, index=False)
    #create testing_set with the rest of the data

    # Save the testing set to a new CSV file
    testing_set.to_csv(test_outname, index=False)


    return  train_outname, test_outname


def getstage1trainData(baselineData, shareThroughputData , kernelData, outname, targetMPS):
    #combine baseline kernel and shared throughput as training target
    
    base_df = pd.read_csv(baselineData)
    #drop row with nan values
    base_df.dropna(inplace=True)
    #store sm in a dict wuth key as Type from base_df, base_df["sm%"] as value
    base_sm_dict = base_df.set_index('Type')['sm%'].to_dict()
    base_mem_dict = base_df.set_index('Type')['mem%'].to_dict()
    base_memcap_dict = base_df.set_index('Type')['memcap'].to_dict()
    base_throughput_dict  = base_df.set_index('Type')[f'Exclusive{targetMPS}'].to_dict()

    #print(base_sm_dict)

    # Step 1: Read the first CSV file
    shared_throughput = pd.read_csv(shareThroughputData )
    if targetMPS != 100 :
        targetLS, targetBE = targetMPS, 100-targetMPS 
    else:
        targetLS, targetBE = 100,100
        
    # Step 2: Read the second CSV file
    kernel_file = pd.read_csv(kernelData)
    columns = shared_throughput.columns
    #get column pos if  targetMPS in column
    targetMPS_pos = [i for i, col in enumerate(columns) if f'LS{targetLS}, BE{targetBE}' in col][0]
    #print(targetMPS_pos)
    target_throughput = [eval(y) for y in shared_throughput.iloc[:, targetMPS_pos].to_list()]
    #get the entrue row of targetMPS_pos
    shared_throughput[[f'LS{targetLS}', f'BE{targetBE}']] = pd.DataFrame(target_throughput, index=shared_throughput.index)
    #print(shared_throughput[[f'LS{targetMPS}', f'BE{100-targetMPS}']])
    # Step 3: Extract the (LS100, BE100) pairs and create a DataFrame
    
    #shared_throughput = shared_throughput.drop(columns=[f'LS{targetMPS}, BE{100-targetMPS}'])

    # Step 4: Create a mapping for both regular and reversed (workload1, workload2) pairs
    mapping = {}
    for _, row in shared_throughput.iterrows():
        w1, w2 = row['workload1'], row['workload2']
        ls, be = row[f'LS{targetLS}'], row[ f'BE{targetBE}']
        mapping[(w1, w2)] = (ls, be)
        mapping[(w2, w1)] = (be, ls)

    # Step 5: Merge the DataFrames based on workload1 and workload2
    merged_rows = []
    no_data = []
    for _, row in kernel_file.iterrows():
        workload1 = row['workload1']
        workload2 = row['workload2']
        ls100_be100 = mapping.get((workload1, workload2), (None, None))
        if ls100_be100[0] is None or ls100_be100[1] is None:
            no_data.append((workload1, workload2))
            #continue  # Skip if any element of the tuple is None
        merged_row = row.tolist() + [ls100_be100[0], ls100_be100[1]]
        merged_rows.append(merged_row)

    # Step 6: Create a DataFrame from the merged rows
    merged_df = pd.DataFrame(merged_rows, columns=list(kernel_file.columns) + ['w1throughput', 'w2throughput'])
    #drop nan values
    merged_df.dropna(inplace=True)

    print(f"No MPS{targetMPS} throughput  for the following pairs: {no_data}")
    merged_df ['w1exclusive_throughput'] = merged_df['workload1'].map(base_throughput_dict)
    merged_df ['w2exclusive_throughput'] = merged_df['workload2'].map(base_throughput_dict)

    #apply sm and mem% to csv_data using base_sm_dict and base_mem_dict
    merged_df ['w1sm%'] = merged_df ['workload1'].map(base_sm_dict)
    merged_df ['w2sm%'] = merged_df ['workload2'].map(base_sm_dict)
    merged_df ['w1mem%'] = merged_df ['workload1'].map(base_mem_dict)
    merged_df ['w2mem%'] = merged_df ['workload2'].map(base_mem_dict)
    merged_df ['w1memcap'] = merged_df ['workload1'].map(base_memcap_dict)
    merged_df ['w2memcap'] = merged_df ['workload2'].map(base_memcap_dict)
    merged_df.dropna(inplace=True)

    #kernel_file.dropna(inplace=True)
    merged_df.to_csv(f"{outname}_targetMPS{targetMPS}.csv", index=False)
    #print noData
   
    print("stage1  training data saved to: ", f"{outname}_targetMPS{targetMPS}.csv")
    return outname
    
def filter_data(data, workload,  custom_col_exclude, n_combination ,isbatchThroughput = False):
                         

    print("FILTERED workload: ", workload)
    # Filter data by workload name == workload1 or workload2
    #remove all workload that contains mobilenet_v2_1.0_224 and resnet-50 and mobilenet_
    #data = data[~data['workload1'].str.contains('mobilenet_v2_1.0_224') & ~data['workload2'].str.contains('mobilenet_v2_1.0_224')]
    #data = data[~data['workload1'].str.contains('resnet-50') & ~data['workload2'].str.contains('resnet-50')]
    #data = data[~data['workload1'].str.contains('mobilenet_') & ~data['workload2'].str.contains('mobilenet_')]

    #filter out data with batch size 8
    for i in range(1, n_combination+1):
        data[f'w{i}batch_size'] = data[f'workload{i}'].str.extract(r'_batch(\d+)', expand=False).astype(int)
        for col in custom_col_exclude:
            data = data.drop(columns=[f'w{i}{col}'])
    #data['w1batch_size'] = data['workload1'].str.extract(r'_batch(\d+)', expand=False).astype(int)
    #data['w2batch_size'] = data['workload2'].str.extract(r'_batch(\d+)', expand=False).astype(int)
    #data = data[(data['w1batch_size'] == 8) & (data['w2batch_size'] == 8)]

    #exclude data with both trains
    #data = data[~data['workload1'].str.contains('train') | ~data['workload2'].str.contains('train')]
    #data = data[~data['workload1'].str.contains('wav2vec2-base-960h') & ~data['workload2'].str.contains('wav2vec2-base-960h')]
    #filter data = data with workload1 == workload or workload2 == workload
    
    # Divide throughput and exclusive throughput by batch size
    if isbatchThroughput:
        for i in range(1, n_combination+1):
            data[f'w{i}throughput'] = data[f'w{i}throughput'] / data[f'w{i}batch_size']
            data[f'w{i}exclusive_throughput'] = data[f'w{i}exclusive_throughput'] / data[f'w{i}batch_size']

    #print(f"cols custom to be excluded{custom_col_exclude}")
    #print(f"prefilter data columns{data.columns}")
    #if custom_col_exclude == []:
    #    raise ValueError("should include something")
    if workload != "":
        condition = pd.Series([False] * len(df))
        for i in range(1, n_combination+1):
            condition = condition | df[f'workload{i}'].str.contains(workload)
        data = data[condition]

    for i in range(1, n_combination+1):
        data = data.drop(columns=[f'w{i}batch_size'])
    if workload == "":
        #drop workload1 == mobile-inf_batch64  workload2 == mobile-inf_batch64
        return data
    #concat data with workload1 and workload2 that contains workload
    #print(data[data['workload1'].str.contains(workload)])
    #data = data[(data['workload1'].str.contains(workload) ) | (data['workload2'].str.contains(workload) )]
    
    #df = df1.append(data[data['workload2'].str.contains(workload)])
    #drop batch size columns
    
    

    return data

def preprocess_data(data, categorical_columns, target, correlation, n_combination):
        #Drop non-numerical and non-relevant columns
        #drop categorical columns
        logging.info(f"preprocess columns: {data.columns.tolist()}")
        if target != "threadclass":
            data[target] = 0 

        columns_excluded = categorical_columns 
            
        for i in range(1, n_combination+1):
            if target == 'L2norm':
                
                data[target] += np.sqrt((data[f'w{i}throughput'] / data[f'w{i}exclusive_throughput'])**2)
                #data[target] = np.sqrt((data['w1throughput'] / data['w1exclusive_throughput'])**2 + (data['w2throughput'] / data['w2exclusive_throughput'])**2)
            elif target == 'sum_throughput':
                data[target] += data[f'w{i}throughput']
                columns_excluded +=  [f'w{i}throughput']
                #logging.info(data[f'w{i}throughput'])
                #data[target] = data['w1throughput'] + data['w2throughput'] 
            elif target == "threadclass":
                data[target] = data['Label']
                columns_excluded +=  ['Label']
                break# only iterate once
            elif target == "threadregression":
                data[target] = data['weight_Throughput_Sum']
                columns_excluded +=  ['weight_Throughput_Sum']
                #drop Power column
                data = data.drop(columns=['Power'])
                columns_excluded +=  ['Power']
                break
            else:
                logging.error("target not found")
                exit(1)
        #logging.info(f"target: {data[target]}")
        #data = data.drop(columns=categorical_columns)
        #get a set of feature colunmn names after removing w1 prefix
        feat_names = [col[2:] for col in data.columns if 'w1' in col] 
        logging.info(f"all feature names should be used with target : {feat_names}")
        #remove throughput  columen since they were used to calculate L2norm
        
        if target != "threadregression":
            for feat in feat_names:
                #add a new column with the name feat
                if f'w1{feat}' in  columns_excluded:
                    print("excluded", f'w1{feat}')
                    continue 
                data[feat] = 0
                for i in range(1, n_combination+1):
                    #print(f"adding w{i}{feat} = \n{data[f'w{i}{feat}']}")
                    data[feat] += data[f'w{i}{feat}']
                    columns_excluded.append(f'w{i}{feat}')
                #print(f"aggreate {feat} = \n{data[feat]}")
                if '%' in feat:
                    data[feat] = data[feat] / n_combination
                #data[feat] = (data[f'w1{feat}'] + data[f'w2{feat}'])/2  if '%' in feat else data[f'w1{feat}'] + data[f'w2{feat}']
                    
                #columns_excluded.append(f'w1{feat}')
            #columns_excluded.append(f'w2{feat}')
       
        logging.info(f"columns excluded: {columns_excluded}")
        #columns_excluded = categorical_columns + ['w1throughput', 'w2throughput', 'w1exclusive_throughput', 'w2exclusive_throughput', 'w1sm%', 'w2sm%', 'w1mem%', 'w2mem%']
        #columns_excluded = categorical_columns 
        least_correlated_columns = get_least_corr_columns(data=data, target_column=target, excluded_columns=columns_excluded, CORRELATION=correlation)
        #data = data.drop(least_correlated_columns, axis=1)  # Drop least correlated columns
        drop_columns = columns_excluded
        return data, drop_columns 
    


def get_least_corr_columns(data, target_column, excluded_columns, CORRELATION):
        
    #drop categorical columns or not using columns
    corr_data = data.drop(excluded_columns, axis=1)

    #z scale data
    corr_data = (corr_data - corr_data.mean()) / corr_data.std()  
    correlations = corr_data.corr()
    logging.info(f"correlation matrix: {correlations}")
    least_correlated_columns = list(correlations[target_column][abs(correlations[target_column]) < CORRELATION].index)
    logging.info(correlations[target_column].sort_values(ascending=False))
    logging.info(f"drop columns {least_correlated_columns}")
    logging.info(f"using columns {list(correlations[target_column][abs(correlations[target_column]) >= CORRELATION].index)}")
    return least_correlated_columns


def train_test_custom_split(data, test_workload, target, RANDOMSEED, CORRELATION, n_combination):
    test_workloads = pd.DataFrame()

    #get categorical columns
    categorical_columns = []
    for i in range(1, n_combination+1):
        categorical_columns += [f'workload{i}', f'idx{i}']
    
    data, columns_excluded = preprocess_data(data, categorical_columns, target, correlation=CORRELATION, n_combination=n_combination)
    print(f"columns excluded {columns_excluded}")
    #split data with workload1 == berttrain of workload2 == berttrain

    if test_workload != "":

        condition = pd.Series([False] * len(data))
        for i in range(1, n_combination+1):
            condition = condition | data[f'workload{i}'].str.startswith(test_workload)
        test_set = data[condition]
        #test_set = data[(data['workload1'].str.startswith(test_workload) ) | (data['workload2'].str.startswith(test_workload) )]
        
        #set train_set to data excluding test_set
        train_set = data[~data.index.isin(test_set.index)]
    #print(train_set['workload1'], train_set['workload2'])
    #print(len(train_set))

        # Drop non-numerical and non-relevant columns
        train_set = train_set.drop(columns=columns_excluded)
        #add test_workloads
        for i in range(1, n_combination+1):
            test_workloads[f'workload{i}'] = test_set[f'workload{i}']

        test_set = test_set.drop(columns=columns_excluded)
        #normalize all data in train_set and test_set
        #do log normalization on train_set and test_set
        #train_set = np.log1p(train_set)
        #test_set = np.log1p(test_set)
        
        
        train_set = (train_set - train_set.mean()) / train_set.std()
        test_set = (test_set - test_set.mean()) / test_set.std()
        #set train and test
        X_train , X_test = train_set.drop(target, axis=1), test_set.drop(target, axis=1)
        y_train, y_test = train_set[target], test_set[target]
    else:
        #standard train test split
        from sklearn.model_selection import train_test_split
        #data = data[~data['workload1'].str.contains('mobilenet_v2_1.0_224') & ~data['workload2'].str.contains('mobilenet_v2_1.0_224')]
        #data = data[~data['workload1'].str.contains('resnet-50') & ~data['workload2'].str.contains('resnet-50')]
        #data = data[~data['workload1'].str.contains('mobilenet_') & ~data['workload2'].str.contains('mobilenet_')]
        
        X = data.drop(columns=[target])
        y = data[target]
        
        #X_train, X_test, y_train, y_test = train_test_split_with_counts(data, workload_counts=3, target=target)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=RANDOMSEED)
        for i in range(1, n_combination+1):
            test_workloads[f'workload{i}'] = X_test[f'workload{i}']
        #test_workloads["workload1"] = X_test['workload1'] 
        #test_workloads["workload2"] = X_test['workload2']
        #drop non relevant columns
        X_train = X_train.drop(columns=columns_excluded)
        X_test = X_test.drop(columns=columns_excluded)
        #normalize all data in train_set and test_set
        X_train = (X_train - X_train.mean()) / X_train.std()
        X_test = (X_test - X_test.mean()) / X_test.std()
        if target != "threadclass":
            y_train = (y_train - y_train.mean()) / y_train.std()
            y_test = (y_test - y_test.mean()) / y_test.std()
        print("Xtest", X_test)
        print("y_test", y_test)
        print(f"X_test columns: {X_test.columns}")
    
    return X_train, X_test, y_train, y_test, test_workloads, columns_excluded

def import_notebook(nb_path):
    # Load the notebook content
    with open(nb_path) as f:
        nb_content = nbformat.read(f, as_version=4)
    
    # Convert notebook to a Python script
    exporter = PythonExporter()
    source, _ = exporter.from_notebook_node(nb_content)
    
    # Save the converted script
    script_path = nb_path.replace('.ipynb', '.py')
    with open(script_path, 'w') as f:
        f.write(source)
    
    # Import the script as a module
    spec = importlib.util.spec_from_file_location("notebook_module", script_path)
    notebook_module = importlib.util.module_from_spec(spec)
    sys.modules["notebook_module"] = notebook_module
    spec.loader.exec_module(notebook_module)
    
    # Clean up the temporary Python script
    os.remove(script_path)
    
    return notebook_module

def get_multiinstance_stage1trainData(baselineData, shareThroughputData , kernelData):
    #combine baseline kernel and shared throughput as training target
    
    

    #print(base_sm_dict)

    # Step 1: create target MPS keys based on combinations
    
    if targetMPS != 100 :
        raise ValueError("targetMPS that != 100 is not yet implemented!")
        #targetLS, targetBE = targetMPS, 100-targetMPS 
    else:
        target_MPS_list = [100] * n_combination
        for i, mps_value in enumerate(target_MPS_list):
            target_MPS_list[i] = f"w{i+1}_{mps_value}"
        target_MPS_key = ", ".join(target_MPS_list)
        #targetLS, targetBE = 100,100
        
    # Step 2: Read the second CSV file
    kernel_file = pd.read_csv(kernelData)
    if hotcloud:
        #keep only kernels that we use columns
        selected_feats = ["PCIe read bandwidth", "PCIe write bandwidth", "Long_Kernel",  "ave_Kernel_Length", "long/short_Ratio", "avg_Thread"]
        #add w1_ and w2_ prefix to all columns in kernel_file
        selected_feat_cols = []
        for i in range(1, n_combination+1):
            selected_feat_cols += [f'w{i}_'+ col for col in selected_feats]
            selected_feat_cols += [f'workload{i}']
            selected_feat_cols += [f'idx{i}']

        kernel_file = kernel_file[selected_feat_cols]

    #read shared throughput data
    shared_throughput = pd.read_csv(shareThroughputData)
    columns = shared_throughput.columns
    targetMPS_pos = [i for i, col in enumerate(columns) if target_MPS_key in col][0]

    print(targetMPS_pos)
    print(target_MPS_key)


   # target_throughput = defaultdict(tuple)
    #target_throughput = [eval(y) for y in shared_throughput.iloc[:, targetMPS_pos].to_list() if y != np.nan]
    #create a dict with (workloads) as keys, target_throughput as values
    #target_throughput = {row['workload1']: eval(row[target_MPS_key]) for _,row in shared_throughput.iterrows()}
    target_throughput_dict = defaultdict(list)
    for _, row in shared_throughput.iterrows():
        workload_instances = []
        #print( np.isnan(row.iloc[targetMPS_pos]))

        throughput_values =eval(row.iloc[targetMPS_pos]) # Assuming `row.iloc[targetMPS_pos]` returns the list of throughput values
        
        # Collect all workload instances
        for i in range(n_combination):
            workload_instances.append(row[f'workload{i+1}'])
        
        # Zip workload instances with throughput values
        workload_with_values = list(zip(workload_instances, throughput_values))
        # Sort the zipped pairs based on workload instances
        workload_with_values_sorted = sorted(workload_with_values, key=lambda x: x[0])  # Sort by workload instance (the first item in each tuple)
        # Unzip the sorted pairs back into two lists: sorted workloads and sorted throughput values
        sorted_workloads, sorted_values = zip(*workload_with_values_sorted)
        # Insert the sorted key-value pair into the target_throughput_dict
        target_throughput_dict[tuple(sorted_workloads)] = sorted_values
    #get the entrue row of targetMPS_pos
    print(len(target_throughput_dict))
    # Step 5: Merge the DataFrames based on workload1 and workload2
    merged_rows = []
    no_data = []
    print(kernel_file.columns)
    for idx, row in kernel_file.iterrows():
        
        workload_instances = []
            
        # Collect all workload instances from the row
        for i in range(n_combination):
            workload_instances.append(row[f'workload{i+1}'])
        
        if idx == 299:
            print(workload_instances)
        # Sort the tuple to ensure the order doesn't matter for lookup
        sorted_workload_instances = tuple(sorted(workload_instances))
        if idx == 299:
            print(sorted_workload_instances)
        
        # None tuple is default if not getting mappings
        None_tuple = tuple([None] * n_combination)
        
        # Check the target throughput dictionary for the sorted tuple
        throughput_of_instances = target_throughput_dict.get(sorted_workload_instances, None_tuple)
        
        
        # If we found a throughput, rematch the order of throughput with original (unsorted) workload instances
        if None in throughput_of_instances:
            no_data.append(tuple(workload_instances))
            #print("no data in shared_throughput for the following pairs: ", tuple(workload_instances))

        else:
            # Map sorted workloads to the throughput values using list (not dict)
            
            sorted_pairs = list(zip(sorted_workload_instances, throughput_of_instances))
            # Create a list to hold the reordered throughput values
            reordered_throughput = []
            
            # Reorder the throughput according to the original order of workload_instances
            for workload in workload_instances:
                for sorted_workload, throughput in sorted_pairs:
                    if sorted_workload == workload:
                        reordered_throughput.append(throughput)
                        sorted_pairs.remove((sorted_workload, throughput))  # Avoid double-counting
                        break
            
            throughput_of_instances = tuple(reordered_throughput)
            #if "vit_h_14_batch2-train"in workload_instances[2]  and "albert-base-v2_batch2-train" in workload_instances[0] and "albert-base-v2_batch2-train" in workload_instances[1]:
            #    print(throughput_of_instances)
        
            #continue  # Skip if any element of the tuple is None
            merged_row = row.tolist() + list(throughput_of_instances)
            merged_rows.append(merged_row)
    print(f"No MPS{targetMPS} throughput  for  {len(no_data)} colocations")

    # Step 6: Create a DataFrame from the merged rows
    throughput_cols = [f"w{i+1}throughput" for i in range(n_combination)]
    merged_df = pd.DataFrame(merged_rows, columns=list(kernel_file.columns) + throughput_cols)

    #drop nan values
    #merged_df.dropna(inplace=True)

    #print(f"No MPS{targetMPS} throughput  for the following pairs: {no_data}")
    #Step 7: apply exclusive throughput, sm and mem% to csv_data using base_sm_dict and base_mem_dict
    
    
    #drop row with nan values
    #base_df.dropna(inplace=True)
    #store sm in a dict wuth key as Type from base_df, base_df["sm%"] as value
    if baselineData != "":
        print("baseline data is not empty")
        base_df = pd.read_csv(baselineData)
        if not hotcloud:
            base_sm_dict = base_df.set_index('Type')['sm%'].to_dict()
            base_mem_dict = base_df.set_index('Type')['mem%'].to_dict()
            base_memcap_dict = base_df.set_index('Type')['memcap'].to_dict()
            base_throughput_dict  = base_df.set_index('Type')[f'Exclusive{targetMPS}'].to_dict()
            
            for i in range(n_combination): 
                merged_df[f'w{i+1}exclusive_throughput'] = merged_df[f'workload{i+1}'].map(base_throughput_dict)
                merged_df[f'w{i+1}sm%'] = merged_df[f'workload{i+1}'].map(base_sm_dict)
                merged_df[f'w{i+1}mem%'] = merged_df[f'workload{i+1}'].map(base_mem_dict)
                merged_df[f'w{i+1}memcap'] = merged_df[f'workload{i+1}'].map(base_memcap_dict)
            merged_df.dropna(inplace=True)
            print(merged_df)

        else:
            base_sm_dict = base_df.set_index('Type')['sm%'].to_dict()
            base_mem_dict = base_df.set_index('Type')['mem%'].to_dict()
            #base_memcap_dict = base_df.set_index('Type')['memcap'].to_dict()
            base_throughput_dict  = base_df.set_index('Type')[f'Exclusive{targetMPS}'].to_dict()
            base_CPU_dict = base_df.set_index('Type')['AvgCPU'].to_dict()
            base_MainMem_dict = base_df.set_index('Type')['AvgMem'].to_dict()
            #keep only kernels that we use columns
            #["PCIe read bandwidth", "PCIe write bandwidth", "Long_Kernel",  "ave_Kernel_Length", "long/short_Ratio", "avg_Thread"]
            
            for i in range(n_combination): 
                #merged_df[f'w{i+1}exclusive_throughput'] = merged_df[f'workload{i+1}'].map(base_throughput_dict)
                merged_df[f'w{i+1}sm%'] = merged_df[f'workload{i+1}'].map(base_sm_dict)
                merged_df[f'w{i+1}mem%'] = merged_df[f'workload{i+1}'].map(base_mem_dict)
                merged_df[f'w{i+1}CPU%'] = merged_df[f'workload{i+1}'].map(base_CPU_dict)
                merged_df[f'w{i+1}MainMem%'] = merged_df[f'workload{i+1}'].map(base_MainMem_dict)

            
            merged_df.dropna(inplace=True)

    import os# Get the parent directory from the output file path
    parent_dir = os.path.dirname(outname)
    # Create parent directory if it doesn't exist
    os.makedirs(parent_dir, exist_ok=True)
    #kernel_file.dropna(inplace=True)
    merged_df.to_csv(f"{outname}_targetMPS{targetMPS}.csv", index=False)
    #print noData
   
    print("stage1  training data saved to: ", f"{outname}_targetMPS{targetMPS}.csv")
    return outname


class Dataloader:
    def __init__(self, args, baselineData, kernelData, shareThroughputData, sharePowerData, shareDurationData, shareEnergyData):
        self.args = args
        self.baselineData = baselineData
        self.kernelData = kernelData
        self.shareThroughputData = shareThroughputData
        self.sharePowerData = sharePowerData
        self.shareDurationData = shareDurationData
        self.shareEnergyData = shareEnergyData
        self.weight_dict = defaultdict(tuple)
        self.power_limit_dict = defaultdict(float)
        
        self.throughput_power_dict = defaultdict(lambda: defaultdict(tuple))
        self.policy = args.label_policy

        # Initialize a logger for this class/module
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Dataloader initialized with debug logging enabled.")
        

    '''
    Preprocess the data including aggregation and dropping irrelevant columns
    '''
    def preprocess_data(self, data, target, correlation, n_combination):
        """Preprocess data by handling targets, aggregating features, and dropping irrelevant columns."""
        logging.info(f"Preprocess columns: {data.columns.tolist()}")
        logging.info(f"process policy: {self.policy}")

        excluded_columns = self.handle_target_column(data, target, n_combination)

        # Aggregate features
        self.aggregate_feature(data, n_combination, excluded_columns, self.policy)

        # Identify least correlated columns
        #least_correlated_columns = self.get_least_corr_columns(data, target, excluded_columns, correlation)
        #excluded_columns += least_correlated_columns
        #special handling for threadclass
        #if target == "threadclass":
        #    excluded_columns += ["_Exclusive10", "_Exclusive20"]
        self.logger.info(f"Excluded columns: {excluded_columns}")
        return data, excluded_columns

    def handle_target_column(self, data, target, n_combination):
        """Handle the target column based on the specified target."""
        excluded_columns = []
        if target != "threadclass" or target != "maxthroughput-powercap":
            data[target] = 0
        for i in range(1, n_combination + 1):
            excluded_columns += [f'workload{i}', f'idx{i}']
        for i in range(1, n_combination + 1):
            if target == "L2norm":
                data[target] += np.sqrt((data[f'w{i}throughput'] / data[f'w{i}exclusive_throughput']) ** 2)
            elif target == "sum_throughput":
                data[target] += data[f'w{i}throughput']
            elif target == "threadclass":
                data[target] = data["Label"]
                excluded_columns += ["Label"]
                break
            elif target == "separate_throughputpower_regression":
                data[target] = data["weight_Throughput_Sum"]
                excluded_columns += ["weight_Throughput_Sum"]
                #need to remove this power
                excluded_columns += [f"w{i}_power_Exclusive" for i in range(1, n_combination + 1)]
                data.drop(columns=["Power"], inplace=True)
                #excluded_columns += ["Power"]
                break
            elif target == "power_regression":
                data[target] = data["Power"]
                nan_mask = data[target].isna()
                if nan_mask.any():
                    nan_rows = data[nan_mask][['workload1', 'workload2', 'Power']].to_string()
                    raise ValueError(
                        f"{nan_mask.sum()} rows have NaN in Power column:\n{nan_rows}"
                    )
                excluded_columns += ["Power"]
                data.drop(columns=["weight_Throughput_Sum"], inplace=True)
                excluded_columns += [f'w{i}_Exclusive' for i in range(1, n_combination + 1)]
                break

            else:
                raise ValueError("Target not found.")

            

        return excluded_columns

    @staticmethod
    def sum_up_workloadfeatures(data, feat_names, n_combination, excluded_columns):
        
        for feat in feat_names:
            if f"w1{feat}" in excluded_columns:
                return

            data[feat] = sum(data[f"w{i}{feat}"] for i in range(1, n_combination + 1))

            if "%" in feat:
                data[feat] /= n_combination

            excluded_columns += [f"w{i}{feat}" for i in range(1, n_combination + 1)]

    def aggregate_feature(self, data, n_combination, excluded_columns, policy):
        """Aggregate features by summing values across workloads."""
       
        if policy == "maxthroughput-powercap":
            #raise error if n_combination is not 2
            if n_combination != 2:
                raise ValueError("maxthroughput-powercap policy is only implemented for 2 workloads for now")
            
            feat_names = list(set(data.columns.tolist()) - set(excluded_columns))
            #get weight for throughput
            baseline_df = pd.read_csv(self.baselineData)
            baseline_throughput = baseline_df.set_index('Type')[f'Exclusive100'].to_dict()
            #self.weight_dict = baseline_throughput
            #baseline_power = baseline_df.set_index('Type')['power_Exclusive100'].to_dict()
            for _, row in data.iterrows():
                #get workload tuple
                workloads = tuple(row[f'workload{i}'] for i in range(1, n_combination + 1))
                self.weight_dict[workloads] = self.get_weight(baseline_throughput, workloads)
               
            
            # Calculate weighted throughput sums
            for _, row in data.iterrows():
                 # Get the workload pair
                workloads = tuple(row[f'workload{i}'] for i in range(1, n_combination + 1))

                # Calculate the weighted throughput for the current row
                weight = self.weight_dict.get(workloads, None)  # Default to 1 if not found
                if weight == None:
                    raise ValueError(f"Weight not found for workloads: {workloads}")
                logging.debug(f"workloads: {workloads}, weight: {weight}")
                for j in range(10, 100, 10):  # We exclude '100' here as requested
                    # Construct the column names based on the percentage
                    col_w1 = f'w1_Exclusive{j}'
                    col_w2 = f'w2_Exclusive{100 - j}'

                    if col_w1 in data.columns and col_w2 in data.columns:
                        # Iterate over the rows and calculate the weighted throughput sum
                        if  col_w1 not in excluded_columns and col_w2 not in excluded_columns:
                            excluded_columns += [col_w1, col_w2]
                        
                        if row[col_w1] is None or row[col_w2] is None:
                            continue
                       
                        weighted_throughput = (row[col_w1] / weight) + row[col_w2]

                        # Update the column with the calculated value
                        data.loc[_, f'weighted_Exclusive_throughput_{j}_{100-j}'] = weighted_throughput

                        #exclude columns col_w1 and col_w2
                        
            #deal with rest of features that are not in excluded columns
            #exclude features caontaining Exclusive
            rest_features = [feat for feat in list(set(data.columns.tolist()) - set(excluded_columns)) if "Exclusive" not in feat]
            feat_names = [feat[2:] for feat in rest_features if "w1" in feat]
            self.sum_up_workloadfeatures(data, feat_names, n_combination, excluded_columns)
            logging.info(f"data.columns={data.columns}")
            #save data to csv
            if self.args.debug:
                data.to_csv(f"{self.args.output_dir}/aggregated_train.csv", index=False)
            
            #special handing for 100 column
            excluded_columns += [f'w1_Exclusive100', f'w2_Exclusive100']
        elif policy == "separate_throughputpower_regression":
            """
            custom_include = [
                policy,
                'w1_Exclusive',
                'w2_Threadpercent',
                'w1_sensitivity',
                'w1_FP32A%',
                'w1_SMOCC%',
                'w1_PCIRX',
                'w1_SMACT%',
                'w1_power_Exclusive',
                'w1_DRAMA%',
                'w2_Exclusive',
                'w1_Threadpercent',
                'w1_mem%',
                'w2_SMACT%',
                'w1_sm%',
                'w2_SMOCC%',
                'w1_memcap',
                'w1_PCITX',
                'w2_FP32A%',
                'w2_DRAMA%',
                'w2_mem%'
            ]
            excluded_feats = list(set(data.columns) - set(custom_include))
            self.logger.debug(f"Excluded features: {excluded_feats}")
            excluded_columns += excluded_feats
            """
            
            #raise ValueError("this section now disabled")
            excluded_feats = ["_power_Exclusive", "_Threadpercent", "_TENSO%", "_sm%", "_PCITX", "_PCIRX", "_memcap"]
            for i in range(1, n_combination+1):
                for feat in excluded_feats:
                   excluded_columns += [f"w{i}{feat}"]
            #excluded_columns += [f"w1_ExclusiveThroughput", f"w2_ExclusiveThroughput"]
            #excluded_columns += ["w1_sensitivity", "w2_sensitivity"]
            #excluded_columns += ["w1_Threadpercent", "w2_Threadpercent"]
            system_features = [feat for feat in list(set(data.columns.tolist()) - set(excluded_columns)) if "Exclusive" not in feat]
            feat_names = [feat[2:] for feat in system_features if "w1" in feat and "sensitivity" not in feat and "Threadpercent" not in feat]
            
            feat_names += ["_Exclusive"]
            

            self.sum_up_workloadfeatures(data, feat_names, n_combination, excluded_columns)
            
            self.logger.info(f"Data columns: {data.columns}")

        elif policy == "power_regression":
            excluded_feats = ["_power_Exclusive","_Exclusive", "_Threadpercent", "_TENSO%", "_sensitivity", "_sm%", "_mem%", "_memcap"]
            for i in range(1, n_combination+1):
                for feat in excluded_feats:
                   excluded_columns += [f"w{i}{feat}"]
            #excluded_columns += [f"w1_ExclusiveThroughput", f"w2_ExclusiveThroughput"]
            #excluded_columns += ["w1_sensitivity", "w2_sensitivity"]
            #excluded_columns += ["w1_Threadpercent", "w2_Threadpercent"]
            system_features = [feat for feat in list(set(data.columns.tolist()) - set(excluded_columns)) if "Exclusive" not in feat]
            feat_names = [feat[2:] for feat in system_features if "w1" in feat and "sensitivity" not in feat and "Threadpercent" not in feat]
            feat_names += ["_power_Exclusive"]

            self.sum_up_workloadfeatures(data, feat_names, n_combination, excluded_columns)
            self.logger.info(f"Data columns: {data.columns}")
            


        else:
            feat_names = [col[2:] for col in data.columns if "w1" in col]
            self.sum_up_workloadfeatures(data, feat_names, n_combination, excluded_columns)

            
    def _get_feature_correlation(self, X, y, target):
        # Combine X_train and y_train into a single DataFrame
        combined = pd.concat([X, y], axis=1)
        self.logger.debug(f"combined matrix: \n{combined}")
        #calculate correlation 
        correlations = combined.corr()[target]
        self.logger.info(f"Correlation matrix: {correlations.sort_values(ascending=False)}")
        return



    def _analyze_workload_sensitivity(self,baseline_path, policy):
        """
        Perform linear fitting on workload sensitivity data and save the results to a file.

        Args:
            file_path (str): Path to the input CSV file containing workload data.
            output_file (str): Path to the output CSV file to save the results.

        Returns:
            pd.DataFrame: DataFrame containing linear fitting results.
        """
        
        # Read the data
        data = pd.read_csv(baseline_path)
        if policy == "separate_throughputpower_regression":
            # Ensure required columns are present
            required_columns = ['Exclusive10', 'Exclusive20', 'Exclusive30', 'Exclusive40', 
                                'Exclusive50', 'Exclusive60', 'Exclusive70', 'Exclusive80', 'Exclusive90']
            if not all(col in data.columns for col in required_columns):
                raise ValueError("Input data is missing required columns.")
        elif policy == "power_regression":
            # Ensure required columns are present
            required_columns = ['Exclusive10', 'Exclusive20', 'Exclusive30', 'Exclusive40', 
                                'Exclusive50', 'Exclusive60', 'Exclusive70', 'Exclusive80', 'Exclusive90']
            required_columns = ["power_" + feat for feat in required_columns]
            
        else:
            raise ValueError(f"Policy {policy} not implemented.")
        if not all(col in data.columns for col in required_columns):
                raise ValueError("Input data is missing required columns.")

        # Linear fitting function
        def linear_fit(x, y):
            x = np.array(x, dtype=float)
            y = np.array(y, dtype=float)

            # Filter out invalid data
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]

            if len(x) < 2 or len(y) < 2:
                raise ValueError("Insufficient data for linear regression")

            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            return slope, intercept, r_value

        # Prepare for analysis
        thread_percentages = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
        sensitivity_results_linear = {}

        # Perform linear fitting for each workload
        for _, row in data.iterrows():
            workload = row['Type']
            exclusive_throughputs = row[
                required_columns
            ].values

            # Perform linear fitting
            try:
                slope, intercept, r_value = linear_fit(thread_percentages, exclusive_throughputs)
                sensitivity_results_linear[workload] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value ** 2
                }
            except ValueError as e:
                print(f"Skipping workload {workload} due to insufficient data: {e}")

        # Convert results to DataFrame
        sensitivity_df_linear = pd.DataFrame.from_dict(sensitivity_results_linear, orient='index')

        #save sensivity results to csv
        sensitivity_df_linear.to_csv(f"{self.args.output_dir}/sensitivity_results_linear_{policy}.csv")

        return sensitivity_results_linear

    def get_least_corr_columns(self, data, target_column, excluded_columns, correlation_threshold):
        """Identify columns least correlated with the target."""
        corr_data = data.drop(excluded_columns, axis=1)
        corr_data = (corr_data - corr_data.mean()) / corr_data.std()
        correlations = corr_data.corr()
        least_correlated = list(correlations[target_column][abs(correlations[target_column]) < correlation_threshold].index)
        self.logger.info(f"Correlation matrix: {correlations}")
        logging.info(f"Dropping columns: {least_correlated}")
        return least_correlated

    def aggregate_and_split_train_test(self, data, train_postprocess_data, test_workload, mode, target, correlation, n_combination):
        """Split data into train and test sets based on workload and target."""
        # Preprocess the data including aggregation and dropping irrelevant columns
        data_dir = str(os.path.dirname(data))
        post_process_data_dir = str(os.path.dirname(train_postprocess_data))

        data = pd.read_csv(data)
        #read traindata to get columns
        data, excluded_columns = self.preprocess_data(data, target, correlation, n_combination)
        self.logger.info(f"Excluded columns: {excluded_columns}")
        
        if mode == "train":
            if test_workload:
                # Split by test workload
                condition = pd.Series([False] * len(data))
                for i in range(1, n_combination + 1):
                    condition |= data[f"workload{i}"].str.startswith(test_workload)
                test_set = data[condition]
                train_set = data[~condition]

                # Drop excluded columns
                train_set = train_set.drop(columns=excluded_columns)
                test_set = test_set.drop(columns=excluded_columns)
                train_set = (train_set - train_set.mean()) / train_set.std()
                test_set = (test_set - test_set.mean()) / test_set.std()
            else:
                # Standard train-test split
                from sklearn.model_selection import train_test_split
                X = data.drop(columns=[target])
                y = data[target]
                X.drop(columns=excluded_columns, inplace=True)
                self._get_feature_correlation(X,y, target)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=self.args.split_randomseed)
                
                
                ## Drop excluded columns
                #X_train  = X_train.drop(columns=excluded_columns)
                #X_test = X_test.drop(columns=excluded_columns)
                #check correlation between X_train and y_train

                #X_train = (X_train - X_train.mean()) / X_train.std()
                #normalize X_test with X_train mean and std
                #X_test = (X_test - X_train.mean()) / X_train.std()
                if target == "power_regression" or target == "separate_throughputpower_regression":
                    #adopt min max scalar
                    scalar = MinMaxScaler()
                    scalar.fit(X_train)
                    X_train = pd.DataFrame(scalar.transform(X_train), columns=X_train.columns) 
                    X_test = pd.DataFrame(scalar.transform(X_test), columns=X_test.columns)
                    #save scalar to file
                    joblib.dump(scalar, f"{data_dir}/minmax_scalar_{target}.joblib")
                    #y_train = (y_train - y_train.mean()) / y_train.std()
                    #y_test = (y_test - y_train.mean()) / y_train.std()
                elif target == "threadclass" :
                    # Normalize data
                    y_train = (y_train - y_train.mean()) / y_train.std()
                    y_test = (y_test - y_test.mean()) / y_test.std()

                #numpy to csv
                #get data directory where data is located
                    
                X_train.to_csv(f"{data_dir}/X_train_postprocess_{target}.csv", index=False)
                y_train.to_csv(f"{data_dir}/y_train_postprocess_{target}.csv", index=False)
        if mode == "test":
            logging.info("testing. do not split data")
            if test_workload:
                self.logger.info(f"filter test workload: {test_workload}")
                #use n_combination to match workload[i] by condition to filter out test workload rows
                condition = pd.Series([True] * len(data))
                for i in range(1, n_combination + 1):
                    condition &= data[f"workload{i}"].str.startswith(test_workload[i-1])
                data = data[condition]
                self.logger.debug(f"test data: {data}")
                
            
            X_test = data.drop(columns=[target])
            y_test = data[target]
            test_mean, test_std = y_test.mean(), y_test.std()
            X_test = X_test.drop(columns=excluded_columns)
            self.logger.info(f"X_test columns: {X_test.columns}")
            self._get_feature_correlation(X_test,y_test, target)
            self.logger.debug(f"X_test:\n{X_test}")
            self.logger.debug(f"y_test: \n{y_test}")
            


            if target == "power_regression" or target == "separate_throughputpower_regression":
                
                #reorder X_test with X_train columns
                X_train = pd.read_csv(train_postprocess_data)
                X_train_columns = X_train.columns

                X_test = X_test[X_train_columns]
                #save X_test
                if self.args.debug:
                    X_test.to_csv(f"{self.args.output_dir}/X_test_postprocess_{target}_noscale.csv", index=False)
                #read scalar from file
                scalar = joblib.load(f"{post_process_data_dir}/minmax_scalar_{target}.joblib")
                X_test = pd.DataFrame(scalar.transform(X_test), columns=X_test.columns)
                
                #self.logger.debug(f"scalar loaded {scalar}")

                #load X_train from training data
                #X_test = (X_test - X_train.mean()) / X_train.std()
                #y_test = (y_test - y_train.mean()) / y_train.std()
                
            elif target == "threadclass":
                y_test = (y_test - y_test.mean()) / y_test.std()
                
            if self.args.debug:
                #create outputdir if not exist
                if not os.path.exists(self.args.output_dir):
                    os.makedirs(self.args.output_dir)
                
                if test_workload:
                    os.makedirs(f"{self.args.output_dir}/{test_workload}", exist_ok=True)
                    X_test.to_csv(f"{self.args.output_dir}/{test_workload}/X_test_postprocess_{target}.csv", index=False)
                    y_test.to_csv(f"{self.args.output_dir}/{test_workload}/y_test_postprocess_{target}.csv", index=False)
                else:
                    X_test.to_csv(f"{self.args.output_dir}/X_test_postprocess_{target}.csv", index=False)
                    y_test.to_csv(f"{self.args.output_dir}/y_test_postprocess_{target}.csv", index=False)


            return X_test, y_test, test_workload, excluded_columns, (test_mean, test_std)


        
      
        return X_train, X_test, y_train, y_test, test_workload, excluded_columns



    def get_colocate_throughput(self, throughput_data, workloads, weight, col):

        #get corresponding row in throughput data with corresponding workloads
        #get col in throughput data with corresponding col
        logging.debug(f"throughput data: {eval(throughput_data.filter(like=col).values[0])}")
        throughput_data = eval(throughput_data.filter(like=col).values[0])

        if all(value is not None for value in tuple(throughput_data)):
            return tuple(throughput_data)
        else:
            return tuple([None] * len(throughput_data))
       
    def get_colocate_power(self, power_data, workloads, col):
        #get corresponding row in power data with corresponding workloads
        condition = True
        logging.debug(f"processing power data for workloads: {workloads}")
        for i in range(1, self.args.n_combination + 1):
            condition &= (power_data[f'workload{i}'] == workloads[i-1])
        power_data = power_data[condition]
        logging.debug(f"power data: {power_data}")
        #find column position that contains col
        
        power_data = power_data.filter(like=col, axis=1).values[0][0]
        return power_data
    
    def get_throughput_power_per_workload(self, throughput_data,workloads, thread_columns, power_data, duration_data, energy_data):
        #get throughput and power for each workload by aggregating read power and throughput files
        #return throughput power dict - thread_column as key, throughput sum and power as values in tuple'
        #throughput_data - read throughput data
        #power_data - read power data
        #workloads - tuple of workloads
        for col in thread_columns:
            colocate_throughput = self.get_colocate_throughput(throughput_data, workloads, self.weight_dict[workloads], col)
            colocate_power = self.get_colocate_power(power_data, workloads, col)
            #self.logger.debug(f"durationdata: {duration_data}")
            if not duration_data.empty:
                colocate_duration = self.get_colocate_power(duration_data, workloads, col)
            if not energy_data.empty:
                colocate_energy = self.get_colocate_power(energy_data, workloads, col)
            self.throughput_power_dict[workloads][col] = (colocate_throughput, colocate_power, colocate_duration, colocate_energy)
        return

    def get_classes(self, thread_columns):
        string_to_int_col = {}
        thread_classes = {}
        workload1_classes = { 10: 1, 20: 1, 30: 2, 40: 2, 50: 3, 60: 3, 70: 4, 80: 4, 90: 5, 100:5}
        for thread_col in thread_columns:
            col_s = thread_col.replace("\"", "").replace("(", "").replace(")", "").replace(" ","").split(",")
            int_col_s = [int(c.split("_")[1]) for c in col_s]
            thread_classes[thread_col] = workload1_classes [int_col_s[0]]
        logging.info(f"thread_classes={thread_classes}")
        #raise ValueError("stop here")
        return thread_classes
            
            
            
        
        #
    def implementPolicy(self, throughput_power_dict, policy):
        if policy == "maxthroughput-powercap":
            labels_dict = defaultdict(tuple)
            
            #label class of thread column from self.thread_columns
            #self.thread_classes = {col: i + 1 for i, col in enumerate(self.thread_columns)}
            self.thread_classes =  self.get_classes(self.thread_columns)

            #get average power across all thread columns of each workload
            for workloads, throughput_power in throughput_power_dict.items():
                power_values = []
                throughput_sum_power_dict = defaultdict(tuple)
                logging.debug(f"current workload={workloads} workload weights: {self.weight_dict[workloads]}" )
                for col, (throughput, power) in throughput_power.items():
                    if any(value is None for value in throughput) or power is None:
                        continue
                    #exclude (w1_100, w2_100) col
                    if col.count("100") == self.args.n_combination: 
                        continue
                    else:
                        logging.debug(f"current col={col} throughput={throughput} power={power}")
                        power_values.append(power)
                        throughput_sum = throughput[0]/self.weight_dict[workloads] + sum(throughput[1:])
                        throughput_sum_power_dict[col] = (throughput_sum, power)
                #get average power across all thread columns of each workload 
                if len(power_values) == 0 or len(throughput_sum_power_dict) == 0:
                    continue
                average_power = sum(power_values) / len(power_values)
                #remove entry (100,100) from throughput_power_dict
                logging.debug(f"throughput_sum_power_dict: {throughput_sum_power_dict}")
                logging.debug(f"average power: {average_power}")
                #delete entry with power over average power
                max_throughput_under_powercap = -1
                max_thread_column = None
                for col, (throughput, power) in throughput_sum_power_dict.items():
                    if power <= average_power:
                        max_throughput_under_powercap = max(max_throughput_under_powercap, throughput)
                        max_thread_column = col
                #save labels_dict with the max throughput without 100, 100
                labels_dict[workloads] = (self.thread_classes[max_thread_column], max_thread_column, max_throughput_under_powercap) 
                #label rows
                label_rows = [f"Workload{i}" for i in range(1, self.args.n_combination + 1)] + ["Class","Configuration","Throughput"]
            return labels_dict, label_rows
        elif policy == "separate_throughputpower_regression" or policy == "power_regression":
            #get max throughput sum and power 
            labels_dict = defaultdict(lambda: defaultdict(float))
            #self.logger.debug(f"throughput_power_dict: {throughput_power_dict}")
            #exit(1)
            
            #get average power across all thread columns of each workload
            for workloads, throughput_power in throughput_power_dict.items():
                power_values = []
                throughput_sum_power_dict = defaultdict(tuple)
                self.logger.debug(f"current workload={workloads} workload weights={self.weight_dict[workloads]}" )
                
                for col, (throughput, power, duration, energy) in throughput_power.items():
                    if any(value is None for value in throughput) or power is None:
                        continue
                    #exclude (w1_100, w2_100) col
                    #if col.count("100") == self.args.n_combination:
                    #    self.logger.debug(f"process 100,100")
                        #continue
                    #else:
                    logging.debug(f"current col={col} throughput={throughput} power={power}")
                    power_values.append(power)
                    #throughput_sum = throughput[0]/self.weight_dict[workloads] + sum(throughput[1:])
                    self.logger.info(f"current workload weights: {self.weight_dict[workloads]}")
                    throughput_sum = sum([throughput[i]/self.weight_dict[workloads][i] for i in range(self.args.n_combination)])
                    self.logger.debug(f"throughput after weight: {[throughput[i]/self.weight_dict[workloads][i] for i in range(self.args.n_combination)]}")
                    key_workload_col = tuple(list(workloads) + [col])
                    labels_dict[key_workload_col] = {"weight_Throughput_sum": throughput_sum, "Power": power, "Duration": duration, "Energy": energy}
                        
                        
                #average_power = sum(power_values) / len(power_values)
                #remove entry (100,100) from throughput_power_dict
                self.logger.debug(f"labels_dict: {labels_dict}")
                label_rows = [f"Workload{i}" for i in range(1, self.args.n_combination + 1)] + ["Thread_combination"] + ["weight_Throughput_sum", "Power", "Duration", "Energy"]
                
                
            return labels_dict, label_rows
        elif policy == "freq-scale":
                        #get max throughput sum and power 
            labels_dict = defaultdict(lambda: defaultdict(float))
            #self.logger.debug(f"throughput_power_dict: {throughput_power_dict}")
            #exit(1)
            
            #get average power across all thread columns of each workload
            for workloads, throughput_power in throughput_power_dict.items():
                power_values = []
                throughput_sum_power_dict = defaultdict(tuple)
                self.logger.debug(f"current workload={workloads} workload weights={self.weight_dict[workloads]}" )
                
                for col, (throughput, power) in throughput_power.items():
                    if any(value is None for value in throughput) or power is None:
                        continue
                    #exclude (w1_100, w2_100) col
                    
                    logging.debug(f"current col={col} throughput={throughput} power={power}")
                    power_values.append(power)
                    #throughput_sum = throughput[0]/self.weight_dict[workloads] + sum(throughput[1:])
                    self.logger.info(f"current workload weights: {self.weight_dict[workloads]}")
                    throughput_sum = sum([throughput[i]/self.weight_dict[workloads][i] for i in range(self.args.n_combination)])
                    self.logger.debug(f"throughput after weight: {[throughput[i]/self.weight_dict[workloads][i] for i in range(self.args.n_combination)]}")
                    key_workload_col = tuple(list(workloads) + [col])
                    labels_dict[key_workload_col] = {"weight_Throughput_sum": throughput_sum, "Power": power}
                        
                        
                #average_power = sum(power_values) / len(power_values)
                #remove entry (100,100) from throughput_power_dict
                self.logger.debug(f"labels_dict: {labels_dict}")
                label_rows = [f"Workload{i}" for i in range(1, self.args.n_combination + 1)] + ["Thread_combination"] + ["weight_Throughput_sum", "Power"]
                
                
            return labels_dict, label_rows
            
            
            
        else:
            raise ValueError(f"policy {policy} is not yet implemented")
        
    def get_weight(self, base_throughput_dict, workloads):
        # Get Exclusive100 values from baseline data
        exclusive100_values = []
        for i in range(len(workloads)):
            exclusive100 = base_throughput_dict.get(workloads[i], None)
            if exclusive100 is None:
                raise ValueError(f"baseline data is missing for workload {workloads[i]}")
            exclusive100_values.append(exclusive100)
            logging.debug(f"exclusive100_w{i+1}: {exclusive100}")

        # Return tuple of weights (currently set to 1 for all workloads)
        weight = tuple([1] * len(workloads))

        # Store the weight in the weight_dict
        return weight

    
    def get_thread_num_from_str(self, thread_comb_str):
        self.logger.debug(f"thread_comb_str: {thread_comb_str}")
        col_s = thread_comb_str.replace("\"", "").replace("(", "").replace(")", "").replace(" ", "").split(",")
        col_s = [c.split("_")[1] for c in col_s]
        return col_s
    
    def reorder_labels(self, label_dict):
        #reordered_dict = defaultdict(tuple)
        reordered_dict = defaultdict(lambda: defaultdict(float))
        #self.logger.debug(self.thread_classes)
        # Sort the dictionary keys
        self.logger.debug(f"label_dict: {label_dict}")
        for i, key in enumerate(label_dict.keys()):
            value = label_dict[key]
            label_thread_comb = ""
            if self.policy == "separate_throughputpower_regression" or self.policy == "power_regression" or self.policy == "freq-scale":
                workloads = key[:-1]
                label_thread_comb = key[-1]
                sorted_workloads = tuple(sorted(list(workloads)))    
                #sorted_keys = tuple(list(sorted_workloads) + label_thread_comb) 
                

            else:
                workloads = key
                sorted_workloads = tuple(sorted(list(workloads)))  
                label_thread_comb = label_dict[key][1]  
                sorted_keys = sorted_workloads
                
            
            #logging.debug(f"sorted_keys: {sorted_keys}")
            
            
             
            # If the order of keys has changed, reverse the thread-class combinations
            if workloads != sorted_workloads:
                # Extract and modify thread-class combination
                #label_thread_comb = value[1]  # For example: '(w1_90, w2_10)'
                col_nums = self.get_thread_num_from_str(label_thread_comb)
                col_nums = col_nums[::-1]  # Reverse the order (w1_90 -> w2_10, w1_10 -> w2_90)
            
                # Rebuild the thread-class combination tuple
                col_nums = tuple([f"w{i+1}_{col_nums[i]}" for i in range(len(col_nums))])
                if self.policy in ["separate_throughputpower_regression","power_regression", "freq-scale" ]:
                    #append reordereddict with original value
                    # Dynamically build thread combination string for any number of workloads
                    thread_comb_str = f"({', '.join(col_nums)})"
                    sorted_keys = tuple(list(sorted_workloads) + [thread_comb_str])
                    reordered_dict[sorted_keys] = value
                elif self.policy == "maxthroughput-powercap":

                    # Rebuild the new value with possibly reversed order
                    # Dynamically build thread combination string for any number of workloads
                    thread_comb_str = f"({', '.join(col_nums)})"
                    new_value = (self.thread_classes[thread_comb_str], thread_comb_str, value[2])  # Recreate the value with reversed or original order

                    reordered_dict[sorted_keys] = new_value
                else:
                    #raise error
                    raise ValueError(f"policy {self.policy} is not yet implemented")
            # Store the reordered key-value pair
            else:
                reordered_dict[key] = value

        return reordered_dict
    
    """
    label the training data based on the policy
    policy: maxthroughput-powercap,...
    """
    def get_labels(self, policy):
        if policy == "maxthroughput-powercap" or policy == "separate_throughputpower_regression" or policy == "power_regression":
           
            # Step 1: Read the baseline data
            baseline_df = pd.read_csv(self.baselineData)
            base_throughput_dict = baseline_df.set_index('Type')[['Exclusive100']].to_dict()['Exclusive100']


            # Step 2: Read the shared throughput and power data
            shared_throughput = pd.read_csv(self.shareThroughputData)
            power_data = pd.read_csv(self.sharePowerData)
            if self.shareDurationData:
                shared_duration = pd.read_csv(self.shareDurationData)
            else:
                shared_duration = pd.DataFrame()
            if self.shareEnergyData:
                shared_energy = pd.read_csv(self.shareEnergyData)
            else:
                shared_energy = pd.DataFrame()
            #get thread columns
            self.thread_columns = get_thread_columns(shared_throughput)
            
            # Step 3: Calculate the weight for each pair of workloads based on throughput
            for _, row in shared_throughput.iterrows():
                
                workloads = []
                for i in range(1, self.args.n_combination + 1):
                    workloads.append(row[f'workload{i}'])
                workloads = tuple(workloads)
                logging.info(f"processing workloads: {workloads}")

                #get weight for workload combination
                self.weight_dict[workloads] = self.get_weight(base_throughput_dict, workloads)
                #get workload weight in reverse order
                self.weight_dict[tuple(reversed(workloads))] = self.get_weight(base_throughput_dict, tuple(reversed(workloads)))

                #get throughput and power for each workload
                self.get_throughput_power_per_workload(row, workloads, self.thread_columns, power_data, shared_duration, shared_energy)
            logging.info(self.throughput_power_dict)
            labels_dict, label_rows = self.implementPolicy(self.throughput_power_dict, self.policy)
            labels_dict = self.reorder_labels(labels_dict)
            
            self.logger.debug(f"labels_dict: {labels_dict}")
        elif policy == "freq-scale":
            # Step 1: Read the baseline data
            baseline_df = pd.read_csv(self.baselineData)
            base_throughput_dict = baseline_df.set_index('Type')[['Exclusive100']].to_dict()['Exclusive100']


            # Step 2: Read the shared throughput and power data
            shared_throughput = pd.read_csv(self.shareThroughputData)
            power_data = pd.read_csv(self.sharePowerData)
            #get thread columns
            self.thread_columns = get_thread_columns(shared_throughput)
            self.logger.debug(f"thread_columns: {self.thread_columns}")

            
            # Step 3: Calculate the weight for each pair of workloads based on throughput
            for _, row in shared_throughput.iterrows():
                workloads = []
                for i in range(1, self.args.n_combination + 1):
                    workloads.append(row[f'workload{i}'])
                workloads = tuple(workloads)
                logging.info(f"processing workloads: {workloads}")

                #get weight for workload combination
                self.weight_dict[workloads] = self.get_weight(base_throughput_dict, workloads)
                #get workload weight in reverse order
                self.weight_dict[tuple(reversed(workloads))] = self.get_weight(base_throughput_dict, tuple(reversed(workloads)))

                #get throughput and power for each workload
                self.get_throughput_power_per_workload(row, workloads, self.thread_columns, power_data)

            logging.info(self.throughput_power_dict)
            labels_dict, label_rows = self.implementPolicy(self.throughput_power_dict, self.policy)
            labels_dict = self.reorder_labels(labels_dict)
        else:
            raise ValueError(f"policy {policy} is not yet implemented")
        return labels_dict, label_rows

    def write_labels2csv(self, outname, label_rows):
        # Save defaultdict to CSV
       
        import os
        import csv
        # Get the parent directory from the output file path
        parent_dir = os.path.dirname(outname)
        # Create parent directory if it doesn't exist
        os.makedirs(parent_dir, exist_ok=True)
        csv_file = f"{outname}_labels.csv"
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(label_rows)

            # Write the rows 
            for key, value_dict in self.labels_dict.items():
                writer.writerow(list(key) + list(value_dict.values()))
        
        logging.info(f"Label data saved to {csv_file}")
        return
    def get_base_feat_names(self, baselineData):

        base_df = pd.read_csv(baselineData)
        feat_columns = []
        for feat in base_df.columns:
            if "workload" in feat or "Type" in feat or "freq" in feat:
                continue
            match = re.search(r'(\d+)$', feat)
            
            if match:
                trailing_number = match.group(1)
                self.logger.debug(f"trailing number: {trailing_number}")
                feat_columns.append(feat[:-len(str(trailing_number))])
            else: 
                raise ValueError(f"trailing number not found for {feat}")
        feat_columns = list(set(feat_columns))
        self.logger.info(f"feat_columns in baselinefile: {feat_columns}")
        return feat_columns
    # Function to strip trailing digits from dictionary keys
    @staticmethod
    def strip_trailing_digits(key):
        return re.sub(r'\d+$', '', key)  
    
    def merge_kernel_labels(self, kernel_file):
        """
        Refactored Step 3: Merge the kernel file with the labels based on workloads
        and reorder if necessary, according to the input policy.
        """

        
        merged_rows = []
        no_data = []
        if self.policy == "separate_throughputpower_regression" or self.policy == "power_regression":
             #init relevant columns = all columns - colname contains Exclusive
            base_feat_names = self.get_base_feat_names(self.baselineData)
            for key, value in self.labels_dict.items():
                workload_instances, thread_combination =  key[:-1], key[-1]
                throughput_sum = value.get('weight_Throughput_sum', None)
                power = value.get('Power', None)
                self.logger.debug(f"workload pairs: {workload_instances}, thread_combination: {thread_combination}, throughput_sum: {throughput_sum}, power: {power}")
                # Check if the workloads exist in the kernel file
               # Construct the condition dynamically for matching workloads
                condition = True  # Start with a True condition (meaning no exclusion)
                logging.debug(f"Processing power data for workloads: {workload_instances}")

                # Check the workload instances against each of the combinations
                for i in range(1, self.args.n_combination + 1):
                    condition &= (kernel_file[f'workload{i}'] == workload_instances[i-1])
                # Filter rows based on the condition
                matching_rows = kernel_file[condition]
                
                if matching_rows.empty:
                    self.logger.debug(f"No matching rows found for {workload_instances}")
                    no_data.append(workload_instances)
                    continue  # Skip if no matching rows found
                    #raise error if more than 1 row in matching rows
                if len(matching_rows) > 1:
                    self.logger.debug(f"matching rows: {matching_rows}")
                    raise ValueError(f"more than 1 workload instance found for {workload_instances}")
                matching_row = matching_rows.iloc[0]

                # Filter relevant Exclusive columns based on thread combination
               
                # relevant_columns = list(set(matching_rows.columns.tolist()) - set([col for col in matching_rows.columns if "Exclusive" in col]))
                # filtered_row = {col: matching_row[col] for col in relevant_columns if col in matching_row}
                filtered_row = {}
                #get category columns
                categories = ["workload", "idx"]
                #expand with combination columns
                category_columns = []
                for category in categories:
                    for i in range(1, self.args.n_combination + 1):
                        category_columns.append(f'{category}{i}')
                #update filter row with matching category columns
                filtered_row.update({col: matching_row[col] for col in category_columns if col in matching_row})
                exclusive_columns = []
                # exclusive_power_columns = []
                col_nums =  self.get_thread_num_from_str(thread_combination)
                for feat in base_feat_names:
                    for i, comb in enumerate(col_nums):
                        exclusive_columns.append(f"w{i+1}_{feat}{comb}")
                
                exclusive_dict = {col: matching_row[col] for col in exclusive_columns if col in matching_row}
                #strip percenetage num from keys
                stripped_dict = {self.strip_trailing_digits(k): v for k, v in exclusive_dict.items()}
                
                #Special handle Exclusive by dividing w{i}_Exclusive with self.weight
                for i in range(1, self.args.n_combination + 1):
                    stripped_dict[f"w{i}_Exclusive"] = stripped_dict[f"w{i}_Exclusive"] / self.weight_dict[workload_instances][i-1]
                

                filtered_row.update(stripped_dict)

                #update filtered power row
                #exclusive_power_row = {col: matching_row[col] for col in exclusive_power_columns if col in matching_row}
                #exclusive_power_row = {col.split("_")[0]+ "_power_Exclusive": value for col, value in exclusive_power_row.items() if "power_Exclusive" in col}
                #filtered_row.update(exclusive_power_row)
                
                # Add additional label information
                # Dynamically create thread percent fields for all workloads
                dynamic_fields = {}
                for i in range(1, self.args.n_combination + 1):
                    dynamic_fields[f'w{i}_Threadpercent'] = col_nums[i-1]

                # Add sensitivity fields for all workloads
                for i in range(1, self.args.n_combination + 1):
                    dynamic_fields[f'w{i}_sensitivity'] = self.workload_sensivity[workload_instances[i-1]]["slope"]

                dynamic_fields.update({
                    'weight_Throughput_Sum': throughput_sum,
                    'Power': power
                })
                filtered_row.update(dynamic_fields)
                self.logger.debug(f"filtered_row = {filtered_row}")
                #raise ValueError("stop here")
                
                # Append the processed row
                merged_rows.append(filtered_row)

            self.logger.info(f"No data for {len(no_data)} workload pairs: {no_data}")

            # Create a DataFrame from the merged rows
            merged_df = pd.DataFrame(merged_rows)

            return merged_df
            
        else:
        # Example: Print columns just for debug
            print(kernel_file.columns)

            for idx, row in kernel_file.iterrows():
                workload_instances = []
                # Collect all workload instances from the row
                for i in range(self.args.n_combination):
                    workload_instances.append(row[f'workload{i+1}'])

                # None tuple is default if not getting mappings
                None_tuple = tuple([None] * self.args.n_combination)

                # Check the target throughput dictionary for the sorted tuple
                # label is index0
                workload_label = self.labels_dict.get(tuple(workload_instances), None_tuple)
                if workload_label is not None_tuple:
                    # Found direct label
                    workload_label = [workload_label[0]]  # 0 is the class
                else:
                    # manage reordered tuple if policy is maxthroughput-powercap
                    if self.policy == "maxthroughput-powercap":
                        workload_label = self.labels_dict.get(tuple(sorted(workload_instances)), None_tuple)
                        if workload_label is not None_tuple:
                            logging.info(
                                f"need to reorder workload instances: {workload_instances} original label: {workload_label}"
                            )
                            label_thread_comb = workload_label[1]
                            col_s = label_thread_comb.replace("\"", "").replace("(", "").replace(")", "").replace(" ","").split(",")
                            col_s = [c.split("_")[1] for c in col_s]
                            # reorder the col_s (reverse, for instance)
                            col_s = col_s[::-1]
                            col_s = tuple([f"w{i+1}_{col_s[i]}" for i in range(len(col_s))])
                            # convert tuple
                            new_label_key = str(col_s).replace("'", "")
                            workload_label = [self.thread_classes[new_label_key]]
                            logging.info(f"reordered thread column: {col_s} label: {workload_label}")
                        else:
                            self.logger.debug("workload pair label snot found ")
                    else:
                        raise ValueError(f"policy {self.args.policy} sorted is not yet implemented")

                # If we found a label but it’s None, track the missing data
                if None in workload_label:
                    no_data.append(tuple(workload_instances))
                else:
                    merged_row = row.tolist() + workload_label
                    merged_rows.append(merged_row)

            print(f"No label for {len(no_data)} colocations")

            if self.policy == "maxthroughput-powercap":
                label_cols = ["Label"]
            else:
                raise ValueError(f"policy {self.args.policy} is not yet implemented")

            merged_df = pd.DataFrame(merged_rows, columns=list(kernel_file.columns) + label_cols)
            return merged_df
        
    def getstage2trainData(self, outname):
        #combine baseline kernel and shared throughput as training target
        #get policy label
        self.labels_dict, label_rows = self.get_labels(self.args.label_policy)
        import csv
        self.write_labels2csv(outname,  label_rows)
        #save labels_dict to csv
        # Save defaultdict to CSV
        self.logger.debug(f"labels_dict: {self.labels_dict}")
        self.logger.debug(f"thread column {self.thread_columns}")
     
        # Step 2: Read the kernel/baseline CSV file as input
        kernel_file = pd.read_csv(self.kernelData)
        if self.args.modeltype == "hotcloud":
            #keep only kernels that we use columns
            selected_feats = ["PCIe read bandwidth", "PCIe write bandwidth", "Long_Kernel",  "ave_Kernel_Length", "long/short_Ratio", "avg_Thread"]
            #add w1_ and w2_ prefix to all columns in kernel_file
            selected_feat_cols = []
            for i in range(1, self.args.n_combination+1):
                selected_feat_cols += [f'w{i}_'+ col for col in selected_feats]
                selected_feat_cols += [f'workload{i}']
                selected_feat_cols += [f'idx{i}']

            kernel_file = kernel_file[selected_feat_cols]
        #get workload sensitivity
        self.workload_sensivity = self._analyze_workload_sensitivity(self.baselineData, self.args.label_policy)
        self.logger.debug(f"workload sensitivity: {self.workload_sensivity}")

        # Step 3: Merge the DataFrames based on workload1 and workload2
        merged_df = self.merge_kernel_labels(kernel_file)
        merged_df.to_csv(f"{outname}.csv", index=False)
        
        #save args to csv
        with open(f"{outname}_args.txt", "w") as file:
            file.write(str(self.args))
        
        #drop nan values
        #merged_df.dropna(inplace=True)

        #print(f"No MPS{targetMPS} throughput  for the following pairs: {no_data}")
        #Step 7: apply exclusive throughput, sm and mem% to csv_data using base_sm_dict and base_mem_dict
        
        
        #drop row with nan values
        #base_df.dropna(inplace=True)
        #store sm in a dict wuth key as Type from base_df, base_df["sm%"] as value
        """
        if self.baselineData != "":
            print("baseline data is not empty")
            base_df = pd.read_csv(self.baselineData)
            if not self.args.hotcloud:
                base_sm_dict = base_df.set_index('Type')['sm%'].to_dict()
                base_mem_dict = base_df.set_index('Type')['mem%'].to_dict()
                base_memcap_dict = base_df.set_index('Type')['memcap'].to_dict()
                base_throughput_dict  = base_df.set_index('Type')[f'Exclusive{self.args.targetMPS}'].to_dict()
                
                for i in range(self.args.n_combination): 
                    merged_df[f'w{i+1}exclusive_throughput'] = merged_df[f'workload{i+1}'].map(base_throughput_dict)
                    merged_df[f'w{i+1}sm%'] = merged_df[f'workload{i+1}'].map(base_sm_dict)
                    merged_df[f'w{i+1}mem%'] = merged_df[f'workload{i+1}'].map(base_mem_dict)
                    merged_df[f'w{i+1}memcap'] = merged_df[f'workload{i+1}'].map(base_memcap_dict)
                merged_df.dropna(inplace=True)
                print(merged_df)

            else:
                base_sm_dict = base_df.set_index('Type')['sm%'].to_dict()
                base_mem_dict = base_df.set_index('Type')['mem%'].to_dict()
                #base_memcap_dict = base_df.set_index('Type')['memcap'].to_dict()
                base_throughput_dict  = base_df.set_index('Type')[f'Exclusive{self.args.targetMPS}'].to_dict()
                base_CPU_dict = base_df.set_index('Type')['AvgCPU'].to_dict()
                base_MainMem_dict = base_df.set_index('Type')['AvgMem'].to_dict()
                #keep only kernels that we use columns
                #["PCIe read bandwidth", "PCIe write bandwidth", "Long_Kernel",  "ave_Kernel_Length", "long/short_Ratio", "avg_Thread"]
                
                for i in range(self.args.n_combination): 
                    #merged_df[f'w{i+1}exclusive_throughput'] = merged_df[f'workload{i+1}'].map(base_throughput_dict)
                    merged_df[f'w{i+1}sm%'] = merged_df[f'workload{i+1}'].map(base_sm_dict)
                    merged_df[f'w{i+1}mem%'] = merged_df[f'workload{i+1}'].map(base_mem_dict)
                    merged_df[f'w{i+1}CPU%'] = merged_df[f'workload{i+1}'].map(base_CPU_dict)
                    merged_df[f'w{i+1}MainMem%'] = merged_df[f'workload{i+1}'].map(base_MainMem_dict)

                
                merged_df.dropna(inplace=True)
        """

        
        #kernel_file.dropna(inplace=True)
        
        #print noData
    
        print("stage2 training data saved to: ", f"{outname}.csv")
        return outname
def train_test_split_multiinstance(data,  train_outname, test_outname, random_seed, train_ratio):
    #split train test
    # Load the CSV file into a DataFrame
    df = pd.read_csv(data)
    # Create a dictionary to count the appearances of each workload
    #shuffle df
    training_set = df.sample(frac=train_ratio, random_state=random_seed)
    testing_set = df.drop(training_set.index)



    """
    workload_count = defaultdict(int)

    # Create a list to store the indices of rows to include in the training set
    training_indices = []

    # Iterate over each row in the DataFrame
    #ensure that each workload appears at most n_workload_counts times
    while (all(value < n_workload_counts for value in workload_count.values())):
        for idx, row in df.iterrows():
            workload1 = row['workload1']
            workload2 = row['workload2']
            if workload1 == workload2:
                continue
            # Check if adding this row will exceed the count of 2 for either workload
            if workload_count[workload1] < n_workload_counts or workload_count[workload2] < n_workload_counts:
                training_indices.append(idx)
                workload_count[workload1] += 1
                workload_count[workload2] += 1

    # Create the training set DataFrame
    training_set = df.loc[training_indices]
    testing_set = df.drop(training_indices)
    """
    #get train_outname absoulte path
    #train_outname = os.path.abspath(train_outname)
    print(f"training set save to {os.path.abspath(train_outname)}")
    print(f"testing set save to {os.path.abspath(test_outname)}")
    # Save the training set to a new CSV file
    training_set.to_csv(train_outname, index=False)
    #create testing_set with the rest of the data

    # Save the testing set to a new CSV file
    testing_set.to_csv(test_outname, index=False)


    return  train_outname, test_outname

import os
import pandas as pd
import numpy as np

def shuffle_split_and_save_csv_by_pairs(csv_path, seed, output_dir, num_pieces):
    """
    Reads a CSV from `csv_path`, groups by workload pairs (workload1, workload2),
    shuffles the pairs using `seed`, splits them into 10 folds, and saves each fold
    with all corresponding rows to fold_1.csv, fold_2.csv, ..., fold_10.csv.
    
    Parameters:
    -----------
    csv_path : str
        Path to the input CSV file.
    seed : int
        The random seed for deterministic shuffling.
    output_dir : str
        Directory where the fold CSV files will be saved.
    num_pieces : int, optional
        Number of folds to split into (default=10).
    """
    
    # 1. Check if all fold files exist in output_dir
    all_exist = True
    for i in range(1, num_pieces + 1):
        fold_file = os.path.join(output_dir, f"fold_{i}.csv")
        if not os.path.isfile(fold_file):
            all_exist = False
            break

    if all_exist:
        print("All fold files already exist. Skipping creation.")
        return
    print(f"start splitting original csv {csv_path} into {num_pieces} folds")
    # 2. Read the CSV into a DataFrame
    df = pd.read_csv(csv_path)
    
    # 3. Get unique workload pairs
    workload_pairs = df[['workload1', 'workload2']].drop_duplicates().reset_index(drop=True)
    
    # 4. Shuffle the pairs with a fixed random seed
    np.random.seed(seed)
    shuffled_pairs = workload_pairs.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # 5. Calculate the number of pairs per fold
    total_pairs = len(shuffled_pairs)
    pairs_per_fold = total_pairs // num_pieces
    remainder = total_pairs % num_pieces  # Extra pairs to distribute
    
    # 6. Assign pairs to folds
    fold_assignments = []
    start_idx = 0
    for i in range(num_pieces):
        # Add one extra pair to the first `remainder` folds to handle uneven division
        extra = 1 if i < remainder else 0
        end_idx = start_idx + pairs_per_fold + extra
        fold_pairs = shuffled_pairs.iloc[start_idx:end_idx]
        fold_assignments.append(fold_pairs)
        start_idx = end_idx
    
    # 7. Create folds by collecting all rows for each pair in the fold
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for fold_idx, fold_pairs in enumerate(fold_assignments, 1):
        # Merge the fold_pairs back with the original DataFrame to get all rows
        fold_df = df.merge(fold_pairs, on=['workload1', 'workload2'], how='inner')
        output_name = os.path.join(output_dir, f"fold_{fold_idx}.csv")
        fold_df.to_csv(output_name, index=False)
        print(f"Saved fold_{fold_idx}.csv with {len(fold_df)} rows and {len(fold_pairs)} unique pairs.")
    
    print(f"Shuffling and splitting complete. {num_pieces} folds saved as fold_1.csv ... fold_{num_pieces}.csv.")
    return


def train_test_split_multiinstance_with_all_threads(
    data,
    train_outname,
    test_outname,
    random_seed,
    train_ratio,
    rm_100partitions,
    n_combinations,
    is_cross_validation,
    test_folds,
    train_folds,
    read_workload_list=None,
):

    df = pd.read_csv(data)
    if rm_100partitions:
        print("remove 100,100 partitions...")
        #print(df['w1_Threadpercent'].dtype)
        #print(df['w2_Threadpercent'].dtype)
        #remove rows with w1_Threadpercent== 100 and w2_Threadpercent == 100
        df = df[~((df['w1_Threadpercent'] == 100) & (df['w2_Threadpercent'] == 100))]
        dirname = os.path.dirname(data)
        #df.to_csv(os.path.join(str(dirname), "debug.csv"), index=False)
        #print(f"save no 100 partitions to {os.path.join(str(dirname), 'debug.csv')}")
    workload_columns = [f"workload{i}" for i in range(1, n_combinations + 1)]
    # get uniquie workload comninations based on workload columns
    unique_workloads = df[workload_columns].drop_duplicates()
    #split workload columns by train ratio
    print(f"unique_workloads: \n{unique_workloads}")
    if is_cross_validation:
        # Ensure test_folds and train_folds are provided
        if not test_folds or not train_folds:
            raise ValueError("Both --test_folds and --train_folds must be provided for cross-validation")
        test_fold_nums = [int(f) for f in test_folds.split(',')]
        train_fold_nums = [int(f) for f in train_folds.split(',')]
        print(f"Cross-validation: Test folds = {test_fold_nums}, Train folds = {train_fold_nums}")

        if read_workload_list:
            workload_base_dir = os.path.abspath(read_workload_list)
            candidate = os.path.join(workload_base_dir, f"rand{random_seed}")
            if os.path.isdir(candidate):
                workload_base_dir = candidate
            if not os.path.isdir(workload_base_dir):
                raise FileNotFoundError(
                    f"Cannot locate workload folds under {workload_base_dir}"
                )

            def _load_pairs(path):
                assert os.path.isfile(path), f"Workload list {path} not found."
                pairs_df = pd.read_csv(path)
                missing_cols = [c for c in workload_columns if c not in pairs_df.columns]
                if missing_cols:
                    raise ValueError(
                        f"{path} is missing workload columns {missing_cols}"
                    )
                return pairs_df[workload_columns].drop_duplicates()

            if len(test_fold_nums) == 1:
                test_label = str(test_fold_nums[0])
            else:
                test_label = f"{test_fold_nums[0]}-{test_fold_nums[-1]}"
            aggregated_test_path = os.path.join(
                workload_base_dir,
                f"fold_{len(test_fold_nums)}_test_{test_label}.csv",
            )
            if os.path.isfile(aggregated_test_path):
                print(f"Reading aggregated test workloads from {aggregated_test_path}")
                test_pairs = _load_pairs(aggregated_test_path)
            else:
                test_pairs_list = [
                    _load_pairs(os.path.join(workload_base_dir, f"fold_{fold}.csv"))
                    for fold in test_fold_nums
                ]
                test_pairs = (
                    pd.concat(test_pairs_list, ignore_index=True).drop_duplicates()
                    if test_pairs_list
                    else pd.DataFrame(columns=workload_columns)
                )

            train_pairs_list = [
                _load_pairs(os.path.join(workload_base_dir, f"fold_{fold}.csv"))
                for fold in train_fold_nums
            ]
            train_pairs = (
                pd.concat(train_pairs_list, ignore_index=True).drop_duplicates()
                if train_pairs_list
                else pd.DataFrame(columns=workload_columns)
            )

            testing_set = df.merge(test_pairs, on=workload_columns, how="inner")
            training_set = df.merge(train_pairs, on=workload_columns, how="inner")

            missing_test = len(test_pairs) - len(testing_set[workload_columns].drop_duplicates())
            missing_train = len(train_pairs) - len(training_set[workload_columns].drop_duplicates())
            if missing_test:
                print(f"Warning: {missing_test} test workload combinations not present in {data}")
            if missing_train:
                print(f"Warning: {missing_train} train workload combinations not present in {data}")
        else:
            #check if fold_*.csv exists
            folds_dir = os.path.dirname(train_outname)
            #folds dir: move up one level
            train_dir = os.path.dirname(train_outname)
            folds_dir = os.path.abspath(os.path.join(train_dir, os.pardir))
            #check if folds exists
            shuffle_split_and_save_csv_by_pairs(data, random_seed, folds_dir, num_pieces=10)

            # Load and concatenate test folds
            test_dfs = []
            for fold_num in test_fold_nums:
                fold_file = os.path.join(folds_dir, f"fold_{fold_num}.csv")
                assert os.path.isfile(fold_file), f"Error: Fold file {fold_file} not found."
                try:
                    fold_df = pd.read_csv(fold_file)
                    test_dfs.append(fold_df)
                    print(f"Loaded test fold {fold_num} from {fold_file} (rows: {len(fold_df)})")
                except Exception as e:
                    assert False, f"Error reading fold file {fold_file}: {str(e)}"
                    return

            # Load and concatenate train folds
            train_dfs = []
            for fold_num in train_fold_nums:
                fold_file = os.path.join(folds_dir, f"fold_{fold_num}.csv")
                assert os.path.isfile(fold_file), f"Error: Fold file {fold_file} not found."
                try:
                    fold_df = pd.read_csv(fold_file)
                    train_dfs.append(fold_df)
                    print(f"Loaded train fold {fold_num} from {fold_file} (rows: {len(fold_df)})")
                except Exception as e:
                    assert False, f"Error reading fold file {fold_file}: {str(e)}"
                    return

            # Concatenate the DataFrames
            testing_set = pd.concat(test_dfs, ignore_index=True)
            training_set = pd.concat(train_dfs, ignore_index=True)

    else:
        # Standard train-test split based on train_ratio
        train_workloads = unique_workloads.sample(frac=train_ratio, random_state=random_seed)
        test_workloads = unique_workloads.drop(train_workloads.index)

    #print("Train workload combinations:")
    #print(train_workloads)
    #print("Test workload combinations:")
    #print(test_workloads)

    # 7) Merge with original df to get full rows for those workload combos
    #    (Inner join on [workload1, ..., workloadN])
        training_set = df.merge(train_workloads, on=workload_columns, how="inner")
        testing_set = df.merge(test_workloads, on=workload_columns, how="inner")

    # 8) Save to CSV
    training_set.to_csv(train_outname, index=False)
    testing_set.to_csv(test_outname, index=False)

    print(f"Training set saved to: {os.path.abspath(train_outname)}")
    print(f"Testing set saved to:  {os.path.abspath(test_outname)}")
    return train_outname, test_outname


if __name__ == "__main__":
    RANDOMSEED = 30
    CORRELATION = 0
    TARGET="sum_throughput"
    FILTERED_WORKLOAD = ''
    TEST_WORKLOAD = ''
    """
    filename = '/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/training/kernel_labels_L2norm.csv'  
    y_all_test = pd.DataFrame()

    df = pd.read_csv(filename)
    df = filter_data(df, FILTERED_WORKLOAD, isbatchThroughput=False)
    X_train, X_test, y_train, y_test, workloads = train_test_custom_split(df, test_workload=TEST_WORKLOAD, target=TARGET,  RANDOMSEED=RANDOMSEED, CORREATION=CORRELATION)
    print(workloads)
    """
    ##############################
    #MULTI INSTANCE
    ##############################
    """
    datafile = get_multiinstance_stage1trainData(baselineData="/home/cc/mlProfiler/tests/mps/analysis/baseline_labels.csv", 
                       shareThroughputData="/home/cc/mlProfiler/data/colocations/0730_share3_batch2_share_steps_stage2.csv",
                       kernelData="/home/cc/mlProfiler/data/experiment_inputs/colocations/kernel_labels_comb3_batches2.csv",
                       outname="/home/cc/mlProfiler/data/model_datasets/total_labels",
                       targetMPS=100,
                       n_combination=3)
    

    #split into train/test set
    train_test_split_multiinstance(data="/home/cc/mlProfiler/data/model_datasets/total_labels_targetMPS100.csv",
                                    train_outname="/home/cc/mlProfiler/data/model_datasets/training_set.csv",
                                    test_outname="/home/cc/mlProfiler/data/model_datasets/testing.csv",
                                    random_seed=30, train_ratio=0.5)
    """
