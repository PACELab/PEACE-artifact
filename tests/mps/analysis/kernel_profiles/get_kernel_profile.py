import pandas as pd
import sys
#read kernel_profiles/albert-train_ncu.csv and kernel_profiles/albert-train_nsys.csv to merge the two
def getkernel_profile(ncuFile, nsysFile, outType):
    df_ncu = pd.read_csv(ncuFile)
    df_nsys = pd.read_csv(nsysFile)
    #get only top 15 rows from df_nsys and in the category of CUDA_KERNEL
    df_nsys = df_nsys[df_nsys['Category'] == 'CUDA_KERNEL']
    #drop Category column
    df_nsys = df_nsys.drop(['Category', "Med (ns)","Min (ns)","Max (ns)","StdDev (ns)", "Total Time (ns)"], axis=1)
    df_nsys = df_nsys.head(20)

    #match df_nsys Operation column with df_ncu Kernel Name column
    #merge df_ncu and df_nsys by Operation column
    df_ncu = df_ncu.rename(columns={'Kernel Name': 'Operation'})


    df_filter = df_ncu[df_ncu['Metric Name'] == 'Duration']
    #get the row with the maximum duration for each operation
    df_filter = df_filter.groupby('Operation').max()
    #print(df_filter)
    #use df_filter id to filter out the rows in df_ncu
    df_ncu = df_ncu[df_ncu['ID'].isin(df_filter['ID'])]
    df_ncu.to_csv(f"source/ncu/processed_{outType}_ncu.csv", header=True, index=False)
    df_ncu['Operation'].astype(str)
    df_nsys['Operation'].astype(str)
    df_merged = pd.merge(df_ncu, df_nsys, on='Operation', how='outer')
    df_merged.head()
    #select required entry in Metric Name column
    features = ["SM Busy","Memory Throughput","DRAM Throughput","Compute (SM) Throughput","Block Size","Grid Size", "Registers Per Thread", "Static Shared Memory Per Block","Waves Per SM","Achieved Active Warps Per SM"]
    df_merged = df_merged[df_merged['Metric Name'].isin(features)]
    #create a new df, with features as columns, Operation as index, and Average as values
    if 'Average' in df_merged.columns:
        df_merged= df_merged.rename(columns={'Average': 'Metric Value'})
    df_merged = df_merged[['Operation', 'Metric Name', 'Metric Value', "Time (%)", "Instances", "Avg (ns)"]]
    #df_merged = df_merged[['Operation', 'Metric Name', 'Average']]
    #remove all , in Average
    df_merged['Metric Value'] = df_merged['Metric Value'].str.replace(',', '').astype(float)
    #transform Average to float datatype
    

    df = pd.DataFrame(df_merged)

    # Drop duplicates to ensure unique ['Operation', 'Metric Name'] pairs for pivoting
    df = df.drop_duplicates(subset=['Operation', 'Metric Name'])
    # Finding duplicates in the 'Name' column
    # 'keep=False' marks all duplicates as True
    

    # Pivot the DataFrame
    df_pivoted = df.pivot(index="Operation", columns="Metric Name", values="Metric Value")

    #add index to a operation column
    df_pivoted['Operation'] = df_pivoted.index
    df_pivoted.reset_index(drop=True, inplace=True)
    df_merged = pd.merge(df_pivoted, df_nsys, on='Operation')
    #print(df_merged.head())
    print(df_merged.columns)
    #write df_merged to csv
    df_merged.to_csv(f'test_merged.csv', header=True, index=False)
    #get weighted sum
    sum_time = df_merged['Time (%)'].sum()
    print(f"sum % of compute kernel time: {sum_time}")
    #get the sum of compute throughput*time% for each type. store it in another array

    df_agg = pd.DataFrame()
    for col in df_merged.columns:
        if col in ["Time (%)", "Operation"]:
            continue
        #df_kernel[col] = df_kernel[col].str.replace('%', '').astype(float)
        #add normalize by exapnding the column by time% and multiplying by 100/sum_time
        df_agg[f"{col}"] = df_merged[col].astype(float) * df_merged["Time (%)"]/100 * (100/sum_time)
    df_agg = pd.DataFrame(df_agg.sum()).T
    print(df_agg)
    #add type column as output type
    df_agg['Type'] = outType

    df_agg.to_csv(f'output_nsys/{outType}_kernel_info.csv', header=True, index=False)

#getkernel_profile('kernel_profiles/albert-train_ncu.csv', 'kernel_profiles/albert-train_nsys.csv')
if __name__ == "__main__":
    getkernel_profile(sys.argv[1], sys.argv[2], sys.argv[3])