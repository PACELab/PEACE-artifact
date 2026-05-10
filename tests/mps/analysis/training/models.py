#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#coorelation of each feature with L2 norm
# Load the data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
file_path = '/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/training/kernel_labels_L2norm.csv'
data = pd.read_csv(file_path)
#filter out workloads with  name including resnet, mobilenet
data = data[~data['workload1'].str.contains('resnet')]
data = data[~data['workload2'].str.contains('resnet')]
data = data[~data['workload1'].str.contains('mobilenet')]
data = data[~data['workload2'].str.contains('mobilenet')]
#add w1+w2throughput column
#data['w1+w2throughput'] = data['w1throughput'] + data['w2throughput']
#first  plot L2 norm distribution vs (w1,w2)
# Optionally, ensure that there are no duplicate (workload1, workload2) pairs with different L2norms
# This step assumes you want the mean L2norm if duplicates exist. Adjust aggregation as needed.
data['L2norm'] = np.sqrt((data['w1throughput'] / data['w1exclusive_throughput'])**2 + (data['w2throughput'] / data['w2exclusive_throughput'])**2)

data_grouped = data.groupby(['workload1', 'workload2'])['L2norm'].mean().reset_index()

# Create a new column combining workload1 and workload2 for easier plotting
data_grouped['workload_pair'] = data_grouped['workload1'] + ", " + data_grouped['workload2']

# Sorting values for better visualization, if needed
data_grouped = data_grouped.sort_values(by='L2norm')
#get average and std of L2norm
print('Average L2norm:', data_grouped['L2norm'].mean())
print('Standard Deviation of L2norm:', data_grouped['L2norm'].std())
# Plotting
plt.figure(figsize=(14, 7))  # Adjust the size as needed
plt.bar(data_grouped['workload_pair'], data_grouped['L2norm'], color='skyblue')
plt.xlabel('Workload Pair (workload1, workload2)')
plt.ylabel('L2norm')
plt.title('L2norm Values by Workload Pair')
plt.xticks(rotation=90)  # Rotate labels to prevent overlap
plt.tight_layout()  # Adjust layout to make room for label rotation
#add legend that shows L2norm average and std

plt.show()

#plot L2norm distribution with min max scaling of 0,100
L2_minmax = (data['L2norm'] - data['L2norm'].min()) / (data['L2norm'].max() - data['L2norm'].min()) * 100
plt.figure(figsize=(14, 7))
plt.hist(L2_minmax, bins=50, color='skyblue')
plt.xlabel('L2norm (Min-Max Scaled)')
plt.ylabel('Frequency')
plt.title('L2norm Distribution (Min-Max Scaled)')
plt.show()

#plot L2norm distribution with z-score scaling
L2_zscore = (data['L2norm'] - data['L2norm'].mean()) / data['L2norm'].std()
#print workload1 and workload2 with L2norm > 3
data['L2_zscore'] = L2_zscore
print(data[['workload1', 'workload2']][data['L2_zscore'] > 3])
plt.figure(figsize=(14, 7))
plt.hist(L2_zscore, bins=50, color='skyblue')
plt.xlabel('L2norm (Z-Score/ standard Scaled)')
plt.ylabel('Frequency')
plt.title('L2norm Distribution (Z-Score Scaled)')
plt.show()

#plot L2norm distribution with log scaling
L2_log = np.log1p(data['L2norm'])

plt.figure(figsize=(14, 7))
plt.hist(L2_log, bins=50, color='skyblue')
plt.xlabel('L2norm (Log Scaled)')
plt.ylabel('Frequency')

plt.title('L2norm Distribution (Log Scaled)')
plt.show()

#plot scatter plot of w1throughput vs w2throughput
plt.figure(figsize=(10, 10))
plt.scatter(data['w1throughput'], data['w2throughput'], alpha=0.5)
plt.xlabel('Workload 1 Throughput')
plt.ylabel('Workload 2 Throughput')
#label data point with bert-train
for i in range(len(data)):
    if data['workload1'][i] == 'bert-train' or data['workload2'][i] == 'bert-train':
        plt.text(data['w1throughput'][i], data['w2throughput'][i], f"({data['workload1'][i]}, {data['workload2'][i]})")
plt.title('Workload 1 vs Workload 2 Throughput')
plt.show()
# std Normalize L2norm for better visualization
stdL2norm = (data['L2norm'] - data['L2norm'].mean()) / data['L2norm'].std()

#plot scatter plot of w1throughput-w1exclusive_throughput  vs w2throughput-w2exclusive_throughput with L2norm as color
plt.figure(figsize=(10, 10))
plt.scatter(data['w1exclusive_throughput'] - data['w1throughput'], data['w2exclusive_throughput'] - data['w2throughput'], c=stdL2norm, cmap='coolwarm')
#lable data point with (workload1, workload2)
for i in range(len(data)):
    if data['workload1'][i] == 'bert-train' or data['workload2'][i] == 'bert-train':
        plt.text(data['w1exclusive_throughput'][i] - data['w1throughput'][i], data['w2exclusive_throughput'][i] - data['w2throughput'][i]+0.5, f"({data['workload1'][i]}, {data['workload2'][i]})")
        #text with L2norm and stdL2norm
        plt.text(data['w1exclusive_throughput'][i] - data['w1throughput'][i], data['w2exclusive_throughput'][i] - data['w2throughput'][i], f"(L2:{data['L2norm'][i]:.2f}, L2_zscaled:{std_testnorm[i]:.2f})")

plt.xlabel('Workload 1 Exclusive Throughput - Workload 1  Throughput')
plt.ylabel('Workload 2 Exclusive Throughput - Workload 2  Throughput')
plt.title('Workload 1 vs Workload 2 Throughput Difference')
plt.colorbar(label='L2norm (Z-Score Scaled)')

#plot scatter plot of z-scaled w1throughput-w1exclusive_throughput  vs z-scaled  w2throughput-w2exclusive_throughput 
plt.figure(figsize=(10, 10))
w1_zscale_throughputdiff = data['w1exclusive_throughput'] - data['w1throughput']
w2_zscale_throughputdiff = data['w2exclusive_throughput'] - data['w2throughput']
w1_zscale_throughputdiff = (w1_zscale_throughputdiff - w1_zscale_throughputdiff.mean()) / w1_zscale_throughputdiff.std()
w2_zscale_throughputdiff = (w2_zscale_throughputdiff - w2_zscale_throughputdiff.mean()) / w2_zscale_throughputdiff.std()
plt.scatter(w1_zscale_throughputdiff, w2_zscale_throughputdiff, c=stdL2norm, cmap='coolwarm')
#label data point with (workload1, workload2)
for i in range(len(data)):
    if data['workload1'][i] == 'bert-train' or data['workload2'][i] == 'bert-train':
        plt.text(w1_zscale_throughputdiff[i], w2_zscale_throughputdiff[i], f"({data['workload1'][i]}, {data['workload2'][i]})")
        #text with L2norm and stdL2norm
        plt.text(w1_zscale_throughputdiff[i], w2_zscale_throughputdiff[i], f"(L2:{data['L2norm'][i]:.2f}, L2_zscaled:{std_testnorm[i]:.2f})")
plt.xlabel('Workload 1 Exclusive Throughput - Workload 1  Throughput (Z-Score Scaled)')
plt.ylabel('Workload 2 Exclusive Throughput - Workload 2  Throughput (Z-Score Scaled)')
plt.title('Workload 1 vs Workload 2 Throughput Difference (Z-Score Scaled)')
#plt.colorbar(label='L2norm (Z-Score Scaled)')
plt.show()


#plot scatter plot of w1throughput/w1exclusive_throughput vs w2throughput/w2exclusive_throughput with L2norm as color
plt.figure(figsize=(10, 10))
plt.scatter(data['w1throughput'] / data['w1exclusive_throughput'], data['w2throughput'] / data['w2exclusive_throughput'])
plt.xlabel('Workload 1 share/exclusive Throughput Ratio')
plt.ylabel('Workload 2 share/exclusive Throughput Ratio')
plt.title('Workload 1 vs Workload 2 Throughput Ratio')
#label data point with (workload1, workload2)

# std Normalize L2norm for better visualization
stdL2norm = (data['L2norm'] - data['L2norm'].mean()) / data['L2norm'].std()
#square root of sum of squares of w1throughput/w1exclusive_throughput and w2throughput/w2exclusive_throughput

testnorm = np.sqrt((data['w1throughput']/data['w1exclusive_throughput'])**2 + (data['w2throughput'] / data['w2exclusive_throughput'])**2)
std_testnorm = (testnorm - testnorm.mean()) / testnorm.std()
for i in range(len(data)):
    if 'bert-train' in data['workload1'][i] or 'bert-train' in data['workload2'][i]:
        pass
        plt.text(data['w1throughput'][i] / data['w1exclusive_throughput'][i], data['w2throughput'][i] / data['w2exclusive_throughput'][i], f"(L2:{testnorm[i]:.2f}, L2_zscaled:{std_testnorm[i]:.2f})")
        #plot text workload1, workload2
        plt.text(data['w1throughput'][i] / data['w1exclusive_throughput'][i] , data['w2throughput'][i] / data['w2exclusive_throughput'][i]+0.02, f"({data['workload1'][i]}, {data['workload2'][i]})")
    #plt.text(data['w1throughput'][i] / data['w1exclusive_throughput'][i], data['w2throughput'][i] / data['w2exclusive_throughput'][i], f"({(data['L2norm'][i]):2f}, {math.sqrt((data['w1throughput'][i] - data['w1exclusive_throughput'][i])**2 + (data['w2throughput'][i] - data['w2exclusive_throughput'][i])**2):.2f})")
    #plt.text(data['w1throughput'][i] / data['w1exclusive_throughput'][i], data['w2throughput'][i] / data['w2exclusive_throughput'][i], f"{math.sqrt((data['w1throughput'][i] / data['w1exclusive_throughput'][i])**2 + (data['w1throughput'][i] / data['w1exclusive_throughput'][i])**2):.2f}")
        #plt.text(data['w1throughput'][i] / data['w1exclusive_throughput'][i], data['w2throughput'][i] / data['w2exclusive_throughput'][i], f"{stdL2norm[i]:.2f}")
#set x scale min and max = (0.4,1)
plt.xlim(0.4, 5)
#set y scale min and max = (0.4,1)
plt.ylim(0.4, 5)
plt.show()

#plot scatter plot of z-scaled w1throughput/w1exclusive_throughput vs z-scaled w2throughput/w2exclusive_throughput
plt.figure(figsize=(10, 10))
w1_zscale_throughput = data['w1throughput'] / data['w1exclusive_throughput']
w2_zscale_throughput = data['w2throughput'] / data['w2exclusive_throughput']
w1_zscale_throughput = (w1_zscale_throughput - w1_zscale_throughput.mean()) / w1_zscale_throughput.std()
w2_zscale_throughput = (w2_zscale_throughput - w2_zscale_throughput.mean()) / w2_zscale_throughput.std()
plt.scatter(w1_zscale_throughput, w2_zscale_throughput, c=stdL2norm, cmap='coolwarm')
#label data point with (workload1, workload2)
for i in range(len(data)):
    if data['workload1'][i] == 'bert-train' or data['workload2'][i] == 'bert-train':
        plt.text(w1_zscale_throughput[i], w2_zscale_throughput[i], f"({data['workload1'][i]}, {data['workload2'][i]})")
        #text with L2norm and stdL2norm
        plt.text(w1_zscale_throughput[i], w2_zscale_throughput[i], f"(L2:{data['L2norm'][i]:.2f}, L2_zscaled:{std_testnorm[i]:.2f})")
plt.xlabel('Workload 1 share/exclusive Throughput Ratio (Z-Score Scaled)')
plt.ylabel('Workload 2 share/exclusive Throughput Ratio (Z-Score Scaled)')
#set x scale min and max = (-3,3)
plt.xlim(-0.5, 0.5)
#set y scale min and max = (-3,3)
plt.ylim(-0.5, 0.5)
plt.title('Workload 1 vs Workload 2 Throughput Ratio (Z-Score Scaled)')
plt.colorbar(label='L2norm (Z-Score Scaled)')
plt.show()


# Drop non-numerical and non-relevant columns
data = data.drop(columns=['workload1', 'workload2', 'idx1', 'idx2'])
# Calculate correlations
L2norm = data['L2norm']
#data = data.drop(columns = ['L2norm'])

#data['L2norm'] = L2norm
data["L2_div_exclusive"] = np.sqrt((data['w1throughput'] / data['w1exclusive_throughput'])**2 + (data['w2throughput'] / data['w2exclusive_throughput'])**2)
data["cross_sm"] = data['w1sm%'] * data['w2sm%']
data["cross_mem"] = data['w1mem%'] * data['w2mem%']
data["sum_sm"] = data['w1sm%'] + data['w2sm%']
data["sum_mem"] = data['w1mem%'] + data['w2mem%']
data["norm_cross_sm"] = (data['w1sm%'] * data['w2sm%']) / (data['w1sm%'] + data['w2sm%'])   
data["norm_cross_mem"] = (data['w1mem%'] * data['w2mem%']) / (data['w1mem%'] + data['w2mem%'])
data = (data - data.mean()) / data.std()
#data.drop(['L2norm'], axis=1, inplace=True)
#normalize the data
#data = 1/ 1-data
#data['L2norm'] = L2norm

correlations = data.corr()



# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlations, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

# Specifically print correlations of features with L2norm
print(correlations["L2_div_exclusive"].sort_values(ascending=False))
#categorize  the rows with correlations['L2norm'].sort_values values into 10 classes



# Perform PCA
pca = PCA(n_components=2)
X = data.drop(columns=['L2norm'])
#normaloze X
X = (X - X.mean()) / X.std()
X_pca = pca.fit_transform(X)
# Prepare the results for the PCA components and feature importances
pca_results = pca.components_, pca.explained_variance_ratio_
#feature_importance_results = list(zip(X.columns, feature_importance))

# Get the PCA components (loadings)
components = pca.components_.T  # Transpose to align with original features

# Define colors for arrows and text
arrow_colors = text_colors = [
    'r', 'g', 'b', 'c', 'm', 'y', 'k', 
    '#FFA07A', '#20B2AA', '#778899', '#B0C4DE', '#FFFFE0', 
    '#00FF7F', '#4682B4', '#D2B48C', '#008080', '#D8BFD8', 
    '#FF6347', '#40E0D0', '#EE82EE', '#FFD700'
]


# Create a scatter plot of the PCA-transformed data points
plt.figure(figsize=(10, 10))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
print(X_pca[:,0])#mobil-inf, bert-inf, whisper-inf

# Overlay arrows representing the loadings of original features
legend_handles = []
for i, (feature, arrow_color, text_color) in enumerate(zip(X.columns, arrow_colors, text_colors)):
    plt.arrow(0, 0, components[i, 0]*3, components[i, 1]*3, color=arrow_color, alpha=0.5)
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=arrow_color, markersize=10))
# Draw lines for PC1 and PC2 axes
#plt.axhline(0, color='grey', linestyle='--')
#plt.axvline(0, color='grey', linestyle='--')

# Annotate the principal components
#plt.text(max(X_pca[:,0]), 0, 'PC1', color='red', ha='right')
#plt.text(0, max(X_pca[:,1]), 'PC2', color='red', va='top')

# Set labels and title
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Biplot')
# Create custom legend with feature names
plt.legend(legend_handles, X.columns, loc='best')
# Show plot
plt.grid()
plt.show()


#sort data by L2norm, and label the top 10% as 1, the next 10% as 2, and so on
# Sort data by L2norm
data['L2norm_log'] = L2_log



#store the column names that has abs(correlations['L2norm']) < 0.3 into a list
least_corr_columns = list(correlations["L2_div_exclusive"][abs(correlations["L2_div_exclusive"]) < 0.4].index)
least_corr_columns.remove('w1throughput')
print(least_corr_columns)
#print(list(correlations['L2norm'][abs(correlations['L2norm']) < 0.3].index))


# In[63]:


import pandas as pd
import numpy as np
filename_type_dict = {'wav2vec2-base-960h' : 'wav2vec', 'bert-base-cased': 'bert',
                      'mobilenet': 'mobile', 'mobilenet_v2_1.0_224' : 'mobile','vit' : 'vit', 
                      'vit_h_14' : 'vit', 'whisper-large-v2': 'whisper',
                      'albert-base-v2' : 'albert'}
GET_WORKLOAD=""
def filter_data(data, workload, isbatchThroughput = False):
                              

    print("FILTERED workload: ", workload)
    # Filter data by workload name == workload1 or workload2
    #remove all workload that contains mobilenet_v2_1.0_224 and resnet-50 and mobilenet_
    data = data[~data['workload1'].str.contains('mobilenet_v2_1.0_224') & ~data['workload2'].str.contains('mobilenet_v2_1.0_224')]
    data = data[~data['workload1'].str.contains('resnet-50') & ~data['workload2'].str.contains('resnet-50')]
    data = data[~data['workload1'].str.contains('mobilenet_') & ~data['workload2'].str.contains('mobilenet_')]

    #filter out data with batch size 8
    data['w1batch_size'] = data['workload1'].str.extract(r'_batch(\d+)', expand=False).astype(int)
    data['w2batch_size'] = data['workload2'].str.extract(r'_batch(\d+)', expand=False).astype(int)
    #data = data[(data['w1batch_size'] == 8) & (data['w2batch_size'] == 8)]

    #exclude data with both trains
    #data = data[~data['workload1'].str.contains('train') | ~data['workload2'].str.contains('train')]
    #data = data[~data['workload1'].str.contains('wav2vec2-base-960h') & ~data['workload2'].str.contains('wav2vec2-base-960h')]
    #filter data = data with workload1 == workload or workload2 == workload
    
    # Divide throughput and exclusive throughput by batch size
    if isbatchThroughput:
        data['w1throughput'] = data['w1throughput'] / data['w1batch_size']
        data['w2throughput'] = data['w2throughput'] / data['w2batch_size']
        data['w1exclusive_throughput'] = data['w1exclusive_throughput'] / data['w1batch_size']
        data['w2exclusive_throughput'] = data['w2exclusive_throughput'] / data['w2batch_size']

    
    if workload != "":
        data = data[(data['workload1'].str.contains(workload)) | (data['workload2'].str.contains(workload))]
    if workload == "":
        #drop workload1 == mobile-inf_batch64  workload2 == mobile-inf_batch64
        return data
    #concat data with workload1 and workload2 that contains workload
    #print(data[data['workload1'].str.contains(workload)])
    #data = data[(data['workload1'].str.contains(workload) ) | (data['workload2'].str.contains(workload) )]
    
    #df = df1.append(data[data['workload2'].str.contains(workload)])
    return data
filename = '/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/training/kernel_labels_L2norm.csv'  
data = pd.read_csv(filename)

ret = filter_data(data, GET_WORKLOAD)
ret["w1_norm"] = ret['w1throughput'] / ret['w1exclusive_throughput']
ret["w2_norm"] = ret['w2throughput'] / ret['w2exclusive_throughput']
ret["L2norm"] = np.sqrt(ret["w1_norm"]**2 + ret["w2_norm"]**2)
ret["sum_relative throughput"] = ret["w1_norm"] + ret["w2_norm"]
ret["sum_throughput"] = ret["w1throughput"] + ret["w2throughput"]
ret["avg_sm%"] = (ret['w1sm%'] + ret['w2sm%']) / 2
ret["avg_mem%"] = (ret['w1mem%'] + ret['w2mem%']) / 2
ret.to_csv(f'{GET_WORKLOAD}_related.csv', index=False)   
print(ret['workload1'], ret['workload2'], ret['L2norm'])
print(ret["avg_sm%"])
#increase  x axis label size
#increase y axis label size

import matplotlib.pyplot as plt
#1. print l2 norm with x axis = (workload1, workload2) and
#2. two bars on the same plot. one bar with sum_sm% and another bar with sum_throughput

def shorten_label(label):
    for key, value in filename_type_dict.items():
        if key in label:
            #split the label by '-' in last occurence
            batch_type = label.rsplit('_', 1)[1]
            #print(label.rsplit('_', 1))
            return value+ '_' + batch_type
    return label



def  plot_throughput_systemmetric(ret,target):
        
    ret = ret.sort_values(by=target, ascending=False)
    ret['workload1'] = ret['workload1'].apply(shorten_label)
    ret['workload2'] = ret['workload2'].apply(shorten_label)
    x_labels = [f"{w1} / {w2}" for w1, w2 in zip(ret['workload1'], ret['workload2'])]
    x = np.arange(len(x_labels))  # the label locations
    # Plot
    fig, (ax1, ax11, ax2, ax3, ax4, ax5,ax6,ax7, ax8) = plt.subplots(9, 1, figsize=(18, 45), sharex=True)

    # Plot L2norm in the first subplot
    ax1.bar(x, ret[target], color='blue', label=target, width=0.4)
    ax1.set_ylabel(f'{target} (higher the better)',fontsize=18)
    ax1.set_title(f'{target} and Resource Metrics for Workload Pairs of {GET_WORKLOAD}')
    ax1.legend()
    ax1.grid()

    #plot sum of exclusive throughput in the second subplot
    bar_width = 0.4
    ax11.bar(x , ret['w1exclusive_throughput']+ ret['w2exclusive_throughput'], bar_width, label='exclusive_throughput', color='orange')
    ax11.set_xlabel('Workload Pairs',fontsize=20)
    ax11.set_ylabel('Exclusive Throughput',fontsize=20)
    ax11.set_xticks(x)
    ax11.set_xticklabels(x_labels, rotation=90, fontsize=20)
    ax11.legend()
    ax11.grid()
    # Plot sum_sm% and sum_mem% in the second subplot
    bar_width = 0.4
    ax2.bar(x - bar_width/2, ret['avg_sm%'], bar_width, label='avg_sm%', color='orange')
    ax2.bar(x + bar_width/2, ret['avg_mem%'], bar_width, label='avg_mem%', color='green')
    ax2.set_xlabel('Workload Pairs',fontsize=20)
    ax2.set_ylabel('Busy rate (%)',fontsize=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, rotation=90, fontsize=20)
    ax2.legend()
    ax2.grid()


    #plot sum of compute throughput and memory throughput
    bar_width = 0.4
    ax3.bar(x - bar_width/2 , (ret['w1_Compute (SM) Throughput']+ret['w2_Compute (SM) Throughput']) /2, bar_width, label='avg_Compute Throughput', color='orange')
    ax3.bar(x + bar_width/2, (ret['w1_Memory Throughput']+ret['w2_Memory Throughput']) / 2, bar_width, label='avg_Memory Throughput', color='green')
    ax3.set_xlabel('Workload Pairs',fontsize=20)
    ax3.set_ylabel('avg Throughput (%)',fontsize=20)
    ax3.set_xticks(x)
    ax3.set_xticklabels(x_labels, rotation=90, fontsize=20)
    ax3.legend()
    ax3.grid()



    #ax4 is bar of ret['Static Shared Memory'] + ret['Dynamic Shared Memory']
    bar_width = 0.4
    ax4.bar(x , ret['w1_Static Shared Memory']+ret['w2_Static Shared Memory'], bar_width, label='sum_Static Shared Memory', color='orange')
    #ax4.bar(x + bar_width/2, ret['w1_Memory Throughput']+ret['w2_Memory Throughput'], bar_width, label='sum_Memory Throughput', color='green')
    ax4.set_xlabel('Workload Pairs',fontsize=20)
    ax4.set_ylabel('Shared Memory (KB)',fontsize=20)
    ax4.set_xticks(x)
    ax4.set_xticklabels(x_labels, rotation=90, fontsize=20)
    ax4.legend()
    ax4.grid()

    bar_width = 0.4
    #ax3.bar(x - bar_width/2, ret['w1_Registers']+ret['w2_Registers'], bar_width, label='w1_Registers', color='orange')
    ax5.bar(x, ret['w1_Threads']+ret['w2_Threads'], bar_width, label='sum_Threads', color='green')
    ax5.set_xlabel('Workload Pairs',fontsize=20)
    ax5.set_ylabel('num of Threads',fontsize=20)
    ax5.set_xticks(x)
    ax5.set_xticklabels(x_labels, rotation=90, fontsize=20)
    ax5.legend()
    ax5.grid()
    

    #plot dram throughput
    bar_width = 0.4
    ax6.bar(x , ret['w1_DRAM Throughput']+ret['w2_DRAM Throughput'], bar_width, label='sum_DRAM Throughput', color='orange')
    ax6.set_xlabel('Workload Pairs',fontsize=20)
    ax6.set_ylabel('DRAM Throughput (%)',fontsize=20)
    ax6.set_xticks(x)
    ax6.set_xticklabels(x_labels, rotation=90, fontsize=20)
    ax6.legend()
    ax6.grid()

    #plot Registers
    bar_width = 0.4
    ax7.bar(x , ret['w1_Registers']+ret['w2_Registers'], bar_width, label='sum_Registers', color='orange')
    ax7.set_xlabel('Workload Pairs',fontsize=20)
    ax7.set_ylabel('Registers',fontsize=20)
    ax7.set_xticks(x)
    ax7.set_xticklabels(x_labels, rotation=90, fontsize=20)
    ax7.legend()
    ax7.grid()

    #plot memcap sum
    bar_width = 0.4
    ax8.bar(x , ret['w1memcap']+ret['w2memcap'], bar_width, label='sum_Memcap', color='orange')
    ax8.set_xlabel('Workload Pairs',fontsize=20)
    ax8.set_ylabel('Memcap',fontsize=20)
    ax8.set_xticks(x)
    ax8.set_xticklabels(x_labels, rotation=90, fontsize=20)
    ax8.legend()
    ax8.grid()

    #increase x axis label size

    fig.tight_layout()
    plt.show()

plot_throughput_systemmetric(ret, 'L2norm')
plot_throughput_systemmetric(ret, 'sum_throughput')
#plot_throughput_systemmetric(ret, 'sum_relative throughput')



ret


# In[60]:


#train test data split function 
RANDOMSEED = 30
CORRELATION = 0.2
TARGET="sum_throughput"
FILTERED_WORKLOAD = ''
TEST_WORKLOAD = ''
import pandas as pd
y_all_test = pd.DataFrame()
import numpy as np

def train_test_custom_split(data, target, test_workload=""):

    def preprocess_data(data, categorical_columns, target):
        # Drop non-numerical and non-relevant columns
        #drop categorical columns
        #augment sm and mem columns with cross
        #data["cross_sm"] = data['w1sm%'] * data['w2sm%']
        #data["cross_mem"] = data['w1mem%'] * data['w2mem%']
        #define L2norm
        #data['L2norm'] = np.sqrt((data['w1throughput'] / data['w1exclusive_throughput'])**2 + (data['w2throughput'] / data['w2exclusive_throughput'])**2)
        #data[target] = (data['w1throughput'] / data['w1exclusive_throughput']) + (data['w2throughput'] / data['w2exclusive_throughput'])
        
        if target == 'L2norm':
            data[target] = np.sqrt((data['w1throughput'] / data['w1exclusive_throughput'])**2 + (data['w2throughput'] / data['w2exclusive_throughput'])**2)
        elif target == 'sum_throughput':
            data[target] = data['w1throughput'] + data['w2throughput'] 
        else:
            print("target not found")

            return
        #data = data.drop(columns=categorical_columns)
        #get a set of feature colunmn names after removing w1 prefix
        feat_names = [col[2:] for col in data.columns if 'w1' in col] 
        print(feat_names)
        #remove throughput  columen since they were used to calculate L2norm
        columns_excluded = categorical_columns + ['w1throughput', 'w2throughput', 'w1batch_size', 'w2batch_size']
        for feat in feat_names:
            #add a new column with the name feat
            if f'w1{feat}' in  columns_excluded:
                print("excluded", f'w1{feat}')
                continue
            
            data[feat] = (data[f'w1{feat}'] + data[f'w2{feat}'])/2  if '%' in feat else data[f'w1{feat}'] + data[f'w2{feat}']
            
            columns_excluded.append(f'w1{feat}')
            columns_excluded.append(f'w2{feat}')
       

        #columns_excluded = categorical_columns + ['w1throughput', 'w2throughput', 'w1exclusive_throughput', 'w2exclusive_throughput', 'w1sm%', 'w2sm%', 'w1mem%', 'w2mem%']
        #columns_excluded = categorical_columns 
        least_correlated_columns = get_least_corr_columns(data=data, target_column=target, excluded_columns=columns_excluded)
        #data = data.drop(least_correlated_columns, axis=1)  # Drop least correlated columns
        drop_columns = columns_excluded + least_correlated_columns
        return data, drop_columns 
    def get_least_corr_columns(data, target_column, excluded_columns):
          
         
        #drop categorical columns or not using columns
        corr_data = data.drop(excluded_columns, axis=1)

        #z scale data
        corr_data = (corr_data - corr_data.mean()) / corr_data.std()  
        correlations = corr_data.corr()
        least_correlated_columns = list(correlations[target_column][abs(correlations[target_column]) < CORRELATION].index)
        print(correlations[target_column].sort_values(ascending=False))
        print(f"drop columns {least_correlated_columns}")
        print(f"using columns {list(correlations[target_column][abs(correlations[target_column]) >= CORRELATION].index)}")
        return least_correlated_columns
    
    categorical_columns = ['workload1', 'workload2', 'idx1', 'idx2']
    data, columns_excluded = preprocess_data(data, categorical_columns, target)
    print(f"columns excluded {columns_excluded}")
    #split data with wokload1 == berttrain of workload2 == berttrain

    if test_workload != "":
        test_set = data[(data['workload1'].str.startswith(test_workload) ) | (data['workload2'].str.startswith(test_workload) )]
        #exclude (wokload1,workload2)  == (berttrain. mobile-train) and (bert-train,bert-inf) from test-set
        #test_set = test_set[~((test_set['workload1'].str.contains(test_workload) ) & (test_set['workload2'].str.contains('mobile-train')))]
        #test_set = test_set[~((test_set['workload1'].str.contains(test_workload) ) & (test_set['workload2'].str.contains('bert-inf')))]
        #print(test_set['workload1'], test_set['workload2'])
        #set train_set to data excluding test_set
        train_set = data[~data.index.isin(test_set.index)]
    #print(train_set['workload1'], train_set['workload2'])
    #print(len(train_set))

        # Drop non-numerical and non-relevant columns
        train_set = train_set.drop(columns=columns_excluded)

        y_all_test["workload_pair"] = test_set['workload1'] + ', ' + test_set['workload2']
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOMSEED)
        y_all_test["workload_pair"] = X_test['workload1'] + ', ' + X_test['workload2']
        #drop non relevant columns
        X_train = X_train.drop(columns=columns_excluded)
        X_test = X_test.drop(columns=columns_excluded)
        #normalize all data in train_set and test_set
        X_train = (X_train - X_train.mean()) / X_train.std()
        X_test = (X_test - X_test.mean()) / X_test.std()
        y_train = (y_train - y_train.mean()) / y_train.std()
        y_test = (y_test - y_test.mean()) / y_test.std()
        print(X_test)
        print(y_test)
    
    return X_train, X_test, y_train, y_test

filename = '/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/training/kernel_labels_L2norm.csv'  
y_all_test = pd.DataFrame()

df = pd.read_csv(filename)
df = filter_data(df, FILTERED_WORKLOAD, isbatchThroughput=False)
X_train, X_test, y_train, y_test = train_test_custom_split(df,  target=TARGET, test_workload=TEST_WORKLOAD,)

y_all_test['y_test'] = y_test
y_all_test
X_test


# In[120]:


#XGBoost
import xgboost as xgb
filename = '/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/training/kernel_labels_L2norm.csv'  
data = pd.read_csv(filename)
data = filter_data(data, FILTERED_WORKLOAD)
# Splitting the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_custom_split(data, target=TARGET, test_workload=TEST_WORKLOAD)
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

# Define the space of hyperparameters to search
space = {
    'max_depth': hp.choice('max_depth', range(3, 10)),
    'n_estimators': hp.choice('n_estimators', range(50, 200)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2)
}

# Objective function to minimize
def objective(space):
    model = xgb.XGBRegressor(
        n_estimators = space['n_estimators'],
        max_depth = space['max_depth'],
        learning_rate = space['learning_rate']
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    return {'loss': mse, 'status': STATUS_OK}

# Run the algorithm
trials = Trials()
best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)
#use best parameters to train and predict the model
model = xgb.XGBRegressor(
    n_estimators = best_hyperparams['n_estimators'],
    max_depth = best_hyperparams['max_depth'],
    learning_rate = best_hyperparams['learning_rate']
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Root Mean Squared Error: {rmse}')
print(f'R^2 Score: {r2}')
#save y_test,y_pred,rmse to a csv
y_xgbtest = y_test.to_frame()
y_xgbtest['y_pred'] = y_pred
y_xgbtest['rmse'] = rmse
y_xgbtest['r2'] = r2
y_xgbtest.to_csv('y_test_y_pred_XGBoost_kernelonly.csv')
#merge y_all_test with y_xgbtest by index

#store y_pred in y_all_test

y_all_test['y_xgbpred'] = y_pred
#store rmse,r2 in y_all_test
y_all_test['xgb_rmse'] = rmse
y_all_test['xgb_r2'] = r2


best_hyperparams


# In[ ]:


y_all_test


# In[73]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

filename = '/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/training/kernel_labels_L2norm.csv'  
data = pd.read_csv(filename)
data = filter_data(data, FILTERED_WORKLOAD)
X_train, X_test, y_train, y_test = train_test_custom_split(data,target=TARGET,test_workload=TEST_WORKLOAD)
# Set up the parameter grid
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_features': [ "sqrt", "log2", None],
    'max_depth': [10, 20, 30,40,50],
    'min_samples_split': [2, 5, 10]
}

# Initialize the model
rf = RandomForestRegressor(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

# Fit the grid search model
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best score: {-grid_search.best_score_}')

# Evaluate the best model on the test set
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'R^2: {r2}')


# Optionally, save predictions and MSE to a CSV
y_test_df = y_test.reset_index()
y_test_df['Predicted'] = y_pred
y_test_df['MSE'] = mse
y_test_df['r2'] = r2
y_test_df.to_csv('y_test_y_pred_randomForest_kernelonly.csv', index=False)
y_all_test['y_randomForestpred'] = y_pred
#store rmse,r2 in y_all_test
y_all_test['y_randomForest_rmse'] = mse
y_all_test['y_randomForest_r2'] = r2
#plot confusion matrix with y_test and y_pred
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
#print feature importance in random forest
feature_importances = pd.DataFrame(best_rf.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)
# Plotting feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y=feature_importances.index, data=feature_importances)
plt.title('Feature Importance')
plt.show()
y_all_test
#confusion_matrix(y_test, y_pred)



# In[ ]:


import shap
explainer = shap.TreeExplainer(best_rf)
#print(X_test)
#print(X_test.loc[[3]], y_test.loc[[3]])
# Calculate Shap values
#choosen_instance = X_test.loc[[3]]
shap_values = explainer(X_train)
clust = shap.utils.hclust(X_train, y_train, linkage="single")
shap.plots.bar(shap_values, clustering=clust, clustering_cutoff=1)
shap.plots.bar(shap_values)
plt.figure(figsize=(10, 10))
for i, feature in enumerate(X_train):
    fig = shap.plots.scatter(shap_values[:, i],show=False, dot_size=26, ylabel="SHAP value\n(increasing patterns means more impact on L2norm")
    f = plt.gcf()
    f.set_linewidth(5)
    f.set_edgecolor("#04253a")
    plt.show()
#shap.plots.scatter(
#    shap_values, ylabel="SHAP value\n(increasing patterns means more impact on L2norm",
#)
shap.summary_plot(shap_values,X_train)


# In[61]:


#train a linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
filename = '/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/training/kernel_labels_L2norm.csv'  
data = pd.read_csv(filename)
data = filter_data(data, FILTERED_WORKLOAD)
X_train, X_test, y_train, y_test = train_test_custom_split(data,target=TARGET, test_workload=TEST_WORKLOAD)

model = LinearRegression()
model.fit(X_train, y_train)


# Save the model to a file
import pickle
with open('trained_models/linear_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)

#load   the model from a file
with open('trained_models/linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}')
print(f'R^2: {r2}')
#save predictions to a csv
y_test_df = y_test.reset_index()
y_test_df['Predicted'] = y_pred
y_test_df['MSE'] = mse
y_test_df['r2'] = r2
y_test_df.to_csv('y_test_y_pred_linearRegression_kernelonly.csv', index=False)
y_all_test['y_linearRegressionpred'] = y_pred
#store rmse,r2 in y_all_test
y_all_test['y_linearRegression_rmse'] = mse
y_all_test['y_linearRegression_r2'] = r2

#plot feature importance
importance = model.coef_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %s, Score: %.5f' % (X_train.columns[i],v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()
#plot prediction vs features with regression line
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=3)
plt.show()





#plot shapley values
import shap
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer(X_train)
shap.plots.bar(shap_values)
shap.summary_plot(shap_values, X_train)



# In[43]:


#nn approach
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# Assume X and y have been defined and are available from your dataset
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
filename = '/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/training/kernel_labels_L2norm.csv'  
data = pd.read_csv(filename)
data = filter_data(data, FILTERED_WORKLOAD)
X_train_scaled, X_test_scaled, y_train, y_test = train_test_custom_split(data, target= TARGET, test_workload=TEST_WORKLOAD)
# Scale the features
scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)
#y_train = scaler.fit_transform(y_train.values.reshape(-1, 1))
#y_test = scaler.transform(y_test.values.reshape(-1, 1))

import tensorflow as tf
tf.random.set_seed(50)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Output layer for regression
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_squared_error'])
#add dropout layer
model.add(Dropout(0.2))
#apply earlystopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=40)
# Train the model with seeds
history = model.fit(X_train_scaled, y_train, epochs=250, validation_split=0.2, verbose=1, callbacks=[early_stopping])
#history = model.fit(X_train_scaled, y_train, epochs=500, validation_split=0.2, verbose=1)
#plot history's traing and validation loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
#add title and labels
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
#calculate mape
#flatten y_pred
y_pred = y_pred.flatten()
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
#calculate sse
sse = np.sum((y_test - y_pred)**2)
rmse = mse ** 0.5
print(f"RMSE on Test Set: {rmse}")
r2 = r2_score(y_test, y_pred)
print(f'R^2 Score: {r2}')
#save y_test,y_pred,rmse to a csv
y_nntest = y_test.to_frame()
y_nntest['y_pred'] = y_pred
y_nntest['rmse'] = rmse
y_nntest['r2'] = r2
y_nntest['mape'] = mape
y_nntest['sse'] = sse
y_nntest.to_csv('y_test_y_pred_nn_kernelonly.csv')
y_all_test['y_nnpred'] = y_pred
#store rmse,r2 in y_all_test
y_all_test['y_nn_rmse'] = rmse
y_all_test['y_nn_r2'] = r2
y_all_test['y_nn_mape'] = mape
y_all_test['y_nn_sse'] = sse
print(y_pred)
y_all_test





# In[11]:


import h2o
from h2o.automl import H2OAutoML
h2o.init()



# In[12]:


import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# Append to a  global variable y_all_test
def append_metrics(model_name, y_test, y_pred):
    y_test = np.array(y_test).flatten()
    y_pred = np.array(y_pred).flatten()
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    sse = np.sum((y_test - y_pred)**2)
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE on Test Set: {rmse}")
    print(f'R^2 Score: {r2}')
    
    # Save y_test, y_pred, rmse, r2, mape, sse to a DataFrame
    
    y_model_test = pd.DataFrame(y_test)
    y_model_test['y_pred'] = y_pred
    y_model_test['rmse'] = rmse
    y_model_test['r2'] = r2
    y_model_test['mape'] = mape
    y_model_test['sse'] = sse
    
    # Save to a CSV file
    y_model_test.to_csv(f'y_test_y_pred_{model_name}.csv', index=False)
    
    # Append results to the global y_all_test DataFrame
    y_all_test[f'y_{model_name}_pred'] = y_pred
    y_all_test[f'y_{model_name}_rmse'] = rmse
    y_all_test[f'y_{model_name}_r2'] = r2
    y_all_test[f'y_{model_name}_mape'] = mape
    y_all_test[f'y_{model_name}_sse'] = sse
    return y_all_test

# Example usage
# Assuming y_test and y_pred are defined
# append_metrics('nn', y_test, y_pred)


# In[13]:


X_train, X_test, y_train, y_test = train_test_custom_split(data, target= TARGET, test_workload=TEST_WORKLOAD)

#use H2OAutoML to train the model with the inputs
#convert data to h2o frame
h2o_train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
h2o_test = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))
# Identify predictors and response
x = h2o_train.columns
y = TARGET
x.remove(y)
#train the model
aml = H2OAutoML(max_models=30, seed=30, sort_metric='mse', max_runtime_secs=3*60)
aml.train(x=x, y=y, training_frame=h2o_train)



# In[14]:


from h2o.automl import get_leaderboard
lb2 = get_leaderboard(aml, extra_columns='ALL')
lb2.head(rows=lb2.nrows)


# In[44]:


# Get the best model
best_model = aml.leader

model_path = h2o.save_model(model=best_model, path="./trained_models", force=True)
print(f"Best model saved to: {model_path}")

# Predict on the test set
predictions = best_model.predict(h2o_test)

# Convert predictions to a pandas DataFrame for easier handling
predictions_df = predictions.as_data_frame()

# Combine X_test, y_test, and predictions for comparison
results_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1)

# Display the first few rows of the results
print(results_df.head())

# Evaluate model performance on the test set
performance = best_model.model_performance(h2o_test)
print(performance)
#shut down h2o instance
#append to y_all_test
y_all_test = append_metrics('h2o', y_test, predictions_df['predict'])
y_all_test


# In[ ]:


aml.explain(h2o_train)


# In[62]:


#plot y_all_test as bars
plt.figure(figsize=(10, 20))
#plotting_columns = ['y_test', 'y_randomForestpred', 'y_xgbpred', 'y_linearRegressionpred', 'y_nnpred']

#get column names with pred in them
plotting_columns = ['y_test']
plotting_columns.extend([col for col in y_all_test.columns if 'pred' in col ])

#plotting_columns = ['y_test', 'y_nnpred', 'y_linearRegressionpred', 'y_h2o_pred']
#plotting_columns = ['y_test', 'y_nnpred', 'y_linearRegressionpred']

err_columns = ['rmse', 'r2']



#add missing columns in row 1 and 2
#y_all_test.loc[25, 'workload_pair'] = 'vit-inf, albert-train'
#y_all_test.loc[19, 'workload_pair'] = 'wav2vec-inf, albert-train'
y_all_test.dropna(inplace=True)
#add mean of y_test to y_all_test
y_all_test['y_test_mean'] = y_all_test['y_test'].mean()
#plot the y_all_test as bars with only plotting_columns
y_all_test.plot( kind='bar', figsize=(10, 6), y=plotting_columns, hatch='')
#y_all_test.plot( kind='bar', figsize=(10, 6))
#x ticks as workload pair
plt.xticks(range(len(y_all_test)), y_all_test['workload_pair'], rotation=90)
#x label as workload pair
plt.xlabel('Workload Pair')

plt.title(f'Actual vs Predicted {TARGET}_dataset-{FILTERED_WORKLOAD}_testworkload-{TEST_WORKLOAD}-seed-{RANDOMSEED}_multiplyfeats_corr{CORRELATION}')
#xlabel as workload pair
plt.xlabel('Workload Pair')
#plt.xlabel('Index')
plt.ylabel(f'Z-scaled {TARGET} (higher the better)')
plt.grid()

plt.savefig(f'results/Actual vs Predicted {TARGET}_dataset-{FILTERED_WORKLOAD}_testworkload-{TEST_WORKLOAD}-seed-{RANDOMSEED}_feats_multiplyfeats_corr{CORRELATION}.png', bbox_inches='tight')
plt.show()
#save y_all_test to a csv
#make results dir if not exists
import os
if not os.path.exists('results'):
    os.makedirs('results')
y_all_test.to_csv(f'results/y_all_test_{TARGET}_dataset-{FILTERED_WORKLOAD}_testworkload-{TEST_WORKLOAD}-seed-{RANDOMSEED}-feats_multiplyfeats_fearcorr{CORRELATION}.csv')
#print all columns of y_all_test ending with r2 and rmse
y_all_test.filter(like='r2', axis=1)
y_all_test.filter(like='rmse', axis=1)
y_all_test.filter(like='r2', axis=1).mean()
y_all_test.filter(like='rmse', axis=1).mean()
#plot the mean of r2 and rmse
y_all_test.filter(like='r2', axis=1).mean().plot(kind='bar')
plt.title(f'Mean R^2 Score_dataset-{FILTERED_WORKLOAD}_testworkload-{TEST_WORKLOAD}')
plt.show()
y_all_test.filter(like='rmse', axis=1).mean().plot(kind='bar')
plt.title(f'Mean RMSE_dataset-{FILTERED_WORKLOAD}_testworkload-{TEST_WORKLOAD}')
plt.show()
#print max y_test value and its workload pair
print(y_all_test.loc[y_all_test['y_test'].idxmax()])
#print max y_nnpred value and its workload pair
print(y_all_test.loc[y_all_test['y_nnpred'].idxmax()])

y_all_test



# In[ ]:


#write y_test_all parital data to csv
#write y_test_all column names : train workload, test workload, mean of groundtruth, Max ground truth pair,Max ground truth value, NN pred pair, NN pred value, L2norm ground truth with nn pred pair, diff- L2norm colocation with nn pred pair - Max_ground truth value, RMSE, R2, mape,sse




# In[196]:


#read baseline＿label_0422.csv
filename = '../baseline_label_0422.csv'
baseline = pd.read_csv(filename)
#plot x as Type column and y as sm% column with bar
#plot baseline['Type'],baseline['sm%']
#sort rows by sm% column
baseline = baseline.sort_values('sm%')
#plot sm%,mem% columns as bar on the same plot
plt.figure(figsize=(10, 6))



plt.bar(baseline['Type'],baseline['sm%'])
plt.bar(baseline['Type'],baseline['mem%'])
plt.legend(['sm%', 'mem%'])
#rotate x ticks by 90
plt.xticks(rotation=90)
#plot sm% column as bar
#plt.plot(baseline['Type'],baseline['sm%'], kind='bar')
plt.title('Baseline sm%')
#x label as Type coloumn
#x axis to be Type column
plt.xlabel('Type')

plt.show()

#plot mem% column as bar
#sort rows by mem% column
baseline = baseline.sort_values('mem%')
#plot baseline['Type'],baseline['mem%']
plt.bar(baseline['Type'],baseline['mem%'])
#rotate x ticks by 90
plt.xticks(rotation=90)
#plot mem% column as bar


plt.title('Baseline mem%')
plt.show()


# In[373]:


#read csv
df = pd.read_csv("/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/cclogs/ccv100_0505_01/ccv100_logs/analysis/training/results/trainnsys_infncu/total_results_customtestsets_L2norm_dataset--feats_multiplyfeats_corr0.1.csv")
#df = pd.read_csv("/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/cclogs/ccv100_0505_01/ccv100_logs/analysis/training/results/seedtest/total_results_L2norm_dataset-_testworkload--feats_multiplyfeats_corr0.1.csv")

#plot x as test workload and y as  'diff %'
plt.figure(figsize=(10, 6))
plt.bar(df['test_workload'],-df['diff %'])
plt.xticks(rotation=90)
plt.title('Diff of (max L2norm -predicted L2 norm)/max L2norm in %')
plt.show()
#print diff mean
print(df['diff %'].mean())

