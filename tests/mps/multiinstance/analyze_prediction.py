import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot, pareto
import sys
# Step 1: Read the CSV file
file_path = sys.argv[1]  # Replace with your file path
test_set_path = sys.argv[2]  # Replace with your file path

output_dir = sys.argv[3]  # Replace with your file path
data = pd.read_csv(file_path)
test_set = pd.read_csv(test_set_path)
# Step 2: Assume 'y_test' and 'y_pred' are in your dataset or calculated elsewhere.
# For demonstration, let's assume 'y_test' is the actual throughput, and 'y_pred' is predicted throughput.
#add workload1, workload2, w1_Threadpercent, w2_Threadpercent from test_set to data
data["workload1"] = test_set["workload1"]
data["workload2"] = test_set["workload2"]
data["w1_Threadpercent"] = test_set["w1_Threadpercent"]
data["w2_Threadpercent"] = test_set["w2_Threadpercent"]

# If these columns are not in your dataset, replace them with the correct column names or computations.
y_test = data['y_test']  # Replace with actual column for test values
y_pred = data['y_pred']  # Replace with actual column for predicted values

# Step 3: Calculate the Percentage Error
percentage_error = abs((y_test - y_pred) / y_test) * 100
#why percentage error negaive??


# Step 4: Add workload pair column by concatenating workload1 and workload2
data['workload_pair'] = data['workload1'] + " + " + data['workload2']

# Step 5: Create a DataFrame with percentage error and workload pair
df_errors = pd.DataFrame({'percentage_error': percentage_error, 'workload_pair': data['workload_pair'], 'w1_Threadpercent': data["w1_Threadpercent"], 'w2_Threadpercent': data["w2_Threadpercent"]})

#sort df_errors by percentage_error
sorted_highest_errors = df_errors.sort_values(by='percentage_error', ascending=False)




# Step 8: Save the sorted errors to a new CSV file
sorted_highest_errors_df = sorted_highest_errors.reset_index()  # Reset index to include 'workload_pair' as a column
#sorted_highest_errors_df.columns = ['workload_pair', 'max_percentage_error']  # Rename columns
#drop index
sorted_highest_errors_df = sorted_highest_errors_df.drop(columns=['index'])
# Save to CSV
output_file = f"{output_dir}/sorted_highest_error_rates.csv"
sorted_highest_errors_df.to_csv(output_file, index=False)

# Print the highest error rates
print("Top 10 Workload Pairs with Highest Error Rates:")
print(sorted_highest_errors.head(30))


# Step 8: Plot Error Distribution
plt.figure(figsize=(10, 6))

# Histogram plot
sns.histplot(sorted_highest_errors['percentage_error'], bins=30, kde=True, color='blue', stat="density", linewidth=0)

# Title and labels
plt.title("Error Distribution of Workload Pairs", fontsize=16)
plt.xlabel("Absolute Percentage Error (%)", fontsize=14)
plt.ylabel("Density", fontsize=14)

# Display the plot
plt.savefig(f"{output_dir}/error_distribution.png")


# Step 9: Q-Q Plot for Pareto Distribution
# Fit the data to Pareto distribution to estimate 'b' parameter
shape_param, loc_param, scale_param = pareto.fit(sorted_highest_errors['percentage_error'])

plt.figure(figsize=(10, 6))

# Generate Q-Q plot using estimated parameters
probplot(sorted_highest_errors['percentage_error'], dist="pareto", sparams=(shape_param,), plot=plt)

plt.title("Q-Q Plot for Error Distribution (Pareto)", fontsize=16)
plt.savefig(f"{output_dir}/qq_plot.png")