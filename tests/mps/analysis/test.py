file_path  = "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/tests/mps/analysis/1222_baseline_metrics.csv"
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import linregress

data = pd.read_csv(file_path)

# Filter columns related to Exclusive throughput
columns_of_interest = [col for col in data.columns if "Exclusive" in col and "100" not in col]

filtered_data = data[columns_of_interest]
# Linear fitting function
def linear_fit(x, y):
    # Ensure x and y are NumPy arrays
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # Filter out invalid data
    mask = np.isfinite(x) & np.isfinite(y)
    #print(mask)
    x, y = x[mask], y[mask]

    if len(x) < 2 or len(y) < 2:
        raise ValueError("Insufficient data for linear regression")
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope, intercept, r_value

# Define the inverse exponential function
def inverse_exponential(x, a, b, c):
    return a * np.exp(-b * x) + c

# Perform linear fitting for each workload
sensitivity_results_linear = {}
for _, row in data.iterrows():
    workload = row['Type']
    thread_percentages = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    exclusive_throughputs = row[
        ['Exclusive10', 'Exclusive20', 'Exclusive30', 'Exclusive40', 'Exclusive50',
         'Exclusive60', 'Exclusive70', 'Exclusive80', 'Exclusive90']
    ].values

    # Perform linear fitting
    slope, intercept, r_value = linear_fit(thread_percentages, exclusive_throughputs)
    
    # Store results
    sensitivity_results_linear[workload] = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2
    }

# Convert results to a DataFrame
sensitivity_df_linear = pd.DataFrame.from_dict(sensitivity_results_linear, orient='index')


# Save to file
linear_output_file = "./workload_sensitivity_linear_analysis.csv"
sensitivity_df_linear.to_csv(linear_output_file, index=True)
exit(0)
# Display the results
sensitivity_df = pd.DataFrame.from_dict(sensitivity_results, orient='index', columns=['a', 'b', 'c'])
#tools.display_dataframe_to_user(name="Workload Sensitivity Results", dataframe=sensitivity_df)

# Plot an example fit for the first row
example_idx = 0
if example_idx in sensitivity_results and not isinstance(sensitivity_results[example_idx], str):
    params = sensitivity_results[example_idx]
    fitted_values = inverse_exponential(thread_percentages, *params)

    plt.figure(figsize=(8, 6))
    plt.scatter(thread_percentages, filtered_data.iloc[example_idx], label='Original Data', color='blue')
    plt.plot(thread_percentages, fitted_values, label='Fitted Curve', color='red')
    plt.xlabel('Thread Percentage')
    plt.ylabel('Exclusive Throughput')
    plt.title('Inverse Exponential Fit - Example')
    plt.legend()
    plt.show()