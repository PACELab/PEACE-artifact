import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def characterize_workloads(baseline_file, is_mem_mid):
    df = pd.read_csv(baseline_file)
    df = df[~df['Type'].str.contains("BlackScholes")]
    workloads = df['Type']
    compute_info = df['SMACT%100']
    mem_info = df['DRAMA%100']
    
    if is_mem_mid:
        categories = {
            ('C_high', 'M_high'): [], ('C_high', 'M_mid'): [], ('C_high', 'M_low'): [],
            ('C_low', 'M_high'): [], ('C_low', 'M_mid'): [], ('C_low', 'M_low'): []
        }
        for workload, compute, mem in zip(workloads, compute_info, mem_info):
            compute_cat = 'C_high' if compute >= 50 else 'C_low'
            if mem >= 66: mem_cat = 'M_high'
            elif 33 <= mem < 66: mem_cat = 'M_mid'
            else: mem_cat = 'M_low'
            categories[(compute_cat, mem_cat)].append(workload)
    else:
        categories = {
            ('C_high', 'M_high'): [], ('C_high', 'M_low'): [],
            ('C_low', 'M_high'): [], ('C_low', 'M_low'): []
        }
        for workload, compute, mem in zip(workloads, compute_info, mem_info):
            compute_cat = 'C_high' if compute >= 50 else 'C_low'
            mem_cat = 'M_high' if mem >=50 else 'M_low'
            categories[(compute_cat, mem_cat)].append(workload)
    return categories

AXIS_FONT = 22
LEGEND_FONT = 14 
LABEL_FONT = 23
TEXT_BOX_FONT_SIZE_RATIO = AXIS_FONT - 2
TEXT_BOX_FONT_SIZE_FREQ_DIST = AXIS_FONT - 2


def collect_workload_data(directory, workload_names):
    all_dfs = []
    read_cnt = 0
    for workload in workload_names:
        csv_pattern = os.path.join(directory, f'{workload}.csv')
        csv_files = glob.glob(csv_pattern)
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                try:
                    df_temp = pd.read_csv(csv_file)
                    if not df_temp.empty:
                        all_dfs.append(df_temp)
                        read_cnt += 1
                    else:
                        print(f"Warning: Empty CSV file found and skipped: {csv_file}")
                except pd.errors.EmptyDataError: 
                    print(f"Warning: Empty CSV file (pd.errors.EmptyDataError): {csv_file}")
                except Exception as e:
                    print(f"Warning: Error reading {csv_file}: {e}")
            else:
                print(f"Warning: No file found for {csv_pattern} (searched for {workload}.csv)")
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Total read {read_cnt} files into combined DataFrame.")
        return combined_df
    else:
        print("No matching CSV files found to combine.")
        return pd.DataFrame()

def categorize_workload_pairs(df, categories):
    workload_to_cat = {}
    for cat, workloads in categories.items():
        for w in workloads: workload_to_cat[w] = cat
    
    combined_categories = {}
    for cat1_key in categories.keys():
        for cat2_key in categories.keys():
            combined_categories[(cat1_key, cat2_key)] = []

    unmatched_workloads = set()
    if 'workload1' not in df.columns or 'workload2' not in df.columns:
        print("Error: 'workload1' or 'workload2' not in DataFrame for categorization.")
        for key in combined_categories: combined_categories[key] = pd.DataFrame()
        return combined_categories

    for _, row in df.iterrows():
        w1, w2 = row['workload1'], row['workload2']
        cat1, cat2 = workload_to_cat.get(w1), workload_to_cat.get(w2)
        if cat1 and cat2:
            combined_categories[(cat1, cat2)].append(row)
        else:
            if not cat1: unmatched_workloads.add(w1)
            if not cat2: unmatched_workloads.add(w2)
    if unmatched_workloads:
        print(f"Warning: Workloads not in characterization: {unmatched_workloads}")

    for key in combined_categories:
        combined_categories[key] = pd.DataFrame(combined_categories[key]) if combined_categories[key] else pd.DataFrame()
    return combined_categories

def merge_reversed_categories(combined_categories):
    merged_categories = {}
    processed_pairs_canonical = set()
    for (cat1, cat2), df1 in combined_categories.items():
        canonical_pair_key = tuple(sorted(((cat1), (cat2)), key=lambda x: str(x)))
        if canonical_pair_key not in processed_pairs_canonical:
            df_list = [df1] if not df1.empty else []
            if cat1 != cat2:
                df2 = combined_categories.get((cat2, cat1), pd.DataFrame())
                if not df2.empty: df_list.append(df2)
            
            if df_list:
                merged_df = pd.concat(df_list, ignore_index=True)
                merged_categories[(cat1, cat2)] = merged_df
            else:
                 merged_categories[(cat1, cat2)] = pd.DataFrame()
            processed_pairs_canonical.add(canonical_pair_key)
    return merged_categories


def plot_category_ratios(combined_categories_input, output_file):
    # This function now receives merged_categories as combined_categories_input
    compute_order = {'C_high': 0, 'C_low': 1}
    memory_order = {'M_high': 0, 'M_mid': 1, 'M_low': 2} 

    # Define the ratio columns for plotting, including the three specific baselines
    ratio_columns_map = {
        'PEACE': 'pred_vs_oracle_ratio',
        'Baseline 1530MHz': 'baseline_freq1530_actual_xput_vs_oracle_ratio',
        'Baseline 900MHz': 'baseline_freq900_actual_xput_vs_oracle_ratio',
        'Baseline 300MHz': 'baseline_freq300_actual_xput_vs_oracle_ratio',
        'Fair Partition': 'fairpartition_vs_oracle_ratio',
        'No Partition': 'nopartition_vs_oracle_ratio'
    }
    plot_methods = list(ratio_columns_map.keys()) 
    required_value_cols = list(ratio_columns_map.values()) # Actual column names from CSV

    # Helper function to shorten category tuple names for printing
    def shortentuple(tup):
        conv = {'C_high':'hiComp','C_low':'loComp','M_high':'hiMem','M_mid':'midMem','M_low':'loMem'}
        return f"({conv.get(tup[0],tup[0])}, {conv.get(tup[1],tup[1])})"

    # --- Print missing value counts for specified columns ---
    print("\n--- Missing Value Report ---")
    # Define columns to check for missing values specifically
    peace_col_name = 'pred_vs_oracle_ratio' # PEACE solution
    baseline_1530_col_name = 'baseline_freq1530_actual_xput_vs_oracle_ratio'
    baseline_900_col_name = 'baseline_freq900_actual_xput_vs_oracle_ratio'
    baseline_300_col_name = 'baseline_freq300_actual_xput_vs_oracle_ratio'
    
    cols_for_missing_report = [
        ('PEACE', peace_col_name),
        ('Baseline 1530MHz', baseline_1530_col_name),
        ('Baseline 900MHz', baseline_900_col_name),
        ('Baseline 300MHz', baseline_300_col_name)
    ]

    for cat_pair, df_cat in combined_categories_input.items():
        cat_label_for_print = f"{shortentuple(cat_pair[0])} + {shortentuple(cat_pair[1])}"
        print(f"Category: {cat_label_for_print} (Total rows: {len(df_cat)})")
        if df_cat.empty:
            print("  No data in this category.")
            for report_label, _ in cols_for_missing_report:
                 print(f"  Missing '{report_label}': N/A (no data)")
            continue

        for report_label, col_name_to_check in cols_for_missing_report:
            if col_name_to_check in df_cat.columns:
                missing_count = df_cat[col_name_to_check].isnull().sum()
                print(f"  Missing '{report_label}' ({col_name_to_check}): {missing_count} rows")
            else:
                # This case should ideally be caught by the main script's column check
                print(f"  Column '{col_name_to_check}' for '{report_label}' not found in this category's DataFrame.")
    print("--- End of Missing Value Report ---\n")
    # --- End of missing value report ---

    category_data = []
    for cat_pair, df in combined_categories_input.items(): # Iterate through the input (merged) categories
        if not df.empty:
            # Check if all required_value_cols (which now includes all baselines) are present in this df
            missing_data_cols_for_plot = [col for col in required_value_cols if col not in df.columns]
            if missing_data_cols_for_plot:
                print(f"Warning (plot_category_ratios): Category {cat_pair} is missing some data columns required for plotting: {missing_data_cols_for_plot}. This category might be skipped or have incomplete bars.")
                # Continue to try plotting with available data, mean will be NaN for missing ones.

            mean_ratios_for_cat = {}
            for label, col_name in ratio_columns_map.items():
                if col_name in df.columns:
                    # Ensure there are non-NaN values before calculating mean to avoid RuntimeWarning
                    if df[col_name].notna().any():
                        mean_ratios_for_cat[label] = df[col_name].mean()
                    else:
                        mean_ratios_for_cat[label] = np.nan # All values are NaN
                else:
                    mean_ratios_for_cat[label] = np.nan # Column itself is missing

            # Only add category if it has at least one valid mean ratio
            if any(pd.notna(val) for val in mean_ratios_for_cat.values()):
                 category_data.append((cat_pair, mean_ratios_for_cat))
            else:
                print(f"Note (plot_category_ratios): Category {cat_pair} has no valid mean ratios for any method. It will not be plotted.")


    if not category_data:
        print("No data to plot for performance ratios after processing. Check CSV contents, column names, and data validity.")
        plt.figure(figsize=(18, 9)); plt.text(0.5, 0.5, "No data for performance ratios.", ha='center', va='center', fontsize=16); plt.xticks([]); plt.yticks([]); plt.savefig(output_file, bbox_inches='tight'); plt.close(); print(f"Empty plot saved: {output_file}"); return

    def sort_key(item):
        cat1, cat2 = item[0]; cat1_mem = memory_order.get(cat1[1], 99); cat2_mem = memory_order.get(cat2[1], 99)
        return (compute_order.get(cat1[0],99), cat1_mem, compute_order.get(cat2[0],99), cat2_mem)

    category_data.sort(key=sort_key)
    categories_labels = [f"{shortentuple(cp[0])} +\n{shortentuple(cp[1])}" for cp, _ in category_data]
    ratios_for_plotting = {method: [cd[1].get(method, np.nan) for cd in category_data] for method in plot_methods}

    num_bars = len(plot_methods) 
    bar_width = 0.15 # Adjusted bar width for more bars
    index = np.arange(len(categories_labels))
    plt.figure(figsize=(max(18, len(categories_labels) * num_bars * bar_width * 1.5), 9)) # Dynamic width
    
    # Expanded colors for up to 6 methods
    colors = ['black', 'salmon', 'lightgreen', 'gold', 'skyblue', 'mediumorchid'] 
    
    all_bars_objects = [] 

    for i, method_name in enumerate(plot_methods):
        position_offset = (i - (num_bars - 1) / 2.0) * bar_width
        bar_positions = index + position_offset
        # Ensure ratios_for_plotting[method_name] has values, otherwise plot NaNs which plt.bar handles
        current_ratios = ratios_for_plotting.get(method_name, [np.nan] * len(categories_labels))
        bars = plt.bar(bar_positions, current_ratios, bar_width, label=method_name, color=colors[i % len(colors)])
        all_bars_objects.append(bars)

    text_offset = 0.015 
    for bar_container in all_bars_objects:
        for bar_element in bar_container:
            height = bar_element.get_height()
            if pd.isna(height): continue 
            plt.text(bar_element.get_x() + bar_element.get_width() / 2,
                     height + text_offset, 
                     f'{height:.2f}', 
                     ha='center', va='bottom', fontsize=TEXT_BOX_FONT_SIZE_RATIO,
                     bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2', alpha=0.7))
        
    plt.ylabel("Performance Ratio (vs Oracle)", fontsize=LABEL_FONT - 1)
    plt.xticks(index, categories_labels, rotation=35, fontsize=AXIS_FONT - 6, ha='right')
    plt.yticks(fontsize=AXIS_FONT -2)
    plt.legend(fontsize=LEGEND_FONT, loc='upper center', ncol=min(num_bars, 4), bbox_to_anchor=(0.5, 1.16)) # ncol can be adjusted
    plt.grid(axis='y', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 1, 0.93]) # Adjust rect if legend is too large
    plt.savefig(output_file, bbox_inches='tight'); plt.close()
    print(f"Performance ratio plot saved as {output_file}")

    # Print average ratios
    print("\n--- Average Performance Ratios (vs Oracle) ---")
    for method_name in plot_methods:
        # Use the data prepared for plotting, which handles missing methods/categories
        all_ratios_for_method = ratios_for_plotting.get(method_name, [])
        valid_ratios = [r for r in all_ratios_for_method if pd.notna(r)] 
        if valid_ratios: 
            print(f"Average {method_name}: {np.mean(valid_ratios):.3f} (from {len(valid_ratios)} categories)")
        else: 
            print(f"Average {method_name}: N/A (no valid data or all NaNs)")
    print("--- End of Average Performance Ratios ---\n")


def plot_selected_model_freq_distribution(combined_categories_input, output_file, selected_freq_col_name):
    # This function now receives merged_categories as combined_categories_input
    compute_order = {'C_high': 0, 'C_low': 1}
    memory_order = {'M_high': 0, 'M_mid': 1, 'M_low': 2}
    
    category_freq_proportions = []
    all_freq_values = set()

    # Helper function to shorten category tuple names for printing (if needed, or define globally)
    def shortentuple(tup): # Copied here for local use if not global
        conv = {'C_high':'hiComp','C_low':'loComp','M_high':'hiMem','M_mid':'midMem','M_low':'loMem'}
        return f"({conv.get(tup[0],tup[0])}, {conv.get(tup[1],tup[1])})"

    for cat_pair, df in combined_categories_input.items():
        if df.empty or selected_freq_col_name not in df.columns:
            if not df.empty: 
                 print(f"Warning (plot_selected_model_freq_distribution): Category {cat_pair}, missing column: {selected_freq_col_name}. Skipping.")
            # Append with cat_pair for consistent sorting, but empty series
            category_freq_proportions.append((cat_pair, pd.Series(dtype=float))) 
            continue
        
        # Ensure there are non-NaN values before value_counts
        if df[selected_freq_col_name].notna().any():
            freq_counts = df[selected_freq_col_name].value_counts(normalize=True)
            category_freq_proportions.append((cat_pair, freq_counts))
            all_freq_values.update(freq_counts.index)
        else:
            # Append with cat_pair for consistent sorting, but empty series
            category_freq_proportions.append((cat_pair, pd.Series(dtype=float)))
            print(f"Warning (plot_selected_model_freq_distribution): Category {cat_pair}, column '{selected_freq_col_name}' contains all NaN values.")


    if not any(s.any() for _, s in category_freq_proportions) or not all_freq_values: # Check if any series has data
        print(f"No data or no unique frequency values found in column '{selected_freq_col_name}' to plot for frequency distribution.")
        plt.figure(figsize=(18, 7)); plt.text(0.5, 0.5, f"No data for '{selected_freq_col_name}' distribution.", ha='center', va='center', fontsize=16); plt.xticks([]); plt.yticks([]); plt.savefig(output_file, bbox_inches='tight'); plt.close(); print(f"Empty frequency distribution plot saved: {output_file}"); return

    # Filter out categories that ended up with no frequency data before sorting and plotting
    category_freq_proportions = [item for item in category_freq_proportions if item[1].any()]
    if not category_freq_proportions: # Double check after filtering
        print(f"No categories with valid frequency data for '{selected_freq_col_name}'. Skipping plot.")
        # (Similar empty plot generation as above)
        return

    sorted_unique_freqs = sorted(list(all_freq_values), key=lambda x: float(x) if isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).isdigit() else str(x))


    def sort_key(item):
        cat1, cat2 = item[0]; cat1_mem = memory_order.get(cat1[1], 99); cat2_mem = memory_order.get(cat2[1], 99)
        return (compute_order.get(cat1[0],99), cat1_mem, compute_order.get(cat2[0],99), cat2_mem)

    category_freq_proportions.sort(key=sort_key)
    categories_labels = [f"{shortentuple(cp[0])} +\n{shortentuple(cp[1])}" for cp, _ in category_freq_proportions]
    
    num_categories = len(categories_labels)
    index = np.arange(num_categories)
    plt.figure(figsize=(max(18, num_categories * 1.2), 7)) 
    
    plot_data = pd.DataFrame(index=categories_labels, columns=sorted_unique_freqs, dtype=float).fillna(0.0)

    for i, (cat_pair_tuple, freq_series) in enumerate(category_freq_proportions):
        cat_label = categories_labels[i] 
        if not freq_series.empty:
            for freq_val, proportion in freq_series.items():
                if freq_val in plot_data.columns:
                     plot_data.loc[cat_label, freq_val] = proportion
    
    bar_width = 0.7 
    bottoms = np.zeros(num_categories)
    # More robust color cycling
    base_colors = ['dodgerblue', 'darkorange', 'limegreen', 'firebrick', 'purple', 'cyan', 'gold', 'magenta', 'teal', 'sienna']
    freq_colors = [base_colors[i % len(base_colors)] for i in range(len(sorted_unique_freqs))]


    for i_freq, freq_val_str in enumerate(sorted_unique_freqs):
        proportions = plot_data[freq_val_str].values
        bars = plt.bar(index, proportions, bar_width, label=str(freq_val_str), bottom=bottoms, 
                       color=freq_colors[i_freq]) # Use str(freq_val_str) for label
        for bar_idx, bar_element in enumerate(bars):
            height = bar_element.get_height()
            if height > 0.01: # Only add text for visible segments
                plt.text(bar_element.get_x() + bar_element.get_width() / 2., 
                         bottoms[bar_idx] + height / 2., 
                         f'{height:.2f}', 
                         ha='center', va='center', 
                         fontsize=TEXT_BOX_FONT_SIZE_FREQ_DIST, color='white', weight='bold')
        bottoms += proportions

    plt.ylabel("Proportion of Pairs", fontsize=LABEL_FONT - 2)
    plt.xticks(index, categories_labels, rotation=35, fontsize=AXIS_FONT - 6, ha='right')
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=AXIS_FONT - 4)
    plt.ylim(0, 1.05) # Ensure y-axis goes up to at least 1.0
    
    plt.legend(title=f"'{selected_freq_col_name}' Values", fontsize=LEGEND_FONT-1, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=min(len(sorted_unique_freqs), 5))
    plt.title(f"Distribution of '{selected_freq_col_name}' by Workload Category", fontsize=LABEL_FONT -1, y=1.08) # Adjust y for legend
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.92]) # Adjust rect for title/legend
    plt.savefig(output_file, bbox_inches='tight'); plt.close()
    print(f"Frequency distribution plot saved as {output_file}")


if __name__ == "__main__":
    output_dir_prefix = "output_plots_final_mergecudaDL_epsilon2" 
    
    use_single_summary_csv = True
    # IMPORTANT: Update this path to your actual CSV file location
    #single_csv_path = '/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/results/paper/xput_under_powercap_mergecudaDL_unseen_multi_freq_powercap60_epsilon10_pred_vs_baselines_dvfs/summary_xput_under_powercap_all_pairs.csv' 
    single_csv_path = "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/results/paper/xput_under_powercap_mergecudaDL_unseen_multi_freq_powercap60_epsilon2_pred_vs_baselines_dvfs/summary_xput_under_powercap_all_pairs.csv"
    # IMPORTANT: Update this path to your actual baseline characterization file
    baseline_file_for_characterization = '/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/tests/mps/analysis/0206_baseline_metrics.csv'
    is_mid = False # Set to True or False based on your characterization needs
    
    SELECTED_MODEL_FREQ_COLUMN_NAME = "selected_model_freq" # Column name for frequency distribution plot

    if not os.path.exists(baseline_file_for_characterization):
        print(f"ERROR: Baseline file for characterization not found: {baseline_file_for_characterization}"); exit()
    categories = characterize_workloads(baseline_file_for_characterization, is_mem_mid=is_mid)
    
    if use_single_summary_csv:
        if not os.path.exists(single_csv_path):
            print(f"ERROR: Single summary CSV not found: {single_csv_path}"); exit()
        print(f"Loading data from single summary CSV: {single_csv_path}")
        try:
            combined_df = pd.read_csv(single_csv_path)
        except Exception as e:
            print(f"ERROR: Could not read CSV file '{single_csv_path}': {e}"); exit()
    else: 
        example_directory = f'./custom_data_dir_v3' 
        if not os.path.exists(example_directory):
             os.makedirs(example_directory, exist_ok=True)
             print(f"INFO: Example directory created: {example_directory}. Populate with CSVs if not using single CSV.")
        # Get all unique workload names for data collection if not using single CSV
        all_characterized_workloads = list(set(w for cat_list in categories.values() for w in cat_list))
        combined_df = collect_workload_data(example_directory, all_characterized_workloads)

    if combined_df.empty: print("Combined DataFrame is empty. Exiting."); exit()
    
    # Updated default_ratio_cols_map to include the three specific baselines
    default_ratio_cols_map = {
        'PEACE': 'pred_vs_oracle_ratio',
        'Baseline 1530MHz': 'baseline_freq1530_actual_xput_vs_oracle_ratio',
        'Baseline 900MHz': 'baseline_freq900_actual_xput_vs_oracle_ratio',
        'Baseline 300MHz': 'baseline_freq300_actual_xput_vs_oracle_ratio',
        'Fair Partition': 'fairpartition_vs_oracle_ratio',
        'No Partition': 'nopartition_vs_oracle_ratio'
    }
    # Ensure these essential columns are present in the loaded DataFrame
    required_cols_for_script = ['workload1', 'workload2', SELECTED_MODEL_FREQ_COLUMN_NAME] + list(default_ratio_cols_map.values())
    required_cols_for_script = list(set(required_cols_for_script)) # Remove duplicates

    missing_essential_cols = [col for col in required_cols_for_script if col not in combined_df.columns]
    if missing_essential_cols:
        print(f"CRITICAL ERROR: Input CSV ('{single_csv_path if use_single_summary_csv else 'loaded from directory'}') is missing essential columns: {missing_essential_cols}.")
        print(f"Available columns in loaded DataFrame: {combined_df.columns.tolist()}")
        print("Script cannot proceed without these columns. Please check your CSV file or data loading process.")
        exit()
    
    # Drop duplicates based on workload pairs
    if 'workload1' in combined_df.columns and 'workload2' in combined_df.columns: 
        combined_df = combined_df.drop_duplicates(subset=["workload1", "workload2"], keep="last")
    
    # Categorize and merge workload pairs
    categorized_pairs = categorize_workload_pairs(combined_df, categories)
    merged_categories_data = merge_reversed_categories(categorized_pairs) # Renamed to avoid conflict
    
    # Prepare output directory
    debug_output_path = os.path.join(output_dir_prefix, f"ismid{is_mid}")
    os.makedirs(debug_output_path, exist_ok=True)
    print(f"\nOutput plots and category CSVs will be saved in: {debug_output_path}")

    # Save each category's DataFrame to a CSV file for debugging/inspection
    print("\nSaving data for each merged category to CSV files...")
    for cat_pair, df_cat in merged_categories_data.items():
        if not df_cat.empty:
            cat1_label = f"{cat_pair[0][0].replace('_','')}{cat_pair[0][1].replace('_','')}"
            cat2_label = f"{cat_pair[1][0].replace('_','')}{cat_pair[1][1].replace('_','')}"
            category_filename = f"category_{cat1_label}_vs_{cat2_label}.csv"
            
            try:
                df_cat.to_csv(os.path.join(debug_output_path, category_filename), index=False)
                print(f"  Saved: {category_filename} ({len(df_cat)} rows)")
            except Exception as e:
                print(f"  ERROR saving {category_filename}: {e}")
        # else: # Optionally print info about empty categories
            # print(f"  Skipped saving empty category: {cat_pair}")
    
    # Plot 1: Performance Ratios
    ratio_plot_filename = f'performance_ratios_by_category_midmem{is_mid}.png'
    ratio_plot_full_path = os.path.join(debug_output_path, ratio_plot_filename)
    print(f"\nGenerating Performance Ratio plot: {ratio_plot_full_path}")
    # Pass the merged data to the plotting function
    plot_category_ratios(merged_categories_data, output_file=ratio_plot_full_path)

    # Plot 2: Selected Model Frequency Distribution (Stacked Bar Chart)
    freq_dist_plot_filename = f'selected_model_freq_distribution_midmem{is_mid}.png'
    freq_dist_plot_full_path = os.path.join(debug_output_path, freq_dist_plot_filename)
    print(f"\nGenerating Selected Model Frequency Distribution plot: {freq_dist_plot_full_path}")
    # Pass the merged data to the plotting function
    plot_selected_model_freq_distribution(merged_categories_data, 
                                          output_file=freq_dist_plot_full_path,
                                          selected_freq_col_name=SELECTED_MODEL_FREQ_COLUMN_NAME)

    print("\nScript finished.")