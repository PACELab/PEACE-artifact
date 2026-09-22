import os
import pandas as pd
import matplotlib.pyplot as plt

def get_thread_num_from_str(thread_comb_str):
        col_s = thread_comb_str.replace("\"", "").replace("(", "").replace(")", "").replace(" ", "").split(",")
        col_s = [c.split("_")[1] for c in col_s]
        return col_s

def get_pred_oracle_target_comparison(pred_df, pred_power_df, all_oracle_df, powercap_limit_df, target, weight, output_dir,rm_100partition, power_threshold=0.75, is_plot=False):
    #baseline listed here
    no_partition_baseline = "(w1_100, w2_100)"
    fair_partition_baseline = "(w1_50, w2_50)"
    # Revised baselines_gain_list to return None if any value is None
    def safe_div(a, b):
        if pd.isna(a) or pd.isna(b) or b == 0:
            return None
        return a / b
    combined_results = []
    if rm_100partition:
        #remove all points that has w1_Threadpercent = 100 and w2_Threadpercent = 100
        assert("w1_Threadpercent" in pred_df.columns and "w2_Threadpercent" in pred_df.columns)
        assert("w1_Threadpercent" in pred_power_df.columns and "w2_Threadpercent" in pred_power_df.columns)
        pred_df = pred_df[(pred_df["w1_Threadpercent"] != 100) & (pred_df["w2_Threadpercent"] != 100)]
        pred_power_df = pred_power_df[(pred_power_df["w1_Threadpercent"] != 100) & (pred_power_df["w2_Threadpercent"] != 100)]

    #in all_oracle_df, add w1_Threadpercent column, w2_Threadpercent column
    all_oracle_df["w1_Threadpercent"] = all_oracle_df["Thread_combination"].apply(lambda x: get_thread_num_from_str(x)[0]).astype(int)
    all_oracle_df["w2_Threadpercent"] = all_oracle_df["Thread_combination"].apply(lambda x: get_thread_num_from_str(x)[1]).astype(int)
    #replace Workload1 to workload in all_oracle_df
    all_oracle_df = all_oracle_df.rename(columns={"Workload1": "workload1", "Workload2": "workload2"})
    #join all_oracle_df with pred_df, pred_power_df with w1_Threadpercent, w2_Threadpercent
    #print(f"all_oracle_df=\n{all_oracle_df}")
    #print(f"pred_df=\n{pred_df}")
    all_df = pd.merge(all_oracle_df, pred_df,
                             on=["workload1", "workload2", "w1_Threadpercent", "w2_Threadpercent"],
                             how="inner")
    pred_power_df = pred_power_df.rename(columns={"y_pred": "y_pred_power", "y_test": "y_test_power"})
    all_df = pd.merge(all_df, pred_power_df,
                             on=["workload1", "workload2", "w1_Threadpercent", "w2_Threadpercent"],
                             how="inner")
    
    print(f"target={target}")
    #compute target value in all_oracle_df 
    if target == "EDP":#energy delay product
    
        all_oracle_df[target] = all_oracle_df["Energy"] /  all_oracle_df["weight_Throughput_sum"]
        all_df[target] = all_df["Energy"] / all_df["weight_Throughput_sum"]
        all_df[f"pred_{target}"] = all_df["y_pred_power"] * all_df["Duration"] /  all_df["y_pred"]
        all_df["pred_energy"] = all_df["y_pred_power"] * all_df["Duration"]
    elif target == "xput_per_joule":
        all_oracle_df["Energy"] = all_oracle_df["Energy"] / weight
        all_df["Energy"] = all_df["Energy"] / weight
        all_oracle_df[target] = all_oracle_df["weight_Throughput_sum"] / all_oracle_df["Energy"]
        all_df[target] = all_df["weight_Throughput_sum"] / all_df["Energy"]
        all_df[f"pred_{target}"] = all_df["y_pred"] / (all_df["y_pred_power"] * all_df["Duration"] / weight)

    elif target == "xput_per_power":
        all_oracle_df[target] = all_oracle_df["weight_Throughput_sum"] / all_oracle_df["Power"]
        all_df[target] = all_df["weight_Throughput_sum"] / all_df["Power"]
        all_df[f"pred_{target}"] = all_df["y_pred"] / all_df["y_pred_power"]

    elif target == "power_per_xput":
        all_oracle_df[target] = all_oracle_df["Power"] / all_oracle_df["weight_Throughput_sum"]
        all_df[target] = all_df["Power"] / all_df["weight_Throughput_sum"]
        all_df[f"pred_{target}"] = all_df["y_pred_power"] / all_df["y_pred"]
    #save all_oracle_df to csv
    
    #elif target == "xput_plus_energy":#energy+throughput
    #    #energy+throughput
    #    weight = 1
    #    all_oracle_df[target] = all_oracle_df["Energy"] + weight * all_oracle_df["weight_Throughput_sum"]
    #    all_df[target] = all_df["Energy"] + weight * all_df["weight_Throughput_sum"]
    #    all_df[f"pred_{target}"] = all_df["y_pred_power"] * all_df["Duration"] + weight * all_df["y_pred"]
    elif target == "energy_plus_duration":#energy+duration
        all_oracle_df[target] = all_oracle_df["Energy"] + weight * all_oracle_df["Duration"]
        all_df[target] = all_df["Energy"] + weight * all_df["Duration"]
        all_df[f"pred_{target}"] = all_df["y_pred_power"] * all_df["Duration"] + weight * all_df["Duration"]
    else:
        raise ValueError(f"Unsupported target value {target}")

    #all_df.to_csv("all_merged_df.csv", index=False)
    oracle_output_file = os.path.join(output_dir, "all_oracle_labels.csv")
    #all_oracle_df.to_csv(oracle_output_file, index=False)
    workload_pairs = all_df[["workload1", "workload2"]].drop_duplicates().values

    for workload1, workload2 in workload_pairs:
        print(f"Processing workloads: {workload1}, {workload2}")
        filtered_all_df = all_df[(all_df["workload1"] == workload1) & 
                              (all_df["workload2"] == workload2)]
        if filtered_all_df.empty:
            print(f"No data for {workload1}, {workload2}")
            continue

        
       
        #compute target value with filtered_xput_df and filtered_power_df
        if target in ["EDP", "energy_plus_duration", "power_per_xput"]:
            select_idx = filtered_all_df[target].idxmin()
            select_pred_idx = filtered_all_df[f"pred_{target}"].idxmin()
        elif target in ["xput_per_joule", "xput_per_power"]:
            #get max target idx
            select_idx = filtered_all_df[target].idxmax()
            select_pred_idx = filtered_all_df[f"pred_{target}"].idxmax()
        else:
            print(f'is {target in ["EDP", "energy_plus_duration", "xput_per_joule", "xput_per_power"]}')
            raise ValueError(f"Unsupported target value {target}")
            
        best_oracle_target_value = filtered_all_df.loc[select_idx, target]
        best_oracle_throughput = filtered_all_df.loc[select_idx, "weight_Throughput_sum"]
        best_oracle_power = filtered_all_df.loc[select_idx, "Power"]
        best_w1_threadpercent = filtered_all_df.loc[select_idx, "w1_Threadpercent"]
        best_w2_threadpercent = filtered_all_df.loc[select_idx, "w2_Threadpercent"]
        #get max pred target idx
        pred_pred_value = filtered_all_df.loc[select_pred_idx, f"pred_{target}"]
        pred_exec_value = filtered_all_df.loc[select_pred_idx, target]
        pred_exec_throughput = filtered_all_df.loc[select_pred_idx, "weight_Throughput_sum"]
        pred_exec_power = filtered_all_df.loc[select_pred_idx, "Power"]
        pred_w1_threadpercent = filtered_all_df.loc[select_pred_idx, "w1_Threadpercent"]
        pred_w2_threadpercent = filtered_all_df.loc[select_pred_idx, "w2_Threadpercent"]
        #get pred_df
        filtered_pred_df = filtered_all_df.loc[select_pred_idx]


        #get no_parition_baseline target from all_oracle_df
        filtered_nopartition_df = all_oracle_df[(all_oracle_df["workload1"] == workload1) &
                                        (all_oracle_df["workload2"] == workload2) &
                                        (all_oracle_df["Thread_combination"] == no_partition_baseline)]
        if not filtered_nopartition_df.empty:
            #get value of target
            no_partition_baseline_value = filtered_nopartition_df[target].iloc[0]
        else:
            no_partition_baseline_value = None
        
        
        #get fair_partition_baseline target
        fair_partition_baseline_df = filtered_all_df[filtered_all_df["Thread_combination"] == fair_partition_baseline]
        if not fair_partition_baseline_df.empty:
            fair_partition_baseline_value = fair_partition_baseline_df[target].iloc[0]
        else:
            fair_partition_baseline_value = None

        # Powercap limit calculation
        if powercap_limit_df is None:
            powercap_limit = None
        else:
            raise ValueError("recheck. not reimplemented yet")
            filtered_powercap_limit_df = powercap_limit_df[(powercap_limit_df["workload1"] == workload1) & 
                                                            (powercap_limit_df["workload2"] == workload2)]
            if filtered_powercap_limit_df.empty:
                print(f"No powercap limit data for {workload1}, {workload2}")
                powercap_limit = None
            else:
                powercap_columns = [col for col in filtered_powercap_limit_df.columns if "powercap" in col]
                assert len(powercap_columns) == 1, f"Expected 1 powercap column, found {len(powercap_columns)}"
                powercap_column = powercap_columns[0]
                powercap_limit = power_threshold * filtered_powercap_limit_df[powercap_column].iloc[0]



        current_results= {
            "workload1": workload1,
            "workload2": workload2,
            "best_w1_threadpercent": best_w1_threadpercent,
            "best_w2_threadpercent": best_w2_threadpercent,
            "pred_w1_threadpercent": pred_w1_threadpercent,
            "pred_w2_threadpercent": pred_w2_threadpercent,
            f"best_{target}_oracle_value": best_oracle_target_value, 
            f"best_{target}_oracle_throughput": best_oracle_throughput,#executed throughput under best w1, w2 percentage with given target
            f"best_{target}_oracle_power": best_oracle_power,#executed power under best w1, w2 percentage with given target
            f"pred_{target}_exec_value" : pred_exec_value, #the executed value of given target  under predicted throughput, power value 
            f"pred_{target}_exec_throughput": pred_exec_throughput,#executed thorughput under predicted w1, w2 percentage with given target
            f"pred_{target}_exec_power": pred_exec_power,#executed power under predicted w1, w2 percentage with given target
            f"pred_{target}_pred_value": pred_pred_value, #the predicted value of given target  under predicted throughput, power value
            "powercap_limit": powercap_limit,
            f"{target}_no_partition_baseline_value": no_partition_baseline_value,
            f"{target}_fair_partition_baseline_value": fair_partition_baseline_value,
            #get ratio compare value with baseline no partition with safe div
            #ratio vs oracle value
            f"pred_{target}_ratio_oracle": safe_div(pred_exec_value, best_oracle_target_value),
            f"pred_{target}_ratio_nopartition": safe_div(pred_exec_value, no_partition_baseline_value),
            f"pred_{target}_ratio_fairpartition": safe_div(pred_exec_value, fair_partition_baseline_value),
            f"fairpartition_{target}_ratio_oracle": safe_div(fair_partition_baseline_value, best_oracle_target_value),
            f"pred_{target}_ratio_pred_exec": safe_div(pred_pred_value, pred_exec_value),
            
            }
        
        combined_results.append(current_results)
        print(f"current_results= {current_results}")
        if is_plot:
        #plot using filtered_all_df, filtered filtered_nopartition_df, filtered_fairpartition_df, filtered_powercap_limit_df
            plot_each_workload(workload1=workload1, workload2=workload2, x="Power", y=target,  all_MPS_df=filtered_all_df, filtered_pred_df=filtered_pred_df, filtered_nopartition_df=filtered_nopartition_df, 
                            fair_partition_baseline_df=fair_partition_baseline_df, powercap_limit_df=powercap_limit_df, target=target, output_dir=output_dir)
    pred_df_processed = pd.DataFrame(combined_results)
    if pred_df_processed.empty:
        print("No prediction results to process")
        return pred_df_processed

    # No best_dvfs_df defined, so no merge here as per your original function
    final_df = pred_df_processed  # Directly use pred_df_processed if no merge is intended
    
    if final_df.empty:
        print("No prediction results after processing")
        return final_df

    # Ratios require columns not present without merge, adjusting to available data
    #if "best_throughput" in final_df.columns and "pred_best_throughput" in final_df.columns:
    #    final_df["throughput_ratio_mps"] = final_df["pred_best_throughput"] / final_df["best_throughput"]
    #if "best_power" in final_df.columns and "pred_best_power" in final_df.columns:
    #    final_df["power_ratio_mps"] = final_df["pred_best_power"] / final_df["best_power"]
    #final_df[f"{target}_ratio_oracle"] = final_df[f"pred_{target}_exec_value"] / final_df[f"best_{target}_oracle_value"]
    #final_df[f"pred_{target}_exec_throughput_ratio_oracle"] = final_df[f"pred_{target}_exec_throughput"] / final_df[f"best_{target}_oracle_throughput"]
    #final_df[f"pred_{target}_exec_power_ratio_oracle"] = final_df[f"pred_{target}_exec_power"] / final_df[f"best_{target}_oracle_power"]
    #add output path by joining output_dir with worklaod  name
    #output_filename = os.path.join(output_dir, f"{workload1}_{workload2}.csv")
    #final_df.to_csv(output_filename, index=False)
    #raise ValueError("stop")
    return final_df, all_df
def plot_each_workload(workload1, workload2, x , y,  all_MPS_df, filtered_pred_df, filtered_nopartition_df, fair_partition_baseline_df, powercap_limit_df, target, output_dir):
    
    if  all_MPS_df.empty:
        print(f"  No filtered all data for {workload1}, {workload2}")
        return
    #plot all points in filtered_all_df with column x, y in blue
    plt.figure(figsize=(10, 6))
    plt.scatter( all_MPS_df[x],  all_MPS_df[y], color='blue', label='All points')
    #plot no partition baseline data in red
    if not fair_partition_baseline_df.empty:
        plt.scatter(fair_partition_baseline_df[x], fair_partition_baseline_df[y], color='green', label=f'Fair Partition Baseline\n {target}= {fair_partition_baseline_df[y].iloc[0]:2f}')
    if not filtered_pred_df.empty:
        print(f"filtered_pred_df for {workload1}, {workload2}:\n{filtered_pred_df}")
        #print(f"filtered_pred_df=\n{filtered_pred_df}")
        plt.scatter(filtered_pred_df[x], filtered_pred_df[y], color='orange', label=f'Predicted\n{target}= {filtered_pred_df[y]:2f}')
    if not filtered_nopartition_df.empty:
        assert(filtered_nopartition_df.shape[0] == 1)
        #plt.scatter(filtered_nopartition_df[x], filtered_nopartition_df[y], color='red', label=f'No Partition Baseline\n{target}= {filtered_nopartition_df[y].iloc[0]:2f}')    
    if not filtered_pred_df.empty and not fair_partition_baseline_df.empty:
        #add text box of target gain of pred vs fair partition baseline
        plt.text(filtered_pred_df[x] + 1, filtered_pred_df[y],
                f"Pred Gain: {filtered_pred_df[y] / fair_partition_baseline_df[y].iloc[0] :.2f}x", color='orange', 
                ha='left', va='bottom', bbox=dict(facecolor='0.85', alpha=0.8))

    plt.xlabel(f'{x}')
    plt.ylabel(f'{y}')
    plt.title(f'{target} for {workload1} and {workload2}')
    plt.legend()

    plot_file = os.path.join(output_dir, f"plot_{workload1}_{workload2}.png")
    plt.savefig(plot_file)
    plt.close()
    #raise ValueError("stop")
    return


def analyze_predictions_crossvalidate_target_with_baselines(predict_xput_dir, predict_power_dir, dvfs_file, temporal_file, powercap_limit_file, 
                                                           pred_xput_file_common_name, pred_power_file_common_name, target, output_base_name, 
                                                           throughput_model, power_model, num_testset, weight, dvfs_oracle, rm_100partition, 
                                                           is_plot=False, save_csv=True):
    
    
    all_dvfs_df = pd.read_csv(dvfs_file)
    if temporal_file:
        temporal_df = pd.read_csv(temporal_file)
    else:
        temporal_df = None
    if powercap_limit_file:
        powercap_limit_df = pd.read_csv(powercap_limit_file)
    else:
        powercap_limit_df = None

    all_final_dfs = []
    output_dir = os.path.dirname(output_base_name)
    os.makedirs(output_base_name, exist_ok=True)

    # Walk through predict_xput_dir and filter by num_testset
    for root, _, files in os.walk(predict_xput_dir):
        
        filename = os.path.basename(root)
        folder_name = os.path.basename(os.path.dirname(root))
        
        # Check if folder matches the specified num_testset (e.g., "fold_2_*" for num_testset=2)
        if f"fold_{num_testset}_" not in root:
            continue
        print(f"root={root}")
        if pred_xput_file_common_name in files:
            xput_file = os.path.join(root, pred_xput_file_common_name)
            pred_df = pd.read_csv(xput_file)
            print(files)
            # Check if the throughput_model exists in this folder
            #print(f"parent dir name={os.path.basename(os.path.dirname(root))}")
            if throughput_model not in os.path.basename(os.path.dirname(root)):
                continue
            print(f"throughput model={os.path.basename(os.path.dirname(root))}")
            relative_path = os.path.relpath(root, predict_xput_dir)
            print(f"relative_path={relative_path}")
            # Replace throughput_model with power_model in the relative_path
            relative_path_power = relative_path.replace(throughput_model, power_model)
            print(f"relative_path_power={relative_path_power}")
            # Get power root with the modified relative path
            power_root = os.path.join(predict_power_dir, relative_path_power)
            power_file = os.path.join(power_root, pred_power_file_common_name)
            print(f"power_file={power_file}")

            if os.path.exists(power_file):
                pred_power_df = pd.read_csv(power_file)
                # Check if the power_model exists in this folder
                if power_model not in os.path.basename(os.path.dirname(power_root)):
                    continue

                print(f"Processing throughput model {throughput_model}, power model{power_model} for {xput_file} and {power_file}")
                final_df, all_df = get_pred_oracle_target_comparison(pred_df, pred_power_df, all_dvfs_df, powercap_limit_df, 
                                                             target, weight=weight, output_dir=output_base_name, 
                                                             rm_100partition=rm_100partition, is_plot=is_plot)

                if not final_df.empty:
                    all_final_dfs.append(final_df)
                    relative_path_str = "".join(relative_path.split("/")[:1])
                    if save_csv:
                        #all_df.to_csv(f"{output_base_name}/{relative_path_str}_all_df.csv", index=False)
                        final_df.to_csv(f"{output_base_name}/{relative_path_str}.csv", index=False)
                        final_df_mean = final_df.mean()
                        print(f"Mean of {relative_path_str}:\n{final_df_mean}")
                        final_df_mean.to_csv(f"{output_base_name}/{relative_path_str}_mean.csv", index=True)
                else:
                    print(f"No data file saved for {output_dir}")
            else:
                print(f"Warning: No matching power file found for {xput_file} at {power_file}")

    if not all_final_dfs:
        raise ValueError(f"No valid prediction files processed for num_testset={num_testset}")

    # Combine all final_dfs and compute mean gain
    combined_df = pd.concat(all_final_dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["workload1", "workload2"], keep="last")
    
    # Assuming final_df contains a column like 'gain' or 'target_ratio' from get_pred_oracle_target_comparison
    # Adjust the column name based on what get_pred_oracle_target_comparison returns

    #combined_df.to_csv(f"{output_base_name}_combined.csv", index=False)
    mean_ratios = combined_df.mean()
    #mean_ratios.to_csv(f"{output_base_name}_mean_ratios.csv", index=True)
    print(f"Mean ratios for num_testset={num_testset}:\n{mean_ratios}")

    return mean_ratios  # Return mean gain for further use if needed



def analyze_predictions_target_with_baselines(predict_xput_dir, predict_power_dir, dvfs_file, temporal_file, powercap_limit_file, pred_xput_file_common_name, pred_power_file_common_name,  target, output_base_name, weight, dvfs_oracle, rm_100partition, is_plot=False, save_csv=True):
    all_dvfs_df = pd.read_csv(dvfs_file)
    if temporal_file:
        temporal_df = pd.read_csv(temporal_file) 
    else:
        temporal_df = None
    if powercap_limit_file:
        powercap_limit_df = pd.read_csv(powercap_limit_file) 
    else:
        powercap_limit_df = None

    all_final_dfs = []
    all_baseline_gains = []
    output_dir = os.path.dirname(output_base_name)
    os.makedirs(output_dir, exist_ok=True)
    
    
    for root, _, files in os.walk(predict_xput_dir):
        
        if pred_xput_file_common_name in files:
            #if "vit_h_14_batch2-train" not in root or "whisper-large-v2_batch2-inf" not in root:
            #    continue
            xput_file = os.path.join(root, pred_xput_file_common_name)
            pred_df = pd.read_csv(xput_file)
            
            relative_path = os.path.relpath(root, predict_xput_dir)
            power_root = os.path.join(predict_power_dir, relative_path)
            power_file = os.path.join(power_root, pred_power_file_common_name)
            
            if os.path.exists(power_file):
                pred_power_df = pd.read_csv(power_file)
                #output_name = f"{output_base_name}.csv"
                print(f"Processing {relative_path} for {xput_file} and {power_file}")
                workload_name = "".join(relative_path.split("/")[:1])
                #final_df = get_pred_oracle_comparison(pred_df, pred_power_df, all_dvfs_df, powercap_limit_df, output_name)
                final_df, all_df = get_pred_oracle_target_comparison(pred_df, pred_power_df, all_dvfs_df, powercap_limit_df, target, weight=weight, output_dir=output_base_name, rm_100partition=rm_100partition,  is_plot=is_plot,
                                            )

                if not final_df.empty:
                    all_final_dfs.append(final_df)
                    #save final_df with the relative path
                    #strip the relative path to get the workload name
                    relative_path_str = "".join(relative_path.split("/")[:1])
                    if save_csv:

                        #save alldf to csv
                        #all_df.to_csv(f"{output_base_name}/{relative_path_str}_all_df.csv", index=False)
                        final_df.to_csv(f"{output_base_name}/{relative_path_str}.csv", index=False)
                        #get mean of all
                        final_df_mean = final_df.mean()
                        print(f"Mean of {relative_path_str}:\n{final_df_mean}")
                        final_df_mean.to_csv(f"{output_base_name}/{relative_path_str}_mean.csv", index=True)
                    


                    #baseline_gain = plot_target_pred_oracle_comparison(final_df, all_dvfs_df, temporal_df, output_dir, dvfs_oracle=dvfs_oracle)
                    #all_baseline_gains.append(baseline_gain)
                    #print(f"Saved to {output_name}")
                else:
                    print(f"No data file saved for {output_dir}")
            else:
                print(f"Warning: No matching power file found for {xput_file} at {power_file}")

    if not all_final_dfs:
        raise ValueError("No valid prediction files processed")
    #if not all_baseline_gains:
    #    raise ValueError("No valid baseline gains computed")
    #all_baseline_gain_df = pd.concat(all_baseline_gains, ignore_index=True)
    #all_baseline_gain_df.to_csv(f"{output_base_name}_baseline_gains.csv", index=False)
    combined_df = pd.concat(all_final_dfs, ignore_index=True)
    #drop duplicates rows of workload1, workload2
    combined_df = combined_df.drop_duplicates(subset=["workload1", "workload2"], keep="last")
    mean_ratios = combined_df.mean()  # Adjusted to available columns
    #all_baseline_mean_ratios = all_baseline_gain_df.mean()
    #combine mean_ratios and all_baseline_mean_ratios
    #mean_ratios = pd.concat([mean_ratios, all_baseline_mean_ratios], axis=0)


    
    combined_df.to_csv(f"{output_base_name}_combined.csv", index=False)
    mean_ratios.to_csv(f"{output_base_name}_mean_ratios.csv", index=True)
    print(f"Mean ratios:\n{mean_ratios}")


if __name__ == "__main__":

    #predict_xput_dir = "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/data/experiment_inputs/colocations/output/run03072025_data1229_mergecudaDL_comb2/unseen_partition/throughput/rand10/extratrees"
    #predict_power_dir = "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/data/experiment_inputs/colocations/output/run03072025_data1229_mergecudaDL_comb2/unseen_partition/power/rand10/extratrees"
    #oracle_file = "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/data/model_datasets/02072025_noweight/with_energy_duration/0307_data1229_mergecudaDL_throughput_total_labels_comb2_labels.csv"
    
    predict_xput_dir = "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/data/experiment_inputs/colocations/output/run03112025_DL0207_0307_nonDL0311_nodvfs/unseen_partition/throughput/rand10/extratrees"
    predict_power_dir = "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/data/experiment_inputs/colocations/output/run03112025_DL0207_0307_nonDL0311_nodvfs/unseen_partition/power/rand10/extratrees"
    oracle_file = "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/data/model_datasets/03112025_DL0207_0307_nonDL0311_nodvfs/mergecudaDL_nodvfs_throughput_total_labels_comb2_labels.csv"
    pred_xput_file_common_name = "pred_separate_throughputpower_regression.csv"
    pred_power_file_common_name = "pred_power_regression.csv"

    temporal_file = None
    powercap_limit_100_file = None
    target = "xput_per_power" #EDP, xput_per_joule
    weight = 1# convert to kj if weight=1000
    is_plot = False #plot each workload
    dvfs_oracle = False #whether oracle is dvfs enabled
    output_base_dir = f"./{target}_unseen_pred_vs_baselines_nodvfs/"
    rm_100partition=True #remove 100 partition
    save_csv = True
    #output_base_dir = f"./test"

    analyze_predictions_target_with_baselines(predict_xput_dir=predict_xput_dir, predict_power_dir=predict_power_dir, dvfs_file=oracle_file, 
                                              temporal_file=temporal_file, powercap_limit_file=powercap_limit_100_file, 
                                              pred_xput_file_common_name=pred_xput_file_common_name, pred_power_file_common_name=pred_power_file_common_name, 
                                              target=target , output_base_name=output_base_dir, 
                                              weight=weight, dvfs_oracle=dvfs_oracle,rm_100partition=rm_100partition , is_plot=is_plot, save_csv=save_csv)

    predict_xput_dir = "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/data/experiment_inputs/colocations/output/run03112025_DL0207_0307_nonDL0311_nodvfs/seen_partition/crossvalid/throughput/trainratio_/rand10"
    predict_power_dir = "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/data/experiment_inputs/colocations/output/run03112025_DL0207_0307_nonDL0311_nodvfs/seen_partition/crossvalid/power/trainratio_/rand10"
    oracle_file = "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/data/model_datasets/03112025_DL0207_0307_nonDL0311_nodvfs/mergecudaDL_nodvfs_throughput_total_labels_comb2_labels.csv"
    pred_xput_file_common_name = "pred_separate_throughputpower_regression.csv"
    pred_power_file_common_name = "pred_power_regression.csv"

    temporal_file = None
    powercap_limit_100_file = None
    target = "xput_per_power" #EDP, xput_per_joule
    weight = 1# convert to kj if weight=1000
    is_plot = False #plot each workload
    dvfs_oracle = False #whether oracle is dvfs enabled
    output_base_dir = f"./{target}_seen_crossvalidate_pred_vs_baselines_nodvfs"
    rm_100partition=True #remove 100 partition
    save_csv = True
    throughput_model = "extratrees"
    power_model = "extratrees"
    #all_models = ["extratrees", "RF", "AutoML", "linear"]
    all_models = ["RF", "AutoML"]
    num_testsets = [i for i in range(1,10)]
    #num_testsets = [2]
    #for model_type in all_models:
    #    throughput_model = model_type
    #    power_model = model_type
    #    output_per_model = os.path.join(output_base_dir, model_type)
    #    for num_testset in num_testsets:
    #        output_dir = f"{output_per_model}/fold_{num_testset}"
    #        print(f"Analyzing for num_testset={num_testset}, output_dir={output_dir}")
    #        analyze_predictions_crossvalidate_target_with_baselines(
    #            predict_xput_dir=predict_xput_dir,
    #            predict_power_dir=predict_power_dir,
    #            dvfs_file=oracle_file,
    #            temporal_file=temporal_file,
    #            powercap_limit_file=powercap_limit_100_file,
    #            pred_xput_file_common_name=pred_xput_file_common_name,
    #            pred_power_file_common_name=pred_power_file_common_name,
    #            target=target,
    #            output_base_name=output_dir,
    #            throughput_model=throughput_model,
    #            power_model=power_model,
    #            num_testset=num_testset,  # Pass num_testset to filter
    #            weight=weight,
    #            dvfs_oracle=dvfs_oracle,
    #            rm_100partition=rm_100partition,
    #            is_plot=is_plot,
    #            save_csv=save_csv
    #        )
    
    

    