#parse share directory
#PARAMETER
#directory: the directory of the share test
#LStype: the type of the LS, train or inf
#BEtype: the type of the BE, train or inf
#LSpercent: the percentage of the LS
import pandas as pd
from collections import defaultdict
import sys
sys.path.append('../')
from parse_util import *
import sys


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("please provide the input_shared_log_dir  output prefix and num of combinations")
        print("python stage2.py inputfile output_prefix num_combinations [metric_type]")
        print("metric_type: 'throughput' (default) or 'latency'")
        exit(1)
    base_df = pd.read_csv("baseline_steps_stage2.csv")
    share_directory_path = sys.argv[1]
    output_prefix = sys.argv[2]
    N_COMB=int(sys.argv[3])
    # Optional fourth argument for metric type
    metric_type = sys.argv[4] if len(sys.argv) > 4 else "throughput"
    if metric_type not in ["throughput", "latency"]:
        print(f"Invalid metric_type: {metric_type}. Must be 'throughput' or 'latency'")
        exit(1)
    #share_directory_path = "/home/cc/mlProfiler/tests/mps/ccv100_logs/share_3work_batch2-8"
    #share_directory_path = "/home/cc/mlProfiler/tests/mps/multiinstance/ccv100/comb4_testbatch4_trainbatch2-8"
    #get all models from basedf workload column
    all_models = base_df["workload"].tolist()
    print(all_models)
    LStype, BEtype = "train", "train"
    share_file_steps, power_dict = defaultdict(dict), defaultdict(dict)
    #share_directory_path = "/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/cclogs/ccv100_0624_MPS100/sharetest/LStrain_batch2816_BEtrain_batch2816"
    #share_file_steps = get_share_avgStep(share_directory_path, LStype=LStype, BEtype=BEtype, all_models=all_models)
    #save_share_file_steps(share_file_steps=share_file_steps, filename=f"{sys.argv[1]}_{LStype}-{BEtype}_share_steps_stage2.csv")

    idle_power_dict = {
        300 : 31,
        400: 32,
        500: 33, 
        600: 34,
        700: 35,
        800: 35, 
        900: 36,
        1000: 42,
        1100: 44,
        1200: 48, 
        1300: 53,
        1400: 58,
        1500: 59,
        1530: 60,
        0: 50 #0 - dvfs deenabled with init freq of 1300
    }

    #share_directory_path = "/home/cc/mlProfiler/tests/mps/ccv100_logs/share_3work_batch2-8"
    #share_file_steps = get_multiworkloads_share_avgStep(share_directory_path, N_COMB, all_models)
    #share_file_steps, power_dict = get_multiworkloads_share_avgStep_with_power(share_directory_path, N_COMB)
    #file_steps, power_dict = get_multiworkloads_share_avgStep_with_freqscale(share_directory_path, N_COMB, idle_power=48)
    #avg_file_steps,  std_file_steps, avg_power_dict, std_power_dict = aggregate_runs(share_directory_path, N_COMB, idle_power=48, normalize_by_baseline_csv="/home/cc/mlProfiler/tests/mps/analysis/1018_baseline_metrics.csv", cost_savings=True)
    #avg_file_steps,  std_file_steps, avg_power_dict, std_power_dict = aggregate_runs(share_directory_path, N_COMB, idle_power=48, normalize_by_baseline_csv=None, cost_savings=False)
    avg_file_steps,  std_file_steps, avg_power_dict, std_power_dict, avg_duration_dict, std_duration_dict, avg_energy_dict, std_energy_dict = aggregate_runs_threaddir(share_directory_path, N_COMB, output_prefix, idle_power_dict=idle_power_dict, normalize_by_baseline_csv=None, cost_savings=False, metric_type=metric_type)



    #test_share = {('wav2vec2-base-960h_batch8-inf', 'wav2vec2-base-960h_batch2-inf', 'vit_h_14_batch8-train', 'vit_h_14_batch8-train'): {'w1_wav2vec2-base-960h_batch8-inf_MPS100': 50.87042635254873, 'w2_wav2vec2-base-960h_batch2-inf_MPS100': 19.328484606954266}}
    #save_multiinstance_share_file_steps_threads(share_file_steps=share_file_steps, filename=f"{output_prefix}_share_comb{N_COMB}_steps_stage2.csv", n_combination=N_COMB)
    #save_multiinstance_avg_power(power_dict=power_dict, filename=f"{output_prefix}_share_comb{N_COMB}_power_stage2.csv", n_combination=N_COMB)
    
    save_with_freqscale_sum_avg_std(avg_file_steps, f"{output_prefix}_share_comb{N_COMB}_freqscale_{metric_type}_sum_avg_stage2.csv", N_COMB)
    save_with_freqscale_sum_avg_std(std_file_steps, f"{output_prefix}_share_comb{N_COMB}_freqscale_{metric_type}_sum_std_stage2.csv", N_COMB)
    save_with_freqscale_sum_avg_std(avg_power_dict, f"{output_prefix}_share_comb{N_COMB}_freqscale_power_avg_stage2.csv", N_COMB)
    save_with_freqscale_sum_avg_std(std_power_dict, f"{output_prefix}_share_comb{N_COMB}_freqscale_power_std_stage2.csv", N_COMB)
    save_with_freqscale_sum_avg_std(avg_duration_dict, f"{output_prefix}_share_comb{N_COMB}_freqscale_duration_avg_stage2.csv", N_COMB)
    save_with_freqscale_sum_avg_std(std_duration_dict, f"{output_prefix}_share_comb{N_COMB}_freqscale_duration_std_stage2.csv", N_COMB)
    save_with_freqscale_sum_avg_std(avg_energy_dict, f"{output_prefix}_share_comb{N_COMB}_freqscale_energy_avg_stage2.csv", N_COMB)
    save_with_freqscale_sum_avg_std(std_energy_dict, f"{output_prefix}_share_comb{N_COMB}_freqscale_energy_std_stage2.csv", N_COMB)
    # Note: steps_count CSVs are now saved inside aggregate_runs_threaddir
    #save_with_freqscale(avg_file_steps, avg_power_dict, f"{output_prefix}_share_comb{N_COMB}_freqscale_stage2.csv", N_COMB)
    #save std
    #save_with_freqscale(std_file_steps, std_power_dict, f"{output_prefix}_share_comb{N_COMB}_freqscale_std_stage2.csv", N_COMB)

    #save input path, output prefix in txt
    with open(f"{output_prefix}_input_output.txt", "w") as f:
        f.write(f"input: {share_directory_path}\n")
        f.write(f"output: {output_prefix}\n")
        f.write(f"combination: {N_COMB}\n")
        f.write(f"metric_type: {metric_type}\n")
    
