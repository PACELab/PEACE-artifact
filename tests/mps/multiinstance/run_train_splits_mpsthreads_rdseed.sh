
#n_occurs=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)
#n_occurs=(0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)
#n_occurs=(0.7 0.6 0.5 0.4 0.3 0.2 0.1)
rm_100partitions="True"
n_combinations="2"
#dataset path
datapath="/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/tests/mps/multiinstance/dataset/03112025_DL0207_0307_nonDL0311_nodvfs/mergecudaDL_nodvfs_throughput_total_labels_comb2.csv"

#root output of partitioned dataset
root_dataset_output_dir="/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/tests/mps/multiinstance/dataset/03112025_DL0207_0307_nonDL0311_nodvfs/seen_partition"
output_result_dir="output/run03112025_DL0207_0307_nonDL0311_nodvfs"
#throughput_or_power_task="throughput"
n_occurs=(0.1)
#n_occurs=(8 9 10)
#n_occurs=(4 5 6 7)
for n_occur in "${n_occurs[@]}"; do


    #bash rand_run_allworkloads.sh AutoML  0 $n_occur output/run0709_KACE_hotcloud  fe  predict
    #bash rand_run_allworkloads.sh NN  0 $n_occur output/run0709_KACE_hotcloud  fef predict
    #bash rand_run_allworkloads.sh RF  0 $n_occur output/run0709_KACE_hotcloud  fefe  predict
    #bash train_splits.sh KACE  0 $n_occur output/run0730_KACE_batch2  train predict
    #bash train_splits_mpsthreads.sh linear 0 $n_occur output/test_run03072025_data1229_mergecudaDL_comb2  fef predict_all_throughput "${datapath}" "${root_dataset_output_dir}" "${rm_100partitions}" "${n_combinations}" throughput
    #bash train_splits_mpsthreads.sh RF 0 $n_occur output/test_run03072025_data1229_mergecudaDL_comb2  train predict_all_throughput "${datapath}" "${root_dataset_output_dir}" "${rm_100partitions}" "${n_combinations}" throughput
    bash train_splits_mpsthreads_rdseed.sh extratrees 0 $n_occur "$output_result_dir"  train predict_all_throughput "${datapath}" "${root_dataset_output_dir}" "${rm_100partitions}" "${n_combinations}" throughput
   # bash train_splits_mpsthreads_rdseed.sh AutoML 0 $n_occur output/test_run03072025_data1229_mergecudaDL_comb2  train predict_all_throughput "${datapath}" "${root_dataset_output_dir}" "${rm_100partitions}" "${n_combinations}" throughput
    #bash train_splits_mpsthreads.sh linear 0 $n_occur output/run03072025_data1229_mergecudaDL_comb2  train predict_all_throughput "${datapath}" "${root_dataset_output_dir}" "${rm_100partitions}" "${n_combinations}" power
    #bash train_splits_mpsthreads.sh RF 0 $n_occur output/run03072025_data1229_mergecudaDL_comb2  train predict_all_throughput "${datapath}" "${root_dataset_output_dir}" "${rm_100partitions}" "${n_combinations}" power
    #bash train_splits_mpsthreads.sh extratrees 0 $n_occur output/run03072025_data1229_mergecudaDL_comb2  train predict_all_throughput "${datapath}" "${root_dataset_output_dir}" "${rm_100partitions}" "${n_combinations}" power
    #bash train_splits_mpsthreads.sh AutoML 0 $n_occur output/run03072025_data1229_mergecudaDL_comb2  train predict_all_throughput "${datapath}" "${root_dataset_output_dir}" "${rm_100partitions}" "${n_combinations}" power
    #bash train_splits.sh NN 0 $n_occur output/run0730_KACE_batch2-8  wefhuehf predict
    #bash train_splits.sh RF 0 $n_occur output/run0730_KACE_batch2-8  fsefe predict
    #bash rand_run_allworkloads.sh hotcloud  0 $n_occur output/run0709_KACE_hotcloud  fefe  predict
done