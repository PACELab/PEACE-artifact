

#TRAIN and TEST on different directories. ex. train with nodvfs, test on dvfs
###################
#COMMON ARGUMENTS
n_occur=0
rm_100partitions="True" #AICROSSROAD: remove 100,100 partition
#datapath="/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/tests/mps/multiinstance/dataset/02072025_noweight/with_energy_duration/0307_data1229_mergecudaDL_throughput_total_labels_comb2.csv"
#root output of partitioned dataset
#root_dataset_output_dir="/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/tests/mps/multiinstance/dataset/02072025_noweight/with_energy_duration/seen_partition"
#result_output_dir="output/run03112025_DL0207_0307_nonDL0311_nodvfs"
#train / test dataset location
#root_train_dataset_filename="0505_FREQ900_throughput_total_labels_comb2.csv"
#FREQ 1530 only
root_train_dataset_filename="mergecudaDL_nodvfs_throughput_total_labels_comb2.csv"

root_test_dataset_filename="0505_FREQ1530_dvfs_throughput_total_labels_comb2.csv"

#root train dataset dir
#root_train_dataset_dir="../freq_scaling/dataset/05052025_FREQ900_nodvfs"
#FREQ1530 only
root_train_dataset_dir="../freq_scaling/dataset/03112025_DL0207_0307_nonDL0311_nodvfs_FREQ1530"

#root test dataset dir
root_test_dataset_dir="../freq_scaling/dataset/05052025_FREQ1530_DL_powercap60_dvfs"

#where train model is located
#root_train_model_dir="../freq_scaling/output/05052025_FREQ900_nodvfs_fullDL"
#FREQ1530 only
root_train_model_dir="../freq_scaling/output/run03112025_DL0207_0307_nonDL0311_nodvfs_FREQ1530"

#where test results will be placed 
root_test_output_dir="../freq_scaling/output/05052025_train_FREQ1530_test_powercap60_dvfs_fullDL"
#####################

#for n_occur in "${n_occurs[@]}"; do
#bash unseen_partitions.sh KACE 0 $n_occur output/run0730_KACE_batch2-8  fefe  predict
#bash unseen_partitions.sh hotcloud 0 $n_occur output/run0730_KACE_batch2-8  train  predict
#bash unseen_partitions.sh AutoML 0 $n_occur output/run0730_KACE_batch2-8  train  predict
#bash unseen_partitions.sh NN 0 $n_occur output/run0730_KACE_batch2-8  train  predict
#bash unseen_partitions.sh RF 0 $n_occur output/run0730_KACE_batch2-8  train  predict
#bash unseen_partitions.sh KACE 0 $n_occur output/run0906_comb4_KACE_batch2-8  train  predict
#bash unseen_partitions.sh hotcloud 0 $n_occur output/run0906_comb4_KACE_batch2-8  train  predict
#bash unseen_partitions.sh KACE 0 $n_occur output/run0912_comb2_KACE_gpt2xl train  predict
#bash unseen_partitions.sh AutoML 0 $n_occur output/run02062025_all_comb2_batch2 train predict power
#bash unseen_partitions.sh RF 0 $n_occur output/run02112025_comb2_batch2 dede predict power
#bash unseen_partitions.sh AutoML 0 $n_occur output/run02112025_comb2_batch2 train predict throughput
#bash unseen_partitions.sh linear 0 $n_occur output/run02072025_noweights_comb2_batch2 train predict power
#bash unseen_partitions.sh AutoML 0 $n_occur "$result_output_dir" train predict throughput "$rm_100partitions"
#bash unseen_partitions.sh extratrees 0 $n_occur output/run03072025_data1229_mergecudaDL_comb2 train predict power "$rm_100partitions"
#bash unseen_partitions_test_dvfs.sh extratrees 0 $n_occur dede predict throughput "$rm_100partitions" "$root_train_dataset_filename" "$root_test_dataset_filename" "$root_train_dataset_dir" "$root_test_dataset_dir" "$root_train_model_dir" "$root_test_output_dir" 
#bash unseen_partitions_test_dvfs.sh extratrees 0 $n_occur train predict power "$rm_100partitions" "$root_train_dataset_filename" "$root_test_dataset_filename" "$root_train_dataset_dir" "$root_test_dataset_dir" "$root_train_model_dir" "$root_test_output_dir" 
bash unseen_partitions_test_dvfs.sh extratrees 0 $n_occur train predict throughput "$rm_100partitions" "$root_train_dataset_filename" "$root_test_dataset_filename" "$root_train_dataset_dir" "$root_test_dataset_dir" "$root_train_model_dir" "$root_test_output_dir" 

#bash unseen_partitions.sh RF 0 $n_occur "$result_output_dir" train predict power "$rm_100partitions"

#bash unseen_partitions.sh RF 0 $n_occur output/run02202025_powercap100_perworkload_noweights_comb2_batch2 train predict power
#done