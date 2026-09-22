#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/paths.sh"
cd "$REPO_ROOT"

#n_occurs=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)
#n_occur not used in unseen
###################
#COMMON ARGUMENTS
n_occur=0
rm_100partitions="True" #AICROSSROAD: remove 100,100 partition
n_combination="2" #combination number for unseen partitions, 2 or 3
#datapath="$REPO_ROOT/data/model_datasets/02072025_noweight/with_energy_duration/0307_data1229_mergecudaDL_throughput_total_labels_comb2.csv"
#root output of partitioned dataset
#root_dataset_output_dir="$REPO_ROOT/data/model_datasets/02072025_noweight/with_energy_duration/seen_partition"
#300 remerge
#result_output_dir="$REPO_ROOT/artifacts/predictions/05052025_FREQ300_mergecudaDL_nodvfs_remerge"
#900 remerge
#result_output_dir="$REPO_ROOT/artifacts/predictions/05052025_FREQ900_mergecudaDL_nodvfs_remerge"
#1530 remerge
result_output_dir="$REPO_ROOT/artifacts/predictions/09152025DL_0311nonDL_FREQ1530_mergecudaDL_nodvfs_remerge"

########################################################
#[SOCC'26 Rebuttal ONLY - triton kernel data]
#tritonkernel still use original 1530 metrics. not triton baseline metrics.
#result_output_dir="$REPO_ROOT/artifacts/predictions/04132026_triton_FREQ1530_DL_nodvfs_BASELINE_NOTRITON"
#triton kernle with triton baseline metrics
#result_output_dir="$REPO_ROOT/artifacts/predictions/04132026_triton_FREQ1530_DL_nodvfs_basemetric_TRITON"
########################################################
#COMB3 
#freq 300 
#result_output_dir="$REPO_ROOT/artifacts/predictions/09152025_freq300_DL_comb3"
#freq900
#result_output_dir="$REPO_ROOT/artifacts/predictions/09152025_freq900_DL_comb3"
#freq1530
#result_output_dir="$REPO_ROOT/artifacts/predictions/09152025_freq1530_DL_comb3"
#####################

#for n_occur in "${n_occurs[@]}"; do
#bash "$SCRIPT_DIR/unseen_partitions.sh" KACE 0 $n_occur output/run0730_KACE_batch2-8  fefe  predict
#bash "$SCRIPT_DIR/unseen_partitions.sh" hotcloud 0 $n_occur output/run0730_KACE_batch2-8  train  predict
#bash "$SCRIPT_DIR/unseen_partitions.sh" AutoML 0 $n_occur output/run0730_KACE_batch2-8  train  predict
#bash "$SCRIPT_DIR/unseen_partitions.sh" NN 0 $n_occur output/run0730_KACE_batch2-8  train  predict
#bash "$SCRIPT_DIR/unseen_partitions.sh" RF 0 $n_occur output/run0730_KACE_batch2-8  train  predict
#bash "$SCRIPT_DIR/unseen_partitions.sh" KACE 0 $n_occur output/run0906_comb4_KACE_batch2-8  train  predict
#bash "$SCRIPT_DIR/unseen_partitions.sh" hotcloud 0 $n_occur output/run0906_comb4_KACE_batch2-8  train  predict
#bash "$SCRIPT_DIR/unseen_partitions.sh" KACE 0 $n_occur output/run0912_comb2_KACE_gpt2xl train  predict
#bash "$SCRIPT_DIR/unseen_partitions.sh" AutoML 0 $n_occur output/run02062025_all_comb2_batch2 train predict power
#bash "$SCRIPT_DIR/unseen_partitions.sh" RF 0 $n_occur output/run02112025_comb2_batch2 dede predict power
#bash "$SCRIPT_DIR/unseen_partitions.sh" AutoML 0 $n_occur output/run02112025_comb2_batch2 train predict throughput
#bash "$SCRIPT_DIR/unseen_partitions.sh" linear 0 $n_occur output/run02072025_noweights_comb2_batch2 train predict power
#bash "$SCRIPT_DIR/unseen_partitions.sh" AutoML 0 $n_occur "$result_output_dir" train predict throughput "$rm_100partitions"
#bash "$SCRIPT_DIR/unseen_partitions.sh" extratrees 0 $n_occur output/run03072025_data1229_mergecudaDL_comb2 train predict power "$rm_100partitions"
bash "$SCRIPT_DIR/unseen_partitions.sh" extratrees 0 $n_occur "$result_output_dir" train predict power "$rm_100partitions" "$n_combination"
bash "$SCRIPT_DIR/unseen_partitions.sh" extratrees 0 $n_occur "$result_output_dir" train predict throughput "$rm_100partitions" "$n_combination"

#bash "$SCRIPT_DIR/unseen_partitions.sh" RF 0 $n_occur "$result_output_dir" train predict power "$rm_100partitions"

#bash "$SCRIPT_DIR/unseen_partitions.sh" RF 0 $n_occur output/run02202025_powercap100_perworkload_noweights_comb2_batch2 train predict power
#done