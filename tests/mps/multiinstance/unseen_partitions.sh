#bash rand_run_main.sh /Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/output/trained_models/rand_shuffles_corr0_excol_sharedmem/8-7-2024_22\:35\:40_LinearRegression_model-corr0.0_datard10_excol_Static_Shared_Memory.pkl  0 output/dataset/MPS100/rand10/testing_set.csv output/dataset/MPS100/rand10/training_set.csv

#mkdir output/MPS100
#bash rand_run_main.sh /Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/output/trained_models/rand_shuffles_corr0_excol_sharedmem/8-7-2024_22\:40\:47_LinearRegression_model-corr0.0_datard50_excol_Static_Shared_Memory.pkl 0  output/dataset/MPS100/rand50/testing_set.csv output/dataset/MPS100/rand50/training_set.csv
#mv output/MPS100 output/rand_run_nosharedmem/rand50
modeltype=$1
correlation=$2
n_occur=$3
outdir=$4
istrain=$5
ispredict=$6
throughput_or_power_task=$7
rm_100partitions=$8
n_combination=$9

#check if model  and correlation is provided
if [ -z "$correlation" ]
then
    echo "Correlation is not provided"
    exit 1
fi
if [ -z "$modeltype" ]
then
    echo "Model is not provided"
    exit 1
fi
#model could only be KACE or hotcloud, AutoML, NN, RF
#check if $throughput_or_power_task == throughput or == power. If not, exit



#if [ "$modeltype" != "KACE" ] && [ "$modeltype" != "hotcloud" ] && [ "$modeltype" != "AutoML" ]
#then
#    echo "Model should be either KACE or hotcloud"
#    exit 1
#fi

#modify if needed:
############################################
#EXCOL="_Static Shared Memory"
EXCOL=""
#for splitting
#n_combination="3"
#root_dataset_dir="./dataset/02072025_noweight/0207_noweight/unseen_partition/$throughput_or_power_task"
#root_dataset_dir="./dataset/03112025_DL0207_0307_nonDL0311_nodvfs/unseen_partition/$throughput_or_power_task"
#FREQ1530 from 0311 workshop data
#root_dataset_dir="../freq_scaling/dataset/03112025_DL0207_0307_nonDL0311_nodvfs_FREQ1530/unseen_partition/$throughput_or_power_task"

#root_dataset_dir="../freq_scaling/dataset/05052025_FREQ900_mergecudaDL_nodvfs/unseen_partition/$throughput_or_power_task"
#root_dataset_dir="../freq_scaling/dataset/05052025_FREQ300_mergecudaDL_nodvfs/unseen_partition/$throughput_or_power_task"
#300 remerge
root_dataset_dir="../freq_scaling/dataset/05052025_FREQ300_mergecudaDL_nodvfs_remerge/unseen_partition/$throughput_or_power_task"
#900 remerge
root_dataset_dir="../freq_scaling/dataset/05052025_FREQ900_mergecudaDL_nodvfs_remerge/unseen_partition/$throughput_or_power_task"
#1530 remerge
root_dataset_dir="../freq_scaling/dataset/09152025DL_0311nonDL_FREQ1530_mergecudaDL_nodvfs_remerge/unseen_partition/$throughput_or_power_task"
########################################################
#[SOCC'26 Rebuttal ONLY - triton kernel data]
#tritonkernel still use original 1530 metrics. not triton baseline metrics.
#root_dataset_dir="../freq_scaling/dataset/04132026_triton_DL_socc26_basemetric_nontriton/unseen_partition/$throughput_or_power_task"
#triton kernle with triton baseline metrics

#root_dataset_dir="../freq_scaling/dataset/04132026_triton_DL_socc26_basemetric_TRITON/unseen_partition/$throughput_or_power_task"
########################################################
### COMB3/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/tests/mps/freq_scaling/dataset/09152025_freq300_DL_comb3/09152025_freq300_DL_comb3_throughput_total_labels_comb3.csv
#freq300
#root_dataset_dir="../freq_scaling/dataset/09152025_freq300_DL_comb3/unseen_partition/$throughput_or_power_task"
#freq900
#root_dataset_dir="../freq_scaling/dataset/09152025_freq900_DL_comb3/unseen_partition/$throughput_or_power_task"
#freq1530
#root_dataset_dir="../freq_scaling/dataset/09152025_freq1530_DL_comb3/unseen_partition/$throughput_or_power_task"

if [ "$modeltype" == "hotcloud" ]
then
    full_data_path="/Users/bing/Documents/mlProfiler/tests/mps/multiinstance/dataset/hotcloud/0907hotcloud_total_labels_comb4_batch2-8_targetMPS100.csv"
else
    #KACE
    #full_data_path="/Users/bing/Documents/mlProfiler/tests/mps/multiinstance/dataset/KACE/0909_total_labels_comb4_testbatch4_batch2-8_targetMPS100.csv"
    #full_data_path="/Users/bing/Documents/mlProfiler/tests/mps/multiinstance/dataset/KACE/0907_total_labels_comb4_batch2-8_targetMPS100.csv"
    
    if [ "$throughput_or_power_task" == "throughput" ]
    then
        #full_data_path="./dataset/02062025_batch2_all/throughputreg/0206_all_throughput_total_labels_comb2.csv"
        #full_data_path="./dataset/02072025_noweight/0207_noweight/0207throughputreg_throughput_total_labels_comb2.csv"
        #full_data_path="../freq_scaling/dataset/05052025_FREQ900_mergecudaDL_nodvfs/0505_FREQ900_nodvfs_throughput_total_labels_comb2.csv"
        #full_data_path="../freq_scaling/dataset/05052025_FREQ300_mergecudaDL_nodvfs/0505_FREQ300_nodvfs_throughput_total_labels_comb2.csv"
        #full_data_path="./dataset/03112025_DL0207_0307_nonDL0311_nodvfs/mergecudaDL_nodvfs_throughput_total_labels_comb2.csv"
        
        #comb 2
        #300 remerge
        #full_data_path="../freq_scaling/dataset/05052025_FREQ300_mergecudaDL_nodvfs_remerge/merged_total_labels_comb2.csv"
        #900 remerge
        #full_data_path="../freq_scaling/dataset/05052025_FREQ900_mergecudaDL_nodvfs_remerge/merged_total_labels_comb2.csv"
        #1530 remerge
        full_data_path="../freq_scaling/dataset/09152025DL_0311nonDL_FREQ1530_mergecudaDL_nodvfs_remerge/merged_total_labels_comb2.csv"

        ########################################################
        #[SOCC'26 Rebuttal ONLY - triton kernel data]
        #tritonkernel still use original 1530 metrics. not triton baseline metrics.
        #full_data_path="../freq_scaling/dataset/04132026_triton_DL_socc26_basemetric_nontriton/triton_DL_throughput_total_labels_comb2.csv"
        #triton kernle with triton baseline metrics
        #full_data_path="../freq_scaling/dataset/04132026_triton_DL_socc26_basemetric_TRITON/triton_DL_throughput_total_labels_comb2.csv"
        ########################################################
        
        #COMB3
        #freq300
        #full_data_path="../freq_scaling/dataset/09152025_freq300_DL_comb3/09152025_freq300_DL_comb3_throughput_total_labels_comb3.csv"
        #freq900
        #full_data_path="../freq_scaling/dataset/09152025_freq900_DL_comb3/09152025_freq900_DL_comb3_throughput_total_labels_comb3.csv"
        #freq1530
        #full_data_path="../freq_scaling/dataset/09152025_freq1530_DL_comb3/09152025_freq1530_DL_comb3_throughput_total_labels_comb3_sampled.csv"
    else
        #full_data_path="/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/tests/mps/freq_scaling/dataset/05052025_FREQ900_nodvfs/0505_FREQ900_throughput_total_labels_comb2.csv"
        #full_data_path="./dataset/02202025_powercap_per_workload/02202025_powercap_perworkload_DL_throughput_total_labels_comb2.csv"
        #full_data_path="../freq_scaling/dataset/05052025_FREQ900_mergecudaDL_nodvfs/0505_FREQ900_nodvfs_throughput_total_labels_comb2.csv"
        #full_data_path="../freq_scaling/dataset/05052025_FREQ300_mergecudaDL_nodvfs/0505_FREQ300_nodvfs_throughput_total_labels_comb2.csv"        
        #full_data_path="./dataset/03112025_DL0207_0307_nonDL0311_nodvfs/mergecudaDL_nodvfs_throughput_total_labels_comb2.csv"
        #300 remerge
        #full_data_path="../freq_scaling/dataset/05052025_FREQ300_mergecudaDL_nodvfs_remerge/merged_total_labels_comb2.csv"
        #900 remerge
        #full_data_path="../freq_scaling/dataset/05052025_FREQ900_mergecudaDL_nodvfs_remerge/merged_total_labels_comb2.csv"
        #1530 remerge
        full_data_path="../freq_scaling/dataset/09152025DL_0311nonDL_FREQ1530_mergecudaDL_nodvfs_remerge/merged_total_labels_comb2.csv"
        ########################################################
        #[SOCC'26 Rebuttal ONLY - triton kernel data]
        #tritonkernel still use original 1530 metrics. not triton baseline metrics.
        #full_data_path="../freq_scaling/dataset/04132026_triton_DL_socc26_basemetric_nontriton/triton_DL_throughput_total_labels_comb2.csv"
        #triton kernle with triton baseline metrics
        #full_data_path="../freq_scaling/dataset/04132026_triton_DL_socc26_basemetric_TRITON/triton_DL_throughput_total_labels_comb2.csv"
        ########################################################
        #comb3
        #full_data_path="../freq_scaling/dataset/09152025_freq300_DL_comb3/09152025_freq300_DL_comb3_throughput_total_labels_comb3.csv"
        #freq900
        #full_data_path="../freq_scaling/dataset/09152025_freq900_DL_comb3/09152025_freq900_DL_comb3_throughput_total_labels_comb3.csv"
        #freq1530
        #full_data_path="../freq_scaling/dataset/09152025_freq1530_DL_comb3/09152025_freq1530_DL_comb3_throughput_total_labels_comb3_sampled.csv"

    fi
fi

test_file="testing_set.csv"
train_file="training_set.csv"
targetMPS=100


#workloads=('whisper-large-v2_batch8-inf'
#'whisper-large-v2_batch2-inf'
#'bert-base-cased_batch8-inf'
#'vit-base-patch16-224_batch8-inf'
#'vit-base-patch16-224_batch2-inf'
#'wav2vec2-base-960h_batch2-inf'
#'wav2vec2-base-960h_batch8-inf'
#'bert-base-cased_batch2-inf'
#'vit_h_14_batch8-train'
#'bert-base-cased_batch8-train'
#'vit_h_14_batch2-train'
#'albert-base-v2_batch2-train'
#'albert-base-v2_batch8-train'
#'bert-base-cased_batch2-train'
#)

workloads=(
#'gpt2-xl_batch20-inf'
#'gpt2-xl_batch100-inf'
#'gpt2-xl_batch214-inf'
#'whisper-large-v2_batch8-inf'
'whisper-large-v2_batch2-inf'
##'bert-base-cased_batch8-inf'
##'vit-base-patch16-224_batch8-inf'
'vit-base-patch16-224_batch2-inf'
'wav2vec2-base-960h_batch2-inf'
##'wav2vec2-base-960h_batch8-inf'
'bert-base-cased_batch2-inf'
##'vit_h_14_batch8-train'
##'bert-base-cased_batch8-train'
'vit_h_14_batch2-train'
'albert-base-v2_batch2-train'
##'albert-base-v2_batch8-train'
'bert-base-cased_batch2-train'
'fastWalshTransform_batch2-cuda_samples'
'cudaTensorCoreGemm_batch2-cuda_samples'
'reductionMultiBlockCG_batch2-cuda_samples'
'transpose_batch2-cuda_samples'
'sortingNetworks_batch2-cuda_samples'
#'BlackScholes_batch2-cuda_samples'
'mobilenet_batch2-train'
'resnet-50_batch2-inf'
'resnet-50_batch2-train'
'mobilenet_v2_1.0_224_batch2-inf'
#[Socc'26 Rebuttal ONLY - triton kernel data]
#'albert-base-v2-triton_batch2-train'
#'mobilenet-triton_batch2-train'
#'resnet-50-triton_batch2-train'
#'resnet-50-triton_batch2-inf'
#'wav2vec2-base-960h-triton_batch2-inf'
#'bert-base-cased-triton_batch2-inf'
#'bert-base-cased-triton_batch2-train'
##'vit_h_14-triton_batch2-train'
##'whisper-large-v2-triton_batch2-inf'
#'mobilenet_v2_1.0_224-triton_batch2-inf'
#'vit-base-patch16-224-triton_batch2-inf'
)


############################################

#create random splits of different 





echo "generate random splits using seed 10 and train model"
#seeds=(10)
seeds=(10)
if [ "$throughput_or_power_task" == "throughput" ]
    then
        label_policy="separate_throughputpower_regression"
    else
        label_policy="power_regression"
fi
#seeds=(50)
#split and train if istrain is set
if [ "$istrain" == "train" ]
then
    for workload in "${workloads[@]}";
    do
        for seed in "${seeds[@]}"
        do
            echo "seed $seed"
            dataset_dir="$root_dataset_dir/rand${seed}/$modeltype/$workload"
            outdir_rand="$outdir/unseen_partition/${throughput_or_power_task}/rand${seed}/$modeltype/$workload"
            mkdir -p "$dataset_dir"
            mkdir -p "$outdir_rand"
            #if data is not generated, generate data
            if [ ! -f "$dataset_dir/$test_file" ] || [ ! -f "$dataset_dir/$train_file" ]
            then
                echo "generate data for $dataset_dir..."
                echo ""
                chmod +x "$dataset_dir"
                #echo "$full_data_path $workload $dataset_dir $n_combination"
                python ../../../output/dataset/MPS100/partition_exp/partition_data_by_workloadname.py \
                "$full_data_path" "$workload" "$dataset_dir" "$n_combination" "$rm_100partitions"
                #python main.py  --test_file $dataset_dir/$test_file \
                #--train_file  $dataset_dir/$train_file \
                #-rd $seed --targetMPS $targetMPS \
                #-d $full_data_path \
                #--n_occur $n_occur \
                #-c
                #echo "data saved in $dataset_dir"
            fi


            echo "train model with $dataset_dir/$train_file..."
            #labelpolicy - set by $throughput_or_power_task
            
            python ../../../main.py --test_file "$dataset_dir/$test_file" --train_file "$dataset_dir/$train_file" \
            -t $targetMPS -rd $seed  --train -corr $correlation \
            --output_dir "$outdir_rand" --modeltype $modeltype -comb $n_combination --label_policy $label_policy --debug
        done
    done
fi


#hotcloud
#test_file=tests/mps/analysis/baselines/hotcloud/testing_set_MPS100.csv
#train_file=tests/mps/analysis/baselines/hotcloud/training_set_MPS100.csv


if [ "$ispredict" == "predict" ]
then
############################################
#Predict throughput for different workloads
    echo "start predicting throughput for different workloads"



    for workload in "${workloads[@]}";
    do
        for seed in "${seeds[@]}";
        do

            echo "seed $seed"
            dataset_dir="$root_dataset_dir/rand${seed}/$modeltype/$workload"
            model_dir="$outdir/unseen_partition/${throughput_or_power_task}/rand${seed}/$modeltype/$workload"
            #model_dir="/Users/bing/Documents/mlProfiler/tests/mps/multiinstance/model"
            outdir_rand="$outdir/unseen_partition/${throughput_or_power_task}/rand${seed}/$modeltype/$workload"
            outdir_pred="$outdir_rand/predicts_by_workloads"
            #outdir_pred="$outdir_rand"
            mkdir -p "$outdir_pred"
            echo "search for model in $outdir_rand..."
            #find model
            if [ $modeltype != "AutoML" ]
            then
                model_file=$(find $model_dir -name "*$modeltype*.pkl")
            else
                model_file=$(find $model_dir -type f -name "*AutoML*" ! -name "*.txt")
            fi
            #model_file=$(find $outdir_rand -name "*$modeltype*.pkl")
            #raise error and exit if model is not found or multiple models are found
            if [ -z "$model_file" ]
            then
                echo "Model $modeltype not found in $outdir_rand"
                exit 1
            fi
            if [ $(echo $model_file | wc -l) -gt 1 ]
            then
                echo "Multiple models found in $outdir_rand"
                exit 1
            fi

            echo "using model $model_file to predict..."
            echo "using dataset  $dataset_dir/$test_file"
            #iterate  over the workloads
        
            
            python ../../../main.py  --test_file "$dataset_dir/$test_file" \
            --train_file  "$dataset_dir/$train_file" \
            --train_postprocess_file "$dataset_dir/X_train_postprocess_$label_policy.csv" \
            --HPworkload "" --targetMPS 100 \
            --model "$model_file" -corr $correlation \
            -rd $seed --output_dir "$outdir_pred" \
            --rulebase --modeltype $modeltype \
            --predAllacc -comb $n_combination --label_policy $label_policy --debug
                #--model $model -corr 0.2
                #no_exclusive_feat model
                #--model output/trained_models/26-6-2024_03:19:04linear_regression_model.pkl
                #with exclusive_throughput as only feat model
                #--model output/trained_models/2d6-6-2024_10:31:13linear_regression_model.pkl --correlation 0.8
                
                #--sharedThroughputData /Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/stage2/merged0505_0617_0520_0624_share_steps_stage2.csv
                
        

            #save  $test_file and $train_file and $model and correlation config in a file
            echo testfile=$dataset_dir/$test_file >  $outdir_pred/config.txt
            echo trainfile=$dataset_dir/$train_file >>  $outdir_pred/config.txt
            echo model=$model_file >>  $outdir_pred/config.txt
            echo correlation=$correlation >>  $outdir_pred/config.txt
            echo train_n_occur=$n_occur >>  $outdir_pred/config.txt
            #plot average throughput
            #python output/count_average.py   $outdir_pred $modeltype
            # Check if the Python script exited with a non-zero status
            if [ $? -ne 0 ]; then
                echo "An error occurred while running thecount_average.py script. Exiting..."
                echo "current output directory is $outdir_pred"
                # Optionally, exit the script if the error is critical
                exit 1
            fi
        done
    done
fi





