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
datapath=$7
root_dataset_output_dir=$8
rm_100partitions=$9
n_combination=${10}
throughput_or_power_task=${11}

#print arguments
echo "modeltype: $modeltype"
echo "correlation: $correlation"
echo "n_occur: $n_occur"
echo "outdir: $outdir"
echo "istrain: $istrain"
echo "ispredict: $ispredict"
echo "datapath: $datapath"
echo "root_dataset_output_dir: $root_dataset_output_dir"
echo "rm_100partitions: $rm_100partitions"
echo "n_combination: $n_combination"
echo "throughput_or_power_task: $throughput_or_power_task"




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

if [ "$throughput_or_power_task" == "throughput" ]
    then
        label_policy="separate_throughputpower_regression"
    else
        label_policy="power_regression"
fi
#model could only be KACE or hotcloud, AutoML, NN, RF

#modify if needed:
############################################
#EXCOL="_Static Shared Memory"
EXCOL=""
#for splitting
#root dataset dir = root_dataset_output_dir + trainratio_n_occur
root_dataset_dir="${root_dataset_output_dir}/trainratio_${n_occur}"
if [ "$modeltype" == "hotcloud" ]
then
    #directly exit 1 with error since not implemented
    echo "hotcloud model not implemented"
    exit 1
    full_data_path="/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/baselines/hotcloud/hotcloud_combined_noexclu_throughput_targetMPS100.csv"
else
    #KACE
    full_data_path=$datapath
fi

test_file="testing_set.csv"
train_file="training_set.csv"
targetMPS=100

############################################

#create random splits of different 





echo "generate random splits using seed 10,20,30 and train model"
seeds=(10)
#seeds=(10 20 30 40 50)
#seeds=(40 50)
#split and train if istrain is set
if [ "$istrain" == "train" ]
then
    for seed in "${seeds[@]}"
    do
        echo "seed $seed"
        dataset_dir="$root_dataset_dir/rand${seed}"
        outdir_rand="$outdir/seen_partition/${throughput_or_power_task}/trainratio_${n_occur}/rand${seed}/$modeltype"
        mkdir -p "$dataset_dir"
        mkdir -p "$outdir_rand"
        #if data is not generated, generate data
        if [ ! -f "$dataset_dir/$test_file" ] || [ ! -f "$dataset_dir/$train_file" ]
        then
            echo "generate data for $dataset_dir..."
            python splitData.py -d "$full_data_path" \
             --train_file  "$dataset_dir/$train_file" \
             --test_file "$dataset_dir/$test_file" \
            -rd $seed --train_ratio $n_occur \
            --rm_100partitions $rm_100partitions \
            --split_workload_with_all_threads \
            --n_combination "$n_combination"
            
            echo "data saved in $dataset_dir"
        fi

        echo "train model with $dataset_dir/$train_file..."
        python ../../../main.py --test_file "$dataset_dir/$test_file" --train_file "$dataset_dir/$train_file" \
            -t $targetMPS -rd $seed  --train -corr $correlation \
            --output_dir "$outdir_rand" --modeltype $modeltype -comb $n_combination --label_policy $label_policy --debug
    done
fi


#hotcloud
#test_file=tests/mps/analysis/baselines/hotcloud/testing_set_MPS100.csv
#train_file=tests/mps/analysis/baselines/hotcloud/training_set_MPS100.csv


if [ "$ispredict" == "predict_by_workload" ]
then
############################################
#Predict throughput for different workloads
    echo "start predicting throughput for different workloads"


    workloads=( 'whisper-large-v2_batch16-inf' 'whisper-large-v2_batch8-inf' 
    'whisper-large-v2_batch2-inf' 'bert-base-cased_batch16-inf'
    'bert-base-cased_batch8-inf' 'vit-base-patch16-224_batch8-inf'
    'vit-base-patch16-224_batch2-inf' 'vit-base-patch16-224_batch16-inf'
    'wav2vec2-base-960h_batch2-inf' 'wav2vec2-base-960h_batch16-inf'
    'wav2vec2-base-960h_batch8-inf' 'bert-base-cased_batch2-inf'
    'vit_h_14_batch8-train' 'vit_h_14_batch16-train'
    'bert-base-cased_batch16-train' 'bert-base-cased_batch8-train'
    'vit_h_14_batch2-train' 'albert-base-v2_batch2-train'
    'albert-base-v2_batch8-train' 'albert-base-v2_batch16-train'
    'bert-base-cased_batch2-train')

    for workload in "${workloads[@]}";
    do
        for seed in "${seeds[@]}";
        do

            echo "seed $seed"
            dataset_dir="$root_dataset_dir/rand${seed}"
            model_dir="$outdir/seen_partition/${throughput_or_power_task}/trainratio_${n_occur}/rand${seed}/$modeltype"
            outdir_rand="$outdir/seen_partition/${throughput_or_power_task}/trainratio_${n_occur}/rand${seed}/$modeltype/$workload"
            #outdir_pred="$outdir_rand/predictthroughput"
            outdir_pred="$outdir_rand/predicts_by_workloads"
            mkdir -p $outdir_pred
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
            #iterate  over the workloads
        
            
            python ../../../main.py  --test_file $dataset_dir/$test_file \
            --train_file  $dataset_dir/$train_file \
            --HPworkload "" --targetMPS 100 \
            --model $model_file -corr $correlation \
            -rd $seed --output_dir $outdir_pred \
            --rulebase --modeltype $modeltype \
            --pred_multiinstance -comb 3
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

if [ "$ispredict" == "predict_all_throughput" ]
then
############################################
#Predict throughput for different workloads
    echo "start predicting throughput for different workloads"


        for seed in "${seeds[@]}";
        do

            echo "seed $seed"
            dataset_dir="$root_dataset_dir/rand${seed}"
            model_dir="$outdir/seen_partition/${throughput_or_power_task}/trainratio_${n_occur}/rand${seed}/$modeltype"
            outdir_rand="$outdir/seen_partition/${throughput_or_power_task}/trainratio_${n_occur}/rand${seed}/$modeltype"
            #outdir_pred="$outdir_rand/predictthroughput"
            outdir_pred="$outdir_rand/predictAllacc"
            mkdir -p "$outdir_pred"
            echo "search for model in $outdir_rand..."
            #find model
            if [ $modeltype != "AutoML" ]
            then
                model_file=$(find "$model_dir" -name "*$modeltype*.pkl")
            else
                model_file=$(find "$model_dir" -type f -name "*AutoML*" ! -name "*.txt")
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
            echo testfile=$dataset_dir/$test_file >  "$outdir_pred/config.txt"
            echo trainfile=$dataset_dir/$train_file >>  "$outdir_pred/config.txt"
            echo model=$model_file >>  "$outdir_pred/config.txt"
            echo correlation=$correlation >>  "$outdir_pred/config.txt"
            echo train_n_occur=$n_occur >>  "$outdir_pred/config.txt"
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

fi


