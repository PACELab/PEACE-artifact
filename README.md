# mlProfiler

## local environment setup
```
//setup conda
conda create -n ml python=3.10
pip install -r requirements.txt

//enable wandb api
vim ~/.bashrc
//add following two lines within ~/.bashrc
export WANDB_API_KEY=[YOUR WANDB API KEY]
export WANDB_SILENT=true
```

## build docker image
```
docker build --build-arg API_KEY=[YOUR WANDB API KEY] -t [dockerhub_Username/image_name] .
docker push dockerhub_Username/image_name
```


## parse baseline metrics
```
cd tests/mps/freq_scaling/baseline_metrics
python parse_baseline_sysmetrics_freqscale.py ../../ccv100_logs/baseline_nonDL/ 0505_cudasample 2
```
## get sharedThroughputData from stage2.
```
#configure input path before execution

python stage2.py [output-prefix]
```
## OPTIONAL - merge share throughput data from different dates
```
#search for merge_share_files from stage2.ipynb

```
## merge stage2 label and feature files
### for fixed freq 300,900,1530 prediction models, make sure kernelfile in main.py is updated for baseline metrics
```
# check baseline, kernel data
python main.py -pd2 -power tests/mps/analysis/stage2/0307debug_nodvfs_DL_mobileinf_share_comb2_freqscale_power_avg_stage2.csv -sd tests/mps/analysis/stage2/0307debug_nodvfs_DL_mobileinf_share_comb2_freqscale_throughput_individual_avg.csv -duration tests/mps/analysis/stage2/0307debug_nodvfs_DL_mobileinf_share_comb2_freqscale_duration_avg_stage2.csv -energy tests/mps/analysis/stage2/0307debug_nodvfs_DL_mobileinf_share_comb2_freqscale_energy_avg_stage2.csv  -comb 2 -t 100  -mt threadclass --output_prefix 0307_debug_resnet50 --output_dir tests/mps/freq_scaling/dataset/0307_resnet50_nodvfs --label_policy separate_throughputpower_regression --baseline_file  ./tests/mps/freq_scaling/baseline_metrics/0502_FREQ900_baseline_metrics.csv
```
## find missing thread configurations with stage2 original testrun file and label file  
```
cd tests/mps/freq_scaling/dataset
python find_missing_threads.py  /home/cc/mlProfiler/tests/mps/multiinstance/nopower_nonDL_data.csv    /home/cc/mlProfiler/tests/mps/multiinstance/dataset/03112025_missingnonDLs_in1229_nodvfs/0311_missingnonDLs1229_nodvfs_throughput_total_labels_comb2_labels.csv
```

## spilt train / test set with workload occurence = 2
```
python main.py  --output_prefix 1228 --train_file tests/mps/multiinstance/dataset/threadclass/1228/train_set.csv --test_file tests/mps/multiinstance/dataset/threadclass/1228/test_set.csv --customsplit --nonSplitData  /home/cc/mlProfiler/tests/mps/multiinstance/dataset/threadclass/1228/1228_batchedthroughput_total_labels_comb2.csv  -comb 2 -t 100  -mt threadclass 

```

## stage2 train
```
python main.py  --output_prefix 0206 --train_file ./tests/mps/multiinstance/dataset/02062025_batch2/powerreg/predAllacc/train_set.csv --test_file ./tests/mps/multiinstance/dataset/02062025_batch2/powerreg/predAllacc/test_set.csv  -comb 2 -t 100  -mt linear --output_dir tests/mps/multiinstance/dataset/02062025_batch2/powerreg/predAllacc  --label_policy power_regression  --train 
```

## stage2 predict all test points(no output partitioning)
```
python main.py  --train_file tests/mps/multiinstance/dataset/threadclass/1229/train_set.csv --test_file tests/mps/multiinstance/dataset/threadclass/1229/test_set.csv   -comb 2 -t 100  -mt threadclass  --output_dir tests/mps/multiinstance/dataset/threadclass/1229 --model /home/cc/mlProfiler/tests/mps/multiinstance/dataset/threadclass/1229/30-12-2024_22:30:07_threadclass_model-corr0.2_datard30_excol.pkl --label_policy power_regression --predAllacc
```
## construct stage1 train/ test file 
```
#construct total dataset from kernel file, share throughput, and baseline
#Double check  getstage1Data() before executing
python main.py  --test_file output/dataset/testing_set.csv --train_file output/dataset/training_set.csv  -t 100 -pd -sd tests/mps/analysis/stage2/merge0624_infinf_traininf_traintrain_share_steps_stage2.csv


#split train / test set
python main.py  --test_file output/dataset/testing_set.csv --train_file output/dataset/training_set.csv  -t 100 -c -d [filepath of total dataset] -rd 10

#example
python main.py  --test_file output/dataset/testing_set.csv --train_file output/dataset/training_set.csv  -t 100 -c -d /Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/output/dataset/test_main_kernel_labels_targetMPS100.csv
```

## train
```
python main.py  --test_file output/dataset/testing_set.csv --train_file output/dataset/training_set.csv  -t 100 --train -corr 0

python main.py --test_file $dataset_dir/$test_file --train_file $dataset_dir/$train_file \
        -t $targetMPS -rd $seed  --train -corr $correlation \
        --output_dir $outdir_rand --modeltype $modeltype
```

## predict stage1 only
```
python main.py  --test_file output/dataset/testing_set.csv --train_file output/dataset/training_set.csv  -t 100 -m [model path] -w [HPworkload] 

#example
python main.py  --test_file output/dataset/testing_set.csv --train_file output/dataset/training_set.csv  -t 100   -m output/trained_models/26-6-2024_14:30:54linear_regression_model-corr0.2.pkl -w bert-base-cased_batch8-inf
```

## predict stage2
```
python main.py  --test_file output/dataset/testing_set.csv --train_file output/dataset/training_set.csv  -t 100 -m [model path] -w [HPworkload] --sharedThroughputData [filepath of sharedThroughputData]
```

## count average throughput
```
#run all colocations
bash main.sh
python output/count_average.py  [all thorughout json file]
#example
python output/count_average.py  output/MPS100/
```

## multiinstance predictions - with seen  workloads

### get kernel data
```
python tests/mps/multiinstance/get_all_combinations.py
```
### get stage2data
```
cd tests/mps/analysis/stage2
python stage2.py [outname_prefix]
```
### Construct stage1 data integrate kernel data, stage2Data, and baseline data
```
#CHECK OUTNAME,KERNELFILE first!!!!!!
python main.py  --test_file output/dataset/testing_set.csv --train_file output/dataset/training_set.csv  -t 100 --processStage1Data -sd /home/cc/mlProfiler/tests/mps/analysis/stage2/0730_share3_batch2-8_share_steps_stage2.csv --pred_multiinstance -comb 3 --modeltype KACE
```
### split train test
```

python splitData.py -d $full_data_path \
             --train_file  $dataset_dir/$train_file \
             --test_file $dataset_dir/$test_file \
            -rd $seed --train_ratio $n_occur
```
### train
```
python main.py  --test_file tests/mps/multiinstance/dataset/testing.csv --train_file tests/mps/multiinstance/dataset/training_set.csv  -t 100 --train -corr 0 -rd 10 --modeltype KACE -comb 3 --output_dir tests/mps/multiinstance/model
```

### predict All acc
```
python main.py  --test_file tests/mps/multiinstance/dataset/testing.csv --train_file tests/mps/multiinstance/dataset/training_set.csv  -t 100 --HPworkload "" --model tests/mps/multiinstance/model/30-7-2024_19\:02\:04_KACE_model-corr0.0_datard10_excol.pkl  -corr 0 -rd 10 --modeltype KACE -comb 3 --output_dir output/mutiinstance/ --rulebase --pred_multiinstance
```

### predict all throughput without partitions
```
#CHANGE predict flags  to predict_all_throughput
bash run_train_splits.sh 
```
### predict throughput with partitions of workloads
```
#CHANGE predict flags to predict_by_workload
bash run_train_splits.sh 
```

### predict unseen partition -  partition data , run both train and prediction
```
bash run_unseen_partitions.sh
```


### End to end
```
python main.py --train_file tests/mps/multiinstance/dataset/02062025_batch2_all/02062025_batch2_all/unseen_partition/throughput/rand10/RF/whisper-large-v2_batch2-inf/training_set.csv --test_file tests/mps/multiinstance/dataset/02062025_batch2_all/02062025_batch2_all/unseen_partition/throughput/rand10/RF/whisper-large-v2_batch2-inf/testing_set.csv   -comb 2 -t 100 --output_dir  tests/mps/multiinstance/output/run02062025_all_comb2_batch2/unseen_partition/e2e   --label_policy power_regression -mt RF --throughput_model tests/mps/multiinstance/output/run02062025_all_comb2_batch2/unseen_partition/throughput/rand10/RF/whisper-large-v2_batch2-inf/8-2-2025_22:34:37_RF_model-corr0.0_datard10_excol.pkl --power_model tests/mps/multiinstance/output/run02062025_all_comb2_batch2/unseen_partition/power/rand10/AutoML/whisper-large-v2_batch2-inf/GBM_5_AutoML_1_20250207_225826  --end_to_end  --debug
```


