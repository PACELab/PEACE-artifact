# multiinstance predictions 

### get kernel data
```
python scripts/datasets/get_all_combinations.py kernel_dir num_combinations output_prefix
```
### get stage2data
```
cd "$PEACE_ROOT/data/colocations"
python stage2.py [outname_prefix]
```
### Construct stage1 data integrate kernel data, stage2Data, and baseline data
```
#CHECK OUTNAME,KERNELFILE, modeltype first!!!!!!
#modeltype = KACE or hotcloud 
python main.py  --test_file output/dataset/testing_set.csv --train_file output/dataset/training_set.csv  -t 100 --processStage1Data -sd data/colocations/0730_share3_batch2-8_share_steps_stage2.csv --pred_multiinstance -comb 3 --modeltype KACE
```

## predict with automative scripts 
### predict all throughput without partitions
```
#CHANGE predict flags  to predict_all_throughput
bash run_train_splits.sh 
```
### Option1: predict throughput with partitions of seen workloads
```
#CHANGE predict flags to predict_by_workload
bash run_train_splits.sh 
```

### Option2: predict unseen partition -  partition data , run both train and prediction
```
#CHANGE n_combination, root_dataset_dir, full_data_path in unseen_partitions.sh 
cd data/experiment_inputs/colocations
bash run_unseen_partitions.sh
```

## Unit test - data split, train, test
### split train test
```

python splitData.py -d $full_data_path \
             --train_file  $dataset_dir/$train_file \
             --test_file $dataset_dir/$test_file \
            -rd $seed --train_ratio $n_occur
```
### train
```
python main.py  --test_file data/model_datasets/testing.csv --train_file data/model_datasets/training_set.csv  -t 100 --train -corr 0 -rd 10 --modeltype KACE -comb 3 --output_dir data/experiment_inputs/colocations/model
```

### predict All acc
```
python main.py  --test_file data/model_datasets/testing.csv --train_file data/model_datasets/training_set.csv  -t 100 --HPworkload "" --model data/experiment_inputs/colocations/model/30-7-2024_19\:02\:04_KACE_model-corr0.0_datard10_excol.pkl  -corr 0 -rd 10 --modeltype KACE -comb 3 --output_dir output/mutiinstance/ --rulebase --pred_multiinstance
```