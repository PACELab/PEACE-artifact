# Run MPS partitions
bash auto_multiinstance_freqscale.sh  ../ccv100_logs/test

# Run MPS partitions with dvfs enabled
bash auto_multiinstance_freqscale_gpumonitor_fixedpower.sh  ../ccv100_logs/test/dvfs

# check parsed data
# check parsed feature + label aggregation

# train/predict values (local)
cd /Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/tests/mps/multiinstance
bash run_train_splits_mpsthreads_crossvalidate.sh
