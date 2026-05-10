
log_dir=$1
mkdir -p $log_dir
#log poisson arrival to $log_dir/arrival_baselineLS100.log and stdout with tee

python poisson_arrival.py  --ip 127.0.0.1 -f IAT/0306_IAT_100reqs_lambd0.05.json 2>&1 | tee $log_dir/arrival_baselineLS100.log &
python vllm_logging.py $log_dir > "$log_dir/LSmetric_baseline_MPS100.log" 
wait
