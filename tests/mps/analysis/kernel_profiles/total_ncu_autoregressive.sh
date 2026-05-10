#!/bin/bash
 job_type=("autoregressive:gpt2-xl")
 worktype="inf"
 stage="steps"
 #stage="prefill_and_decode1token_request"
 batch_size=(20 100 214)
 output_dir=$1
 mkdir -p $output_dir
 #mkd dir
 mkdir -p source/ncu
 for job in "${job_type[@]}"
 do
     # Parse job string into TASK and JOB
     IFS=':' read -ra ADDR <<< "$job"
     TASK="${ADDR[0]}"
     JOB="${ADDR[1]}"

     # Extract pure model name
     if [[ "$JOB" == *"/"* ]]; then
         pure_model="${JOB#*/}"
     else
         pure_model="$JOB"
     fi

     echo "model without slash is $pure_model"
     echo "$TASK"
     echo "$JOB"


     for i in "${batch_size[@]}"
     do
         full_name="${pure_model}_batch${i}-${worktype}_${stage}"
         cd /home/cc/mlProfiler/workloads/inference

         echo "running ncu for $full_name"
         sudo /usr/local/cuda/bin/ncu --nvtx --nvtx-include "${stage}10/" --call-stack --target-processes all --verbose --set full --csv -o /home/cc/mlProfiler/tests/mps/analysis/kernel_profiles/source/ncu/"$full_name" /home/cc/miniconda3/bin/python "${TASK}-inference.py" --model_name "$JOB" --batch_size 1 --output_length "$i" --profile_nstep 12 --profile
         echo "generating profile $full_name for batch size $i"
         cd /home/cc/mlProfiler/tests/mps/analysis/kernel_profiles
         /usr/local/cuda/bin/ncu -i source/ncu/"$full_name".ncu-rep --csv --print-details all > source/ncu/"$full_name"_ncu.csv
         python ncu_orion.py --input_file source/ncu/"$full_name"_ncu.csv --results_dir $output_dir --job_type "$full_name" > tmp/"$full_name"_ncu.log
     done
 done