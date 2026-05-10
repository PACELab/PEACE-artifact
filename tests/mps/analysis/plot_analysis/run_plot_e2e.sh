#!/usr/bin/env bash

# Name this file "run_plots.sh" (for example) and make it executable:
#   chmod +x run_plots.sh
# Then run:
#   ./run_plots.sh

modeltypes=("linear" "AutoML")
workloads=("whisper-large-v2_batch2-inf" "mobilenet_batch2-train" "bert-base-cased_batch2-train")

for mt in "${modeltypes[@]}"; do
  for w in "${workloads[@]}"; do
    
    csv_path="tests/mps/multiinstance/output/run02112025_comb2_batch2/unseen_partition/e2e/${mt}/${w}/e2e_results.csv"
    
    echo "Plotting CSV: ${csv_path}"
    
    # Call the Python script
    python plot_e2e.py \
      --csv_file "${csv_path}" \
      --target_workload "${w}" \
      --modeltype "${mt}" \
      --output_dir "./plots"
      
    echo "Done plotting for model=${mt}, workload=${w}"
    echo
  done
done
