import pandas as pd
import random

# Updated workload lists
workload1_inference = [
    "whisper-large-v2",
    "mobilenet_v2_1.0_224",
    "bert-base-cased",
    "vit-base-patch16-224",
    "resnet-50",
    "wav2vec2-base-960h"
]
workload1_train = [
    "bert-base-cased",
    "mobilenet",
    "vit_h_14",
    "albert-base-v2",
    "resnet-50"
]

# List of workload2 options
workload2_options = ['cudaTensorCoreGemm', 'fastWalshTransform', 'reductionMultiBlockCG',
                     'transpose', 'BlackScholes', 'sortingNetworks']

# Column headers for the table
columns = [
    "workload1", "idx1", "workload2", "idx2",
    "w1_Threads", "w1_Compute (SM) Throughput", "w1_DRAM Throughput", 
    "w1_Memory Throughput", "w1_Registers", "w1_Static Shared Memory",
    "w2_Threads", "w2_Compute (SM) Throughput", "w2_DRAM Throughput",
    "w2_Memory Throughput", "w2_Registers", "w2_Static Shared Memory"
]

# Generate all combinations
data = []
for workload2 in workload2_options:
    for workload1 in workload1_inference + workload1_train:
        idx1 = random.randint(1, 50)
        idx2 = random.randint(1, 50)
        w1_threads = random.uniform(100000, 200000)
        w1_compute = random.uniform(40, 80)
        w1_dram = random.uniform(30, 50)
        w1_memory = random.uniform(60, 80)
        w1_registers = random.uniform(9000000, 12000000)
        w1_static_shared_memory = random.uniform(6000, 9000)
        w2_threads = random.uniform(300000, 500000)
        w2_compute = random.uniform(50, 90)
        w2_dram = random.uniform(20, 50)
        w2_memory = random.uniform(60, 80)
        w2_registers = random.uniform(10000000, 15000000)
        w2_static_shared_memory = random.uniform(8000, 12000)
        data.append([
            workload1 + "_batch2-" + ("inf" if workload1 in workload1_inference else "train"),
            idx1, workload2 + "_batch2-samples", idx2,
            w1_threads, w1_compute, w1_dram, w1_memory, w1_registers, w1_static_shared_memory,
            w2_threads, w2_compute, w2_dram, w2_memory, w2_registers, w2_static_shared_memory
        ])

# Create DataFrame
df = pd.DataFrame(data, columns=columns)

# Save to a CSV file
csv_path = "./workload_combinations_full.csv"
df.to_csv(csv_path, index=False)

# Print a confirmation
print(f"CSV file saved at {csv_path}")
