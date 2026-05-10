# Example usage
from loaddata import shuffle_split_and_save_csv_by_pairs
csv_path = "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/tests/mps/multiinstance/dataset/03112025_DL0207_0307_nonDL0311_nodvfs/mergecudaDL_nodvfs_throughput_total_labels_comb2.csv"  # Replace with your CSV file path
seed = 10
output_dir = "test"
shuffle_split_and_save_csv_by_pairs(csv_path, seed, output_dir, num_pieces=10)