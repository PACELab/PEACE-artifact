import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from utils.loaddata import train_test_split_multiinstance, train_test_split_multiinstance_with_all_threads, shuffle_split_and_save_csv_by_pairs
import argparse

#create arg parser for all parameters
parser = argparse.ArgumentParser()
parser.add_argument("--data","-d", type=str, help="Path to the dataset")
parser.add_argument("--train_file", type=str, help="Path to the training set")
parser.add_argument("--test_file", type=str, help="Path to the testing set")
parser.add_argument("--random_seed", "-rd", type=int, help="Random seed for the split")
parser.add_argument("--train_ratio", type=float, help="Ratio of the training set")
#stage2
parser.add_argument("--rm_100partitions", type=bool, default=False, help="Remove 100 partitions for all workloads")
parser.add_argument("--split_workload_with_all_threads", action="store_true", help="Split base on workload names, ensure all threads go with it")
parser.add_argument(
    "--n_combinations",
    "--n_combination",
    type=int,
    default=2,
    help="Number of combinations to split the data",
)
parser.add_argument(
    "--readbyworkloadlist",
    type=str,
    default="",
    help="Directory containing precomputed workload fold CSVs (e.g., .../rand10).",
)

#cross validation
parser.add_argument("--cross_validation", action="store_true", help="Perform cross validation")
parser.add_argument("--test_folds", type=str, help="Comma-separated list of test fold numbers (1-10)")
parser.add_argument("--train_folds", type=str, help="Comma-separated list of train fold numbers (1-10)")
args = parser.parse_args()
print(f"args: {args}")

if  args.split_workload_with_all_threads:

    train_test_split_multiinstance_with_all_threads(data=args.data, 
                                    train_outname=args.train_file, 
                                    test_outname=args.test_file, 
                                    random_seed=args.random_seed, train_ratio=args.train_ratio, rm_100partitions=args.rm_100partitions,
                                    n_combinations=args.n_combinations, 
                                    is_cross_validation=args.cross_validation,
                                    test_folds=args.test_folds,
                                    train_folds=args.train_folds,
                                    read_workload_list=args.readbyworkloadlist.strip() or None,
                                )
else:
    train_test_split_multiinstance(data=args.data, 
                                    train_outname=args.train_file, 
                                    test_outname=args.test_file, 
                                    random_seed=args.random_seed, train_ratio=args.train_ratio)
