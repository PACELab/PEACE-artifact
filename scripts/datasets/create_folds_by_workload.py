"""Generate deterministic workload folds for cross-validation splits.

This script reads a dataset containing workload columns (e.g., ``workload1``,
``workload2``) and produces canonical fold assignments that can be reused across
different datasets sharing the same workload universe.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

NUM_FOLDS = 10

# Mapping copied from run_train_splits_mpsthreads_crossvalidate.sh to ensure the
# generated artifacts match the driving experiments.
TEST_FOLDS_PER_NUM: Dict[int, List[str]] = {
    1: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    2: ["1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9", "9-10", "10-1"],
    3: ["1-3", "2-4", "3-5", "4-6", "5-7", "6-8", "7-9", "8-10", "9-1", "10-2"],
    4: ["1-4", "2-5", "3-6", "4-7", "5-8", "6-9", "7-10", "8-1", "9-2", "10-3"],
    5: ["1-5", "2-6", "3-7", "4-8", "5-9", "6-10", "7-1", "8-2", "9-3", "10-4"],
    6: ["1-6", "2-7", "3-8", "4-9", "5-10", "6-1", "7-2", "8-3", "9-4", "10-5"],
    7: ["1-7", "2-8", "3-9", "4-10", "5-1", "6-2", "7-3", "8-4", "9-5", "10-6"],
    8: ["1-8", "2-9", "3-10", "4-1", "5-2", "6-3", "7-4", "8-5", "9-6", "10-7"],
    9: ["1-9", "2-10", "3-1", "4-2", "5-3", "6-4", "7-5", "8-6", "9-7", "10-8"],
}


def parse_fold_sequence(label: str, num_folds: int = NUM_FOLDS) -> List[int]:
    """Expand a fold shorthand (e.g., ``\"2-4\"``) into explicit fold indices."""
    if "-" not in label:
        return [int(label)]
    start_str, end_str = label.split("-")
    start, end = int(start_str), int(end_str)
    folds = [start]
    while folds[-1] != end:
        next_fold = (folds[-1] % num_folds) + 1
        folds.append(next_fold)
    return folds


def assign_folds(
    workloads: pd.DataFrame, seed: int, num_folds: int = NUM_FOLDS
) -> Dict[int, pd.DataFrame]:
    """Shuffle and partition workloads into ``num_folds`` chunks."""
    shuffled = workloads.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    indices = np.linspace(0, len(shuffled), num_folds + 1, dtype=int)

    fold_map: Dict[int, pd.DataFrame] = {}
    for fold_idx in range(num_folds):
        start, end = indices[fold_idx], indices[fold_idx + 1]
        fold_map[fold_idx + 1] = shuffled.iloc[start:end].reset_index(drop=True)
    return fold_map


def concat_folds(fold_map: Dict[int, pd.DataFrame], fold_ids: Iterable[int]) -> pd.DataFrame:
    """Concatenate workloads for the specified ``fold_ids``."""
    selected = [fold_map[fid] for fid in fold_ids if not fold_map[fid].empty]
    if not selected:
        return pd.DataFrame(columns=fold_map[next(iter(fold_map))].columns)
    return pd.concat(selected, ignore_index=True)


def save_fold(df: pd.DataFrame, path: Path) -> None:
    """Persist workloads to CSV with parent directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} workload combinations to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create canonical workload folds for multi-instance experiments."
    )
    parser.add_argument(
        "--workload_list_path",
        required=True,
        help="CSV containing workload columns (workload1..workloadN).",
    )
    parser.add_argument(
        "--fold_workloads_output_path",
        required=True,
        help="Directory where fold workload CSVs will be written.",
    )
    parser.add_argument(
        "--n_combination",
        type=int,
        required=True,
        help="Number of workloads per sample (determines workload column count).",
    )
    parser.add_argument(
        "--rdseed",
        type=int,
        default=10,
        help="Random seed used to shuffle workload combinations.",
    )
    args = parser.parse_args()

    workload_cols = [f"workload{i}" for i in range(1, args.n_combination + 1)]

    df = pd.read_csv(args.workload_list_path)
    missing_cols = [col for col in workload_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Expected workload columns {workload_cols}, but missing {missing_cols}"
        )

    unique_workloads = df[workload_cols].drop_duplicates().reset_index(drop=True)
    print(f"Found {len(unique_workloads)} unique workload combinations.")

    fold_map = assign_folds(unique_workloads, seed=args.rdseed)

    base_output_dir = Path(args.fold_workloads_output_path) / f"rand{args.rdseed}"

    # Save individual folds for reuse.
    for fold_id, fold_df in fold_map.items():
        save_fold(fold_df, base_output_dir / f"fold_{fold_id}.csv")

    # Save aggregated fold selections matching TEST_FOLDS_PER_NUM.
    for num_test_fold, ranges in TEST_FOLDS_PER_NUM.items():
        for range_label in ranges:
            fold_ids = parse_fold_sequence(range_label)
            combined = concat_folds(fold_map, fold_ids)
            filename = f"fold_{num_test_fold}_test_{range_label}.csv"
            save_fold(combined, base_output_dir / filename)


if __name__ == "__main__":
    main()
