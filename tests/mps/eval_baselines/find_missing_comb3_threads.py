"""Utilities to report MUDI comb3 optimal thread allocations.

This script compares the optimal allocations produced by the baseline
models against the workload colocations present in the power-cap selection
summaries. It emits CSV reports that list the optimal thread split for each
workload triple and, when available, augments the rows with throughput and power
measurements sourced from the CUDA sampling datasets.

Example:
    python tests/mps/eval_baselines/mudi/find_missing_comb3_threads.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set, Tuple

import pandas as pd
baseline = "gslice"
#baseline = "mudi"
#freq file paths can be used for different baselines.
FREQ_FILE_PATHS: Mapping[str, Path] = {
    "freq1530": Path(
        f"tests/mps/eval_baselines/{baseline}"
        "/summary/freq1530_comb3_summary.csv"
        #"/curvefit/summary/cutoff_freq1530_comb3_optimal_allocation_latency.csv"
    ),
    "freq900": Path(
        f"tests/mps/eval_baselines/{baseline}"
        "/summary/freq900_comb3_summary.csv"
        #"/curvefit/summary/cutoff_freq900_comb3_optimal_allocation_latency.csv"
    ),
    "freq300": Path(
        f"tests/mps/eval_baselines/{baseline}"
        "/summary/freq300_comb3_summary.csv"
        #"/curvefit/summary/cutoff_freq300_comb3_optimal_allocation_latency.csv"
    ),
}


SELECTION_CONFIGS: List[Dict[str, object]] = [
    {
        "label": "powercap200",
        "path": Path(
            #"tests/mps/analysis/plot_analysis/"
            #"09152025_xput_under_powercap_comb3_mergecudaDL_unseen_multi_freq_powercap200_"
            #"epsilon20_pred_vs_baselines_dvfs/summary_xput_under_powercap_all_pairs.csv"
            "/home/cc/"
            "powercap200_dvfs_comb3_summary_xput_under_powercap_all_pairs.csv"
        ),
        "powercap": 200,
        "output": Path(
            f"tests/mps/eval_baselines/{baseline}/{baseline}_missing_thread_powercap200.csv"
        ),
        "metrics_path": [
            Path(
                "tests/mps/freq_scaling/dataset/09152025_shareDL_comb3_powercap200_dvfs/"
                "09152025_shareDL_comb3_powercap200_dvfs_throughput_total_labels_comb3_labels.csv"
            ),
            Path(
                "tests/mps/freq_scaling/dataset/09152025_shareDL_comb3_powercap200_dvfs_mudi_baseline/"
                "09152025_shareDL_comb3_powercap200_dvfs_mudi_baseline_throughput_total_labels_comb3_labels.csv"
            ),
        ],
    },
    {
        "label": "powercap100",
        "path": Path(
            #"tests/mps/analysis/plot_analysis/"
            #"09152025_xput_under_powercap_comb3_mergecudaDL_unseen_multi_freq_powercap100_"
            #"epsilon10_pred_vs_baselines_dvfs/summary_xput_under_powercap_all_pairs.csv"
            "/home/cc/"
            "powercap100_dvfs_comb3_summary_xput_under_powercap_all_pairs.csv"
        ),
        "powercap": 100,
        "output": Path(
            f"tests/mps/eval_baselines/{baseline}/{baseline}_missing_thread_powercap100.csv"
        ),
        "metrics_path": [
            Path(
                "tests/mps/freq_scaling/dataset/09152025_shareDL_comb3_powercap100_dvfs/"
                "09152025_shareDL_comb3_powercap100_dvfs_throughput_total_labels_comb3_labels.csv"
            ),
            Path(
                "tests/mps/freq_scaling/dataset/09152025_shareDL_comb3_powercap100_dvfs_mudi_baseline/"
                "09152025_shareDL_comb3_powercap100_dvfs_mudi_baseline_throughput_total_labels_comb3_labels.csv"
            ),
        ],
    },
    {
        "label": "powercap60",
        "path": Path(
            #"tests/mps/analysis/plot_analysis/"
            #"09152025_xput_under_powercap_comb3_mergecudaDL_unseen_multi_freq_powercap100_"
            #"epsilon10_pred_vs_baselines_dvfs/summary_xput_under_powercap_all_pairs.csv"
            "/home/cc/"
            "powercap60_dvfs_comb3_summary_xput_under_powercap_all_pairs.csv"
        ),
        "powercap": 60,
        "output": Path(
            f"tests/mps/eval_baselines/{baseline}/{baseline}_missing_thread_powercap60.csv"
        ),
        "metrics_path": [
            Path(
                "tests/mps/freq_scaling/dataset/09152025_shareDL_comb3_powercap60_dvfs_gpu012/"
                "09152025_shareDL_comb3_powercap60_dvfs_throughput_total_labels_comb3_labels.csv"
            ),
            Path(
                #"tests/mps/freq_scaling/dataset/09152025_shareDL_comb3_powercap60_dvfs_mudi_baseline/"
                #"09152025_shareDL_comb3_powercap60_dvfs_mudi_baseline_throughput_total_labels_comb3_labels.csv"
                "tests/mps/freq_scaling/dataset/09152025_shareDL_comb3_powercap60_dvfs_gpu012_missing_baseline/"
                "09152025_shareDL_comb3_powercap60_dvfs_missing_baseline_throughput_total_labels_comb3_labels.csv"
            ),
        ],
    },
]


def build_canonical_key(workloads: Sequence[str]) -> Tuple[str, str, str]:
    """Canonically represent a workload colocation for joins.

    Args:
        workloads: Workload identifiers in any order.

    Returns:
        A sorted tuple suitable for dictionary keys.
    """

    return tuple(sorted(workloads))  # type: ignore[return-value]


def load_optimal_percentages(
    freq_paths: Mapping[str, Path]
) -> Dict[str, Dict[Tuple[str, str, str], Dict[str, float]]]:
    """Load optimal allocation percentages per frequency.

    Args:
        freq_paths: Mapping of frequency keys (e.g., ``freq300``) to CSV paths.

    Returns:
        Nested dictionaries where the outer key is the frequency and the inner key is the
        canonical workload colocation. Each value is a workload-to-percentage mapping.
    """

    data: Dict[str, Dict[Tuple[str, str, str], Dict[str, float]]] = {}
    for freq, path in freq_paths.items():
        df = pd.read_csv(path)
        freq_map: Dict[Tuple[str, str, str], Dict[str, float]] = {}
        for _, row in df.iterrows():
            workloads = (row["workload1"], row["workload2"], row["workload3"])
            percentages = {
                row["workload1"]: float(row["w1_optimal_percentage"]),
                row["workload2"]: float(row["w2_optimal_percentage"]),
                row["workload3"]: float(row["w3_optimal_percentage"]),
            }
            freq_map[build_canonical_key(workloads)] = percentages
        data[freq] = freq_map
    return data


def load_powercap_metrics(
    paths: Sequence[Path] | Path,
) -> Dict[Tuple[str, str, str, str], Dict[str, float]]:
    """Load measured metrics for thread combinations from the CUDA datasets.

    Args:
        paths: One or more CSV paths that contain measured throughput/power metrics.

    Returns:
        Mapping from workload/thread combination keys to metric values aggregated across
        all provided files.
    """

    if isinstance(paths, Path):
        iterable_paths: Sequence[Path] = [paths]
    else:
        iterable_paths = list(paths)

    metrics: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
    for path in iterable_paths:
        if not path.exists():
            print(f"[WARN] Power-cap metrics file '{path}' not found; skipping.")
            continue

        df = pd.read_csv(path)
        for _, row in df.iterrows():
            key = (
                row["Workload1"],
                row["Workload2"],
                row["Workload3"],
                row["Thread_combination"],
            )
            if key in metrics:
                continue

            metrics[key] = {
                "weight_Throughput_sum": row["weight_Throughput_sum"],
                "Power": row["Power"],
            }

    if not metrics:
        joined = ", ".join(str(p) for p in iterable_paths)
        print(
            f"[WARN] No power-cap metrics found; verified paths: {joined}."
        )

    return metrics


def format_thread_combination(percentages: Sequence[int]) -> str:
    """Render thread percentages in the canonical string format.

    Args:
        percentages: Thread allocation percentages in workload order.

    Returns:
        A canonical representation such as ``(w1_10, w2_30, w3_60)``.
    """

    return f"(w1_{percentages[0]}, w2_{percentages[1]}, w3_{percentages[2]})"


def ensure_parent_directory(path: Path) -> None:
    """Ensure that the parent directory for ``path`` exists.

    Args:
        path: Target file path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)


def write_output_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Persist optimal thread combinations to CSV.

    Args:
        path: Destination CSV path.
        rows: Row dictionaries ready for DataFrame construction.
    """

    ensure_parent_directory(path)
    df = pd.DataFrame(
        rows,
        columns=[
            "Workload1",
            "Workload2",
            "Workload3",
            "Thread_combination",
            "w1_percentage",
            "w2_percentage",
            "w3_percentage",
            "weight_Throughput_sum",
            "Power",
            "PowerCap",
            "selected_model_freq",
        ],
    )
    df.to_csv(path, index=False)


def main() -> None:
    """Generate optimal-thread CSVs for MUDI comb3 workloads."""

    optimal_percentages = load_optimal_percentages(FREQ_FILE_PATHS)

    processed_total = 0
    combo_frequency_map: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)

    for config in SELECTION_CONFIGS:
        label = str(config["label"])
        path = config["path"]
        powercap = config["powercap"]
        output_path = config["output"]
        metrics_path = config["metrics_path"]

        if not path.exists():
            print(
                f"[WARN] Selection summary for '{label}' not found at '{path}'. "
                "Skipping report generation for this configuration."
            )
            config["output_rows"] = []
            continue

        df = pd.read_csv(path)
        metrics_lookup = load_powercap_metrics(metrics_path)

        rows: List[Dict[str, object]] = []
        seen_signatures: Set[
            Tuple[Tuple[str, str, str], str, Tuple[int, int, int]]
        ] = set()

        for _, row in df.iterrows():
            workloads_order = (row["workload1"], row["workload2"], row["workload3"])
            canonical_key = build_canonical_key(workloads_order)
            freq = str(row["selected_model_freq"])
            combo_frequency_map[canonical_key].add(freq)

            if freq not in optimal_percentages:
                print(
                    f"[WARN] Skipping workloads {workloads_order} from {label} because frequency "
                    f"'{freq}' is not available in curve-fit outputs."
                )
                continue

            percentages_lookup = optimal_percentages[freq].get(canonical_key)
            if not percentages_lookup:
                print(
                    f"[WARN] Missing optimal percentages for workloads {workloads_order} with "
                    f"frequency '{freq}' in {label}."
                )
                continue

            ordered_percentages: List[int] = []
            missing_percentage = False
            for workload in workloads_order:
                if workload not in percentages_lookup:
                    print(
                        f"[WARN] Percentage for workload '{workload}' not found in frequency '{freq}' "
                        f"for {label}."
                    )
                    missing_percentage = True
                    break
                ordered_percentages.append(int(round(percentages_lookup[workload])))
            if missing_percentage:
                continue

            signature = (workloads_order, freq, tuple(ordered_percentages))
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            thread_combination = format_thread_combination(ordered_percentages)
            metrics_key = (
                workloads_order[0],
                workloads_order[1],
                workloads_order[2],
                thread_combination,
            )
            metrics = metrics_lookup.get(metrics_key, {})
            throughput_value = metrics.get("weight_Throughput_sum", "")
            power_value = metrics.get("Power", "")

            if throughput_value != "" and pd.isna(throughput_value):
                throughput_value = ""
            if power_value != "" and pd.isna(power_value):
                power_value = ""

            rows.append(
                {
                    "Workload1": workloads_order[0],
                    "Workload2": workloads_order[1],
                    "Workload3": workloads_order[2],
                    "Thread_combination": thread_combination,
                    "w1_percentage": ordered_percentages[0],
                    "w2_percentage": ordered_percentages[1],
                    "w3_percentage": ordered_percentages[2],
                    "weight_Throughput_sum": throughput_value,
                    "Power": power_value,
                    "PowerCap": powercap,
                    "selected_model_freq": freq,
                }
            )

        write_output_rows(output_path, rows)
        config["output_rows"] = rows
        processed_total += len(rows)

    print("=== MUDI comb3 optimal allocations ===")
    for config in SELECTION_CONFIGS:
        label = str(config["label"])
        rows = config.get("output_rows", [])
        print(f"{label}: {len(rows)} workload colocations processed.")
    print(f"Total colocations processed: {processed_total}")

    multi_freq_count = sum(
        1 for freq_set in combo_frequency_map.values() if len(freq_set) > 1
    )
    if multi_freq_count:
        print(
            f"Observed {multi_freq_count} workload colocations with multiple selected frequencies "
            "across the provided power-cap summary files."
        )


if __name__ == "__main__":
    main()
