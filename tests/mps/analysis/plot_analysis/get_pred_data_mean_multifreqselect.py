import argparse
import os
import sys
from collections import defaultdict, OrderedDict
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
MPS_DIR = REPO_ROOT / "tests" / "mps"
FREQ_DATASET_DIR = MPS_DIR / "freq_scaling" / "dataset"
FREQ_OUTPUT_DIR = MPS_DIR / "freq_scaling" / "output"
STAGE2_DIR = MPS_DIR / "analysis" / "stage2"

def get_thread_num_from_str(thread_comb_str):
    """
    Extracts thread percentages from a string like "(w1_40, w2_60, w3_0)".
    Returns a list whose length matches the number of workload entries present.
    Handles potential NaN or malformed inputs.
    """
    if pd.isna(thread_comb_str):
        return []

    try:
        cleaned_str = (
            str(thread_comb_str)
            .replace("\"", "")
            .replace("(", "")
            .replace(")", "")
        )
        parts = [part.strip() for part in cleaned_str.split(",") if part.strip()]
        percentages = []
        for part in parts:
            if "_" not in part:
                percentages.append(None)
                continue
            _, value_str = part.split("_", 1)
            value_str = value_str.strip()
            try:
                value = float(value_str)
            except ValueError:
                percentages.append(None)
                continue
            percentages.append(int(round(value)))
        return percentages
    except Exception:
        return []


def safe_div(a, b):
    """
    Safely divides two numbers. Returns None if division is not possible or inputs are None/NaN.
    """
    if pd.isna(a) or pd.isna(b) or b == 0:
        return None
    return a / b


def build_workload_column_names(count):
    """Return canonical workload column names: workload1, workload2, ..."""
    return [f"workload{i}" for i in range(1, count + 1)]


def build_threadpercent_column_names(count):
    """Return canonical thread-percent column names: w1_Threadpercent, w2_Threadpercent, ..."""
    return [f"w{i}_Threadpercent" for i in range(1, count + 1)]


def adjust_mudi_percentages(w1_pct, w2_pct):
    """Apply fallback thread splits when MUDI suggests fully dedicating one workload."""
    try:
        w1_val = float(w1_pct)
        w2_val = float(w2_pct)
    except (TypeError, ValueError):
        return None, None

    if abs(w1_val - 0.0) < 1e-6 and abs(w2_val - 100.0) < 1e-6:
        return 10, 90
    if abs(w1_val - 100.0) < 1e-6 and abs(w2_val - 0.0) < 1e-6:
        return 90, 10
    return w1_val, w2_val


def load_optimal_allocations(file_path, baseline_name):
    """Load optimal allocations for a baseline and build a lookup keyed by workload order."""
    if not file_path or not os.path.exists(file_path):
        if file_path:
            print(f"Warning: {baseline_name} allocation file not found at {file_path}")
        return {}

    try:
        opt_df = pd.read_csv(file_path)
    except Exception as exc:
        print(f"Warning: Failed to load {baseline_name} allocation file {file_path}: {exc}")
        return {}

    required_cols = {"workload1", "workload2", "w1_optimal_percentage", "w2_optimal_percentage"}
    missing_cols = required_cols.difference(opt_df.columns)
    if missing_cols:
        print(f"Warning: {baseline_name} allocation file missing columns: {sorted(missing_cols)}")
        return {}

    lookup = {}
    for _, row in opt_df.iterrows():
        w1_name = row["workload1"]
        w2_name = row["workload2"]

        w1_pct_adj, w2_pct_adj = adjust_mudi_percentages(
            row["w1_optimal_percentage"], row["w2_optimal_percentage"]
        )
        if w1_pct_adj is None or w2_pct_adj is None:
            print(
                f"Warning: {baseline_name} allocation percentages missing or invalid for pair {w1_name}, {w2_name} "
                f"in {os.path.basename(file_path) if file_path else 'allocation file'}"
            )
            continue

        w1_pct_final = int(round(w1_pct_adj))
        w2_pct_final = int(round(w2_pct_adj))
        if w1_pct_final + w2_pct_final != 100:
            w2_pct_final = max(0, min(100, 100 - w1_pct_final))

        lookup[(w1_name, w2_name)] = (w1_pct_final, w2_pct_final)
        lookup[(w2_name, w1_name)] = (w2_pct_final, w1_pct_final)

    return lookup


def load_optimal_allocations_by_freq(freq_file_map, baseline_name):
    """Load per-frequency optimal allocation lookups for a baseline."""
    if not freq_file_map:
        return {}

    allocations = {}
    for freq_label, file_path in freq_file_map.items():
        lookup = load_optimal_allocations(file_path, baseline_name)
        if lookup:
            allocations[freq_label] = lookup
    return allocations


def load_mudi_latency_allocations_by_freq(freq_file_map):
    """Load MUDI latency-oriented allocations keyed by frequency."""
    return load_optimal_allocations_by_freq(freq_file_map, baseline_name="MUDI_LATENCY")


def load_mudi_xput_allocations_by_freq(freq_file_map):
    """Load MUDI throughput-oriented allocations keyed by frequency."""
    return load_optimal_allocations_by_freq(freq_file_map, baseline_name="MUDI_XPUT")


def load_gslice_allocations_by_freq(freq_file_map):
    """Load GSLICE allocations keyed by frequency."""
    return load_optimal_allocations_by_freq(freq_file_map, baseline_name="GSLICE")


def load_muxflow_allocations(file_path):
    """Load MuxFlow allocations keyed by workload order from a single CSV."""
    if not file_path or not os.path.exists(file_path):
        if file_path:
            print(f"Warning: MUXFLOW allocation file not found at {file_path}")
        return {}

    try:
        mux_df = pd.read_csv(file_path)
    except Exception as exc:
        print(f"Warning: Failed to load MUXFLOW allocation file {file_path}: {exc}")
        return {}

    mux_df = mux_df.rename(columns={"Workload1": "workload1", "Workload2": "workload2"})
    required_cols = {"workload1", "workload2", "w1_optimal_percentage", "w2_optimal_percentage"}
    missing_cols = required_cols.difference(mux_df.columns)
    if missing_cols:
        print(f"Warning: MUXFLOW allocation file missing columns: {sorted(missing_cols)}")
        return {}

    lookup = {}
    for _, row in mux_df.iterrows():
        w1_name = row["workload1"]
        w2_name = row["workload2"]
        try:
            w1_pct = int(round(float(row["w1_optimal_percentage"])))
            w2_pct = int(round(float(row["w2_optimal_percentage"])))
        except (TypeError, ValueError):
            print(f"Warning: Invalid MUXFLOW percentages for pair {w1_name}, {w2_name} in {os.path.basename(file_path)}")
            continue

        if w1_pct + w2_pct != 100:
            raise ValueError(f"muxflow w1 and w2 should combine as 100")
            #w2_pct = max(0, min(100, 100 - w1_pct))

        lookup[(w1_name, w2_name)] = (w1_pct, w2_pct)
        lookup[(w2_name, w1_name)] = (w2_pct, w1_pct)

    return lookup


def load_mudi_xput_multiworkload_allocations(powercap_file_map, power_cap):
    """Load MUDI throughput-oriented allocations for three-workload combinations."""
    if not powercap_file_map:
        return {}

    file_path = powercap_file_map.get(power_cap)
    if not file_path:
        print(f"Warning: No MUDI throughput allocation file configured for power cap {power_cap}")
        return {}
    if not os.path.exists(file_path):
        print(f"Warning: MUDI throughput allocation file not found at {file_path}")
        return {}

    try:
        allocations_df = pd.read_csv(file_path)
    except Exception as exc:
        print(f"Warning: Failed to load MUDI throughput allocation file {file_path}: {exc}")
        return {}

    required_cols = {
        "Workload1",
        "Workload2",
        "Workload3",
        "w1_percentage",
        "w2_percentage",
        "w3_percentage",
        "weight_Throughput_sum",
        "Power",
    }
    missing_cols = required_cols.difference(allocations_df.columns)
    if missing_cols:
        print(
            f"Warning: MUDI throughput allocation file {file_path} missing columns: "
            f"{sorted(missing_cols)}"
        )
        return {}

    lookup = {}
    for _, row in allocations_df.iterrows():
        workloads = [row["Workload1"], row["Workload2"], row["Workload3"]]
        if any(pd.isna(name) for name in workloads):
            print(f"Warning: Skipping MUDI allocation row with missing workload name in {file_path}")
            continue

        try:
            base_percentages = [
                int(round(float(row["w1_percentage"]))),
                int(round(float(row["w2_percentage"]))),
                int(round(float(row["w3_percentage"]))),
            ]
        except (TypeError, ValueError):
            print(
                "Warning: Invalid MUDI allocation percentages for workloads "
                f"{workloads} in {file_path}"
            )
            continue

        throughput_val = row["weight_Throughput_sum"]
        power_val = row["Power"]
        source_freq = row.get("selected_model_freq")

        for perm_indices in permutations(range(len(workloads))):
            key = tuple(workloads[idx] for idx in perm_indices)
            perm_percentages = [base_percentages[idx] for idx in perm_indices]
            entry = {
                "thread_percentages": perm_percentages,
                "throughput": throughput_val,
                "power": power_val,
                "source_freq": source_freq,
            }

            existing_entry = lookup.get(key)
            if existing_entry:
                if (
                    existing_entry["throughput"] != entry["throughput"]
                    or existing_entry["power"] != entry["power"]
                    or existing_entry["thread_percentages"] != entry["thread_percentages"]
                ):
                    print(
                        "Warning: Conflicting MUDI allocations for workloads "
                        f"{key} in {file_path}; keeping the first entry."
                    )
                continue

            lookup[key] = entry

    return lookup


def discover_cross_validation_folds(freq_dir_map, cross_num_testsets):
    """Return sorted fold directory names common across all frequency directories."""
    if not freq_dir_map:
        return []

    desired_prefixes = []
    if cross_num_testsets:
        desired_prefixes = [f"fold_{num}_" for num in cross_num_testsets]

    common_folds = None
    for freq_value, base_dir in freq_dir_map.items():
        if not base_dir or not os.path.isdir(base_dir):
            print(f"Warning: Cross-validation base directory missing for freq {freq_value}: {base_dir}")
            return []

        current_folds = {
            entry.name
            for entry in os.scandir(base_dir)
            if entry.is_dir()
        }
        if desired_prefixes:
            current_folds = {
                name for name in current_folds
                if any(name.startswith(prefix) for prefix in desired_prefixes)
            }
            

        if not current_folds:
            print(f"Warning: No fold directories discovered in {base_dir}")
            return []

        if common_folds is None:
            common_folds = current_folds
        else:
            common_folds &= current_folds

        if not common_folds:
            print("Warning: No common fold directories found across frequencies.")
            return []
        print(f"common_folds={common_folds}")
        #debug 
        #common_folds={"fold_7_test_1-7"}

    return sorted(common_folds) if common_folds else []


def build_cross_validation_prediction_maps(
    workloads,
    freq_dirs_xput,
    freq_dirs_power,
    fold_name,
    model_name,
    pred_xput_filename,
    pred_power_filename,
):
    """
    Build prediction file path dictionaries for a given cross-validation fold and model.
    Returns:
        tuple(dict, dict): (predict_xput_csv_files_dict, predict_power_csv_files_dict)
    Raises:
        FileNotFoundError: if any required CSV is missing.
    """
    predict_xput_csv_files_dict = {}
    predict_power_csv_files_dict = {}
    missing_paths = []

    for freq_value, xput_base_dir in freq_dirs_xput.items():
        power_base_dir = freq_dirs_power.get(freq_value)
        freq_missing_paths = []
        if not power_base_dir:
            freq_missing_paths.append(f"Power directory missing for freq {freq_value}")
            missing_paths.extend(freq_missing_paths)
            continue

        xput_predict_dir = os.path.join(xput_base_dir, fold_name, model_name, "predictAllacc")
        power_predict_dir = os.path.join(power_base_dir, fold_name, model_name, "predictAllacc")
        xput_csv_path = os.path.join(xput_predict_dir, pred_xput_filename)
        power_csv_path = os.path.join(power_predict_dir, pred_power_filename)

        if not os.path.exists(xput_csv_path):
            freq_missing_paths.append(xput_csv_path)
        if not os.path.exists(power_csv_path):
            freq_missing_paths.append(power_csv_path)
        if freq_missing_paths:
            missing_paths.extend(freq_missing_paths)
            continue

        predict_xput_csv_files_dict[freq_value] = {workload: xput_csv_path for workload in workloads}
        predict_power_csv_files_dict[freq_value] = {workload: power_csv_path for workload in workloads}

    if missing_paths:
        missing_str = ", ".join(sorted(set(missing_paths)))
        raise FileNotFoundError(f"Missing prediction CSVs for fold '{fold_name}', model '{model_name}': {missing_str}")

    return predict_xput_csv_files_dict, predict_power_csv_files_dict



def populate_allocation_baseline(
    result,
    baseline_label,
    thread_percent_lookup_by_freq,
    workload1_name,
    workload2_name,
    oracle_df_for_pair,
    target_metric_name,
    model_freq_labels,
    selected_model_freq,
):
    """Populate baseline metrics from a precomputed optimal allocation."""
    if not thread_percent_lookup_by_freq:
        return

    pair_key = (workload1_name, workload2_name)
    baseline_key = baseline_label.lower()

    freq_priority = []
    if selected_model_freq and selected_model_freq in thread_percent_lookup_by_freq:
        freq_priority.append(selected_model_freq)

    for freq_label in model_freq_labels:
        if freq_label not in freq_priority and freq_label in thread_percent_lookup_by_freq:
            freq_priority.append(freq_label)

    for freq_label in thread_percent_lookup_by_freq.keys():
        if freq_label not in freq_priority:
            freq_priority.append(freq_label)

    baseline_source_freq = None
    baseline_split = None
    for freq_label in freq_priority:
        lookup = thread_percent_lookup_by_freq.get(freq_label)
        if lookup and pair_key in lookup:
            baseline_source_freq = freq_label
            baseline_split = lookup[pair_key]
            break

    if baseline_split is None:
        print(f"Warning: {baseline_label} baseline allocation not found for pair {workload1_name}, {workload2_name}")
        return

    w1_tp, w2_tp = baseline_split
    result[f"{baseline_key}_source_freq"] = baseline_source_freq
    result[f"{baseline_key}_w1_threadpercent"] = w1_tp
    result[f"{baseline_key}_w2_threadpercent"] = w2_tp

    baseline_oracle_row_df = oracle_df_for_pair[
        (oracle_df_for_pair["w1_Threadpercent"] == w1_tp)
        & (oracle_df_for_pair["w2_Threadpercent"] == w2_tp)
    ]
    if baseline_oracle_row_df.empty:
        print(
            f"Warning: Oracle data missing for {baseline_label} baseline split {w1_tp}/{w2_tp} "
            f"for pair {workload1_name}, {workload2_name}"
        )
        return

    baseline_oracle_row = baseline_oracle_row_df.iloc[0]
    result[f"{baseline_key}_oracle_throughput"] = baseline_oracle_row["weight_Throughput_sum_oracle_actual"]
    result[f"{baseline_key}_oracle_power"] = baseline_oracle_row["Power_oracle_actual"]
    result[f"{baseline_key}_vs_oracle_ratio"] = safe_div(
        result[f"{baseline_key}_oracle_throughput"],
        result.get(f"best_{target_metric_name}_oracle_throughput"),
    )


def populate_muxflow_baseline(
    result,
    muxflow_thread_percent_lookup,
    workload1_name,
    workload2_name,
    oracle_df_for_pair,
    target_metric_name,
):
    """Populate MuxFlow baseline metrics using precomputed optimal percentages."""
    if not muxflow_thread_percent_lookup:
        return

    pair_key = (workload1_name, workload2_name)
    muxflow_split = muxflow_thread_percent_lookup.get(pair_key)
    if muxflow_split is None:
        raise ValueError(f"Error: MUXFLOW baseline allocation not found for pair {workload1_name}, {workload2_name}")
        return

    w1_tp, w2_tp = muxflow_split
    result["muxflow_w1_threadpercent"] = w1_tp
    result["muxflow_w2_threadpercent"] = w2_tp

    muxflow_oracle_row_df = oracle_df_for_pair[
        (oracle_df_for_pair["w1_Threadpercent"] == w1_tp)
        & (oracle_df_for_pair["w2_Threadpercent"] == w2_tp)
    ]
    if muxflow_oracle_row_df.empty:
        print(
            f"Warning: Oracle data missing for MUXFLOW baseline split {w1_tp}/{w2_tp} "
            f"for pair {workload1_name}, {workload2_name}"
        )
        return

    muxflow_oracle_row = muxflow_oracle_row_df.iloc[0]
    result["muxflow_oracle_throughput"] = muxflow_oracle_row["weight_Throughput_sum_oracle_actual"]
    result["muxflow_oracle_power"] = muxflow_oracle_row["Power_oracle_actual"]
    result["muxflow_vs_oracle_ratio"] = safe_div(
        result["muxflow_oracle_throughput"],
        result.get(f"best_{target_metric_name}_oracle_throughput"),
    )



def get_pred_oracle_target_comparison_powercap(
    workload_names,
    pred_xput_dfs_list,
    pred_power_dfs_list,
    model_freq_labels,
    oracle_df_for_pair,
    no_partition_oracle_config_for_pair,
    static_power_limit,
    target_metric_name,
    weight,
    output_dir_for_plots,
    rm_100partitions,
    model_label_for_custom_baseline,
    threadpercent_columns,
    workload_columns,
    num_workloads,
    mudi_latency_thread_percent_lookup_by_freq=None,
    mudi_xput_thread_percent_lookup_by_freq=None,
    gslice_thread_percent_lookup_by_freq=None,
    muxflow_thread_percent_lookup=None,
    mudi_xput_multiworkload_allocations=None,
    oracle_latency_lookup=None,
    oracle_steps_lookup=None,
    oracle_latency_lookup_baselines=None,
    oracle_steps_lookup_baselines=None,
    weighted_latency_debug_collector=None,
    is_plot=False,
):
    """Compare predicted throughput/power with oracle data under a power cap.

    The implementation supports an arbitrary number of colocated workloads. Additional
    baselines (MUDI, GSLICE, etc.) remain enabled only for the 2-workload case because
    the allocation tables are pairwise.
    """
    workload_names = tuple(workload_names)
    combo_label = ", ".join(workload_names)
    use_two_workload_baselines = num_workloads == 2

    no_partition_baseline_config_str = "(" + ", ".join(f"w{i}_100" for i in range(1, num_workloads + 1)) + ")"
    if use_two_workload_baselines:
        fair_partition_baseline_config_str = "(w1_50, w2_50)"
    elif num_workloads == 3:
        fair_partition_baseline_config_str = "(w1_30, w2_30, w3_40)"
    else:
        fair_partition_baseline_config_str = None

    all_candidate_configs_for_pair_dfs = []
    merge_cols = workload_columns + threadpercent_columns

    def _filter_df_by_workloads(df):
        if df is None or df.empty:
            return pd.DataFrame()
        missing_cols = [col for col in workload_columns if col not in df.columns]
        if missing_cols:
            return pd.DataFrame()
        mask = pd.Series(True, index=df.index)
        for col, name in zip(workload_columns, workload_names):
            mask &= df[col] == name
        return df[mask].copy()

    def _remove_full_partition_rows(df):
        if df.empty:
            return df
        if not set(threadpercent_columns).issubset(df.columns):
            return df
        full_mask = (df[threadpercent_columns] == 100).all(axis=1)
        return df[~full_mask]

    for i, current_freq_label in enumerate(model_freq_labels):
        if i >= len(pred_xput_dfs_list) or pred_xput_dfs_list[i] is None:
            continue
        if i >= len(pred_power_dfs_list) or pred_power_dfs_list[i] is None:
            continue

        pair_pred_xput_df = _filter_df_by_workloads(pred_xput_dfs_list[i])
        pair_pred_power_df = _filter_df_by_workloads(pred_power_dfs_list[i])

        if rm_100partitions:
            pair_pred_xput_df = _remove_full_partition_rows(pair_pred_xput_df)
            pair_pred_power_df = _remove_full_partition_rows(pair_pred_power_df)

        if pair_pred_xput_df.empty or pair_pred_power_df.empty:
            continue

        pair_pred_xput_df = pair_pred_xput_df.rename(columns={"y_pred": "pred_throughput_sum_val", "y_test": "y_test_xput"})
        pair_pred_power_df = pair_pred_power_df.rename(columns={"y_pred": "pred_power_val", "y_test": "y_test_power"})

        missing_merge_cols = [
            col for col in merge_cols
            if col not in pair_pred_xput_df.columns or col not in pair_pred_power_df.columns
        ]
        if missing_merge_cols:
            print(
                f"Warning: Merge columns missing for {current_freq_label} for workloads [{combo_label}]. "
                f"Missing: {missing_merge_cols}"
            )
            continue

        cols_to_select_xput = [col for col in merge_cols + ["pred_throughput_sum_val"] if col in pair_pred_xput_df.columns]
        cols_to_select_power = [col for col in merge_cols + ["pred_power_val"] if col in pair_pred_power_df.columns]

        merged_pred_for_freq = pd.merge(
            pair_pred_xput_df[cols_to_select_xput],
            pair_pred_power_df[cols_to_select_power],
            on=merge_cols,
            how="inner",
        )

        if merged_pred_for_freq.empty:
            continue

        valid_pred_df_for_freq = merged_pred_for_freq[merged_pred_for_freq["pred_power_val"] <= static_power_limit]
        if valid_pred_df_for_freq.empty:
            continue

        valid_pred_df_for_freq = valid_pred_df_for_freq.copy()
        valid_pred_df_for_freq.loc[:, "selected_model_freq"] = current_freq_label

        configs_with_oracle_actuals = pd.merge(
            valid_pred_df_for_freq,
            oracle_df_for_pair,
            on=merge_cols,
            how="inner",
            suffixes=("_pred_model", "_oracle_actual"),
        )
        if not configs_with_oracle_actuals.empty:
            all_candidate_configs_for_pair_dfs.append(configs_with_oracle_actuals)

    result = {col: name for col, name in zip(workload_columns, workload_names)}

    if not all_candidate_configs_for_pair_dfs:
        print(
            f"No predicted configurations under power cap {static_power_limit} for workloads [{combo_label}] "
            "from any model after merging with oracle."
        )
    else:
        all_candidates_df_for_pair = pd.concat(all_candidate_configs_for_pair_dfs, ignore_index=True)
        if all_candidates_df_for_pair.empty:
            print(f"Concatenated candidates DataFrame is empty for workloads [{combo_label}].")
        else:
            best_idx = all_candidates_df_for_pair["pred_throughput_sum_val"].idxmax()
            best_pred_config_row = all_candidates_df_for_pair.loc[best_idx]
            result["selected_model_freq"] = best_pred_config_row["selected_model_freq"]
            for pos, thread_col in enumerate(threadpercent_columns, start=1):
                result[f"pred_w{pos}_threadpercent_selected"] = best_pred_config_row.get(thread_col)
            result[f"pred_{target_metric_name}_exec_value"] = best_pred_config_row["weight_Throughput_sum_oracle_actual"]
            result[f"pred_{target_metric_name}_exec_throughput"] = best_pred_config_row["weight_Throughput_sum_oracle_actual"]
            result[f"pred_{target_metric_name}_exec_power"] = best_pred_config_row["Power_oracle_actual"]
            result[f"pred_{target_metric_name}_pred_throughput_val"] = best_pred_config_row["pred_throughput_sum_val"]
            result[f"pred_{target_metric_name}_pred_power_val"] = best_pred_config_row["pred_power_val"]

    valid_oracle_configs = oracle_df_for_pair[oracle_df_for_pair["Power_oracle_actual"] <= static_power_limit]
    if not valid_oracle_configs.empty:
        best_oracle_idx = valid_oracle_configs["weight_Throughput_sum_oracle_actual"].idxmax()
        best_oracle_config_row = valid_oracle_configs.loc[best_oracle_idx]
        for pos, thread_col in enumerate(threadpercent_columns, start=1):
            result[f"best_w{pos}_threadpercent_oracle"] = best_oracle_config_row.get(thread_col)
        result[f"best_{target_metric_name}_oracle_value"] = best_oracle_config_row["weight_Throughput_sum_oracle_actual"]
        result[f"best_{target_metric_name}_oracle_throughput"] = best_oracle_config_row["weight_Throughput_sum_oracle_actual"]
        result[f"best_{target_metric_name}_oracle_power"] = best_oracle_config_row["Power_oracle_actual"]
    else:
        print(f"No ORACLE configurations under power cap {static_power_limit} for workloads [{combo_label}]")
        result[f"best_{target_metric_name}_oracle_value"] = None
        result[f"best_{target_metric_name}_oracle_throughput"] = None
        result[f"best_{target_metric_name}_oracle_power"] = None
        for pos in range(1, num_workloads + 1):
            result[f"best_w{pos}_threadpercent_oracle"] = None

    if fair_partition_baseline_config_str is not None:
        fair_partition_baseline_df = oracle_df_for_pair[
            oracle_df_for_pair["Thread_combination"] == fair_partition_baseline_config_str
        ]
        if fair_partition_baseline_df.empty:
            result[f"{target_metric_name}_fair_partition_baseline_value"] = None
        else:
            result[f"{target_metric_name}_fair_partition_baseline_value"] = (
                fair_partition_baseline_df["weight_Throughput_sum_oracle_actual"].iloc[0]
            )
    else:
        result[f"{target_metric_name}_fair_partition_baseline_value"] = None

    if no_partition_oracle_config_for_pair is not None and not no_partition_oracle_config_for_pair.empty:
        result[f"{target_metric_name}_true_no_partition_baseline_value"] = (
            no_partition_oracle_config_for_pair["weight_Throughput_sum_oracle_actual"].iloc[0]
        )
        result[f"{target_metric_name}_true_no_partition_baseline_power"] = (
            no_partition_oracle_config_for_pair["Power_oracle_actual"].iloc[0]
        )
    else:
        result[f"{target_metric_name}_true_no_partition_baseline_value"] = None
        result[f"{target_metric_name}_true_no_partition_baseline_power"] = None

    result.setdefault("baseline_custom_exceeds_cap", None)
    result.setdefault("baseline_custom_pred_throughput", None)
    result.setdefault("baseline_custom_pred_power", None)
    result.setdefault("baseline_custom_actual_throughput", None)
    result.setdefault("baseline_custom_actual_power", None)
    result.setdefault("baseline_custom_actual_xput_vs_oracle_ratio", None)
    for pos in range(1, min(num_workloads, 2) + 1):
        result.setdefault(f"baseline_custom_w{pos}_threadpercent", None)

    if num_workloads == 3:
        for pos in range(1, num_workloads + 1):
            result.setdefault(f"mudi_xput_w{pos}_threadpercent", None)
        result.setdefault("mudi_xput_oracle_throughput", None)
        result.setdefault("mudi_xput_oracle_power", None)
        result.setdefault("mudi_xput_vs_oracle_ratio", None)

    if use_two_workload_baselines:
        workload1_name, workload2_name = workload_names[:2]
        for i, cb_freq_label in enumerate(model_freq_labels):
            baseline_prefix = f"baseline_{cb_freq_label}"
            for suffix in ("w1_tp", "w2_tp", "pred_xput", "pred_power", "actual_xput", "actual_power", "exceeds_cap", "actual_xput_vs_oracle_ratio"):
                result.setdefault(f"{baseline_prefix}_{suffix}", None)

            if i >= len(pred_xput_dfs_list) or pred_xput_dfs_list[i] is None:
                print(
                    f"Warning: Prediction data for custom baseline freq '{cb_freq_label}' not available for "
                    f"{workload1_name},{workload2_name}"
                )
                continue
            if i >= len(pred_power_dfs_list) or pred_power_dfs_list[i] is None:
                print(
                    f"Warning: Power prediction data for custom baseline freq '{cb_freq_label}' not available for "
                    f"{workload1_name},{workload2_name}"
                )
                continue

            cb_xput_df = _filter_df_by_workloads(pred_xput_dfs_list[i])
            cb_power_df = _filter_df_by_workloads(pred_power_dfs_list[i])
            if rm_100partitions:
                cb_xput_df = _remove_full_partition_rows(cb_xput_df)
                cb_power_df = _remove_full_partition_rows(cb_power_df)
            if cb_xput_df.empty or cb_power_df.empty:
                print(
                    f"Warning: Prediction data for custom baseline freq '{cb_freq_label}' empty after filtering for "
                    f"{workload1_name},{workload2_name}"
                )
                continue

            cb_xput_df = cb_xput_df.rename(columns={"y_pred": "pred_throughput_sum_val"})
            cb_power_df = cb_power_df.rename(columns={"y_pred": "pred_power_val"})

            merge_cols_cb = workload_columns[:2] + threadpercent_columns[:2]
            if not set(merge_cols_cb).issubset(cb_xput_df.columns) or not set(merge_cols_cb).issubset(cb_power_df.columns):
                continue

            merged_cb_pred_df = pd.merge(
                cb_xput_df[merge_cols_cb + ["pred_throughput_sum_val"]],
                cb_power_df[merge_cols_cb + ["pred_power_val"]],
                on=merge_cols_cb,
                how="inner",
            )
            if merged_cb_pred_df.empty:
                continue

            selected_cb_pred_idx_for_freq = merged_cb_pred_df["pred_throughput_sum_val"].idxmax()
            selected_cb_pred_row_for_freq = merged_cb_pred_df.loc[selected_cb_pred_idx_for_freq]
            result[f"{baseline_prefix}_w1_tp"] = selected_cb_pred_row_for_freq["w1_Threadpercent"]
            result[f"{baseline_prefix}_w2_tp"] = selected_cb_pred_row_for_freq["w2_Threadpercent"]
            result[f"{baseline_prefix}_pred_xput"] = selected_cb_pred_row_for_freq["pred_throughput_sum_val"]
            result[f"{baseline_prefix}_pred_power"] = selected_cb_pred_row_for_freq["pred_power_val"]

            oracle_cb_actual_row_df = oracle_df_for_pair[
                (oracle_df_for_pair["w1_Threadpercent"] == selected_cb_pred_row_for_freq["w1_Threadpercent"]) &
                (oracle_df_for_pair["w2_Threadpercent"] == selected_cb_pred_row_for_freq["w2_Threadpercent"])
            ]
            if oracle_cb_actual_row_df.empty and                selected_cb_pred_row_for_freq["w1_Threadpercent"] == 100 and                selected_cb_pred_row_for_freq["w2_Threadpercent"] == 100 and                no_partition_oracle_config_for_pair is not None and not no_partition_oracle_config_for_pair.empty:
                oracle_cb_actual_row_df = no_partition_oracle_config_for_pair

            if not oracle_cb_actual_row_df.empty:
                oracle_cb_actual_row = oracle_cb_actual_row_df.iloc[0]
                result[f"{baseline_prefix}_actual_xput"] = oracle_cb_actual_row["weight_Throughput_sum_oracle_actual"]
                result[f"{baseline_prefix}_actual_power"] = oracle_cb_actual_row["Power_oracle_actual"]
                result[f"{baseline_prefix}_exceeds_cap"] = oracle_cb_actual_row["Power_oracle_actual"] > static_power_limit
                result[f"{baseline_prefix}_actual_xput_vs_oracle_ratio"] = safe_div(
                    result[f"{baseline_prefix}_actual_xput"],
                    result.get(f"best_{target_metric_name}_oracle_throughput"),
                )

        if model_label_for_custom_baseline in model_freq_labels:
            freq_index = model_freq_labels.index(model_label_for_custom_baseline)
            if freq_index < len(pred_xput_dfs_list) and freq_index < len(pred_power_dfs_list):
                model_specific_xput_df = _filter_df_by_workloads(pred_xput_dfs_list[freq_index])
                model_specific_power_df = _filter_df_by_workloads(pred_power_dfs_list[freq_index])
                if rm_100partitions:
                    model_specific_xput_df = _remove_full_partition_rows(model_specific_xput_df)
                    model_specific_power_df = _remove_full_partition_rows(model_specific_power_df)
                if not model_specific_xput_df.empty and not model_specific_power_df.empty:
                    model_specific_xput_df = model_specific_xput_df.rename(columns={"y_pred": "pred_throughput_sum_val"})
                    model_specific_power_df = model_specific_power_df.rename(columns={"y_pred": "pred_power_val"})
                    merge_cols_baseline = workload_columns[:2] + threadpercent_columns[:2]
                    merged_baseline_df = pd.merge(
                        model_specific_xput_df[merge_cols_baseline + ["pred_throughput_sum_val"]],
                        model_specific_power_df[merge_cols_baseline + ["pred_power_val"]],
                        on=merge_cols_baseline,
                        how="inner",
                    )
                    if not merged_baseline_df.empty:
                        max_pred_idx = merged_baseline_df["pred_throughput_sum_val"].idxmax()
                        max_pred_row = merged_baseline_df.loc[max_pred_idx]
                        result["baseline_custom_w1_threadpercent"] = max_pred_row["w1_Threadpercent"]
                        result["baseline_custom_w2_threadpercent"] = max_pred_row["w2_Threadpercent"]
                        result["baseline_custom_pred_throughput"] = max_pred_row["pred_throughput_sum_val"]
                        result["baseline_custom_pred_power"] = max_pred_row["pred_power_val"]
                        result["baseline_custom_exceeds_cap"] = max_pred_row["pred_power_val"] > static_power_limit

                        oracle_baseline_row_df = oracle_df_for_pair[
                            (oracle_df_for_pair["w1_Threadpercent"] == max_pred_row["w1_Threadpercent"]) &
                            (oracle_df_for_pair["w2_Threadpercent"] == max_pred_row["w2_Threadpercent"])
                        ]
                        if not oracle_baseline_row_df.empty:
                            oracle_baseline_row = oracle_baseline_row_df.iloc[0]
                            result["baseline_custom_actual_throughput"] = oracle_baseline_row["weight_Throughput_sum_oracle_actual"]
                            result["baseline_custom_actual_power"] = oracle_baseline_row["Power_oracle_actual"]
                            result["baseline_custom_actual_xput_vs_oracle_ratio"] = safe_div(
                                result["baseline_custom_actual_throughput"],
                                result.get(f"best_{target_metric_name}_oracle_throughput"),
                            )
                else:
                    print(
                        f"Warning: Custom baseline data empty for {model_label_for_custom_baseline} for "
                        f"{workload1_name}, {workload2_name}"
                    )
            else:
                print(
                    f"Warning: Custom baseline freq '{model_label_for_custom_baseline}' not found in predictions list "
                    f"for {workload1_name}, {workload2_name}"
                )
        else:
            print(
                f"Warning: Custom baseline freq '{model_label_for_custom_baseline}' not present in model frequency labels."
            )

        populate_allocation_baseline(
            result,
            "MUDI_LATENCY",
            mudi_latency_thread_percent_lookup_by_freq,
            workload1_name,
            workload2_name,
            oracle_df_for_pair,
            target_metric_name,
            model_freq_labels,
            result.get("selected_model_freq"),
        )
        populate_allocation_baseline(
            result,
            "MUDI_XPUT",
            mudi_xput_thread_percent_lookup_by_freq,
            workload1_name,
            workload2_name,
            oracle_df_for_pair,
            target_metric_name,
            model_freq_labels,
            result.get("selected_model_freq"),
        )
        populate_allocation_baseline(
            result,
            "GSLICE",
            gslice_thread_percent_lookup_by_freq,
            workload1_name,
            workload2_name,
            oracle_df_for_pair,
            target_metric_name,
            model_freq_labels,
            result.get("selected_model_freq"),
        )
        populate_muxflow_baseline(
            result,
            muxflow_thread_percent_lookup,
            workload1_name,
            workload2_name,
            oracle_df_for_pair,
            target_metric_name,
        )
    elif num_workloads == 3 and mudi_xput_multiworkload_allocations:
        baseline_entry = mudi_xput_multiworkload_allocations.get(workload_names)
        if baseline_entry is None:
            combo_label = ", ".join(workload_names)
            print(f"Warning: MUDI throughput baseline not found for workloads [{combo_label}]")
        else:
            thread_percentages = baseline_entry.get("thread_percentages", [])
            for idx, percentage in enumerate(thread_percentages, start=1):
                result[f"mudi_xput_w{idx}_threadpercent"] = percentage
            result["mudi_xput_oracle_throughput"] = baseline_entry.get("throughput")
            result["mudi_xput_oracle_power"] = baseline_entry.get("power")
            result["mudi_xput_vs_oracle_ratio"] = safe_div(
                result["mudi_xput_oracle_throughput"],
                result.get(f"best_{target_metric_name}_oracle_throughput"),
            )

    if f"pred_{target_metric_name}_exec_value" in result:
        result[f"pred_vs_oracle_ratio"] = safe_div(
            result[f"pred_{target_metric_name}_exec_value"],
            result.get(f"best_{target_metric_name}_oracle_value"),
        )
        result[f"pred_vs_nopartition_ratio"] = safe_div(
            result[f"pred_{target_metric_name}_exec_value"],
            result.get(f"{target_metric_name}_true_no_partition_baseline_value"),
        )
        result[f"pred_vs_fairpartition_ratio"] = safe_div(
            result[f"pred_{target_metric_name}_exec_value"],
            result.get(f"{target_metric_name}_fair_partition_baseline_value"),
        )

    result[f"fairpartition_vs_oracle_ratio"] = safe_div(
        result.get(f"{target_metric_name}_fair_partition_baseline_value"),
        result.get(f"best_{target_metric_name}_oracle_value"),
    )
    result[f"nopartition_vs_oracle_ratio"] = safe_div(
        result.get(f"{target_metric_name}_true_no_partition_baseline_value"),
        result.get(f"best_{target_metric_name}_oracle_value"),
    )

    # Latency comparison logic
    if oracle_latency_lookup is not None and use_two_workload_baselines:
        workload1_name, workload2_name = workload_names[:2]
        
        # Compute oracle minimum latency across all thread percentage configurations
        oracle_min_latency = None
        oracle_min_latency_w1_pct = None
        oracle_min_latency_w2_pct = None
        
        for _, row in oracle_df_for_pair.iterrows():
            w1_pct = row.get("w1_Threadpercent")
            w2_pct = row.get("w2_Threadpercent")
            
            if pd.isna(w1_pct) or pd.isna(w2_pct):
                continue
            
            latency_key = (workload1_name, workload2_name, int(w1_pct), int(w2_pct))
            latency_data = oracle_latency_lookup.get(latency_key)
            
            if latency_data is not None:
                # Extract individual latencies and average from tuple
                lat1, lat2, latency_avg = latency_data
                
                # Compute weighted latency - require steps data (no fallback)
                weighted_latency = None
                w1_steps = None
                w2_steps = None
                if oracle_steps_lookup is None:
                    print(f"Warning: No steps lookup available for weighted latency calculation for {latency_key}")
                else:
                    steps_data = oracle_steps_lookup.get(latency_key)
                    if steps_data is None:
                        print(f"Warning: Missing steps data for {latency_key}, weighted latency set to None")
                    else:
                        w1_steps, w2_steps = steps_data
                        total_steps = w1_steps + w2_steps
                        if total_steps > 0:
                            weighted_latency = (w1_steps * lat1 + w2_steps * lat2) / total_steps
                        else:
                            print(f"Warning: Total steps is zero for {latency_key}, weighted latency set to None")
                
                # Collect debug data for CSV
                if weighted_latency_debug_collector is not None:
                    weighted_latency_debug_collector.append({
                        "workload1": workload1_name,
                        "workload2": workload2_name,
                        "w1_threadpercent": int(w1_pct),
                        "w2_threadpercent": int(w2_pct),
                        "w1_steps": w1_steps,
                        "w2_steps": w2_steps,
                        "w1_latency": lat1,
                        "w2_latency": lat2,
                        "latency_avg_simple": latency_avg,
                        "weighted_latency": weighted_latency,
                        "context": "oracle_minimum_search"
                    })
                
                # Use weighted_latency for oracle minimum comparison (only if available)
                if weighted_latency is not None:
                    if oracle_min_latency is None or weighted_latency < oracle_min_latency:
                        oracle_min_latency = weighted_latency
                        oracle_min_latency_w1_pct = int(w1_pct)
                        oracle_min_latency_w2_pct = int(w2_pct)
        
        result["oracle_min_latency"] = oracle_min_latency
        result["oracle_min_latency_w1_threadpercent"] = oracle_min_latency_w1_pct
        result["oracle_min_latency_w2_threadpercent"] = oracle_min_latency_w2_pct
        
        # Compute fairpartition and nopartition latency ratios early (before rm_100partitions filtering)
        # Use the unfiltered baseline lookups to ensure (100,100) data is available
        for baseline_name, baseline_w1_pct, baseline_w2_pct in [("fairpartition", 50, 50), ("nopartition", 100, 100)]:
            # Use the unfiltered baseline lookup directly - don't check oracle_df_for_pair 
            # since it may have (100,100) filtered out
            latency_key = (workload1_name, workload2_name, baseline_w1_pct, baseline_w2_pct)
            latency_data = None
            if oracle_latency_lookup_baselines is not None:
                latency_data = oracle_latency_lookup_baselines.get(latency_key)
            
            if latency_data is not None:
                lat1, lat2, latency_avg = latency_data
                
                # Compute weighted latency using unfiltered baseline steps lookup
                weighted_latency = None
                w1_steps = None
                w2_steps = None
                if oracle_steps_lookup_baselines is not None:
                    steps_data = oracle_steps_lookup_baselines.get(latency_key)
                    if steps_data is not None:
                        w1_steps, w2_steps = steps_data
                        total_steps = w1_steps + w2_steps
                        if total_steps > 0:
                            weighted_latency = (w1_steps * lat1 + w2_steps * lat2) / total_steps
                
                # Collect debug data
                if weighted_latency_debug_collector is not None:
                    weighted_latency_debug_collector.append({
                        "workload1": workload1_name,
                        "workload2": workload2_name,
                        "w1_threadpercent": baseline_w1_pct,
                        "w2_threadpercent": baseline_w2_pct,
                        "w1_steps": w1_steps,
                        "w2_steps": w2_steps,
                        "w1_latency": lat1,
                        "w2_latency": lat2,
                        "latency_avg_simple": latency_avg,
                        "weighted_latency": weighted_latency,
                        "context": f"baseline_{baseline_name}"
                    })
                
                result[f"{baseline_name}_latency_avg"] = latency_avg
                if weighted_latency is not None and oracle_min_latency is not None:
                    result[f"{baseline_name}_vs_oracle_latency_ratio"] = weighted_latency / oracle_min_latency
                else:
                    result[f"{baseline_name}_vs_oracle_latency_ratio"] = None
            else:
                # Latency data not available in baseline lookup
                result[f"{baseline_name}_latency_avg"] = None
                result[f"{baseline_name}_vs_oracle_latency_ratio"] = None
        
        # Look up latency for each policy's selected thread percentages
        policy_configs = []
        
        # Pred policy
        if f"pred_w1_threadpercent_selected" in result and f"pred_w2_threadpercent_selected" in result:
            pred_w1 = result[f"pred_w1_threadpercent_selected"]
            pred_w2 = result[f"pred_w2_threadpercent_selected"]
            if not pd.isna(pred_w1) and not pd.isna(pred_w2):
                policy_configs.append(("pred", int(pred_w1), int(pred_w2)))
        
        # MUDI latency policy
        if "mudi_latency_w1_threadpercent" in result and "mudi_latency_w2_threadpercent" in result:
            mudi_lat_w1 = result["mudi_latency_w1_threadpercent"]
            mudi_lat_w2 = result["mudi_latency_w2_threadpercent"]
            if not pd.isna(mudi_lat_w1) and not pd.isna(mudi_lat_w2):
                policy_configs.append(("mudi_latency", int(mudi_lat_w1), int(mudi_lat_w2)))
        
        # MUDI xput policy
        if "mudi_xput_w1_threadpercent" in result and "mudi_xput_w2_threadpercent" in result:
            mudi_xput_w1 = result["mudi_xput_w1_threadpercent"]
            mudi_xput_w2 = result["mudi_xput_w2_threadpercent"]
            if not pd.isna(mudi_xput_w1) and not pd.isna(mudi_xput_w2):
                policy_configs.append(("mudi_xput", int(mudi_xput_w1), int(mudi_xput_w2)))
        
        # GSLICE policy
        if "gslice_w1_threadpercent" in result and "gslice_w2_threadpercent" in result:
            gslice_w1 = result["gslice_w1_threadpercent"]
            gslice_w2 = result["gslice_w2_threadpercent"]
            if not pd.isna(gslice_w1) and not pd.isna(gslice_w2):
                policy_configs.append(("gslice", int(gslice_w1), int(gslice_w2)))
        
        # MUXFLOW policy
        if "muxflow_w1_threadpercent" in result and "muxflow_w2_threadpercent" in result:
            muxflow_w1 = result["muxflow_w1_threadpercent"]
            muxflow_w2 = result["muxflow_w2_threadpercent"]
            if not pd.isna(muxflow_w1) and not pd.isna(muxflow_w2):
                policy_configs.append(("muxflow", int(muxflow_w1), int(muxflow_w2)))
        
        # Look up latency for each policy
        for policy_name, w1_pct, w2_pct in policy_configs:
            latency_key = (workload1_name, workload2_name, w1_pct, w2_pct)
            latency_data = oracle_latency_lookup.get(latency_key)
            
            if latency_data is not None:
                # Extract individual latencies and average from tuple
                lat1, lat2, latency_avg = latency_data
                
                # Store simple average for backward compatibility
                result[f"{policy_name}_latency_avg"] = latency_avg
                
                # Compute weighted latency - require steps data (no fallback)
                weighted_latency = None
                w1_steps = None
                w2_steps = None
                if oracle_steps_lookup is None:
                    print(f"Warning: No steps lookup available for weighted latency calculation for {policy_name} at {latency_key}")
                else:
                    steps_data = oracle_steps_lookup.get(latency_key)
                    if steps_data is None:
                        print(f"Warning: Missing steps data for {policy_name} at {latency_key}, weighted latency set to None")
                    else:
                        w1_steps, w2_steps = steps_data
                        total_steps = w1_steps + w2_steps
                        if total_steps > 0:
                            weighted_latency = (w1_steps * lat1 + w2_steps * lat2) / total_steps
                        else:
                            print(f"Warning: Total steps is zero for {policy_name} at {latency_key}, weighted latency set to None")
                
                # Collect debug data for CSV
                if weighted_latency_debug_collector is not None:
                    weighted_latency_debug_collector.append({
                        "workload1": workload1_name,
                        "workload2": workload2_name,
                        "w1_threadpercent": w1_pct,
                        "w2_threadpercent": w2_pct,
                        "w1_steps": w1_steps,
                        "w2_steps": w2_steps,
                        "w1_latency": lat1,
                        "w2_latency": lat2,
                        "latency_avg_simple": latency_avg,
                        "weighted_latency": weighted_latency,
                        "context": f"policy_{policy_name}"
                    })
                
                # Use weighted_latency for ratio comparison with oracle (only if both are available)
                if weighted_latency is not None and oracle_min_latency is not None:
                    result[f"{policy_name}_vs_oracle_latency_ratio"] = weighted_latency / oracle_min_latency
                else:
                    result[f"{policy_name}_vs_oracle_latency_ratio"] = None
            else:
                result[f"{policy_name}_latency_avg"] = None
                result[f"{policy_name}_vs_oracle_latency_ratio"] = None

    ordered_result = OrderedDict()
    for col in workload_columns:
        if col in result:
            ordered_result[col] = result.pop(col)
    if "selected_model_freq" in result:
        ordered_result["selected_model_freq"] = result.pop("selected_model_freq")

    thread_key_order = [f"best_w{idx}_threadpercent_oracle" for idx in range(1, num_workloads + 1)]
    thread_key_order += [f"pred_w{idx}_threadpercent_selected" for idx in range(1, num_workloads + 1)]
    if use_two_workload_baselines:
        thread_key_order += [f"baseline_custom_w{idx}_threadpercent" for idx in range(1, 3)]
        thread_key_order += [f"mudi_latency_w{idx}_threadpercent" for idx in range(1, 3)]
        thread_key_order += [f"mudi_xput_w{idx}_threadpercent" for idx in range(1, 3)]
        thread_key_order += [f"gslice_w{idx}_threadpercent" for idx in range(1, 3)]
        thread_key_order += [f"muxflow_w{idx}_threadpercent" for idx in range(1, 3)]
    elif num_workloads == 3:
        thread_key_order += [f"mudi_xput_w{idx}_threadpercent" for idx in range(1, num_workloads + 1)]

    for key in thread_key_order:
        if key in result:
            ordered_result[key] = result.pop(key)

    remaining_thread_keys = sorted(
        [key for key in result if "threadpercent" in key.lower() or key.endswith("_tp")]
    )
    for key in remaining_thread_keys:
        if key in result:
            ordered_result[key] = result.pop(key)

    ratio_key_order = [f"pred_vs_oracle_ratio"]
    if use_two_workload_baselines:
        ratio_key_order.append("baseline_custom_actual_xput_vs_oracle_ratio")
        ratio_key_order.append("mudi_latency_vs_oracle_ratio")
        ratio_key_order.append("mudi_xput_vs_oracle_ratio")
        ratio_key_order.append("gslice_vs_oracle_ratio")
        ratio_key_order.append("muxflow_vs_oracle_ratio")
    elif num_workloads == 3:
        ratio_key_order.append("mudi_xput_vs_oracle_ratio")
    ratio_key_order.append(f"fairpartition_vs_oracle_ratio")
    ratio_key_order.append(f"nopartition_vs_oracle_ratio")
    ratio_key_order.append(f"pred_vs_nopartition_ratio")
    ratio_key_order.append(f"pred_vs_fairpartition_ratio")

    for key in ratio_key_order:
        if key in result:
            ordered_result[key] = result.pop(key)

    # Add latency columns after ratio columns
    latency_key_order = []
    if use_two_workload_baselines:
        latency_key_order.append("oracle_min_latency")
        latency_key_order.append("oracle_min_latency_w1_threadpercent")
        latency_key_order.append("oracle_min_latency_w2_threadpercent")
        latency_key_order.append("pred_latency_avg")
        latency_key_order.append("pred_vs_oracle_latency_ratio")
        latency_key_order.append("mudi_latency_latency_avg")
        latency_key_order.append("mudi_latency_vs_oracle_latency_ratio")
        latency_key_order.append("mudi_xput_latency_avg")
        latency_key_order.append("mudi_xput_vs_oracle_latency_ratio")
        latency_key_order.append("gslice_latency_avg")
        latency_key_order.append("gslice_vs_oracle_latency_ratio")
        latency_key_order.append("muxflow_latency_avg")
        latency_key_order.append("muxflow_vs_oracle_latency_ratio")
        latency_key_order.append("fairpartition_latency_avg")
        latency_key_order.append("fairpartition_vs_oracle_latency_ratio")
        latency_key_order.append("nopartition_latency_avg")
        latency_key_order.append("nopartition_vs_oracle_latency_ratio")
    
    for key in latency_key_order:
        if key in result:
            ordered_result[key] = result.pop(key)

    for key in sorted(result.keys()):
        ordered_result[key] = result[key]

    return ordered_result

    #if not all_candidate_configs_for_pair_dfs or all_candidates_df_for_pair.empty : #if main policy did not find any config
    #    return result # Return partial results if main policy failed but custom baseline might have run
    #return result



def analyze_xput_under_powercap_policy(
    predict_xput_csv_files_dict,
    predict_power_csv_files_dict,
    model_freq_string_labels,
    model_freq_numeric_labels,
    oracle_file_path,
    static_power_limit,
    dvfs_power_limit,
    target_metric_name,
    output_base_name,
    weight,
    rm_100partitions,
    model_label_for_custom_baseline,
    mudi_latency_optimal_allocation_paths=None,
    mudi_xput_optimal_allocation_paths=None,
    mudi_xput_optimal_allocation_paths_powercap=None,
    gslice_optimal_allocation_paths=None,
    muxflow_optimal_allocation_path=None,
    oracle_latency_files=None,
    oracle_steps_files=None,
    is_plot=False,
    save_csv=True,
    is_cross_validation=False,
):
    """Main analysis entrypoint for the 'xput_under_powercap' policy."""
    # Global collection for all weighted latency debug data
    all_weighted_latency_debug_rows = []
    
    # Parse oracle latency files if provided
    oracle_latency_lookup = None
    oracle_latency_lookup_baselines = None
    if oracle_latency_files:
        oracle_latency_lookup_by_powercap = parse_oracle_latency_files(oracle_latency_files, rm_100partitions=rm_100partitions)
        # Get the latency lookup for the current dvfs_power_limit
        oracle_latency_lookup = oracle_latency_lookup_by_powercap.get(dvfs_power_limit)
        if oracle_latency_lookup is None and oracle_latency_lookup_by_powercap:
            print(f"Warning: No latency data found for power cap {dvfs_power_limit}")
        
        # Also load unfiltered version for baseline lookups (fairpartition/nopartition)
        oracle_latency_lookup_baselines_by_powercap = parse_oracle_latency_files(oracle_latency_files, rm_100partitions=False)
        oracle_latency_lookup_baselines = oracle_latency_lookup_baselines_by_powercap.get(dvfs_power_limit)
    
    # Parse oracle steps files if provided
    oracle_steps_lookup = None
    oracle_steps_lookup_baselines = None
    if oracle_steps_files:
        oracle_steps_lookup_by_powercap = parse_oracle_steps_files(oracle_steps_files, rm_100partitions=rm_100partitions)
        # Get the steps lookup for the current dvfs_power_limit
        oracle_steps_lookup = oracle_steps_lookup_by_powercap.get(dvfs_power_limit)
        if oracle_steps_lookup is None and oracle_steps_lookup_by_powercap:
            print(f"Warning: No steps data found for power cap {dvfs_power_limit}")
        
        # Also load unfiltered version for baseline lookups (fairpartition/nopartition)
        oracle_steps_lookup_baselines_by_powercap = parse_oracle_steps_files(oracle_steps_files, rm_100partitions=False)
        oracle_steps_lookup_baselines = oracle_steps_lookup_baselines_by_powercap.get(dvfs_power_limit)
    
    try:
        all_oracle_df_original = pd.read_csv(oracle_file_path)
    except FileNotFoundError:
        print(f"Error: Oracle file not found at {oracle_file_path}")
        return pd.DataFrame(), 0, 0

    all_oracle_df = all_oracle_df_original.copy()
    if "Thread_combination" not in all_oracle_df.columns:
        print("Error: 'Thread_combination' column missing in oracle file.")
        return pd.DataFrame(), 0, 0

    workload_columns_original = sorted(
        [col for col in all_oracle_df.columns if col.startswith("Workload")],
        key=lambda name: int(name.replace("Workload", "")),
    )
    if not workload_columns_original:
        print("Error: No workload columns detected in oracle file.")
        return pd.DataFrame(), 0, 0

    num_workloads = len(workload_columns_original)
    workload_columns = build_workload_column_names(num_workloads)
    threadpercent_columns = build_threadpercent_column_names(num_workloads)

    rename_map = {orig: dest for orig, dest in zip(workload_columns_original, workload_columns)}
    rename_map.update(
        {
            "Power": "Power_oracle_actual",
            "weight_Throughput_sum": "weight_Throughput_sum_oracle_actual",
            "Energy": "Energy_oracle_actual",
            "Duration": "Duration_oracle_actual",
        }
    )
    all_oracle_df = all_oracle_df.rename(columns=rename_map)

    thread_values_series = all_oracle_df["Thread_combination"].apply(get_thread_num_from_str)
    for idx, column_name in enumerate(threadpercent_columns):
        all_oracle_df[column_name] = thread_values_series.apply(
            lambda values: values[idx] if len(values) > idx else None
        )

    all_oracle_df.dropna(subset=threadpercent_columns, inplace=True)
    for column_name in threadpercent_columns:
        all_oracle_df[column_name] = all_oracle_df[column_name].astype(int)

    true_no_partition_mask = (all_oracle_df[threadpercent_columns] == 100).all(axis=1)
    true_no_partition_all_pairs_df = all_oracle_df[true_no_partition_mask]

    if rm_100partitions:
        all_oracle_df = all_oracle_df[~true_no_partition_mask]

    mudi_xput_multiworkload_lookup = {}
    if num_workloads == 2:
        mudi_latency_thread_percent_lookup_by_freq = load_mudi_latency_allocations_by_freq(
            mudi_latency_optimal_allocation_paths
        )
        mudi_xput_thread_percent_lookup_by_freq = load_mudi_xput_allocations_by_freq(
            mudi_xput_optimal_allocation_paths
        )
        gslice_thread_percent_lookup_by_freq = load_gslice_allocations_by_freq(
            gslice_optimal_allocation_paths
        )
        muxflow_thread_percent_lookup = load_muxflow_allocations(muxflow_optimal_allocation_path)
    elif num_workloads == 3:
        mudi_latency_thread_percent_lookup_by_freq = {}
        mudi_xput_thread_percent_lookup_by_freq = {}
        gslice_thread_percent_lookup_by_freq = {}
        muxflow_thread_percent_lookup = {}
        mudi_xput_multiworkload_lookup = load_mudi_xput_multiworkload_allocations(
            mudi_xput_optimal_allocation_paths_powercap,
            dvfs_power_limit,
        )
    else:
        mudi_latency_thread_percent_lookup_by_freq = {}
        mudi_xput_thread_percent_lookup_by_freq = {}
        gslice_thread_percent_lookup_by_freq = {}
        muxflow_thread_percent_lookup = {}

    os.makedirs(output_base_name, exist_ok=True)
    predict_xput_cache = {}
    predict_power_cache = {}

    def _load_csv_with_cache(csv_path, cache_dict):
        if csv_path not in cache_dict:
            cache_dict[csv_path] = pd.read_csv(csv_path)
        return cache_dict[csv_path]

    def _filter_df_for_combo(df, combo):
        if df.empty:
            return df
        mask = pd.Series(True, index=df.index)
        for col, name in zip(workload_columns, combo):
            mask &= df[col] == name
        return df[mask]

    def _combo_exists_in_predictions(workload_combo):
        if num_workloads != 2:
            return False
        pair_w1, pair_w2 = workload_combo[:2]
        for freq_numeric in model_freq_numeric_labels:
            workload_map = predict_xput_csv_files_dict.get(freq_numeric, {})
            if not workload_map:
                continue
            for workload_key in (pair_w1, pair_w2):
                csv_path = workload_map.get(workload_key)
                if not csv_path or not os.path.exists(csv_path):
                    continue
                try:
                    df = _load_csv_with_cache(csv_path, predict_xput_cache)
                except Exception as exc:
                    print(f"Warning: Unable to read prediction file {csv_path}: {exc}")
                    continue
                required_cols = {"workload1", "workload2"}
                if not required_cols.issubset(df.columns):
                    continue
                matches = df[(df["workload1"] == pair_w1) & (df["workload2"] == pair_w2)]
                if not matches.empty:
                    return True
        return False

    workload_combinations = [
        tuple(row)
        for row in all_oracle_df[workload_columns].drop_duplicates().values
    ]

    aggregated_results_all_pairs = []
    total_pairs_no_pred_config = 0
    total_freq1530_baseline_power_violations = 0

    for combo in workload_combinations:
        combo_label = ", ".join(combo)
        oracle_df_current_combo = _filter_df_for_combo(all_oracle_df, combo)
        if oracle_df_current_combo.empty:
            print(f"Warning: No oracle data found for workloads [{combo_label}]. Skipping this combination.")
            continue

        if is_cross_validation and num_workloads == 2 and not _combo_exists_in_predictions(combo):
            print(
                f"Skipping workloads [{combo_label}] for cross-validation: predictions missing across frequencies."
            )
            continue

        no_partition_oracle_config_current_combo = _filter_df_for_combo(
            true_no_partition_all_pairs_df, combo
        )

        loaded_xput_dfs_for_combo = []
        loaded_power_dfs_for_combo = []
        active_labels_for_combo = []

        for freq_numeric, freq_label_str in zip(model_freq_numeric_labels, model_freq_string_labels):
            primary_workload_for_path = combo[0]
            xput_path = predict_xput_csv_files_dict.get(freq_numeric, {}).get(primary_workload_for_path)
            power_path = predict_power_csv_files_dict.get(freq_numeric, {}).get(primary_workload_for_path)

            if not xput_path or not power_path or not os.path.exists(xput_path) or not os.path.exists(power_path):
                print(
                    f"Info: Missing prediction CSVs for workloads [{combo_label}] at freq {freq_label_str} "
                    f"(lookup key {primary_workload_for_path})"
                )
                continue

            try:
                df_xput = _load_csv_with_cache(xput_path, predict_xput_cache).copy()
                df_power = _load_csv_with_cache(power_path, predict_power_cache).copy()
            except Exception as exc:
                print(
                    f"Error loading prediction files for workloads [{combo_label}], freq {freq_label_str}: {exc}"
                )
                continue

            loaded_xput_dfs_for_combo.append(df_xput)
            loaded_power_dfs_for_combo.append(df_power)
            active_labels_for_combo.append(freq_label_str)

        if not loaded_xput_dfs_for_combo:
            print(
                f"Skipping workloads [{combo_label}] due to missing prediction files for all frequencies."
            )
            total_pairs_no_pred_config += 1
            continue

        pair_result_dict = get_pred_oracle_target_comparison_powercap(
            workload_names=combo,
            pred_xput_dfs_list=loaded_xput_dfs_for_combo,
            pred_power_dfs_list=loaded_power_dfs_for_combo,
            model_freq_labels=active_labels_for_combo,
            oracle_df_for_pair=oracle_df_current_combo,
            no_partition_oracle_config_for_pair=no_partition_oracle_config_current_combo,
            static_power_limit=static_power_limit,
            target_metric_name=target_metric_name,
            weight=weight,
            output_dir_for_plots=output_base_name,
            rm_100partitions=rm_100partitions,
            model_label_for_custom_baseline=model_label_for_custom_baseline,
            threadpercent_columns=threadpercent_columns,
            workload_columns=workload_columns,
            num_workloads=num_workloads,
            mudi_latency_thread_percent_lookup_by_freq=mudi_latency_thread_percent_lookup_by_freq,
            mudi_xput_thread_percent_lookup_by_freq=mudi_xput_thread_percent_lookup_by_freq,
            gslice_thread_percent_lookup_by_freq=gslice_thread_percent_lookup_by_freq,
            muxflow_thread_percent_lookup=muxflow_thread_percent_lookup,
            mudi_xput_multiworkload_allocations=mudi_xput_multiworkload_lookup,
            oracle_latency_lookup=oracle_latency_lookup,
            oracle_steps_lookup=oracle_steps_lookup,
            oracle_latency_lookup_baselines=oracle_latency_lookup_baselines,
            oracle_steps_lookup_baselines=oracle_steps_lookup_baselines,
            weighted_latency_debug_collector=all_weighted_latency_debug_rows,
            is_plot=is_plot,
        )

        if pair_result_dict:
            aggregated_results_all_pairs.append(pair_result_dict)
            if (
                num_workloads == 2
                and pair_result_dict.get("baseline_custom_exceeds_cap") is True
            ):
                total_freq1530_baseline_power_violations += 1
        else:
            total_pairs_no_pred_config += 1

    final_summary_df = pd.DataFrame(aggregated_results_all_pairs)

    if final_summary_df.empty:
        print("No prediction results to process after iterating all workload combinations.")
        return final_summary_df, total_pairs_no_pred_config, total_freq1530_baseline_power_violations

    if save_csv:
        combined_filename = os.path.join(output_base_name, f"summary_{target_metric_name}_all_pairs.csv")
        final_summary_df.to_csv(combined_filename, index=False)
        print(f"Saved combined summary to {combined_filename}")

        mean_summary = final_summary_df.mean(numeric_only=True)
        mean_filename = os.path.join(output_base_name, f"summary_{target_metric_name}_mean_ratios.csv")
        mean_summary.to_csv(mean_filename, header=["mean_value"])
        print(f"Mean summary statistics:\n{mean_summary}")
        print(f"Saved mean summary to {mean_filename}")

        def _sanitize_filename(raw_name: str) -> str:
            if raw_name is None:
                return "workload"
            safe = "".join(
                ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
                for ch in str(raw_name)
            )
            return safe or "workload"

        safe_name_counts = defaultdict(int)
        for workload_col in workload_columns:
            if workload_col not in final_summary_df.columns:
                continue
            grouped = final_summary_df.groupby(workload_col)
            for workload_name, workload_subset in grouped:
                base_safe_name = _sanitize_filename(workload_name)
                count = safe_name_counts[base_safe_name]
                safe_name_counts[base_safe_name] = count + 1
                safe_name = base_safe_name if count == 0 else f"{base_safe_name}_{count}"

                workload_all_pairs_path = os.path.join(output_base_name, f"{safe_name}_all_df.csv")
                workload_subset.to_csv(workload_all_pairs_path, index=False)

                workload_mean = workload_subset.mean(numeric_only=True)
                workload_mean.loc["pair_count"] = len(workload_subset)
                workload_mean_path = os.path.join(output_base_name, f"{safe_name}_mean.csv")
                workload_mean.to_csv(workload_mean_path, header=["mean_value"])

    print(
        "Total workload combinations where no suitable predicted configuration was found under powercap (main policy): "
        f"{total_pairs_no_pred_config}"
    )
    print(
        f"Total workload combinations where '{model_label_for_custom_baseline}_only_max_pred_xput' baseline "
        f"exceeded static power limit: {total_freq1530_baseline_power_violations}"
    )
    
    # Save single weighted latency debug CSV with all pairs and partitions
    if save_csv and all_weighted_latency_debug_rows:
        try:
            debug_df = pd.DataFrame(all_weighted_latency_debug_rows)
            debug_path = os.path.join(output_base_name, "weighted_latency_debug_all_pairs.csv")
            debug_df.to_csv(debug_path, index=False)
            print(f"Saved weighted latency debug CSV with {len(debug_df)} entries (all pairs, all partitions) to {debug_path}")
        except Exception as e:
            print(f"Warning: Failed to save weighted latency debug CSV: {e}")
    
    return final_summary_df, total_pairs_no_pred_config, total_freq1530_baseline_power_violations


# --- Placeholder for original get_pred_oracle_target_comparison ---
def get_pred_oracle_target_comparison(pred_df, pred_power_df, all_oracle_df, powercap_limit_df, target, static_power_limit, weight, output_dir,rm_100partitions, power_threshold=0.75, is_plot=False):
    print(f"INFO: Using original get_pred_oracle_target_comparison for target: {target}")
    
    no_partition_baseline = "(w1_100, w2_100)"
    fair_partition_baseline = "(w1_50, w2_50)"
    
    combined_results = []
    # Make copies to avoid modifying original DataFrames passed to the function
    pred_df_copy = pred_df.copy()
    pred_power_df_copy = pred_power_df.copy()
    all_oracle_df_copy = all_oracle_df.copy()


    if rm_100partitions:
        if 'w1_Threadpercent' in pred_df_copy.columns and 'w2_Threadpercent' in pred_df_copy.columns:
            pred_df_copy = pred_df_copy[~((pred_df_copy['w1_Threadpercent'] == 100) & (pred_df_copy['w2_Threadpercent'] == 100))]
        if 'w1_Threadpercent' in pred_power_df_copy.columns and 'w2_Threadpercent' in pred_power_df_copy.columns:
            pred_power_df_copy = pred_power_df_copy[~((pred_power_df_copy['w1_Threadpercent'] == 100) & (pred_power_df_copy['w2_Threadpercent'] == 100))]

    # Ensure Thread_combination parsing happens on the copy
    if 'Thread_combination' not in all_oracle_df_copy.columns:
        print("Error in get_pred_oracle_target_comparison: 'Thread_combination' missing from oracle_df.")
        return pd.DataFrame(), pd.DataFrame(), 0 # Return empty DFs and 0 invalid

    all_oracle_df_copy["w1_Threadpercent"] = all_oracle_df_copy["Thread_combination"].apply(lambda x: get_thread_num_from_str(x)[0])
    all_oracle_df_copy["w2_Threadpercent"] = all_oracle_df_copy["Thread_combination"].apply(lambda x: get_thread_num_from_str(x)[1])
    all_oracle_df_copy.dropna(subset=['w1_Threadpercent', 'w2_Threadpercent'], inplace=True) # Drop if parsing failed
    all_oracle_df_copy["w1_Threadpercent"] = all_oracle_df_copy["w1_Threadpercent"].astype(int)
    all_oracle_df_copy["w2_Threadpercent"] = all_oracle_df_copy["w2_Threadpercent"].astype(int)
    
    all_oracle_df_copy = all_oracle_df_copy.rename(columns={"Workload1": "workload1", "Workload2": "workload2", 
                                                  "Power": "Power_oracle_actual", 
                                                  "weight_Throughput_sum": "weight_Throughput_sum_oracle_actual",
                                                  "Energy": "Energy_oracle_actual",
                                                  "Duration": "Duration_oracle_actual"})

    all_df = pd.merge(all_oracle_df_copy, pred_df_copy,
                             on=["workload1", "workload2", "w1_Threadpercent", "w2_Threadpercent"],
                             how="inner", suffixes=('_oracle', '_xput_pred_model'))
    
    pred_power_df_copy = pred_power_df_copy.rename(columns={"y_pred": "y_pred_power", "y_test": "y_test_power"})
    all_df = pd.merge(all_df, pred_power_df_copy[['workload1', 'workload2', 'w1_Threadpercent', 'w2_Threadpercent', 'y_pred_power']],
                             on=["workload1", "workload2", "w1_Threadpercent", "w2_Threadpercent"],
                             how="inner")
    all_df = all_df.rename(columns={'y_pred_xput_pred_model': 'y_pred'})

    if target == "EDP":
        all_oracle_df_copy[target] = all_oracle_df_copy["Energy_oracle_actual"] /  all_oracle_df_copy["weight_Throughput_sum_oracle_actual"]
        all_df[target] = all_df["Energy_oracle_actual"] / all_df["weight_Throughput_sum_oracle_actual"]
        all_df[f"pred_{target}"] = all_df["y_pred_power"] * all_df["Duration_oracle_actual"] /  all_df["y_pred"]
    elif target == "xput_per_joule":
        all_oracle_df_copy["Energy_oracle_actual_weighted"] = all_oracle_df_copy["Energy_oracle_actual"] / weight
        all_df["Energy_oracle_actual_weighted"] = all_df["Energy_oracle_actual"] / weight
        all_oracle_df_copy[target] = all_oracle_df_copy["weight_Throughput_sum_oracle_actual"] / all_oracle_df_copy["Energy_oracle_actual_weighted"]
        all_df[target] = all_df["weight_Throughput_sum_oracle_actual"] / all_df["Energy_oracle_actual_weighted"]
        all_df[f"pred_{target}"] = all_df["y_pred"] / (all_df["y_pred_power"] * all_df["Duration_oracle_actual"] / weight)
    elif target == "xput_per_power":
        all_oracle_df_copy[target] = all_oracle_df_copy["weight_Throughput_sum_oracle_actual"] / all_oracle_df_copy["Power_oracle_actual"]
        all_df[target] = all_df["weight_Throughput_sum_oracle_actual"] / all_df["Power_oracle_actual"]
        all_df[f"pred_{target}"] = all_df["y_pred"] / all_df["y_pred_power"]
    elif target == "power_per_xput":
        all_oracle_df_copy[target] = all_oracle_df_copy["Power_oracle_actual"] / all_oracle_df_copy["weight_Throughput_sum_oracle_actual"]
        all_df[target] = all_df["Power_oracle_actual"] / all_df["weight_Throughput_sum_oracle_actual"]
        all_df[f"pred_{target}"] = all_df["y_pred_power"] / all_df["y_pred"]
    elif target == "max_xput":
        all_oracle_df_copy[target] = all_oracle_df_copy["weight_Throughput_sum_oracle_actual"]
        all_df[target] = all_df["weight_Throughput_sum_oracle_actual"]
        all_df[f"pred_{target}"] = all_df["y_pred"]
    elif target == "energy_plus_duration":
        all_oracle_df_copy[target] = all_oracle_df_copy["Energy_oracle_actual"] + weight * all_oracle_df_copy["Duration_oracle_actual"]
        all_df[target] = all_df["Energy_oracle_actual"] + weight * all_df["Duration_oracle_actual"]
        all_df[f"pred_{target}"] = all_df["y_pred_power"] * all_df["Duration_oracle_actual"] + weight * all_df["Duration_oracle_actual"]
    # xput_under_powercap is NOT handled by this function path
    elif target == "xput_under_powercap":
        print("WARNING: 'xput_under_powercap' should be handled by 'get_pred_oracle_target_comparison_powercap'. This path in original function is likely an error.")
        # Fallback or error, as this logic is specific
        all_oracle_df_copy[target] = all_oracle_df_copy["weight_Throughput_sum_oracle_actual"]
        all_df[target] = all_df["weight_Throughput_sum_oracle_actual"]
        all_df[f"pred_{target}"] = all_df["y_pred"]
    else:
        raise ValueError(f"Unsupported target value {target} in original comparison function")

    workload_pairs = all_df[["workload1", "workload2"]].drop_duplicates().values
    num_invalid_pre_configs = 0

    for workload1, workload2 in workload_pairs:
        filtered_all_df = all_df[(all_df["workload1"] == workload1) & (all_df["workload2"] == workload2)]
        if filtered_all_df.empty: continue

        select_idx, select_pred_idx = None, None
        if target in ["EDP", "energy_plus_duration", "power_per_xput"]:
            if not filtered_all_df[target].empty: select_idx = filtered_all_df[target].idxmin()
            if not filtered_all_df[f"pred_{target}"].empty: select_pred_idx = filtered_all_df[f"pred_{target}"].idxmin()
        elif target in ["xput_per_joule", "xput_per_power", "max_xput", "xput_under_powercap"]: # xput_under_powercap handled by other func
            if not filtered_all_df[target].empty: select_idx = filtered_all_df[target].idxmax()
            if not filtered_all_df[f"pred_{target}"].empty: select_pred_idx = filtered_all_df[f"pred_{target}"].idxmax()
        
        if pd.isna(select_idx) or pd.isna(select_pred_idx):
            num_invalid_pre_configs +=1
            continue
            
        best_oracle_target_value = filtered_all_df.loc[select_idx, target]
        best_oracle_throughput = filtered_all_df.loc[select_idx, "weight_Throughput_sum_oracle_actual"]
        best_oracle_power = filtered_all_df.loc[select_idx, "Power_oracle_actual"]
        best_w1_threadpercent = filtered_all_df.loc[select_idx, "w1_Threadpercent"]
        best_w2_threadpercent = filtered_all_df.loc[select_idx, "w2_Threadpercent"]
        
        pred_pred_value = filtered_all_df.loc[select_pred_idx, f"pred_{target}"]
        pred_exec_value = filtered_all_df.loc[select_pred_idx, target]
        pred_exec_throughput = filtered_all_df.loc[select_pred_idx, "weight_Throughput_sum_oracle_actual"]
        pred_exec_power = filtered_all_df.loc[select_pred_idx, "Power_oracle_actual"]
        pred_w1_threadpercent = filtered_all_df.loc[select_pred_idx, "w1_Threadpercent"]
        pred_w2_threadpercent = filtered_all_df.loc[select_pred_idx, "w2_Threadpercent"]

        oracle_df_current_pair_for_baselines = all_oracle_df_copy[
            (all_oracle_df_copy["workload1"] == workload1) & (all_oracle_df_copy["workload2"] == workload2)
        ]
        no_partition_baseline_row = oracle_df_current_pair_for_baselines[oracle_df_current_pair_for_baselines["Thread_combination"] == no_partition_baseline]
        fair_partition_baseline_row = oracle_df_current_pair_for_baselines[oracle_df_current_pair_for_baselines["Thread_combination"] == fair_partition_baseline]

        no_partition_baseline_value = no_partition_baseline_row[target].iloc[0] if not no_partition_baseline_row.empty else None
        fair_partition_baseline_value = fair_partition_baseline_row[target].iloc[0] if not fair_partition_baseline_row.empty else None
        
        current_results= {
            "workload1": workload1, "workload2": workload2,
            "best_w1_threadpercent": best_w1_threadpercent, "best_w2_threadpercent": best_w2_threadpercent,
            "pred_w1_threadpercent": pred_w1_threadpercent, "pred_w2_threadpercent": pred_w2_threadpercent,
            f"best_{target}_oracle_value": best_oracle_target_value, 
            f"best_{target}_oracle_throughput": best_oracle_throughput,
            f"best_{target}_oracle_power": best_oracle_power,
            f"pred_{target}_exec_value" : pred_exec_value, 
            f"pred_{target}_exec_throughput": pred_exec_throughput,
            f"pred_{target}_exec_power": pred_exec_power,
            f"pred_{target}_pred_value": pred_pred_value,
            "powercap_limit": static_power_limit if target == "xput_under_powercap" else None,
            f"{target}_no_partition_baseline_value": no_partition_baseline_value,
            f"{target}_fair_partition_baseline_value": fair_partition_baseline_value,
            f"pred_vs_oracle_ratio": safe_div(pred_exec_value, best_oracle_target_value),
            f"pred_vs_nopartition_ratio": safe_div(pred_exec_value, no_partition_baseline_value),
            f"pred_vs_fairpartition_ratio": safe_div(pred_exec_value, fair_partition_baseline_value),
            f"fairpartition_vs_oracle_ratio": safe_div(fair_partition_baseline_value, best_oracle_target_value),
        }
        combined_results.append(current_results)

    pred_df_processed = pd.DataFrame(combined_results)
    return pred_df_processed, all_df, num_invalid_pre_configs


def parse_oracle_latency_files(oracle_latency_path_dict, rm_100partitions=False):
    """
    Parse oracle latency CSV files and build latency lookup dictionaries.
    
    Args:
        oracle_latency_path_dict: Dict mapping power_cap to list of CSV file paths
                                 Format: {power_cap: [file_paths]}
        rm_100partitions: If True, exclude (100, 100) partition entries
    
    Returns:
        Dict mapping power_cap to latency lookup dict
        Format: {power_cap: {(w1, w2, w1_pct, w2_pct): latency_avg}}
    """
    import re
    
    result = {}
    
    for power_cap, file_paths in oracle_latency_path_dict.items():
        if not file_paths:
            continue
            
        latency_lookup = {}
        
        for file_path in file_paths:
            if not file_path or not os.path.exists(file_path):
                print(f"Warning: Oracle latency file not found at {file_path}")
                continue
                
            try:
                df = pd.read_csv(file_path)
            except Exception as exc:
                print(f"Warning: Failed to read oracle latency file {file_path}: {exc}")
                continue
            
            # Expected columns: workload1, workload2, freq1, freq2, "(w1_10, w2_90)", "(w1_20, w2_80)", ...
            if "workload1" not in df.columns or "workload2" not in df.columns:
                print(f"Warning: Missing workload columns in {file_path}")
                continue
            
            # Find thread percentage columns - they should be in format "(w1_X, w2_Y)"
            thread_pct_pattern = re.compile(r'\(w1_(\d+),\s*w2_(\d+)\)')
            thread_cols = []
            for col in df.columns:
                match = thread_pct_pattern.match(col)
                if match:
                    w1_pct = int(match.group(1))
                    w2_pct = int(match.group(2))
                    thread_cols.append((col, w1_pct, w2_pct))
            
            if not thread_cols:
                print(f"Warning: No thread percentage columns found in {file_path}")
                continue
            
            # Parse each row
            for _, row in df.iterrows():
                w1_name = row["workload1"]
                w2_name = row["workload2"]
                
                if pd.isna(w1_name) or pd.isna(w2_name):
                    continue
                
                # Parse each thread split configuration
                for col_name, w1_pct, w2_pct in thread_cols:
                    cell_value = row.get(col_name)
                    
                    if pd.isna(cell_value):
                        continue
                    
                    # Parse latency tuple: "(lat1, lat2)" -> extract floats
                    latency_pattern = re.compile(r'\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)')
                    match = latency_pattern.match(str(cell_value))
                    
                    if match:
                        try:
                            lat1 = float(match.group(1))
                            lat2 = float(match.group(2))
                            latency_avg = (lat1 + lat2) / 2.0
                            
                            # Skip (100, 100) partitions if rm_100partitions is True
                            if rm_100partitions and w1_pct == 100 and w2_pct == 100:
                                continue
                            
                            # Store in lookup (lat1, lat2, avg)
                            key = (w1_name, w2_name, w1_pct, w2_pct)
                            latency_lookup[key] = (lat1, lat2, latency_avg)
                        except ValueError as e:
                            print(f"Warning: Failed to parse latency values in {file_path}: {e}")
                            continue
        
        if latency_lookup:
            result[power_cap] = latency_lookup
            filter_msg = " (filtered 100,100 partitions)" if rm_100partitions else ""
            print(f"Loaded {len(latency_lookup)} latency entries for power cap {power_cap}{filter_msg}")
    
    return result


def parse_oracle_steps_files(oracle_steps_path_dict, rm_100partitions=False):
    """
    Parse oracle steps CSV files and build steps lookup dictionaries.
    
    Args:
        oracle_steps_path_dict: Dict mapping power_cap to list of CSV file paths
                                Format: {power_cap: [file_paths]}
        rm_100partitions: If True, exclude (100, 100) partition entries
    
    Returns:
        Dict mapping power_cap to steps lookup dict
        Format: {power_cap: {(w1, w2, w1_pct, w2_pct): (w1_steps, w2_steps)}}
    """
    import re
    
    result = {}
    
    for power_cap, file_paths in oracle_steps_path_dict.items():
        if not file_paths:
            continue
            
        steps_lookup = {}
        
        for file_path in file_paths:
            if not file_path or not os.path.exists(file_path):
                print(f"Warning: Oracle steps file not found at {file_path}")
                continue
                
            try:
                df = pd.read_csv(file_path)
            except Exception as exc:
                print(f"Warning: Failed to read oracle steps file {file_path}: {exc}")
                continue
            
            # Expected columns: workload1, workload2, freq1, freq2, "(w1_10, w2_90)", "(w1_20, w2_80)", ...
            if "workload1" not in df.columns or "workload2" not in df.columns:
                print(f"Warning: Missing workload columns in {file_path}")
                continue
            
            # Find thread percentage columns - they should be in format "(w1_X, w2_Y)"
            thread_pct_pattern = re.compile(r'\(w1_(\d+),\s*w2_(\d+)\)')
            thread_cols = []
            for col in df.columns:
                match = thread_pct_pattern.match(col)
                if match:
                    w1_pct = int(match.group(1))
                    w2_pct = int(match.group(2))
                    thread_cols.append((col, w1_pct, w2_pct))
            
            if not thread_cols:
                print(f"Warning: No thread percentage columns found in {file_path}")
                continue
            
            # Parse each row
            for _, row in df.iterrows():
                w1_name = row["workload1"]
                w2_name = row["workload2"]
                
                if pd.isna(w1_name) or pd.isna(w2_name):
                    continue
                
                # Parse each thread split configuration
                for col_name, w1_pct, w2_pct in thread_cols:
                    cell_value = row.get(col_name)
                    
                    if pd.isna(cell_value):
                        continue
                    
                    # Parse steps tuple: "(steps1, steps2)" -> extract floats
                    steps_pattern = re.compile(r'\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)')
                    match = steps_pattern.match(str(cell_value))
                    
                    if match:
                        try:
                            w1_steps = float(match.group(1))
                            w2_steps = float(match.group(2))
                            
                            # Skip (100, 100) partitions if rm_100partitions is True
                            if rm_100partitions and w1_pct == 100 and w2_pct == 100:
                                continue
                            
                            # Store in lookup
                            key = (w1_name, w2_name, w1_pct, w2_pct)
                            steps_lookup[key] = (w1_steps, w2_steps)
                        except ValueError as e:
                            print(f"Warning: Failed to parse steps values in {file_path}: {e}")
                            continue
        
        if steps_lookup:
            result[power_cap] = steps_lookup
            filter_msg = " (filtered 100,100 partitions)" if rm_100partitions else ""
            print(f"Loaded {len(steps_lookup)} steps entries for power cap {power_cap}{filter_msg}")
    
    return result


def analyze_predictions_target_with_baselines(
    predict_xput_dir, predict_power_dir, dvfs_file, temporal_file, 
    powercap_limit_file, static_power_limit, 
    pred_xput_file_common_name, pred_power_file_common_name,  
    target, output_base_name, weight, dvfs_oracle, 
    rm_100partitions, is_plot=False, save_csv=True
):
    all_dvfs_df_original = pd.read_csv(dvfs_file)
    
    os.makedirs(output_base_name, exist_ok=True) 
    all_final_dfs = []
    all_invald_pred_pairs_counts = []

    for root, _, files in os.walk(predict_xput_dir):
        if pred_xput_file_common_name in files:
            xput_file = os.path.join(root, pred_xput_file_common_name)
            relative_path = os.path.relpath(root, predict_xput_dir)
            power_root = os.path.join(predict_power_dir, relative_path)
            power_file = os.path.join(power_root, pred_power_file_common_name)

            if os.path.exists(power_file):
                print(f"Processing (single model policy: {target}): {relative_path}")
                pred_df = pd.read_csv(xput_file)
                pred_power_df = pd.read_csv(power_file)
                workload_folder_name = os.path.basename(relative_path) if relative_path else "overall"
                
                current_oracle_df = all_dvfs_df_original.copy()

                final_df, _, num_invalid_pred_configs = get_pred_oracle_target_comparison(
                    pred_df, pred_power_df, current_oracle_df, None, 
                    target, static_power_limit, weight, 
                    output_dir=os.path.join(output_base_name, workload_folder_name),
                    rm_100partitions=rm_100partitions, is_plot=is_plot
                )

                if not final_df.empty:
                    all_final_dfs.append(final_df)
                    if save_csv:
                        current_output_path = os.path.join(output_base_name, workload_folder_name)
                        os.makedirs(current_output_path, exist_ok=True)
                        final_df.to_csv(os.path.join(current_output_path, f"{workload_folder_name}_results.csv"), index=False)
                        final_df_mean = final_df.mean(numeric_only=True)
                        print(f"Mean of {workload_folder_name}:\n{final_df_mean}")
                        final_df_mean.to_csv(os.path.join(current_output_path, f"{workload_folder_name}_mean.csv"), index=True)
                else:
                    print(f"No data generated for {relative_path}")
                all_invald_pred_pairs_counts.append({'path': relative_path, 'invalid_count': num_invalid_pred_configs})
            else:
                print(f"Warning: No matching power file found for {xput_file} at {power_file}")
    
    if not all_final_dfs:
        print(f"No valid prediction files processed for target {target} using original method.")
        return pd.DataFrame(), pd.DataFrame(all_invald_pred_pairs_counts)

    combined_df = pd.concat(all_final_dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["workload1", "workload2"], keep="last")
    
    if save_csv:
        combined_df.to_csv(os.path.join(output_base_name, f"summary_{target}_combined_all_workloads.csv"), index=False)
        mean_ratios = combined_df.mean(numeric_only=True)
        mean_ratios.to_csv(os.path.join(output_base_name, f"summary_{target}_mean_ratios_all_workloads.csv"), index=True)
        print(f"Overall Mean ratios for target {target}:\n{mean_ratios}")
        
        invalid_counts_df = pd.DataFrame(all_invald_pred_pairs_counts)
        invalid_counts_df.to_csv(os.path.join(output_base_name, f"summary_{target}_invalid_pred_counts.csv"), index=False)

    return combined_df, pd.DataFrame(all_invald_pred_pairs_counts)


# --- Main execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recompute the PEACE paper policy summaries.")
    parser.add_argument("--power-limit", type=int, choices=(60, 100, 200, 250), default=60)
    parser.add_argument("--power-epsilon", type=int, default=None)
    parser.add_argument("--combinations", type=int, choices=(2, 3), default=2)
    parser.add_argument("--cross-validation", action="store_true")
    parser.add_argument("--fold-size", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    target = "xput_under_powercap" 
    # target = "max_xput" # For testing the other policy path

    n_combinations = args.combinations
    weight = 1.0
    is_plot = False
    rm_100partitions = True
    save_csv = True
    set_power_limit = args.power_limit
    power_epsilon = args.power_epsilon
    if power_epsilon is None:
        power_epsilon = int(round(set_power_limit * 0.10))
    static_power_limit = set_power_limit + power_epsilon

    """CROSS-VALIDATE arguments
    is_cross_validation = True
    CROSS_MODEL =  extratrees
    cross_num_testsets = [i for i in range(1,10)] # trainining set ratio frpm 10,20,...90%
    """
    is_cross_validation = args.cross_validation
    cross_models = ["extratrees"]
    cross_num_testsets = [args.fold_size]
    #prediction filenames
    PRED_XPUT_FILE_COMMON_NAME = "pred_separate_throughputpower_regression.csv"
    PRED_POWER_FILE_COMMON_NAME = "pred_power_regression.csv"
    #output name 
    scenario_tag = "crossvalid" if is_cross_validation else "unseen"
    output_base_name_dir = args.output_dir or Path(
        f"./reproduced_{target}_comb{n_combinations}_mergecudaDL_{scenario_tag}_"
        f"multi_freq_powercap{set_power_limit}_epsilon{power_epsilon}_pred_vs_baselines_dvfs"
    )
    model_freq_numeric_labels = [300, 900, 1530]

    # Initialize defaults so we always have defined variables for args logging.
    CROSS_PREDICT_XPUT_DIR_FREQ300 = ""
    CROSS_PREDICT_POWER_DIR_FREQ300 = ""
    CROSS_PREDICT_XPUT_DIR_FREQ900 = ""
    CROSS_PREDICT_POWER_DIR_FREQ900 = ""
    CROSS_PREDICT_XPUT_DIR_FREQ1530 = ""
    CROSS_PREDICT_POWER_DIR_FREQ1530 = ""
    MUDI_LATENCY_OPTIMAL_ALLOCATION_FILES = {}
    MUDI_XPUT_OPTIMAL_ALLOCATION_FILES = {}
    MUDI_XPUT_OPTIMAL_ALLOCATION_FILES_POWERCAP = {}
    GSLICE_OPTIMAL_ALLOCATION_FILES = {}
    MUXFLOW_OPTIMAL_ALLOCATION_FILES = {}
    muxflow_optimal_allocation_path = None
    ORACLE_PATH = {}

    if n_combinations == 2:
        experiment_by_freq = {
            300: "05052025_FREQ300_mergecudaDL_nodvfs_remerge",
            900: "05052025_FREQ900_mergecudaDL_nodvfs_remerge",
            1530: "09152025DL_0311nonDL_FREQ1530_mergecudaDL_nodvfs_remerge",
        }

        def prediction_dir(freq, partition, metric):
            base = FREQ_OUTPUT_DIR / experiment_by_freq[freq] / partition
            if partition == "seen_partition":
                return base / "crossvalid" / metric / "trainratio_" / "rand10"
            return base / metric / "rand10" / "extratrees"

        PREDICT_XPUT_DIR_FREQ300 = prediction_dir(300, "unseen_partition", "throughput")
        PREDICT_POWER_DIR_FREQ300 = prediction_dir(300, "unseen_partition", "power")
        PREDICT_XPUT_DIR_FREQ900 = prediction_dir(900, "unseen_partition", "throughput")
        PREDICT_POWER_DIR_FREQ900 = prediction_dir(900, "unseen_partition", "power")
        PREDICT_XPUT_DIR_FREQ1530 = prediction_dir(1530, "unseen_partition", "throughput")
        PREDICT_POWER_DIR_FREQ1530 = prediction_dir(1530, "unseen_partition", "power")

        #Cross validate output
        CROSS_PREDICT_XPUT_DIR_FREQ300 = prediction_dir(300, "seen_partition", "throughput")
        CROSS_PREDICT_POWER_DIR_FREQ300 = prediction_dir(300, "seen_partition", "power")
        CROSS_PREDICT_XPUT_DIR_FREQ900 = prediction_dir(900, "seen_partition", "throughput")
        CROSS_PREDICT_POWER_DIR_FREQ900 = prediction_dir(900, "seen_partition", "power")
        CROSS_PREDICT_XPUT_DIR_FREQ1530 = prediction_dir(1530, "seen_partition", "throughput")
        CROSS_PREDICT_POWER_DIR_FREQ1530 = prediction_dir(1530, "seen_partition", "power")
        
        ORACLE_PATH = {#powerlimit: oracle 
            60: FREQ_DATASET_DIR / "05052025_mergecudaDL_powercap60_dvfs" / "0505_nonDL_powercap60_dvfs_throughput_total_labels_comb2_labels.csv",
            100: FREQ_DATASET_DIR / "05052025_nonDL_09152025_DL_mergecudaDL_powercap100_dvfs" / "merged_labels.csv",
            200: FREQ_DATASET_DIR / "09102025_mergecudaDL_powercap200_dvfs" / "merged_labels.csv",
            250: FREQ_DATASET_DIR / "10032025_mergecudaDL_powercap250_dvfs" / "merged_labels.csv"
        }
        #ORACLE Latency file for latency comparison  
        ORACLE_LATENCY_PATH = {
            60: [STAGE2_DIR / "v100_socc26" / "02062026_nonDL_gpu011_powercap60_dvfs_share_comb2_freqscale_latency_individual_avg.csv",
            STAGE2_DIR / "05052025_powercap60_DL_dvfs_share_comb2_freqscale_latency_individual_avg.csv"],
            100: None,
            200: [STAGE2_DIR / "09102025_powercap200_nonDL_dvfs_share_comb2_freqscale_latency_individual_avg.csv",
            STAGE2_DIR / "09102025_powercap200_DL_dvfs_share_comb2_freqscale_latency_individual_avg.csv"],
            250: None,
        }
        ORACLE_STEP_PATH = {
            60: [STAGE2_DIR / "v100" / "01152026_rebut_share_comb2_freqscale_steps_count_individual_avg.csv",
            STAGE2_DIR / "v100_socc26" / "02062026_nonDL_gpu011_powercap60_dvfs_share_comb2_freqscale_steps_count_individual_avg.csv"]
        }
        #baselines
        mudi_summary_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../eval_baselines/mudi/curvefit/summary"))
        gslice_summary_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../eval_baselines/gslice/summary"))
        muxflow_summary_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../eval_baselines/muxflow"))
        MUDI_LATENCY_OPTIMAL_ALLOCATION_FILES = {
            "freq300": os.path.join(mudi_summary_dir, "cutoff_freq300_optimal_allocation_latency.csv"),
            "freq900": os.path.join(mudi_summary_dir, "cutoff_freq900_optimal_allocation_latency.csv"),
            "freq1530": os.path.join(mudi_summary_dir, "cutoff_freq1530_optimal_allocation_latency.csv"),
        }
        MUDI_XPUT_OPTIMAL_ALLOCATION_FILES = {
            "freq300": os.path.join(mudi_summary_dir, "cutoff_freq300_optimal_allocation.csv"),
            "freq900": os.path.join(mudi_summary_dir, "cutoff_freq900_optimal_allocation.csv"),
            "freq1530": os.path.join(mudi_summary_dir, "cutoff_freq1530_optimal_allocation.csv"),
        }

        GSLICE_OPTIMAL_ALLOCATION_FILES = {
            "freq300": os.path.join(gslice_summary_dir, "colocated_freq300.csv"),
            "freq900": os.path.join(gslice_summary_dir, "colocated_freq900.csv"),
            "freq1530": os.path.join(gslice_summary_dir, "colocated_freq1530.csv"),
        }
        MUXFLOW_OPTIMAL_ALLOCATION_FILES = {
            60: os.path.join(muxflow_summary_dir, "final-powercap60_result_optimal_percentages.csv"),
            100: os.path.join(muxflow_summary_dir, "final-powercap100_result_optimal_percentages.csv"),
            200: os.path.join(muxflow_summary_dir, "final-powercap200_result_optimal_percentages.csv"),
        }
        muxflow_optimal_allocation_path = MUXFLOW_OPTIMAL_ALLOCATION_FILES.get(set_power_limit)
        if muxflow_optimal_allocation_path is None:
            print(f"Warning: No MUXFLOW allocation file configured for power cap {set_power_limit}")
        elif not os.path.exists(muxflow_optimal_allocation_path):
            print(f"Warning: MUXFLOW allocation file not found at {muxflow_optimal_allocation_path}")
        

    elif n_combinations == 3:
        def comb3_prediction_dir(freq, metric):
            return FREQ_OUTPUT_DIR / f"09152025_freq{freq}_DL_comb3" / "unseen_partition" / metric / "rand10" / "extratrees"

        PREDICT_XPUT_DIR_FREQ300 = comb3_prediction_dir(300, "throughput")
        PREDICT_POWER_DIR_FREQ300 = comb3_prediction_dir(300, "power")
        PREDICT_XPUT_DIR_FREQ900 = comb3_prediction_dir(900, "throughput")
        PREDICT_POWER_DIR_FREQ900 = comb3_prediction_dir(900, "power")
        PREDICT_XPUT_DIR_FREQ1530 = comb3_prediction_dir(1530, "throughput")
        PREDICT_POWER_DIR_FREQ1530 = comb3_prediction_dir(1530, "power")
        ORACLE_PATH = {
            100: FREQ_DATASET_DIR / "09152025_shareDL_comb3_powercap100_dvfs" / "09152025_shareDL_comb3_powercap100_dvfs_throughput_total_labels_comb3_labels_sampled.csv",
            200: FREQ_DATASET_DIR / "09152025_shareDL_comb3_powercap200_dvfs" / "09152025_shareDL_comb3_powercap200_dvfs_throughput_total_labels_comb3_labels.csv",
            60: FREQ_DATASET_DIR / "09152025_shareDL_comb3_powercap60_dvfs_gpu012" / "09152025_shareDL_comb3_powercap60_dvfs_throughput_total_labels_comb3_labels.csv"
        }
        #added mudi baseline
        mudi_comb3_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../eval_baselines/mudi"))
        MUDI_XPUT_OPTIMAL_ALLOCATION_FILES_POWERCAP = {
            60 :  os.path.join(mudi_comb3_dir, "mudi_comb3_final_powercap60_v2.csv"),
            100: os.path.join(mudi_comb3_dir, "mudi_comb3_final_powercap100.csv"),
            200: os.path.join(mudi_comb3_dir, "mudi_comb3_final_powercap200.csv"),
        }
        #print(MUDI_XPUT_OPTIMAL_ALLOCATION_FILES_POWERCAP)
    else:
        raise ValueError(f"prediction and oracle file of {n_combinations} combinations not implemented!")

    try:
        oracle_file = ORACLE_PATH[set_power_limit]
    except KeyError:
        available_caps = ", ".join(str(k) for k in sorted(ORACLE_PATH.keys()))
        print(f"Error: No oracle file configured for power cap {set_power_limit}. Available caps: {available_caps}")
        sys.exit(1)

    MODEL_FREQ_LABEL_1530MHZ = "freq1530"  # Constant for the custom baseline

    try:
        temp_oracle_df_for_workloads = pd.read_csv(oracle_file)
        workload_cols_in_oracle = [
            col for col in temp_oracle_df_for_workloads.columns if col.startswith("Workload")
        ]
        workloads_set = set()
        for col in workload_cols_in_oracle:
            workloads_set.update(
                value for value in temp_oracle_df_for_workloads[col].dropna().unique()
                if pd.notna(value)
            )
        workloads = sorted(workloads_set)
    except Exception as e:
        print(f"Error deriving workloads from oracle file: {e}. Please define the 'workloads' list manually.")
        workloads = []

    if not workloads:
        print("Error: 'workloads' list is empty. Cannot proceed with dictionary creation for prediction files.")
        sys.exit(1)

    sorted_model_freqs = sorted(model_freq_numeric_labels)
    model_freq_string_labels = [f"freq{num}" for num in sorted_model_freqs]



    os.makedirs(output_base_name_dir, exist_ok=True)
    with open(os.path.join(output_base_name_dir, "args.txt"), "w") as f:
        f.write(f"target={target}\n")
        f.write(f"oracle_file={oracle_file}\n")
        f.write(f"is_cross_validation={is_cross_validation}\n")
        f.write(f"cross_models={','.join(cross_models)}\n")
        f.write(f"cross_num_testsets={','.join(str(n) for n in cross_num_testsets)}\n")
        f.write(f"predict_xput_dirs_freq300={PREDICT_XPUT_DIR_FREQ300}\n")
        f.write(f"predict_power_dirs_freq300={PREDICT_POWER_DIR_FREQ300}\n")
        f.write(f"predict_xput_dirs_freq900={PREDICT_XPUT_DIR_FREQ900}\n")
        f.write(f"predict_power_dirs_freq900={PREDICT_POWER_DIR_FREQ900}\n")
        f.write(f"predict_xput_dirs_freq1530={PREDICT_XPUT_DIR_FREQ1530}\n")
        f.write(f"predict_power_dirs_freq1530={PREDICT_POWER_DIR_FREQ1530}\n")
        f.write(f"cross_predict_xput_dirs_freq300={CROSS_PREDICT_XPUT_DIR_FREQ300}\n")
        f.write(f"cross_predict_power_dirs_freq300={CROSS_PREDICT_POWER_DIR_FREQ300}\n")
        f.write(f"cross_predict_xput_dirs_freq900={CROSS_PREDICT_XPUT_DIR_FREQ900}\n")
        f.write(f"cross_predict_power_dirs_freq900={CROSS_PREDICT_POWER_DIR_FREQ900}\n")
        f.write(f"cross_predict_xput_dirs_freq1530={CROSS_PREDICT_XPUT_DIR_FREQ1530}\n")
        f.write(f"cross_predict_power_dirs_freq1530={CROSS_PREDICT_POWER_DIR_FREQ1530}\n")
        f.write(f"pred_xput_file_common_name={PRED_XPUT_FILE_COMMON_NAME}\n")
        f.write(f"pred_power_file_common_name={PRED_POWER_FILE_COMMON_NAME}\n")
        f.write(f"weight={weight}\n")
        f.write(f"static_power_limit={static_power_limit}\n")
        f.write(f"power_epsilon={power_epsilon}\n")
        f.write(f"rm_100partitions={rm_100partitions}\n")
        f.write(f"is_plot={is_plot}\n")
        f.write(f"save_csv={save_csv}\n")
        for freq_label, allocation_path in MUDI_LATENCY_OPTIMAL_ALLOCATION_FILES.items():
            f.write(f"mudi_latency_allocation_{freq_label}={allocation_path}\n")
        for freq_label, allocation_path in MUDI_XPUT_OPTIMAL_ALLOCATION_FILES.items():
            f.write(f"mudi_xput_allocation_{freq_label}={allocation_path}\n")
        for power_cap, allocation_path in MUDI_XPUT_OPTIMAL_ALLOCATION_FILES_POWERCAP.items():
            f.write(f"mudi_xput_allocation_powercap_{power_cap}={allocation_path}\n")
        for freq_label, allocation_path in GSLICE_OPTIMAL_ALLOCATION_FILES.items():
            f.write(f"gslice_allocation_{freq_label}={allocation_path}\n")
        for power_cap, allocation_path in MUXFLOW_OPTIMAL_ALLOCATION_FILES.items():
            f.write(f"muxflow_allocation_{power_cap}={allocation_path}\n")
        f.write(f"muxflow_active_allocation_path={muxflow_optimal_allocation_path}\n")

    if is_cross_validation:
        cross_freq_base_dirs_xput = {
            300: CROSS_PREDICT_XPUT_DIR_FREQ300,
            900: CROSS_PREDICT_XPUT_DIR_FREQ900,
            1530: CROSS_PREDICT_XPUT_DIR_FREQ1530,
        }
        cross_freq_base_dirs_power = {
            300: CROSS_PREDICT_POWER_DIR_FREQ300,
            900: CROSS_PREDICT_POWER_DIR_FREQ900,
            1530: CROSS_PREDICT_POWER_DIR_FREQ1530,
        }

        fold_suffix_fragment = ""
        if cross_num_testsets:
            fold_identifiers = [f"fold{int(num)}" for num in cross_num_testsets]
            fold_suffix_fragment = "_" + "_".join(fold_identifiers)

        xput_fold_names = discover_cross_validation_folds(cross_freq_base_dirs_xput, cross_num_testsets)
        power_fold_names = discover_cross_validation_folds(cross_freq_base_dirs_power, cross_num_testsets)
        fold_names = sorted(set(xput_fold_names).intersection(power_fold_names))

        if not fold_names:
            print("Error: Could not find matching cross-validation fold directories across throughput and power predictions.")
            sys.exit(1)

        aggregated_cross_results = []
        cross_fold_stats = []

        for fold_name in fold_names:
            for model in cross_models:
                try:
                    predict_xput_csv_files_dict, predict_power_csv_files_dict = build_cross_validation_prediction_maps(
                        workloads=workloads,
                        freq_dirs_xput=cross_freq_base_dirs_xput,
                        freq_dirs_power=cross_freq_base_dirs_power,
                        fold_name=fold_name,
                        model_name=model,
                        pred_xput_filename=PRED_XPUT_FILE_COMMON_NAME,
                        pred_power_filename=PRED_POWER_FILE_COMMON_NAME,
                    )
                except FileNotFoundError as exc:
                    print(f"Skipping fold '{fold_name}' model '{model}': {exc}")
                    continue

                fold_output_dir = os.path.join(output_base_name_dir, fold_name, model)
                os.makedirs(fold_output_dir, exist_ok=True)

                print(f"Running cross-validation analysis: fold={fold_name}, model={model}")
                final_df, pairs_without_pred, baseline_violations = analyze_xput_under_powercap_policy(
                    predict_xput_csv_files_dict=predict_xput_csv_files_dict,
                    predict_power_csv_files_dict=predict_power_csv_files_dict,
                    model_freq_string_labels=model_freq_string_labels,
                    model_freq_numeric_labels=sorted_model_freqs,
                    oracle_file_path=oracle_file,
                    static_power_limit=static_power_limit,
                    dvfs_power_limit=set_power_limit,
                    target_metric_name=target,
                    output_base_name=fold_output_dir,
                    weight=weight,
                    rm_100partitions=rm_100partitions,
                    model_label_for_custom_baseline=MODEL_FREQ_LABEL_1530MHZ,
                    mudi_latency_optimal_allocation_paths=MUDI_LATENCY_OPTIMAL_ALLOCATION_FILES,
                    mudi_xput_optimal_allocation_paths=MUDI_XPUT_OPTIMAL_ALLOCATION_FILES,
                    mudi_xput_optimal_allocation_paths_powercap=MUDI_XPUT_OPTIMAL_ALLOCATION_FILES_POWERCAP,
                    gslice_optimal_allocation_paths=GSLICE_OPTIMAL_ALLOCATION_FILES,
                    muxflow_optimal_allocation_path=muxflow_optimal_allocation_path,
                    oracle_latency_files=ORACLE_LATENCY_PATH if n_combinations == 2 else None,
                    oracle_steps_files=ORACLE_STEP_PATH if n_combinations == 2 else None,
                    is_plot=is_plot,
                    save_csv=save_csv,
                    is_cross_validation=is_cross_validation
                )

                cross_fold_stats.append({
                    "fold_name": fold_name,
                    "model_name": model,
                    "pairs_without_pred_config": pairs_without_pred,
                    "custom_baseline_violations": baseline_violations,
                })

                if final_df is None or final_df.empty:
                    continue

                final_df = final_df.copy()
                final_df["fold_name"] = fold_name
                final_df["model_name"] = model
                aggregated_cross_results.append(final_df)

        if aggregated_cross_results:
            combined_cross_df = pd.concat(aggregated_cross_results, ignore_index=True)
            combined_cross_filename = f"summary_{target}_crossvalid{fold_suffix_fragment}_all_folds.csv"
            combined_cross_path = os.path.join(output_base_name_dir, combined_cross_filename)
            combined_cross_df.to_csv(combined_cross_path, index=False)
            cross_mean = combined_cross_df.mean(numeric_only=True)
            cross_mean_filename = f"summary_{target}_crossvalid{fold_suffix_fragment}_mean.csv"
            cross_mean_path = os.path.join(output_base_name_dir, cross_mean_filename)
            cross_mean.to_csv(cross_mean_path, header=["mean_value"])
        else:
            print("No cross-validation results were generated.")

        if cross_fold_stats:
            fold_stats_df = pd.DataFrame(cross_fold_stats)
            fold_stats_filename = f"summary_{target}_crossvalid{fold_suffix_fragment}_fold_stats.csv"
            fold_stats_path = os.path.join(output_base_name_dir, fold_stats_filename)
            fold_stats_df.to_csv(fold_stats_path, index=False)

    else:
        freq_base_dirs_xput = {
            300: PREDICT_XPUT_DIR_FREQ300,
            900: PREDICT_XPUT_DIR_FREQ900,
            1530: PREDICT_XPUT_DIR_FREQ1530,
        }
        freq_base_dirs_power = {
            300: PREDICT_POWER_DIR_FREQ300,
            900: PREDICT_POWER_DIR_FREQ900,
            1530: PREDICT_POWER_DIR_FREQ1530,
        }

        predict_xput_csv_files_dict = defaultdict(dict)
        predict_power_csv_files_dict = defaultdict(dict)

        for freq_num_key in sorted_model_freqs:
            for workload_name_key in workloads:
                xpath = os.path.join(freq_base_dirs_xput[freq_num_key], workload_name_key, "predicts_by_workloads", PRED_XPUT_FILE_COMMON_NAME)
                ppath = os.path.join(freq_base_dirs_power[freq_num_key], workload_name_key, "predicts_by_workloads", PRED_POWER_FILE_COMMON_NAME)
                predict_xput_csv_files_dict[freq_num_key][workload_name_key] = xpath
                predict_power_csv_files_dict[freq_num_key][workload_name_key] = ppath

        if target == "xput_under_powercap":
            if static_power_limit is None:
                print("Error: For 'xput_under_powercap', static_power_limit must be set.")
                sys.exit(1)

            print("Running XPUT Under Powercap Policy with Multiple Frequency Models...")
            print(f"  Oracle: {oracle_file}")
            print(f"  Output Dir: {output_base_name_dir}")

            analyze_xput_under_powercap_policy(
                predict_xput_csv_files_dict=predict_xput_csv_files_dict,
                predict_power_csv_files_dict=predict_power_csv_files_dict,
                model_freq_string_labels=model_freq_string_labels,
                model_freq_numeric_labels=sorted_model_freqs,
                oracle_file_path=oracle_file,
                static_power_limit=static_power_limit,
                dvfs_power_limit=set_power_limit,
                target_metric_name=target,
                output_base_name=output_base_name_dir,
                weight=weight,
                rm_100partitions=rm_100partitions,
                model_label_for_custom_baseline=MODEL_FREQ_LABEL_1530MHZ,
                mudi_latency_optimal_allocation_paths=MUDI_LATENCY_OPTIMAL_ALLOCATION_FILES,
                mudi_xput_optimal_allocation_paths=MUDI_XPUT_OPTIMAL_ALLOCATION_FILES,
                mudi_xput_optimal_allocation_paths_powercap=MUDI_XPUT_OPTIMAL_ALLOCATION_FILES_POWERCAP,
                gslice_optimal_allocation_paths=GSLICE_OPTIMAL_ALLOCATION_FILES,
                muxflow_optimal_allocation_path=muxflow_optimal_allocation_path,
                oracle_latency_files=ORACLE_LATENCY_PATH if n_combinations == 2 else None,
                oracle_steps_files=ORACLE_STEP_PATH if n_combinations == 2 else None,
                is_plot=is_plot,
                save_csv=save_csv,
                is_cross_validation=False,
            )
        else:
            single_predict_xput_dir = PREDICT_XPUT_DIR_FREQ1530
            single_predict_power_dir = PREDICT_POWER_DIR_FREQ1530
            single_oracle_file = FREQ_DATASET_DIR / "05052025_mergecudaDL_powercap60_dvfs" / "0505_nonDL_powercap60_dvfs_throughput_total_labels_comb2_labels.csv"
            print(f"Running original analysis for target: {target}")
            print(f"  Xput Dir: {single_predict_xput_dir}")
            print(f"  Power Dir: {single_predict_power_dir}")
            print(f"  Oracle: {single_oracle_file}")
            print(f"  Output Dir: {output_base_name_dir}")

            analyze_predictions_target_with_baselines(
                predict_xput_dir=single_predict_xput_dir,
                predict_power_dir=single_predict_power_dir,
                dvfs_file=single_oracle_file,
                temporal_file=None,
                powercap_limit_file=None,
                static_power_limit=static_power_limit,
                pred_xput_file_common_name=PRED_XPUT_FILE_COMMON_NAME,
                pred_power_file_common_name=PRED_POWER_FILE_COMMON_NAME,
                target=target,
                output_base_name=output_base_name_dir,
                weight=weight,
                dvfs_oracle=True,
                rm_100partitions=rm_100partitions,
                is_plot=is_plot,
                save_csv=save_csv,
            )
