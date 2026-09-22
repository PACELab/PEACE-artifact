import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from plot_power import (
    VIOLATION_START_DELAY_SECONDS,
    calculate_max_violation_duration,
    compute_violation_result,
    read_power_data,
)


SHARE_RATIOS: List[str] = [
    "10_90",
    "20_80",
    "30_70",
    "40_60",
    "50_50",
    "60_40",
    "70_30",
    "80_20",
    "90_10",
]

RUN_PATTERN = re.compile(r"^RUN(\d+)$")
RUNNING_AVG_WINDOW_SECONDS = 5.0


def _sorted_run_paths(log_dir: str) -> List[str]:
    entries = [
        entry
        for entry in os.listdir(log_dir)
        if os.path.isdir(os.path.join(log_dir, entry))
    ]

    run_entries = []
    for entry in entries:
        match = RUN_PATTERN.match(entry)
        if match:
            run_entries.append((int(match.group(1)), entry))

    if run_entries:
        return [
            os.path.join(log_dir, entry)
            for _, entry in sorted(run_entries)
        ]

    return [log_dir]


def _max_power_after_start(
    df: pd.DataFrame,
    start_offset_seconds: float,
) -> Optional[float]:
    if df.empty:
        return None

    valid_mask = df["time_seconds"] >= start_offset_seconds
    if not valid_mask.any():
        return None

    return float(df.loc[valid_mask, "power.draw"].max())


def _running_average_violation_stats(
    df: Optional[pd.DataFrame],
    powercap_value: float,
    start_offset_seconds: float,
    window_seconds: float = RUNNING_AVG_WINDOW_SECONDS,
    debug: bool = False,
) -> Tuple[
    bool,
    Optional[pd.Timestamp],
    Optional[pd.Timestamp],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    """Return running-average violation stats.

    Returns a tuple with:
    - violation_flag: whether any window mean exceeded the cap
    - violation_start_ts / violation_end_ts: timestamps for the first and last sample of
      the first violating window encountered
    - max_mean: maximum window mean power across all windows
    - violation_percentage: percent of windows whose mean exceeded the cap
    - violation_windows_mean: mean of window means for only the violating windows
    """

    if df is None or df.empty:
        return False, None, None, None, None, None

    mask = df["time_seconds"] >= start_offset_seconds
    if not mask.any():
        return False, None, None, None, None, None

    filtered = (
        df.loc[mask, ["timestamp", "time_seconds", "power.draw"]]
        .dropna(subset=["timestamp", "time_seconds", "power.draw"])
        .sort_values("timestamp")
    )
    if filtered.empty:
        return False, None, None, None, None, None

    filtered = filtered.set_index("timestamp")
    if filtered.empty:
        return False, None, None, None, None, None

    filtered.reset_index(inplace=True)

    time_seconds = filtered["time_seconds"].to_numpy()
    if time_seconds.size == 0:
        return False, None, None, None, None, None

    window_seconds = float(window_seconds)
    last_time_seconds = float(time_seconds[-1])
    window_start_seconds = float(start_offset_seconds)

    violation_flag = False
    violation_start_ts: Optional[pd.Timestamp] = None
    violation_end_ts: Optional[pd.Timestamp] = None
    max_mean: Optional[float] = None
    total_windows = 0
    violation_windows = 0
    violation_windows_sum = 0.0

    while window_start_seconds <= last_time_seconds:
        window_end_seconds = window_start_seconds + window_seconds
        window_mask = (time_seconds >= window_start_seconds) & (
            time_seconds < window_end_seconds
        )
        window_df = filtered.loc[window_mask]

        if window_df.empty:
            window_start_seconds = window_end_seconds
            continue

        total_windows += 1
        window_mean = float(window_df["power.draw"].mean())
        if max_mean is None or window_mean > max_mean:
            max_mean = window_mean

        if window_mean > powercap_value:
            violation_windows += 1
            violation_windows_sum += window_mean
            if not violation_flag:
                violation_flag = True
                violation_start_ts = window_df["timestamp"].iloc[0]
            violation_end_ts = window_df["timestamp"].iloc[-1]

        if debug:
            timestamps = window_df["timestamp"].astype(str).tolist()
            print(
                " | ".join(
                    [
                        "[debug-running-avg]",
                        f"window_start_seconds={window_start_seconds:.2f}",
                        f"window_end_seconds={window_end_seconds:.2f}",
                        f"mean_power={window_mean:.2f}",
                        f"over_cap={'yes' if window_mean > powercap_value else 'no'}",
                        f"samples={len(timestamps)}",
                        f"timestamps=[{', '.join(timestamps)}]",
                    ]
                )
            )

        window_start_seconds = window_end_seconds

    violation_percentage = (
        (violation_windows / total_windows) * 100.0
        if total_windows > 0
        else None
    )
    violation_windows_mean: Optional[float] = (
        float(violation_windows_sum / violation_windows)
        if violation_windows > 0
        else None
    )
    if debug:
        print(
            " | ".join(
                [
                    "[debug-running-avg-summary]",
                    f"total_windows={total_windows}",
                    f"violation_windows={violation_windows}",
                    f"violation_percentage={violation_percentage:.2f}%" if violation_percentage is not None else "violation_percentage=NA",
                    f"max_mean={max_mean:.2f}" if max_mean is not None else "max_mean=NA",
                    f"violation_windows_mean={violation_windows_mean:.2f}" if violation_windows_mean is not None else "violation_windows_mean=NA",
                ]
            )
        )

    return (
        violation_flag,
        violation_start_ts,
        violation_end_ts,
        max_mean,
        violation_percentage,
        violation_windows_mean,
    )


def collect_violation_times(
    log_dirs: List[str],
    powercap_value: float,
    start_delay: float,
    include_timestamps: bool,
    window_seconds: float = RUNNING_AVG_WINDOW_SECONDS,
    debug_windows: bool = False,
) -> List[Dict[str, object]]:
    summary_map: Dict[Tuple[str, str], Dict[str, object]] = {}

    for log_dir in log_dirs:
        for run_path in _sorted_run_paths(log_dir):
            for workload1 in sorted(os.listdir(run_path)):
                workload1_path = os.path.join(run_path, workload1)
                if not os.path.isdir(workload1_path):
                    continue

                for workload2 in sorted(os.listdir(workload1_path)):
                    workload2_path = os.path.join(workload1_path, workload2)
                    if not os.path.isdir(workload2_path):
                        continue

                    key = (workload1, workload2)
                    if key not in summary_map:
                        base_row: Dict[str, object] = {
                            "workload1": workload1,
                            "workload2": workload2,
                        }
                        for ratio in SHARE_RATIOS:
                            base_row[f"violation_{ratio}"] = np.nan
                            base_row[f"max_violation_watt_{ratio}"] = np.nan
                            # Power stats to also record in summary CSV
                            base_row[f"power_max_{ratio}"] = np.nan
                            base_row[f"power_mean_{ratio}"] = np.nan
                            base_row[f"power_p95_{ratio}"] = np.nan
                            base_row[f"power_p99_{ratio}"] = np.nan
                            base_row[f"running_avg_violation_{ratio}"] = np.nan
                            base_row[f"running_avg_violation_pct_{ratio}"] = np.nan
                            base_row[f"running_avg_max_mean_{ratio}"] = np.nan
                            base_row[f"running_avg_violation_overcap_mean_{ratio}"] = np.nan
                            if include_timestamps:
                                base_row[f"violation_{ratio}_start"] = pd.NaT
                                base_row[f"violation_{ratio}_end"] = pd.NaT
                                base_row[f"running_avg_violation_{ratio}_start"] = pd.NaT
                                base_row[f"running_avg_violation_{ratio}_end"] = pd.NaT
                        summary_map[key] = base_row

                    row = summary_map[key]

                    for ratio in SHARE_RATIOS:
                        best_result = None
                        max_watt_candidates: List[float] = []
                        # Keep refs to per-source dfs and their max power for selecting the lower-max-power source
                        df_log: Optional[pd.DataFrame] = None
                        df_csv: Optional[pd.DataFrame] = None
                        max_log: Optional[float] = None
                        max_csv: Optional[float] = None
                        has_power_data = False
                        running_stats: List[
                            Tuple[
                                str,
                                bool,
                                Optional[pd.Timestamp],
                                Optional[pd.Timestamp],
                                Optional[float],
                                Optional[float],
                                Optional[float],
                            ]
                        ] = []

                        log_name = f"gpu_monitor_FREQ0_{ratio}_SLEEP0.log"
                        log_path = os.path.join(workload2_path, ratio, log_name)
                        if os.path.exists(log_path):
                            df_log = read_power_data(log_path)
                            has_power_data = True
                            result = compute_violation_result(
                                df_log,
                                powercap_value,
                                start_offset_seconds=start_delay,
                            )
                            if result is not None:
                                best_result = result

                            max_log = _max_power_after_start(
                                df_log,
                                start_delay,
                            )
                            if max_log is not None:
                                max_watt_candidates.append(max_log)
                            (
                                violation,
                                start_ts,
                                end_ts,
                                max_mean,
                                violation_pct,
                                violation_windows_mean,
                            ) = _running_average_violation_stats(
                                df_log,
                                powercap_value,
                                start_delay,
                                window_seconds=window_seconds,
                                debug=debug_windows,
                            )
                            running_stats.append((
                                "log",
                                violation,
                                start_ts,
                                end_ts,
                                max_mean,
                                violation_pct,
                                violation_windows_mean,
                            ))

                        csv_name = f"gpu_mem_FREQ0_{ratio}_SLEEP0.csv"
                        csv_path = os.path.join(workload2_path, ratio, csv_name)
                        if os.path.exists(csv_path):
                            df_csv = read_power_data(csv_path)
                            has_power_data = True
                            result = compute_violation_result(
                                df_csv,
                                powercap_value,
                                start_offset_seconds=start_delay,
                            )
                            if result is not None:
                                if (
                                    best_result is None
                                    or result["duration"] < best_result["duration"]
                                ):
                                    best_result = result

                            max_csv = _max_power_after_start(
                                df_csv,
                                start_delay,
                            )
                            if max_csv is not None:
                                max_watt_candidates.append(max_csv)
                            (
                                violation,
                                start_ts,
                                end_ts,
                                max_mean,
                                violation_pct,
                                violation_windows_mean,
                            ) = _running_average_violation_stats(
                                df_csv,
                                powercap_value,
                                start_delay,
                                window_seconds=window_seconds,
                                debug=debug_windows,
                            )
                            running_stats.append((
                                "csv",
                                violation,
                                start_ts,
                                end_ts,
                                max_mean,
                                violation_pct,
                                violation_windows_mean,
                            ))

                        max_watt_value: Optional[float] = None
                        if max_watt_candidates:
                            max_watt_value = min(max_watt_candidates)
                            row[f"max_violation_watt_{ratio}"] = max_watt_value
                            print(
                                " | ".join(
                                    [
                                        f"workload1={workload1}",
                                        f"workload2={workload2}",
                                        f"ratio={ratio}",
                                        f"max_watt={max_watt_value:.2f}",
                                    ]
                                )
                            )

                        # Choose dataset with the LOWER max power for power stats (consistent with plotting)
                        selected_df: Optional[pd.DataFrame] = None
                        if max_log is not None and (max_csv is None or max_log <= max_csv):
                            selected_df = df_log
                        elif max_csv is not None:
                            selected_df = df_csv

                        running_violation_flag = False
                        running_violation_start: Optional[pd.Timestamp] = None
                        running_violation_end: Optional[pd.Timestamp] = None
                        max_running_mean: Optional[float] = None
                        running_violation_percentage: Optional[float] = None
                        running_violation_overcap_mean: Optional[float] = None

                        if running_stats:
                            violation_flags = [violation_flag for (_, violation_flag, *_) in running_stats]
                            # Extract the violation pct (2nd from last element) and mean-overcap (last element)
                            violation_pcts = [
                                tpl[-2]
                                for tpl in running_stats
                                if tpl[-2] is not None
                            ]
                            violation_overcap_means = [
                                tpl[-1]
                                for tpl in running_stats
                                if tpl[-1] is not None
                            ]

                            # Require all observed sources to violate before flagging the ratio
                            running_violation_flag = bool(violation_flags) and all(violation_flags)

                            if violation_pcts:
                                running_violation_percentage = min(violation_pcts)

                            for (
                                _source_label,
                                violation_flag,
                                start_ts,
                                end_ts,
                                max_mean,
                                _violation_pct,
                                _violation_windows_mean,
                            ) in running_stats:
                                if max_mean is not None and (
                                    max_running_mean is None or max_mean > max_running_mean
                                ):
                                    max_running_mean = max_mean
                                    if violation_flag:
                                        running_violation_start = start_ts
                                        running_violation_end = end_ts

                            row[f"running_avg_violation_{ratio}"] = running_violation_flag
                            if max_running_mean is not None:
                                row[f"running_avg_max_mean_{ratio}"] = max_running_mean
                            if running_violation_percentage is not None:
                                row[f"running_avg_violation_pct_{ratio}"] = running_violation_percentage
                            if violation_overcap_means:
                                # Per request: record the minimum mean across sources for this combination
                                running_violation_overcap_mean = min(violation_overcap_means)
                                row[f"running_avg_violation_overcap_mean_{ratio}"] = running_violation_overcap_mean

                            if include_timestamps:
                                row[f"running_avg_violation_{ratio}_start"] = (
                                    running_violation_start
                                    if running_violation_start is not None and running_violation_flag
                                    else pd.NaT
                                )
                                row[f"running_avg_violation_{ratio}_end"] = (
                                    running_violation_end
                                    if running_violation_end is not None and running_violation_flag
                                    else pd.NaT
                                )

                        if running_violation_flag and max_running_mean is not None:
                            print(
                                " | ".join(
                                    [
                                        "running-avg violation detected",
                                        f"workload1={workload1}",
                                        f"workload2={workload2}",
                                        f"ratio={ratio}",
                                        f"mean_power={max_running_mean:.2f}",
                                        (
                                            f"window_start={running_violation_start}"
                                            if running_violation_start is not None
                                            else "window_start=NA"
                                        ),
                                        (
                                            f"window_end={running_violation_end}"
                                            if running_violation_end is not None
                                            else "window_end=NA"
                                        ),
                                        (
                                            f"violation_windows_pct={running_violation_percentage:.2f}%"
                                            if running_violation_percentage is not None
                                            else "violation_windows_pct=NA"
                                        ),
                                        (
                                            f"mean_overcap={running_violation_overcap_mean:.2f}"
                                            if running_violation_overcap_mean is not None
                                            else "mean_overcap=NA"
                                        ),
                                    ]
                                )
                            )

                        if selected_df is not None and not selected_df.empty:
                            mask = selected_df["time_seconds"] >= start_delay
                            powers = selected_df.loc[mask, "power.draw"].to_numpy(dtype=float)
                            powers = powers[~np.isnan(powers)]
                            if powers.size:
                                row[f"power_max_{ratio}"] = float(np.max(powers))
                                row[f"power_mean_{ratio}"] = float(np.mean(powers))
                                row[f"power_p95_{ratio}"] = float(np.percentile(powers, 95))
                                row[f"power_p99_{ratio}"] = float(np.percentile(powers, 99))

                        if best_result is None or not has_power_data:
                            continue

                        row[f"violation_{ratio}"] = best_result["duration"]
                        if include_timestamps:
                            row[f"violation_{ratio}_start"] = (
                                best_result["start"]
                                if best_result["start"] is not None
                                else pd.NaT
                            )
                            row[f"violation_{ratio}_end"] = (
                                best_result["end"]
                                if best_result["end"] is not None
                                else pd.NaT
                            )

    return list(summary_map.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute longest powercap violations for all share ratios.",
    )
    parser.add_argument(
        "log_dirs",
        nargs="+",
        help="One or more directories containing run subfolders.",
    )
    parser.add_argument(
        "--powercap",
        required=True,
        help="Fixed powercap value (float) applied to all workload pairs.",
    )
    parser.add_argument(
        "--start_delay",
        type=float,
        default=VIOLATION_START_DELAY_SECONDS,
        help="Seconds to skip before evaluating violations.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the summary CSV. Defaults to <log_dir>/violation_summary.csv.",
    )
    parser.add_argument(
        "--ts",
        action="store_true",
        help="Include start/end timestamps for each violation interval.",
    )
    parser.add_argument(
        "--debug_windows",
        action="store_true",
        help="Print window timestamps and mean power for running-average calculations.",
    )
    parser.add_argument(
        "--window_seconds",
        type=float,
        default=RUNNING_AVG_WINDOW_SECONDS,
        help=f"Window duration in seconds for running-average violation checks (default {RUNNING_AVG_WINDOW_SECONDS}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.log_dirs:
        raise SystemExit("At least one log directory must be provided.")

    try:
        powercap_value = float(args.powercap)
    except ValueError as exc:
        raise SystemExit(f"--powercap must be numeric: {args.powercap}") from exc

    summary_rows = collect_violation_times(
        log_dirs=args.log_dirs,
        powercap_value=powercap_value,
        start_delay=args.start_delay,
        include_timestamps=args.ts,
        window_seconds=args.window_seconds,
        debug_windows=args.debug_windows,
    )

    if not summary_rows:
        print("No violation data collected. Check the log directories and powercap value.")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_records: List[Dict[str, object]] = []

    duration_columns = [
        column
        for column in summary_df.columns
        if column.startswith("violation_")
        and not column.endswith("_start")
        and not column.endswith("_end")
    ]

    if duration_columns:
        flat_values = summary_df[duration_columns].to_numpy(dtype=float).ravel()
        valid_values = flat_values[~np.isnan(flat_values)]

        if valid_values.size:
            mean_violation = float(np.mean(valid_values))
            max_violation = float(np.max(valid_values))
            p95_violation = float(np.percentile(valid_values, 95))
            p99_violation = float(np.percentile(valid_values, 99))
            print(
                "Violation stats (seconds) -> "
                f"mean: {mean_violation:.2f}, "
                f"max: {max_violation:.2f}, "
                f"p95: {p95_violation:.2f}, "
                f"p99: {p99_violation:.2f}"
            )
            summary_records.extend([
                {"metric": "violation_mean_seconds", "value": mean_violation},
                {"metric": "violation_max_seconds", "value": max_violation},
                {"metric": "violation_p95_seconds", "value": p95_violation},
                {"metric": "violation_p99_seconds", "value": p99_violation},
            ])

    # Aggregate power stats across all ratios/workloads for each power metric
    for metric in ("power_max", "power_mean", "power_p95", "power_p99"):
        power_columns = [
            column
            for column in summary_df.columns
            if column.startswith(f"{metric}_")
        ]

        if not power_columns:
            continue

        flat_power_values = summary_df[power_columns].to_numpy(dtype=float).ravel()
        valid_power_values = flat_power_values[~np.isnan(flat_power_values)]

        if not valid_power_values.size:
            continue

        mean_power = float(np.mean(valid_power_values))
        max_power = float(np.max(valid_power_values))
        p95_power = float(np.percentile(valid_power_values, 95))
        p99_power = float(np.percentile(valid_power_values, 99))

        print(
            f"{metric} stats (W) -> "
            f"mean: {mean_power:.2f}, "
            f"max: {max_power:.2f}, "
            f"p95: {p95_power:.2f}, "
            f"p99: {p99_power:.2f}"
        )
        summary_records.extend([
            {"metric": f"{metric}_mean_watts", "value": mean_power},
            {"metric": f"{metric}_max_watts", "value": max_power},
            {"metric": f"{metric}_p95_watts", "value": p95_power},
            {"metric": f"{metric}_p99_watts", "value": p99_power},
        ])

    running_violation_columns = [
        column
        for column in summary_df.columns
        if column.startswith("running_avg_violation_")
        and "_pct_" not in column
        and not column.endswith("_start")
        and not column.endswith("_end")
        and not column.startswith("running_avg_violation_overcap_mean_")
    ]
    violation_pct_columns = [
        column
        for column in summary_df.columns
        if column.startswith("running_avg_violation_pct_")
    ]
    total_pct_combinations = 0
    if violation_pct_columns:
        flat_pct_values = summary_df[violation_pct_columns].to_numpy(dtype=float).ravel()
        valid_pct_values = flat_pct_values[~np.isnan(flat_pct_values)]
        total_pct_combinations = valid_pct_values.size
        if total_pct_combinations:
            avg_pct_value = float(np.mean(valid_pct_values))
            print(
                "Window violation percentage -> "
                f"mean: {avg_pct_value:5f}% across {total_pct_combinations} workload/ratio combinations"
            )
            summary_records.extend([
                {"metric": "window_violation_pct_mean", "value": avg_pct_value},
                {"metric": "window_violation_pct_combinations", "value": total_pct_combinations},
            ])

    # Aggregate across all combinations the recorded overcap mean (minimum across sources per combo)
    overcap_mean_columns = [
        column
        for column in summary_df.columns
        if column.startswith("running_avg_violation_overcap_mean_")
    ]
    total_overcap_mean_combos = 0
    if overcap_mean_columns:
        flat_overcap_values = summary_df[overcap_mean_columns].to_numpy(dtype=float).ravel()
        valid_overcap_values = flat_overcap_values[~np.isnan(flat_overcap_values)]
        total_overcap_mean_combos = valid_overcap_values.size
        if total_overcap_mean_combos:
            avg_overcap_mean = float(np.mean(valid_overcap_values))
            print(
                "Running-avg overcap mean (W) -> "
                f"mean: {avg_overcap_mean:.2f} across {total_overcap_mean_combos} workload/ratio combinations"
            )
            summary_records.extend([
                {"metric": "running_avg_overcap_mean_mean_watts", "value": avg_overcap_mean},
                {"metric": "running_avg_overcap_mean_combinations", "value": total_overcap_mean_combos},
            ])

    if running_violation_columns:
        running_violation_df = summary_df[running_violation_columns]
        valid_mask = running_violation_df.notna()
        total_windows = int(valid_mask.sum().sum())
        violation_count = int((running_violation_df == True).sum().sum())
        if total_windows:
            violation_rate = (violation_count / total_windows) * 100.0
            print(
                "Workload/ratio combinations with >=1 violating window -> "
                f"{violation_count}/{total_windows} ({violation_rate:.1f}%)"
            )
            summary_records.extend([
                {"metric": "running_violation_combinations", "value": violation_count},
                {"metric": "running_violation_total_combinations", "value": total_windows},
                {"metric": "running_violation_rate_pct", "value": violation_rate},
            ])

    running_avg_mean_columns = [
        column
        for column in summary_df.columns
        if column.startswith("running_avg_max_mean_")
    ]
    if running_avg_mean_columns:
        running_avg_values = summary_df[running_avg_mean_columns].to_numpy(dtype=float).ravel()
        valid_running_avg_values = running_avg_values[~np.isnan(running_avg_values)]
        if valid_running_avg_values.size:
            mean_running_avg = float(np.mean(valid_running_avg_values))
            max_running_avg = float(np.max(valid_running_avg_values))
            p95_running_avg = float(np.percentile(valid_running_avg_values, 95))
            p99_running_avg = float(np.percentile(valid_running_avg_values, 99))
            print(
                "Running-average max-mean stats (W) -> "
                f"mean: {mean_running_avg:.2f}, "
                f"max: {max_running_avg:.2f}, "
                f"p95: {p95_running_avg:.2f}, "
                f"p99: {p99_running_avg:.2f}"
            )
            summary_records.extend([
                {"metric": "running_avg_max_mean_mean_watts", "value": mean_running_avg},
                {"metric": "running_avg_max_mean_max_watts", "value": max_running_avg},
                {"metric": "running_avg_max_mean_p95_watts", "value": p95_running_avg},
                {"metric": "running_avg_max_mean_p99_watts", "value": p99_running_avg},
            ])
    if args.output:
        output_file = args.output.replace(".csv", "") + f"_vio{args.powercap}_windowsize{args.window_seconds}.csv"
    output_path = (
        output_file
        if args.output
        else os.path.join(args.log_dirs[0], "violation_summary.csv")
    )
    #create output dir if not exist 
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    print(f"Saved violation summary to {output_path}")

    if summary_records:
        summary_metrics_df = pd.DataFrame(summary_records)
        output_dir = os.path.dirname(output_path) or "."
        metrics_path = os.path.join(output_dir, "violation_metrics_summary.csv")
        summary_metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved aggregate metrics to {metrics_path}")


if __name__ == "__main__":
    main()
