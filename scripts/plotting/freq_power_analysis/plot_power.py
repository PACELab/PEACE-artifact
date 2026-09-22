import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]


VIOLATION_START_DELAY_SECONDS = 60
# Number of trailing samples to drop to avoid end-of-run edge cases
TRIM_TAIL_COUNT = 5
POWER_LINE_REGEX = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?Power=(?P<power>[\d\.]+) W"
)


def calculate_max_violation_duration(
    df,
    powercap,
    start_offset_seconds=0.0,
) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Return longest violation duration plus start/end timestamps."""

    times = df['time_seconds'].values
    power_values = df['power.draw'].values
    timestamps = df['timestamp'].values

    valid_mask = times >= start_offset_seconds
    if not np.any(valid_mask):
        return 0.0, None, None

    times = times[valid_mask]
    power_values = power_values[valid_mask]
    timestamps = timestamps[valid_mask]

    if len(power_values) == 0:
        return 0.0, None, None

    violation_mask = power_values > powercap

    if not np.any(violation_mask):
        return 0.0, None, None

    interval = float(np.median(np.diff(times))) if len(times) > 1 else 0.0

    max_duration = 0.0
    best_start_idx: Optional[int] = None
    best_end_idx: Optional[int] = None
    current_start_idx: Optional[int] = None

    for idx in range(len(times)):
        if violation_mask[idx]:
            if current_start_idx is None:
                current_start_idx = idx
        elif current_start_idx is not None:
            next_time = times[idx]
            duration = next_time - times[current_start_idx]
            if duration > max_duration:
                max_duration = duration
                best_start_idx = current_start_idx
                best_end_idx = idx - 1
            current_start_idx = None

    if current_start_idx is not None:
        duration = (times[-1] + interval) - times[current_start_idx]
        if duration > max_duration:
            max_duration = duration
            best_start_idx = current_start_idx
            best_end_idx = len(times) - 1

    if best_start_idx is None or best_end_idx is None:
        return 0.0, None, None

    start_timestamp = pd.to_datetime(timestamps[best_start_idx])

    if best_end_idx < len(times) - 1:
        end_timestamp = pd.to_datetime(timestamps[best_end_idx + 1])
    else:
        end_base = pd.to_datetime(timestamps[best_end_idx])
        end_timestamp = end_base + pd.to_timedelta(interval, unit='s')

    return max_duration, start_timestamp, end_timestamp


def compute_violation_result(
    df,
    powercap,
    start_offset_seconds,
) -> Optional[Dict[str, Optional[float]]]:
    if df.empty:
        return None

    duration, start_ts, end_ts = calculate_max_violation_duration(
        df,
        powercap,
        start_offset_seconds=start_offset_seconds,
    )

    return {
        "duration": duration,
        "start": start_ts,
        "end": end_ts,
    }


def read_power_data(path: str) -> pd.DataFrame:
    """Load power readings from a CSV or monitor log."""

    if path.endswith(".log"):
        timestamps = []
        powers = []
        with open(path, "r", encoding="utf-8") as log_file:
            lines = log_file.readlines()

        last_init_idx = -1
        for idx, line in enumerate(lines):
            if "__init__" in line:
                last_init_idx = idx

        for line in lines[last_init_idx + 1 :]:
            match = POWER_LINE_REGEX.search(line)
            if match:
                timestamps.append(match.group("timestamp"))
                powers.append(float(match.group("power")))

        if not timestamps:
            return pd.DataFrame(columns=["timestamp", "time_seconds", "power.draw"])

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(timestamps, format="%Y-%m-%d %H:%M:%S,%f"),
            "power.draw": powers,
        })
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["time_seconds"] = (
            df["timestamp"] - df["timestamp"].iloc[0]
        ).dt.total_seconds()

        # Trim trailing samples to avoid edge effects at shutdown/end
        if len(df) > TRIM_TAIL_COUNT:
            df = df.iloc[:-TRIM_TAIL_COUNT].reset_index(drop=True)

        return df

    df = pd.read_csv(path)
    timestamp_str = df['timestamp'].astype(str).str.strip()
    parsed_ts = pd.to_datetime(
        timestamp_str,
        format="%Y/%m/%d %H:%M:%S.%f",
        errors='coerce',
    )
    if parsed_ts.isna().any():
        parsed_ts = parsed_ts.fillna(
            pd.to_datetime(
                timestamp_str,
                format="%Y/%m/%d %H:%M:%S",
                errors='coerce',
            )
        )
    parsed_ts = parsed_ts.fillna(pd.to_datetime(timestamp_str, errors='coerce'))
    df['timestamp'] = parsed_ts
    df = df.dropna(subset=['timestamp']).reset_index(drop=True)
    df['time_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()

    power_column = None
    for column in df.columns:
        if column.strip() == 'power.draw [W]':
            power_column = column
            break

    if power_column is None:
        raise ValueError(f"Unable to locate power column in {path}")

    df['power.draw'] = (
        df[power_column]
        .astype(str)
        .str.replace(' W', '', regex=False)
        .astype(float)
    )

    # Trim trailing samples to avoid end-of-run spikes in power/freq
    if len(df) > TRIM_TAIL_COUNT:
        df = df.iloc[:-TRIM_TAIL_COUNT].reset_index(drop=True)

    return df


def _find_freq_column(df: pd.DataFrame) -> Optional[str]:
    """Find a usable GPU frequency column in df, tolerant to spacing/case.

    Prefers graphics clock; falls back to SM clock if needed.
    """
    if df is None or df.empty:
        return None
    norm_map = {c: c.strip().lower() for c in df.columns}
    for key in ["clocks.current.graphics", "clocks.current.sm"]:
        for original, normalized in norm_map.items():
            if key in normalized and "[mhz]" in normalized:
                return original
    return None


def _to_numeric_mhz(series: pd.Series) -> pd.Series:
    """Convert a column with values like '1305 MHz' or numeric to float MHz."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    s = series.astype(str).str.replace(" MHz", "", regex=False).str.strip()
    s = s.str.replace(r"[^0-9.]+", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def _paired_path(path: str) -> Optional[str]:
    """Return the counterpart path between gpu_monitor(.log) and gpu_mem(.csv) if plausible."""
    if path.endswith(".log") and "gpu_monitor" in path:
        return path.replace("gpu_monitor", "gpu_mem").replace(".log", ".csv")
    if path.endswith(".csv") and "gpu_mem" in path:
        return path.replace("gpu_mem", "gpu_monitor").replace(".csv", ".log")
    return None


def plot_power(data, title, powercap_value, powerratio, increase_freq_ratio):
    # Read primary and paired power sources, prefer the one with lower max power
    sources = []  # (src_name, path, df)
    try:
        df_primary = read_power_data(data)
        if not df_primary.empty:
            sources.append(("log" if data.endswith(".log") else "csv", data, df_primary))
    except Exception as e:
        print(f"Failed to read primary power data {data}: {e}")

    other_path = _paired_path(data)
    if other_path and os.path.exists(other_path):
        try:
            df_other = read_power_data(other_path)
            if not df_other.empty:
                sources.append(("csv" if other_path.endswith(".csv") else "log", other_path, df_other))
        except Exception as e:
            print(f"Failed to read paired power data {other_path}: {e}")

    if not sources:
        print(f"No power readings found in {data}")
        return None

    selected_src, selected_path, df = min(
        sources, key=lambda item: float(item[2]["power.draw"].max()) if not item[2].empty else float("inf")
    )
    alt_df = None
    for s in sources:
        if s[1] != selected_path:
            alt_df = s[2]
            break

    workload1, workload2 = title.split("vs")
    workload1 = workload1.replace("inference", "inf")
    workload2 = workload2.replace("inference", "inf")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['time_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()

    print(
        f"Using {selected_src} source ({os.path.basename(selected_path)}) | "
        f"Avg Power: {df['power.draw'].mean():.2f} W | Max Power: {df['power.draw'].max():.2f} W"
    )

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

    if os.path.isfile(powercap_value):
        powercap100_df = pd.read_csv(powercap_value)
        powercap100_df = powercap100_df[(powercap100_df["workload1"] == workload1) & (powercap100_df["workload2"] == workload2)]
        if powercap100_df.empty:
            print(f"No powercap100 data for {workload1}, {workload2}")
            return None
        powercap = powerratio * powercap100_df["powercap50_50"].iloc[0]
    else:
        powercap = powercap_value
    increase_freq_cap = powercap * increase_freq_ratio

    violation = compute_violation_result(
        df,
        powercap,
        start_offset_seconds=VIOLATION_START_DELAY_SECONDS,
    )
    if violation is None:
        print(f"No violation data available for {title}")
        return None

    max_violation_time = violation["duration"]
    violation_start_ts = violation["start"]
    violation_end_ts = violation["end"]
    print(
        f"w1: {workload1}, w2: {workload2}, powercap: {powercap}, "
        f"increase_freq_cap: {increase_freq_cap}, max_violation_time: {max_violation_time:.2f}s "
        f"(start_offset {VIOLATION_START_DELAY_SECONDS}s, start={violation_start_ts}, end={violation_end_ts}, source={selected_src})"
    )

    # Power vs Time
    axes[0].plot(
        df['time_seconds'],
        df['power.draw'],
        'b-',
        marker='o',
        markersize=4,
        label=(
            f"Avg Power: {df['power.draw'].mean():.2f} W | "
            f"Max Violation: {max_violation_time:.2f} s"
        ),
    )
    axes[0].set_title(f'{title} - Power vs Time')
    axes[0].set_ylabel('Power (W)')
    axes[0].grid(True)
    # Always draw powercap lines
    axes[0].axhline(y=powercap, color='yellow', label=f'Powercap Limit ({powercap:.2f})', linewidth=3)
    axes[0].axhline(y=increase_freq_cap, color='red', label=f'freq increase threshold ({increase_freq_cap:.2f})', linewidth=3)
    axes[0].legend()

    # Frequency vs Time
    freq_col = _find_freq_column(df)
    freq_df = df
    if freq_col is None and alt_df is not None and not alt_df.empty:
        # Try alternate dataset for frequency
        fcol_alt = _find_freq_column(alt_df)
        if fcol_alt is not None:
            freq_col = fcol_alt
            freq_df = alt_df

    if freq_col is not None:
        t0 = df['timestamp'].iloc[0]
        t_seconds = (pd.to_datetime(freq_df['timestamp']) - t0).dt.total_seconds()
        gpu_clock = _to_numeric_mhz(freq_df[freq_col])
        axes[1].plot(t_seconds, gpu_clock, 'r-', marker='o', markersize=4)
        axes[1].set_title(f'{title} - GPU Clock vs Time')
    else:
        axes[1].set_title(f'{title} - GPU Clock vs Time (no data)')

    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('GPU Clock (MHz)')
    axes[1].grid(True)

    # Layout and save
    plt.tight_layout()
    if not os.path.isfile(powercap_value):
        output_dir = f"power_time/powercap{int(powercap_value)}"
    else:
        output_dir = f"power_time/powercapfile"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"./{output_dir}/{title}.png")
    plt.close()

    return {
        "duration": max_violation_time,
        "start": violation_start_ts,
        "end": violation_end_ts,
        "output_dir": output_dir,
        "source": selected_src,
    }

def plot_power_probability(data, title, powercap_value, powerratio, increase_freq_ratio):
    # Read power data
    df = read_power_data(data)
    if df.empty:
        print(f"No power readings found in {data}")
        return
    workload1, workload2 = title.split("vs")
    workload1 = workload1.replace("inference", "inf")
    workload2 = workload2.replace("inference", "inf")

    # Determine powercap value
    if os.path.isfile(powercap_value):
        powercap100_df = pd.read_csv(powercap_value)
        powercap100_df = powercap100_df[(powercap100_df["workload1"] == workload1) & (powercap100_df["workload2"] == workload2)]
        if powercap100_df.empty:
            print(f"No powercap100 data for {workload1}, {workload2}")
            return
        powercap = powerratio * powercap100_df["powercap50_50"].iloc[0]
    else:
        powercap = powercap_value

    # Create probability distribution of power consumption
    power_values = df['power.draw'].values

    # Calculate histogram for probability density
    bins = np.linspace(power_values.min(), power_values.max(), 50)
    hist, bin_edges = np.histogram(power_values, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate percentage of time exceeding powercap
    exceed_powercap_count = np.sum(power_values > powercap)
    total_count = len(power_values)
    exceed_percentage = (exceed_powercap_count / total_count) * 100

    # Calculate percentiles
    p95 = np.percentile(power_values, 95)
    p99 = np.percentile(power_values, 99)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot probability density
    ax.bar(bin_centers, hist, width=np.diff(bin_edges)[0], alpha=0.7, color='skyblue',
           label=f'Power Distribution (Avg: {power_values.mean():.1f}W)')

    # Add vertical line for powercap
    ax.axvline(x=powercap, color='red', linestyle='--', linewidth=2,
               label=f'Powercap: {powercap:.1f}W')

    # Add vertical lines for percentiles
    ax.axvline(x=p95, color='orange', linestyle=':', linewidth=2,
               label=f'P95: {p95:.1f}W')
    ax.axvline(x=p99, color='purple', linestyle=':', linewidth=2,
               label=f'P99: {p99:.1f}W')

    # Add text annotation for statistics
    max_prob = hist.max()
    stats_text = f'{exceed_percentage:.1f}% exceeds powercap\nP95: {p95:.1f}W\nP99: {p99:.1f}W'
    ax.text(powercap + (power_values.max() - powercap) * 0.1, max_prob * 0.8,
            stats_text,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            fontsize=11, ha='left')

    # Formatting
    ax.set_xlabel('Power (W)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'{title} - Power Probability Distribution')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Save the plot
    if not os.path.isfile(powercap_value):
        output_dir = f"power_probability/powercap{int(powercap_value)}"
    else:
        output_dir = f"power_probability/powercapfile"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(f"./{output_dir}/{title}_probability.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Power probability plot saved for {title} - {exceed_percentage:.1f}% exceeds powercap, P95: {p95:.1f}W, P99: {p99:.1f}W")

if __name__ == "__main__":
    #200
    #log_dir = f"{REPO_ROOT}/tests/mps/ccv100_logs/sharenonDL_powercap200_dvfs/RUN1"
    #100
    #log_dir = f"{REPO_ROOT}/tests/mps/ccv100_logs/sharenonDL_powercap100_dvfs/RUN1"
    #60
    #log_dir = f"{REPO_ROOT}/tests/mps/ccv100_logs/sharenonDL_powercap60_dvfs/RUN1"
    log_dir =f"{REPO_ROOT}/tests/mps/ccv100_logs/sharenonDL_powercap200_dvfs/RUN1/fastWalshTransform_batch2-cuda_samples/whisper-large-v2_batch2-inference" 
    #powercap100_file = "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/data/experiment_inputs/colocations/comb2/02062025_powercap100_DL_baseline_labels_comb2_batches2.csv"
    powercap = 220
    w1_percentage = 10
    w2_percentage = 90
    violation_records = defaultdict(list)

    for root, dirs, files in os.walk(log_dir):
        # Check if the current directory has the target monitor log
        if f'gpu_monitor_FREQ0_{w1_percentage}_{w2_percentage}_SLEEP0.log' in files:
            # Extract workload pair (w1 and w2) from log files
            w1_file = [f for f in files if f.startswith('w1_') and f.endswith('.log')]
            w2_file = [f for f in files if f.startswith('w2_') and f.endswith('.log')]
            
            if w1_file and w2_file:
                w1 = w1_file[0].replace('w1_', '').replace(f'_FREQ0_MPS{w1_percentage}_SLEEP0.log', '')
                w2 = w2_file[0].replace('w2_', '').replace(f'_FREQ0_MPS{w2_percentage}_SLEEP0.log', '')
                title = f"{w1}vs{w2}"
                
                # Path to the power monitor log
                log_path = os.path.join(root, f'gpu_monitor_FREQ0_{w1_percentage}_{w2_percentage}_SLEEP0.log')

                # Plot and save the power data
                print(f"Plotting power for {title}")
                result = plot_power(log_path, title, powercap, powerratio=0.75, increase_freq_ratio=0.9)
                if result is not None:
                    output_dir = result["output_dir"]
                    violation_records[output_dir].append(
                        {
                            "workload1": w1,
                            "workload2": w2,
                            "title": title,
                            "max_consecutive_violation_seconds": result["duration"],
                            "violation_start_timestamp": result["start"],
                            "violation_end_timestamp": result["end"],
                        }
                    )
                plot_power_probability(log_path, title, powercap, powerratio=0.75, increase_freq_ratio=0.9)

    for output_dir, records in violation_records.items():
        if records:
            summary_path = os.path.join(output_dir, "max_violation_summary.csv")
            pd.DataFrame(records).to_csv(summary_path, index=False)
            print(f"Saved violation summary to {summary_path}")

    print("Plotting complete. Check the saved PNG files in the working directory.")
