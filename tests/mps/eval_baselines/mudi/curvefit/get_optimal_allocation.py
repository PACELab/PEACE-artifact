#!/usr/bin/env python3
"""
Throughput Sum Analysis for Colocated Workloads

This script analyzes the throughput sum of colocated workloads using piecewise linear
curve fitting parameters from Mudi's latency profiler. It calculates the optimal
GPU resource allocation that maximizes combined throughput.

Based on the methodology from:
"Multiplexing Dynamic Deep Learning Workloads with SLO-awareness in GPU Clusters"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

def piecewise_throughput(gpu_percent, cutoff, value0, slope_left, slope_right):
    """
    Calculate throughput using piecewise linear function.

    Args:
        gpu_percent: GPU percentage allocation (0-100)
        cutoff: Breakpoint in piecewise linear function
        value0: Throughput value at cutoff point
        slope_left: Slope before cutoff point
        slope_right: Slope after cutoff point

    Returns:
        Calculated throughput value
    """
    if gpu_percent <= cutoff:
        return slope_left * (gpu_percent - cutoff) + value0
    else:
        return slope_right * (gpu_percent - cutoff) + value0

def parse_csv_data(csv_path):
    """
    Parse the colocated workload CSV data.

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of dictionaries containing workload data
    """
    df = pd.read_csv(csv_path)
    workload_pairs = []

    workload_indices = []
    for column in df.columns:
        if column.startswith('workload'):
            suffix = column.replace('workload', '')
            if suffix.isdigit():
                workload_indices.append(int(suffix))
    max_workloads = max(workload_indices, default=0)

    for _, row in df.iterrows():
        pair_data = {}
        workloads = []

        for idx in range(1, max_workloads + 1):
            workload_key = f'workload{idx}'
            if workload_key not in df.columns:
                continue

            workload_name = row[workload_key]
            if pd.isna(workload_name):
                continue

            pair_data[workload_key] = workload_name

            workload_meta = {'name': workload_name}
            for field in ('freq', 'cutoff', 'value0', 'slope_left', 'slope_right'):
                column_key = f'w{idx}_{field}'
                if column_key in df.columns:
                    value = row[column_key]
                    pair_data[column_key] = value
                    workload_meta[field] = value
                else:
                    workload_meta[field] = None

            workloads.append(workload_meta)

        if not workloads:
            continue

        pair_data['workloads'] = workloads
        workload_pairs.append(pair_data)

    return workload_pairs

def calculate_throughput_sum(pair_data, gpu_percentages):
    """
    Calculate throughput sum for a workload pair across different GPU allocations.

    Args:
        pair_data: Dictionary containing workload pair curve fitting parameters
        gpu_percentages: Array of GPU percentages for w1 (0-100)

    Returns:
        Tuple of (w1_throughputs, w2_throughputs, throughput_sums)
    """
    w1_throughputs = []
    w2_throughputs = []
    throughput_sums = []

    for w1_gpu in gpu_percentages:
        w2_gpu = 100 - w1_gpu

        # Calculate w1 throughput at w1_gpu%
        w1_throughput = piecewise_throughput(
            w1_gpu,
            pair_data['w1_cutoff'],
            pair_data['w1_value0'],
            pair_data['w1_slope_left'],
            pair_data['w1_slope_right']
        )

        # Calculate w2 throughput at w2_gpu%
        w2_throughput = piecewise_throughput(
            w2_gpu,
            pair_data['w2_cutoff'],
            pair_data['w2_value0'],
            pair_data['w2_slope_left'],
            pair_data['w2_slope_right']
        )

        w1_throughputs.append(w1_throughput)
        w2_throughputs.append(w2_throughput)
        throughput_sums.append(w1_throughput + w2_throughput)

    return w1_throughputs, w2_throughputs, throughput_sums

def find_optimal_allocation(gpu_percentages, throughput_sums):
    """
    Find the GPU allocation that maximizes throughput sum.

    Args:
        gpu_percentages: Array of GPU percentages
        throughput_sums: Array of corresponding throughput sums

    Returns:
        Tuple of (optimal_w1_percentage, optimal_w2_percentage, max_throughput_sum)
    """
    max_idx = np.argmax(throughput_sums)
    optimal_w1 = gpu_percentages[max_idx]
    optimal_w2 = 100 - optimal_w1
    max_sum = throughput_sums[max_idx]

    return optimal_w1, optimal_w2, max_sum

def calculate_cutoff_allocation(pair_data):
    """
    Calculate allocation using the primary workload's cutoff and evenly distributing
    the remaining GPU percentage across other workloads.

    Args:
        pair_data: Dictionary containing workload curve fitting parameters

    Returns:
        Tuple containing:
            - allocations: list of dictionaries with workload name, allocation, throughput
            - total_throughput: sum of throughputs at the calculated allocations
    """
    workloads = pair_data.get('workloads')

    if not workloads:
        workloads = []
        idx = 1
        while True:
            workload_key = f'workload{idx}'
            cutoff_key = f'w{idx}_cutoff'
            value0_key = f'w{idx}_value0'
            slope_left_key = f'w{idx}_slope_left'
            slope_right_key = f'w{idx}_slope_right'

            if workload_key not in pair_data:
                break

            workloads.append({
                'name': pair_data[workload_key],
                'cutoff': pair_data.get(cutoff_key),
                'value0': pair_data.get(value0_key),
                'slope_left': pair_data.get(slope_left_key),
                'slope_right': pair_data.get(slope_right_key)
            })
            idx += 1

    if len(workloads) < 2:
        raise ValueError("Cutoff-based allocation requires at least two workloads.")

    primary = workloads[0]
    primary_cutoff = primary['cutoff']

    if primary_cutoff is None or pd.isna(primary_cutoff):
        raise ValueError("Primary workload cutoff value is missing.")

    remaining_gpu = 100 - primary_cutoff
    if remaining_gpu < 0:
        remaining_gpu = 0

    remaining_count = len(workloads) - 1
    shared_allocation = remaining_gpu / remaining_count if remaining_count else 0

    allocations = []
    total_throughput = 0

    for idx, workload in enumerate(workloads):
        allocation = primary_cutoff if idx == 0 else shared_allocation

        throughput = piecewise_throughput(
            allocation,
            workload['cutoff'],
            workload['value0'],
            workload['slope_left'],
            workload['slope_right']
        )

        allocations.append({
            'name': workload['name'],
            'allocation': allocation,
            'throughput': throughput
        })
        total_throughput += throughput

    return allocations, total_throughput

def create_summary_csv(workload_pairs, output_path, use_cutoff=False):
    """
    Create summary CSV with optimal allocations for all workload pairs.

    Args:
        workload_pairs: List of workload pair data
        output_path: Path to save the summary CSV
        use_cutoff: If True, use cutoff-based allocation; if False, use xput_sum optimization
    """
    summary_data = []

    for pair_data in workload_pairs:
        if use_cutoff:
            # Use cutoff-based allocation
            allocations, total_throughput = calculate_cutoff_allocation(pair_data)

            summary_row = {'max_xput_sum': total_throughput}

            for idx, allocation in enumerate(allocations, start=1):
                summary_row[f'workload{idx}'] = allocation['name']
                summary_row[f'w{idx}_optimal_percentage'] = allocation['allocation']
                summary_row[f'w{idx}_throughput_at_optimal'] = allocation['throughput']

            summary_data.append(summary_row)
        else:
            # Use xput_sum optimization (original logic)
            gpu_percentages = np.arange(0, 101, 1)  # 0 to 100% in 1% increments
            w1_throughputs, w2_throughputs, throughput_sums = calculate_throughput_sum(
                pair_data, gpu_percentages
            )

            optimal_w1, optimal_w2, max_sum = find_optimal_allocation(
                gpu_percentages, throughput_sums
            )

            # Get individual throughputs at optimal allocation
            optimal_idx = int(optimal_w1)
            w1_optimal_throughput = w1_throughputs[optimal_idx]
            w2_optimal_throughput = w2_throughputs[optimal_idx]

            summary_data.append({
                'workload1': pair_data['workload1'],
                'workload2': pair_data['workload2'],
                'w1_optimal_percentage': optimal_w1,
                'w2_optimal_percentage': optimal_w2,
                'max_xput_sum': max_sum,
                'w1_throughput_at_optimal': w1_optimal_throughput,
                'w2_throughput_at_optimal': w2_optimal_throughput
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False)
    print(f"Summary CSV saved to: {output_path}")

def create_visualization(workload_pairs, fig_dir):
    """
    Create visualization plots for throughput sum analysis.

    Args:
        workload_pairs: List of workload pair data
        fig_dir: Directory to save figures
    """
    gpu_percentages = np.arange(0, 100, 1)

    # Create a summary plot with all workload pairs
    plt.figure(figsize=(15, 10))

    for i, pair_data in enumerate(workload_pairs[:12]):  # Limit to first 12 pairs for clarity
        w1_throughputs, w2_throughputs, throughput_sums = calculate_throughput_sum(
            pair_data, gpu_percentages
        )

        optimal_w1, optimal_w2, max_sum = find_optimal_allocation(
            gpu_percentages, throughput_sums
        )

        # Plot throughput sum curve
        label = f"{pair_data['workload1'][:10]} + {pair_data['workload2'][:10]}"
        plt.plot(gpu_percentages, throughput_sums, label=label, alpha=0.7)

        # Mark optimal point
        plt.scatter([optimal_w1], [max_sum], marker='o', s=50,
                   color=plt.gca().lines[-1].get_color())

    plt.xlabel('W1 GPU Percentage (%) [W2 gets 100-x%]')
    plt.ylabel('Combined Throughput Sum')
    plt.title('Throughput Sum vs GPU Allocation for Colocated Workloads (Freq 1530)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    summary_plot_path = os.path.join(fig_dir, 'throughput_sum_summary.png')
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Summary plot saved to: {summary_plot_path}")

    # Create individual plots for top 6 workload pairs
    for i, pair_data in enumerate(workload_pairs[:6]):
        plt.figure(figsize=(10, 6))

        w1_throughputs, w2_throughputs, throughput_sums = calculate_throughput_sum(
            pair_data, gpu_percentages
        )

        optimal_w1, optimal_w2, max_sum = find_optimal_allocation(
            gpu_percentages, throughput_sums
        )

        # Plot individual throughputs
        plt.plot(gpu_percentages, w1_throughputs, label=f'{pair_data["workload1"]} throughput',
                linestyle='--', alpha=0.6)
        plt.plot(gpu_percentages, w2_throughputs, label=f'{pair_data["workload2"]} throughput (at 100-x%)',
                linestyle='--', alpha=0.6)

        # Plot sum
        plt.plot(gpu_percentages, throughput_sums, label='Combined throughput sum',
                linewidth=2)

        # Mark optimal point
        plt.scatter([optimal_w1], [max_sum], marker='o', s=100, color='red',
                   zorder=5, label=f'Optimal: W1={optimal_w1:.0f}%, W2={optimal_w2:.0f}%')

        plt.xlabel('W1 GPU Percentage (%) [W2 gets 100-x%]')
        plt.ylabel('Throughput')
        plt.title(f'Throughput Analysis: {pair_data["workload1"]} + {pair_data["workload2"]}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save individual plot
        safe_name1 = pair_data['workload1'].replace('/', '_').replace('-', '_')
        safe_name2 = pair_data['workload2'].replace('/', '_').replace('-', '_')
        plot_path = os.path.join(fig_dir, f'throughput_sum_{safe_name1}_{safe_name2}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Individual plots saved to: {fig_dir}")

def main():
    """Main function to run the throughput sum analysis."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Throughput Sum Analysis for Colocated Workloads')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--xput_sum', action='store_true',
                       help='Use throughput sum optimization (original logic)')
    group.add_argument('--cutoff', action='store_true',
                      help='Use cutoff-based allocation (primary workload gets cutoff, remainder evenly split)')

    args = parser.parse_args()

    # Set up paths
    base_dir = Path(__file__).parent
    #add n_comb
    n_comb = 3
    freq = 1530
    csv_path = base_dir / 'summary' / f'colocated_freq{freq}_comb{n_comb}_latency.csv'
    summary_file_str = f'freq{freq}_comb{n_comb}_optimal_allocation_latency.csv'

    # Set output filename based on mode
    if args.cutoff:
        mode_str = "cutoff-based"
        summary_csv_path = base_dir / 'summary' / f'cutoff_{summary_file_str}'
    else:
        mode_str = "throughput sum optimization"
        summary_csv_path = base_dir / 'summary' / f'xput_sum_{summary_file_str}'

    fig_dir = base_dir / 'fig' / f'colocated_freq{freq}'
    # Ensure directories exist
    fig_dir.mkdir(exist_ok=True)

    print(f"Starting throughput sum analysis using {mode_str}...")
    print(f"Input CSV: {csv_path}")
    print(f"Output summary CSV: {summary_csv_path}")
    print(f"Output figures directory: {fig_dir}")

    # Parse CSV data
    workload_pairs = parse_csv_data(csv_path)
    print(f"Loaded {len(workload_pairs)} workload pairs")

    # Create summary CSV with appropriate mode
    create_summary_csv(workload_pairs, summary_csv_path, use_cutoff=args.cutoff)

    # Create visualizations (only for xput_sum mode to avoid confusion)
    if args.xput_sum:
        create_visualization(workload_pairs, fig_dir)
    else:
        print("Skipping visualizations for cutoff mode (visualizations show optimization curves)")

    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()
