#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_e2e(csv_file, target_workload, modeltype, output_dir):
    """
    Read a CSV of end-to-end results and produce a single figure with
    three subplots:
      A) Ground Truth vs. Predicted
      B) Ground Truth vs. Pred-execution
      C) Predicted vs. Pred-execution
    
    Each row is drawn in its own color, but the legend references marker shapes
    (GT vs. Pred vs. Pred-execution) rather than row colors.
    """
    # Read DataFrame
    csv_file = os.path.join(os.getcwd(), "../../../../", csv_file)
    df = pd.read_csv(csv_file)
    if df.empty:
        print(f"WARNING: CSV {csv_file} is empty. No plot generated.")
        return
    
    # Prepare figure: 1 row, 3 columns, shared y-axis
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4), sharey=True)
    
    # Generate one color per row
    colors = plt.cm.tab20(np.linspace(0, 1, len(df)))  # or use another colormap if many rows
    
    # ----- Subplot A: GT vs. Pred -----
    axes[0].scatter([], [], marker='o', color='black', label='GT')
    axes[0].scatter([], [], marker='x', color='black', label='Predicted')
    axes[0].set_title("A) GT vs. Pred")
    axes[0].set_xlabel("Power (W)")
    axes[0].set_ylabel("Throughput")
    axes[0].legend()
    axes[0].set_ylim(0.8, 1.8)
    
    # ----- Subplot B: Pred vs. Pred-execution (SWAPPED) -----
    axes[1].scatter([], [], marker='x', color='black', label='Predicted')
    axes[1].scatter([], [], marker='^', color='black', label='Pred-execution')
    axes[1].set_title("B) Pred vs. Pred-execution")
    axes[1].set_xlabel("Power (W)")
    axes[1].legend()
    
    axes[1].set_ylim(0.8, 1.8)
    
    # ----- Subplot C: GT vs. Pred-execution (SWAPPED) -----
    axes[2].scatter([], [], marker='o', color='black', label='GT')
    axes[2].scatter([], [], marker='^', color='black', label='Pred-execution')
    axes[2].set_title("C) GT vs. Pred-execution")
    axes[2].set_xlabel("Power (W)")
    axes[2].legend()
    #limit y axis to 0.8-1.8
    
    axes[2].set_ylim(0.8, 1.8)
    
    # Plot each row's data with different color but no label
    for idx, row in df.iterrows():
        c = colors[idx]
        # Subplot A: GT vs. Pred
        axes[0].scatter(
            row["best_power"], row["best_throughput"],
            color=c, marker='o'
        )
        axes[0].scatter(
            row["pred_power"], row["pred_throughput"],
            color=c, marker='x'
        )

        # Subplot B: Pred vs. Pred-execution
        axes[1].scatter(
            row["pred_power"], row["pred_throughput"],
            color=c, marker='x'
        )
        axes[1].scatter(
            row["pred_actual_power"], row["pred_actual_throughput"],
            color=c, marker='^'
        )

        # Subplot C: GT vs. Pred-execution
        axes[2].scatter(
            row["best_power"], row["best_throughput"],
            color=c, marker='o'
        )
        axes[2].scatter(
            row["pred_actual_power"], row["pred_actual_throughput"],
            color=c, marker='^'
        )
    
    # Overall title
    fig.suptitle(f"Model: {modeltype} | Workload: {target_workload}", fontsize=14)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Build a filename
    out_file = os.path.join(output_dir, f"{modeltype}_{target_workload}_e2e.png")
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot E2E results.")
    parser.add_argument(
        "--csv_file", required=True,
        help="Path to the CSV (e2e_results.csv)."
    )
    parser.add_argument(
        "--target_workload", default="whisper-large-v2_batch2-inf",
        choices=[
            "whisper-large-v2_batch2-inf",
            "mobilenet_batch2-train",
            "bert-base-cased_batch2-train"
        ],
        help="Target workload name (just for labeling)."
    )
    parser.add_argument(
        "--modeltype", default="linear",
        choices=["linear", "AutoML"],
        help="Model type (just for labeling)."
    )
    parser.add_argument(
        "--output_dir", default=".",
        help="Directory to save the resulting plot."
    )
    args = parser.parse_args()

    plot_e2e(
        csv_file=args.csv_file,
        target_workload=args.target_workload,
        modeltype=args.modeltype,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
