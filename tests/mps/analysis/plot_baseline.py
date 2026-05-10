import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

# Example: Reading your data into a DataFrame
data = pd.DataFrame({
    'Type': [
        'resnet-50_batch2-train',
        'bert-base-cased_batch2-train',
        'albert-base-v2_batch2-train',
        'vit_h_14_batch2-train',
        'mobilenet_batch2-train',
        'resnet-50_batch2-inf',
        'vit-base-patch16-224_batch2-inf',
        'mobilenet_v2_1.0_224_batch2-inf',
        'bert-base-cased_batch2-inf',
        'wav2vec2-base-960h_batch2-inf',
        'whisper-large-v2_batch2-inf',
        'sortingNetworks_batch2-cuda_samples',
        'BlackScholes_batch2-cuda_samples'
    ],
    'sm%': [
        33.25, 96.71428571, 97.16666667, 89.3, 17.0,
        16.5, 24.4, 7.0, np.nan, 64.8, 86.24137931,
        96.46, 99.0
    ],
    'mem%': [
        8.75, 47.0, 46.0, 45.83333333, 0.0,
        3.0, 3.8, 0.0, np.nan, 24.2, 30.15517241,
        84.42, 99.0
    ]
})


# Boundaries for categories
x_boundaries = [0, 15, 40, 110]  # extended to 110 just in case
y_boundaries = [0, 40, 80, 110]  # extended to 110 just in case

# Colors for each of the 9 regions (3x3)
colors = [
    ["#ffcccc", "#ffd9cc", "#ffe5cc"],  # bottom row colors
    ["#ccffcc", "#ccffd9", "#ccffe5"],  # middle row
    ["#ccccff", "#d9ccff", "#e5ccff"]   # top row
]

fig, ax = plt.subplots(figsize=(10, 6))

# Draw rectangles for each category region
for i in range(3):
    for j in range(3):
        # Calculate the width and height of the rectangle
        x_start = x_boundaries[j]
        x_end = x_boundaries[j+1]
        y_start = y_boundaries[i]
        y_end = y_boundaries[i+1]

        rect = Rectangle((x_start, y_start),
                         x_end - x_start,
                         y_end - y_start,
                         facecolor=colors[i][j],
                         edgecolor='none',
                         alpha=0.3)
        ax.add_patch(rect)

# Plot the scatter points on top
sc = ax.scatter(data['mem%'], data['sm%'], s=100, alpha=0.7, edgecolors='black')

# Annotate points
for i, row in data.iterrows():
    ax.text(row['mem%']+1, row['sm%']+1, row['Type'], fontsize=8)

# Draw the boundary lines in bold
ax.axvline(x=15, color='red', linewidth=2.5)
ax.axvline(x=60, color='red', linewidth=2.5)
ax.axhline(y=40, color='red', linewidth=2.5)
ax.axhline(y=80, color='red', linewidth=2.5)

# Set labels
ax.set_xlabel('Memory Utilization (mem%)')
ax.set_ylabel('Compute Utilization (sm%)')
ax.set_title('Workloads: Compute vs Memory Utilization with Categorized Regions')

# Add a grid
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('workloads_compute_vs_memory.png')

