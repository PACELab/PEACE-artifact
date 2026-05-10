import json
import sys
import random
import numpy as np
with open(sys.argv[1], 'r') as file:
    data = json.load(file)
    
    



print(len(data))
print(f"mean of {sys.argv[1]} Inter arrival time-  {np.mean(data)}")
import matplotlib.pyplot as plt

plt.hist(data, bins=len(data)//5)
plt.savefig(sys.argv[1]+ ".jpg")

fig, axs = plt.subplots(2, 3)
test_lambda = [1, 0.1]
morereq, downsampling = [], []
for  lamb in test_lambda:
    morereq.append([random.expovariate(lamb) for _ in range(100000)])
    print(f"mean of sampled Inter arrival time-  {np.mean(morereq[-1])}")
    downsampling.append([random.expovariate(lamb) for _ in range(200)])
    print(f"mean of downsampled Inter arrival time-  {np.mean(downsampling[-1])}")


# Plot histograms
axs[0, 0].hist(morereq[0], bins=100, density=True)
axs[0, 0].set_title(f'lambda={test_lambda[0]} num_samples=100000', fontsize=5)

axs[0, 1].hist(downsampling[0], bins=200//10, density=True)
axs[0, 1].set_title(f'lambda={test_lambda[0]}  - num_samples=200', fontsize=5)

axs[1, 0].hist(morereq[1], bins=100, density=True)
axs[1, 0].set_title(f'lambda={test_lambda[1]}  num_samples=100000', fontsize=5)

axs[1, 1].hist(downsampling[1], bins=200//10, density=True)
axs[1, 1].set_title(f'lambda={test_lambda[1]}  num_samples=200', fontsize=5)

axs[0, 2].hist(data, bins=len(data)//10, density=True)
axs[0, 2].set_title(f'SAMPLED DATA - num_samples=200', fontsize=5)

# Adjust layout to prevent overlap of titles
plt.tight_layout()

# Save the figure
plt.savefig("combined_exponential.jpg")
