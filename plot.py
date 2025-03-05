import matplotlib.pyplot as plt

import numpy as np
plt.figure(figsize=(8, 6))

plt.plot(np.arange(1,6), [75.62,78.19,70.11,67.85,62.12], marker='o', color='b', linestyle='-', linewidth=2, markersize=8, label="LOGO")
plt.plot(np.arange(1,6), [75.99,86.20,82.98,79.11,77.83], marker='*', color='r', linestyle='-', linewidth=2, markersize=8, label="Fis-V PCS")
plt.plot(np.arange(1,6), [65.49,72.10,67.44,65.84,65.31], marker='s', color='g', linestyle='-', linewidth=2, markersize=8, label="Fis-V TES")

plt.title('SRCC vs. Number of Layers', fontsize=16)
plt.xticks(np.arange(1,6))
plt.xlabel('Number of Layers', fontsize=14)
plt.ylabel('SRCC', fontsize=14)
plt.grid(True)
plt.legend(loc="upper right")  

plt.savefig("layers.png")