"""Print the class distribution of the training set and save one example image per class.

Usage: python explore.py   (writes examples.png)
"""

import matplotlib.pyplot as plt
import numpy as np
from medmnist import DermaMNIST

from common import CLASS_NAMES, SIZE

# Raw images (no transform) so they can be displayed.
train = DermaMNIST(split="train", download=True, size=SIZE)
labels = train.labels.flatten()

print("Training images:", len(train))
counts = np.bincount(labels, minlength=len(CLASS_NAMES))
for name, count in zip(CLASS_NAMES, counts):
    print(f"{count:5d}  {name}")

# One example per class in a 2 x 4 grid (the 8th cell stays empty).
fig, axes = plt.subplots(2, 4, figsize=(11, 6))
for c, ax in enumerate(axes.flat):
    ax.axis("off")
    if c < len(CLASS_NAMES):
        i = np.where(labels == c)[0][0]
        ax.imshow(train[i][0])
        ax.set_title(CLASS_NAMES[c][:25], fontsize=9)

plt.tight_layout()
plt.savefig("examples.png")
plt.show()
