"""
Visualization script for the mnist dataset.
"""

import torch
import matplotlib.pyplot as plt

from datasets.mnist import create_dataset, DataConfig

config = DataConfig(
    root="./data",
    raw="./data/raw",
    module="datasets.mnist",
    batch_size=64,
)
(img, y), _ = create_dataset(config, dev=True)


# |%%--%%| <NzAIAeSNEF|t74oiTfdST>


# First image analysis.

print("Shape of image tensor:", img.shape)
print("Min value in the images:", img.min())
print("Max value in the images:", img.max())

# # Plot a grid of images.
# fig, axes = plt.subplots(nrows=1, ncols=3)
# for im, axis in zip(img, axes.T):
#     axis.imshow(im)
#     axis.set_aspect(1)
#     axis.axis("off")
# plt.tight_layout()


# |%%--%%| <t74oiTfdST|V7Hi0DR9be>

from collections import Counter

print("Class distribution")
counter = Counter(y.tolist())

print("Label", "\t", "Count")
for label, count in sorted(counter.items()):
    print(label, "\t", count)


# plt.hist(y)
# plt.plot()
