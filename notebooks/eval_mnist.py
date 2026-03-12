%load_ext autoreload    
%autoreload 2
import torch
from skimage.transform import iradon
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import torchvision

# Load the generated tensors

im = (torch.load("results/scratch/unet_mnist_test_0000_generated.pt") + 1) / 2

print("----global min max----")
print(im.min())
print(im.max())

print("----local min max----")
print(im[0].min())
print(im[0].max())

im = (torch.load("results/scratch/unet_0000_generated.pt") + 1) / 2
im = torch.clamp(im, 0, 1)
grid = make_grid(im, nrow=8)
img = torchvision.transforms.ToPILImage()(grid[:, :, :])
#
plt.imshow(img)


#|%%--%%| <02GfimQmvW|jPzjAcjYTe>

"""
Plot the training metrics.
"""

import pandas as pd

df = pd.read_csv("results/scratch/logs/unet_zero_snr/trainlogs/metrics.csv")
df_base = pd.read_csv("results/scratch/logs/unet/trainlogs/metrics.csv")

# Plot validation loss 
plt.plot(df["train_loss"])
plt.plot(df[["val_loss"]].dropna())
plt.plot(df_base["train_loss"])
plt.plot(df_base[["val_loss"]].dropna())





# |%%--%%| <jPzjAcjYTe|0bzTG7eEGC>


"""
Evaluate the metrics.
"""

from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import Resize
import glob

resize = Resize(299)

real_mnist = torch.load("data/mnist/dev/test_ims.pt")[:64,:,:]
real_mnist = (255*(real_mnist + 1 ) / 2).to(torch.uint8)
real_mnist = resize(real_mnist).repeat(1,3,1,1)
print(real_mnist.shape)

checkpoint_files = glob.glob(f"results/scratch/unet*.pt")
for checkpoint in checkpoint_files:
    im = (torch.load(checkpoint) + 1) / 2
    im = torch.clamp(im, 0, 1)
    generated_mnist =(255*im).to(torch.uint8) 
    generated_mnist = resize(generated_mnist).repeat(1,3,1,1)
    fid = FrechetInceptionDistance(feature=64)
    fid.update(real_mnist, real=True)
    fid.update(generated_mnist, real=False)
    print(checkpoint,"\t",fid.compute())
