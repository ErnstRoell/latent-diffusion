%load_ext autoreload    
%autoreload 2
from datasets.qm9 import DataConfig, get_dataloaders, create_dataset, QM9
import torch
from skimage.transform import iradon
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import torchvision
import matplotlib.pyplot as plt

# |%%--%%| <hQ8fqzQmde|lyt9mIzG9q>

ects = (torch.load("results/qm9/generated_ects_0.pt") + 1) / 2
# ects = torch.load("results/qm9/generated_ects_0.pt") 
ects = torch.clip(ects,min=0,max=1)

print(ects.min())
print(ects.max())

plt.imshow(ects[0].cpu().squeeze())

# |%%--%%| <lyt9mIzG9q|rgfp2PMuml>
import torch

diffs = torch.diff(ects[:64, :1, :, :].squeeze(), dim=1)
theta = torch.linspace(0, 360, 28).numpy()

print(diffs.shape)

imgs = []
for idx, rec in enumerate(diffs.squeeze()):
    recon_fbp = iradon(rec.squeeze().numpy(), theta=theta, filter_name=None)
    imgs.append(recon_fbp)

recons = torch.tensor(np.stack(imgs)).unsqueeze(1)
recons.shape

grid = make_grid(recons, nrow=8)
img = torchvision.transforms.ToPILImage()(grid[:, :, :])
#
plt.imshow(img)

