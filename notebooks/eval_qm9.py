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
import torch

#|%%--%%| <hQ8fqzQmde|YcPac3Eaka>
"""
Vis of the dataset
"""

ects = (torch.load("data/qm9/dev/test.pt") + 1) / 2
ects = ects[:64,:1,:,:]
print(ects.shape)

grid = make_grid(ects, nrow=8)
img = torchvision.transforms.ToPILImage()(grid[:, :, :])
#
plt.imshow(img)
plt.show()

diffs = torch.diff(ects[:64, :1, :, :].squeeze(), dim=1)
theta = torch.linspace(0, 360, 64).numpy()

grid = make_grid(diffs.unsqueeze(1), nrow=8)
img = torchvision.transforms.ToPILImage()(grid[:, :, :])
#
plt.imshow(img)
plt.show()

print(diffs.min())
print(diffs.max())


imgs = []
for idx, rec in enumerate(diffs.squeeze()):
    recon_fbp = iradon(rec.squeeze().numpy(), 
                       theta=theta, 
                       filter_name=None)
    imgs.append(5*recon_fbp)

recons = torch.tensor(np.stack(imgs)).unsqueeze(1)
recons.shape

grid = make_grid(recons, nrow=8)
img = torchvision.transforms.ToPILImage()(grid[:, :, :])
#
plt.imshow(img)


# |%%--%%| <YcPac3Eaka|lyt9mIzG9q>


ects = (torch.load("results/qm9/generated_ects_0.pt") + 1) / 2
# ects = torch.load("results/qm9/generated_ects_0.pt") 
ects = torch.clip(ects,min=0,max=1)

print(ects.shape)

print(ects.min())
print(ects.max())

plt.imshow(ects[0].cpu().squeeze())


# |%%--%%| <lyt9mIzG9q|rgfp2PMuml>

diffs = torch.diff(ects[:64, :1, :, :].squeeze(), dim=1)

plt.imshow(diffs[0].cpu().squeeze())

print(diffs[0].sum(dim=0).round(decimals=4))



#|%%--%%| <rgfp2PMuml|8t7drSdY7C>

grid = make_grid(diffs.unsqueeze(1), nrow=8)
img = torchvision.transforms.ToPILImage()(grid[:, :, :])
#
plt.imshow(img)

#|%%--%%| <8t7drSdY7C|Lk0S0wQDlT>


theta = torch.linspace(0, 360, 64).numpy()

imgs = []
for idx, rec in enumerate(diffs.squeeze()):
    recon_fbp = iradon(rec.squeeze().numpy(), 
                       theta=theta, 
                       filter_name=None)
    imgs.append(5*recon_fbp)

recons = torch.tensor(np.stack(imgs)).unsqueeze(1)
recons.shape

grid = make_grid(recons, nrow=8)
img = torchvision.transforms.ToPILImage()(grid[:, :, :])
#
plt.imshow(img)
