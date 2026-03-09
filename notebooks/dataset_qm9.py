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

config = DataConfig(
    root="./data",
    raw="./data/raw",
    batch_size=64,
    resolution=28,
    use_diracs=False,
)
train_ds, _ = create_dataset(config, dev=True, force_reload=False)

train_ds.shape

plt.imshow(train_ds[20, 0])



# get_dataloaders(config, dev=True)
# create_dataset(config, dev=False, force_reload=True)
#|%%--%%| <Ne6hNxgNeA|hQ8fqzQmde>


# |%%--%%| <hQ8fqzQmde|lyt9mIzG9q>


# |%%--%%| <lyt9mIzG9q|rgfp2PMuml>
import torch

diffs = torch.diff(train_ds[:64, :1, :, :].squeeze(), dim=1)

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

# |%%--%%| <rgfp2PMuml|6OOwVHLOS4>
import os
import torch

dataset_type = "prod"

path = f"{config.root}/qm9/{dataset_type}"
raw_path = f"{config.raw}/qm9"
print(dataset_type)
print("Creating:", os.path.dirname(path))
print("Creating:", os.path.dirname(raw_path))

os.makedirs(path, exist_ok=True)
os.makedirs(raw_path, exist_ok=True)

torch.manual_seed("1337")

# Download the full QM9 dataset.
dataset = QM9(
    root=raw_path,
    force_reload=False,
).shuffle()

rads = []

for data in dataset:
    rads.append(torch.tensor(data.pos).norm(dim=-1).max())

# |%%--%%| <6OOwVHLOS4|HQ8c0sVm9f>

print(max(torch.tensor(rads)))
