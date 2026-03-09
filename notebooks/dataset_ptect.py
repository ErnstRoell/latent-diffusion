from dect.directions import generate_uniform_directions, generate_2d_directions
from dect.ect import compute_ect_point_cloud
import torch
import matplotlib.pyplot as plt

g = torch.Generator(device="cpu").manual_seed(9)

pts = (torch.rand(10, 1, 2, generator=g) * 2 - 1) * 0.7
v = generate_2d_directions(28)

perm = torch.randperm(28, generator=g)
v = v[:, perm]
ect = compute_ect_point_cloud(pts, v, radius=1, resolution=28, scale=500)

plt.imshow(ect[5].squeeze())


# |%%--%%| <F2cRGBjj5K|roGUtyABut>

inv_perm = torch.argsort(perm)
ect_inv = ect[:, :, inv_perm]
plt.imshow(ect_inv[5].squeeze())


# |%%--%%| <roGUtyABut|h2CR45RJT4>

from datasets.ptect import DataConfig, create_dataset, get_dataloaders

config = DataConfig(
    root="./data",
    raw="./data/raw",
    structured_directions=True,
    module="datasets.ptect",
    batch_size=64,
)
create_dataset(config, dev=True)
train_ect, _ = create_dataset(config, dev=False)
train_dl, test_dl = get_dataloaders(config, dev=True)

plt.imshow(train_ect[0].squeeze().cpu())

print(train_ect.min())
print(train_ect.max())

print(train_ect[0].min())
print(train_ect[0].max())
