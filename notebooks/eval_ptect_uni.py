import torch
from skimage.transform import iradon
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import torchvision

# Load the generated tensors

ects = (torch.load("results/ptect_uniform/generated_ects_0.pt") + 1) / 2

print("----global min max----")
print(ects.min())
print(ects.max())

print("----local min max----")
print(ects[0].min())
print(ects[0].max())

ects = torch.clamp(ects, 0, 1)

plt.imshow(ects[0].detach().squeeze())

grid = make_grid(ects, nrow=8)
img = torchvision.transforms.ToPILImage()(grid[:, :, :])
#
plt.imshow(img)


# |%%--%%| <02GfimQmvW|1E2bcLyGPq>

perm = torch.load("data/ptect/prod/perm.pt")
perm_inv = torch.argsort(perm)

ect_inv = ects[:, :, :, perm_inv]

grid = make_grid(ect_inv, nrow=8)
img = torchvision.transforms.ToPILImage()(grid[:, :, :])
#
plt.imshow(img)


# |%%--%%| <1E2bcLyGPq|0bzTG7eEGC>

diffs = torch.diff(ect_inv, dim=2)

theta = torch.linspace(0, 360, 28).numpy()

imgs = []
for idx, rec in enumerate(diffs.squeeze()):
    recon_fbp = iradon(rec.numpy(), theta=theta, filter_name=None)
    imgs.append(recon_fbp)

recons = torch.tensor(np.stack(imgs)).unsqueeze(1)
recons.shape

grid = make_grid(recons, nrow=8)
img = torchvision.transforms.ToPILImage()(grid[:, :, :])
#
plt.imshow(img)


# fig, axes = plt.subplots(8, 8, figsize=(5, 5))

# for axis, rec in zip(axes.T, diffs.squeeze()):
#
#     ax[0].imshow(recon_fbp)
#     ax[0].axis("off")
#     # ax[2].scatter(pc[:, 0], pc[:, 1])
#     # ax[2].axis("off")
#
# fig.tight_layout()
