"""
This is a single point ECT dataset.
Computes the ECT of a single point either with random directions or with
structured directions.
"""

import os
from dataclasses import dataclass

import torch
from dect.directions import generate_uniform_directions, generate_2d_directions
from dect.ect import compute_ect_point_cloud
import torch


class Normalize:
    def __call__(self, img):
        img = 2 * img - 1
        return img


@dataclass
class DataConfig:
    module: str
    structured_directions: bool
    root: str
    raw: str
    batch_size: int


def create_dataset(config: DataConfig, dev: bool = False):
    """
    Create the datasets for processing. Creates either
    the dev dataset or the full dataset.
    """
    g = torch.Generator(device="cpu").manual_seed(9)

    dataset_name = "ptect"
    dataset_type = "dev" if dev else "prod"

    path = f"{config.root}/{dataset_name}/{dataset_type}"
    raw_path = f"{config.raw}/{dataset_name}"
    print(dataset_type)
    print("Creating:", os.path.dirname(path))
    print("Creating:", os.path.dirname(raw_path))

    os.makedirs(path, exist_ok=True)
    os.makedirs(raw_path, exist_ok=True)

    pts_train = (torch.rand(5000, 1, 2, generator=g) * 2 - 1) * 0.7
    pts_test = (torch.rand(5000, 1, 2, generator=g) * 2 - 1) * 0.7
    v = generate_2d_directions(28)

    ect_train = (
        2
        * compute_ect_point_cloud(
            pts_train, v, radius=1, resolution=28, scale=500
        ).unsqueeze(1)
        - 1
    )
    ect_test = (
        2
        * compute_ect_point_cloud(
            pts_test, v, radius=1, resolution=28, scale=500
        ).unsqueeze(1)
        - 1
    )

    # Create the permutation.
    perm = torch.randperm(28, generator=g)

    ect_train_perm = ect_train[:, :, :, perm]
    ect_test_perm = ect_test[:, :, :, perm]

    if dev:
        ect_train = ect_train[:32]
        ect_test = ect_test[:32]
        ect_train_perm = ect_train_perm[:32]
        ect_test_perm = ect_test_perm[:32]

    torch.save(ect_train, f"{path}/train_ims.pt")
    torch.save(ect_test, f"{path}/test_ims.pt")
    torch.save(ect_train_perm, f"{path}/train_ims_perm.pt")
    torch.save(ect_test_perm, f"{path}/test_ims_perm.pt")
    torch.save(perm, f"{path}/perm.pt")

    return ect_train, ect_test


def get_dataloaders(config: DataConfig, dev: bool = False):
    """Returns two dataloaders, train and test."""

    dataset_type = "dev" if dev else "prod"
    dataset_name = "ptect"

    path = f"{config.root}/{dataset_name}/{dataset_type}"

    if config.structured_directions:
        train_ims = torch.load(f"{path}/train_ims.pt")
        test_ims = torch.load(f"{path}/test_ims.pt")
    else:
        train_ims = torch.load(f"{path}/train_ims_perm.pt")
        test_ims = torch.load(f"{path}/test_ims_perm.pt")

    train_ds = torch.utils.data.TensorDataset(train_ims)

    test_ds = torch.utils.data.TensorDataset(test_ims)
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True if not dev else False,
    )

    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_dl, test_dl


if __name__ == "__main__":
    config = DataConfig(
        root="./data",
        raw="./data/raw",
        structured_directions=False,
        module="datasets.ptect",
        batch_size=64,
    )
    create_dataset(config, dev=True)
    create_dataset(config, dev=False)
    train_dl, test_dl = get_dataloaders(config, dev=True)
