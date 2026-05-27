import os
from configs import Configuration

import torch
from torchvision.datasets import MNIST
from torchvision import transforms


class Normalize:
    def __call__(self, img):
        img = 2 * img - 1
        return img


# @dataclass
class DataConfig(Configuration):
    module: str
    root: str
    raw: str
    batch_size: int


def create_dataset(config: DataConfig, dev: bool = False):
    """
    Create the datasets for processing. Creates either
    the dev dataset or the full dataset.
    """

    dataset_type = "dev" if dev else "prod"

    path = f"{config.root}/mnist/{dataset_type}"
    raw_path = f"{config.raw}/mnist"
    print(dataset_type)
    print("Creating:", os.path.dirname(path))
    print("Creating:", os.path.dirname(raw_path))

    os.makedirs(path, exist_ok=True)
    os.makedirs(raw_path, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            Normalize(),
        ]
    )

    # Download the full MNIST dataset.
    mnist_train = MNIST(root=raw_path, transform=transform, train=True, download=True)
    mnist_test = MNIST(root=raw_path, transform=transform, train=False, download=True)

    if dev:
        mnist_train = torch.utils.data.Subset(mnist_train, torch.arange(0, 2048))  # type: ignore
        mnist_test = torch.utils.data.Subset(mnist_test, torch.arange(0, 2048))  # type: ignore

    # Convert to list.
    train_ims = []
    train_ys = []

    for el in mnist_train:
        train_ims.append(el[0])
        train_ys.append(torch.tensor([el[1]]))

    train_ims = torch.cat(train_ims).unsqueeze(1)
    train_ys = torch.cat(train_ys)

    test_ims = []
    test_ys = []

    for el in mnist_test:
        test_ims.append(el[0])
        test_ys.append(torch.tensor([el[1]]))

    test_ims = torch.cat(test_ims).unsqueeze(1)
    test_ys = torch.cat(test_ys)

    torch.save(train_ims, f"{path}/train_ims.pt")
    torch.save(train_ys, f"{path}/train_ys.pt")

    torch.save(test_ims, f"{path}/test_ims.pt")
    torch.save(test_ys, f"{path}/test_ys.pt")

    return (train_ims, train_ys), (test_ims, test_ys)


def get_dataloaders(config: DataConfig, dev: bool = False):
    """Returns two dataloaders, train and test."""

    dataset_type = "dev" if dev else "prod"
    path = f"{config.root}/mnist/{dataset_type}"

    train_ims = torch.load(f"{path}/train_ims.pt")
    train_ys = torch.load(f"{path}/train_ys.pt")

    test_ims = torch.load(f"{path}/test_ims.pt")
    test_ys = torch.load(f"{path}/test_ys.pt")

    train_ds = torch.utils.data.TensorDataset(train_ims,train_ys)

    test_ds = torch.utils.data.TensorDataset(test_ims, test_ys)
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


def main():
    config = DataConfig(
        root="./data",
        raw="./data/raw",
        module="datasets.mnist",
        batch_size=64,
    )
    create_dataset(config, dev=True)
    create_dataset(config, dev=False)
    # train_dl, test_dl = get_dataloaders(config, dev=True)

def setup():
    return DataConfig, get_dataloaders

if __name__ == "__main__":
    main()
