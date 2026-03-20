"""Integration test for configurations."""

import argparse
import os
import pathlib

import torch
from lightning.fabric import Fabric

from structlog import get_logger

from loaders import load_config, load_datamodule

from models.linear_scheduler import LinearNoiseScheduler
from models.unet import Unet

from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = get_logger()

torch.set_float32_matmul_precision("medium")


def validate(args):
    # Set up Fabric
    fabric = Fabric(
        accelerator="cuda",
        precision="16-mixed",
    )

    ###################
    #  Parse configs  #
    ###################

    # Parse the args
    config_path = args.config_path

    config = load_config(config_path)

    #################################
    #  Set up directory structures  #
    #################################

    # Create paths for the results
    result_folder_list = ["results"] + list(pathlib.Path(config_path).parent.parts[1:])
    result_folder = pathlib.Path(*result_folder_list)

    if not result_folder.exists():
        os.makedirs(result_folder, exist_ok=True)

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(config=config.scheduler)
    scheduler = fabric.setup(scheduler)  # type: ignore

    # Dataloaders
    dm = load_datamodule(config.dataset)
    train_loader, test_loader = dm.get_dataloaders(config.dataset, dev=True)
    train_loader = fabric.setup_dataloaders(train_loader)
    test_loader = fabric.setup_dataloaders(test_loader)

    # Instantiate the model
    model = Unet(config.model)
    epoch = 0

    optimizer = Adam(model.parameters(), lr=config.trainer.lr)
    criterion = torch.nn.MSELoss()

    model, optimizer = fabric.setup(model, optimizer)  # type: ignore

    model.train()
    logger.info(f"Start training from epoch {epoch} ...")
    for epoch_idx in range(2):
        logger.info(f"Epoch {epoch} ...")
        im = torch.empty(5, 1, 28, 28).to(fabric.device)

        # Sample random noise
        noise = scheduler.sample_noise(im)

        t = scheduler.sample_timestep(im)

        # Add noise to images according to timestep
        noisy_im = scheduler.add_noise(im, noise, t)

        noise_pred = model(noisy_im, t)

        loss = criterion(noise_pred, noise)
        fabric.backward(loss)

        optimizer.step()
        optimizer.zero_grad()


def main():
    parser = argparse.ArgumentParser(description="Arguments for ddpm training")
    _ = parser.add_argument(
        "--config",
        dest="config_path",
        default="configs/scratch/mnist.yaml",
        type=str,
    )
    args = parser.parse_args()

    validate(args)


if __name__ == "__main__":
    main()
