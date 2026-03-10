import argparse
import os
import pathlib

import numpy as np
import torch
from lightning.fabric import Fabric

from loaders import load_config, load_datamodule

from models.linear_scheduler import LinearNoiseScheduler
from models.unet import Unet

from torch.optim import Adam
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision("medium")


def train(args):
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
    dev: bool = args.dev
    compile: bool = args.compile
    resume: bool = args.resume

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
    if compile:
        scheduler = torch.compile(scheduler)
    scheduler = fabric.setup(scheduler)  # type: ignore

    # Dataloaders
    dm = load_datamodule(config.dataset)
    train_loader, _ = dm.get_dataloaders(config.dataset, dev=dev)
    train_loader = fabric.setup_dataloaders(train_loader)

    # Instantiate the model
    model = Unet(config.model)

    # If resume training, load the model.
    if resume:
        state = {"model": model}
        fabric.load(result_folder / "model.ckpt", state)
        model.train()
        model.to(fabric.device)

    if compile:
        model = torch.compile(model)

    optimizer = Adam(model.parameters(), lr=config.trainer.lr)
    criterion = torch.nn.MSELoss()

    model, optimizer = fabric.setup(model, optimizer)  # type: ignore

    model.train()
    for epoch_idx in range(config.trainer.num_epochs):
        losses = []
        for step, (im,) in enumerate(tqdm(train_loader)):
            # Accumulate gradient 8 batches at a time
            is_accumulating = step % 8 != 0
            im = im[:, :1, :, :]

            with fabric.no_backward_sync(model, enabled=is_accumulating):

                # Sample random noise
                noise = torch.randn_like(im)

                # Sample timestep
                t = torch.randint(
                    0,
                    config.scheduler.num_timesteps,
                    (im.shape[0],),
                    device=fabric.device,
                )

                # Add noise to images according to timestep
                noisy_im = scheduler.add_noise(im, noise, t)

                noise_pred = model(noisy_im, t)

                loss = criterion(noise_pred, noise)
                fabric.backward(loss)
                losses.append(loss.item())

            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()

        print(
            "Finished epoch:{} | Loss : {:.4f}".format(epoch_idx + 1, np.mean(losses))
        )

        if epoch_idx % 50 == 0:
            state = {"model": model}
            fabric.save(result_folder / f"model_{epoch_idx}.ckpt", state)

    print("Done Training ...")
    state = {"model": model}
    fabric.save(result_folder / "model.ckpt", state)


def main():
    parser = argparse.ArgumentParser(description="Arguments for ddpm training")
    _ = parser.add_argument(
        "--config",
        dest="config_path",
        default="configs/scratch/mnist.yaml",
        type=str,
    )
    _ = parser.add_argument(
        "--dev",
        default=False,
        action="store_true",
        help="Run a small subset.",
    )
    _ = parser.add_argument(
        "--compile",
        default=False,
        action="store_true",
        help="Compile modules",
    )
    _ = parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="Compile modules",
    )
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
