import argparse
import os
import pathlib

import numpy as np
import torch
from lightning.fabric import Fabric

from loaders import load_config, load_datamodule

from models.linear_scheduler import LinearNoiseScheduler
from torchvision.utils import make_grid
from models.unet import Unet

# from models.vqvae import VQVAE
from torch.optim import Adam
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision("medium")


def train(args):
    # Set up Fabric
    fabric = Fabric(accelerator="cuda")  # , precision="16-mixed")

    # Parse the args
    config_path = args.config_path
    dev: bool = args.dev
    compile: bool = args.compile

    config = load_config(config_path)

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
    _, train_loader = dm.get_dataloaders(config.dataset, dev=dev)
    train_loader = fabric.setup_dataloaders(train_loader)

    # Instantiate the model
    model = Unet(config.model)

    if compile:
        model = torch.compile(model)
    optimizer = Adam(model.parameters(), lr=config.trainer.lr)
    criterion = torch.nn.MSELoss()

    model, optimizer = fabric.setup(model, optimizer)  # type: ignore

    model.train()
    for epoch_idx in range(config.trainer.num_epochs):
        losses = []
        for (im,) in tqdm(train_loader):

            im = im[:, :1, :, :]

            optimizer.zero_grad()

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
            losses.append(loss.item())
            fabric.backward(loss)
            optimizer.step()
        print(
            "Finished epoch:{} | Loss : {:.4f}".format(epoch_idx + 1, np.mean(losses))
        )

    state = {"model": model}
    fabric.save(result_folder / "model.ckpt", state)

    print("Done Training ...")

    # model.eval()
    # with torch.no_grad():
    #     for batch in range(1):
    #         xt = torch.randn((9, 1, 28, 28)).to(device)
    #         for i in tqdm(reversed(range(config.scheduler.num_timesteps))):
    #             # Get prediction of noise
    #             noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
    #
    #             # Use scheduler to get x0 and xt-1
    #             xt, _ = scheduler.sample_prev_timestep(
    #                 xt, noise_pred, torch.as_tensor(i).to(device)
    #             )
    #             #
    #             # Save x0
    #             if i == 0:
    #                 ims = torch.clamp(xt, -1.0, 1.0).detach().cpu()
    #                 ims = (1 + torch.clamp(ims, -1.0, 1.0).detach().cpu()) / 2
    #                 grid = make_grid(ims, nrow=2)
    #                 img = torchvision.transforms.ToPILImage()(grid[:3, :, :])
    #                 img.save(result_folder / f"generated_{i}.png")
    #                 img.close()


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
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
