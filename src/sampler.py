import argparse
import os
import pathlib

import torch
import torchvision
from lightning.fabric import Fabric
from torchvision.utils import make_grid
from tqdm import tqdm

from loaders import load_config
from models.linear_scheduler import LinearNoiseScheduler
from models.unet import Unet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def main(args):

    # Set up Fabric
    fabric = Fabric(accelerator="cuda")  # , precision="16-mixed")
    config_path = args.config_path

    config = load_config(config_path)

    # Make result path
    result_folder_list = ["results"] + list(pathlib.Path(config_path).parent.parts[1:])
    result_folder = pathlib.Path(*result_folder_list)

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(config=config.scheduler)

    model = Unet(config.model)
    state = {"model": model}
    fabric.load(result_folder / "model.ckpt", state)

    model.eval()
    model.to(fabric.device)

    for batch in range(1):
        xt = torch.randn((64, 1, 28, 28)).to(device)
        for i in tqdm(reversed(range(config.scheduler.num_timesteps))):
            # Get prediction of noise
            noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))

            # Use scheduler to get x0 and xt-1
            xt, _ = scheduler.sample_prev_timestep(
                xt, noise_pred, torch.as_tensor(i).to(device)
            )
            #
            # Save x0
            if i == 0:
                # Save as (raw) tensors

                torch.save(xt.cpu().detach(), f"{result_folder}/generated_ects_{i}.pt")

                # Save as imgs
                ims = torch.clamp(xt, -1.0, 1.0).detach().cpu()
                ims = (1 + torch.clamp(ims, -1.0, 1.0).detach().cpu()) / 2
                grid = make_grid(ims, nrow=8)
                img = torchvision.transforms.ToPILImage()(grid[:, :, :])
                img.save(result_folder / f"generated_{i}.png")
                img.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for ddpm image generation")
    parser.add_argument(
        "--config",
        dest="config_path",
        default="configs/mnist.yaml",
        type=str,
    )

    # parser.add_argument(
    #     "--vae_config",
    #     dest="vae_config",
    #     default="configs/vqvae_airplane.yaml",
    #     type=str,
    # )
    # parser.add_argument(
    #     "--encoder_config",
    #     dest="encoder_config",
    #     default="configs/encoder_airplane.yaml",
    #     type=str,
    # )
    args = parser.parse_args()
    main(args)
