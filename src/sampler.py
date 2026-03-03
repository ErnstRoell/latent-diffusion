import argparse
import os

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

    config = load_config(args.config_path)

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(config=config.scheduler)

    model = Unet(config.model)
    # if compile:
    #     vae = torch.compile(vae)

    state = {"model": model}
    fabric.load("trained_models/ddpm.ckpt", state)

    model.eval()
    model.to(fabric.device)

    for batch in range(1):
        xt = torch.randn((1, 1, 28, 28)).to(device)
        for i in tqdm(reversed(range(config.scheduler.num_timesteps))):
            # Get prediction of noise
            noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))

            # Use scheduler to get x0 and xt-1
            xt, x0_pred = scheduler.sample_prev_timestep(
                xt, noise_pred, torch.as_tensor(i).to(device)
            )
            #
            # Save x0
            if i == 0:
                ims = torch.clamp(xt, -1.0, 1.0).detach().cpu()
                ims = (1 + torch.clamp(ims, -1.0, 1.0).detach().cpu()) / 2
                grid = make_grid(ims, nrow=2)
                img = torchvision.transforms.ToPILImage()(grid[:3, :, :])
                img.save(f"results/generated_{i}.png")
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
