import argparse
import pathlib

import torch
import torchvision
from lightning.fabric import Fabric
from torchvision.utils import make_grid
from tqdm import tqdm

from loaders import load_config
from models.linear_scheduler import LinearNoiseScheduler
from models.unet import Unet
import glob
import structlog
import logging

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def main(args):

    # Set up Fabric
    fabric = Fabric(accelerator="cuda", precision="16-mixed")
    config_path = args.config_path
    num_samples = args.num_samples

    config = load_config(config_path)

    # Make result path
    result_folder_list = ["results"] + list(pathlib.Path(config_path).parent.parts[1:])
    result_folder = pathlib.Path(*result_folder_list)

    if args.sample_all:
        checkpoint_files = glob.glob(f"{result_folder}/{config.meta.modelname}*.ckpt")
    else:
        checkpoint_files = [f"{result_folder}/{config.meta.modelname}_9999.ckpt"]

    for checkpoint in checkpoint_files:
        # Get model name:
        modelname = pathlib.Path(checkpoint).stem
        print(modelname)

        # Create the noise scheduler
        scheduler = LinearNoiseScheduler(config=config.scheduler).to(fabric.device)

        model = Unet(config.model)
        state = {"model": model}
        fabric.load(checkpoint, state)

        if compile:
            model = torch.compile(model)

        model.eval()
        model.to(fabric.device)

        for batch in range(1):
            xt = torch.randn((64, 1, 28, 28)).to(fabric.device)
            for i in tqdm(reversed(range(config.scheduler.num_timesteps - 1))):
                # Get prediction of noise
                noise_pred = model(
                    xt, torch.as_tensor(i).unsqueeze(0).to(fabric.device)
                )

                # Use scheduler to get x0 and xt-1
                xt, _ = scheduler.sample_prev_timestep(
                    xt, noise_pred, torch.as_tensor(i).to(fabric.device)
                )
                #
                # Save x0
                if i == 0:

                    # Save as (raw) tensors
                    torch.save(
                        xt.cpu().detach(), f"{result_folder}/{modelname}_generated.pt"
                    )

                    # Save as imgs
                    ims = torch.clamp(xt, -1.0, 1.0).detach().cpu()
                    ims = (1 + torch.clamp(ims, -1.0, 1.0).detach().cpu()) / 2
                    grid = make_grid(ims, nrow=8)
                    img = torchvision.transforms.ToPILImage()(grid[:, :, :])
                    img.save(result_folder / f"{modelname}_generated.png")
                    img.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for ddpm image generation")
    parser.add_argument(
        "--config",
        dest="config_path",
        default="configs/mnist.yaml",
        type=str,
    )
    _ = parser.add_argument(
        "--sample-all",
        default=False,
        action="store_true",
        help="Samples all stored models.",
    )
    parser.add_argument(
        "--num-samples",
        dest="num_samples",
        default=64,
        type=int,
    )
    _ = parser.add_argument(
        "--compile",
        default=False,
        action="store_true",
        help="Compile modules",
    )
    args = parser.parse_args()
    main(args)
