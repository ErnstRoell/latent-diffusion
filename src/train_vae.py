import argparse
from torchvision.utils import make_grid
import torchvision
import os
import pathlib

import numpy as np
import torch
from lightning.fabric import Fabric

from structlog import get_logger
from loggers import setup_loggers

from loaders import load_config, load_datamodule

from models.vae import VAE
from models.discriminator import Discriminator
from torch.optim import Adam
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = get_logger()

torch.set_float32_matmul_precision("medium")


def train(args, config):
    # Set up Fabric
    fabric = Fabric(
        accelerator="cuda",
        precision="16-mixed",
    )

    # Parse the args
    config_path = args.config_path
    dev: bool = args.dev
    compile: bool = args.compile
    resume: bool = args.resume

    #################################
    #  Set up directory structures  #
    #################################

    # Create paths for the results
    result_folder_list = ["results"] + list(pathlib.Path(config_path).parent.parts[1:])
    result_folder = pathlib.Path(*result_folder_list)

    if not result_folder.exists():
        os.makedirs(result_folder, exist_ok=True)

    # Dataloaders
    dm = load_datamodule(config.dataset)
    train_loader, test_loader = dm.get_dataloaders(config.dataset, dev=dev)
    train_loader = fabric.setup_dataloaders(train_loader)
    test_loader = fabric.setup_dataloaders(test_loader)

    # Instantiate the model
    model = VAE(config.model)
    epoch = 0

    # If resume training, load the model.
    if resume:
        state = {"model": model, "epoch": epoch}
        fabric.load(result_folder / f"{config.meta.modelname}_9999.ckpt", state)
        epoch = state["epoch"]
        model.train()
        model.to(fabric.device)

    if compile:
        model = torch.compile(model)

    optimizer = Adam(model.parameters(), lr=config.trainer.lr)
    criterion = torch.nn.MSELoss()
    model, optimizer = fabric.setup(model, optimizer)  # type: ignore

    ##########################
    #  Discriminator setup.  #
    ##########################

    disc_criterion = torch.nn.MSELoss()
    discriminator = Discriminator()
    optimizer_d = Adam(
        discriminator.parameters(),
        lr=config.trainer.lr,
        betas=(0.5, 0.999),
    )

    discriminator, optimizer_d = fabric.setup(discriminator, optimizer_d)  # type: ignore

    ###############################################
    ###############################################

    logger.info(f"Start training from epoch {epoch} ...")
    for epoch_idx in range(epoch, epoch + config.trainer.num_epochs):
        model.train()
        g_losses = []
        d_losses = []
        for step, (im,) in enumerate(tqdm(train_loader)):
            # Accumulate gradient 8 batches at a time
            # is_accumulating = step % 8 != 0
            # is_accumulating = False

            # Hardcoded for now.
            # is_discriminator_training = epoch_idx > 1
            # is_discriminator_training = True

            # with fabric.no_backward_sync(model, enabled=is_accumulating):

            ##################
            #  Optimize VAE  #
            ##################

            recon, _ = model(im)
            recon_loss = criterion(recon, im)

            # optimizer_d.zero_grad()
            disc_fake_pred = discriminator(recon)
            disc_fake_loss = disc_criterion(
                disc_fake_pred,
                torch.ones(disc_fake_pred.shape, device=disc_fake_pred.device),
            )

            generator_loss = recon_loss + disc_fake_loss
            g_losses.append(generator_loss.item())

            fabric.backward(generator_loss)

            ############################
            #  Optimize Discriminator  #
            ############################

            disc_fake_pred = discriminator(recon.detach())
            disc_fake_loss = disc_criterion(
                disc_fake_pred,
                torch.ones_like(disc_fake_pred, device=fabric.device),
            )

            disc_real_pred = discriminator(im)
            disc_real_loss = disc_criterion(
                disc_real_pred,
                torch.ones_like(disc_real_pred, device=fabric.device),
            )
            disc_loss = disc_real_loss + disc_fake_loss
            d_losses.append(disc_loss.item())

            fabric.backward(disc_loss)

            optimizer.step()
            optimizer.zero_grad()
            optimizer_d.step()
            optimizer_d.zero_grad()

            logger.debug(
                "Training",
                loss=generator_loss.item(),
                epoch=epoch_idx,
                type="metric",
                split="train",
                model=config.meta.modelname,
            )
            logger.debug(
                "Training",
                loss=disc_loss.item(),
                epoch=epoch_idx,
                type="metric",
                split="train",
                model="Discriminator",
            )
        logger.info(
            "Training epoch:{} | G Loss : {:.4f} | D Loss: {:.4f} ".format(
                epoch_idx, np.mean(g_losses), np.mean(d_losses)
            )
        )

        if epoch_idx % config.trainer.validation_interval == 0:
            model.eval()
            g_losses_val = []
            d_losses_val = []
            with torch.no_grad():
                for step, (im,) in enumerate(tqdm(test_loader)):
                    recon, _ = model(im)

                    recon_loss = criterion(recon, im)

                    disc_fake_pred = discriminator(recon.detach())
                    disc_fake_loss = disc_criterion(
                        disc_fake_pred,
                        torch.ones_like(disc_fake_pred, device=fabric.device),
                    )
                    generator_loss = recon_loss + disc_fake_loss
                    g_losses_val.append(generator_loss.item())

                    logger.debug(
                        "Validation",
                        loss=generator_loss.item(),
                        epoch=epoch_idx,
                        type="metric",
                        split="val",
                        model=config.meta.modelname,
                    )

                    disc_real_pred = discriminator(im)
                    disc_real_loss = disc_criterion(
                        disc_real_pred,
                        torch.ones_like(disc_real_pred, device=fabric.device),
                    )
                    disc_loss = disc_real_loss + disc_fake_loss
                    d_losses_val.append(disc_loss.item())

                    logger.debug(
                        "Validation",
                        loss=disc_loss.item(),
                        epoch=epoch_idx,
                        type="metric",
                        split="val",
                        model="Discriminator",
                    )

                    if step == 0:
                        # Save as imgs
                        recon = (1 + torch.clamp(recon, -1.0, 1.0).detach().cpu()) / 2
                        im = (1 + im) / 2

                        ims = torch.cat(
                            [recon[:32, :, :, :], im[:32, :, :, :].cpu()], dim=0
                        )
                        grid = make_grid(ims, nrow=8)
                        img = torchvision.transforms.ToPILImage()(grid[:, :, :])
                        img.save(
                            result_folder
                            / f"{config.meta.modelname}_recon_{epoch_idx:04}.png"
                        )
                        img.close()

                logger.info(
                    "Validation epoch:{} | G Loss : {:.4f} | D Loss : {:.4f}".format(
                        epoch_idx, np.mean(g_losses_val), np.mean(d_losses).item()
                    ),
                )

        if epoch_idx % config.trainer.checkpoint_interval == 0:
            state = {"model": model, "epoch": epoch_idx + 1}
            fabric.save(
                result_folder / f"{config.meta.modelname}_{epoch_idx:04}.ckpt", state
            )

    logger.info("Done Training ...")
    state = {"model": model, "epoch": epoch_idx + 1}
    logger.info(f"Epoch: {epoch_idx}")
    fabric.save(result_folder / f"{config.meta.modelname}_9999.ckpt", state)


def main():
    parser = argparse.ArgumentParser(description="Arguments for ddpm training")
    _ = parser.add_argument(
        "--config",
        dest="config_path",
        default="configs/scratch/mnist_vae.yaml",
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

    # Create paths for the results
    result_folder_list = ["results"] + list(
        pathlib.Path(args.config_path).parent.parts[1:]
    )
    result_folder = pathlib.Path(*result_folder_list)

    if not result_folder.exists():
        os.makedirs(result_folder, exist_ok=True)

    ###################
    #  Parse configs  #
    ###################

    config = load_config(args.config_path)
    setup_loggers(
        str(result_folder),
        name=config.meta.modelname,
        remove_logs=True,
    )

    train(args, config)


if __name__ == "__main__":
    main()
