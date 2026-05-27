import argparse
import os
import pathlib
from torchmetrics.classification import MulticlassAccuracy

import numpy as np
import torch
from lightning.fabric import Fabric

from structlog import get_logger
from loggers import setup_loggers

from loaders import load_config, load_datamodule

from torch.optim import Adam
from tqdm import tqdm

from models.vit import VisionTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = get_logger()

torch.set_float32_matmul_precision("medium")

def train(args, config):
    # Set up Fabric
    fabric = Fabric(
        accelerator="cuda",
        # precision="16-mixed",
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
    model = VisionTransformer(config.model)
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
    criterion = torch.nn.CrossEntropyLoss()
    model, optimizer = fabric.setup(model, optimizer)  # type: ignore

    ###############################################
    ###############################################

    logger.info(f"Start training from epoch {epoch} ...")
    for epoch_idx in range(epoch, epoch + config.trainer.num_epochs):
        model.train()
        losses = []
        for step, (im,y) in enumerate(tqdm(train_loader)):

            ##################
            #  Optimize VAE  #
            ##################


            optimizer.zero_grad()
            logits = model(im)

            loss = criterion(logits, y.squeeze().to(torch.long))

            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())


            logger.debug(
                "Training",
                loss=loss.item(),
                epoch=epoch_idx,
                type="metric",
                split="train",
                model=config.meta.modelname,
            )
        logger.info(
            "Training epoch:{} | Loss : {:.4f}".format(
                epoch_idx, np.mean(losses)
            )
        )

        if epoch_idx % config.trainer.validation_interval == 0:
            acc = MulticlassAccuracy(num_classes=10).cuda()
            model.eval()
            losses_val = []
            with torch.no_grad():
                for step, (im,y) in enumerate(tqdm(test_loader)):

                    logits = model(im)

                    acc.update(torch.softmax(logits,dim=-1),y.squeeze().to(torch.long).cuda())


                    loss = criterion(logits, y.squeeze().to(torch.long))
                    losses_val.append(loss.item())
                    logger.debug(
                        "Validation",
                        loss=loss.item(),
                        epoch=epoch_idx,
                        type="metric",
                        split="val",
                        model=config.meta.modelname,
                    )

                logger.info(
                        "Validation epoch:{} | Loss : {:.4f} | Acc: {:.4f}".format(
                        epoch_idx, np.mean(losses_val), acc.compute()
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
    parser = argparse.ArgumentParser(description="Arguments for classifier training")
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

    if args.resume:
        remove_logs = False
    else:
        remove_logs = True

    config = load_config(args.config_path)
    setup_loggers(
        str(result_folder),
        name=config.meta.modelname,
        remove_logs=remove_logs,
    )

    train(args, config)


if __name__ == "__main__":
    main()
