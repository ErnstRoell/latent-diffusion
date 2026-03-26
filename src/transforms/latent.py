"""Maps image to latent space."""

import pathlib
from models.vae import VAE
import torch
import pydantic
from loaders import load_config
from torch import nn
import structlog
from lightning.fabric import Fabric

logger = structlog.get_logger()


class LatentConfig(pydantic.BaseModel):
    """Contains the path of the config used to train the VAE"""

    module: str
    vae_config: str


class ToLatent(nn.Module):

    def __init__(self, config: LatentConfig):
        super().__init__()
        vae_config = load_config(config.vae_config)
        self.model = VAE(vae_config.model)

        # Make checkpoint path
        result_folder_list = ["results"] + list(
            pathlib.Path(config.vae_config).parent.parts[1:]
        )
        result_folder = pathlib.Path(*result_folder_list)
        checkpoint = f"{result_folder}/{vae_config.meta.modelname}_9999.ckpt"

        logger.debug(f"Loading model from {checkpoint}")

        state = {"model": self.model}
        Fabric().load(checkpoint, state)

    @torch.no_grad()
    def __call__(self, img):
        z, _ = self.model.encode(img)
        return z

    @torch.no_grad()
    def decode(self, latents):
        return self.model.decode(latents)
