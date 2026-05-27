"""Maps image to latent space."""

import pathlib
from types import SimpleNamespace
from models.vae import VAE
import torch
import yaml
import importlib
import pydantic
from torch import nn
import structlog
from lightning.fabric import Fabric

logger = structlog.get_logger()


class LatentConfig(pydantic.BaseModel):
    """Contains the path of the config used to train the VAE"""

    module: str
    vae_config: str


def load_config(path: str):
    """
    Loads the configuration yaml and parses it into an object with dot access.
    """
    with open(path, encoding="utf-8") as stream:
        # Load dict
        config_dict = yaml.safe_load(stream)

    loaded_config_dict = {}
    for key, val in config_dict.items():
        print(key)
        if "module" in config_dict[key].keys():
            if val["module"] is not None:
                module = importlib.import_module(config_dict[key]["module"])
                configs = [c for c in dir(module) if c.endswith("Config")]
                if len(configs) > 1:
                    print(f"Found multiple configs in {val['module']}")
                else:
                    config_cls = getattr(module, configs[0])
                    loaded_config_dict[key] = config_cls(**val)
            else:
                loaded_config_dict[key] = SimpleNamespace(**val)

        else:
            print(f"Module not found in {key}")

    return SimpleNamespace(**loaded_config_dict)



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


def setup():
    return LatentConfig, ToLatent
