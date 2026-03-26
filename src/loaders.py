"""
Helpers to load a class from a configuration file.
Sectioned into models, datasets, loggers.
"""

import importlib
from types import SimpleNamespace

import yaml
from torch import nn


class Compose(nn.Module):
    """Custom compose class, compatible with fabric."""

    def __init__(self, transforms):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


#######################################################################
### Configuration
#######################################################################


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


#######################################################################
### Datasets
#######################################################################


def load_datamodule(config):
    # Validation
    if not hasattr(config, "module"):
        raise ValueError("Path to the module is missing.")
    module = importlib.import_module(config.module)
    return module


#######################################################################
### Transforms
#######################################################################


def load_latent(config):
    module = importlib.import_module(config.module)
    return module.ToLatent(config)


def load_transforms(config):
    # Validation

    tr_list = []
    for tr_config in config:
        module = importlib.import_module(tr_config["module"])
        cfg = module.TransformConfig(**tr_config)
        tr_list.append(module.Transform(cfg))

    return Compose(tr_list)
