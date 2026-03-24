"""
Helpers to load a class from a configuration file.
Sectioned into models, datasets, loggers.
"""

from dataclasses import dataclass

import importlib
import json
from types import SimpleNamespace
from typing import Any
import pydantic

import yaml

#######################################################################
### Configuration
#######################################################################


#######################################################################
### Configuration
#######################################################################


# @timeit_decorator
def load_object(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**obj)
    else:
        return obj


def load_config_pydantic(path: str):
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


def load_config(path: str):
    """
    Loads the configuration yaml and parses it into an object with dot access.
    """
    with open(path, encoding="utf-8") as stream:
        # Load dict
        config_dict = yaml.safe_load(stream)

        # Convert to namespace (access via config.data etc)
        config = json.loads(json.dumps(config_dict), object_hook=load_object)
    return config


@dataclass
class Config:
    dataset: Any
    model: Any
    scheduler: Any
    trainer: Any
    meta: Any


def save_config(config, path: str):
    """
    Save the configuration yaml.
    """
    print(f"Saving config to {path}")
    with open(path, "w", encoding="utf-8") as stream:
        # Load dict
        yaml.dump(
            json.loads(json.dumps(config, default=lambda s: vars(s))),
            default_flow_style=False,
            stream=stream,
        )


def config_to_dict(config):
    """
    Converts nested namespace to nested dictionary.
    Needed for printing."""
    return json.loads(
        json.dumps(config, default=lambda s: vars(s)),
    )


def print_config(config):
    print(
        yaml.dump(
            config_to_dict(config),
            default_flow_style=False,
        )
    )


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
### Models
#######################################################################


# def load_datamodule(config):
#     # Validation
#     if not hasattr(config, "module"):
#         raise ValueError("Path to the module is missing.")
#     module = importlib.import_module(config.module)
#     config_class = getattr(module, "DataConfig")
#     datamodule_class = getattr(module, "DataModule")
#     loaded_config = config_class(**config_to_dict(config))
#     datamodule = datamodule_class(loaded_config)
#     return datamodule


def load_model(config):
    # Validation
    if not hasattr(config, "module"):
        raise ValueError("Path to the module is missing.")
    module = importlib.import_module(config.module)
    model_class = getattr(module, "Model")
    return model_class(config)


def load_model_config(config):
    # Validation
    if not hasattr(config, "module"):
        raise ValueError("Path to the module is missing.")
    module = importlib.import_module(config.module)
    model_config_class = getattr(module, "ModelConfig")
    return model_config_class


#######################################################################
### Loggers
#######################################################################


def load_logger(config, log_path: str):
    # Validation
    module = importlib.import_module("loggers")
    return module.get_logger(config, log_path)


def load_logger_config(_):
    # Validation
    module = importlib.import_module("loggers")
    logger_config_class = getattr(module, "LogConfig")
    return logger_config_class
