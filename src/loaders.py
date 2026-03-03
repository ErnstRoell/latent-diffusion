"""
Helpers to load a class from a configuration file.
Sectioned into models, datasets, loggers.
"""

import importlib
import json
from types import SimpleNamespace

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
    return importlib.import_module(config.module)


#######################################################################
### Models
#######################################################################


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
