"""
Helpers to load a class from a configuration file.
Sectioned into models, datasets, loggers.
"""


import importlib
from types import SimpleNamespace 

def load_module(config: dict[str,str]):
    # Validation
    module = importlib.import_module(config["module"])
    if not hasattr(module,"setup"):
        raise ValueError(f"Trying to load {config['module']}, but the setup function is missing.")
    config, cls = module.setup()
    return config, cls


def load_context(full_config_dict):
    """Loads the context from the configuration."""
    ctx_dict = {} 
    for key, config_dict in full_config_dict.items():
        if "module" not in config_dict.keys():
            raise ValueError(f"Trying to load {key}, but the 'module' key is missing.")

        if config_dict["module"] is not None:
            config_cls, module_cls = load_module(config_dict)
            config_instance = config_cls(**config_dict)
            ctx_dict[key] = module_cls(config_instance)
        else: 
            ctx_dict[key] = SimpleNamespace(**config_dict)
    return SimpleNamespace(**ctx_dict)


def load_configs(config_dict):
    config_inst_dict = {} 
    for key, config in config_dict.items():
        c, _ = load_module(config)
        config_inst_dict[key] = c(**config)
    return SimpleNamespace(**config_inst_dict)
