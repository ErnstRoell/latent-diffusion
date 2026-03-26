import torch
import pytest
import glob
from loaders import load_config
import structlog

logger = structlog.get_logger()


def all_configs():
    files = glob.glob("configs/**/**.yaml")

    # file = "configs/scratch/mnist_unet_test.yaml"
    # files = [file]
    return files


@pytest.mark.parametrize("config_path", all_configs())
def test_load_configs(config_path):
    logger.info(
        "config",
        config=config_path,
    )

    config = load_config(config_path)
    logger.info("Config", config=config)
