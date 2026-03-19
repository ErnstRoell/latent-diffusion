import torch
import pytest
from models.blocks.resnet import ResNetConfig, ResNetBlock
from structlog import get_logger

logger = get_logger()

good_configs = [
    ResNetConfig(
        in_channels=64,
        out_channels=128,
        down_sample=True,
        t_emb_dim=128,
        norm_channels=2,
        normtype="group",
    ),
    ResNetConfig(
        in_channels=64,
        out_channels=128,
        down_sample=True,
        t_emb_dim=128,
        norm_channels=2,
        normtype="group",
    ),
    ResNetConfig(
        in_channels=64,
        out_channels=128,
        down_sample=False,
        t_emb_dim=128,
        norm_channels=2,
        normtype="group",
    ),
    ResNetConfig(
        in_channels=64,
        out_channels=128,
        down_sample=False,
        t_emb_dim=128,
        norm_channels=2,
        normtype="group",
    ),
    ResNetConfig(
        in_channels=64,
        out_channels=64,
        down_sample=False,
        t_emb_dim=128,
        norm_channels=2,
        normtype="group",
    ),
    ResNetConfig(
        in_channels=128,
        out_channels=64,
        down_sample=True,
        t_emb_dim=64,
        norm_channels=4,
        normtype="group",
    ),
    ResNetConfig(
        in_channels=128,
        out_channels=64,
        down_sample=True,
        t_emb_dim=1234,
        norm_channels=4,
        normtype="group",
    ),
]


@pytest.mark.parametrize("config", good_configs)
def test_init(config: ResNetConfig):

    block = ResNetBlock(config=config)

    img = torch.empty(10, config.in_channels, 28, 28)
    t_emb = torch.empty(10, config.t_emb_dim)
    out = block(img, t_emb)

    if config.down_sample:
        assert out.shape == (10, config.out_channels, 14, 14)
    elif not config.down_sample:
        assert out.shape == (10, config.out_channels, 28, 28)


@pytest.mark.parametrize("config", good_configs)
def test_no_time_emb(config: ResNetConfig):

    block = ResNetBlock(config=config)

    img = torch.empty(10, config.in_channels, 28, 28)
    out = block(img)

    if config.down_sample:
        assert out.shape == (10, config.out_channels, 14, 14)
    elif not config.down_sample:
        assert out.shape == (10, config.out_channels, 28, 28)


@pytest.mark.parametrize("config", good_configs)
def test_no_time_emb_at_all(config: ResNetConfig):

    config.t_emb_dim = None

    block = ResNetBlock(config=config)

    img = torch.empty(10, config.in_channels, 28, 28)
    out = block(img)

    if config.down_sample:
        assert out.shape == (10, config.out_channels, 14, 14)
    elif not config.down_sample:
        assert out.shape == (10, config.out_channels, 28, 28)


@pytest.mark.parametrize("config", good_configs)
def test_group_norm_not_equal(config: ResNetConfig):

    # Will never work.
    config.norm_channels = 13

    logger.info("PyTest Config:", config=config)

    with pytest.raises(AssertionError) as exc_info:
        block = ResNetBlock(config=config)  # type: ignore
