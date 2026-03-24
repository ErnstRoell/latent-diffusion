import torch
import pytest
from models.blocks.attention import AttentionConfig, AttentionBlock
from structlog import get_logger

logger = get_logger()

good_configs = [
    AttentionConfig(
        in_channels=64,
        t_emb_dim=128,
        norm_channels=2,
        num_heads=8,
    ),
    AttentionConfig(
        in_channels=64,
        t_emb_dim=128,
        norm_channels=2,
        num_heads=8,
    ),
    AttentionConfig(
        in_channels=64,
        t_emb_dim=128,
        norm_channels=2,
        num_heads=8,
    ),
    AttentionConfig(
        in_channels=64,
        t_emb_dim=128,
        norm_channels=2,
        num_heads=8,
    ),
    AttentionConfig(
        in_channels=64,
        t_emb_dim=128,
        norm_channels=2,
        num_heads=8,
    ),
    AttentionConfig(
        in_channels=128,
        t_emb_dim=64,
        norm_channels=4,
        num_heads=8,
    ),
    AttentionConfig(
        in_channels=128,
        t_emb_dim=128,
        norm_channels=4,
        num_heads=8,
    ),
]


@pytest.mark.parametrize("config", good_configs)
def test_init(config: AttentionConfig):

    block = AttentionBlock(config=config)

    img = torch.empty(10, config.in_channels, 28, 28)
    out = block(img)

    assert out.shape == (10, config.in_channels, 28, 28)


@pytest.mark.parametrize("config", good_configs)
def test_group_norm_not_equal(config: AttentionConfig):

    # Will never work.
    config.norm_channels = 13

    logger.info("PyTest Config:", config=config)

    with pytest.raises(AssertionError) as exc_info:
        block = AttentionBlock(config=config)  # type: ignore


@pytest.mark.parametrize("config", good_configs)
def test_attention_shape_not_equal(config: AttentionConfig):
    config.num_heads = 13

    with pytest.raises(AssertionError) as exc_info:
        block = AttentionBlock(config=config)


@pytest.mark.parametrize("config", good_configs)
def test_t_emb_not_equal(config: AttentionConfig):
    config.t_emb_dim = 13

    with pytest.raises(AssertionError) as exc_info:
        block = AttentionBlock(config=config)
