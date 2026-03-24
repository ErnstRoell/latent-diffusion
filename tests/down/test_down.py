import torch
import pytest
from models.down import DownConfig, DownBlock

good_configs = [
    DownConfig(
        in_channels=64,
        out_channels=128,
        down_sample=True,
        t_emb_dim=128,
        num_heads=32,
        num_layers=6,
        attn=True,
        norm_channels=2,
        bias=True,
    ),
    DownConfig(
        in_channels=64,
        out_channels=128,
        down_sample=True,
        t_emb_dim=128,
        num_heads=32,
        num_layers=2,
        attn=False,
        norm_channels=2,
        bias=True,
    ),
    DownConfig(
        in_channels=64,
        out_channels=128,
        down_sample=False,
        t_emb_dim=128,
        num_heads=32,
        num_layers=2,
        attn=True,
        norm_channels=2,
        bias=True,
    ),
    DownConfig(
        in_channels=64,
        out_channels=128,
        down_sample=False,
        t_emb_dim=128,
        num_heads=32,
        bias=True,
        num_layers=2,
        attn=False,
        norm_channels=2,
    ),
]


@pytest.mark.parametrize("config", good_configs)
def test_init(config: DownConfig):

    block = DownBlock(config=config)

    img = torch.empty(10, 64, 28, 28)
    t_emb = torch.empty(10, 128)
    out = block(img, t_emb)

    if config.down_sample:
        assert out.shape == (10, 128, 14, 14)
    elif not config.down_sample:
        assert out.shape == (10, 128, 28, 28)


def test_attention_shape_not_equal():
    config = DownConfig(
        in_channels=64,
        out_channels=128,
        down_sample=True,
        t_emb_dim=128,
        num_heads=13,
        num_layers=6,
        attn=True,
        norm_channels=2,
        bias=True,
    )

    with pytest.raises(AssertionError) as exc_info:
        block = DownBlock(config=config)


def test_group_norm_not_equal():
    config = DownConfig(
        in_channels=64,
        out_channels=128,
        down_sample=True,
        t_emb_dim=128,
        num_heads=32,
        num_layers=6,
        attn=True,
        norm_channels=13,
        bias=True,
    )

    with pytest.raises(AssertionError) as exc_info:
        block = DownBlock(config=config)  # type: ignore
