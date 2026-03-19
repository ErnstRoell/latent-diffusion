import torch
import pytest
from models.up import UpConfig, UpBlock

good_configs = [
    UpConfig(
        in_channels=64,
        out_channels=128,
        up_sample=True,
        t_emb_dim=128,
        num_heads=32,
        num_layers=2,
        attn=True,
        norm_channels=2,
    ),
    UpConfig(
        in_channels=64,
        out_channels=128,
        up_sample=True,
        t_emb_dim=128,
        num_heads=32,
        num_layers=2,
        attn=False,
        norm_channels=2,
    ),
    UpConfig(
        in_channels=64,
        out_channels=128,
        up_sample=False,
        t_emb_dim=128,
        num_heads=32,
        num_layers=2,
        attn=True,
        norm_channels=2,
    ),
    UpConfig(
        in_channels=64,
        out_channels=128,
        up_sample=False,
        t_emb_dim=128,
        num_heads=32,
        num_layers=2,
        attn=False,
        norm_channels=2,
    ),
]


@pytest.mark.parametrize("config", good_configs)
def test_init(config: UpConfig):

    block = UpBlock(config=config)

    img = torch.empty(10, 32, 14, 14)
    down_out = torch.empty(10, 32, 14, 14)
    t_emb = torch.empty(10, 128)
    out = block(img, down_out, t_emb)

    if config.up_sample:
        assert out.shape == (10, 128, 28, 28)
    elif not config.up_sample:
        assert out.shape == (10, 128, 14, 14)


def test_attention_shape_not_equal():
    config = UpConfig(
        in_channels=64,
        out_channels=128,
        up_sample=True,
        t_emb_dim=128,
        num_heads=13,
        num_layers=6,
        attn=True,
        norm_channels=2,
    )

    with pytest.raises(AssertionError) as exc_info:
        block = UpBlock(config=config)


def test_group_norm_not_equal():
    config = UpConfig(
        in_channels=64,
        out_channels=128,
        up_sample=True,
        t_emb_dim=128,
        num_heads=32,
        num_layers=6,
        attn=True,
        norm_channels=13,
    )

    with pytest.raises(AssertionError) as exc_info:
        block = UpBlock(config=config)  # type: ignore
