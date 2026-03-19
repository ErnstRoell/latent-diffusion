import torch
from models.unet import ModelConfig, Unet


import pytest

good_configs = [
    ModelConfig(
        module="",
        im_channels=1,
        down_channels=[64, 128, 256],
        mid_channels=[256, 256],
        down_sample=[False, True],
        attn_down=[False, True],
        attn_up=[False, False],
        time_emb_dim=128,
        norm_channels=16,
        num_heads=16,
        conv_out_channels=128,
        num_down_layers=2,
        num_mid_layers=2,
        num_up_layers=2,
    ),
    ModelConfig(
        module="",
        im_channels=1,
        down_channels=[16, 64, 64],
        mid_channels=[64, 64],
        down_sample=[True, True],
        attn_down=[True, True],
        attn_up=[True, True],
        time_emb_dim=32,
        norm_channels=8,
        num_heads=4,
        conv_out_channels=64,
        num_down_layers=2,
        num_mid_layers=2,
        num_up_layers=2,
    ),
]


@pytest.mark.parametrize("config", good_configs)
def test_unet_config(config):

    model = Unet(config)

    im = torch.zeros(size=(32, 1, 32, 32))
    t = torch.tensor([0.1])

    out = model(im, t)

    assert out.shape == im.shape
