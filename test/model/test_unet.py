import torch
from models.unet import ModelConfig, Unet

from models.down import DownConfig
from models.mid import MidConfig

import pytest

good_configs = [
    ModelConfig(
        module="",
        im_channels=1,
        time_emb_dim=128,  # Should be the same everywhere.
        down_blocks=[
            DownConfig(
                in_channels=64,
                out_channels=128,
                t_emb_dim=128,
                down_sample=False,
                num_heads=16,
                num_layers=2,
                attn=False,
                norm_channels=16,
                normtype="group",
            ),
            DownConfig(
                in_channels=128,
                out_channels=256,
                t_emb_dim=128,
                down_sample=True,
                num_heads=16,
                num_layers=2,
                attn=True,
                norm_channels=16,
                normtype="group",
            ),
        ],
        mid_blocks=[
            MidConfig(
                in_channels=256,
                out_channels=256,
                t_emb_dim=128,
                num_heads=16,
                num_layers=2,
                attn=True,
                norm_channels=16,
            ),
        ],
    ),
    ModelConfig(
        module="",
        im_channels=1,
        time_emb_dim=128,  # Should be the same everywhere.
        down_blocks=[
            DownConfig(
                in_channels=64,
                out_channels=128,
                t_emb_dim=128,
                down_sample=False,
                num_heads=16,
                num_layers=2,
                attn=False,
                norm_channels=16,
                normtype="group",
            ),
            DownConfig(
                in_channels=128,
                out_channels=256,
                t_emb_dim=128,
                down_sample=True,
                num_heads=16,
                num_layers=2,
                attn=True,
                norm_channels=16,
                normtype="group",
            ),
        ],
        mid_blocks=[
            MidConfig(
                in_channels=256,
                out_channels=256,
                t_emb_dim=128,
                num_heads=16,
                num_layers=2,
                attn=True,
                norm_channels=16,
            ),
        ],
    ),
]


@pytest.mark.parametrize("config", good_configs)
def test_unet_config(config):

    model = Unet(config)

    im = torch.zeros(size=(32, 1, 32, 32))
    t = torch.tensor([0.1])

    out = model(im, t)

    assert out.shape == im.shape
