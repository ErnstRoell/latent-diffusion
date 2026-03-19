import torch.nn as nn
import torch

from typing import Union
from dataclasses import dataclass, asdict

import structlog
from hooks.forward import forward_hook

logger = structlog.get_logger()


@dataclass
class DownConfig:
    in_channels: int = 64
    out_channels: int = 128
    t_emb_dim: int = 128
    down_sample: bool = True
    num_heads: int = 16
    num_layers: int = 2
    attn: bool = True
    norm_channels: int = 32
    normtype: str = "group"


def get_normlayer(
    norm_channels,
    in_channels,
    normtype="group",
) -> Union[nn.GroupNorm, nn.BatchNorm2d]:
    if normtype == "group":
        return nn.GroupNorm(norm_channels, in_channels)
    else:
        return nn.BatchNorm2d(in_channels)


class DownBlock(nn.Module):
    r"""
    Down conv block with attention.
    Sequence of following block
    1. Resnet block with time embedding
    2. Attention block
    3. Downsample
    """

    def __init__(self, config):
        super().__init__()

        assert config.t_emb_dim % config.num_heads == 0, ValueError(
            f"Expected embedding dimension / number of heads ({config.t_emb_dim} / {config.num_heads}) to be an integer."
        )

        assert config.in_channels % config.norm_channels == 0, ValueError(
            f"Expected input channels / number of groups ({config.in_channels} / {config.norm_channels}) to be an integer."
        )

        self.config = config

        logger.info("DownConfig:", **asdict(self.config))

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    get_normlayer(
                        config.norm_channels,
                        config.in_channels if i == 0 else config.out_channels,
                        normtype=config.normtype,
                    ),
                    nn.SiLU(),
                    nn.Conv2d(
                        config.in_channels if i == 0 else config.out_channels,
                        config.out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(config.num_layers)
            ]
        )
        if self.config.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.SiLU(), nn.Linear(self.config.t_emb_dim, config.out_channels)
                    )
                    for _ in range(config.num_layers)
                ]
            )
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    get_normlayer(
                        config.norm_channels,
                        config.out_channels,
                        normtype=config.normtype,
                    ),
                    nn.SiLU(),
                    nn.Conv2d(
                        config.out_channels,
                        config.out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for _ in range(config.num_layers)
            ]
        )

        if self.config.attn:
            self.attention_norms = nn.ModuleList(
                [
                    get_normlayer(
                        config.norm_channels, config.out_channels, normtype="group"
                    )
                    for _ in range(config.num_layers)
                ]
            )

            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        config.out_channels, config.num_heads, batch_first=True
                    )
                    for _ in range(config.num_layers)
                ]
            )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    config.in_channels if i == 0 else config.out_channels,
                    config.out_channels,
                    kernel_size=1,
                )
                for i in range(config.num_layers)
            ]
        )

        if self.config.down_sample:
            self.down_sample_conv = nn.Conv2d(
                self.config.out_channels, self.config.out_channels, 4, 2, 1
            )
        else:
            self.down_sample_conv = nn.Identity()

        # if dev:
        #     self.register_forward_hook(forward_hook)
        # self.hooks = {}
        # for name, module in self.named_modules():
        #     self.hooks[name] = module.register_forward_hook(forward_hook)

    def forward(self, x, t_emb=None):
        out = x
        for i in range(self.config.num_layers):
            # Resnet block of Unet
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.config.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            if self.config.attn:
                # Attention block of Unet
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn

        # Downsample
        out = self.down_sample_conv(out)
        return out


if __name__ == "__main__":
    import torch

    config = DownConfig(
        in_channels=64,
        out_channels=128,
        t_emb_dim=128,
        down_sample=True,
        num_heads=16,
        num_layers=2,
        attn=True,
        norm_channels=32,
        normtype="group",
    )
    block_1 = DownConfig(
        in_channels=64,
        out_channels=128,
        t_emb_dim=128,
        down_sample=True,
        num_heads=8,
        num_layers=7,
        attn=True,
        norm_channels=64,
        normtype="group",
    )

    block = DownBlock(config=config)

    img = torch.zeros(10, 64, 28, 28)
    t_emb = torch.ones(10, 128)
    block(img, t_emb)
    # block_2(img, t_emb)

    # from torchinfo import summary
    # print(summary(block))
