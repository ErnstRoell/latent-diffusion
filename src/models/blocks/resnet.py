import torch.nn as nn
import torch

from typing import Union
from dataclasses import dataclass, asdict
from hooks.forward import forward_hook

import structlog

logger = structlog.get_logger()


@dataclass
class ResNetConfig:
    in_channels: int
    out_channels: int
    t_emb_dim: int | None
    down_sample: bool
    norm_channels: int
    bias: bool


class ResNetBlock(nn.Module):
    """
    ResNetBlock with time embedding.

    """

    def __init__(self, config: ResNetConfig):
        super().__init__()

        self.config = config

        assert config.in_channels % config.norm_channels == 0, ValueError(
            f"Expected input channels / number of groups ({config.in_channels} / {config.norm_channels}) to be an integer."
        )

        logger.debug(
            f"Config {self.__class__.__name__}", type="config", **asdict(config)
        )

        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(
                config.norm_channels,
                config.in_channels,
            ),
            nn.SiLU(),
            nn.Conv2d(
                config.in_channels,
                config.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=config.bias,
            ),
        )

        if self.config.t_emb_dim is not None:
            self.t_emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    self.config.t_emb_dim,
                    config.out_channels,
                ),
            )

        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(
                config.norm_channels,
                config.out_channels,
            ),
            nn.SiLU(),
            nn.Conv2d(
                config.out_channels,
                config.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=config.bias,
            ),
        )

        self.residual_input_conv = nn.Conv2d(
            config.in_channels, config.out_channels, kernel_size=1, bias=config.bias
        )

        if self.config.down_sample:
            self.down_sample_conv = nn.Conv2d(
                config.out_channels,
                config.out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=config.bias,
            )
        else:
            self.down_sample_conv = nn.Identity()

        self.register_forward_hook(forward_hook)
        # self.hooks = {}
        # for name, module in self.named_modules():
        #     self.hooks[name] = module.register_forward_hook(forward_hook)

    def forward(self, x, t_emb=None):
        # Resnet block of Unet
        out = x
        resnet_input = x

        # First conv layer
        out = self.resnet_conv_first(out)

        # Time embedding.
        if t_emb is not None:
            out = out + self.t_emb_layers(t_emb)[:, :, None, None]

        out = self.resnet_conv_second(out)

        # Add the skip connection
        out = out + self.residual_input_conv(resnet_input)

        # Downsample
        out = self.down_sample_conv(out)
        return out


if __name__ == "__main__":
    import torch

    config = ResNetConfig(
        in_channels=64,
        out_channels=64,
        t_emb_dim=128,
        down_sample=False,
        norm_channels=32,
        bias=True,
    )

    block = ResNetBlock(config=config)

    img = torch.zeros(10, 64, 28, 28)
    t_emb = torch.ones(10, 128)
    out = block(img, t_emb)
    out2 = block(out, t_emb)
    logger.info("Out", shape=out.shape)

    # from torchinfo import summary
    # print(summary(block))
