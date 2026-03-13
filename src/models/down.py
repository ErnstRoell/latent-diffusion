import torch.nn as nn

from typing import Union
from dataclasses import dataclass, asdict

import structlog

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


def forward_hook(module, input, output):
    log = logger.bind(module=module.__class__.__name__)
    if isinstance(output, torch.Tensor):
        outs = output.shape
    elif isinstance(output, tuple):
        outs = [el.shape for el in output]
    log.info("Shape", in_shape=input[0].shape, out_shape=outs)


class DownBlock(nn.Module):
    r"""
    Down conv block with attention.
    Sequence of following block
    1. Resnet block with time embedding
    2. Attention block
    3. Downsample
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        t_emb_dim,
        down_sample,
        num_heads,
        num_layers,
        attn,
        norm_channels,
        normtype="group",
    ):
        super().__init__()

        config = DownConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            t_emb_dim=t_emb_dim,
            down_sample=down_sample,
            num_heads=num_heads,
            num_layers=num_layers,
            attn=attn,
            norm_channels=norm_channels,
            normtype=normtype,
        )
        self.config = config

        logger.info("Config:", **asdict(self.config))

        self.config.num_layers = config.num_layers
        self.config.down_sample = config.down_sample
        self.config.attn = config.attn
        self.config.t_emb_dim = config.t_emb_dim
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    get_normlayer(
                        norm_channels,
                        in_channels if i == 0 else config.out_channels,
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
                        norm_channels,
                        out_channels,
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
                    get_normlayer(norm_channels, out_channels, normtype="group")
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

        # self.register_forward_hook(forward_hook)
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

    block_1 = DownBlock(
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
    block_2 = DownBlock(
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

    img = torch.zeros(10, 64, 28, 28)
    t_emb = torch.ones(10, 128)
    block_1(img, t_emb)
    block_2(img, t_emb)

    # from torchinfo import summary
    # print(summary(block))
