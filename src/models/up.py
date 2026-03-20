import torch
import torch.nn as nn
from typing import Union
from hooks.forward import forward_hook
from dataclasses import dataclass, asdict
from models.blocks.resnet import ResNetBlock, ResNetConfig
from models.blocks.attention import AttentionBlock, AttentionConfig

import structlog

logger = structlog.get_logger()


@dataclass
class UpConfig:
    in_channels: int
    out_channels: int
    t_emb_dim: int | None
    up_sample: bool
    num_heads: int
    num_layers: int
    norm_channels: int
    attn: bool


class UpBlock(nn.Module):
    r"""
    Up conv block with attention.
    Sequence of following blocks
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block with time embedding
    3. Attention Block
    """

    def __init__(self, config: UpConfig):
        super().__init__()
        self.config = config
        logger.info(
            f"Config {self.__class__.__name__}", type="config", **asdict(config)
        )

        channel_pairs = []
        for i in range(self.config.num_layers):
            # First layer
            if i == 0:
                channel_pairs.append(
                    [self.config.in_channels, self.config.out_channels]
                )
            else:
                channel_pairs.append(
                    [self.config.out_channels, self.config.out_channels]
                )

        logger.debug("ResNet channelpairs", channelpairs=channel_pairs)

        # Either attention blocks or identity.
        self.resnet_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        self.emb_blocks = nn.ModuleList()

        for in_channels, out_channels in channel_pairs:

            ###################
            #  ResNet Blocks  #
            ###################
            self.resnet_blocks.append(
                ResNetBlock(
                    ResNetConfig(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        t_emb_dim=self.config.t_emb_dim,
                        down_sample=False,
                        norm_channels=self.config.norm_channels,
                        normtype="group",
                    )
                )
            )

            ####################
            #  Self Attention  #
            ####################
            if self.config.attn:
                self.attn_blocks.append(
                    AttentionBlock(
                        AttentionConfig(
                            in_channels=out_channels,  # Out channels from resnet.
                            t_emb_dim=self.config.t_emb_dim,
                            norm_channels=self.config.norm_channels,
                            num_heads=self.config.num_heads,
                            normtype="group",
                        )
                    )
                )
            else:
                self.attn_blocks.append(nn.Identity())

        self.up_sample_conv = (
            nn.ConvTranspose2d(
                self.config.in_channels,
                self.config.in_channels,
                3,
                2,
                1,
                output_padding=1,
            )
            if self.config.up_sample
            else nn.Identity()
        )

        self.register_forward_hook(forward_hook)

    def forward(self, x, out_down=None, t_emb=None, context=None):
        if out_down is not None:
            assert x.shape[1] == out_down.shape[1]
            x = torch.cat([x, out_down], dim=1)
        x = self.up_sample_conv(x)

        out = x

        for resnet, attn in zip(self.resnet_blocks, self.attn_blocks):
            # Resnet block of Unet
            out = resnet(out, t_emb)

            # Attention block of Unet
            out = attn(out)

        # logger.info("DONE")
        return out
