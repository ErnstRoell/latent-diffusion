import torch.nn as nn
import torch

from configs import Configuration
from hooks.forward import forward_hook

import structlog
from models.blocks.resnet import ResNetBlock, ResNetConfig
from models.blocks.attention import AttentionBlock, AttentionConfig

logger = structlog.get_logger()


class DownConfig(Configuration):
    """
    in_channels: int = 64
    out_channels: int = 128
    t_emb_dim: int = 128
    down_sample: bool = True
    num_heads: int = 16
    num_layers: int = 2
    attn: bool = True
    norm_channels: int = 32
    """

    in_channels: int
    out_channels: int
    t_emb_dim: int | None
    down_sample: bool
    num_heads: int
    num_layers: int
    attn: bool
    norm_channels: int
    bias: bool


class DownBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        logger.info(
            f"Config {self.__class__.__name__}", type="config", **config.model_dump()
        )

        # Either attention blocks or identity.
        self.resnet_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        self.emb_blocks = nn.ModuleList()

        channel_pairs = []
        down_sample = []
        for i in range(self.config.num_layers):
            # First layer
            if i == 0:
                channel_pairs.append([config.in_channels, config.out_channels])
            else:
                channel_pairs.append([config.out_channels, config.out_channels])

            if self.config.down_sample and (i == self.config.num_layers - 1):
                down_sample.append(True)
            else:
                down_sample.append(False)

        logger.debug("ResNet channelpairs", channelpairs=channel_pairs)
        logger.debug("ResNet downsample", downsample=down_sample)

        for (in_channels, out_channels), down_sample_bool in zip(
            channel_pairs, down_sample
        ):

            ###################
            #  ResNet Blocks  #
            ###################
            self.resnet_blocks.append(
                ResNetBlock(
                    ResNetConfig(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        t_emb_dim=config.t_emb_dim,
                        down_sample=down_sample_bool,
                        norm_channels=config.norm_channels,
                        bias=config.bias,
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
                            t_emb_dim=config.t_emb_dim,
                            norm_channels=config.norm_channels,
                            num_heads=config.num_heads,
                        )
                    )
                )
            else:
                self.attn_blocks.append(nn.Identity())

        self.register_forward_hook(forward_hook)

    def forward(self, x, t_emb=None):
        out = x
        for resnet, attn in zip(self.resnet_blocks, self.attn_blocks):

            # Resnet block of Unet
            out = resnet(out, t_emb)

            # Attention block of Unet
            out = attn(out)

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
        bias=True,
    )

    block = DownBlock(config=config)

    img = torch.zeros(10, 64, 28, 28)
    t_emb = torch.ones(10, 128)
    block(img, t_emb)
    # block_2(img, t_emb)

    # from torchinfo import summary
    # print(summary(block))
