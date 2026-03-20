import torch.nn as nn
from hooks.forward import forward_hook
from models.blocks.resnet import ResNetBlock, ResNetConfig
from models.blocks.attention import AttentionBlock, AttentionConfig
from dataclasses import dataclass, asdict

import structlog

logger = structlog.get_logger()


@dataclass
class MidConfig:
    in_channels: int
    out_channels: int
    t_emb_dim: int | None
    num_heads: int
    num_layers: int
    norm_channels: int
    attn: bool


class MidBlock(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.config = config
        logger.info(
            f"Config {self.__class__.__name__}", type="config", **asdict(config)
        )

        # Either attention blocks or identity.
        self.resnet_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        self.emb_blocks = nn.ModuleList()
        self.first_resnet_block = ResNetBlock(
            ResNetConfig(
                in_channels=self.config.in_channels,
                out_channels=self.config.out_channels,
                t_emb_dim=self.config.t_emb_dim,
                down_sample=False,
                norm_channels=self.config.norm_channels,
                normtype="group",
            )
        )

        for _ in range(self.config.num_layers):
            ###################
            #  ResNet Blocks  #
            ###################
            self.resnet_blocks.append(
                ResNetBlock(
                    ResNetConfig(
                        in_channels=self.config.out_channels,
                        out_channels=self.config.out_channels,
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
                            in_channels=self.config.out_channels,  # Out channels from resnet.
                            t_emb_dim=self.config.t_emb_dim,
                            norm_channels=self.config.norm_channels,
                            num_heads=self.config.num_heads,
                            normtype="group",
                        )
                    )
                )
            else:
                self.attn_blocks.append(nn.Identity())

        self.register_forward_hook(forward_hook)

    def forward(self, x, t_emb=None, context=None):
        out = x

        out = self.first_resnet_block(out, t_emb)

        for resnet, attn in zip(self.resnet_blocks, self.attn_blocks):
            # Resnet block of Unet
            out = resnet(out, t_emb)

            # Attention block of Unet
            out = attn(out)

        return out
