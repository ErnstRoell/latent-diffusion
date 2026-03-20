import torch.nn as nn
import torch

from dataclasses import dataclass, asdict

import structlog
from hooks.forward import forward_hook

logger = structlog.get_logger()


@dataclass
class AttentionConfig:
    in_channels: int
    t_emb_dim: int | None
    num_heads: int
    norm_channels: int
    normtype: str


class AttentionBlock(nn.Module):
    r"""
    Down conv block with attention.
    Sequence of following block
    1. Resnet block with time embedding
    2. Attention block
    3. Downsample
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()

        if config.t_emb_dim is not None:
            assert config.t_emb_dim % config.num_heads == 0, ValueError(
                f"Expected embedding dimension / number of heads ({config.t_emb_dim} / {config.num_heads}) to be an integer."
            )

        assert config.in_channels % config.norm_channels == 0, ValueError(
            f"Expected input channels / number of groups ({config.in_channels} / {config.norm_channels}) to be an integer."
        )

        self.config = config

        logger.debug(
            f"Config {self.__class__.__name__}", type="config", **asdict(config)
        )

        self.attention_norms = nn.GroupNorm(
            config.norm_channels,
            config.in_channels,
        )

        self.attentions = nn.MultiheadAttention(
            config.in_channels,
            config.num_heads,
            batch_first=True,
        )

        self.register_forward_hook(forward_hook)

    def forward(self, x):
        out = x
        # Attention block of Unet
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h * w)
        in_attn = self.attention_norms(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attentions(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        out = out + out_attn

        return out


# if __name__ == "__main__":
#     import torch
#
#     config = AttentionConfig(
#         in_channels=64,
#         t_emb_dim=64,
#         num_heads=16,
#         norm_channels=32,
#         normtype="group",
#     )
#
#     block = AttentionBlock(config=config)
#
#     img = torch.zeros(10, 64, 28, 28)
#     block(img)
