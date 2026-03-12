import torch
import torch.nn as nn
from typing import Union

import structlog

logger = structlog.get_logger()


def forward_hook(module, input, output):
    logger.info(f"Inside forward hook for {module.__class__.__name__}")
    logger.info(f"Input shape: {input[0].shape}")
    logger.info(f"Output shape: {output.shape}")
    logger.info("--------")


def get_normlayer(
    norm_channels,
    in_channels,
    normtype="group",
) -> Union[nn.GroupNorm, nn.BatchNorm2d]:
    if normtype == "group":
        return nn.GroupNorm(norm_channels, in_channels)
    else:
        return nn.BatchNorm2d(in_channels)


class UpBlock(nn.Module):
    r"""
    Up conv block with attention.
    Sequence of following blocks
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block with time embedding
    3. Attention Block
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        t_emb_dim,
        up_sample,
        num_heads,
        num_layers,
        norm_channels,
        cross_attn=False,
        context_dim=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.t_emb_dim = t_emb_dim
        self.cross_attn = cross_attn
        self.context_dim = context_dim
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(
                        norm_channels, in_channels if i == 0 else out_channels
                    ),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(num_layers)
            ]
        )

        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList(
                [
                    nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels))
                    for _ in range(num_layers)
                ]
            )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        out_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),
                )
                for _ in range(num_layers)
            ]
        )

        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)]
        )

        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            ]
        )

        if self.cross_attn:
            assert (
                context_dim is not None
            ), "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels) for _ in range(num_layers)]
            )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels if i == 0 else out_channels, out_channels, kernel_size=1
                )
                for i in range(num_layers)
            ]
        )
        self.up_sample_conv = (
            nn.ConvTranspose2d(in_channels, in_channels // 2, 4, 2, 1)
            if self.up_sample
            else nn.Identity()
        )

        self.register_forward_hook(forward_hook)
        self.hooks = {}
        for name, module in self.named_modules():
            self.hooks[name] = module.register_forward_hook(forward_hook)

    def forward(self, x, out_down=None, t_emb=None, context=None):
        x = self.up_sample_conv(x)
        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)

        out = x
        for i in range(self.num_layers):
            # Resnet
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            # Self Attention
            # batch_size, channels, h, w = out.shape
            # in_attn = out.reshape(batch_size, channels, h * w)
            # in_attn = self.attention_norms[i](in_attn)
            # in_attn = in_attn.transpose(1, 2)
            # out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            # out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            # out = out + out_attn
            # Cross Attention
            if self.cross_attn:
                assert (
                    context is not None
                ), "context cannot be None if cross attention layers are used"
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert (
                    len(context.shape) == 3
                ), "Context shape does not match B,_,CONTEXT_DIM"
                assert (
                    context.shape[0] == x.shape[0]
                    and context.shape[-1] == self.context_dim
                ), "Context shape does not match B,_,CONTEXT_DIM"
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](
                    in_attn, context_proj, context_proj
                )
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn

        return out
