from dataclasses import dataclass

import torch
import torch.nn as nn
from models.down import DownBlock
from models.mid import MidBlock
from models.up import UpBlock
from torch import nn
import structlog

logger = structlog.get_logger()


def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    # factor = 10000^(2i/d_model)
    factor = 10000 ** (
        (
            torch.arange(
                start=0,
                end=temb_dim // 2,
                dtype=torch.float32,
                device=time_steps.device,
            )
            / (temb_dim // 2)
        )
    )

    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


@dataclass
class ModelConfig:
    module: str
    time_emb_dim: int
    down_sample: list[bool]
    im_channels: int
    down_channels: list[int]
    num_heads: int
    mid_channels: list[int]
    attn_down: list[bool]
    attn_up: list[bool]
    norm_channels: int
    conv_out_channels: int
    num_down_layers: int
    num_mid_layers: int
    num_up_layers: int


class Unet(nn.Module):
    """
    UNet Architecture.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        ###########################
        #  Time embedding layer.  #
        ###########################

        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.config.time_emb_dim, self.config.time_emb_dim),  # type: ignore
            nn.SiLU(),
            nn.Linear(self.config.time_emb_dim, self.config.time_emb_dim),  # type: ignore
        )

        self.conv_in = nn.Conv2d(
            self.config.im_channels,
            self.config.down_channels[0],  # type: ignore
            kernel_size=3,
            padding=1,
        )

        ######################
        #  Down blocks UNet  #
        ######################

        self.downs = nn.ModuleList([])
        for i in range(len(self.config.down_channels) - 1):  # type: ignore
            self.downs.append(
                DownBlock(
                    self.config.down_channels[i],  # type: ignore
                    self.config.down_channels[i + 1],  # type: ignore
                    self.config.time_emb_dim,
                    down_sample=self.config.down_sample[i],  # type: ignore
                    num_heads=self.config.num_heads,
                    num_layers=self.config.num_down_layers,
                    attn=self.config.attn_down[i],  # type: ignore
                    norm_channels=self.config.norm_channels,
                )
            )

        #####################
        #  Mid blocks UNet  #
        #####################

        self.mids = nn.ModuleList([])
        for i in range(len(self.config.mid_channels) - 1):  # type: ignore
            self.mids.append(
                MidBlock(
                    self.config.mid_channels[i],  # type: ignore
                    self.config.mid_channels[i + 1],  # type: ignore
                    self.config.time_emb_dim,
                    num_heads=self.config.num_heads,
                    num_layers=self.config.num_mid_layers,
                    norm_channels=self.config.norm_channels,
                )
            )

        ####################
        #  Up Blocks UNet  #
        ####################

        self.up_sample = list(reversed(self.config.down_sample))  # type: ignore
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.config.down_channels) - 1)):  # type: ignore
            self.ups.append(
                UpBlock(
                    self.config.down_channels[i] * 2,  # type: ignore
                    self.config.down_channels[i - 1] if i != 0 else self.config.conv_out_channels,  # type: ignore
                    self.config.time_emb_dim,
                    up_sample=self.config.down_sample[i],  # type: ignore
                    num_heads=self.config.num_heads,
                    num_layers=self.config.num_up_layers,
                    norm_channels=self.config.norm_channels,
                    # attn=self.config.attn_up[i],  # type: ignore
                )
            )

        self.norm_out = nn.GroupNorm(
            self.config.norm_channels,
            self.config.conv_out_channels,
        )  # type: ignore
        self.conv_out = nn.Conv2d(
            self.config.conv_out_channels,
            self.config.im_channels,
            kernel_size=3,
            padding=1,  # type: ignore
        )

    def forward(self, x, t):
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        # B x C x H x W
        out = self.conv_in(x)
        # B x C1 x H x W

        # t_emb -> B x t_emb_dim
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.config.time_emb_dim)
        t_emb = self.t_proj(t_emb)

        down_outs = []

        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, t_emb)
        # down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
        # out B x C4 x H/4 x W/4

        for mid in self.mids:
            out = mid(out, t_emb)
        # out B x C3 x H/4 x W/4

        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)
            # out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        # out B x C x H x W
        return out


if __name__ == "__main__":

    config = ModelConfig(
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
    )
    model = Unet(config).cuda()

    im = torch.zeros(size=(32, 1, 28, 28)).cuda()
    t = torch.tensor([0.5]).cuda()

    out = model(im, t)
    print(out.shape)
    print(out.min())
    print(out.max())
