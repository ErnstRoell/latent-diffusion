from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
from models.down import DownBlock, DownConfig
from models.mid import MidBlock, MidConfig
from models.up import UpBlock, UpConfig
from torch import nn
import structlog

logger = structlog.get_logger()
import json


def config_to_dict(config):
    """
    Converts nested namespace to nested dictionary.
    Needed for printing."""
    return json.loads(
        json.dumps(config, default=lambda s: vars(s)),
    )


def flip_and_multiply(up_config: dict):
    # Flip in and out channel dims.
    in_channels = up_config["out_channels"]
    out_channels = up_config["in_channels"]
    up_config["out_channels"] = out_channels
    up_config["in_channels"] = in_channels
    up_config["in_channels"] = 2 * in_channels
    up_config["up_sample"] = up_config["down_sample"]  # type: ignore
    up_config.pop("down_sample")
    return UpConfig(**up_config)


def get_time_embedding(time_steps, temb_dim):
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

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
    im_channels: int
    down_blocks: list[DownConfig]
    mid_blocks: list[MidConfig]
    bias: bool


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
            self.config.down_blocks[0].in_channels,  # type: ignore
            kernel_size=3,
            padding=1,
            bias=config.bias,
        )

        logger.info(
            f"Config {self.__class__.__name__}", type="config", **config_to_dict(config)
        )

        ######################
        #  Down blocks UNet  #
        ######################

        self.conv_out_channels = self.config.down_blocks[0].in_channels
        self.norm_channels = self.config.down_blocks[0].norm_channels

        self.downs = nn.ModuleList([])
        for down_config in self.config.down_blocks:  # type: ignore
            down_config = config_to_dict(down_config)
            self.downs.append(DownBlock(DownConfig(**down_config)))

        #####################
        #  Mid blocks UNet  #
        #####################

        self.mids = nn.ModuleList([])
        for mid_config in self.config.mid_blocks:  # type: ignore
            mid_config = MidConfig(**config_to_dict(mid_config))
            self.mids.append(MidBlock(mid_config))

        ####################
        #  Up Blocks UNet  #
        ####################

        self.ups = nn.ModuleList([])
        for down_config in reversed(self.config.down_blocks):
            # Flip and multiply input and output channels
            down_config = config_to_dict(down_config)
            up_config = flip_and_multiply(down_config)
            self.ups.append(UpBlock(up_config))  # type: ignore

        #########################
        #  Output convolutions  #
        #########################

        self.norm_out = nn.GroupNorm(
            self.norm_channels,
            self.conv_out_channels,
        )  # type: ignore
        self.conv_out = nn.Conv2d(
            self.conv_out_channels,
            self.config.im_channels,
            kernel_size=3,
            padding=1,  # type: ignore
            bias=config.bias,
        )

    def forward(self, x, t):
        out = self.conv_in(x)
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.config.time_emb_dim)
        t_emb = self.t_proj(t_emb)

        down_outs = []
        for idx, down in enumerate(self.downs):
            out = down(out, t_emb)
            down_outs.append(out)

        for mid in self.mids:
            out = mid(out, t_emb)

        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)

        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        return out


if __name__ == "__main__":
    config = ModelConfig(
        module="",
        im_channels=1,
        time_emb_dim=128,  # Should be the same everywhere.
        bias=True,
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
                bias=True,
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
                bias=True,
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
                bias=True,
            ),
        ],
    )

    model = Unet(config).cuda()

    im = torch.zeros(size=(32, 1, 28, 28)).cuda()
    t = torch.tensor([0.5]).cuda()

    out = model(im, t)
    print(out.shape)
    print(out.min())
    print(out.max())
