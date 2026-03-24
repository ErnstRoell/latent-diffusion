import torch
import pydantic
import torch.nn as nn
import structlog

logger = structlog.get_logger()
import json

from models.down import DownBlock, DownConfig
from models.mid import MidBlock, MidConfig
from models.up import UpBlock, UpConfig


def config_to_dict(config):
    """
    Converts nested namespace to nested dictionary.
    Needed for printing."""
    return json.loads(
        json.dumps(config, default=lambda s: vars(s)),
    )


def flip_config(up_config: dict):
    # Flip in and out channel dims.
    in_channels = up_config["out_channels"]
    out_channels = up_config["in_channels"]
    up_config["out_channels"] = out_channels
    up_config["in_channels"] = in_channels
    if "down_sample" in up_config.keys():
        up_config["up_sample"] = up_config["down_sample"]  # type: ignore
        up_config.pop("down_sample")

    if "normtype" in up_config.keys():
        up_config.pop("normtype")
    return up_config


class ModelConfig(pydantic.BaseModel):
    module: str
    im_channels: int
    bias: bool
    z_channels: int
    down_blocks: list[DownConfig]
    mid_blocks: list[MidConfig]


class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        logger.info(
            f"Config {self.__class__.__name__}", type="config", **config_to_dict(config)
        )

        ##################### Encoder ######################

        self.conv_in = nn.Conv2d(
            self.config.im_channels,
            self.config.down_blocks[0].in_channels,  # type: ignore
            kernel_size=3,
            padding=1,
        )

        ######################
        #  Down blocks UNet  #
        ######################

        self.conv_out_channels = self.config.down_blocks[0].in_channels
        self.norm_channels = self.config.down_blocks[0].norm_channels

        self.encoder_downs = nn.ModuleList([])
        for down_config in self.config.down_blocks:  # type: ignore
            down_config = config_to_dict(down_config)
            self.encoder_downs.append(DownBlock(DownConfig(**down_config)))

        #####################
        #  Mid blocks UNet  #
        #####################

        self.encoder_mids = nn.ModuleList([])
        for mid_config in self.config.mid_blocks:  # type: ignore
            mid_config = MidConfig(**config_to_dict(mid_config))
            self.encoder_mids.append(MidBlock(mid_config))

        ####################################
        #  Encoder to latent convolutions  #
        ####################################

        self.encoder_norm_out = nn.GroupNorm(
            self.norm_channels, self.config.down_blocks[-1].out_channels
        )
        self.encoder_conv_out = nn.Conv2d(
            self.config.down_blocks[-1].out_channels,
            2 * self.config.z_channels,
            kernel_size=3,
            padding=1,
        )

        # Latent Dimension is 2*Latent because we are predicting mean & variance
        self.pre_quant_conv = nn.Conv2d(
            2 * self.config.z_channels,
            2 * self.config.z_channels,
            kernel_size=1,
        )

        ####################################
        #  Latent to Decoder convolutions  #
        ####################################

        self.post_quant_conv = nn.Conv2d(
            self.config.z_channels,
            self.config.z_channels,
            kernel_size=1,
        )
        self.decoder_conv_in = nn.Conv2d(
            self.config.z_channels,
            self.config.mid_blocks[-1].in_channels,
            kernel_size=3,
            padding=1,
        )

        self.decoder_mids = nn.ModuleList([])
        for mid_config in reversed(self.config.mid_blocks):  # type: ignore
            # Flip and multiply input and output channels
            mid_config = config_to_dict(mid_config)
            mid_config = flip_config(mid_config)
            mid_config = MidConfig(**config_to_dict(mid_config))
            self.decoder_mids.append(MidBlock(mid_config))

        self.decoder_ups = nn.ModuleList([])
        for down_config in reversed(self.config.down_blocks):
            # Flip and multiply input and output channels
            down_config = config_to_dict(down_config)
            up_config = flip_config(down_config)
            self.decoder_ups.append(UpBlock(UpConfig(**up_config)))  # type: ignore

        self.decoder_norm_out = nn.GroupNorm(
            self.norm_channels,
            self.conv_out_channels,
        )  # type: ignore
        self.decoder_conv_out = nn.Conv2d(
            self.conv_out_channels,
            self.config.im_channels,
            kernel_size=3,
            padding=1,  # type: ignore
        )

    def encode(self, x):
        out = self.conv_in(x)
        for idx, down in enumerate(self.encoder_downs):
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        mean_logvar = self.pre_quant_conv(out)
        mean, logvar = torch.chunk(mean_logvar, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        sample = mean + std * torch.randn(mean.shape).to(device=x.device)
        return sample, mean_logvar

    def decode(self, z):
        out = z
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)
        for mid in self.decoder_mids:
            out = mid(out)
        for idx, up in enumerate(self.decoder_ups):
            out = up(out)

        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        return out

    def forward(self, x):
        z, mean_logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mean_logvar


if __name__ == "__main__":
    config = ModelConfig(
        module="",
        z_channels=8,
        im_channels=1,
        bias=True,
        down_blocks=[
            DownConfig(
                in_channels=32,
                out_channels=64,
                t_emb_dim=None,
                down_sample=False,
                num_heads=16,
                num_layers=2,
                attn=False,
                norm_channels=16,
                bias=True,
            ),
            DownConfig(
                in_channels=64,
                out_channels=128,
                t_emb_dim=None,
                down_sample=True,
                num_heads=16,
                num_layers=2,
                attn=False,
                norm_channels=16,
                bias=True,
            ),
        ],
        mid_blocks=[
            MidConfig(
                in_channels=128,
                out_channels=128,
                t_emb_dim=None,
                num_heads=16,
                num_layers=2,
                attn=True,
                norm_channels=16,
                bias=True,
            ),
        ],
    )

    model = VAE(config).cuda()

    im = torch.zeros(size=(33, 1, 28, 28)).cuda()

    out, encoder_out = model(im)

    print(out.shape)
    print(encoder_out.shape)
    print(out.min())
    print(out.max())
