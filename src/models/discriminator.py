import torch
import torch.nn as nn
import torchinfo


class Discriminator(nn.Module):
    r"""
    PatchGAN Discriminator.
    Rather than taking IMG_CHANNELSxIMG_HxIMG_W all the way to
    1 scalar value , we instead predict grid of values.
    Where each grid is prediction of how likely
    the discriminator thinks that the image patch corresponding
    to the grid cell is real
    """

    def __init__(
        self,
    ):
        super().__init__()

        self.im_channels = 1
        self.conv_channels = [16, 32, 64]
        self.kernel_size = [3, 3, 3, 3]
        self.strides = [2, 2, 2, 1]
        self.paddings = [1, 1, 1, 1]
        layers_dim = [self.im_channels] + self.conv_channels + [1]

        activation = nn.LeakyReLU(0.2)
        m_list = []
        for i in range(len(layers_dim) - 1):
            m_list.extend(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            layers_dim[i],
                            layers_dim[i + 1],
                            kernel_size=self.kernel_size[i],
                            stride=self.strides[i],
                            padding=self.paddings[i],
                            bias=False if i != 0 else True,
                        ),
                        (
                            nn.BatchNorm2d(layers_dim[i + 1])
                            if i != len(layers_dim) - 2 and i != 0
                            else nn.Identity()
                        ),
                        activation if i != len(layers_dim) - 2 else nn.Identity(),
                    )
                ]
            )
        self.layers = nn.ModuleList(m_list)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


if __name__ == "__main__":
    x = torch.randn((2, 3, 256, 256))
    prob = Discriminator()(x)
    print(prob.shape)
    print(torchinfo.summary(Discriminator()))

    """
    Original model definition.
    Non-trainable params: 0
    =================================================================
    =================================================================
    Layer (type:depth-idx)                   Param #
    =================================================================
    Discriminator                            --
    ├─ModuleList: 1-1                        --
    │    └─Sequential: 2-1                   --
    │    │    └─Conv2d: 3-1                  3,136
    │    │    └─Identity: 3-2                --
    │    │    └─LeakyReLU: 3-3               --
    │    └─Sequential: 2-2                   --
    │    │    └─Conv2d: 3-4                  131,072
    │    │    └─BatchNorm2d: 3-5             256
    │    │    └─LeakyReLU: 3-6               --
    │    └─Sequential: 2-3                   --
    │    │    └─Conv2d: 3-7                  524,288
    │    │    └─BatchNorm2d: 3-8             512
    │    │    └─LeakyReLU: 3-9               --
    │    └─Sequential: 2-4                   --
    │    │    └─Conv2d: 3-10                 4,096
    │    │    └─Identity: 3-11               --
    │    │    └─Identity: 3-12               --
    =================================================================
    Total params: 663,360
    Trainable params: 663,360
    Non-trainable params: 0
    =================================================================
    """
