import torch
from torch import nn
import structlog

logger = structlog.get_logger()

# |%%--%%| <SYvAplFRhf|MOQHqZh3Xu>

num_channels = 16
num_groups = 2


class BlockNormAct(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.GroupNorm(num_channels=num_channels, num_groups=num_groups),
            self.activation,
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.GroupNorm(num_channels=num_channels, num_groups=num_groups),
            self.activation,
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, img):
        return img + self.conv(img)


class BlockActNorm(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            self.activation,
            nn.GroupNorm(num_channels=num_channels, num_groups=num_groups),
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            self.activation,
            nn.GroupNorm(num_channels=num_channels, num_groups=num_groups),
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, img):
        return img + self.conv(img)


class Net(nn.Module):
    def __init__(self, activation, blk_cls, num_layers):
        super().__init__()
        self.activation = activation
        self.conv = nn.Sequential(*[blk_cls(activation) for _ in range(num_layers)])

    def forward(self, img):
        return self.conv(img)


def compute_norms(x):
    x_flat = torch.flatten(x, start_dim=1)
    return x_flat.pow(2).mean(dim=1).sqrt()


act_list = [
    nn.SELU(),
    nn.ReLU(),
    nn.SiLU(),
    nn.GELU(),
    nn.LeakyReLU(negative_slope=0.5),
    nn.LeakyReLU(negative_slope=0.05),
]
blk_list = [
    BlockActNorm,
    BlockNormAct,
]

results = []
for num_l in range(0, 5):
    for act in act_list:
        for blk_cls in blk_list:
            img = torch.randn(512, num_channels, 28, 28)
            model = Net(act, blk_cls, num_l)
            with torch.no_grad():
                out = model(img)

            out_norms = compute_norms(out)

            for el in out_norms:
                results.append(
                    {
                        "block": blk_cls.__name__,
                        "act": str(act),
                        "norm": el.item(),
                        "num_layers": num_l,
                    }
                )


# |%%--%%| <MOQHqZh3Xu|KgOYfHcetp>

import pandas as pd
import seaborn as sns

df = pd.DataFrame(results)
df.groupby(by=["block", "act", "num_layers"]).agg(["min", "mean", "max"])

# |%%--%%| <KgOYfHcetp|AeZoiTCjDA>
sns.relplot(data=df, x="num_layers", y="norm", hue="act", col="block", kind="line")
