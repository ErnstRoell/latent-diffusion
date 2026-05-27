from dataclasses import dataclass
from models.blocks.transformer import TransformerBlockConfig as TrCfg 
from models.blocks.transformer import TransformerBlock
from models.blocks.patchembed import PatchEmbedConfig as PCfg 
from models.blocks.patchembed import PatchEmbed


import torch 
from torch import nn 

# resolution=11,
# patch_size=16,
# in_chans=1,
# n_classes=1000,
# embed_dim=768,
# depth=12,
# n_heads=12,
# mlp_ratio=4.,
# qkv_bias=True,
# p=0.,
# attn_p=0.,

@dataclass
class ModelConfig: 
    module: str
    img_size: int 
    patch_size: int
    in_chans: int 
    n_classes: int 
    embed_dim: int 
    n_heads: int 
    mlp_ratio: float 
    qkv_bias: bool
    p: float 
    attn_p: float
    depth: int
    


class VisionTransformer(nn.Module):
    def __init__(
            self,
            config: ModelConfig 
    ):
        super().__init__()

        pconfig = PCfg(
                in_chans=config.in_chans,
                embed_dim=config.embed_dim,
                patch_size=config.patch_size,
                img_size=config.img_size,
            )
        self.patch_embed = PatchEmbed(
                pconfig
        )
        # self.patch_embed = CurveEnc1D(
        #         pconfig
        # )
        # self.patch_embed = CurveEnc1D(config.in_chans,d_model=config.embed_dim,hidden_dim=64)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(
                torch.zeros(1, 1 + self.patch_embed.n_patches, config.embed_dim)
        )
        self.pos_drop = nn.Dropout(p=config.p)

        trconfig = TrCfg(
                    dim=config.embed_dim,
                    n_heads=config.n_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    p=config.p,
                    attn_p=config.attn_p)

        self.blocks = nn.ModuleList(
            [

                TransformerBlock(trconfig) for _ in range(config.depth)
            ]
        )

        self.norm = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.head = nn.Linear(config.embed_dim, config.n_classes)


    def forward(self, x):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`.

        Returns
        -------
        logits : torch.Tensor
            Logits over all the classes - `(n_samples, n_classes)`.
        """
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(
                n_samples, -1, -1
        )  # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_token_final = x[:, 0]  # just the CLS token
        x = self.head(cls_token_final)

        return x

if __name__ == "__main__":
    config = ModelConfig(
            module="",
        resolution=11,
        in_chans=1,
        num_thetas=17,
        n_classes=1000,
        embed_dim=768,
        depth=12,
        n_heads=1,
        mlp_ratio=4.,
        qkv_bias=True,
        p=0.,
        attn_p=0.
        )

    model = VisionTransformer(config)
    img = torch.ones(size=(13,1,17,11))
    out = model(img)
    print(out.shape)
