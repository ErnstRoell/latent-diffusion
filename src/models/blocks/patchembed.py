from dataclasses import dataclass
from torch import nn 
import torch 

@dataclass
class PatchEmbedConfig:
    img_size: int 
    patch_size: int 
    in_chans: int
    embed_dim: int

def get_patch_position_embedding(pos_emb_dim, grid_size, device):
    assert pos_emb_dim % 4 == 0, 'Position embedding dimension must be divisible by 4'
    grid_size_h, grid_size_w = grid_size
    grid_h = torch.arange(grid_size_h, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32, device=device)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0)
    print("grid",grid.shape)

    # grid_h_positions -> (Number of patch tokens,)
    grid_h_positions = grid[0].reshape(-1)
    grid_w_positions = grid[1].reshape(-1)

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0,
        end=pos_emb_dim // 4,
        dtype=torch.float32,
        device=device) / (pos_emb_dim // 4))
    )
    print("factor",factor.shape)

    grid_h_emb = grid_h_positions[:, None].repeat(1, pos_emb_dim // 4) / factor
    grid_h_emb = torch.cat([torch.sin(grid_h_emb), torch.cos(grid_h_emb)], dim=-1)
    # grid_h_emb -> (Number of patch tokens, pos_emb_dim // 2)

    grid_w_emb = grid_w_positions[:, None].repeat(1, pos_emb_dim // 4) / factor
    grid_w_emb = torch.cat([torch.sin(grid_w_emb), torch.cos(grid_w_emb)], dim=-1)
    pos_emb = torch.cat([grid_h_emb, grid_w_emb], dim=-1)

    # pos_emb -> (Number of patch tokens, pos_emb_dim)
    return pos_emb



class PatchEmbedding(nn.Module):
    def __init__(self, config: PatchEmbedConfig):
        super().__init__()
        self.config = config

        self.project = nn.Conv2d(
                config.in_chans,
                config.embed_dim,
                kernel_size=config.patch_size,
                stride=config.patch_size,
        )

        ############################
        # DiT Layer Initialization #
        ############################
        nn.init.xavier_uniform_(self.project.weight)
        nn.init.constant_(self.project.bias, 0)

    def forward(self, x):

        out = self.project(x)

        # Add 2d sinusoidal position embeddings
        pos_embed = get_patch_position_embedding(pos_emb_dim=self.config.embed_dim,
                                                 grid_size=(self.config.img_size//self.config.patch_size, self.config.img_size//self.config.patch_size),
                                                 device=x.device)
        print("pos emb",pos_embed.shape)
        print(out.shape)
        # out += pos_embed
        return out


class PatchEmbed(nn.Module):
    def __init__(self, config: PatchEmbedConfig):
        super().__init__()
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.n_patches = (config.img_size // config.patch_size) ** 2


        self.proj = nn.Conv2d(
                config.in_chans,
                config.embed_dim,
                kernel_size=config.patch_size,
                stride=config.patch_size,
        )

    def forward(self, x):
        x = self.proj(
                x
            )  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)

        return x



if __name__ == "__main__":
    config = PatchEmbedConfig(img_size=28,in_chans=1,patch_size=2,embed_dim=4*17)
    pe = PatchEmbedding(config)
    im = torch.rand(size=(11,1,28,28))
    print(pe(im).shape)

    # model = PatchEmbed(config)
    #
    # patch = model(ect)
    # print(patch.shape)
