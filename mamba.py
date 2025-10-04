import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from common import ConditionalVectorField

from mamba_ssm import Mamba


# Reuse PatchEmbed and FourierEncoder from DiT
class PatchEmbed(nn.Module):
    """Converts image into patches and embeds them"""

    def __init__(
        self, img_size: int, patch_size: int, in_channels: int, embed_dim: int
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (bs, embed_dim, h/p, w/p)
        x = x.flatten(2).transpose(1, 2)  # (bs, num_patches, embed_dim)
        return x


class FourierEncoder(nn.Module):
    """Time embedding using Fourier features"""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)
        freqs = t * self.weights * 2 * math.pi
        sin_embed = torch.sin(freqs)
        cos_embed = torch.cos(freqs)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2)


class MambaConditioningBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, expand_factor: int = 2):
        super().__init__()
        # Use official Mamba implementation
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            expand=expand_factor,
        )

        # Add conditioning
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(d_model, 3 * d_model)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale, shift, gate = self.adaLN_modulation(c).chunk(3, dim=1)

        residual = x
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.mamba(x)
        x = residual + gate.unsqueeze(1) * x

        return x


class MNISTMamba(ConditionalVectorField):
    """
    Mamba-based diffusion model for MNIST
    Drop-in replacement for MNISTUNet and MNISTDiT
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 1,
        d_model: int = 256,
        d_state: int = 16,
        depth: int = 6,
        expand_factor: int = 2,
        t_embed_dim: int = 256,
        y_embed_dim: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding (same as DiT)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, d_model)

        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

        # Time embedding
        self.time_embedder = FourierEncoder(t_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

        # Class embedding
        self.y_embedder = nn.Embedding(num_embeddings=11, embedding_dim=y_embed_dim)
        self.y_mlp = nn.Sequential(
            nn.Linear(y_embed_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

        # Mamba blocks
        self.blocks = nn.ModuleList(
            [
                MambaConditioningBlock(d_model, d_state, expand_factor)
                for _ in range(depth)
            ]
        )

        # Final layer
        self.final_layer = nn.Sequential(
            nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6),
            nn.Linear(d_model, patch_size * patch_size * self.out_channels),
        )

        # Final adaLN
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize patch embedding
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)

        # Zero-out final layer
        nn.init.constant_(self.final_layer[-1].weight, 0)
        nn.init.constant_(self.final_layer[-1].bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (bs, num_patches, patch_size^2 * c)
        Returns:
            imgs: (bs, c, h, w)
        """
        bs = x.shape[0]
        h = w = self.img_size // self.patch_size
        x = x.reshape(bs, h, w, self.patch_size, self.patch_size, self.out_channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(bs, self.out_channels, self.img_size, self.img_size)
        return imgs

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (bs, 1, 32, 32)
            t: (bs, 1, 1, 1)
            y: (bs,)
        Returns:
            u_t^theta(x|y): (bs, 1, 32, 32)
        """
        # Embed time and class
        t_embed = self.time_embedder(t)  # (bs, t_embed_dim)
        t_embed = self.time_mlp(t_embed)  # (bs, d_model)

        y_embed = self.y_embedder(y)  # (bs, y_embed_dim)
        y_embed = self.y_mlp(y_embed)  # (bs, d_model)

        # Combine conditioning
        c = t_embed + y_embed  # (bs, d_model)

        # Patchify and add positional embedding
        x = self.patch_embed(x)  # (bs, num_patches, d_model)
        x = x + self.pos_embed  # (bs, num_patches, d_model)

        # Apply Mamba blocks
        for block in self.blocks:
            x = block(x, c)  # (bs, num_patches, d_model)

        # Final layer with adaLN
        scale, shift = self.final_adaLN(c).chunk(2, dim=1)  # Each is (bs, d_model)
        x = self.final_layer[0](x)  # LayerNorm
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.final_layer[1](x)  # Linear projection

        # Unpatchify to image
        x = self.unpatchify(x)  # (bs, 1, 32, 32)

        return x


# Example usage
if __name__ == "__main__":
    # Create Mamba model (can directly replace MNISTUNet or MNISTDiT)
    mamba = MNISTMamba(
        img_size=32,
        patch_size=4,
        d_model=256,  # Model dimension
        d_state=16,  # SSM state dimension
        depth=6,  # Number of Mamba blocks
        expand_factor=2,  # Inner dimension expansion
        t_embed_dim=256,
        y_embed_dim=256,
    ).to("cuda")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 32, 32).to("cuda")
    t = torch.rand(batch_size, 1, 1, 1).to("cuda")
    y = torch.randint(0, 11, (batch_size,)).to("cuda")

    output = mamba(x, t, y)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in mamba.parameters()):,}")
    print(f"\nMamba uses linear complexity O(n) vs Transformer's O(n²)")
