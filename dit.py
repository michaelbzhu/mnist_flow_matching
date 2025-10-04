import torch
import torch.nn as nn
import math
from common import ConditionalVectorField


# FourierEncoder from the original code (needed for time embedding)
class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: (bs, 1, 1, 1)
        Returns:
        - embeddings: (bs, dim)
        """
        t = t.view(-1, 1)  # (bs, 1)
        freqs = t * self.weights * 2 * math.pi  # (bs, half_dim)
        sin_embed = torch.sin(freqs)  # (bs, half_dim)
        cos_embed = torch.cos(freqs)  # (bs, half_dim)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2)  # (bs, dim)


class PatchEmbed(nn.Module):
    """
    Converts image into patches and embeds them
    """

    def __init__(
        self, img_size: int, patch_size: int, in_channels: int, embed_dim: int
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Convolutional patch embedding
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (bs, c, h, w)
        Returns:
            patches: (bs, num_patches, embed_dim)
        """
        x = self.proj(x)  # (bs, embed_dim, h/p, w/p)
        x = x.flatten(2)  # (bs, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (bs, num_patches, embed_dim)
        return x


class DiTBlock(nn.Module):
    """
    Transformer block with adaptive layer norm (adaLN) conditioning
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )

        # adaLN modulation - outputs 6 values per block (scale/shift for norm1, norm2, and gate)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (bs, num_patches, hidden_dim)
            c: conditioning (bs, hidden_dim) - combined time and class embedding
        Returns:
            output: (bs, num_patches, hidden_dim)
        """
        # Get adaptive modulation parameters
        scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )  # Each is (bs, hidden_dim)

        # Self-attention block with adaLN
        x_norm = self.norm1(x)  # (bs, num_patches, hidden_dim)
        # Apply scale and shift
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        # Self-attention
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        # Apply gate and residual
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP block with adaLN
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class MNISTDiT(ConditionalVectorField):
    """
    Diffusion Transformer for MNIST
    Drop-in replacement for MNISTUNet
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 1,
        hidden_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        t_embed_dim: int = 384,
        y_embed_dim: int = 384,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_dim)

        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))

        # Time embedding (reuse the FourierEncoder from the original code)
        self.time_embedder = FourierEncoder(t_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Class embedding (reuse from original code)
        self.y_embedder = nn.Embedding(num_embeddings=11, embedding_dim=y_embed_dim)
        self.y_mlp = nn.Sequential(
            nn.Linear(y_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )

        # Final layer
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_dim, patch_size * patch_size * self.out_channels),
        )

        # Final adaLN for the final layer
        self.final_adaLN = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim)
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize patch embedding like nn.Linear
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
        t_embed = self.time_mlp(t_embed)  # (bs, hidden_dim)

        y_embed = self.y_embedder(y)  # (bs, y_embed_dim)
        y_embed = self.y_mlp(y_embed)  # (bs, hidden_dim)

        # Combine conditioning
        c = t_embed + y_embed  # (bs, hidden_dim)

        # Patchify and add positional embedding
        x = self.patch_embed(x)  # (bs, num_patches, hidden_dim)
        x = x + self.pos_embed  # (bs, num_patches, hidden_dim)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, c)  # (bs, num_patches, hidden_dim)

        # Final layer with adaLN
        scale, shift = self.final_adaLN(c).chunk(2, dim=1)  # Each is (bs, hidden_dim)
        x = self.final_layer[0](x)  # LayerNorm
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.final_layer[1](x)  # Linear projection

        # Unpatchify to image
        x = self.unpatchify(x)  # (bs, 1, 32, 32)

        return x


# Example usage - swap this in for your UNet:
if __name__ == "__main__":
    # Create DiT model (can directly replace MNISTUNet)
    dit = MNISTDiT(
        img_size=32,
        patch_size=4,  # 32/4 = 8, so 8x8 = 64 patches
        hidden_dim=384,  # Model dimension
        depth=6,  # Number of transformer blocks
        num_heads=6,  # Number of attention heads
        t_embed_dim=384,
        y_embed_dim=384,
    )

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 32, 32)
    t = torch.rand(batch_size, 1, 1, 1)
    y = torch.randint(0, 11, (batch_size,))

    output = dit(x, t, y)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in dit.parameters()):,}")
