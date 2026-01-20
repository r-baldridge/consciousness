"""
Consistency Models Layers

Architecture-specific layers for consistency models including
UNet components, preconditioning, and specialized blocks.
"""

from typing import Optional, Tuple, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Preconditioner(nn.Module):
    """Input/output preconditioning for consistency models.

    Implements the preconditioning from Karras et al. (2022) that
    improves training stability and convergence.
    """

    def __init__(
        self,
        sigma_data: float = 0.5,
    ):
        """Initialize preconditioner.

        Args:
            sigma_data: Standard deviation of data distribution
        """
        super().__init__()
        self.sigma_data = sigma_data

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute skip connection coefficient."""
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute output scaling coefficient."""
        return sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute input scaling coefficient."""
        return 1 / (sigma ** 2 + self.sigma_data ** 2).sqrt()

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute noise conditioning coefficient."""
        return 0.25 * torch.log(sigma)

    def forward(
        self,
        x: torch.Tensor,
        F_x: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Apply preconditioning to network output.

        Args:
            x: Noisy input
            F_x: Raw network output
            sigma: Noise level

        Returns:
            Preconditioned output with skip connection
        """
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1, 1) if x.dim() == 4 else sigma.unsqueeze(-1)

        c_skip = self.c_skip(sigma)
        c_out = self.c_out(sigma)

        return c_skip * x + c_out * F_x


class FourierEmbedding(nn.Module):
    """Fourier feature embedding for noise level conditioning."""

    def __init__(self, dim: int, scale: float = 16.0):
        """Initialize Fourier embedding.

        Args:
            dim: Output dimension
            scale: Scale factor for frequencies
        """
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.register_buffer("freqs", torch.randn(dim // 2) * scale)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute Fourier embedding.

        Args:
            sigma: Noise level of shape (batch,)

        Returns:
            Embedding of shape (batch, dim)
        """
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)

        # Use log(sigma) for better numerical properties
        log_sigma = torch.log(sigma.clamp(min=1e-8))

        freqs = log_sigma * self.freqs * 2 * math.pi
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class AdaGroupNorm(nn.Module):
    """Adaptive Group Normalization with sigma conditioning.

    Applies group normalization with scale and shift predicted
    from the noise level sigma.
    """

    def __init__(
        self,
        num_channels: int,
        cond_dim: int,
        num_groups: int = 32,
    ):
        """Initialize adaptive group norm.

        Args:
            num_channels: Number of input channels
            cond_dim: Dimension of conditioning signal
            num_groups: Number of groups for group norm
        """
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.scale = nn.Linear(cond_dim, num_channels)
        self.shift = nn.Linear(cond_dim, num_channels)

        # Initialize to identity
        nn.init.ones_(self.scale.bias)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Apply adaptive group norm.

        Args:
            x: Input of shape (batch, channels, ...)
            cond: Conditioning of shape (batch, cond_dim)

        Returns:
            Normalized and modulated output
        """
        x = self.norm(x)
        scale = self.scale(cond)
        shift = self.shift(cond)

        # Reshape for broadcasting
        if x.dim() == 4:  # BCHW
            scale = scale.view(-1, scale.shape[-1], 1, 1)
            shift = shift.view(-1, shift.shape[-1], 1, 1)
        else:
            scale = scale.unsqueeze(-1)
            shift = shift.unsqueeze(-1)

        return x * (1 + scale) + shift


class ConsistencyResBlock(nn.Module):
    """Residual block for consistency model UNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True,
    ):
        """Initialize residual block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            cond_dim: Conditioning dimension
            dropout: Dropout probability
            use_scale_shift_norm: Use adaptive normalization
        """
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        if use_scale_shift_norm:
            self.norm2 = AdaGroupNorm(out_channels, cond_dim)
        else:
            self.norm2 = nn.GroupNorm(32, out_channels)
            self.cond_proj = nn.Linear(cond_dim, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Apply residual block.

        Args:
            x: Input of shape (batch, in_channels, H, W)
            cond: Conditioning of shape (batch, cond_dim)

        Returns:
            Output of shape (batch, out_channels, H, W)
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        if self.use_scale_shift_norm:
            h = self.norm2(h, cond)
        else:
            h = self.norm2(h)
            cond_emb = self.cond_proj(F.silu(cond))
            h = h + cond_emb[:, :, None, None]

        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Self-attention for 2D feature maps."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
    ):
        """Initialize self-attention.

        Args:
            channels: Number of input/output channels
            num_heads: Number of attention heads
            head_dim: Dimension per head
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim or channels // num_heads

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj_out = nn.Linear(channels, channels)

        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention.

        Args:
            x: Input of shape (batch, channels, H, W)

        Returns:
            Output of same shape
        """
        b, c, h, w = x.shape

        # Normalize and flatten
        x_flat = self.norm(x).view(b, c, h * w).transpose(1, 2)  # B, HW, C

        # QKV projection
        qkv = self.qkv(x_flat)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention
        out = (attn @ v).transpose(1, 2).reshape(b, h * w, c)
        out = self.proj_out(out)

        # Reshape and residual
        return x + out.transpose(1, 2).view(b, c, h, w)


class DownsampleBlock(nn.Module):
    """Downsampling block with residual connections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        dropout: float = 0.0,
    ):
        """Initialize downsample block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            cond_dim: Conditioning dimension
            num_res_blocks: Number of residual blocks
            use_attention: Whether to use attention
            dropout: Dropout probability
        """
        super().__init__()

        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()

        for i in range(num_res_blocks):
            self.res_blocks.append(
                ConsistencyResBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    cond_dim,
                    dropout,
                )
            )
            if use_attention:
                self.attn_blocks.append(SelfAttention2d(out_channels))
            else:
                self.attn_blocks.append(nn.Identity())

        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Apply downsample block.

        Args:
            x: Input tensor
            cond: Conditioning tensor

        Returns:
            Tuple of (downsampled output, list of skip connections)
        """
        skips = []

        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            x = res_block(x, cond)
            x = attn_block(x)
            skips.append(x)

        x = self.downsample(x)

        return x, skips


class UpsampleBlock(nn.Module):
    """Upsampling block with skip connections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        dropout: float = 0.0,
    ):
        """Initialize upsample block.

        Args:
            in_channels: Input channels (including skip)
            out_channels: Output channels
            cond_dim: Conditioning dimension
            num_res_blocks: Number of residual blocks
            use_attention: Whether to use attention
            dropout: Dropout probability
        """
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
        )

        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()

        for i in range(num_res_blocks):
            self.res_blocks.append(
                ConsistencyResBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    cond_dim,
                    dropout,
                )
            )
            if use_attention:
                self.attn_blocks.append(SelfAttention2d(out_channels))
            else:
                self.attn_blocks.append(nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        skips: List[torch.Tensor],
    ) -> torch.Tensor:
        """Apply upsample block.

        Args:
            x: Input tensor
            cond: Conditioning tensor
            skips: Skip connection tensors

        Returns:
            Upsampled output
        """
        x = self.upsample(x)

        for res_block, attn_block, skip in zip(
            self.res_blocks, self.attn_blocks, reversed(skips)
        ):
            x = torch.cat([x, skip], dim=1)
            x = res_block(x, cond)
            x = attn_block(x)

        return x


class ConsistencyUNet(nn.Module):
    """UNet architecture for consistency models.

    Predicts denoised samples with preconditioning for
    improved training stability.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        dropout: float = 0.0,
        sigma_data: float = 0.5,
    ):
        """Initialize consistency UNet.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            base_channels: Base channel count
            channel_mult: Channel multipliers per level
            num_res_blocks: Residual blocks per level
            attention_resolutions: Resolutions to apply attention
            dropout: Dropout probability
            sigma_data: Data standard deviation
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sigma_data = sigma_data

        # Preconditioning
        self.precond = Preconditioner(sigma_data)

        # Sigma embedding
        cond_dim = base_channels * 4
        self.sigma_embed = nn.Sequential(
            FourierEmbedding(base_channels),
            nn.Linear(base_channels, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Input
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            use_attn = (64 // (2 ** level)) in attention_resolutions
            self.down_blocks.append(
                DownsampleBlock(
                    ch, out_ch, cond_dim, num_res_blocks, use_attn, dropout
                )
            )
            ch = out_ch

        # Middle
        self.mid_block1 = ConsistencyResBlock(ch, ch, cond_dim, dropout)
        self.mid_attn = SelfAttention2d(ch)
        self.mid_block2 = ConsistencyResBlock(ch, ch, cond_dim, dropout)

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            use_attn = (64 // (2 ** level)) in attention_resolutions
            self.up_blocks.append(
                UpsampleBlock(
                    ch * 2, out_ch, cond_dim, num_res_blocks, use_attn, dropout
                )
            )
            ch = out_ch

        # Output
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

        # Initialize output to zero
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Predict denoised sample.

        Args:
            x: Noisy input of shape (batch, in_channels, H, W)
            sigma: Noise level of shape (batch,)

        Returns:
            Denoised prediction with preconditioning
        """
        # Input preconditioning
        c_in = self.precond.c_in(sigma)
        x_in = x * c_in.view(-1, 1, 1, 1)

        # Sigma conditioning
        cond = self.sigma_embed(sigma)

        # Encoder
        h = self.conv_in(x_in)

        skip_connections = []
        for down_block in self.down_blocks:
            h, skips = down_block(h, cond)
            skip_connections.append(skips)

        # Middle
        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond)

        # Decoder
        for up_block, skips in zip(self.up_blocks, reversed(skip_connections)):
            h = up_block(h, cond, skips)

        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        F_x = self.conv_out(h)

        # Apply preconditioning with skip connection
        return self.precond(x, F_x, sigma)


# Export all public classes
__all__ = [
    "Preconditioner",
    "FourierEmbedding",
    "AdaGroupNorm",
    "ConsistencyResBlock",
    "SelfAttention2d",
    "DownsampleBlock",
    "UpsampleBlock",
    "ConsistencyUNet",
]
