"""
Flow Matching Layers

Architecture-specific layers for flow matching models including
UNet components, attention mechanisms, and specialized normalizations.
"""

from typing import Optional, Tuple, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization with time/condition modulation.

    Applies layer norm then scales and shifts based on conditioning signal.
    Used for injecting time information into the network.
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        """Initialize adaptive layer norm.

        Args:
            hidden_dim: Dimension of features to normalize
            cond_dim: Dimension of conditioning signal
        """
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.scale = nn.Linear(cond_dim, hidden_dim)
        self.shift = nn.Linear(cond_dim, hidden_dim)

        # Initialize to identity transformation
        nn.init.zeros_(self.scale.weight)
        nn.init.ones_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply adaptive layer norm.

        Args:
            x: Input tensor of shape (batch, ..., hidden_dim)
            cond: Conditioning tensor of shape (batch, cond_dim)

        Returns:
            Normalized and modulated tensor
        """
        x = self.norm(x)
        scale = self.scale(cond)
        shift = self.shift(cond)

        # Expand dimensions for broadcasting
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)

        return x * scale + shift


class FourierFeatures(nn.Module):
    """Random Fourier features for continuous time embedding.

    Provides a richer time representation than simple sinusoidal embeddings
    using random frequency sampling.
    """

    def __init__(self, dim: int, scale: float = 16.0):
        """Initialize Fourier features.

        Args:
            dim: Output dimension (will be doubled for sin/cos)
            scale: Scale factor for random frequencies
        """
        super().__init__()
        self.dim = dim
        self.scale = scale
        # Random frequencies (not trainable)
        self.register_buffer("freqs", torch.randn(dim // 2) * scale)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute Fourier features.

        Args:
            t: Time tensor of shape (batch,) or (batch, 1)

        Returns:
            Fourier features of shape (batch, dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        freqs = t * self.freqs * 2 * math.pi
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class ResidualBlock(nn.Module):
    """Residual block with time conditioning for UNet architectures."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.0,
    ):
        """Initialize residual block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            time_dim: Time embedding dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_channels)

        self.dropout = nn.Dropout(dropout)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Apply residual block.

        Args:
            x: Input of shape (batch, in_channels, height, width)
            t_emb: Time embedding of shape (batch, time_dim)

        Returns:
            Output of shape (batch, out_channels, height, width)
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding
        t_emb = self.time_proj(F.silu(t_emb))
        h = h + t_emb[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention block for UNet architectures."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
    ):
        """Initialize attention block.

        Args:
            channels: Number of input/output channels
            num_heads: Number of attention heads
            head_dim: Dimension per head (defaults to channels // num_heads)
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim or channels // num_heads

        self.norm = nn.GroupNorm(32, channels)

        # QKV projection
        self.qkv = nn.Conv1d(channels, 3 * channels, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)

        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention.

        Args:
            x: Input of shape (batch, channels, height, width)

        Returns:
            Output of same shape
        """
        b, c, h, w = x.shape

        # Flatten spatial dimensions
        x_flat = x.view(b, c, h * w)

        # Normalize
        h_norm = self.norm(x).view(b, c, h * w)

        # QKV projection
        qkv = self.qkv(h_norm)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for attention
        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w)

        # Attention
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        out = out.view(b, c, h * w)

        # Project out
        out = self.proj_out(out)

        return (x_flat + out).view(b, c, h, w)


class Downsample(nn.Module):
    """Downsampling layer using strided convolution."""

    def __init__(self, channels: int, use_conv: bool = True):
        """Initialize downsample.

        Args:
            channels: Number of channels
            use_conv: Use convolution (True) or average pooling (False)
        """
        super().__init__()
        self.use_conv = use_conv

        if use_conv:
            self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample input by factor of 2."""
        return self.op(x)


class Upsample(nn.Module):
    """Upsampling layer using nearest neighbor + convolution."""

    def __init__(self, channels: int, use_conv: bool = True):
        """Initialize upsample.

        Args:
            channels: Number of channels
            use_conv: Use convolution after upsampling
        """
        super().__init__()
        self.use_conv = use_conv

        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample input by factor of 2."""
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class VelocityUNet(nn.Module):
    """UNet architecture for velocity field prediction in flow matching.

    A simplified UNet that predicts the velocity field v(x, t) for
    transporting samples along the probability path.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        time_embed_dim: int = 256,
        dropout: float = 0.0,
        num_heads: int = 8,
    ):
        """Initialize velocity UNet.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            base_channels: Base channel count
            channel_mult: Channel multipliers for each level
            num_res_blocks: Number of residual blocks per level
            attention_resolutions: Resolutions at which to apply attention
            time_embed_dim: Time embedding dimension
            dropout: Dropout probability
            num_heads: Number of attention heads
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions

        # Time embedding
        self.time_embed = nn.Sequential(
            FourierFeatures(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock(ch, out_ch, time_embed_dim, dropout)
                )
                ch = out_ch
            if level < len(channel_mult) - 1:
                self.down_blocks.append(Downsample(ch))

        # Middle blocks
        self.mid_block1 = ResidualBlock(ch, ch, time_embed_dim, dropout)
        self.mid_attn = AttentionBlock(ch, num_heads)
        self.mid_block2 = ResidualBlock(ch, ch, time_embed_dim, dropout)

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResidualBlock(ch + out_ch, out_ch, time_embed_dim, dropout)
                )
                ch = out_ch
            if level > 0:
                self.up_blocks.append(Upsample(ch))

        # Output
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

        # Initialize output to zero
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field.

        Args:
            x: Input image of shape (batch, in_channels, height, width)
            t: Time values of shape (batch,)

        Returns:
            Predicted velocity of same shape as input
        """
        # Time embedding
        t_emb = self.time_embed(t)

        # Input
        h = self.conv_in(x)

        # Downsampling
        hs = [h]
        for block in self.down_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, t_emb)
            else:
                h = block(h)
            hs.append(h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Upsampling with skip connections
        for block in self.up_blocks:
            if isinstance(block, ResidualBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = block(h, t_emb)
            else:
                h = block(h)

        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


class OTCoupler(nn.Module):
    """Optimal Transport coupler for mini-batch OT coupling.

    Computes optimal transport coupling within mini-batches to produce
    straighter flow paths.
    """

    def __init__(self, cost_type: str = "l2", sinkhorn_iterations: int = 100):
        """Initialize OT coupler.

        Args:
            cost_type: Type of cost ('l2', 'cosine')
            sinkhorn_iterations: Number of Sinkhorn iterations
        """
        super().__init__()
        self.cost_type = cost_type
        self.sinkhorn_iterations = sinkhorn_iterations

    def compute_cost_matrix(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise cost matrix.

        Args:
            x: Source samples of shape (n, dim)
            y: Target samples of shape (m, dim)

        Returns:
            Cost matrix of shape (n, m)
        """
        if self.cost_type == "l2":
            # Squared L2 distance
            x_sq = (x ** 2).sum(dim=-1, keepdim=True)
            y_sq = (y ** 2).sum(dim=-1, keepdim=True)
            return x_sq + y_sq.T - 2 * x @ y.T
        elif self.cost_type == "cosine":
            # Cosine distance
            x_norm = F.normalize(x, dim=-1)
            y_norm = F.normalize(y, dim=-1)
            return 1 - x_norm @ y_norm.T
        else:
            raise ValueError(f"Unknown cost type: {self.cost_type}")

    def sinkhorn(
        self,
        cost: torch.Tensor,
        epsilon: float = 0.1,
    ) -> torch.Tensor:
        """Compute Sinkhorn optimal transport plan.

        Args:
            cost: Cost matrix of shape (n, m)
            epsilon: Regularization strength

        Returns:
            Transport plan of shape (n, m)
        """
        n, m = cost.shape

        # Initialize
        log_alpha = torch.zeros(n, device=cost.device)
        log_beta = torch.zeros(m, device=cost.device)

        log_K = -cost / epsilon

        # Sinkhorn iterations
        for _ in range(self.sinkhorn_iterations):
            log_alpha = -torch.logsumexp(log_K + log_beta, dim=1)
            log_beta = -torch.logsumexp(log_K.T + log_alpha, dim=1)

        # Transport plan
        log_P = log_K + log_alpha[:, None] + log_beta[None, :]
        return torch.exp(log_P)

    def couple(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        epsilon: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute OT coupling and return coupled samples.

        Args:
            x0: Noise samples of shape (batch, dim)
            x1: Data samples of shape (batch, dim)
            epsilon: Sinkhorn regularization

        Returns:
            Tuple of (coupled_x0, coupled_x1)
        """
        # Compute cost and transport plan
        cost = self.compute_cost_matrix(x0, x1)
        P = self.sinkhorn(cost, epsilon)

        # Sample from transport plan (for now, just use argmax)
        # In practice, would sample from the coupling
        indices = P.argmax(dim=1)

        return x0, x1[indices]


# Export all public classes
__all__ = [
    "AdaptiveLayerNorm",
    "FourierFeatures",
    "ResidualBlock",
    "AttentionBlock",
    "Downsample",
    "Upsample",
    "VelocityUNet",
    "OTCoupler",
]
