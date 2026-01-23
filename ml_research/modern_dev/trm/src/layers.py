"""
TRM Layers - Core building blocks for Tiny Recursive Model

This module implements the layer components for TRM:
- TRMBlock: Single transformer-style block
- DeepRecursion: The core recursive refinement mechanism
- QHead: Halting probability prediction head
- OutputHead: Solution prediction head
- GridEmbedding: 2D positional embedding for grid tasks
- MLPSequence: MLP-only variant for small grids

Reference:
    "Less is More: Recursive Reasoning with Tiny Networks"
    https://arxiv.org/abs/2510.04871
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Applies rotary embeddings to queries and keys for relative position encoding.
    """

    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for given sequence length."""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos/sin for given sequence length."""
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to queries and keys."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU activation function.

    Combines Swish activation with gated linear units.
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, out_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm without mean centering.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional rotary embeddings."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float = 0.0,
        use_rotary: bool = True,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.use_rotary = use_rotary

        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        if use_rotary:
            self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Transpose for attention: (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply rotary embeddings
        if self.use_rotary:
            cos, sin = self.rotary_emb(x, seq_len)
            cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
            sin = sin.unsqueeze(0).unsqueeze(0)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)

        return self.out_proj(attn_output)


class TRMBlock(nn.Module):
    """Single TRM transformer block.

    Pre-norm architecture with attention and feed-forward layers.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_attention: bool = True,
        use_rotary: bool = True,
        use_swiglu: bool = True,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_attention = use_attention

        # Normalization layers
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)

        # Attention (optional)
        if use_attention:
            self.attn = MultiHeadAttention(
                embed_dim, n_heads, dropout, use_rotary, max_seq_len
            )

        # Feed-forward
        hidden_dim = int(embed_dim * mlp_ratio)
        if use_swiglu:
            self.ffn = SwiGLU(embed_dim, hidden_dim, embed_dim)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
                nn.Dropout(dropout),
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention block (if enabled)
        if self.use_attention:
            x = x + self.attn(self.norm1(x), mask)

        # Feed-forward block
        x = x + self.ffn(self.norm2(x))

        return x


class MLPSequence(nn.Module):
    """MLP-only variant for small grid tasks.

    Uses position-wise MLPs without attention.
    Better for small grids like 9x9 Sudoku.
    """

    def __init__(
        self,
        embed_dim: int,
        n_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_swiglu: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TRMBlock(
                embed_dim=embed_dim,
                n_heads=1,  # Not used
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_attention=False,  # MLP only
                use_swiglu=use_swiglu,
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class GridEmbedding(nn.Module):
    """2D positional embedding for grid-based tasks.

    Combines token embedding with learned 2D position embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        grid_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # 2D positional embeddings
        self.row_embed = nn.Embedding(grid_size, embed_dim // 2)
        self.col_embed = nn.Embedding(grid_size, embed_dim // 2)

        self.dropout = nn.Dropout(dropout)

        # Initialize
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.row_embed.weight, std=0.02)
        nn.init.normal_(self.col_embed.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, grid_size, grid_size) or (batch, seq_len)

        Returns:
            Embedded tensor of shape (batch, seq_len, embed_dim)
        """
        batch_size = x.shape[0]

        # Flatten if 2D grid
        if x.dim() == 3:
            x = x.view(batch_size, -1)

        seq_len = x.shape[1]

        # Token embeddings
        token_emb = self.token_embed(x)

        # Generate position indices
        positions = torch.arange(seq_len, device=x.device)
        rows = positions // self.grid_size
        cols = positions % self.grid_size

        # 2D positional embeddings
        row_emb = self.row_embed(rows)
        col_emb = self.col_embed(cols)
        pos_emb = torch.cat([row_emb, col_emb], dim=-1)

        # Combine
        return self.dropout(token_emb + pos_emb)


class OutputHead(nn.Module):
    """Output prediction head.

    Projects hidden states to vocabulary logits.
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        use_stable_max: bool = True,
    ):
        super().__init__()
        self.use_stable_max = use_stable_max

        self.norm = RMSNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, vocab_size, bias=False)

        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Hidden states (batch, seq_len, embed_dim)

        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        x = self.norm(x)
        logits = self.proj(x)

        # Stable max for numerical stability
        if self.use_stable_max:
            logits = logits - logits.max(dim=-1, keepdim=True).values

        return logits


class QHead(nn.Module):
    """Halting probability prediction head.

    Predicts whether the current solution is correct (q_hat).
    Used for early stopping during inference.
    """

    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim

        self.norm = RMSNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize output to predict 0 initially
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent reasoning state (batch, seq_len, embed_dim)

        Returns:
            Halting probability (batch,) - single value per batch
        """
        z = self.norm(z)
        # Global average pooling over sequence
        z_pooled = z.mean(dim=1)
        # Predict halting probability
        q_hat = self.mlp(z_pooled).squeeze(-1)
        return q_hat


class DeepRecursion(nn.Module):
    """Core deep recursion mechanism.

    Implements the recursive refinement loop from the TRM paper:

    For t in 1..T:
        For i in 1..n:
            z <- net(x, y, z)    # Latent update
        y <- net(y, z)           # Solution update

    The final cycle backpropagates through all n+1 evaluations.
    """

    def __init__(
        self,
        embed_dim: int,
        n_layers: int = 2,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        T_cycles: int = 3,
        n_cycles: int = 6,
        dropout: float = 0.0,
        use_attention: bool = True,
        use_rotary: bool = True,
        use_swiglu: bool = True,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.T_cycles = T_cycles
        self.n_cycles = n_cycles

        # Network for z updates: net(x, y, z) -> z'
        # Takes concatenated [x, y, z] input
        self.z_net = nn.ModuleList([
            TRMBlock(
                embed_dim=embed_dim * 3,  # Concatenated input
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_attention=use_attention,
                use_rotary=use_rotary,
                use_swiglu=use_swiglu,
                max_seq_len=max_seq_len,
            )
            for _ in range(n_layers)
        ])
        self.z_proj = nn.Linear(embed_dim * 3, embed_dim, bias=False)

        # Network for y updates: net(y, z) -> y'
        # Takes concatenated [y, z] input
        self.y_net = nn.ModuleList([
            TRMBlock(
                embed_dim=embed_dim * 2,  # Concatenated input
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_attention=use_attention,
                use_rotary=use_rotary,
                use_swiglu=use_swiglu,
                max_seq_len=max_seq_len,
            )
            for _ in range(n_layers)
        ])
        self.y_proj = nn.Linear(embed_dim * 2, embed_dim, bias=False)

    def _update_z(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Update latent state z given x, y, z."""
        # Concatenate inputs
        combined = torch.cat([x, y, z], dim=-1)

        # Process through network
        for layer in self.z_net:
            combined = layer(combined)

        # Project back to embed_dim
        return self.z_proj(combined)

    def _update_y(
        self,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Update solution state y given y, z."""
        # Concatenate inputs
        combined = torch.cat([y, z], dim=-1)

        # Process through network
        for layer in self.y_net:
            combined = layer(combined)

        # Project back to embed_dim
        return self.y_proj(combined)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        detach_between_T: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform deep recursion.

        Args:
            x: Input embedding (batch, seq_len, embed_dim) - fixed
            y: Solution embedding (batch, seq_len, embed_dim)
            z: Latent state (batch, seq_len, embed_dim)
            detach_between_T: Whether to detach between T cycles (training mode)

        Returns:
            Tuple of (y, z) after recursion
        """
        for t in range(self.T_cycles):
            # Detach between T cycles (except last one for gradient flow)
            if detach_between_T and t < self.T_cycles - 1:
                y = y.detach()
                z = z.detach()

            # n low-level cycles for z updates
            for _ in range(self.n_cycles):
                z = self._update_z(x, y, z)

            # One y update per T cycle
            y = self._update_y(y, z)

        return y, z

    def forward_single_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single recursion step (for analysis/visualization).

        Performs one complete T cycle.
        """
        # n low-level cycles
        for _ in range(self.n_cycles):
            z = self._update_z(x, y, z)

        # One y update
        y = self._update_y(y, z)

        return y, z
