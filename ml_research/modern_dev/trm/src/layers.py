"""
TRM Layers - Core building blocks for Tiny Recursive Model

This module implements the layer components for TRM:
- TRMBlock: Single transformer-style block
- DeepRecursion: The core recursive refinement mechanism
- QHead: Halting probability prediction head
- OutputHead: Solution prediction head
- GridEmbedding: 2D positional embedding for grid tasks
- MLPSequence: MLP-only variant for small grids

Code Repair Extensions (64x48 grid):
- GridPositionalEncoding: 2D learned positional encodings for rectangular grids
- GridAttention: 2D attention with relative position encoding
- RecursiveBlock: Weight-shared recursive transformer block
- FeedForward: SwiGLU feed-forward network
- IterationController: Early stopping based on confidence threshold

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


# =============================================================================
# Code Repair Extensions (64x48 Grid Architecture)
# =============================================================================


class GridPositionalEncoding(nn.Module):
    """2D learned positional encodings for rectangular grids.

    Supports arbitrary grid dimensions (e.g., 64x48 for code repair).
    Uses separate row and column embeddings that are added together.
    """

    def __init__(
        self,
        embed_dim: int,
        max_height: int = 64,
        max_width: int = 48,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_height = max_height
        self.max_width = max_width

        # Learned row and column embeddings
        self.row_embed = nn.Embedding(max_height, embed_dim)
        self.col_embed = nn.Embedding(max_width, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize
        nn.init.normal_(self.row_embed.weight, std=0.02)
        nn.init.normal_(self.col_embed.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate positional encodings for grid input.

        Args:
            x: Input tensor of shape (batch, height, width) or (batch, height, width, embed_dim)

        Returns:
            Positional encoding of shape (batch, height, width, embed_dim)
        """
        if x.dim() == 3:
            batch_size, height, width = x.shape
        else:
            batch_size, height, width, _ = x.shape

        device = x.device

        # Create position indices
        row_pos = torch.arange(height, device=device)
        col_pos = torch.arange(width, device=device)

        # Get embeddings
        row_emb = self.row_embed(row_pos)  # (height, embed_dim)
        col_emb = self.col_embed(col_pos)  # (width, embed_dim)

        # Broadcast and add: (height, 1, dim) + (1, width, dim) -> (height, width, dim)
        pos_encoding = row_emb.unsqueeze(1) + col_emb.unsqueeze(0)

        # Expand for batch
        pos_encoding = pos_encoding.unsqueeze(0).expand(batch_size, -1, -1, -1)

        return self.dropout(pos_encoding)


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation.

    Implements: FFN(x) = W2 * (SiLU(W1 * x) * W3 * x)
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_swiglu: bool = True,
    ):
        super().__init__()
        hidden_dim = hidden_dim or int(embed_dim * 4)
        self.use_swiglu = use_swiglu

        if use_swiglu:
            self.w1 = nn.Linear(embed_dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, embed_dim, bias=False)
            self.w3 = nn.Linear(embed_dim, hidden_dim, bias=False)
        else:
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
                nn.Dropout(dropout),
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        return self.net(x)


class GridAttention(nn.Module):
    """2D attention that operates over rectangular grids.

    Flattens the grid to a sequence for attention computation,
    using relative position encoding for grid structure awareness.
    Supports causal masking for autoregressive generation.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        max_height: int = 64,
        max_width: int = 48,
        use_relative_pos: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.use_relative_pos = use_relative_pos
        self.max_height = max_height
        self.max_width = max_width

        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        # QKV projection
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Relative position bias (2D)
        if use_relative_pos:
            # Range: [-(max_height-1), max_height-1] and [-(max_width-1), max_width-1]
            self.rel_row_bias = nn.Embedding(2 * max_height - 1, n_heads)
            self.rel_col_bias = nn.Embedding(2 * max_width - 1, n_heads)
            nn.init.normal_(self.rel_row_bias.weight, std=0.02)
            nn.init.normal_(self.rel_col_bias.weight, std=0.02)

    def _compute_relative_position_bias(
        self,
        height: int,
        width: int,
        device: torch.device
    ) -> torch.Tensor:
        """Compute 2D relative position bias for attention."""
        # Create position indices for rows and columns
        seq_len = height * width

        # Row and column indices for each position
        row_idx = torch.arange(height, device=device).unsqueeze(1).expand(-1, width).reshape(-1)
        col_idx = torch.arange(width, device=device).unsqueeze(0).expand(height, -1).reshape(-1)

        # Relative positions: (seq_len, seq_len)
        rel_row = row_idx.unsqueeze(1) - row_idx.unsqueeze(0)  # (seq_len, seq_len)
        rel_col = col_idx.unsqueeze(1) - col_idx.unsqueeze(0)  # (seq_len, seq_len)

        # Shift to non-negative indices
        rel_row = rel_row + self.max_height - 1
        rel_col = rel_col + self.max_width - 1

        # Clamp to valid range
        rel_row = rel_row.clamp(0, 2 * self.max_height - 2)
        rel_col = rel_col.clamp(0, 2 * self.max_width - 2)

        # Get bias values: (seq_len, seq_len, n_heads)
        row_bias = self.rel_row_bias(rel_row)
        col_bias = self.rel_col_bias(rel_col)

        # Combine and transpose to (n_heads, seq_len, seq_len)
        bias = (row_bias + col_bias).permute(2, 0, 1)

        return bias

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with grid-aware attention.

        Args:
            x: Input tensor of shape (batch, height, width, embed_dim)
            mask: Optional attention mask (batch, height, width)
            causal: Whether to use causal masking

        Returns:
            Output tensor of shape (batch, height, width, embed_dim)
        """
        batch_size, height, width, _ = x.shape
        seq_len = height * width

        # Flatten spatial dimensions: (batch, seq_len, embed_dim)
        x_flat = x.view(batch_size, seq_len, self.embed_dim)

        # QKV projection
        qkv = self.qkv(x_flat)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        if self.use_relative_pos:
            rel_bias = self._compute_relative_position_bias(height, width, x.device)
            attn = attn + rel_bias.unsqueeze(0)  # (1, heads, seq, seq)

        # Apply mask if provided
        if mask is not None:
            # Flatten mask: (batch, seq_len)
            mask_flat = mask.view(batch_size, seq_len)
            # Expand for attention: (batch, 1, 1, seq_len)
            mask_expanded = mask_flat.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask_expanded == 0, float('-inf'))

        # Causal mask
        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn = attn.masked_fill(causal_mask, float('-inf'))

        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, heads, seq, head_dim)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)

        # Restore grid shape
        out = out.view(batch_size, height, width, self.embed_dim)

        return out


class RecursiveBlock(nn.Module):
    """Single recursive transformer block with weight sharing.

    Architecture: LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        max_height: int = 64,
        max_width: int = 48,
        use_swiglu: bool = True,
    ):
        super().__init__()
        ffn_dim = ffn_dim or int(embed_dim * 4)

        self.norm1 = RMSNorm(embed_dim)
        self.attn = GridAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            dropout=dropout,
            max_height=max_height,
            max_width=max_width,
        )
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=ffn_dim,
            dropout=dropout,
            use_swiglu=use_swiglu,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through recursive block.

        Args:
            x: Input tensor (batch, height, width, embed_dim)
            mask: Optional attention mask
            causal: Whether to use causal masking

        Returns:
            Output tensor (batch, height, width, embed_dim)
        """
        # Attention with residual
        x = x + self.attn(self.norm1(x), mask, causal)
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class IterationController:
    """Controller for early stopping based on confidence threshold.

    Monitors hidden state to determine when the model is confident
    enough in its predictions to stop iterating.
    """

    def __init__(
        self,
        q_threshold: float = 0.95,
        min_iterations: int = 2,
        max_iterations: int = 8,
    ):
        self.q_threshold = q_threshold
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations

    def _compute_confidence(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence score from hidden state.

        Uses inverse of normalized variance as a confidence measure.
        Low variance = high confidence (model is settled on a solution).

        Args:
            hidden: Hidden state (batch, height, width, embed_dim)

        Returns:
            Confidence scores (batch,)
        """
        # Flatten spatial dimensions
        batch_size = hidden.shape[0]
        hidden_flat = hidden.view(batch_size, -1, hidden.shape[-1])

        # Compute variance across spatial positions
        variance = hidden_flat.var(dim=1).mean(dim=-1)  # (batch,)

        # Convert to confidence (lower variance = higher confidence)
        # Use sigmoid to normalize to [0, 1]
        confidence = torch.sigmoid(-torch.log(variance + 1e-8) - 2.0)

        return confidence

    def should_stop(
        self,
        hidden: torch.Tensor,
        iteration: int,
        q_hat: Optional[torch.Tensor] = None,
    ) -> Tuple[bool, torch.Tensor]:
        """
        Determine whether to stop iteration.

        Args:
            hidden: Current hidden state
            iteration: Current iteration number (0-indexed)
            q_hat: Optional explicit confidence from Q-head

        Returns:
            Tuple of (should_stop, confidence)
        """
        if iteration < self.min_iterations - 1:
            # Always do minimum iterations
            confidence = self._compute_confidence(hidden)
            return False, confidence

        if iteration >= self.max_iterations - 1:
            # Always stop at max
            confidence = self._compute_confidence(hidden) if q_hat is None else q_hat
            return True, confidence

        # Use provided q_hat or compute from hidden state
        if q_hat is not None:
            confidence = torch.sigmoid(q_hat)
        else:
            confidence = self._compute_confidence(hidden)

        # Stop if all samples exceed threshold
        should_stop = (confidence > self.q_threshold).all().item()

        return should_stop, confidence
