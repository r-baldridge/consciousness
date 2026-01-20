"""
Ring Attention Layers

Architecture-specific layers for ring attention including
communication primitives, memory-efficient attention, and parallelism utilities.
"""

from typing import Optional, Tuple, List, Callable
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for ring attention.

    Implements position-dependent rotation of query and key vectors
    that extends naturally to very long sequences.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 1048576,  # 1M tokens
        base: float = 10000.0,
    ):
        """Initialize rotary embedding.

        Args:
            dim: Dimension to apply rotation (typically head_dim)
            max_seq_len: Maximum sequence length
            base: Base for frequency computation
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache for cos/sin values
        self._cos_cache = None
        self._sin_cache = None

    def _build_cache(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Build cos/sin cache for given sequence length."""
        if (
            self._cos_cache is not None and
            self._cos_cache.shape[0] >= seq_len and
            self._cos_cache.device == device
        ):
            return

        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self._cos_cache = emb.cos().to(dtype)
        self._sin_cache = emb.sin().to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """Apply rotary embedding.

        Args:
            x: Input tensor (B, H, S, D) or (B, S, H, D)
            position_offset: Position offset for this block

        Returns:
            Rotated tensor
        """
        seq_len = x.shape[-2] if x.dim() == 4 else x.shape[1]

        self._build_cache(
            seq_len + position_offset,
            x.device,
            x.dtype,
        )

        # Get relevant portion of cache
        cos = self._cos_cache[position_offset:position_offset + seq_len]
        sin = self._sin_cache[position_offset:position_offset + seq_len]

        return self._apply_rotary(x, cos, sin)

    def _apply_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary transformation."""
        # Split into pairs
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]

        # Rotate
        return torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin,
        ], dim=-1)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm for transformer models.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize RMS normalization.

        Args:
            dim: Dimension to normalize
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor (..., dim)

        Returns:
            Normalized tensor
        """
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class ParallelLinear(nn.Module):
    """Linear layer with support for tensor parallelism.

    Can split weight matrix across multiple devices for
    model-parallel training.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        parallel_mode: str = "none",
        world_size: int = 1,
        rank: int = 0,
    ):
        """Initialize parallel linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Whether to include bias
            parallel_mode: Parallelism mode ('none', 'column', 'row')
            world_size: Number of devices
            rank: This device's rank
        """
        super().__init__()
        self.parallel_mode = parallel_mode
        self.world_size = world_size
        self.rank = rank

        if parallel_mode == "column":
            # Split output dimension
            local_out = out_features // world_size
            self.weight = nn.Parameter(torch.randn(local_out, in_features))
            self.bias = nn.Parameter(torch.zeros(local_out)) if bias else None
        elif parallel_mode == "row":
            # Split input dimension
            local_in = in_features // world_size
            self.weight = nn.Parameter(torch.randn(out_features, local_in))
            self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        else:
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        output = F.linear(x, self.weight, self.bias)

        # Handle parallel reduction if needed
        if self.parallel_mode == "row":
            # Would need all-reduce here in distributed setting
            pass

        return output


class GatedMLP(nn.Module):
    """Gated MLP block commonly used in modern transformers."""

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """Initialize gated MLP.

        Args:
            hidden_dim: Input/output dimension
            intermediate_dim: Intermediate dimension (default: 4 * hidden_dim)
            dropout: Dropout probability
        """
        super().__init__()

        if intermediate_dim is None:
            intermediate_dim = 4 * hidden_dim

        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated MLP.

        Args:
            x: Input tensor (..., hidden_dim)

        Returns:
            Output tensor (..., hidden_dim)
        """
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class AsyncKVBuffer:
    """Double-buffered KV storage for overlapping communication.

    Manages two buffers for key-value pairs to enable concurrent
    communication and computation in ring attention.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        block_size: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Initialize async KV buffer.

        Args:
            batch_size: Batch size
            num_heads: Number of attention heads
            block_size: Sequence block size
            head_dim: Dimension per head
            device: Device to allocate on
            dtype: Data type
        """
        self.shape = (batch_size, num_heads, block_size, head_dim)

        # Double buffer for K
        self.k_buffer_a = torch.empty(*self.shape, device=device, dtype=dtype)
        self.k_buffer_b = torch.empty(*self.shape, device=device, dtype=dtype)

        # Double buffer for V
        self.v_buffer_a = torch.empty(*self.shape, device=device, dtype=dtype)
        self.v_buffer_b = torch.empty(*self.shape, device=device, dtype=dtype)

        # Track which buffer is active
        self._use_buffer_a = True

    def get_compute_buffer(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the buffer currently being computed on.

        Returns:
            Tuple of (key_buffer, value_buffer)
        """
        if self._use_buffer_a:
            return self.k_buffer_a, self.v_buffer_a
        else:
            return self.k_buffer_b, self.v_buffer_b

    def get_comm_buffer(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the buffer for communication.

        Returns:
            Tuple of (key_buffer, value_buffer)
        """
        if self._use_buffer_a:
            return self.k_buffer_b, self.v_buffer_b
        else:
            return self.k_buffer_a, self.v_buffer_a

    def swap_buffers(self) -> None:
        """Swap active and communication buffers."""
        self._use_buffer_a = not self._use_buffer_a


class CausalMaskCache:
    """Cached causal mask for efficient repeated use.

    Pre-computes and caches causal masks for different block
    position combinations.
    """

    def __init__(self, block_size: int, device: torch.device):
        """Initialize mask cache.

        Args:
            block_size: Size of sequence blocks
            device: Device to create masks on
        """
        self.block_size = block_size
        self.device = device

        # Pre-compute same-block causal mask
        self._same_block_mask = torch.triu(
            torch.ones(block_size, block_size, device=device, dtype=torch.bool),
            diagonal=1
        )

    def get_mask(
        self,
        query_block_idx: int,
        kv_block_idx: int,
        q_len: Optional[int] = None,
        k_len: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Get causal mask for given block pair.

        Args:
            query_block_idx: Index of query block
            kv_block_idx: Index of key-value block
            q_len: Actual query length (for partial blocks)
            k_len: Actual key length (for partial blocks)

        Returns:
            Boolean mask (True = masked) or None if no masking needed
        """
        if kv_block_idx > query_block_idx:
            # Future block - fully masked
            q_len = q_len or self.block_size
            k_len = k_len or self.block_size
            return torch.ones(q_len, k_len, device=self.device, dtype=torch.bool)

        elif kv_block_idx < query_block_idx:
            # Past block - no masking
            return None

        else:
            # Same block - triangular mask
            mask = self._same_block_mask
            if q_len is not None or k_len is not None:
                q_len = q_len or self.block_size
                k_len = k_len or self.block_size
                mask = mask[:q_len, :k_len]
            return mask


class MemoryEfficientCrossAttention(nn.Module):
    """Memory-efficient cross-attention for ring attention.

    Computes attention in chunks to reduce peak memory usage,
    compatible with ring communication pattern.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        chunk_size: int = 1024,
    ):
        """Initialize memory-efficient attention.

        Args:
            hidden_dim: Model dimension
            num_heads: Number of attention heads
            head_dim: Dimension per head
            dropout: Attention dropout
            chunk_size: Size of query chunks for memory efficiency
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute memory-efficient attention.

        Args:
            query: Query tensor (B, Q, D)
            key: Key tensor (B, K, D)
            value: Value tensor (B, K, D)
            mask: Optional attention mask

        Returns:
            Attention output (B, Q, D)
        """
        batch_size, q_len, _ = query.shape
        k_len = key.shape[1]

        # Project
        q = self.q_proj(query).view(batch_size, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, k_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, k_len, self.num_heads, self.head_dim)

        # Transpose for attention
        q = q.transpose(1, 2)  # B, H, Q, D
        k = k.transpose(1, 2)  # B, H, K, D
        v = v.transpose(1, 2)  # B, H, K, D

        # Chunked attention computation
        outputs = []
        for i in range(0, q_len, self.chunk_size):
            chunk_end = min(i + self.chunk_size, q_len)
            q_chunk = q[:, :, i:chunk_end, :]

            # Compute attention for this chunk
            scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale

            if mask is not None:
                chunk_mask = mask[i:chunk_end, :]
                scores = scores.masked_fill(chunk_mask, float("-inf"))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            chunk_output = torch.matmul(attn_weights, v)
            outputs.append(chunk_output)

        # Concatenate chunks
        output = torch.cat(outputs, dim=2)

        # Reshape and project
        output = output.transpose(1, 2).reshape(batch_size, q_len, -1)
        return self.out_proj(output)


class GradientCheckpointWrapper(nn.Module):
    """Wrapper for gradient checkpointing in ring attention.

    Enables memory-efficient training by recomputing activations
    during backward pass.
    """

    def __init__(self, module: nn.Module, use_checkpoint: bool = True):
        """Initialize checkpoint wrapper.

        Args:
            module: Module to wrap
            use_checkpoint: Whether to enable checkpointing
        """
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward with optional checkpointing.

        Args:
            *args: Positional arguments for module
            **kwargs: Keyword arguments for module

        Returns:
            Module output
        """
        if self.use_checkpoint and self.training:
            # Checkpoint the forward pass
            return torch.utils.checkpoint.checkpoint(
                self.module,
                *args,
                use_reentrant=False,
                **kwargs
            )
        else:
            return self.module(*args, **kwargs)


# Export all public classes
__all__ = [
    "RotaryEmbedding",
    "RMSNorm",
    "ParallelLinear",
    "GatedMLP",
    "AsyncKVBuffer",
    "CausalMaskCache",
    "MemoryEfficientCrossAttention",
    "GradientCheckpointWrapper",
]
