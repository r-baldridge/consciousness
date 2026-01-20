"""
Griffin Layers - Architecture-Specific Layer Implementations

This module contains the core layer components for Griffin:
- RGLRUCell: Single step of Real-Gated Linear Recurrence
- ParallelScan: Efficient parallel implementation of linear recurrence
- CausalConv1d: Causal convolution for local context
- RotaryEmbedding: Rotary position embeddings (RoPE)
- SlidingWindowMask: Efficient sliding window attention mask
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGLRUCell(nn.Module):
    """Single-step Real-Gated Linear Recurrence Unit.

    Used for efficient incremental inference where we process
    one token at a time with a running hidden state.

    Recurrence:
        h_t = a_t * h_{t-1} + sqrt(1 - a_t^2) * (i_t * Bx_t)

    Where:
        a_t = sigmoid(-8 * softplus(lambda) * r_t)
        r_t = sigmoid(W_r @ x_t)
        i_t = sigmoid(W_i @ x_t)
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        use_bias: bool = True,
    ):
        """Initialize RG-LRU cell.

        Args:
            input_dim: Input dimension.
            state_dim: Hidden state dimension.
            use_bias: Whether to use bias.
        """
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim

        # Projections
        self.input_proj = nn.Linear(input_dim, state_dim, bias=use_bias)
        self.recurrence_gate = nn.Linear(input_dim, state_dim, bias=use_bias)
        self.input_gate = nn.Linear(input_dim, state_dim, bias=use_bias)

        # Learnable decay
        self.log_decay = nn.Parameter(torch.zeros(state_dim))

        # Output projection
        self.output_proj = nn.Linear(state_dim, input_dim, bias=use_bias)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single step forward.

        Args:
            x: Input of shape (batch, input_dim).
            h: Hidden state of shape (batch, state_dim).

        Returns:
            Tuple of (output, new_hidden_state).
        """
        # Compute gates
        r = torch.sigmoid(self.recurrence_gate(x))
        i = torch.sigmoid(self.input_gate(x))

        # Compute decay
        decay_base = F.softplus(self.log_decay)
        a = torch.sigmoid(-8 * decay_base * r)

        # Input scaling for norm preservation
        input_scale = torch.sqrt(1 - a ** 2 + 1e-6)

        # Compute input projection
        x_proj = self.input_proj(x)

        # Update hidden state
        h_new = a * h + input_scale * (i * x_proj)

        # Output projection
        output = self.output_proj(h_new)

        return output, h_new

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Initialize hidden state.

        Args:
            batch_size: Batch size.
            device: Device for tensor.
            dtype: Data type.

        Returns:
            Zero-initialized hidden state.
        """
        return torch.zeros(batch_size, self.state_dim, device=device, dtype=dtype)


class ParallelScan(nn.Module):
    """Parallel scan implementation for efficient linear recurrence.

    Implements the associative scan algorithm to parallelize the computation
    of linear recurrence over the sequence dimension.

    For a linear recurrence:
        h_t = a_t * h_{t-1} + b_t

    The associative operation is:
        (a1, b1) * (a2, b2) = (a1 * a2, a1 * b2 + b1)
    """

    def __init__(self):
        """Initialize parallel scan."""
        super().__init__()

    @staticmethod
    def _binary_associative_scan(
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Perform parallel associative scan.

        Args:
            a: Decay coefficients of shape (batch, seq_len, dim).
            b: Input terms of shape (batch, seq_len, dim).

        Returns:
            Hidden states of shape (batch, seq_len, dim).
        """
        # Placeholder implementation - sequential for correctness
        # In practice, would use custom CUDA kernel or JAX-style scan

        batch_size, seq_len, dim = a.shape
        h = torch.zeros(batch_size, dim, device=a.device, dtype=a.dtype)

        outputs = []
        for t in range(seq_len):
            h = a[:, t, :] * h + b[:, t, :]
            outputs.append(h)

        return torch.stack(outputs, dim=1)

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply parallel scan.

        Args:
            a: Decay coefficients of shape (batch, seq_len, dim).
            b: Input terms of shape (batch, seq_len, dim).
            h0: Optional initial hidden state of shape (batch, dim).

        Returns:
            Hidden states of shape (batch, seq_len, dim).
        """
        if h0 is not None:
            # Prepend initial state contribution
            # h_1 = a_1 * h_0 + b_1 = a_1 * h_0 + b_1
            b = b.clone()
            b[:, 0, :] = b[:, 0, :] + a[:, 0, :] * h0

        return self._binary_associative_scan(a, b)


class CausalConv1d(nn.Module):
    """Causal 1D convolution layer.

    A convolution that only attends to past positions (and current),
    ensuring autoregressive behavior.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
        use_bias: bool = True,
    ):
        """Initialize causal convolution.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of convolution kernel.
            groups: Number of groups for grouped convolution.
            use_bias: Whether to use bias.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1  # Causal padding

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            groups=groups,
            bias=use_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal convolution.

        Args:
            x: Input of shape (batch, seq_len, channels).

        Returns:
            Output of shape (batch, seq_len, channels).
        """
        # Transpose for conv1d: (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # Apply convolution
        y = self.conv(x)

        # Remove future padding to maintain causality
        if self.padding > 0:
            y = y[:, :, :-self.padding]

        # Transpose back: (batch, seq_len, channels)
        return y.transpose(1, 2)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Applies rotary embeddings to query and key tensors for
    relative position encoding in attention.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
    ):
        """Initialize RoPE.

        Args:
            dim: Dimension of embeddings (should be head_dim).
            max_seq_len: Maximum sequence length.
            base: Base for frequency computation.
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute sin/cos cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build sin/cos cache for given sequence length.

        Args:
            seq_len: Sequence length.
        """
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0))

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings.

        Args:
            q: Query tensor of shape (batch, heads, seq_len, head_dim).
            k: Key tensor of shape (batch, heads, seq_len, head_dim).
            position_ids: Optional position indices.

        Returns:
            Tuple of (rotated_q, rotated_k).
        """
        seq_len = q.shape[2]

        # Extend cache if needed
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)

        # Get cached values
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        # Apply rotation
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed


class SlidingWindowMask(nn.Module):
    """Efficient sliding window attention mask generator.

    Creates and caches attention masks for sliding window attention,
    supporting both training and incremental inference.
    """

    def __init__(
        self,
        window_size: int,
        max_seq_len: int = 8192,
    ):
        """Initialize sliding window mask.

        Args:
            window_size: Size of attention window.
            max_seq_len: Maximum sequence length for caching.
        """
        super().__init__()
        self.window_size = window_size
        self.max_seq_len = max_seq_len

        # Build and cache mask
        self._build_mask(max_seq_len)

    def _build_mask(self, seq_len: int) -> None:
        """Build sliding window mask.

        Args:
            seq_len: Sequence length.
        """
        # Create mask: True = mask out, False = attend
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)

        for i in range(seq_len):
            # Window start (inclusive)
            start = max(0, i - self.window_size + 1)
            # Window end (inclusive) - current position
            end = i + 1
            mask[i, start:end] = False

        # Convert to float mask
        float_mask = torch.zeros(seq_len, seq_len)
        float_mask.masked_fill_(mask, float("-inf"))

        self.register_buffer("mask", float_mask)

    def forward(
        self,
        seq_len: int,
        key_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Get sliding window mask.

        Args:
            seq_len: Query sequence length.
            key_len: Key sequence length (for KV cache scenarios).

        Returns:
            Attention mask of shape (seq_len, key_len).
        """
        if key_len is None:
            key_len = seq_len

        # Extend mask if needed
        if seq_len > self.mask.shape[0]:
            self._build_mask(seq_len)

        # Return appropriate slice
        if key_len == seq_len:
            return self.mask[:seq_len, :seq_len]
        else:
            # Handle KV cache scenario
            # Need to construct mask for (seq_len, key_len)
            mask = torch.zeros(seq_len, key_len, device=self.mask.device)

            for i in range(seq_len):
                # Position in full sequence
                pos = key_len - seq_len + i
                # Window start
                start = max(0, pos - self.window_size + 1)
                # Mask everything outside window
                mask[i, :start] = float("-inf")
                # Mask future (shouldn't happen with causal but be safe)
                if pos + 1 < key_len:
                    mask[i, pos + 1:] = float("-inf")

            return mask


class GatedMLP(nn.Module):
    """Gated MLP used in Griffin blocks.

    Implements: out = (gate * activation(up)) @ down
    Similar to SwiGLU but with configurable activation.
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_factor: int = 8,
        dropout: float = 0.0,
        activation: str = "gelu",
        use_bias: bool = True,
    ):
        """Initialize gated MLP.

        Args:
            hidden_dim: Input/output dimension.
            expansion_factor: MLP expansion ratio.
            dropout: Dropout probability.
            activation: Activation function name.
            use_bias: Whether to use bias.
        """
        super().__init__()
        intermediate_dim = hidden_dim * expansion_factor

        # Up projections (2x for gate)
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=use_bias)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=use_bias)

        # Down projection
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=use_bias)

        # Activation
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, hidden_dim).

        Returns:
            Output of shape (batch, seq_len, hidden_dim).
        """
        gate = self.act(self.gate_proj(x))
        up = self.up_proj(x)

        hidden = gate * up
        hidden = self.dropout(hidden)

        return self.down_proj(hidden)


class ChunkedAttention(nn.Module):
    """Memory-efficient chunked attention for very long sequences.

    Processes attention in chunks to reduce peak memory usage
    while maintaining exact computation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        chunk_size: int = 2048,
    ):
        """Initialize chunked attention.

        Args:
            hidden_dim: Model dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            chunk_size: Size of each attention chunk.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.scale = head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with chunked computation.

        Args:
            x: Input of shape (batch, seq_len, hidden_dim).
            mask: Optional attention mask.

        Returns:
            Output of shape (batch, seq_len, hidden_dim).
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention
        q = q.transpose(1, 2)  # (batch, heads, seq, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Process in chunks
        outputs = []
        for chunk_start in range(0, seq_len, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, seq_len)

            # Query chunk
            q_chunk = q[:, :, chunk_start:chunk_end, :]

            # For causal: keys up to chunk_end
            k_chunk = k[:, :, :chunk_end, :]
            v_chunk = v[:, :, :chunk_end, :]

            # Compute attention for this chunk
            attn_weights = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * self.scale

            # Apply mask
            if mask is not None:
                chunk_mask = mask[chunk_start:chunk_end, :chunk_end]
                attn_weights = attn_weights + chunk_mask

            attn_weights = F.softmax(attn_weights, dim=-1)
            chunk_output = torch.matmul(attn_weights, v_chunk)

            outputs.append(chunk_output)

        # Concatenate chunks
        output = torch.cat(outputs, dim=2)

        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, -1)
        output = self.out_proj(output)

        return output
