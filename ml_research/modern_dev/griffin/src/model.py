"""
Griffin Model Implementation

Core model architecture for Griffin - hybrid recurrent/attention architecture.
Based on "Griffin: Mixing Gated Linear Recurrences with Local Attention" (2024).

This module contains:
- GriffinConfig: Configuration dataclass
- RecurrentBlock: Gated linear recurrence block
- LocalAttention: Sliding window attention
- GriffinBlock: Combined hybrid block
- GriffinModel: Complete language model
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GriffinConfig:
    """Configuration for Griffin model.

    Attributes:
        vocab_size: Size of the vocabulary.
        hidden_dim: Hidden dimension size.
        num_layers: Total number of blocks.
        num_heads: Number of attention heads (for local attention blocks).
        head_dim: Dimension per attention head.
        mlp_expansion: MLP expansion factor (mlp_dim = hidden_dim * expansion).
        window_size: Sliding window size for local attention.
        context_length: Maximum context length.
        recurrent_state_dim: Dimension of recurrent hidden state.
        block_pattern: Pattern of block types (e.g., ["recurrent", "recurrent", "attention"]).
        dropout: Dropout probability.
        layer_norm_eps: Layer normalization epsilon.
        use_bias: Whether to use bias in linear layers.
        tie_embeddings: Whether to tie input/output embeddings.
    """
    vocab_size: int = 256000
    hidden_dim: int = 2048
    num_layers: int = 26
    num_heads: int = 8
    head_dim: int = 256
    mlp_expansion: int = 8
    window_size: int = 2048
    context_length: int = 8192
    recurrent_state_dim: int = 2048
    block_pattern: List[str] = field(default_factory=lambda: ["recurrent", "recurrent", "attention"])
    dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    use_bias: bool = True
    tie_embeddings: bool = True
    initializer_range: float = 0.02


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize RMSNorm.

        Args:
            dim: Dimension to normalize.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class GatedLinearRecurrence(nn.Module):
    """Real-Gated Linear Recurrence Unit (RG-LRU).

    The core recurrent cell in Griffin:
        h_t = a_t * h_{t-1} + sqrt(1 - a_t^2) * (B @ x_t)

    Where a_t is an input-dependent gate controlling decay.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        use_bias: bool = True,
    ):
        """Initialize RG-LRU.

        Args:
            input_dim: Input dimension.
            state_dim: Recurrent state dimension.
            use_bias: Whether to use bias.
        """
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, state_dim, bias=use_bias)

        # Gate projections
        self.recurrence_gate = nn.Linear(input_dim, state_dim, bias=use_bias)
        self.input_gate = nn.Linear(input_dim, state_dim, bias=use_bias)

        # Learnable decay parameter
        self.log_decay = nn.Parameter(torch.zeros(state_dim))

        # Output projection
        self.output_proj = nn.Linear(state_dim, input_dim, bias=use_bias)

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of RG-LRU.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).
            h: Optional initial hidden state of shape (batch, state_dim).

        Returns:
            Tuple of (output, final_hidden_state).
        """
        batch_size, seq_len, _ = x.shape

        # Initialize hidden state if not provided
        if h is None:
            h = torch.zeros(batch_size, self.state_dim, device=x.device, dtype=x.dtype)

        # Compute projections for all timesteps
        x_proj = self.input_proj(x)  # (batch, seq_len, state_dim)
        r = torch.sigmoid(self.recurrence_gate(x))  # (batch, seq_len, state_dim)
        i = torch.sigmoid(self.input_gate(x))  # (batch, seq_len, state_dim)

        # Compute decay: a = sigmoid(-8 * softplus(lambda) * r)
        decay_base = F.softplus(self.log_decay)
        a = torch.sigmoid(-8 * decay_base.unsqueeze(0).unsqueeze(0) * r)

        # Compute input scaling: sqrt(1 - a^2) for norm preservation
        input_scale = torch.sqrt(1 - a ** 2 + 1e-6)

        # Sequential recurrence (can be parallelized with associative scan)
        outputs = []
        for t in range(seq_len):
            # h_t = a_t * h_{t-1} + sqrt(1 - a_t^2) * input_t
            h = a[:, t, :] * h + input_scale[:, t, :] * (i[:, t, :] * x_proj[:, t, :])
            outputs.append(h)

        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, state_dim)

        # Project to output dimension
        output = self.output_proj(output)

        return output, h


class LocalAttention(nn.Module):
    """Local Sliding Window Attention.

    Standard multi-head attention restricted to a local context window.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        window_size: int,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        """Initialize local attention.

        Args:
            hidden_dim: Model hidden dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            window_size: Size of sliding attention window.
            dropout: Dropout probability.
            use_bias: Whether to use bias in projections.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.scale = head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=use_bias)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=use_bias)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=use_bias)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=use_bias)

        self.dropout = nn.Dropout(dropout)

    def _create_window_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create sliding window attention mask.

        Args:
            seq_len: Sequence length.
            device: Device for mask tensor.

        Returns:
            Attention mask of shape (seq_len, seq_len).
        """
        # Create causal mask
        mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)  # Upper triangular (future tokens)

        # Create window mask (tokens outside window)
        for i in range(seq_len):
            start = max(0, i - self.window_size + 1)
            mask[i, :start] = True

        # Convert to float mask (0 for attend, -inf for mask)
        float_mask = torch.zeros_like(mask, dtype=torch.float)
        float_mask.masked_fill_(mask, float("-inf"))

        return float_mask

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass of local attention.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            cache: Optional KV cache for incremental decoding.

        Returns:
            Tuple of (output, updated_cache).
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Handle KV cache
        if cache is not None:
            k_cache, v_cache = cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)

            # Truncate cache to window size
            if k.shape[1] > self.window_size:
                k = k[:, -self.window_size:, :, :]
                v = v[:, -self.window_size:, :, :]

        # Transpose for attention: (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply window mask
        kv_len = k.shape[2]
        mask = self._create_window_mask(seq_len, x.device)

        # Adjust mask for KV cache scenario
        if kv_len > seq_len:
            # Extend mask for cached keys
            full_mask = torch.zeros(seq_len, kv_len, device=x.device)
            full_mask[:, -seq_len:] = mask
            # Mask out-of-window cached positions
            for i in range(seq_len):
                window_start = max(0, kv_len - self.window_size + i - seq_len + 1)
                full_mask[i, :window_start] = float("-inf")
            mask = full_mask

        attn_weights = attn_weights + mask.unsqueeze(0).unsqueeze(0)

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)

        # Update cache
        new_cache = (k.transpose(1, 2), v.transpose(1, 2))

        return output, new_cache


class RecurrentBlock(nn.Module):
    """Griffin recurrent block with RG-LRU and MLP.

    Structure:
        x = x + RG-LRU(RMSNorm(x))
        x = x + MLP(RMSNorm(x))
    """

    def __init__(self, config: GriffinConfig):
        """Initialize recurrent block.

        Args:
            config: Griffin configuration.
        """
        super().__init__()

        # Normalizations
        self.norm1 = RMSNorm(config.hidden_dim, config.layer_norm_eps)
        self.norm2 = RMSNorm(config.hidden_dim, config.layer_norm_eps)

        # RG-LRU
        self.rglru = GatedLinearRecurrence(
            input_dim=config.hidden_dim,
            state_dim=config.recurrent_state_dim,
            use_bias=config.use_bias,
        )

        # MLP
        mlp_dim = config.hidden_dim * config.mlp_expansion
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, mlp_dim, bias=config.use_bias),
            nn.GELU(),
            nn.Linear(mlp_dim, config.hidden_dim, bias=config.use_bias),
            nn.Dropout(config.dropout),
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of recurrent block.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            h: Optional recurrent hidden state.

        Returns:
            Tuple of (output, new_hidden_state).
        """
        # RG-LRU with residual
        normed = self.norm1(x)
        rglru_out, h = self.rglru(normed, h)
        x = x + self.dropout(rglru_out)

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x, h


class AttentionBlock(nn.Module):
    """Griffin attention block with local attention and MLP.

    Structure:
        x = x + LocalAttention(RMSNorm(x))
        x = x + MLP(RMSNorm(x))
    """

    def __init__(self, config: GriffinConfig):
        """Initialize attention block.

        Args:
            config: Griffin configuration.
        """
        super().__init__()

        # Normalizations
        self.norm1 = RMSNorm(config.hidden_dim, config.layer_norm_eps)
        self.norm2 = RMSNorm(config.hidden_dim, config.layer_norm_eps)

        # Local attention
        self.attention = LocalAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            window_size=config.window_size,
            dropout=config.dropout,
            use_bias=config.use_bias,
        )

        # MLP
        mlp_dim = config.hidden_dim * config.mlp_expansion
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, mlp_dim, bias=config.use_bias),
            nn.GELU(),
            nn.Linear(mlp_dim, config.hidden_dim, bias=config.use_bias),
            nn.Dropout(config.dropout),
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass of attention block.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            cache: Optional KV cache.

        Returns:
            Tuple of (output, updated_cache).
        """
        # Attention with residual
        normed = self.norm1(x)
        attn_out, cache = self.attention(normed, cache)
        x = x + self.dropout(attn_out)

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x, cache


class GriffinBlock(nn.Module):
    """Unified Griffin block that can be either recurrent or attention.

    Acts as a factory/wrapper that instantiates the appropriate block type.
    """

    def __init__(
        self,
        config: GriffinConfig,
        block_type: Literal["recurrent", "attention"],
    ):
        """Initialize Griffin block.

        Args:
            config: Griffin configuration.
            block_type: Type of block ("recurrent" or "attention").
        """
        super().__init__()
        self.block_type = block_type

        if block_type == "recurrent":
            self.block = RecurrentBlock(config)
        elif block_type == "attention":
            self.block = AttentionBlock(config)
        else:
            raise ValueError(f"Unknown block type: {block_type}")

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]]:
        """Forward pass.

        Args:
            x: Input tensor.
            state: Hidden state (for recurrent) or KV cache (for attention).

        Returns:
            Tuple of (output, updated_state).
        """
        return self.block(x, state)


class GriffinModel(nn.Module):
    """Complete Griffin language model.

    Stacks Griffin blocks according to the block pattern with token
    embeddings and output projection.
    """

    def __init__(self, config: GriffinConfig):
        """Initialize Griffin model.

        Args:
            config: Griffin configuration.
        """
        super().__init__()
        self.config = config

        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Create blocks according to pattern
        self.blocks = nn.ModuleList()
        self.block_types = []

        pattern = config.block_pattern
        for i in range(config.num_layers):
            block_type = pattern[i % len(pattern)]
            self.blocks.append(GriffinBlock(config, block_type))
            self.block_types.append(block_type)

        # Final normalization
        self.final_norm = RMSNorm(config.hidden_dim, config.layer_norm_eps)

        # Output projection
        if config.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize module weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List] = None,
        return_logits: bool = True,
    ) -> Tuple[torch.Tensor, List]:
        """Forward pass of Griffin model.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len).
            states: Optional list of states for each block.
            return_logits: Whether to compute and return logits.

        Returns:
            Tuple of (logits or hidden states, updated states).
        """
        # Token embedding
        x = self.embedding(input_ids)
        x = self.dropout(x)

        # Initialize states if not provided
        if states is None:
            states = [None] * len(self.blocks)

        # Pass through blocks
        new_states = []
        for i, block in enumerate(self.blocks):
            x, state = block(x, states[i])
            new_states.append(state)

        # Final normalization
        x = self.final_norm(x)

        # Compute logits
        if return_logits:
            if self.lm_head is not None:
                logits = self.lm_head(x)
            else:
                logits = F.linear(x, self.embedding.weight)
            return logits, new_states

        return x, new_states

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            input_ids: Starting token IDs of shape (batch, seq_len).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Nucleus sampling parameter.

        Returns:
            Generated token IDs.
        """
        # Initialize states
        states = None

        for _ in range(max_new_tokens):
            # Forward pass
            logits, states = self.forward(input_ids, states)

            # Get logits for last token
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append (only need last token for recurrent blocks)
            input_ids = next_token

            # Yield or accumulate (simplified: just accumulate)
            # In practice, would yield for streaming

        return input_ids

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters.

        Args:
            non_embedding: Exclude embedding parameters.

        Returns:
            Number of parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embedding.weight.numel()
        return n_params
