"""
Hyena Model Implementation

Core model architecture for Hyena - long convolutions for sequence modeling.
Based on "Hyena Hierarchy: Towards Larger Convolutional Language Models" (2023).

This module contains:
- HyenaConfig: Configuration dataclass
- HyenaOperator: Core Hyena operator with implicit convolutions
- HyenaBlock: Full transformer-style block with Hyena
- HyenaModel: Complete language model
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HyenaConfig:
    """Configuration for Hyena model.

    Attributes:
        d_model: Hidden dimension size.
        n_layer: Number of Hyena blocks.
        vocab_size: Size of the vocabulary.
        order: Order of the Hyena hierarchy (number of gating operations).
        filter_order: Number of frequencies for implicit filter parameterization.
        emb_dim: Dimension of positional encoding for filters.
        short_filter_order: Kernel size for short convolution.
        max_seq_len: Maximum sequence length.
        dropout: Dropout probability.
        filter_dropout: Dropout for implicit filters.
        activation: Activation function name.
        bidirectional: Whether to use bidirectional convolutions.
        use_flash_fft: Whether to use optimized FFT implementations.
        tie_embeddings: Whether to tie input/output embeddings.
    """
    d_model: int = 512
    n_layer: int = 12
    vocab_size: int = 50257
    order: int = 2
    filter_order: int = 64
    emb_dim: int = 3
    short_filter_order: int = 3
    max_seq_len: int = 2048
    dropout: float = 0.0
    filter_dropout: float = 0.0
    activation: str = "gelu"
    bidirectional: bool = False
    use_flash_fft: bool = False
    tie_embeddings: bool = True

    # Additional fields with defaults
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_bias: bool = True


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for implicit filter generation.

    Generates positional encodings using sin/cos frequencies and normalized
    position, which are used to parameterize the implicit convolution filters.
    """

    def __init__(
        self,
        emb_dim: int,
        max_seq_len: int,
        learnable_frequencies: bool = True,
    ):
        """Initialize positional encoding.

        Args:
            emb_dim: Number of frequency components.
            max_seq_len: Maximum sequence length.
            learnable_frequencies: Whether frequencies are learnable.
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len

        # Initialize frequencies
        if learnable_frequencies:
            self.frequencies = nn.Parameter(
                torch.randn(emb_dim) * 0.01
            )
        else:
            # Fixed log-spaced frequencies
            frequencies = torch.exp(
                torch.linspace(0, -4, emb_dim)
            )
            self.register_buffer("frequencies", frequencies)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Generate positional encoding.

        Args:
            seq_len: Length of sequence to encode.

        Returns:
            Positional encoding tensor of shape (seq_len, 2 * emb_dim + 1).
        """
        # Position indices normalized to [0, 1]
        t = torch.linspace(0, 1, seq_len, device=self.frequencies.device)

        # Compute sin and cos encodings
        # Shape: (seq_len, emb_dim)
        angles = 2 * torch.pi * self.frequencies.unsqueeze(0) * t.unsqueeze(1)

        # Concatenate sin, cos, and normalized position
        # Shape: (seq_len, 2 * emb_dim + 1)
        encoding = torch.cat([
            torch.sin(angles),
            torch.cos(angles),
            t.unsqueeze(1),
        ], dim=-1)

        return encoding


class ImplicitFilter(nn.Module):
    """Implicit filter parameterization using an MLP.

    Instead of storing a full convolution filter explicitly, we parameterize
    it as a neural network that maps positional encodings to filter values.
    This allows for constant parameter count regardless of sequence length.
    """

    def __init__(
        self,
        d_model: int,
        filter_order: int,
        emb_dim: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        """Initialize implicit filter.

        Args:
            d_model: Output filter dimension.
            filter_order: Hidden dimension of filter MLP.
            emb_dim: Positional encoding dimension.
            max_seq_len: Maximum sequence length.
            dropout: Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Positional encoding for filter generation
        self.pos_enc = PositionalEncoding(emb_dim, max_seq_len)

        # Input dimension: sin, cos, and position
        input_dim = 2 * emb_dim + 1

        # MLP to generate filter values
        self.filter_mlp = nn.Sequential(
            nn.Linear(input_dim, filter_order),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(filter_order, filter_order),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(filter_order, d_model),
        )

        # Learnable decay parameter for windowing
        self.decay = nn.Parameter(torch.ones(d_model) * 0.5)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Generate implicit convolution filter.

        Args:
            seq_len: Length of filter to generate.

        Returns:
            Filter tensor of shape (seq_len, d_model).
        """
        # Get positional encoding
        pos_enc = self.pos_enc(seq_len)  # (seq_len, input_dim)

        # Generate filter through MLP
        h = self.filter_mlp(pos_enc)  # (seq_len, d_model)

        # Apply exponential window for stability
        t = torch.linspace(0, 1, seq_len, device=h.device).unsqueeze(1)
        window = torch.exp(-self.decay.abs() * t * seq_len)

        return h * window


class DataControlledGating(nn.Module):
    """Data-controlled gating mechanism for Hyena.

    Computes input-dependent gating signals that modulate the convolution
    outputs, enabling context-dependent processing.
    """

    def __init__(
        self,
        d_model: int,
        short_filter_order: int = 3,
        use_bias: bool = True,
    ):
        """Initialize gating mechanism.

        Args:
            d_model: Model dimension.
            short_filter_order: Kernel size for short convolution.
            use_bias: Whether to use bias in projections.
        """
        super().__init__()
        self.d_model = d_model

        # Short convolution for local context
        self.short_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=short_filter_order,
            padding=short_filter_order // 2,
            groups=d_model,
            bias=use_bias,
        )

        # Gating projection
        self.gate_proj = nn.Linear(d_model, d_model, bias=use_bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gating signal and gated input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tuple of (gating signal, gated input).
        """
        # Short convolution: (batch, seq_len, d_model) -> transpose -> conv -> transpose
        x_conv = self.short_conv(x.transpose(1, 2)).transpose(1, 2)

        # Gating signal
        gate = torch.sigmoid(self.gate_proj(x_conv))

        return gate, x_conv * gate


class HyenaOperator(nn.Module):
    """Core Hyena operator implementing the hierarchy formula.

    The Hyena operator recursively applies:
        H^(1)(v, x_1) = (h * v) . x_1
        H^(N)(v, x_1, ..., x_N) = H^(1)(H^(N-1)(v, x_1, ..., x_{N-1}), x_N)

    where * is long convolution and . is element-wise multiplication.
    """

    def __init__(self, config: HyenaConfig):
        """Initialize Hyena operator.

        Args:
            config: Hyena configuration.
        """
        super().__init__()
        self.config = config
        self.order = config.order

        # Input projections for value and control signals
        self.in_proj = nn.Linear(
            config.d_model,
            (config.order + 1) * config.d_model,
            bias=config.use_bias,
        )

        # Output projection
        self.out_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=config.use_bias,
        )

        # Implicit filters for each order
        self.filters = nn.ModuleList([
            ImplicitFilter(
                config.d_model,
                config.filter_order,
                config.emb_dim,
                config.max_seq_len,
                config.filter_dropout,
            )
            for _ in range(config.order)
        ])

        # Data-controlled gating for each control signal
        self.gates = nn.ModuleList([
            DataControlledGating(
                config.d_model,
                config.short_filter_order,
                config.use_bias,
            )
            for _ in range(config.order)
        ])

        self.dropout = nn.Dropout(config.dropout)

    def _fft_conv(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Perform convolution using FFT.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            h: Filter tensor of shape (seq_len, d_model).

        Returns:
            Convolved tensor of shape (batch, seq_len, d_model).
        """
        seq_len = x.shape[1]

        # Pad to power of 2 for efficient FFT
        fft_len = 2 ** (2 * seq_len - 1).bit_length()

        # FFT of input and filter
        x_f = torch.fft.rfft(x, n=fft_len, dim=1)
        h_f = torch.fft.rfft(h, n=fft_len, dim=0).unsqueeze(0)

        # Multiply in frequency domain
        y_f = x_f * h_f

        # Inverse FFT
        y = torch.fft.irfft(y_f, n=fft_len, dim=1)

        # Truncate to original length
        return y[:, :seq_len, :]

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of Hyena operator.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            cache: Optional cache for incremental inference.

        Returns:
            Tuple of (output tensor, updated cache).
        """
        batch_size, seq_len, _ = x.shape

        # Project input to value and control signals
        projections = self.in_proj(x)

        # Split into value and control signals
        v, *controls = projections.chunk(self.order + 1, dim=-1)

        # Apply Hyena hierarchy
        for i in range(self.order):
            # Get implicit filter
            h = self.filters[i](seq_len)

            # Apply long convolution
            v = self._fft_conv(v, h)

            # Get gating signal and apply
            gate, gated_control = self.gates[i](controls[i])
            v = v * gated_control

        # Output projection
        output = self.out_proj(v)
        output = self.dropout(output)

        # TODO: Implement caching for incremental inference
        return output, None


class HyenaBlock(nn.Module):
    """Full Hyena block with residual connection and normalization.

    Follows standard transformer block pattern:
        x = x + HyenaOperator(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    """

    def __init__(self, config: HyenaConfig):
        """Initialize Hyena block.

        Args:
            config: Hyena configuration.
        """
        super().__init__()

        # Layer norms
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Hyena operator
        self.hyena = HyenaOperator(config)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model, bias=config.use_bias),
            nn.GELU() if config.activation == "gelu" else nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.d_model, config.d_model, bias=config.use_bias),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of Hyena block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            cache: Optional cache for incremental inference.

        Returns:
            Tuple of (output tensor, updated cache).
        """
        # Hyena with residual
        hyena_out, cache = self.hyena(self.norm1(x), cache)
        x = x + hyena_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x, cache


class HyenaModel(nn.Module):
    """Complete Hyena language model.

    Stacks multiple HyenaBlocks with token embeddings and optional
    tied output projection.
    """

    def __init__(self, config: HyenaConfig):
        """Initialize Hyena model.

        Args:
            config: Hyena configuration.
        """
        super().__init__()
        self.config = config

        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Hyena blocks
        self.blocks = nn.ModuleList([
            HyenaBlock(config) for _ in range(config.n_layer)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Output projection
        if config.tie_embeddings:
            self.lm_head = None  # Will use embedding weight
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

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
        cache: Optional[List[torch.Tensor]] = None,
        return_logits: bool = True,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Forward pass of Hyena model.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len).
            cache: Optional list of caches for incremental inference.
            return_logits: Whether to compute and return logits.

        Returns:
            Tuple of (logits or hidden states, updated caches).
        """
        # Token embedding
        x = self.embedding(input_ids)
        x = self.dropout(x)

        # Pass through blocks
        new_caches = []
        for i, block in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            x, new_cache = block(x, layer_cache)
            new_caches.append(new_cache)

        # Final normalization
        x = self.final_norm(x)

        # Compute logits
        if return_logits:
            if self.lm_head is not None:
                logits = self.lm_head(x)
            else:
                logits = F.linear(x, self.embedding.weight)
            return logits, new_caches

        return x, new_caches

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
            Generated token IDs of shape (batch, seq_len + max_new_tokens).
        """
        # TODO: Implement efficient generation with caching
        for _ in range(max_new_tokens):
            # Forward pass (truncate if too long)
            if input_ids.shape[1] > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]

            logits, _ = self.forward(input_ids)

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

                # Remove tokens with cumulative probability above threshold
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

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
