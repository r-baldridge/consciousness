"""
xLSTM - Extended Long Short-Term Memory Architecture

This module implements the xLSTM architecture with exponential gating
and matrix-valued memory cells.

Reference:
    "xLSTM: Extended Long Short-Term Memory"
    https://arxiv.org/abs/2405.04517
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import sLSTMCell, mLSTMCell


@dataclass
class xLSTMConfig:
    """Configuration for xLSTM model.

    Attributes:
        vocab_size: Vocabulary size for embedding layer.
        embedding_dim: Model embedding/hidden dimension.
        num_layers: Number of xLSTM blocks.
        layer_types: List of layer types ('s' for sLSTM, 'm' for mLSTM).
        num_heads: Number of attention heads for mLSTM.
        head_dim: Dimension per head.
        up_proj_factor: Expansion factor for up projection.
        conv_kernel_size: Kernel size for causal convolution (sLSTM).
        use_conv: Whether to use convolution in sLSTM blocks.
        layer_norm_epsilon: Epsilon for layer normalization.
        initializer_range: Standard deviation for weight initialization.
        tie_word_embeddings: Whether to tie input and output embeddings.
        dropout: Dropout probability.
        qkv_proj_blocksize: Block size for qkv projection (for efficiency).
        slstm_at_layer_idx: Indices of sLSTM layers (alternative to layer_types).
    """

    vocab_size: int = 50257
    embedding_dim: int = 768
    num_layers: int = 12
    layer_types: Optional[List[str]] = None
    num_heads: int = 8
    head_dim: int = 64
    up_proj_factor: float = 2.0
    conv_kernel_size: int = 4
    use_conv: bool = True
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    dropout: float = 0.0
    qkv_proj_blocksize: int = 4
    slstm_at_layer_idx: Optional[List[int]] = None

    def __post_init__(self):
        """Compute derived attributes."""
        # Default layer types: mostly mLSTM with some sLSTM
        if self.layer_types is None:
            if self.slstm_at_layer_idx is not None:
                # Use provided indices
                self.layer_types = [
                    "s" if i in self.slstm_at_layer_idx else "m"
                    for i in range(self.num_layers)
                ]
            else:
                # Default: sLSTM every 4th layer (ratio 7:1)
                self.layer_types = [
                    "s" if (i + 1) % 4 == 0 else "m"
                    for i in range(self.num_layers)
                ]

        # Validate
        assert len(self.layer_types) == self.num_layers, \
            f"layer_types length ({len(self.layer_types)}) must match num_layers ({self.num_layers})"

        # Compute inner dimension
        self.inner_dim = int(self.embedding_dim * self.up_proj_factor)


class xLSTMBlock(nn.Module):
    """Single xLSTM block (either sLSTM or mLSTM variant).

    Architecture:
        Input -> LayerNorm -> UpProj -> [Conv] -> LSTM Cell -> DownProj -> Residual

    Args:
        config: xLSTMConfig instance.
        layer_idx: Index of this layer.
        block_type: Type of block ('s' for sLSTM, 'm' for mLSTM).
    """

    def __init__(
        self,
        config: xLSTMConfig,
        layer_idx: int,
        block_type: Literal["s", "m"] = "m",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.block_type = block_type

        # Pre-layer norm
        self.ln = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_epsilon)

        # Up projection
        self.up_proj = nn.Linear(
            config.embedding_dim,
            config.inner_dim,
            bias=False
        )

        # Optional causal convolution (mainly for sLSTM)
        if config.use_conv and block_type == "s":
            self.conv = nn.Conv1d(
                config.inner_dim,
                config.inner_dim,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.inner_dim,
            )
        else:
            self.conv = None

        # LSTM cell
        if block_type == "s":
            self.lstm_cell = sLSTMCell(
                input_dim=config.inner_dim,
                hidden_dim=config.inner_dim,
                num_heads=config.num_heads,
            )
        else:  # mLSTM
            self.lstm_cell = mLSTMCell(
                input_dim=config.inner_dim,
                hidden_dim=config.inner_dim,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
            )

        # Down projection
        self.down_proj = nn.Linear(
            config.inner_dim,
            config.embedding_dim,
            bias=False
        )

        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """Forward pass of xLSTM block.

        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim).
            state: Optional tuple of state tensors from previous step.

        Returns:
            Tuple of (output, new_state).
        """
        residual = x

        # Pre-norm
        x = self.ln(x)

        # Up projection
        x = self.up_proj(x)

        # Optional convolution
        if self.conv is not None:
            # (batch, seq, dim) -> (batch, dim, seq)
            x_conv = x.transpose(1, 2)
            x_conv = self.conv(x_conv)
            # Trim for causality
            x_conv = x_conv[..., :x.shape[1]]
            x = x_conv.transpose(1, 2)

        # LSTM cell
        x, new_state = self.lstm_cell(x, state=state)

        # Down projection
        x = self.down_proj(x)

        # Residual and dropout
        x = residual + self.dropout(x)

        return x, new_state

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, ...]:
        """Initialize state for this block.

        Args:
            batch_size: Batch size.
            device: Device for tensors.
            dtype: Data type.

        Returns:
            Initial state tuple.
        """
        return self.lstm_cell.init_state(batch_size, device, dtype)


class xLSTM(nn.Module):
    """xLSTM language model.

    Combines sLSTM and mLSTM blocks with exponential gating
    for competitive performance with Transformers.

    Args:
        config: xLSTMConfig instance.
    """

    def __init__(self, config: xLSTMConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        # xLSTM blocks
        self.blocks = nn.ModuleList([
            xLSTMBlock(
                config,
                layer_idx=i,
                block_type=config.layer_types[i],
            )
            for i in range(config.num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_epsilon)

        # Language model head
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Tie weights
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight

    def _init_weights(self, module: nn.Module):
        """Initialize weights following xLSTM conventions."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Optional[List[Tuple[torch.Tensor, ...]]] = None,
        return_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[torch.Tensor, ...]]]]:
        """Forward pass of xLSTM model.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len).
            state: Optional list of state tuples for each layer.
            return_state: Whether to return updated states.

        Returns:
            If return_state is False:
                Logits tensor of shape (batch, seq_len, vocab_size).
            If return_state is True:
                Tuple of (logits, list of state tuples).
        """
        # Embed tokens
        x = self.embedding(input_ids)

        # Initialize state list if needed
        if state is None and return_state:
            state = [None] * self.config.num_layers

        new_states = []

        # Process through blocks
        for i, block in enumerate(self.blocks):
            layer_state = state[i] if state else None
            x, new_layer_state = block(x, state=layer_state)

            if return_state:
                new_states.append(new_layer_state)

        # Final layer norm
        x = self.ln_f(x)

        # Language model head
        logits = self.lm_head(x)

        if return_state:
            return logits, new_states
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p (nucleus) sampling parameter.
            eos_token_id: Token ID to stop generation.

        Returns:
            Generated token IDs including input tokens.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Process prompt and get initial state
        logits, state = self.forward(input_ids, return_state=True)

        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Get logits for last position
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-k
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # Single-token forward with state
            logits, state = self.forward(next_token, state=state, return_state=True)

        return generated

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[Tuple[torch.Tensor, ...]]:
        """Initialize empty state for generation.

        Args:
            batch_size: Batch size.
            device: Device for state tensors.
            dtype: Data type for state tensors.

        Returns:
            List of state tuples for each layer.
        """
        if device is None:
            device = self.embedding.weight.device
        if dtype is None:
            dtype = self.embedding.weight.dtype

        return [
            block.init_state(batch_size, device, dtype)
            for block in self.blocks
        ]

    @classmethod
    def from_config(cls, config: xLSTMConfig) -> "xLSTM":
        """Create an xLSTM model from configuration.

        Args:
            config: xLSTMConfig instance.

        Returns:
            xLSTM model instance.
        """
        return cls(config)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "xLSTM":
        """Load a pretrained xLSTM model.

        Args:
            model_name_or_path: Model name or path to checkpoint.

        Returns:
            xLSTM model instance.

        Note:
            This is a placeholder. Implement checkpoint loading logic.
        """
        raise NotImplementedError("Pretrained model loading not yet implemented")

    def num_parameters(self, only_trainable: bool = True) -> int:
        """Count model parameters.

        Args:
            only_trainable: Whether to count only trainable parameters.

        Returns:
            Number of parameters.
        """
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_layer_stats(self) -> dict:
        """Get statistics about layer types.

        Returns:
            Dictionary with layer type counts.
        """
        slstm_count = sum(1 for t in self.config.layer_types if t == "s")
        mlstm_count = sum(1 for t in self.config.layer_types if t == "m")

        return {
            "total_layers": self.config.num_layers,
            "slstm_layers": slstm_count,
            "mlstm_layers": mlstm_count,
            "mlstm_to_slstm_ratio": mlstm_count / max(slstm_count, 1),
        }
