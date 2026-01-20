"""
RWKV - Receptance Weighted Key Value Architecture

This module implements the RWKV architecture, combining efficient RNN inference
with parallelizable Transformer-like training.

Reference:
    "RWKV: Reinventing RNNs for the Transformer Era"
    https://arxiv.org/abs/2305.13048
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import TimeMixing, ChannelMixing


@dataclass
class RWKVConfig:
    """Configuration for RWKV model.

    Attributes:
        vocab_size: Vocabulary size for embedding layer.
        hidden_dim: Model hidden dimension.
        num_layers: Number of RWKV blocks.
        head_size: Size of each attention head.
        num_heads: Number of attention heads (computed if not provided).
        context_length: Maximum context length.
        use_data_dependent_decay: Whether to use RWKV-6 style data-dependent decay.
        decay_lora_dim: LoRA dimension for decay projection in RWKV-6.
        time_mix_extra_dim: Extra dimension for time mixing projections.
        channel_mix_extra_dim: Extra dimension for channel mixing projections.
        layer_norm_epsilon: Epsilon for layer normalization.
        initializer_range: Standard deviation for weight initialization.
        rescale_layer: Rescale every N layers (0 to disable).
        tie_word_embeddings: Whether to tie input and output embeddings.
        dropout: Dropout probability.
        version: RWKV version (4, 5, 6, or 7).
    """

    vocab_size: int = 50277
    hidden_dim: int = 768
    num_layers: int = 12
    head_size: int = 64
    num_heads: Optional[int] = None
    context_length: int = 4096
    use_data_dependent_decay: bool = False
    decay_lora_dim: int = 64
    time_mix_extra_dim: int = 0
    channel_mix_extra_dim: int = 0
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    rescale_layer: int = 0
    tie_word_embeddings: bool = True
    dropout: float = 0.0
    version: int = 4

    def __post_init__(self):
        """Compute derived attributes."""
        if self.num_heads is None:
            self.num_heads = self.hidden_dim // self.head_size

        # Validate
        assert self.hidden_dim % self.head_size == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by head_size ({self.head_size})"


class RWKVBlock(nn.Module):
    """Single RWKV block with time mixing and channel mixing.

    Architecture:
        Input -> LayerNorm -> TimeMixing -> Residual
              -> LayerNorm -> ChannelMixing -> Residual -> Output

    Args:
        config: RWKVConfig instance.
        layer_idx: Index of this layer.
    """

    def __init__(self, config: RWKVConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Layer norms
        self.ln1 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)

        # Time mixing (attention-like)
        self.time_mixing = TimeMixing(
            hidden_dim=config.hidden_dim,
            head_size=config.head_size,
            num_heads=config.num_heads,
            layer_idx=layer_idx,
            num_layers=config.num_layers,
            use_data_dependent_decay=config.use_data_dependent_decay,
            decay_lora_dim=config.decay_lora_dim,
        )

        # Channel mixing (FFN-like)
        self.channel_mixing = ChannelMixing(
            hidden_dim=config.hidden_dim,
            layer_idx=layer_idx,
            num_layers=config.num_layers,
        )

        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass of RWKV block.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            state: Optional state dict for recurrent mode.

        Returns:
            Tuple of (output tensor, updated state).
        """
        # Time mixing with residual
        time_state = state.get("time", None) if state else None
        residual = x
        x = self.ln1(x)
        x, time_state_out = self.time_mixing(x, state=time_state)
        x = residual + self.dropout(x)

        # Channel mixing with residual
        channel_state = state.get("channel", None) if state else None
        residual = x
        x = self.ln2(x)
        x, channel_state_out = self.channel_mixing(x, state=channel_state)
        x = residual + self.dropout(x)

        # Update state
        new_state = None
        if state is not None or time_state_out is not None:
            new_state = {
                "time": time_state_out,
                "channel": channel_state_out,
            }

        return x, new_state


class RWKV(nn.Module):
    """RWKV language model.

    Supports both parallel (training) and recurrent (inference) modes.

    Args:
        config: RWKVConfig instance.
    """

    def __init__(self, config: RWKVConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Initial layer norm (RWKV uses LN before first block)
        self.ln0 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)

        # RWKV blocks
        self.blocks = nn.ModuleList([
            RWKVBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Final layer norm
        self.ln_out = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)

        # Language model head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Tie weights
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight

    def _init_weights(self, module: nn.Module):
        """Initialize weights following RWKV conventions."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Optional[List[Dict[str, torch.Tensor]]] = None,
        return_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]]:
        """Forward pass of RWKV model.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len).
            state: Optional list of state dicts for each layer.
            return_state: Whether to return updated states.

        Returns:
            If return_state is False:
                Logits tensor of shape (batch, seq_len, vocab_size).
            If return_state is True:
                Tuple of (logits, list of state dicts).
        """
        # Embed tokens
        x = self.embedding(input_ids)

        # Initial layer norm
        x = self.ln0(x)

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

            # Optional rescaling for numerical stability
            if self.config.rescale_layer > 0 and (i + 1) % self.config.rescale_layer == 0:
                x = x / 2.0

        # Final layer norm
        x = self.ln_out(x)

        # Language model head
        logits = self.lm_head(x)

        if return_state:
            return logits, new_states
        return logits

    def forward_recurrent(
        self,
        token: torch.Tensor,
        state: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """Single-token forward pass for efficient inference.

        Args:
            token: Single token ID of shape (batch,) or (batch, 1).
            state: List of state dicts for each layer.

        Returns:
            Tuple of (logits for single position, updated states).
        """
        if token.dim() == 1:
            token = token.unsqueeze(1)

        return self.forward(token, state=state, return_state=True)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        use_recurrent: bool = True,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p (nucleus) sampling parameter.
            eos_token_id: Token ID to stop generation.
            use_recurrent: Whether to use recurrent mode for efficiency.

        Returns:
            Generated token IDs including input tokens.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        generated = input_ids.clone()

        if use_recurrent:
            # Process prompt and get initial state
            logits, state = self.forward(input_ids, return_state=True)
            next_token_logits = logits[:, -1, :] / temperature
        else:
            state = None

        for _ in range(max_new_tokens):
            if use_recurrent and state is not None:
                # Single-token forward
                logits, state = self.forward_recurrent(generated[:, -1:], state)
                next_token_logits = logits[:, -1, :] / temperature
            else:
                # Full sequence forward
                logits = self.forward(generated)
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

        return generated

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Initialize empty state for recurrent inference.

        Args:
            batch_size: Batch size.
            device: Device for state tensors.
            dtype: Data type for state tensors.

        Returns:
            List of state dicts for each layer.
        """
        if device is None:
            device = self.embedding.weight.device
        if dtype is None:
            dtype = self.embedding.weight.dtype

        states = []
        for i in range(self.config.num_layers):
            layer_state = {
                "time": self.blocks[i].time_mixing.init_state(batch_size, device, dtype),
                "channel": self.blocks[i].channel_mixing.init_state(batch_size, device, dtype),
            }
            states.append(layer_state)

        return states

    @classmethod
    def from_config(cls, config: RWKVConfig) -> "RWKV":
        """Create an RWKV model from configuration.

        Args:
            config: RWKVConfig instance.

        Returns:
            RWKV model instance.
        """
        return cls(config)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "RWKV":
        """Load a pretrained RWKV model.

        Args:
            model_name_or_path: Model name or path to checkpoint.

        Returns:
            RWKV model instance.

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
