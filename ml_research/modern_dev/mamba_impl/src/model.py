"""
Mamba - Selective State Space Model Architecture

This module implements the Mamba architecture with selective state space models,
providing linear-time sequence modeling with input-dependent selection mechanisms.

Reference:
    "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    https://arxiv.org/abs/2312.00752
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SelectiveSSM, CausalConv1d


@dataclass
class MambaConfig:
    """Configuration for Mamba model.

    Attributes:
        d_model: Model dimension (embedding size).
        n_layer: Number of Mamba blocks.
        vocab_size: Vocabulary size for embedding layer.
        d_state: SSM state dimension (N in the paper).
        d_conv: Convolution kernel size.
        expand: Expansion factor for inner dimension.
        dt_rank: Rank for delta projection. If "auto", uses ceil(d_model / 16).
        dt_min: Minimum value for delta (discretization step).
        dt_max: Maximum value for delta.
        dt_init: Initialization method for delta ("random" or "constant").
        dt_scale: Scaling factor for delta initialization.
        dt_init_floor: Floor value for delta initialization.
        conv_bias: Whether to use bias in convolution layer.
        bias: Whether to use bias in linear projections.
        use_fast_path: Whether to use fused CUDA kernel (if available).
        layer_idx: Layer index (for caching in generation).
        pad_vocab_size_multiple: Pad vocab to multiple of this value.
        norm_epsilon: Epsilon for layer normalization.
        residual_in_fp32: Whether to keep residual connections in fp32.
        initializer_range: Standard deviation for weight initialization.
    """

    d_model: int = 768
    n_layer: int = 24
    vocab_size: int = 50280
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False
    use_fast_path: bool = True
    layer_idx: Optional[int] = None
    pad_vocab_size_multiple: int = 8
    norm_epsilon: float = 1e-5
    residual_in_fp32: bool = True
    initializer_range: float = 0.02

    def __post_init__(self):
        """Compute derived attributes."""
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)

        # Pad vocab size
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )


class MambaBlock(nn.Module):
    """Single Mamba block with selective SSM.

    Architecture:
        Input -> Norm -> [Linear -> Conv1D -> SiLU -> SSM] * [Linear -> SiLU] -> Linear -> Output
                                    |                               |
                                    +---------- Gating -------------+

    Args:
        config: MambaConfig instance.
        layer_idx: Index of this layer in the model.
    """

    def __init__(self, config: MambaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.d_inner = config.d_inner
        self.dt_rank = config.dt_rank

        # Input projection: projects to 2x d_inner (for x and z paths)
        self.in_proj = nn.Linear(
            self.d_model,
            self.d_inner * 2,
            bias=config.bias
        )

        # Causal convolution
        self.conv1d = CausalConv1d(
            in_channels=self.d_inner,
            kernel_size=self.d_conv,
            bias=config.conv_bias
        )

        # Selective SSM
        self.ssm = SelectiveSSM(
            d_model=self.d_inner,
            d_state=self.d_state,
            dt_rank=self.dt_rank,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            dt_init=config.dt_init,
            dt_scale=config.dt_scale,
            dt_init_floor=config.dt_init_floor,
        )

        # Output projection
        self.out_proj = nn.Linear(
            self.d_inner,
            self.d_model,
            bias=config.bias
        )

        # Layer norm
        self.norm = nn.LayerNorm(self.d_model, eps=config.norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        inference_params: Optional[dict] = None,
    ) -> torch.Tensor:
        """Forward pass of Mamba block.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, d_model).
            inference_params: Optional dict for caching during generation.

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        batch, seq_len, _ = hidden_states.shape

        # Pre-norm
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        # Input projection -> (batch, seq_len, 2 * d_inner)
        xz = self.in_proj(hidden_states)

        # Split into x and z paths
        x, z = xz.chunk(2, dim=-1)  # Each: (batch, seq_len, d_inner)

        # Convolution path
        # Rearrange for conv: (batch, d_inner, seq_len)
        x = x.transpose(1, 2)
        x = self.conv1d(x, inference_params=inference_params)
        x = x.transpose(1, 2)  # Back to (batch, seq_len, d_inner)

        # Activation
        x = F.silu(x)

        # Selective SSM
        x = self.ssm(x, inference_params=inference_params)

        # Gating with z path
        z = F.silu(z)
        x = x * z

        # Output projection
        output = self.out_proj(x)

        # Residual connection
        output = output + residual

        return output

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> dict:
        """Allocate cache for inference.

        Args:
            batch_size: Batch size for inference.
            max_seqlen: Maximum sequence length.
            dtype: Data type for cache tensors.
            device: Device to allocate tensors on.

        Returns:
            Dictionary containing cache tensors.
        """
        return {
            "conv_state": torch.zeros(
                batch_size, self.d_inner, self.d_conv,
                dtype=dtype, device=device
            ),
            "ssm_state": torch.zeros(
                batch_size, self.d_inner, self.d_state,
                dtype=dtype, device=device
            ),
        }


class Mamba(nn.Module):
    """Mamba language model.

    A stack of Mamba blocks with embedding and language model head.

    Args:
        config: MambaConfig instance.
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(config, layer_idx=i)
            for i in range(config.n_layer)
        ])

        # Final norm
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)

        # Language model head (tied weights with embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Tie weights
        self.lm_head.weight = self.embedding.weight

    def _init_weights(self, module: nn.Module):
        """Initialize weights following Mamba paper conventions."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        inference_params: Optional[dict] = None,
        return_hidden_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass of the Mamba model.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len).
            inference_params: Optional dict for caching during generation.
            return_hidden_states: Whether to return all hidden states.

        Returns:
            If return_hidden_states is False:
                Logits tensor of shape (batch, seq_len, vocab_size).
            If return_hidden_states is True:
                Tuple of (logits, list of hidden states from each layer).
        """
        # Embed tokens
        hidden_states = self.embedding(input_ids)

        all_hidden_states = []

        # Process through Mamba blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states, inference_params=inference_params)
            if return_hidden_states:
                all_hidden_states.append(hidden_states)

        # Final norm
        hidden_states = self.norm_f(hidden_states)

        # Language model head
        logits = self.lm_head(hidden_states)

        if return_hidden_states:
            return logits, all_hidden_states
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
        # Placeholder implementation
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Allocate inference cache
        inference_params = self._allocate_inference_cache(
            batch_size, seq_len + max_new_tokens, device=device
        )

        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Get logits for last position
            logits = self.forward(generated, inference_params=inference_params)
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

    def _allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> dict:
        """Allocate inference cache for all layers.

        Args:
            batch_size: Batch size for inference.
            max_seqlen: Maximum sequence length.
            dtype: Data type for cache tensors.
            device: Device to allocate tensors on.

        Returns:
            Dictionary containing per-layer cache.
        """
        if dtype is None:
            dtype = self.embedding.weight.dtype

        return {
            "layer_caches": [
                layer.allocate_inference_cache(batch_size, max_seqlen, dtype, device)
                for layer in self.layers
            ],
            "seqlen_offset": 0,
        }

    @classmethod
    def from_config(cls, config: MambaConfig) -> "Mamba":
        """Create a Mamba model from configuration.

        Args:
            config: MambaConfig instance.

        Returns:
            Mamba model instance.
        """
        return cls(config)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "Mamba":
        """Load a pretrained Mamba model.

        Args:
            model_name_or_path: Model name or path to checkpoint.

        Returns:
            Mamba model instance.

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
