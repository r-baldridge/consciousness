"""
RWKV Architecture-Specific Layers

This module implements the core layers for RWKV:
- WKVOperator: The core WKV attention mechanism
- TimeMixing: Time mixing layer (attention-like)
- ChannelMixing: Channel mixing layer (FFN-like)
- TokenShift: Token shifting operation for mixing previous token information
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenShift(nn.Module):
    """Token shifting operation for RWKV.

    Mixes current token with previous token using learnable mixing weights.
    x' = mu * x_t + (1 - mu) * x_{t-1}

    Args:
        hidden_dim: Hidden dimension.
        shift_amount: Number of positions to shift (default 1).
    """

    def __init__(self, hidden_dim: int, shift_amount: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.shift_amount = shift_amount

    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply token shift.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            mu: Mixing weight tensor of shape (hidden_dim,) or (batch, seq_len, hidden_dim).
            state: Optional previous token state of shape (batch, hidden_dim).

        Returns:
            Tuple of (shifted tensor, last token for state update).
        """
        batch, seq_len, hidden_dim = x.shape

        if seq_len == 1:
            # Single token mode (inference)
            if state is None:
                # No previous state, use zeros
                prev = torch.zeros_like(x[:, 0])
            else:
                prev = state

            shifted = mu * x[:, 0] + (1 - mu) * prev
            new_state = x[:, 0]
            return shifted.unsqueeze(1), new_state

        # Multi-token mode (training)
        # Shift tokens: [0, x_0, x_1, ..., x_{n-2}]
        if state is not None:
            # Prepend state
            prev = torch.cat([state.unsqueeze(1), x[:, :-1]], dim=1)
        else:
            # Zero-pad
            prev = F.pad(x[:, :-1], (0, 0, 1, 0))

        # Apply mixing
        shifted = mu * x + (1 - mu) * prev

        # Return last token as new state
        new_state = x[:, -1]

        return shifted, new_state


class WKVOperator(nn.Module):
    """WKV (Weighted Key Value) attention operator.

    The core attention mechanism of RWKV:
        wkv_t = (sum_{i<t} e^{-(t-1-i)w + k_i} * v_i + e^{u+k_t} * v_t)
              / (sum_{i<t} e^{-(t-1-i)w + k_i} + e^{u+k_t})

    In recurrence form:
        a_t = e^{-w} * a_{t-1} + e^{k_t} * v_t
        b_t = e^{-w} * b_{t-1} + e^{k_t}
        wkv_t = a_t / b_t

    Args:
        hidden_dim: Hidden dimension.
        head_size: Size of each attention head.
        num_heads: Number of attention heads.
        layer_idx: Index of the layer (for initialization).
        num_layers: Total number of layers (for initialization).
    """

    def __init__(
        self,
        hidden_dim: int,
        head_size: int,
        num_heads: int,
        layer_idx: int,
        num_layers: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_size = head_size
        self.num_heads = num_heads
        self.layer_idx = layer_idx

        # Learnable decay weight (per channel)
        # Initialized based on layer index for diversity
        self.time_decay = nn.Parameter(self._init_time_decay(layer_idx, num_layers))

        # Bonus for current token
        self.time_first = nn.Parameter(torch.zeros(hidden_dim))

    def _init_time_decay(self, layer_idx: int, num_layers: int) -> torch.Tensor:
        """Initialize time decay weights.

        Args:
            layer_idx: Current layer index.
            num_layers: Total number of layers.

        Returns:
            Initialized decay tensor.
        """
        # Layer-dependent initialization for diversity
        ratio = layer_idx / max(num_layers - 1, 1)

        # Decay values typically in range [-5, -0.5] in log space
        decay = torch.empty(self.hidden_dim)
        for i in range(self.hidden_dim):
            channel_ratio = i / max(self.hidden_dim - 1, 1)
            # Mix of layer and channel position
            zigzag = ((i + 1) % 3 - 1) * 0.1
            decay[i] = -5 + 4.5 * (ratio * 0.3 + channel_ratio * 0.7) + zigzag

        return decay

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute WKV attention.

        Args:
            k: Key tensor of shape (batch, seq_len, hidden_dim).
            v: Value tensor of shape (batch, seq_len, hidden_dim).
            state: Optional tuple of (a, b) state tensors.

        Returns:
            Tuple of (output, (new_a, new_b) state).
        """
        batch, seq_len, hidden_dim = k.shape

        # Get decay as positive values
        w = -torch.exp(self.time_decay)  # Negative for decay
        u = self.time_first

        if seq_len == 1:
            # Single token mode (efficient recurrent)
            return self._forward_single(k, v, w, u, state)
        else:
            # Multi-token mode (parallel)
            return self._forward_parallel(k, v, w, u, state)

    def _forward_single(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Single token WKV computation.

        Args:
            k: Key of shape (batch, 1, hidden_dim).
            v: Value of shape (batch, 1, hidden_dim).
            w: Decay weights.
            u: Current token bonus.
            state: Previous (a, b) state.

        Returns:
            Output and new state.
        """
        k = k.squeeze(1)  # (batch, hidden_dim)
        v = v.squeeze(1)

        if state is None:
            a = torch.zeros_like(k)
            b = torch.zeros_like(k)
        else:
            a, b = state

        # exp(k) for this token
        ek = torch.exp(k)

        # Weighted sum of current + history
        # Current contribution: e^{u+k} * v
        # History contribution: a (already accumulated)
        wkv_num = a + torch.exp(u + k) * v
        wkv_den = b + torch.exp(u + k)

        # Output
        wkv = wkv_num / (wkv_den + 1e-8)

        # Update state: a_new = e^w * a + e^k * v, b_new = e^w * b + e^k
        ew = torch.exp(w)
        new_a = ew * a + ek * v
        new_b = ew * b + ek

        return wkv.unsqueeze(1), (new_a, new_b)

    def _forward_parallel(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Parallel WKV computation for training.

        This is a placeholder sequential implementation.
        In production, use a parallel scan or CUDA kernel.

        Args:
            k: Key of shape (batch, seq_len, hidden_dim).
            v: Value of shape (batch, seq_len, hidden_dim).
            w: Decay weights.
            u: Current token bonus.
            state: Previous (a, b) state.

        Returns:
            Output and final state.
        """
        batch, seq_len, hidden_dim = k.shape

        if state is None:
            a = torch.zeros(batch, hidden_dim, device=k.device, dtype=k.dtype)
            b = torch.zeros(batch, hidden_dim, device=k.device, dtype=k.dtype)
        else:
            a, b = state

        ew = torch.exp(w)  # (hidden_dim,)

        outputs = []
        for t in range(seq_len):
            k_t = k[:, t]  # (batch, hidden_dim)
            v_t = v[:, t]

            ek_t = torch.exp(k_t)

            # Current token contribution with bonus u
            eu_k = torch.exp(u + k_t)

            # WKV = (history + current) / normalization
            wkv_num = a + eu_k * v_t
            wkv_den = b + eu_k
            wkv_t = wkv_num / (wkv_den + 1e-8)

            outputs.append(wkv_t)

            # Update state
            a = ew * a + ek_t * v_t
            b = ew * b + ek_t

        output = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_dim)
        return output, (a, b)


class TimeMixing(nn.Module):
    """Time mixing layer (RWKV's attention-like mechanism).

    Computes:
        r = sigmoid(W_r @ shift(x, mu_r))  # Receptance (gate)
        k = W_k @ shift(x, mu_k)           # Key
        v = W_v @ shift(x, mu_v)           # Value
        output = W_o @ (r * wkv(k, v))

    Args:
        hidden_dim: Hidden dimension.
        head_size: Size of each attention head.
        num_heads: Number of attention heads.
        layer_idx: Index of this layer.
        num_layers: Total number of layers.
        use_data_dependent_decay: Whether to use RWKV-6 style decay.
        decay_lora_dim: LoRA dimension for decay (RWKV-6).
    """

    def __init__(
        self,
        hidden_dim: int,
        head_size: int,
        num_heads: int,
        layer_idx: int,
        num_layers: int,
        use_data_dependent_decay: bool = False,
        decay_lora_dim: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_size = head_size
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        self.use_data_dependent_decay = use_data_dependent_decay

        # Token shift module
        self.token_shift = TokenShift(hidden_dim)

        # Mixing weights for token shift
        self.time_mix_r = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.time_mix_k = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(hidden_dim) * 0.5)

        # Projections
        self.receptance = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # WKV operator
        self.wkv = WKVOperator(
            hidden_dim=hidden_dim,
            head_size=head_size,
            num_heads=num_heads,
            layer_idx=layer_idx,
            num_layers=num_layers,
        )

        # Optional RWKV-6 data-dependent decay
        if use_data_dependent_decay:
            self.decay_lora_a = nn.Linear(hidden_dim, decay_lora_dim, bias=False)
            self.decay_lora_b = nn.Linear(decay_lora_dim, hidden_dim, bias=False)
            self.decay_scale = nn.Parameter(torch.ones(hidden_dim) * 0.1)

        # Optional layer normalization for output
        self.ln_x = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass of time mixing layer.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            state: Optional state dict with 'shift' and 'wkv' keys.

        Returns:
            Tuple of (output, updated state).
        """
        batch, seq_len, _ = x.shape

        # Get previous token state for shift
        shift_state = state.get("shift", None) if state else None
        wkv_state = state.get("wkv", None) if state else None

        # Token shift for r, k, v
        x_r, shift_state_r = self.token_shift(x, self.time_mix_r, shift_state)
        x_k, shift_state_k = self.token_shift(x, self.time_mix_k, shift_state)
        x_v, shift_state_v = self.token_shift(x, self.time_mix_v, shift_state)

        # Compute r, k, v
        r = torch.sigmoid(self.receptance(x_r))
        k = self.key(x_k)
        v = self.value(x_v)

        # WKV attention
        wkv_out, new_wkv_state = self.wkv(k, v, state=wkv_state)

        # Gate and project
        output = self.output(r * self.ln_x(wkv_out))

        # Update state
        new_state = {
            "shift": shift_state_v,  # Use v's shift state (they should be same)
            "wkv": new_wkv_state,
        }

        return output, new_state

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Initialize state for recurrent inference.

        Args:
            batch_size: Batch size.
            device: Device for tensors.
            dtype: Data type.

        Returns:
            Initial state dict.
        """
        return {
            "shift": torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype),
            "wkv": (
                torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype),
                torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype),
            ),
        }


class ChannelMixing(nn.Module):
    """Channel mixing layer (RWKV's FFN-like mechanism).

    Computes:
        r = sigmoid(W_r @ shift(x, mu_r))  # Receptance (gate)
        k = W_k @ shift(x, mu_k)           # Key (FFN input)
        output = r * (W_v @ squared_relu(k))

    Args:
        hidden_dim: Hidden dimension.
        layer_idx: Index of this layer.
        num_layers: Total number of layers.
        expand_factor: Expansion factor for FFN (default 4).
    """

    def __init__(
        self,
        hidden_dim: int,
        layer_idx: int,
        num_layers: int,
        expand_factor: float = 4.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_idx = layer_idx

        intermediate_dim = int(hidden_dim * expand_factor)

        # Token shift
        self.token_shift = TokenShift(hidden_dim)

        # Mixing weights
        self.time_mix_r = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.time_mix_k = nn.Parameter(torch.ones(hidden_dim) * 0.5)

        # Projections
        self.receptance = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.value = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of channel mixing layer.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            state: Optional previous token state.

        Returns:
            Tuple of (output, updated state).
        """
        # Token shift
        x_r, state_r = self.token_shift(x, self.time_mix_r, state)
        x_k, state_k = self.token_shift(x, self.time_mix_k, state)

        # Compute r and k
        r = torch.sigmoid(self.receptance(x_r))
        k = self.key(x_k)

        # Squared ReLU activation
        k = torch.square(F.relu(k))

        # Gated output
        output = r * self.value(k)

        return output, state_k

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Initialize state for recurrent inference.

        Args:
            batch_size: Batch size.
            device: Device for tensors.
            dtype: Data type.

        Returns:
            Initial state tensor.
        """
        return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)


class GroupNorm(nn.Module):
    """Group normalization with optional affine transform.

    Used in some RWKV variants for per-head normalization.

    Args:
        num_groups: Number of groups.
        num_channels: Number of channels.
        eps: Epsilon for numerical stability.
        affine: Whether to use learnable parameters.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply group normalization.

        Args:
            x: Input tensor of shape (batch, ..., channels).

        Returns:
            Normalized tensor.
        """
        # Reshape for group norm
        original_shape = x.shape
        x = x.view(*original_shape[:-1], self.num_groups, -1)

        # Normalize per group
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)

        # Reshape back
        x = x.view(*original_shape)

        # Apply affine
        if self.affine:
            x = x * self.weight + self.bias

        return x
