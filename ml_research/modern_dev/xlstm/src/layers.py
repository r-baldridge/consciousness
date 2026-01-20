"""
xLSTM Architecture-Specific Layers

This module implements the core layers for xLSTM:
- sLSTMCell: Scalar LSTM with exponential gating
- mLSTMCell: Matrix LSTM with key-value memory
- ExponentialGating: Exponential gating mechanism
- MatrixMemory: Matrix-valued memory cell for mLSTM
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExponentialGating(nn.Module):
    """Exponential gating mechanism for xLSTM.

    Unlike sigmoid gating (bounded 0-1), exponential gating is unbounded,
    allowing better gradient flow over long sequences.

    The normalizer state prevents values from exploding.

    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden dimension.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Gate projections
        self.input_gate_proj = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.forget_gate_proj = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute exponential gates.

        Args:
            x: Input tensor of shape (batch, input_dim).
            h: Hidden state of shape (batch, hidden_dim).

        Returns:
            Tuple of (input_gate, forget_gate) with exp() applied.
        """
        # Concatenate input and hidden
        combined = torch.cat([x, h], dim=-1)

        # Compute gates with exponential activation
        i_gate = torch.exp(self.input_gate_proj(combined))
        f_gate = torch.exp(self.forget_gate_proj(combined))

        return i_gate, f_gate


class sLSTMCell(nn.Module):
    """Scalar LSTM cell with exponential gating.

    xLSTM's sLSTM variant uses exponential gates instead of sigmoid,
    plus a normalizer state to prevent explosion.

    Equations:
        i_t = exp(W_i [x_t, h_{t-1}] + b_i)  # exponential input gate
        f_t = exp(W_f [x_t, h_{t-1}] + b_f)  # exponential forget gate
        o_t = sigmoid(W_o [x_t, h_{t-1}] + b_o)  # sigmoid output gate
        c_tilde = tanh(W_c [x_t, h_{t-1}] + b_c)  # candidate

        n_t = f_t * n_{t-1} + i_t  # normalizer
        c_t = f_t * c_{t-1} + i_t * c_tilde  # cell state
        h_t = o_t * (c_t / max(n_t, 1))  # normalized hidden state

    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden (cell) dimension.
        num_heads: Number of heads (for multi-head variant).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Combined projection for efficiency
        # Projects to: i_gate, f_gate, o_gate, cell_candidate
        self.combined_proj = nn.Linear(
            input_dim,
            4 * hidden_dim,
            bias=True
        )

        # Recurrent projection (from hidden state)
        self.recurrent_proj = nn.Linear(
            hidden_dim,
            4 * hidden_dim,
            bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass of sLSTM cell.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).
            state: Optional tuple of (h, c, n) states.

        Returns:
            Tuple of (output, (h, c, n) new state).
        """
        batch, seq_len, _ = x.shape

        # Initialize state if not provided
        if state is None:
            h = torch.zeros(batch, self.hidden_dim, device=x.device, dtype=x.dtype)
            c = torch.zeros(batch, self.hidden_dim, device=x.device, dtype=x.dtype)
            n = torch.ones(batch, self.hidden_dim, device=x.device, dtype=x.dtype)
        else:
            h, c, n = state

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t]  # (batch, input_dim)

            # Combined projections
            gates = self.combined_proj(x_t) + self.recurrent_proj(h)

            # Split into gates
            i_pre, f_pre, o_pre, c_tilde_pre = gates.chunk(4, dim=-1)

            # Apply activations
            i_gate = torch.exp(i_pre)  # Exponential input gate
            f_gate = torch.exp(f_pre)  # Exponential forget gate
            o_gate = torch.sigmoid(o_pre)  # Sigmoid output gate
            c_tilde = torch.tanh(c_tilde_pre)  # Cell candidate

            # Update normalizer: n_t = f_t * n_{t-1} + i_t
            n = f_gate * n + i_gate

            # Update cell state: c_t = f_t * c_{t-1} + i_t * c_tilde
            c = f_gate * c + i_gate * c_tilde

            # Compute hidden state with normalization
            # h_t = o_t * (c_t / max(n_t, 1))
            h = o_gate * (c / torch.clamp(n, min=1.0))

            outputs.append(h)

        output = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_dim)

        return output, (h, c, n)

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize state for this cell.

        Args:
            batch_size: Batch size.
            device: Device for tensors.
            dtype: Data type.

        Returns:
            Tuple of (h, c, n) initial states.
        """
        h = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        n = torch.ones(batch_size, self.hidden_dim, device=device, dtype=dtype)
        return (h, c, n)


class MatrixMemory(nn.Module):
    """Matrix-valued memory for mLSTM.

    Stores key-value associations in a matrix C of shape (d, d):
        C_t = f_t * C_{t-1} + i_t * v_t @ k_t^T

    Retrieval is done via query:
        output = C_t @ q_t

    Args:
        hidden_dim: Hidden dimension (d).
        num_heads: Number of heads.
        head_dim: Dimension per head.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 1,
        head_dim: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Total dimension across heads
        self.total_head_dim = num_heads * head_dim

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        q: torch.Tensor,
        i_gate: torch.Tensor,
        f_gate: torch.Tensor,
        C: Optional[torch.Tensor] = None,
        n: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update matrix memory and retrieve.

        Args:
            k: Key tensor of shape (batch, head_dim).
            v: Value tensor of shape (batch, head_dim).
            q: Query tensor of shape (batch, head_dim).
            i_gate: Input gate (scalar per sample).
            f_gate: Forget gate (scalar per sample).
            C: Previous memory matrix of shape (batch, head_dim, head_dim).
            n: Previous normalizer of shape (batch, head_dim).

        Returns:
            Tuple of (output, new_C, new_n).
        """
        batch = k.shape[0]
        device = k.device
        dtype = k.dtype

        # Initialize if needed
        if C is None:
            C = torch.zeros(batch, self.head_dim, self.head_dim, device=device, dtype=dtype)
        if n is None:
            n = torch.zeros(batch, self.head_dim, device=device, dtype=dtype)

        # Normalize key
        k = k / math.sqrt(self.head_dim)

        # Update memory: C_t = f_t * C_{t-1} + i_t * v_t @ k_t^T
        # i_gate and f_gate are scalars, need to expand
        f_gate = f_gate.view(batch, 1, 1)
        i_gate_expand = i_gate.view(batch, 1, 1)

        # Outer product v @ k^T
        vk_outer = torch.bmm(v.unsqueeze(2), k.unsqueeze(1))  # (batch, head_dim, head_dim)

        C = f_gate * C + i_gate_expand * vk_outer

        # Update normalizer: n_t = f_t * n_{t-1} + i_t * k_t
        n = f_gate.squeeze(-1) * n + i_gate.view(batch, 1) * k

        # Retrieve: output = C_t @ q_t
        output = torch.bmm(C, q.unsqueeze(2)).squeeze(2)  # (batch, head_dim)

        # Normalize by |n^T @ q|
        denom = torch.abs(torch.sum(n * q, dim=-1, keepdim=True))
        denom = torch.clamp(denom, min=1.0)
        output = output / denom

        return output, C, n


class mLSTMCell(nn.Module):
    """Matrix LSTM cell with key-value memory.

    The mLSTM uses a matrix-valued cell state C that stores key-value
    associations, enabling much larger memory capacity than scalar LSTM.

    Equations:
        q_t = W_q @ x_t                     # query
        k_t = (1/sqrt(d)) * W_k @ x_t       # key (normalized)
        v_t = W_v @ x_t                     # value

        i_t = exp(w_i^T @ x_t + b_i)        # exponential input gate (scalar)
        f_t = exp(w_f^T @ x_t + b_f)        # exponential forget gate (scalar)
        o_t = sigmoid(W_o @ x_t + b_o)      # output gate (vector)

        C_t = f_t * C_{t-1} + i_t * v_t @ k_t^T   # matrix memory
        n_t = f_t * n_{t-1} + i_t * k_t           # normalizer

        h_t = o_t * (C_t @ q_t) / max(|n_t^T @ q_t|, 1)

    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden dimension (for output).
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_head_dim = num_heads * head_dim

        # Query, Key, Value projections
        self.q_proj = nn.Linear(input_dim, self.total_head_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, self.total_head_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, self.total_head_dim, bias=False)

        # Gate projections (scalar gates)
        self.i_gate_proj = nn.Linear(input_dim, num_heads, bias=True)
        self.f_gate_proj = nn.Linear(input_dim, num_heads, bias=True)

        # Output gate (vector)
        self.o_gate_proj = nn.Linear(input_dim, self.total_head_dim, bias=True)

        # Output projection
        self.out_proj = nn.Linear(self.total_head_dim, hidden_dim, bias=False)

        # Matrix memory modules (one per head conceptually, but batched)
        self.memory = MatrixMemory(
            hidden_dim=self.total_head_dim,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # Layer norm for output
        self.ln = nn.LayerNorm(self.total_head_dim)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of mLSTM cell.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).
            state: Optional tuple of (C, n) states.
                C: Matrix memory of shape (batch, num_heads, head_dim, head_dim)
                n: Normalizer of shape (batch, num_heads, head_dim)

        Returns:
            Tuple of (output, (C, n) new state).
        """
        batch, seq_len, _ = x.shape

        # Initialize state if not provided
        if state is None:
            C = torch.zeros(
                batch, self.num_heads, self.head_dim, self.head_dim,
                device=x.device, dtype=x.dtype
            )
            n = torch.zeros(
                batch, self.num_heads, self.head_dim,
                device=x.device, dtype=x.dtype
            )
        else:
            C, n = state

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t]  # (batch, input_dim)

            # Compute Q, K, V
            q = self.q_proj(x_t)  # (batch, total_head_dim)
            k = self.k_proj(x_t)
            v = self.v_proj(x_t)

            # Reshape to heads: (batch, num_heads, head_dim)
            q = q.view(batch, self.num_heads, self.head_dim)
            k = k.view(batch, self.num_heads, self.head_dim)
            v = v.view(batch, self.num_heads, self.head_dim)

            # Compute gates
            i_gate = torch.exp(self.i_gate_proj(x_t))  # (batch, num_heads)
            f_gate = torch.exp(self.f_gate_proj(x_t))
            o_gate = torch.sigmoid(self.o_gate_proj(x_t))  # (batch, total_head_dim)
            o_gate = o_gate.view(batch, self.num_heads, self.head_dim)

            # Process each head
            head_outputs = []
            new_C_heads = []
            new_n_heads = []

            for h in range(self.num_heads):
                h_output, new_C_h, new_n_h = self.memory(
                    k=k[:, h],
                    v=v[:, h],
                    q=q[:, h],
                    i_gate=i_gate[:, h],
                    f_gate=f_gate[:, h],
                    C=C[:, h],
                    n=n[:, h],
                )

                # Apply output gate
                h_output = o_gate[:, h] * h_output

                head_outputs.append(h_output)
                new_C_heads.append(new_C_h)
                new_n_heads.append(new_n_h)

            # Combine heads
            output_t = torch.cat(head_outputs, dim=-1)  # (batch, total_head_dim)
            output_t = self.ln(output_t)
            output_t = self.out_proj(output_t)  # (batch, hidden_dim)

            # Update state
            C = torch.stack(new_C_heads, dim=1)  # (batch, num_heads, head_dim, head_dim)
            n = torch.stack(new_n_heads, dim=1)  # (batch, num_heads, head_dim)

            outputs.append(output_t)

        output = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_dim)

        return output, (C, n)

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize state for this cell.

        Args:
            batch_size: Batch size.
            device: Device for tensors.
            dtype: Data type.

        Returns:
            Tuple of (C, n) initial states.
        """
        C = torch.zeros(
            batch_size, self.num_heads, self.head_dim, self.head_dim,
            device=device, dtype=dtype
        )
        n = torch.zeros(
            batch_size, self.num_heads, self.head_dim,
            device=device, dtype=dtype
        )
        return (C, n)


class CausalConv1d(nn.Module):
    """Causal 1D convolution for sLSTM blocks.

    Args:
        in_channels: Number of input channels.
        kernel_size: Convolution kernel size.
        bias: Whether to use bias.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 4,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal convolution.

        Args:
            x: Input tensor of shape (batch, channels, seq_len).

        Returns:
            Output tensor of same shape.
        """
        x = self.conv(x)
        # Trim to make causal
        x = x[..., :-(self.kernel_size - 1)] if self.kernel_size > 1 else x
        return x


class GroupedLinear(nn.Module):
    """Grouped linear layer for efficient multi-head projections.

    Args:
        in_features: Input features.
        out_features: Output features.
        num_groups: Number of groups.
        bias: Whether to use bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

        # Standard linear for now (can be optimized with grouped ops)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply grouped linear.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.linear(x)
