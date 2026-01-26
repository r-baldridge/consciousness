"""
S4D: Diagonal State Space Model

Implements the core SSM computation with diagonal A matrix for efficiency.
This is the foundational layer for the Mamba architecture, providing both
convolutional (training) and recurrent (inference) modes.

Reference:
    "On the Parameterization and Initialization of Diagonal State Space Models"
    https://arxiv.org/abs/2206.11893
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class S4DKernel(nn.Module):
    """
    S4D: Diagonal State Space layer.

    Uses diagonal A matrix for efficiency while maintaining expressiveness.
    Supports both recurrent (inference) and convolutional (training) modes.

    The continuous-time state space model is:
        h'(t) = A h(t) + B x(t)    # State equation
        y(t) = C h(t) + D x(t)     # Output equation

    Discretized using zero-order hold:
        h[k] = A_bar h[k-1] + B_bar x[k]
        y[k] = C h[k] + D x[k]

    where:
        A_bar = exp(dt * A)
        B_bar = (dt * A)^{-1} (exp(dt * A) - I) * dt * B
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",  # "random" or "constant"
        dt_scale: float = 1.0,
    ):
        """
        Args:
            d_model: Model dimension (number of features)
            d_state: State dimension (N in the paper)
            dt_min: Minimum discretization step
            dt_max: Maximum discretization step
            dt_init: How to initialize dt
            dt_scale: Scale factor for dt
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # A: diagonal matrix (stored as vector for efficiency)
        # Initialize with HiPPO-LegS diagonal approximation: A[n] = -(n + 1/2)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = -0.5 * A  # HiPPO diagonal approximation
        self.register_buffer("A", A)  # [d_state]

        # Learnable discretization step (log scale for stability)
        if dt_init == "random":
            log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        else:
            log_dt = torch.full((d_model,), math.log((dt_min + dt_max) / 2))
        self.log_dt = nn.Parameter(log_dt * dt_scale)

        # B and C: learnable projection matrices
        # Initialize with small random values
        self.B = nn.Parameter(torch.randn(d_model, d_state) * (1.0 / d_state ** 0.5))
        self.C = nn.Parameter(torch.randn(d_model, d_state) * (1.0 / d_state ** 0.5))

        # D: skip connection (feedthrough)
        self.D = nn.Parameter(torch.ones(d_model))

        # Cache for discretized parameters
        self._cache = {}

    def _discretize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discretize continuous parameters using zero-order hold.

        Returns:
            A_bar: [d_model, d_state] discretized state matrix
            B_bar: [d_model, d_state] discretized input matrix
        """
        dt = torch.exp(self.log_dt)  # [d_model]

        # A_bar = exp(dt * A)
        # For diagonal A, this is element-wise: exp(dt_i * A_n) for each (i, n)
        dtA = dt.unsqueeze(-1) * self.A.unsqueeze(0)  # [d_model, d_state]
        A_bar = torch.exp(dtA)

        # B_bar = (A)^{-1} (A_bar - I) B
        # For diagonal: B_bar = (exp(dtA) - 1) / A * B
        # Numerically stable version using expm1
        B_bar = (torch.expm1(dtA) / self.A.unsqueeze(0)) * self.B  # [d_model, d_state]

        return A_bar, B_bar

    def forward_recurrent(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recurrent mode: O(L) sequential, O(1) per step.
        Best for inference.

        Args:
            x: [batch, d_model] single time step input
            h: [batch, d_model, d_state] hidden state (optional)

        Returns:
            y: [batch, d_model] output
            h_new: [batch, d_model, d_state] new hidden state
        """
        batch = x.shape[0]
        device = x.device

        # Initialize hidden state if not provided
        if h is None:
            h = torch.zeros(batch, self.d_model, self.d_state, device=device, dtype=x.dtype)

        # Get discretized parameters
        A_bar, B_bar = self._discretize()

        # State update: h_new = A_bar * h + B_bar * x
        # h: [batch, d_model, d_state]
        # A_bar: [d_model, d_state]
        # B_bar: [d_model, d_state]
        # x: [batch, d_model]
        h_new = A_bar.unsqueeze(0) * h + B_bar.unsqueeze(0) * x.unsqueeze(-1)

        # Output: y = C @ h + D * x
        # C: [d_model, d_state]
        y = torch.einsum('bdn,dn->bd', h_new, self.C) + self.D * x

        return y, h_new

    def forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convolutional mode: O(L log L) parallel via FFT.
        Best for training.

        Args:
            x: [batch, length, d_model]

        Returns:
            y: [batch, length, d_model]
        """
        batch, length, d_model = x.shape

        # Compute convolution kernel
        kernel = self._compute_kernel(length)  # [d_model, length]

        # Transpose for convolution: [batch, d_model, length]
        x_t = x.transpose(1, 2)

        # FFT convolution (circular, need to handle properly)
        # Pad to avoid circular convolution artifacts
        fft_size = 2 * length

        x_f = torch.fft.rfft(x_t, n=fft_size)  # [batch, d_model, fft_size//2+1]
        k_f = torch.fft.rfft(kernel, n=fft_size)  # [d_model, fft_size//2+1]

        # Multiply in frequency domain
        y_f = x_f * k_f.unsqueeze(0)

        # Inverse FFT and truncate
        y = torch.fft.irfft(y_f, n=fft_size)[..., :length]  # [batch, d_model, length]

        # Add skip connection
        y = y + self.D.view(1, -1, 1) * x_t

        return y.transpose(1, 2)  # [batch, length, d_model]

    def _compute_kernel(self, length: int) -> torch.Tensor:
        """
        Compute convolution kernel K.

        K[k] = C @ A_bar^k @ B_bar for k = 0, 1, ..., L-1

        For diagonal A_bar, this simplifies to:
        K[k] = sum_n C[n] * A_bar[n]^k * B_bar[n]

        Args:
            length: Sequence length

        Returns:
            kernel: [d_model, length] convolution kernel
        """
        A_bar, B_bar = self._discretize()  # [d_model, d_state]

        # Compute powers of A_bar efficiently
        # A_bar^k for k = 0, 1, ..., L-1
        # Shape: [d_model, d_state, length]
        k_range = torch.arange(length, device=A_bar.device, dtype=A_bar.dtype)

        # A_bar^k = exp(k * log(A_bar))
        # For stability with values in (0, 1), use log-space computation
        # powers[d, n, k] = A_bar[d, n]^k
        log_A_bar = torch.log(A_bar.clamp(min=1e-10))
        powers = torch.exp(log_A_bar.unsqueeze(-1) * k_range.unsqueeze(0).unsqueeze(0))

        # K[d, k] = sum_n C[d, n] * powers[d, n, k] * B_bar[d, n]
        # = sum_n (C[d, n] * B_bar[d, n]) * powers[d, n, k]
        CB = self.C * B_bar  # [d_model, d_state]
        kernel = torch.einsum('dn,dnk->dk', CB, powers)  # [d_model, length]

        return kernel

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        mode: str = "auto"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with automatic mode selection.

        Args:
            x: Input tensor
               - If [batch, d_model]: single step (recurrent)
               - If [batch, length, d_model]: sequence (conv or recurrent)
            h: Hidden state for recurrent mode
            mode: "auto", "conv", or "recurrent"

        Returns:
            y: Output tensor (same shape as x)
            h: Hidden state (only for recurrent mode)
        """
        if x.dim() == 2:
            # Single step: always recurrent
            return self.forward_recurrent(x, h)

        # Sequence input
        if mode == "auto":
            # Use conv for training (parallel), recurrent for inference
            mode = "conv" if self.training else "recurrent"

        if mode == "conv":
            y = self.forward_conv(x)
            return y, None
        else:
            # Sequential recurrent
            outputs = []
            for t in range(x.shape[1]):
                y_t, h = self.forward_recurrent(x[:, t], h)
                outputs.append(y_t)
            return torch.stack(outputs, dim=1), h


class S4DLayer(nn.Module):
    """
    Full S4D layer with normalization and residual connection.

    This wraps the S4DKernel with layer normalization, dropout,
    and optional bidirectional processing for use in deep networks.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        bidirectional: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
    ):
        """
        Args:
            d_model: Model dimension
            d_state: State dimension
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional processing
            dt_min: Minimum discretization step
            dt_max: Maximum discretization step
            dt_init: How to initialize dt
            dt_scale: Scale factor for dt
        """
        super().__init__()
        self.d_model = d_model
        self.bidirectional = bidirectional

        # Forward SSM
        self.ssm = S4DKernel(
            d_model, d_state,
            dt_min=dt_min, dt_max=dt_max,
            dt_init=dt_init, dt_scale=dt_scale
        )

        # Backward SSM for bidirectional
        if bidirectional:
            self.ssm_back = S4DKernel(
                d_model, d_state,
                dt_min=dt_min, dt_max=dt_max,
                dt_init=dt_init, dt_scale=dt_scale
            )
            self.mix = nn.Linear(2 * d_model, d_model)

        # Normalization and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mode: str = "auto") -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: [batch, length, d_model]
            mode: "auto", "conv", or "recurrent"

        Returns:
            y: [batch, length, d_model]
        """
        residual = x
        x = self.norm(x)

        # Forward pass
        y_fwd, _ = self.ssm(x, mode=mode)

        if self.bidirectional:
            # Backward pass (flip, process, flip back)
            x_flip = torch.flip(x, dims=[1])
            y_bwd, _ = self.ssm_back(x_flip, mode=mode)
            y_bwd = torch.flip(y_bwd, dims=[1])

            # Mix forward and backward
            y = self.mix(torch.cat([y_fwd, y_bwd], dim=-1))
        else:
            y = y_fwd

        return residual + self.dropout(y)
