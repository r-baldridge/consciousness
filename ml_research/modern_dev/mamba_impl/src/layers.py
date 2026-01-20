"""
Mamba Architecture-Specific Layers

This module implements the core layers for the Mamba selective state space model:
- SelectiveSSM: The main selective state space module
- S6Layer: Structured state space layer (S6)
- CausalConv1d: Causal 1D convolution
- Discretization: Methods for discretizing continuous-time SSM
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discretization:
    """Discretization methods for continuous-time SSM.

    Converts continuous-time parameters (A, B) to discrete-time (A_bar, B_bar)
    using various discretization schemes.
    """

    @staticmethod
    def zero_order_hold(
        A: torch.Tensor,
        B: torch.Tensor,
        delta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero-order hold (ZOH) discretization.

        The standard discretization method used in Mamba:
            A_bar = exp(delta * A)
            B_bar = (delta * A)^{-1} * (exp(delta * A) - I) * delta * B
                  approx delta * B  (for small delta)

        Args:
            A: State matrix of shape (..., d_state).
            B: Input matrix of shape (..., d_state).
            delta: Discretization step of shape (...).

        Returns:
            Tuple of (A_bar, B_bar) in discrete time.
        """
        # A_bar = exp(delta * A)
        # For diagonal A (which Mamba uses), this is element-wise
        delta_A = delta.unsqueeze(-1) * A  # (..., d_state)
        A_bar = torch.exp(delta_A)

        # Simplified B_bar = delta * B (first-order approximation)
        # This is valid when delta is small
        B_bar = delta.unsqueeze(-1) * B

        return A_bar, B_bar

    @staticmethod
    def bilinear(
        A: torch.Tensor,
        B: torch.Tensor,
        delta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bilinear (Tustin) discretization.

        Alternative discretization method with better frequency response:
            A_bar = (I + delta/2 * A) * (I - delta/2 * A)^{-1}
            B_bar = delta * (I - delta/2 * A)^{-1} * B

        Args:
            A: State matrix of shape (..., d_state).
            B: Input matrix of shape (..., d_state).
            delta: Discretization step of shape (...).

        Returns:
            Tuple of (A_bar, B_bar) in discrete time.
        """
        # For diagonal A (element-wise operations)
        delta_A_half = 0.5 * delta.unsqueeze(-1) * A

        # (1 + delta/2 * A) / (1 - delta/2 * A)
        A_bar = (1 + delta_A_half) / (1 - delta_A_half)

        # delta / (1 - delta/2 * A) * B
        B_bar = delta.unsqueeze(-1) * B / (1 - delta_A_half)

        return A_bar, B_bar


class CausalConv1d(nn.Module):
    """Causal 1D convolution layer.

    Implements causal convolution where output at position t only depends
    on inputs at positions <= t.

    Args:
        in_channels: Number of input channels.
        kernel_size: Size of the convolution kernel.
        bias: Whether to include a bias term.
        groups: Number of groups for grouped convolution.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        bias: bool = True,
        groups: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # Use groups=in_channels for depthwise conv (as in Mamba)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=kernel_size - 1,  # Will trim to make causal
            groups=groups if groups > 1 else in_channels,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        inference_params: Optional[dict] = None,
    ) -> torch.Tensor:
        """Apply causal convolution.

        Args:
            x: Input tensor of shape (batch, channels, seq_len).
            inference_params: Optional dict containing conv_state for caching.

        Returns:
            Output tensor of shape (batch, channels, seq_len).
        """
        if inference_params is not None and "conv_state" in inference_params:
            # Inference mode: use cached state
            conv_state = inference_params["conv_state"]
            # Update state with new input
            conv_state = torch.cat([conv_state[..., 1:], x[..., :1]], dim=-1)
            inference_params["conv_state"] = conv_state
            # Apply conv to state
            x = F.conv1d(
                conv_state,
                self.conv.weight,
                self.conv.bias,
                groups=self.conv.groups,
            )
            return x

        # Training mode: standard causal conv
        x = self.conv(x)
        # Remove future padding (keep only causal part)
        x = x[..., :-(self.kernel_size - 1)] if self.kernel_size > 1 else x
        return x


class S6Layer(nn.Module):
    """Structured State Space (S6) layer.

    Implements the S6 (simplified S4) layer with diagonal state matrix.
    This is the non-selective version used as a building block.

    Args:
        d_model: Model dimension.
        d_state: State space dimension.
        bidirectional: Whether to use bidirectional processing.
        discretization: Discretization method ("zoh" or "bilinear").
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        bidirectional: bool = False,
        discretization: str = "zoh",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.bidirectional = bidirectional
        self.discretization = discretization

        # State matrix A (diagonal, learnable)
        # Initialized with HiPPO-inspired values
        self.A_log = nn.Parameter(self._init_A_log(d_model, d_state))

        # Input projection B
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.02)

        # Output projection C
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.02)

        # Skip connection D
        self.D = nn.Parameter(torch.ones(d_model))

        # Delta (discretization step) - fixed in S6
        self.delta = nn.Parameter(torch.ones(d_model) * 0.01)

    def _init_A_log(self, d_model: int, d_state: int) -> torch.Tensor:
        """Initialize A in log space using HiPPO-inspired initialization.

        Args:
            d_model: Model dimension.
            d_state: State dimension.

        Returns:
            Initialized A_log tensor.
        """
        # HiPPO matrix initialization
        # A[i] = -(i + 1) for i in 0..d_state-1
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(d_model, -1)  # (d_model, d_state)
        A_log = torch.log(A)
        return A_log

    @property
    def A(self) -> torch.Tensor:
        """Get A from log space."""
        return -torch.exp(self.A_log)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of S6 layer.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        batch, seq_len, d_model = x.shape

        # Get discretized parameters
        A = self.A  # (d_model, d_state)
        B = self.B  # (d_model, d_state)
        C = self.C  # (d_model, d_state)

        # Discretize
        if self.discretization == "zoh":
            A_bar, B_bar = Discretization.zero_order_hold(A, B, self.delta)
        else:
            A_bar, B_bar = Discretization.bilinear(A, B, self.delta)

        # Run SSM (placeholder - use parallel scan in production)
        y = self._ssm_scan(x, A_bar, B_bar, C)

        # Add skip connection
        y = y + self.D * x

        return y

    def _ssm_scan(
        self,
        x: torch.Tensor,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Sequential SSM scan (placeholder for parallel scan).

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            A_bar: Discretized state matrix.
            B_bar: Discretized input matrix.
            C: Output matrix.

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        batch, seq_len, d_model = x.shape

        # Initialize state
        h = torch.zeros(batch, d_model, self.d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, d_model)

            # State update: h = A_bar * h + B_bar * x
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)

            # Output: y = C * h
            y_t = (C * h).sum(dim=-1)  # (batch, d_model)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        return y


class SelectiveSSM(nn.Module):
    """Selective State Space Model (the core of Mamba).

    Implements the selective SSM where B, C, and delta are input-dependent,
    allowing the model to selectively propagate or forget information.

    Args:
        d_model: Model dimension (inner dimension in Mamba block).
        d_state: State space dimension.
        dt_rank: Rank for delta projection.
        dt_min: Minimum delta value.
        dt_max: Maximum delta value.
        dt_init: Delta initialization method.
        dt_scale: Delta scaling factor.
        dt_init_floor: Floor value for delta initialization.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_rank: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor

        # A matrix (diagonal, in log space)
        # Not input-dependent, but learnable
        self.A_log = nn.Parameter(self._init_A_log())

        # Skip connection D
        self.D = nn.Parameter(torch.ones(d_model))

        # Input-dependent projections
        # Project x to B, C, and dt
        self.x_proj = nn.Linear(d_model, dt_rank + 2 * d_state, bias=False)

        # Project dt from low rank to full dimension
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        # Initialize dt_proj bias
        self._init_dt_proj_bias()

    def _init_A_log(self) -> torch.Tensor:
        """Initialize A in log space."""
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_model, -1)
        return torch.log(A)

    def _init_dt_proj_bias(self):
        """Initialize dt projection bias for proper dt range."""
        dt = torch.exp(
            torch.rand(self.d_model) * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        ).clamp(min=self.dt_init_floor)

        # Inverse of softplus to get bias
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    @property
    def A(self) -> torch.Tensor:
        """Get A from log space."""
        return -torch.exp(self.A_log)

    def forward(
        self,
        x: torch.Tensor,
        inference_params: Optional[dict] = None,
    ) -> torch.Tensor:
        """Forward pass of selective SSM.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            inference_params: Optional dict for caching during generation.

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        batch, seq_len, d_model = x.shape

        # Get A
        A = self.A  # (d_model, d_state)

        # Input-dependent projections
        x_proj = self.x_proj(x)  # (batch, seq_len, dt_rank + 2*d_state)

        # Split projections
        dt_proj = x_proj[..., :self.dt_rank]  # (batch, seq_len, dt_rank)
        B = x_proj[..., self.dt_rank:self.dt_rank + self.d_state]  # (batch, seq_len, d_state)
        C = x_proj[..., self.dt_rank + self.d_state:]  # (batch, seq_len, d_state)

        # Project dt to full dimension and apply softplus
        dt = F.softplus(self.dt_proj(dt_proj))  # (batch, seq_len, d_model)

        # Run selective scan
        y = self._selective_scan(x, dt, A, B, C, inference_params)

        # Add skip connection
        y = y + self.D * x

        return y

    def _selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        inference_params: Optional[dict] = None,
    ) -> torch.Tensor:
        """Selective scan implementation.

        This is a placeholder sequential implementation.
        In production, use a parallel scan (associative scan) or fused CUDA kernel.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            dt: Delta tensor of shape (batch, seq_len, d_model).
            A: State matrix of shape (d_model, d_state).
            B: Input projection of shape (batch, seq_len, d_state).
            C: Output projection of shape (batch, seq_len, d_state).
            inference_params: Optional dict for caching.

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        batch, seq_len, d_model = x.shape

        # Get or initialize state
        if inference_params is not None and "ssm_state" in inference_params:
            h = inference_params["ssm_state"]
        else:
            h = torch.zeros(
                batch, d_model, self.d_state,
                device=x.device, dtype=x.dtype
            )

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, d_model)
            dt_t = dt[:, t, :]  # (batch, d_model)
            B_t = B[:, t, :]  # (batch, d_state)
            C_t = C[:, t, :]  # (batch, d_state)

            # Discretize (ZOH)
            # A_bar = exp(dt * A), B_bar = dt * B
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A)  # (batch, d_model, d_state)
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (batch, d_model, d_state)

            # State update: h = A_bar * h + B_bar * x
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)

            # Output: y = C * h (sum over state dimension)
            y_t = (C_t.unsqueeze(1) * h).sum(dim=-1)  # (batch, d_model)
            outputs.append(y_t)

        # Update cache if in inference mode
        if inference_params is not None:
            inference_params["ssm_state"] = h

        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        return y


class ParallelScan(nn.Module):
    """Parallel (associative) scan for efficient SSM computation.

    Implements the associative scan operation that enables O(log n) parallel
    computation of the SSM recurrence.

    The associative operation is:
        (a1, b1) * (a2, b2) = (a1 * a2, a2 * b1 + b2)

    This is a placeholder - production implementation would use Triton/CUDA.
    """

    @staticmethod
    def forward(
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute parallel scan.

        Given recurrence h[t] = a[t] * h[t-1] + b[t],
        compute all h[t] in parallel.

        Args:
            a: Multiplicative coefficients of shape (batch, seq_len, d_model, d_state).
            b: Additive coefficients of shape (batch, seq_len, d_model, d_state).

        Returns:
            Output tensor of shape (batch, seq_len, d_model, d_state).
        """
        # Placeholder: use sequential scan
        # In production, implement associative scan using work-efficient algorithm
        batch, seq_len, d_model, d_state = a.shape

        h = torch.zeros(batch, d_model, d_state, device=a.device, dtype=a.dtype)
        outputs = []

        for t in range(seq_len):
            h = a[:, t] * h + b[:, t]
            outputs.append(h)

        return torch.stack(outputs, dim=1)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Alternative to LayerNorm that only normalizes by RMS without centering.
    Often used in modern architectures for efficiency.

    Args:
        d_model: Dimension to normalize.
        eps: Epsilon for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Normalized tensor of same shape.
        """
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms
