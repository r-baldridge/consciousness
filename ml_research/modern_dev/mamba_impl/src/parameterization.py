"""
State Space Model Parameterization

Provides initialization schemes and numerical utilities for SSMs.
Includes HiPPO (High-order Polynomial Projection Operators) initialization
which enables long-range dependency modeling in state space models.

References:
    "HiPPO: Recurrent Memory with Optimal Polynomial Projections"
    https://arxiv.org/abs/2008.07669

    "On the Parameterization and Initialization of Diagonal State Space Models"
    https://arxiv.org/abs/2206.11893
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


class HiPPO:
    """
    HiPPO (High-order Polynomial Projection Operators) initialization.

    Provides principled initialization for SSM matrices that enables
    long-range dependency modeling. The key insight is that certain
    polynomial bases can efficiently approximate sliding window memory.
    """

    @staticmethod
    def legs(N: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        HiPPO-LegS: Scaled Legendre polynomials.

        Best for general sequence modeling. Projects onto a basis that
        optimally approximates uniform weighting over the history.

        A[n,k] = -sqrt(2n+1) * sqrt(2k+1)  if n > k
               = -(n+1)                     if n == k
               = 0                          otherwise

        Args:
            N: State dimension
            dtype: Data type for the tensor

        Returns:
            A: [N, N] HiPPO-LegS matrix
        """
        A = torch.zeros(N, N, dtype=dtype)
        for n in range(N):
            for k in range(n + 1):
                if n > k:
                    A[n, k] = -math.sqrt(2 * n + 1) * math.sqrt(2 * k + 1)
                elif n == k:
                    A[n, k] = -(n + 1)
        return A

    @staticmethod
    def legs_diagonal(N: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Diagonal approximation of HiPPO-LegS.

        A[n] = -(n + 1/2) for n = 0, 1, ..., N-1

        This is what S4D uses. The diagonal approximation maintains
        the essential memory properties while enabling efficient computation.

        Args:
            N: State dimension
            dtype: Data type for the tensor

        Returns:
            A: [N] diagonal HiPPO matrix
        """
        return -(torch.arange(N, dtype=dtype) + 0.5)

    @staticmethod
    def legt(N: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        HiPPO-LegT: Translated Legendre polynomials.

        Better for tasks with strong recency bias. Projects onto a basis
        that emphasizes recent history more strongly.

        Args:
            N: State dimension
            dtype: Data type for the tensor

        Returns:
            A: [N, N] HiPPO-LegT matrix
        """
        A = torch.zeros(N, N, dtype=dtype)
        for n in range(N):
            for k in range(n + 1):
                if n > k:
                    A[n, k] = (2 * n + 1) ** 0.5 * (2 * k + 1) ** 0.5
                else:
                    A[n, k] = n + 1
        return -A

    @staticmethod
    def lmu(N: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        LMU (Legendre Memory Unit) initialization.

        A[n,k] = (2n+1)^{1/2} (2k+1)^{1/2}  if n >= k

        Args:
            N: State dimension
            dtype: Data type for the tensor

        Returns:
            A: [N, N] LMU matrix
        """
        A = torch.zeros(N, N, dtype=dtype)
        for n in range(N):
            for k in range(n + 1):
                A[n, k] = (2 * n + 1) ** 0.5 * (2 * k + 1) ** 0.5
        return -A

    @staticmethod
    def fourier(N: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Fourier basis initialization.

        Uses complex exponentials for frequency-based memory.
        Good for periodic or quasi-periodic sequences.

        Args:
            N: State dimension (should be even)
            dtype: Data type for the tensor

        Returns:
            A: [N] diagonal matrix (imaginary frequencies)
        """
        # Pairs of complex conjugate eigenvalues
        freqs = torch.arange(N // 2, dtype=dtype) + 1
        # Return as diagonal (real part is 0, imaginary is the frequency)
        A_real = torch.zeros(N, dtype=dtype)
        A_real[::2] = -freqs  # Decay terms
        return A_real

    @staticmethod
    def random_diagonal(
        N: int,
        min_val: float = -1.0,
        max_val: float = -0.1,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Random diagonal initialization with bounded negative values.

        Args:
            N: State dimension
            min_val: Minimum value (most negative)
            max_val: Maximum value (least negative, should be < 0)
            dtype: Data type for the tensor

        Returns:
            A: [N] random diagonal matrix
        """
        return torch.rand(N, dtype=dtype) * (max_val - min_val) + min_val


class SSMInit:
    """
    Initialization utilities for SSM parameters.
    """

    @staticmethod
    def init_dt(
        d_model: int,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Initialize discretization step.

        The discretization step (delta) controls how the continuous-time
        SSM is converted to discrete time. Larger delta means the model
        responds more strongly to inputs (less filtering).

        Args:
            d_model: Number of features
            dt_min: Minimum dt value
            dt_max: Maximum dt value
            dt_init: "random" or "constant"
            dtype: Data type for the tensor

        Returns:
            log_dt: [d_model] log-scale discretization steps
        """
        if dt_init == "random":
            log_dt = torch.rand(d_model, dtype=dtype) * (
                math.log(dt_max) - math.log(dt_min)
            ) + math.log(dt_min)
        elif dt_init == "constant":
            log_dt = torch.full((d_model,), math.log((dt_min + dt_max) / 2), dtype=dtype)
        else:
            raise ValueError(f"Unknown dt_init: {dt_init}")

        return log_dt

    @staticmethod
    def init_BC(
        d_model: int,
        d_state: int,
        init: str = "normal",
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize B and C matrices.

        B is the input projection matrix, C is the output projection matrix.
        Their initialization affects how information flows through the SSM.

        Args:
            d_model: Number of features
            d_state: State dimension
            init: "normal", "uniform", or "ones"
            dtype: Data type for the tensor

        Returns:
            B: [d_model, d_state]
            C: [d_model, d_state]
        """
        if init == "normal":
            std = 1.0 / d_state ** 0.5
            B = torch.randn(d_model, d_state, dtype=dtype) * std
            C = torch.randn(d_model, d_state, dtype=dtype) * std
        elif init == "uniform":
            bound = 1.0 / d_state ** 0.5
            B = torch.empty(d_model, d_state, dtype=dtype).uniform_(-bound, bound)
            C = torch.empty(d_model, d_state, dtype=dtype).uniform_(-bound, bound)
        elif init == "ones":
            B = torch.ones(d_model, d_state, dtype=dtype) / d_state
            C = torch.ones(d_model, d_state, dtype=dtype) / d_state
        else:
            raise ValueError(f"Unknown init: {init}")

        return B, C

    @staticmethod
    def init_D(
        d_model: int,
        init: str = "ones",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Initialize skip connection D.

        D is the direct feedthrough term that adds the input directly
        to the output, bypassing the state dynamics.

        Args:
            d_model: Number of features
            init: "ones", "zeros", or "normal"
            dtype: Data type for the tensor

        Returns:
            D: [d_model] skip connection weights
        """
        if init == "ones":
            return torch.ones(d_model, dtype=dtype)
        elif init == "zeros":
            return torch.zeros(d_model, dtype=dtype)
        elif init == "normal":
            return torch.randn(d_model, dtype=dtype) * 0.02
        else:
            raise ValueError(f"Unknown init: {init}")


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.

    More stable than LayerNorm for SSMs, and computationally cheaper
    since it doesn't require computing the mean.

    RMSNorm(x) = x / RMS(x) * weight
    where RMS(x) = sqrt(mean(x^2))
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Args:
            d_model: Dimension to normalize
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Normalized tensor of same shape
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


class ComplexSSMUtils:
    """
    Utilities for complex-valued SSMs.

    Some SSM variants (like S4) use complex eigenvalues for the state matrix.
    These utilities help with complex arithmetic in a numerically stable way.
    """

    @staticmethod
    def complex_exp(x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute exp(x) for complex x = x_real + i*x_imag.

        exp(a + bi) = exp(a) * (cos(b) + i*sin(b))

        Args:
            x_real: Real part
            x_imag: Imaginary part

        Returns:
            Tuple of (result_real, result_imag)
        """
        exp_real = torch.exp(x_real)
        result_real = exp_real * torch.cos(x_imag)
        result_imag = exp_real * torch.sin(x_imag)
        return result_real, result_imag

    @staticmethod
    def complex_mul(
        a_real: torch.Tensor, a_imag: torch.Tensor,
        b_real: torch.Tensor, b_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute a * b for complex a and b.

        (a_r + i*a_i) * (b_r + i*b_i) = (a_r*b_r - a_i*b_i) + i*(a_r*b_i + a_i*b_r)

        Args:
            a_real, a_imag: Complex number a
            b_real, b_imag: Complex number b

        Returns:
            Tuple of (result_real, result_imag)
        """
        result_real = a_real * b_real - a_imag * b_imag
        result_imag = a_real * b_imag + a_imag * b_real
        return result_real, result_imag

    @staticmethod
    def cauchy_kernel(
        v: torch.Tensor,
        z: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Cauchy kernel: sum_i v_i / (z - w_i).

        This is used in S4 for efficient kernel computation.

        Args:
            v: Numerator coefficients [N]
            z: Query points [L] (complex)
            w: Poles [N] (complex)

        Returns:
            Kernel values [L]
        """
        # v: [N], z: [L], w: [N]
        # Result: sum over N of v[n] / (z[l] - w[n])
        # Shape: [L, N] -> sum -> [L]
        denom = z.unsqueeze(-1) - w.unsqueeze(0)  # [L, N]
        return (v.unsqueeze(0) / denom).sum(dim=-1)  # [L]
