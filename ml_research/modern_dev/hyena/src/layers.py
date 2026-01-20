"""
Hyena Layers - Architecture-Specific Layer Implementations

This module contains the core layer components for Hyena:
- FFTConv: Efficient FFT-based convolution
- FlashFFTConv: Optimized FFT convolution (placeholder)
- ExponentialWindow: Exponential decay window function
- ShortConvolution: Local convolution for gating
- HyenaFilter: Combined filter with implicit parameterization
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTConv(nn.Module):
    """FFT-based convolution for efficient long-range operations.

    Implements convolution via the convolution theorem:
        h * x = IFFT(FFT(h) * FFT(x))

    This achieves O(L log L) complexity vs O(L^2) for naive convolution.
    """

    def __init__(
        self,
        d_model: int,
        bidirectional: bool = False,
        use_bias: bool = True,
    ):
        """Initialize FFT convolution.

        Args:
            d_model: Model dimension.
            bidirectional: Whether to apply bidirectional convolution.
            use_bias: Whether to include a learnable bias.
        """
        super().__init__()
        self.d_model = d_model
        self.bidirectional = bidirectional

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        fft_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply FFT convolution.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            h: Filter tensor of shape (seq_len, d_model) or (filter_len, d_model).
            fft_size: Optional FFT size (defaults to next power of 2).

        Returns:
            Convolved tensor of shape (batch, seq_len, d_model).
        """
        batch_size, seq_len, d_model = x.shape
        filter_len = h.shape[0]

        # Determine FFT size
        if fft_size is None:
            fft_size = 2 ** (seq_len + filter_len - 1).bit_length()

        # Compute FFTs
        x_f = torch.fft.rfft(x, n=fft_size, dim=1)
        h_f = torch.fft.rfft(h, n=fft_size, dim=0)

        # Multiply in frequency domain
        y_f = x_f * h_f.unsqueeze(0)

        # Inverse FFT
        y = torch.fft.irfft(y_f, n=fft_size, dim=1)

        # Truncate and add bias
        y = y[:, :seq_len, :]

        if self.bias is not None:
            y = y + self.bias

        return y


class FlashFFTConv(nn.Module):
    """Optimized FFT convolution using fused CUDA kernels.

    This is a placeholder for the optimized implementation that would
    use custom CUDA kernels for better performance. Falls back to
    standard FFTConv when CUDA kernels are not available.
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 8192,
        bidirectional: bool = False,
    ):
        """Initialize Flash FFT convolution.

        Args:
            d_model: Model dimension.
            max_seq_len: Maximum sequence length for precomputation.
            bidirectional: Whether to apply bidirectional convolution.
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.bidirectional = bidirectional

        # Fallback to standard FFTConv
        self.fft_conv = FFTConv(d_model, bidirectional)

        # Placeholder for precomputed twiddle factors
        self._cuda_available = False

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Apply optimized FFT convolution.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            h: Filter tensor of shape (filter_len, d_model).

        Returns:
            Convolved tensor of shape (batch, seq_len, d_model).
        """
        # TODO: Implement CUDA kernel version
        # For now, fall back to standard FFT
        return self.fft_conv(x, h)


class ExponentialWindow(nn.Module):
    """Exponential decay window function for filter stability.

    Applies an exponential decay to convolution filters to ensure
    stability and prevent gradient explosion in long sequences.

        window(t) = exp(-alpha * t)
    """

    def __init__(
        self,
        d_model: int,
        learnable: bool = True,
        init_decay: float = 0.5,
    ):
        """Initialize exponential window.

        Args:
            d_model: Model dimension.
            learnable: Whether decay is learnable.
            init_decay: Initial decay rate.
        """
        super().__init__()
        self.d_model = d_model

        if learnable:
            self.log_decay = nn.Parameter(
                torch.full((d_model,), torch.log(torch.tensor(init_decay)))
            )
        else:
            self.register_buffer(
                "log_decay",
                torch.full((d_model,), torch.log(torch.tensor(init_decay)))
            )

    @property
    def decay(self) -> torch.Tensor:
        """Get decay rates (always positive)."""
        return self.log_decay.exp()

    def forward(
        self,
        filter: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply exponential window to filter.

        Args:
            filter: Filter tensor of shape (filter_len, d_model).
            seq_len: Sequence length for normalization (default: filter_len).

        Returns:
            Windowed filter tensor.
        """
        filter_len = filter.shape[0]
        if seq_len is None:
            seq_len = filter_len

        # Create position indices normalized to [0, seq_len]
        t = torch.arange(filter_len, device=filter.device, dtype=filter.dtype)

        # Compute window
        window = torch.exp(-self.decay.unsqueeze(0) * t.unsqueeze(1))

        return filter * window


class ShortConvolution(nn.Module):
    """Short convolution for local context in gating mechanism.

    A depthwise separable convolution applied over short local windows
    to capture fine-grained local patterns before gating.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 3,
        groups: Optional[int] = None,
        use_bias: bool = True,
        causal: bool = True,
    ):
        """Initialize short convolution.

        Args:
            d_model: Model dimension.
            kernel_size: Convolution kernel size.
            groups: Number of groups (default: d_model for depthwise).
            use_bias: Whether to use bias.
            causal: Whether to use causal (left) padding.
        """
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.causal = causal

        if groups is None:
            groups = d_model  # Depthwise

        # Padding for causal or symmetric
        if causal:
            self.padding = kernel_size - 1
        else:
            self.padding = kernel_size // 2

        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=self.padding,
            groups=groups,
            bias=use_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply short convolution.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Transpose to (batch, d_model, seq_len) for conv1d
        x = x.transpose(1, 2)

        # Apply convolution
        y = self.conv(x)

        # Truncate if causal to maintain causality
        if self.causal:
            y = y[:, :, :-self.padding] if self.padding > 0 else y

        # Transpose back to (batch, seq_len, d_model)
        return y.transpose(1, 2)


class HyenaFilter(nn.Module):
    """Complete Hyena filter combining implicit parameterization with windowing.

    This module integrates:
    1. Positional encoding generation
    2. MLP-based filter synthesis
    3. Exponential windowing for stability
    """

    def __init__(
        self,
        d_model: int,
        filter_order: int = 64,
        emb_dim: int = 3,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        learnable_frequencies: bool = True,
        learnable_decay: bool = True,
    ):
        """Initialize Hyena filter.

        Args:
            d_model: Output filter dimension.
            filter_order: Hidden dimension of filter MLP.
            emb_dim: Number of frequency components for positional encoding.
            max_seq_len: Maximum sequence length.
            dropout: Dropout probability in filter MLP.
            learnable_frequencies: Whether positional frequencies are learnable.
            learnable_decay: Whether window decay is learnable.
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Positional encoding input dimension: sin + cos + position
        pos_dim = 2 * emb_dim + 1

        # Learnable frequencies
        if learnable_frequencies:
            self.frequencies = nn.Parameter(torch.randn(emb_dim) * 0.01)
        else:
            freqs = torch.exp(torch.linspace(0, -4, emb_dim))
            self.register_buffer("frequencies", freqs)

        # Filter MLP
        self.filter_mlp = nn.Sequential(
            nn.Linear(pos_dim, filter_order),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(filter_order, filter_order),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(filter_order, d_model),
        )

        # Window function
        self.window = ExponentialWindow(d_model, learnable=learnable_decay)

        # Cache for generated filters
        self._filter_cache: Optional[Tuple[int, torch.Tensor]] = None

    def _get_positional_encoding(self, seq_len: int) -> torch.Tensor:
        """Generate positional encoding for filter synthesis.

        Args:
            seq_len: Sequence length.

        Returns:
            Positional encoding of shape (seq_len, pos_dim).
        """
        device = self.frequencies.device
        dtype = self.frequencies.dtype

        # Normalized positions [0, 1]
        t = torch.linspace(0, 1, seq_len, device=device, dtype=dtype)

        # Compute angles
        angles = 2 * torch.pi * self.frequencies.unsqueeze(0) * t.unsqueeze(1)

        # Concatenate [sin, cos, position]
        encoding = torch.cat([
            torch.sin(angles),
            torch.cos(angles),
            t.unsqueeze(1),
        ], dim=-1)

        return encoding

    def forward(
        self,
        seq_len: int,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Generate filter for given sequence length.

        Args:
            seq_len: Desired filter length.
            use_cache: Whether to use cached filter if available.

        Returns:
            Filter tensor of shape (seq_len, d_model).
        """
        # Check cache
        if use_cache and self._filter_cache is not None:
            cached_len, cached_filter = self._filter_cache
            if cached_len == seq_len:
                return cached_filter

        # Generate positional encoding
        pos_enc = self._get_positional_encoding(seq_len)

        # Generate filter through MLP
        h = self.filter_mlp(pos_enc)

        # Apply window
        h = self.window(h)

        # Cache if training
        if use_cache and self.training:
            self._filter_cache = (seq_len, h.detach())

        return h

    def clear_cache(self) -> None:
        """Clear the filter cache."""
        self._filter_cache = None


class GatedMLP(nn.Module):
    """Gated MLP (SwiGLU variant) used in Hyena blocks.

    Implements: output = (gate * activation(up)) @ down
    """

    def __init__(
        self,
        d_model: int,
        expansion_factor: int = 4,
        dropout: float = 0.0,
        activation: str = "silu",
        use_bias: bool = True,
    ):
        """Initialize gated MLP.

        Args:
            d_model: Model dimension.
            expansion_factor: MLP expansion ratio.
            dropout: Dropout probability.
            activation: Activation function ("silu", "gelu", or "relu").
            use_bias: Whether to use bias in linear layers.
        """
        super().__init__()
        hidden_dim = d_model * expansion_factor

        # Up projection (produces 2x for gating)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=use_bias)
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=use_bias)

        # Down projection
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=use_bias)

        # Activation
        if activation == "silu":
            self.act = nn.SiLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Gated activation
        gate = self.act(self.gate_proj(x))
        up = self.up_proj(x)

        # Multiply and project down
        hidden = gate * up
        hidden = self.dropout(hidden)
        output = self.down_proj(hidden)

        return output


class CauchyKernel(nn.Module):
    """Cauchy kernel for efficient filter computation.

    This is a placeholder for the optimized Cauchy multiplication
    used in some Hyena variants for implicit filter generation.
    The Cauchy kernel allows efficient computation of certain
    structured matrix-vector products.
    """

    def __init__(
        self,
        d_model: int,
        num_poles: int = 64,
    ):
        """Initialize Cauchy kernel.

        Args:
            d_model: Model dimension.
            num_poles: Number of poles for Cauchy representation.
        """
        super().__init__()
        self.d_model = d_model
        self.num_poles = num_poles

        # Learnable poles and residues
        self.poles = nn.Parameter(torch.randn(d_model, num_poles) * 0.1)
        self.residues = nn.Parameter(torch.randn(d_model, num_poles) * 0.01)

    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate Cauchy kernel at given points.

        Args:
            z: Evaluation points of shape (seq_len,).

        Returns:
            Kernel values of shape (seq_len, d_model).
        """
        # Cauchy sum: sum_k residue_k / (z - pole_k)
        # Shape: (seq_len, d_model, num_poles)
        z = z.unsqueeze(-1).unsqueeze(-1)  # (seq_len, 1, 1)
        denom = z - self.poles.unsqueeze(0)  # (seq_len, d_model, num_poles)

        # Compute sum
        result = (self.residues.unsqueeze(0) / denom).sum(dim=-1)

        return result.real if result.is_complex() else result
