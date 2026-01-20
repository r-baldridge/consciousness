"""
Consistency Models Implementation

Core model architecture for consistency models enabling one-step generation.
Implements ConsistencyFunction, training objectives, and distillation.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConsistencyConfig:
    """Configuration for Consistency Models.

    Attributes:
        input_dim: Dimension of input data
        hidden_dim: Hidden dimension for the network
        num_layers: Number of layers
        time_embed_dim: Dimension of time embedding
        num_heads: Number of attention heads
        dropout: Dropout probability
        sigma_min: Minimum noise level (epsilon)
        sigma_max: Maximum noise level (T)
        sigma_data: Data standard deviation for skip scaling
        rho: Schedule parameter for Karras schedule
        num_timesteps: Number of discretization timesteps
        distillation: Whether using distillation (True) or training (False)
    """
    input_dim: int = 784
    hidden_dim: int = 256
    num_layers: int = 4
    time_embed_dim: int = 128
    num_heads: int = 8
    dropout: float = 0.0
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.5
    rho: float = 7.0
    num_timesteps: int = 18
    distillation: bool = True


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for conditioning."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        """Initialize sinusoidal embedding.

        Args:
            dim: Embedding dimension
            max_period: Maximum period for sinusoidal functions
        """
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute time embedding.

        Args:
            t: Time/sigma values of shape (batch_size,)

        Returns:
            Embeddings of shape (batch_size, dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) *
            torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t * freqs
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return embedding


class SkipScaling(nn.Module):
    """Skip connection scaling for enforcing boundary condition.

    Computes c_skip(t) and c_out(t) such that:
        f(x, epsilon) = x (boundary condition at t=epsilon)

    Parameterization:
        f_theta(x, t) = c_skip(t) * x + c_out(t) * F_theta(x, t)
    """

    def __init__(self, sigma_data: float = 0.5, sigma_min: float = 0.002):
        """Initialize skip scaling.

        Args:
            sigma_data: Standard deviation of data
            sigma_min: Minimum sigma (epsilon)
        """
        super().__init__()
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min

    def compute_c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute skip connection coefficient.

        Args:
            sigma: Noise level of shape (batch_size,) or (batch_size, 1)

        Returns:
            c_skip coefficient
        """
        sigma_data_sq = self.sigma_data ** 2
        sigma_sq = sigma ** 2
        return sigma_data_sq / (sigma_sq + sigma_data_sq)

    def compute_c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute output scaling coefficient.

        Args:
            sigma: Noise level

        Returns:
            c_out coefficient
        """
        sigma_data = self.sigma_data
        return sigma * sigma_data / torch.sqrt(sigma ** 2 + sigma_data ** 2)

    def compute_c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute input scaling coefficient (for preconditioning).

        Args:
            sigma: Noise level

        Returns:
            c_in coefficient
        """
        return 1.0 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def forward(
        self,
        x: torch.Tensor,
        network_output: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Apply skip scaling to network output.

        Args:
            x: Noisy input
            network_output: Raw network output F_theta(x, t)
            sigma: Noise level

        Returns:
            Final output with skip connection
        """
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)

        c_skip = self.compute_c_skip(sigma)
        c_out = self.compute_c_out(sigma)

        return c_skip * x + c_out * network_output


class ConsistencyFunction(nn.Module):
    """Neural network that maps noisy samples to clean data.

    The consistency function f(x_t, t) maps any point on a diffusion
    ODE trajectory to the clean data x_0.
    """

    def __init__(self, config: ConsistencyConfig):
        """Initialize consistency function.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Skip scaling for boundary condition
        self.skip_scaling = SkipScaling(config.sigma_data, config.sigma_min)

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(config.time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_embed_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        # Main network layers
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(
                ConsistencyBlock(
                    config.hidden_dim,
                    config.num_heads,
                    config.dropout,
                )
            )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.input_dim),
        )

        # Initialize output to zero for stability
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Map noisy input to clean prediction.

        Args:
            x: Noisy input of shape (batch_size, input_dim)
            sigma: Noise level of shape (batch_size,)
            condition: Optional conditioning information

        Returns:
            Clean prediction of shape (batch_size, input_dim)
        """
        # Time embedding
        t_emb = self.time_embed(sigma)
        t_emb = self.time_mlp(t_emb)

        # Input preconditioning
        c_in = self.skip_scaling.compute_c_in(sigma)
        if c_in.dim() == 1:
            c_in = c_in.unsqueeze(-1)
        h = self.input_proj(x * c_in)

        # Apply layers
        for layer in self.layers:
            h = layer(h, t_emb)

        # Output projection (raw network output)
        F_x = self.output_proj(h)

        # Apply skip scaling for boundary condition
        output = self.skip_scaling(x, F_x, sigma)

        return output


class ConsistencyBlock(nn.Module):
    """Single block in the consistency function network."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        """Initialize consistency block.

        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        # Adaptive layer norm with time conditioning
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.time_scale = nn.Linear(hidden_dim, hidden_dim)
        self.time_shift = nn.Linear(hidden_dim, hidden_dim)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Apply consistency block.

        Args:
            x: Input of shape (batch_size, hidden_dim)
            t_emb: Time embedding of shape (batch_size, hidden_dim)

        Returns:
            Output of shape (batch_size, hidden_dim)
        """
        # Adaptive layer norm
        h = self.norm1(x)
        scale = 1 + self.time_scale(t_emb)
        shift = self.time_shift(t_emb)
        h = h * scale + shift

        # Feed-forward with residual
        x = x + self.ff(h)
        x = self.norm2(x)

        return x


class KarrasSchedule:
    """Karras noise schedule for consistency models.

    Implements the schedule from "Elucidating the Design Space of
    Diffusion-Based Generative Models" (Karras et al., 2022).
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        num_timesteps: int = 18,
    ):
        """Initialize Karras schedule.

        Args:
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            rho: Schedule exponent
            num_timesteps: Number of discretization steps
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.num_timesteps = num_timesteps

    def get_sigmas(self, n: Optional[int] = None) -> torch.Tensor:
        """Get discretized sigma schedule.

        Args:
            n: Number of steps (defaults to num_timesteps)

        Returns:
            Tensor of sigma values
        """
        if n is None:
            n = self.num_timesteps

        ramp = torch.linspace(0, 1, n)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho

        return sigmas

    def get_adjacent_sigmas(
        self,
        indices: torch.Tensor,
        n: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get adjacent sigma pairs for training.

        Args:
            indices: Timestep indices of shape (batch_size,)
            n: Number of total steps

        Returns:
            Tuple of (sigma_t, sigma_t_minus_1)
        """
        sigmas = self.get_sigmas(n).to(indices.device)

        sigma_t = sigmas[indices]
        sigma_t_minus_1 = sigmas[indices + 1]

        return sigma_t, sigma_t_minus_1


class ConsistencyTraining(nn.Module):
    """Consistency Training (CT) objective.

    Trains consistency models from scratch without a pre-trained
    diffusion model teacher.
    """

    def __init__(self, config: ConsistencyConfig):
        """Initialize consistency training.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Main model (online)
        self.model = ConsistencyFunction(config)

        # EMA model (target)
        self.model_ema = ConsistencyFunction(config)
        self._copy_params_to_ema()

        # Freeze EMA model
        for param in self.model_ema.parameters():
            param.requires_grad = False

        # Schedule
        self.schedule = KarrasSchedule(
            config.sigma_min,
            config.sigma_max,
            config.rho,
            config.num_timesteps,
        )

    def _copy_params_to_ema(self):
        """Copy parameters from online model to EMA model."""
        self.model_ema.load_state_dict(self.model.state_dict())

    @torch.no_grad()
    def update_ema(self, decay: float = 0.999):
        """Update EMA model parameters.

        Args:
            decay: EMA decay rate
        """
        for p_ema, p_online in zip(
            self.model_ema.parameters(),
            self.model.parameters()
        ):
            p_ema.data.mul_(decay).add_(p_online.data, alpha=1 - decay)

    def forward(
        self,
        x: torch.Tensor,
        num_timesteps: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute consistency training loss.

        Args:
            x: Clean data samples of shape (batch_size, input_dim)
            num_timesteps: Number of discretization steps (curriculum)

        Returns:
            Scalar loss value
        """
        batch_size = x.shape[0]
        device = x.device

        if num_timesteps is None:
            num_timesteps = self.config.num_timesteps

        # Sample timestep indices
        indices = torch.randint(
            0, num_timesteps - 1, (batch_size,), device=device
        )

        # Get adjacent sigma pairs
        sigma_t, sigma_t_minus_1 = self.schedule.get_adjacent_sigmas(
            indices, num_timesteps
        )

        # Sample noise
        noise = torch.randn_like(x)

        # Create noisy samples at both time steps
        x_t = x + sigma_t.unsqueeze(-1) * noise
        x_t_minus_1 = x + sigma_t_minus_1.unsqueeze(-1) * noise

        # Online model prediction at t
        f_t = self.model(x_t, sigma_t)

        # Target model prediction at t-1 (no gradient)
        with torch.no_grad():
            f_t_minus_1 = self.model_ema(x_t_minus_1, sigma_t_minus_1)

        # Consistency loss
        loss = F.mse_loss(f_t, f_t_minus_1)

        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate samples using one-step generation.

        Args:
            batch_size: Number of samples to generate
            device: Device to use

        Returns:
            Generated samples
        """
        # Start from maximum noise
        sigma_max = torch.full(
            (batch_size,), self.config.sigma_max, device=device
        )
        x_T = torch.randn(batch_size, self.config.input_dim, device=device)
        x_T = x_T * self.config.sigma_max

        # One-step generation
        samples = self.model(x_T, sigma_max)

        return samples


class ConsistencyDistillation(nn.Module):
    """Consistency Distillation (CD) objective.

    Distills a pre-trained diffusion model into a consistency model.
    """

    def __init__(
        self,
        config: ConsistencyConfig,
        teacher_model: Optional[nn.Module] = None,
    ):
        """Initialize consistency distillation.

        Args:
            config: Model configuration
            teacher_model: Pre-trained diffusion model (score network)
        """
        super().__init__()
        self.config = config

        # Student model (online)
        self.model = ConsistencyFunction(config)

        # Student EMA (target)
        self.model_ema = ConsistencyFunction(config)
        self._copy_params_to_ema()

        for param in self.model_ema.parameters():
            param.requires_grad = False

        # Teacher model (frozen)
        self.teacher = teacher_model
        if self.teacher is not None:
            for param in self.teacher.parameters():
                param.requires_grad = False

        # Schedule
        self.schedule = KarrasSchedule(
            config.sigma_min,
            config.sigma_max,
            config.rho,
            config.num_timesteps,
        )

    def _copy_params_to_ema(self):
        """Copy parameters from online model to EMA model."""
        self.model_ema.load_state_dict(self.model.state_dict())

    @torch.no_grad()
    def update_ema(self, decay: float = 0.999):
        """Update EMA model parameters."""
        for p_ema, p_online in zip(
            self.model_ema.parameters(),
            self.model.parameters()
        ):
            p_ema.data.mul_(decay).add_(p_online.data, alpha=1 - decay)

    def teacher_denoise_step(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
    ) -> torch.Tensor:
        """Perform one denoising step using teacher model.

        Args:
            x: Noisy input
            sigma: Current noise level
            sigma_next: Target noise level

        Returns:
            Denoised sample at sigma_next
        """
        if self.teacher is None:
            # Placeholder: return input unchanged
            return x

        # Teacher predicts score or denoised sample
        # This is a simplified Euler step
        denoised = self.teacher(x, sigma)

        # Interpolate to next noise level
        d = (x - denoised) / sigma.unsqueeze(-1)
        dt = sigma_next - sigma
        x_next = x + dt.unsqueeze(-1) * d

        return x_next

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute consistency distillation loss.

        Args:
            x: Clean data samples

        Returns:
            Scalar loss value
        """
        batch_size = x.shape[0]
        device = x.device

        # Sample timestep indices
        indices = torch.randint(
            0, self.config.num_timesteps - 1, (batch_size,), device=device
        )

        # Get adjacent sigma pairs
        sigma_t, sigma_t_minus_1 = self.schedule.get_adjacent_sigmas(indices)

        # Sample noise
        noise = torch.randn_like(x)

        # Create noisy sample at t
        x_t = x + sigma_t.unsqueeze(-1) * noise

        # Teacher step: t -> t-1
        x_hat_t_minus_1 = self.teacher_denoise_step(x_t, sigma_t, sigma_t_minus_1)

        # Student predictions
        f_t = self.model(x_t, sigma_t)

        with torch.no_grad():
            f_t_minus_1 = self.model_ema(x_hat_t_minus_1, sigma_t_minus_1)

        # Consistency loss
        loss = F.mse_loss(f_t, f_t_minus_1)

        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        num_steps: int = 1,
    ) -> torch.Tensor:
        """Generate samples.

        Args:
            batch_size: Number of samples
            device: Device to use
            num_steps: Number of sampling steps (1 for one-step generation)

        Returns:
            Generated samples
        """
        # Start from maximum noise
        sigma_max = torch.full(
            (batch_size,), self.config.sigma_max, device=device
        )
        x = torch.randn(batch_size, self.config.input_dim, device=device)
        x = x * self.config.sigma_max

        if num_steps == 1:
            # One-step generation
            return self.model(x, sigma_max)

        # Multi-step generation
        sigmas = self.schedule.get_sigmas(num_steps + 1).to(device)

        for i in range(num_steps):
            sigma = torch.full((batch_size,), sigmas[i], device=device)
            x = self.model(x, sigma)

            if i < num_steps - 1:
                # Add noise for next step
                sigma_next = sigmas[i + 1]
                noise = torch.randn_like(x)
                x = x + sigma_next * noise

        return x


class ConsistencyModel(nn.Module):
    """Main Consistency Model class.

    Wraps all components for a complete consistency model system.
    """

    def __init__(
        self,
        config: ConsistencyConfig,
        teacher_model: Optional[nn.Module] = None,
    ):
        """Initialize Consistency Model.

        Args:
            config: Model configuration
            teacher_model: Optional pre-trained teacher for distillation
        """
        super().__init__()
        self.config = config

        if config.distillation and teacher_model is not None:
            self.trainer = ConsistencyDistillation(config, teacher_model)
        else:
            self.trainer = ConsistencyTraining(config)

    def forward(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute training loss.

        Args:
            x: Clean data samples
            **kwargs: Additional arguments for trainer

        Returns:
            Loss value
        """
        return self.trainer(x, **kwargs)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        num_steps: int = 1,
    ) -> torch.Tensor:
        """Generate samples.

        Args:
            batch_size: Number of samples
            device: Device to use
            num_steps: Number of sampling steps

        Returns:
            Generated samples
        """
        if device is None:
            device = next(self.parameters()).device

        if hasattr(self.trainer, 'sample'):
            if isinstance(self.trainer, ConsistencyDistillation):
                return self.trainer.sample(batch_size, device, num_steps)
            else:
                return self.trainer.sample(batch_size, device)

        raise NotImplementedError("Trainer does not support sampling")

    def update_ema(self, decay: float = 0.999):
        """Update EMA parameters."""
        self.trainer.update_ema(decay)

    def get_consistency_function(self) -> ConsistencyFunction:
        """Get the underlying consistency function."""
        return self.trainer.model


# Export all public classes
__all__ = [
    "ConsistencyConfig",
    "ConsistencyModel",
    "ConsistencyFunction",
    "ConsistencyBlock",
    "ConsistencyTraining",
    "ConsistencyDistillation",
    "SkipScaling",
    "KarrasSchedule",
    "SinusoidalTimeEmbedding",
]
