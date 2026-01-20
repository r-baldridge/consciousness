"""
Flow Matching Model Implementation

Core model architecture for flow matching generative modeling.
Implements VectorField network, ODE solvers, and flow path computations.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FlowMatchingConfig:
    """Configuration for Flow Matching model.

    Attributes:
        input_dim: Dimension of input data (e.g., channels * height * width for images)
        hidden_dim: Hidden dimension for the vector field network
        num_layers: Number of layers in the vector field network
        time_embed_dim: Dimension of time embedding
        num_heads: Number of attention heads (for transformer variant)
        dropout: Dropout probability
        use_attention: Whether to use attention in the network
        sigma_min: Minimum noise level for numerical stability
        solver: ODE solver type ('euler', 'heun', 'rk4', 'dopri5')
        num_steps: Number of ODE solver steps for sampling
        use_ot_coupling: Whether to use optimal transport coupling
    """
    input_dim: int = 784
    hidden_dim: int = 256
    num_layers: int = 4
    time_embed_dim: int = 128
    num_heads: int = 8
    dropout: float = 0.0
    use_attention: bool = False
    sigma_min: float = 1e-5
    solver: str = "euler"
    num_steps: int = 50
    use_ot_coupling: bool = True


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for conditioning the vector field."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        """Initialize sinusoidal time embedding.

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
            t: Time values of shape (batch_size,) or (batch_size, 1)

        Returns:
            Time embeddings of shape (batch_size, dim)
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


class VectorField(nn.Module):
    """Neural network that predicts the velocity field v(x, t).

    The velocity field is used to transport samples from noise distribution
    to data distribution along optimal transport paths.
    """

    def __init__(self, config: FlowMatchingConfig):
        """Initialize vector field network.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

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
                VectorFieldBlock(
                    config.hidden_dim,
                    config.num_heads,
                    config.dropout,
                    use_attention=config.use_attention,
                )
            )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.input_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict velocity field at (x, t).

        Args:
            x: Input samples of shape (batch_size, input_dim)
            t: Time values of shape (batch_size,)
            condition: Optional conditioning information

        Returns:
            Predicted velocity of shape (batch_size, input_dim)
        """
        # Time embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)

        # Input projection
        h = self.input_proj(x)

        # Apply layers with time conditioning
        for layer in self.layers:
            h = layer(h, t_emb)

        # Output projection
        v = self.output_proj(h)

        return v


class VectorFieldBlock(nn.Module):
    """Single block in the vector field network."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_attention: bool = False,
    ):
        """Initialize vector field block.

        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_attention: Whether to include self-attention
        """
        super().__init__()
        self.use_attention = use_attention

        # Time conditioning via adaptive layer norm
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

        if use_attention:
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Apply vector field block.

        Args:
            x: Input of shape (batch_size, hidden_dim)
            t_emb: Time embedding of shape (batch_size, hidden_dim)

        Returns:
            Output of shape (batch_size, hidden_dim)
        """
        # Adaptive layer norm with time conditioning
        h = self.norm1(x)
        scale = 1 + self.time_scale(t_emb)
        shift = self.time_shift(t_emb)
        h = h * scale + shift

        # Feed-forward with residual
        x = x + self.ff(h)

        # Optional attention
        if self.use_attention:
            h = self.norm2(x)
            h = h.unsqueeze(1)  # Add sequence dimension
            attn_out, _ = self.attn(h, h, h)
            x = x + attn_out.squeeze(1)

        return x


class ODESolver:
    """ODE solver for sampling from flow matching models."""

    def __init__(self, solver_type: str = "euler"):
        """Initialize ODE solver.

        Args:
            solver_type: Type of solver ('euler', 'heun', 'rk4')
        """
        self.solver_type = solver_type

    def solve(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
        t_span: Tuple[float, float] = (0.0, 1.0),
        num_steps: int = 50,
    ) -> torch.Tensor:
        """Solve ODE to generate samples.

        Args:
            velocity_fn: Function that computes velocity v(x, t)
            x0: Initial samples (noise) of shape (batch_size, dim)
            t_span: Time interval (t_start, t_end)
            num_steps: Number of solver steps

        Returns:
            Final samples of shape (batch_size, dim)
        """
        t_start, t_end = t_span
        dt = (t_end - t_start) / num_steps

        x = x0
        t = t_start

        for _ in range(num_steps):
            t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)

            if self.solver_type == "euler":
                x = self._euler_step(velocity_fn, x, t_tensor, dt)
            elif self.solver_type == "heun":
                x = self._heun_step(velocity_fn, x, t_tensor, dt)
            elif self.solver_type == "rk4":
                x = self._rk4_step(velocity_fn, x, t_tensor, dt)
            else:
                raise ValueError(f"Unknown solver type: {self.solver_type}")

            t += dt

        return x

    def _euler_step(
        self,
        velocity_fn: Callable,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Single Euler step."""
        v = velocity_fn(x, t)
        return x + dt * v

    def _heun_step(
        self,
        velocity_fn: Callable,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Single Heun (improved Euler) step."""
        v1 = velocity_fn(x, t)
        x_pred = x + dt * v1

        t_next = t + dt
        v2 = velocity_fn(x_pred, t_next)

        return x + 0.5 * dt * (v1 + v2)

    def _rk4_step(
        self,
        velocity_fn: Callable,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Single RK4 step."""
        k1 = velocity_fn(x, t)
        k2 = velocity_fn(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = velocity_fn(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = velocity_fn(x + dt * k3, t + dt)

        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class FlowPath:
    """Defines the probability path for flow matching.

    Implements optimal transport (OT) paths that linearly interpolate
    between noise and data distributions.
    """

    def __init__(self, sigma_min: float = 1e-5):
        """Initialize flow path.

        Args:
            sigma_min: Minimum noise for numerical stability
        """
        self.sigma_min = sigma_min

    def interpolate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate between noise (x0) and data (x1).

        For OT path: x_t = (1 - t) * x0 + t * x1

        Args:
            x0: Noise samples of shape (batch_size, dim)
            x1: Data samples of shape (batch_size, dim)
            t: Time values of shape (batch_size,) or (batch_size, 1)

        Returns:
            Interpolated samples of shape (batch_size, dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        return (1 - t) * x0 + t * x1

    def velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute target velocity field.

        For OT path: u_t = x1 - x0 (constant velocity)

        Args:
            x0: Noise samples
            x1: Data samples
            t: Time (unused for OT path, included for API consistency)

        Returns:
            Target velocity
        """
        return x1 - x0


class ConditionalFlowMatching(nn.Module):
    """Conditional Flow Matching training objective.

    Trains the vector field network to predict velocities that transport
    samples from noise to data distribution.
    """

    def __init__(self, config: FlowMatchingConfig):
        """Initialize CFM training.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.vector_field = VectorField(config)
        self.flow_path = FlowPath(config.sigma_min)
        self.solver = ODESolver(config.solver)

    def forward(
        self,
        x1: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute flow matching loss.

        Args:
            x1: Data samples of shape (batch_size, dim)
            condition: Optional conditioning information

        Returns:
            Scalar loss value
        """
        batch_size = x1.shape[0]
        device = x1.device

        # Sample noise
        x0 = torch.randn_like(x1)

        # Sample time uniformly
        t = torch.rand(batch_size, device=device)

        # Interpolate
        x_t = self.flow_path.interpolate(x0, x1, t)

        # Target velocity
        u_t = self.flow_path.velocity(x0, x1, t)

        # Predicted velocity
        v_t = self.vector_field(x_t, t, condition)

        # MSE loss
        loss = F.mse_loss(v_t, u_t)

        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate samples using ODE solver.

        Args:
            batch_size: Number of samples to generate
            device: Device to generate on
            num_steps: Number of solver steps (defaults to config)

        Returns:
            Generated samples of shape (batch_size, input_dim)
        """
        if num_steps is None:
            num_steps = self.config.num_steps

        # Start from noise
        x0 = torch.randn(batch_size, self.config.input_dim, device=device)

        # Solve ODE
        def velocity_fn(x, t):
            return self.vector_field(x, t)

        samples = self.solver.solve(velocity_fn, x0, num_steps=num_steps)

        return samples


class FlowMatchingModel(nn.Module):
    """Main Flow Matching model for generative modeling.

    Wraps all components for a complete flow matching system including
    training, sampling, and optional rectification (reflow).
    """

    def __init__(self, config: FlowMatchingConfig):
        """Initialize Flow Matching model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.cfm = ConditionalFlowMatching(config)

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute training loss.

        Args:
            x: Data samples
            condition: Optional conditioning

        Returns:
            Loss value
        """
        return self.cfm(x, condition)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate samples.

        Args:
            batch_size: Number of samples
            device: Device to use
            num_steps: Number of solver steps

        Returns:
            Generated samples
        """
        if device is None:
            device = next(self.parameters()).device
        return self.cfm.sample(batch_size, device, num_steps)

    def get_velocity_field(self) -> VectorField:
        """Get the vector field network."""
        return self.cfm.vector_field


# Export all public classes
__all__ = [
    "FlowMatchingConfig",
    "FlowMatchingModel",
    "VectorField",
    "VectorFieldBlock",
    "ODESolver",
    "FlowPath",
    "ConditionalFlowMatching",
    "SinusoidalTimeEmbedding",
]
