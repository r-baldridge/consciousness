"""
CTM Architecture-Specific Layers

Core components for Continuous Thought Machine:
- NeuronLevelModel: Individual neuron processing with temporal weights
- SynchronizationLayer: Neural synchronization computation
- TemporalHistory: Buffer for maintaining activation history

Reference: https://arxiv.org/abs/2505.05522
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalHistory(nn.Module):
    """Buffer for maintaining neuron activation history.

    Stores the temporal history of neuron activations for use in
    computing next-step activations based on past patterns.
    """

    def __init__(
        self,
        num_neurons: int,
        history_length: int,
    ):
        """Initialize temporal history buffer.

        Args:
            num_neurons: Number of neurons to track.
            history_length: Number of past timesteps to maintain.
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.history_length = history_length

        # Register buffer (not a parameter, but saved with model)
        self.register_buffer(
            "history",
            torch.zeros(1, num_neurons, history_length),
            persistent=False,
        )
        self.register_buffer("current_idx", torch.tensor(0), persistent=False)

    def reset(self, batch_size: int, device: torch.device) -> None:
        """Reset history buffer for new forward pass.

        Args:
            batch_size: Batch size for the new pass.
            device: Device to create tensors on.
        """
        self.history = torch.zeros(
            batch_size,
            self.num_neurons,
            self.history_length,
            device=device,
        )
        self.current_idx = torch.tensor(0, device=device)

    def update(self, activations: torch.Tensor) -> None:
        """Add new activations to history buffer.

        Uses circular buffer pattern for efficient updates.

        Args:
            activations: Current neuron activations (batch_size, num_neurons).
        """
        idx = self.current_idx.item() % self.history_length
        self.history[:, :, idx] = activations
        self.current_idx = self.current_idx + 1

    def get_history(self) -> torch.Tensor:
        """Get full history tensor in temporal order.

        Returns:
            History tensor of shape (batch_size, num_neurons, history_length).
            Ordered from oldest to newest activation.
        """
        if self.current_idx.item() < self.history_length:
            # Not yet filled - return as-is with zero padding
            return self.history

        # Reorder circular buffer to temporal order
        idx = self.current_idx.item() % self.history_length
        # Oldest first: [idx, idx+1, ..., history_length-1, 0, 1, ..., idx-1]
        indices = torch.cat([
            torch.arange(idx, self.history_length, device=self.history.device),
            torch.arange(0, idx, device=self.history.device),
        ])
        return self.history.index_select(dim=2, index=indices)

    def get_recent_window(self, window_size: int) -> torch.Tensor:
        """Get most recent activations within a window.

        Args:
            window_size: Number of recent timesteps to retrieve.

        Returns:
            Tensor of shape (batch_size, num_neurons, window_size).
        """
        window_size = min(window_size, self.history_length)
        history = self.get_history()
        return history[:, :, -window_size:]


class NeuronLevelModel(nn.Module):
    """Neuron-Level Model (NLM).

    Each neuron has its own internal weights that process the history
    of incoming signals to compute the next activation. This is a
    mid-level abstraction between single weights and full layers.

    The key innovation is that neurons decide their next state based
    on their temporal context, enabling adaptive computation.
    """

    def __init__(
        self,
        num_neurons: int,
        history_length: int,
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        """Initialize Neuron-Level Model.

        Args:
            num_neurons: Number of neurons.
            history_length: Length of history to process.
            activation: Activation function name.
            dropout: Dropout probability.
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.history_length = history_length

        # Per-neuron temporal weights
        # Shape: (num_neurons, history_length)
        # Each neuron has unique weights for processing its history
        self.temporal_weights = nn.Parameter(
            torch.randn(num_neurons, history_length) * 0.02
        )

        # Mixing weights for cross-neuron influence
        # Sparse interactions to keep computation tractable
        self.mixing = nn.Linear(num_neurons, num_neurons)

        # Gating mechanism for residual connection
        self.gate = nn.Sequential(
            nn.Linear(num_neurons, num_neurons),
            nn.Sigmoid(),
        )

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(num_neurons)

        # Activation function
        self.activation = self._get_activation(activation)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name.

        Args:
            name: Activation function name.

        Returns:
            Activation module.
        """
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(name.lower(), nn.GELU())

    def forward(
        self,
        current_activations: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        """Compute next neuron activations.

        Args:
            current_activations: Current activations (batch_size, num_neurons).
            history: Activation history (batch_size, num_neurons, history_length).

        Returns:
            Next activations (batch_size, num_neurons).
        """
        batch_size = current_activations.size(0)

        # Apply temporal weights to history (per-neuron)
        # history: (batch, neurons, history_len)
        # weights: (neurons, history_len)
        # Result: (batch, neurons)
        temporal_contribution = torch.einsum(
            "bnh,nh->bn",
            history,
            self.temporal_weights,
        )

        # Apply activation
        temporal_contribution = self.activation(temporal_contribution)

        # Cross-neuron mixing
        mixed = self.mixing(temporal_contribution)
        mixed = self.dropout(mixed)

        # Gated residual connection
        gate = self.gate(current_activations)
        next_activations = gate * current_activations + (1 - gate) * mixed

        # Layer norm
        next_activations = self.layer_norm(next_activations)

        return next_activations


class SynchronizationLayer(nn.Module):
    """Computes neural synchronization patterns.

    Information in CTM is encoded in the TIMING of neural activity.
    Synchronization patterns across neurons form the latent representation
    used for downstream tasks.

    This layer computes pairwise synchronization metrics and aggregates
    them into features suitable for task prediction.
    """

    def __init__(
        self,
        num_neurons: int,
        num_heads: int = 8,
        window_size: int = 4,
    ):
        """Initialize synchronization layer.

        Args:
            num_neurons: Number of neurons.
            num_heads: Number of synchronization heads (for multi-head sync).
            window_size: Temporal window for synchronization computation.
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.num_heads = num_heads
        self.window_size = window_size

        assert num_neurons % num_heads == 0, "num_neurons must be divisible by num_heads"
        self.neurons_per_head = num_neurons // num_heads

        # Projection for computing sync keys and values
        self.sync_key_proj = nn.Linear(window_size, window_size)
        self.sync_value_proj = nn.Linear(window_size, window_size)

        # Output projection
        self.output_proj = nn.Linear(num_heads * window_size, num_heads * window_size)

        # Layer norm
        self.layer_norm = nn.LayerNorm(num_heads * window_size)

    def forward(
        self,
        activations: torch.Tensor,
        recent_window: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute synchronization patterns.

        Args:
            activations: Current activations (batch_size, num_neurons).
            recent_window: Recent activation window (batch_size, num_neurons, window_size).

        Returns:
            Tuple of:
                - sync_matrix: Synchronization matrix (batch_size, num_heads, neurons_per_head, neurons_per_head).
                - sync_features: Aggregated sync features (batch_size, num_heads * window_size).
        """
        batch_size = activations.size(0)

        # Reshape for multi-head processing
        # (batch, num_neurons, window) -> (batch, num_heads, neurons_per_head, window)
        recent_reshaped = recent_window.view(
            batch_size,
            self.num_heads,
            self.neurons_per_head,
            self.window_size,
        )

        # Compute sync keys and values
        # Keys capture "when" neurons fire
        # Values capture "what" they contribute
        sync_keys = self.sync_key_proj(recent_reshaped)  # (batch, heads, neurons_per_head, window)
        sync_values = self.sync_value_proj(recent_reshaped)

        # Compute pairwise synchronization via correlation
        # (batch, heads, n, w) x (batch, heads, w, n) -> (batch, heads, n, n)
        sync_matrix = torch.matmul(
            sync_keys,
            sync_values.transpose(-2, -1),
        ) / (self.window_size ** 0.5)

        # Apply softmax for attention-like normalization
        sync_matrix = F.softmax(sync_matrix, dim=-1)

        # Aggregate sync features
        # Use mean pooling over neuron dimension
        sync_aggregated = sync_values.mean(dim=2)  # (batch, heads, window)

        # Flatten and project
        sync_features = sync_aggregated.view(batch_size, -1)  # (batch, heads * window)
        sync_features = self.output_proj(sync_features)
        sync_features = self.layer_norm(sync_features)

        return sync_matrix, sync_features


class AdaptiveHaltingMechanism(nn.Module):
    """Adaptive computation time mechanism for CTM.

    Learns when to stop internal processing based on synchronization
    stability. When sync patterns stabilize, further computation
    provides diminishing returns.

    Note: In CTM, this often emerges naturally without explicit
    mechanism, but this provides explicit control when needed.
    """

    def __init__(
        self,
        num_neurons: int,
        hidden_dim: int,
        threshold: float = 0.01,
    ):
        """Initialize adaptive halting mechanism.

        Args:
            num_neurons: Number of neurons.
            hidden_dim: Hidden dimension for halt predictor.
            threshold: Base halting threshold.
        """
        super().__init__()
        self.threshold = threshold

        # Halt probability predictor
        self.halt_predictor = nn.Sequential(
            nn.Linear(num_neurons, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Running ponder cost for regularization
        self.register_buffer("ponder_cost", torch.tensor(0.0), persistent=False)

    def forward(
        self,
        activations: torch.Tensor,
        prev_sync: Optional[torch.Tensor] = None,
        current_sync: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, bool]:
        """Compute halting probability.

        Args:
            activations: Current neuron activations (batch_size, num_neurons).
            prev_sync: Previous synchronization matrix.
            current_sync: Current synchronization matrix.

        Returns:
            Tuple of:
                - halt_prob: Halting probability (batch_size, 1).
                - should_halt: Boolean indicating if processing should stop.
        """
        # Compute learned halt probability
        halt_prob = self.halt_predictor(activations)

        # Also check sync stability if provided
        should_halt = False
        if prev_sync is not None and current_sync is not None:
            sync_diff = (current_sync - prev_sync).abs().mean()
            should_halt = sync_diff < self.threshold

        # Combine learned and stability-based halting
        should_halt = should_halt or (halt_prob.mean() > 0.5)

        return halt_prob, should_halt

    def compute_ponder_cost(
        self,
        halt_probs: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute ponder cost for regularization.

        Encourages model to halt early when confident.

        Args:
            halt_probs: List of halt probabilities from each step.

        Returns:
            Ponder cost scalar.
        """
        if not halt_probs:
            return torch.tensor(0.0)

        # Stack and compute cumulative remainder
        probs = torch.stack(halt_probs, dim=1)  # (batch, steps, 1)
        remainder = 1.0 - probs.cumsum(dim=1)
        ponder_cost = remainder.sum(dim=1).mean()

        return ponder_cost
