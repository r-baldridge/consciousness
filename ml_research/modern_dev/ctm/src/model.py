"""
Continuous Thought Machine (CTM) Model Implementation

This module contains the main CTM model architecture with:
- Configuration dataclass
- Main CTM model class
- Core forward method with temporal unrolling

Reference: https://arxiv.org/abs/2505.05522
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import NeuronLevelModel, SynchronizationLayer, TemporalHistory


@dataclass
class CTMConfig:
    """Configuration for Continuous Thought Machine.

    Attributes:
        hidden_dim: Dimension of hidden representations.
        num_neurons: Number of neurons in the NLM layer.
        history_length: Number of past timesteps to maintain in history.
        max_internal_steps: Maximum internal time steps for processing.
        sync_window: Window size for synchronization computation.
        halt_threshold: Threshold for adaptive halting (when enabled).
        neuron_activation: Activation function for neurons.
        input_dim: Input dimension (set based on task).
        output_dim: Output dimension (set based on task).
        dropout: Dropout probability.
        use_adaptive_halt: Whether to use adaptive halting mechanism.
        num_sync_heads: Number of heads for synchronization layer.
    """
    hidden_dim: int = 512
    num_neurons: int = 1024
    history_length: int = 8
    max_internal_steps: int = 32
    sync_window: int = 4
    halt_threshold: float = 0.01
    neuron_activation: str = "gelu"
    input_dim: int = 768
    output_dim: int = 1000
    dropout: float = 0.1
    use_adaptive_halt: bool = False
    num_sync_heads: int = 8

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "hidden_dim": self.hidden_dim,
            "num_neurons": self.num_neurons,
            "history_length": self.history_length,
            "max_internal_steps": self.max_internal_steps,
            "sync_window": self.sync_window,
            "halt_threshold": self.halt_threshold,
            "neuron_activation": self.neuron_activation,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "dropout": self.dropout,
            "use_adaptive_halt": self.use_adaptive_halt,
            "num_sync_heads": self.num_sync_heads,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CTMConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


class CTMEmbedding(nn.Module):
    """Input embedding layer for CTM.

    Transforms input data into initial neuron activations.
    """

    def __init__(self, config: CTMConfig):
        """Initialize embedding layer.

        Args:
            config: CTM configuration.
        """
        super().__init__()
        self.config = config
        self.projection = nn.Linear(config.input_dim, config.num_neurons)
        self.layer_norm = nn.LayerNorm(config.num_neurons)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input to neuron space.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor of shape (batch_size, num_neurons) with initial activations.
        """
        x = self.projection(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class CTMOutputHead(nn.Module):
    """Output head for CTM.

    Transforms synchronization patterns into task outputs.
    """

    def __init__(self, config: CTMConfig):
        """Initialize output head.

        Args:
            config: CTM configuration.
        """
        super().__init__()
        self.config = config

        # Project from synchronization features to hidden
        sync_feature_dim = config.num_sync_heads * config.sync_window
        self.sync_projection = nn.Linear(sync_feature_dim, config.hidden_dim)

        # Combine with neuron activations
        self.neuron_projection = nn.Linear(config.num_neurons, config.hidden_dim)

        # Final output projection
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.output_projection = nn.Linear(config.hidden_dim, config.output_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        neuron_activations: torch.Tensor,
        sync_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute output from neural state.

        Args:
            neuron_activations: Current neuron activations (batch_size, num_neurons).
            sync_features: Synchronization features (batch_size, num_heads * sync_window).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        # Project both components to hidden space
        sync_hidden = self.sync_projection(sync_features)
        neuron_hidden = self.neuron_projection(neuron_activations)

        # Combine
        combined = sync_hidden + neuron_hidden
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)

        # Final projection
        output = self.output_projection(combined)
        return output


class CTM(nn.Module):
    """Continuous Thought Machine.

    A neural network architecture that leverages neural dynamics as its core
    representation, with neurons that maintain temporal history and synchronize
    to encode information.

    The key innovation is decoupling internal time from input sequence time,
    allowing variable computation depth based on input complexity.

    Reference: https://arxiv.org/abs/2505.05522
    """

    def __init__(self, config: CTMConfig):
        """Initialize CTM model.

        Args:
            config: CTM configuration object.
        """
        super().__init__()
        self.config = config

        # Input embedding
        self.embedding = CTMEmbedding(config)

        # Core components
        self.neuron_model = NeuronLevelModel(
            num_neurons=config.num_neurons,
            history_length=config.history_length,
            activation=config.neuron_activation,
            dropout=config.dropout,
        )

        self.temporal_history = TemporalHistory(
            num_neurons=config.num_neurons,
            history_length=config.history_length,
        )

        self.sync_layer = SynchronizationLayer(
            num_neurons=config.num_neurons,
            num_heads=config.num_sync_heads,
            window_size=config.sync_window,
        )

        # Output head
        self.output_head = CTMOutputHead(config)

        # Optional adaptive halting
        if config.use_adaptive_halt:
            self.halt_predictor = nn.Sequential(
                nn.Linear(config.num_neurons, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, 1),
                nn.Sigmoid(),
            )
        else:
            self.halt_predictor = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with temporal unrolling.

        Args:
            x: Input tensor of shape (batch_size, input_dim).
            num_steps: Number of internal time steps (defaults to max_internal_steps).
            return_intermediates: Whether to return intermediate states.

        Returns:
            Dictionary containing:
                - output: Final output tensor (batch_size, output_dim).
                - activations: Final neuron activations (batch_size, num_neurons).
                - sync_matrix: Final synchronization matrix.
                - intermediates: List of intermediate states (if requested).
                - num_steps_used: Actual number of steps taken (for adaptive halt).
        """
        batch_size = x.size(0)
        device = x.device

        if num_steps is None:
            num_steps = self.config.max_internal_steps

        # Initialize neuron activations from input
        activations = self.embedding(x)  # (batch_size, num_neurons)

        # Reset temporal history
        self.temporal_history.reset(batch_size, device)

        # Store intermediates if requested
        intermediates: List[Dict[str, torch.Tensor]] = []

        # Previous sync matrix for adaptive halting
        prev_sync_matrix = None
        actual_steps = num_steps

        # Temporal unrolling loop
        for step in range(num_steps):
            # Update history with current activations
            self.temporal_history.update(activations)

            # Get history tensor
            history = self.temporal_history.get_history()  # (batch, num_neurons, history_length)

            # Compute next activations using NLM
            activations = self.neuron_model(activations, history)

            # Compute synchronization features
            sync_matrix, sync_features = self.sync_layer(
                activations,
                self.temporal_history.get_recent_window(self.config.sync_window),
            )

            # Store intermediate state
            if return_intermediates:
                intermediates.append({
                    "step": step,
                    "activations": activations.clone(),
                    "sync_matrix": sync_matrix.clone(),
                })

            # Check adaptive halting condition
            if self.config.use_adaptive_halt and prev_sync_matrix is not None:
                sync_diff = (sync_matrix - prev_sync_matrix).abs().mean()
                if sync_diff < self.config.halt_threshold:
                    actual_steps = step + 1
                    break

            prev_sync_matrix = sync_matrix

        # Compute final output
        output = self.output_head(activations, sync_features)

        result = {
            "output": output,
            "activations": activations,
            "sync_matrix": sync_matrix,
            "sync_features": sync_features,
            "num_steps_used": actual_steps,
        }

        if return_intermediates:
            result["intermediates"] = intermediates

        return result

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get synchronization patterns as attention-like weights.

        Returns:
            Synchronization matrix if available, None otherwise.
        """
        # Placeholder - would return sync patterns for visualization
        return None

    @classmethod
    def from_pretrained(cls, path: str) -> "CTM":
        """Load pretrained CTM model.

        Args:
            path: Path to pretrained model checkpoint.

        Returns:
            Loaded CTM model.
        """
        # Placeholder for checkpoint loading
        checkpoint = torch.load(path, map_location="cpu")
        config = CTMConfig.from_dict(checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def save_pretrained(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        checkpoint = {
            "config": self.config.to_dict(),
            "model_state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)
