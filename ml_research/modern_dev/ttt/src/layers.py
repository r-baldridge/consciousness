"""
TTT Architecture-Specific Layers

Core components for Test-Time Training:
- TTTLayer: Base class for TTT layers
- TTTLinear: Linear hidden state layer
- TTTMLP: MLP hidden state layer
- InnerOptimizer: Optimizer for test-time weight updates
- TestTimeTrainer: Manages test-time training process

Reference: https://arxiv.org/abs/2407.04620
"""

from __future__ import annotations

from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, base: int = 10000):
        """Initialize rotary embedding.

        Args:
            dim: Embedding dimension (per head).
            base: Base for frequency computation.
        """
        super().__init__()
        self.dim = dim
        self.base = base

        # Compute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary embeddings.

        Args:
            x: Input tensor (for device/dtype).
            seq_len: Sequence length.
            offset: Position offset for caching.

        Returns:
            Tuple of (cos, sin) tensors.
        """
        t = torch.arange(offset, offset + seq_len, device=x.device, dtype=x.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to queries and keys.

    Args:
        q: Query tensor (batch, heads, seq, head_dim).
        k: Key tensor (batch, heads, seq, head_dim).
        cos: Cosine tensor.
        sin: Sine tensor.

    Returns:
        Rotated (q, k) tensors.
    """
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class InnerOptimizer:
    """Optimizer for test-time weight updates.

    Performs gradient descent on the hidden state (weights)
    during inference. Uses the self-supervised reconstruction
    objective.
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        momentum: float = 0.0,
    ):
        """Initialize inner optimizer.

        Args:
            learning_rate: Learning rate for weight updates.
            momentum: Momentum coefficient (0 = no momentum).
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity: Optional[torch.Tensor] = None

    def step(
        self,
        weights: torch.Tensor,
        gradient: torch.Tensor,
    ) -> torch.Tensor:
        """Perform one optimization step.

        Args:
            weights: Current weights tensor.
            gradient: Gradient tensor.

        Returns:
            Updated weights tensor.
        """
        if self.momentum > 0:
            if self.velocity is None:
                self.velocity = torch.zeros_like(gradient)
            self.velocity = self.momentum * self.velocity + gradient
            update = self.velocity
        else:
            update = gradient

        return weights - self.learning_rate * update

    def reset(self) -> None:
        """Reset optimizer state."""
        self.velocity = None


class TestTimeTrainer:
    """Manages test-time training for TTT layers.

    Coordinates the self-supervised learning that happens
    during inference, computing gradients and updating
    hidden state weights.
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        mini_batch_size: int = 16,
    ):
        """Initialize test-time trainer.

        Args:
            learning_rate: Learning rate for TTT updates.
            mini_batch_size: Mini-batch size for batched updates.
        """
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.optimizer = InnerOptimizer(learning_rate=learning_rate)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute self-supervised loss.

        L(x; W) = ||f(xW_K; W) - xW_V||^2

        Args:
            predictions: Model predictions.
            targets: Target values.

        Returns:
            MSE loss.
        """
        return F.mse_loss(predictions, targets, reduction="mean")

    def update_weights(
        self,
        weights: torch.Tensor,
        inputs: torch.Tensor,
        key_proj: nn.Linear,
        value_proj: nn.Linear,
        hidden_model: nn.Module,
    ) -> torch.Tensor:
        """Update hidden state weights via gradient descent.

        Args:
            weights: Current hidden state weights.
            inputs: Input token embeddings.
            key_proj: Key projection layer.
            value_proj: Value projection layer.
            hidden_model: The TTT hidden model (linear or MLP).

        Returns:
            Updated weights.
        """
        # This is a placeholder for the actual gradient computation
        # In practice, this would involve:
        # 1. Computing the forward pass through hidden_model
        # 2. Computing the self-supervised loss
        # 3. Computing gradients with respect to weights
        # 4. Applying the optimizer step

        # Placeholder implementation
        return weights


class TTTLayer(nn.Module):
    """Base class for TTT layers.

    TTT layers replace attention with a learnable hidden state
    that is updated via gradient descent during inference.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        learning_rate: float = 1.0,
        mini_batch_size: int = 16,
        use_rope: bool = True,
        rope_base: int = 10000,
    ):
        """Initialize TTT layer.

        Args:
            hidden_dim: Hidden dimension.
            num_heads: Number of heads.
            learning_rate: TTT learning rate.
            mini_batch_size: Mini-batch size for TTT.
            use_rope: Whether to use rotary embeddings.
            rope_base: Base for rotary embeddings.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size

        # Projections (fixed during inference)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Rotary embeddings
        if use_rope:
            self.rotary = RotaryEmbedding(self.head_dim, base=rope_base)
        else:
            self.rotary = None

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Test-time trainer
        self.trainer = TestTimeTrainer(
            learning_rate=learning_rate,
            mini_batch_size=mini_batch_size,
        )

    def _init_hidden_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Initialize hidden state weights.

        Args:
            batch_size: Batch size.
            device: Device to create tensor on.

        Returns:
            Initial hidden state.
        """
        raise NotImplementedError

    def _apply_hidden_model(
        self,
        x: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the hidden model (linear or MLP).

        Args:
            x: Input tensor.
            hidden_state: Hidden state weights.

        Returns:
            Output tensor.
        """
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        return_hidden_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with test-time training.

        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim).
            hidden_state: Optional previous hidden state.
            return_hidden_state: Whether to return updated hidden state.

        Returns:
            Tuple of output tensor and optional hidden state.
        """
        raise NotImplementedError


class TTTLinear(TTTLayer):
    """TTT layer with linear hidden state.

    Hidden state is a single linear layer: W in R^{d x d}.
    Self-supervised task: reconstruct input from projection.

    z = f(xW_K; W) = xW_K @ W
    loss = ||z - xW_V||^2
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        learning_rate: float = 1.0,
        mini_batch_size: int = 16,
        use_rope: bool = True,
        rope_base: int = 10000,
    ):
        """Initialize TTT-Linear layer."""
        super().__init__(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            learning_rate=learning_rate,
            mini_batch_size=mini_batch_size,
            use_rope=use_rope,
            rope_base=rope_base,
        )

        # Initialize hidden state parameters (W_0)
        # Shape: (num_heads, head_dim, head_dim) per batch element
        self.register_buffer(
            "init_hidden_state",
            torch.eye(self.head_dim).unsqueeze(0).expand(num_heads, -1, -1),
            persistent=True,
        )

    def _init_hidden_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Initialize hidden state as identity matrix.

        Args:
            batch_size: Batch size.
            device: Device to create tensor on.

        Returns:
            Initial hidden state (batch, heads, head_dim, head_dim).
        """
        return self.init_hidden_state.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        ).clone().to(device)

    def _apply_hidden_model(
        self,
        x: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """Apply linear transformation.

        f(x; W) = x @ W

        Args:
            x: Input (batch, heads, seq, head_dim).
            hidden_state: Weights (batch, heads, head_dim, head_dim).

        Returns:
            Output (batch, heads, seq, head_dim).
        """
        # x: (batch, heads, seq, head_dim)
        # W: (batch, heads, head_dim, head_dim)
        # Output: (batch, heads, seq, head_dim)
        return torch.einsum("bhsd,bhde->bhse", x, hidden_state)

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        return_hidden_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with test-time training.

        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim).
            hidden_state: Optional previous hidden state.
            return_hidden_state: Whether to return updated hidden state.

        Returns:
            Tuple of output tensor and optional hidden state.
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden_state(batch_size, device)

        # Project to keys and values
        keys = self.key_proj(x)  # (batch, seq, hidden)
        values = self.value_proj(x)  # (batch, seq, hidden)

        # Reshape to multi-head
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.permute(0, 2, 1, 3)  # (batch, heads, seq, head_dim)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.permute(0, 2, 1, 3)

        # Apply rotary embeddings
        if self.rotary is not None:
            cos, sin = self.rotary(keys, seq_len)
            cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
            sin = sin.unsqueeze(0).unsqueeze(0)
            keys, _ = apply_rotary_pos_emb(keys, keys, cos, sin)

        # Process in mini-batches for efficient TTT
        outputs = []
        current_hidden = hidden_state

        for i in range(0, seq_len, self.mini_batch_size):
            end_idx = min(i + self.mini_batch_size, seq_len)
            mini_keys = keys[:, :, i:end_idx]
            mini_values = values[:, :, i:end_idx]

            # Apply current hidden model
            mini_output = self._apply_hidden_model(mini_keys, current_hidden)

            # Compute gradient and update hidden state (simplified)
            # In full implementation, would use proper gradient computation
            with torch.no_grad():
                # Self-supervised loss: ||f(k; W) - v||^2
                loss_grad = 2 * (mini_output - mini_values)
                # Gradient w.r.t. W: k^T @ loss_grad
                weight_grad = torch.einsum(
                    "bhsd,bhse->bhde",
                    mini_keys,
                    loss_grad,
                ) / (end_idx - i)

                # Update hidden state
                current_hidden = current_hidden - self.learning_rate * weight_grad

            outputs.append(mini_output)

        # Concatenate outputs
        output = torch.cat(outputs, dim=2)  # (batch, heads, seq, head_dim)

        # Reshape back
        output = output.permute(0, 2, 1, 3)  # (batch, seq, heads, head_dim)
        output = output.reshape(batch_size, seq_len, self.hidden_dim)

        # Add residual and project
        output = self.output_proj(output) + x

        if return_hidden_state:
            return output, current_hidden
        return output, None


class TTTMLP(TTTLayer):
    """TTT layer with MLP hidden state.

    Hidden state is a two-layer MLP, more expressive than linear.
    Higher capacity for modeling complex dependencies.
    """

    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden_dim: int,
        num_heads: int,
        learning_rate: float = 1.0,
        mini_batch_size: int = 16,
        use_rope: bool = True,
        rope_base: int = 10000,
    ):
        """Initialize TTT-MLP layer.

        Args:
            hidden_dim: Hidden dimension.
            mlp_hidden_dim: MLP hidden dimension.
            num_heads: Number of heads.
            learning_rate: TTT learning rate.
            mini_batch_size: Mini-batch size for TTT.
            use_rope: Whether to use rotary embeddings.
            rope_base: Base for rotary embeddings.
        """
        super().__init__(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            learning_rate=learning_rate,
            mini_batch_size=mini_batch_size,
            use_rope=use_rope,
            rope_base=rope_base,
        )

        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_head_dim = mlp_hidden_dim // num_heads

        # MLP hidden state initialization parameters
        # W1: head_dim -> mlp_head_dim
        # W2: mlp_head_dim -> head_dim
        self.register_buffer(
            "init_w1",
            torch.randn(num_heads, self.head_dim, self.mlp_head_dim) * 0.02,
            persistent=True,
        )
        self.register_buffer(
            "init_w2",
            torch.randn(num_heads, self.mlp_head_dim, self.head_dim) * 0.02,
            persistent=True,
        )

    def _init_hidden_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize MLP hidden state weights.

        Args:
            batch_size: Batch size.
            device: Device to create tensor on.

        Returns:
            Tuple of (W1, W2) initial weights.
        """
        w1 = self.init_w1.unsqueeze(0).expand(batch_size, -1, -1, -1).clone().to(device)
        w2 = self.init_w2.unsqueeze(0).expand(batch_size, -1, -1, -1).clone().to(device)
        return (w1, w2)

    def _apply_hidden_model(
        self,
        x: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Apply MLP transformation.

        f(x; W1, W2) = GELU(x @ W1) @ W2

        Args:
            x: Input (batch, heads, seq, head_dim).
            hidden_state: Tuple of (W1, W2) weights.

        Returns:
            Output (batch, heads, seq, head_dim).
        """
        w1, w2 = hidden_state

        # First layer
        hidden = torch.einsum("bhsd,bhde->bhse", x, w1)
        hidden = F.gelu(hidden)

        # Second layer
        output = torch.einsum("bhsd,bhde->bhse", hidden, w2)

        return output

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_hidden_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with test-time training.

        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim).
            hidden_state: Optional previous hidden state (W1, W2 tuple).
            return_hidden_state: Whether to return updated hidden state.

        Returns:
            Tuple of output tensor and optional hidden state.
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden_state(batch_size, device)

        w1, w2 = hidden_state

        # Project to keys and values
        keys = self.key_proj(x)
        values = self.value_proj(x)

        # Reshape to multi-head
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.permute(0, 2, 1, 3)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.permute(0, 2, 1, 3)

        # Apply rotary embeddings
        if self.rotary is not None:
            cos, sin = self.rotary(keys, seq_len)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
            keys, _ = apply_rotary_pos_emb(keys, keys, cos, sin)

        # Process in mini-batches
        outputs = []
        current_w1, current_w2 = w1, w2

        for i in range(0, seq_len, self.mini_batch_size):
            end_idx = min(i + self.mini_batch_size, seq_len)
            mini_keys = keys[:, :, i:end_idx]
            mini_values = values[:, :, i:end_idx]

            # Apply current hidden model
            mini_output = self._apply_hidden_model(mini_keys, (current_w1, current_w2))

            # Update hidden state (simplified - full implementation needs
            # proper gradient computation through MLP)
            with torch.no_grad():
                # Placeholder: Simple weight update
                # Full implementation would compute gradients properly
                loss_grad = 2 * (mini_output - mini_values)

                # Simplified W2 gradient
                hidden = torch.einsum("bhsd,bhde->bhse", mini_keys, current_w1)
                hidden_activated = F.gelu(hidden)
                w2_grad = torch.einsum(
                    "bhsd,bhse->bhde",
                    hidden_activated,
                    loss_grad,
                ) / (end_idx - i)

                # Update W2
                current_w2 = current_w2 - self.learning_rate * w2_grad

            outputs.append(mini_output)

        # Concatenate outputs
        output = torch.cat(outputs, dim=2)

        # Reshape back
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(batch_size, seq_len, self.hidden_dim)

        # Add residual and project
        output = self.output_proj(output) + x

        if return_hidden_state:
            return output, (current_w1, current_w2)
        return output, None
