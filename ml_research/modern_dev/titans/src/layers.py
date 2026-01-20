"""
Titans Layers - Architecture-Specific Layer Implementations

This module contains the core layer components for Titans:
- TestTimeLearner: Manages gradient-based test-time updates
- SurpriseGate: Gates memory updates based on surprise
- MemoryAttention: Cross-attention to memory slots
- AdaptiveLearningRate: Learns per-parameter learning rates
- MemorySnapshot: Saves and restores memory states
"""

from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TestTimeLearner(nn.Module):
    """Manages test-time learning for memory modules.

    Wraps a module and provides methods for performing gradient-based
    updates during inference.
    """

    def __init__(
        self,
        module: nn.Module,
        learning_rate: float = 0.01,
        max_grad_norm: float = 1.0,
        update_fraction: float = 1.0,
    ):
        """Initialize test-time learner.

        Args:
            module: Module to update at test time.
            learning_rate: Base learning rate for updates.
            max_grad_norm: Maximum gradient norm for clipping.
            update_fraction: Fraction of parameters to update (for efficiency).
        """
        super().__init__()
        self.module = module
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.update_fraction = update_fraction

        # Store initial state for reset
        self._initial_state: Optional[Dict] = None
        self._save_initial_state()

    def _save_initial_state(self) -> None:
        """Save initial parameter state for reset."""
        self._initial_state = {
            name: param.clone().detach()
            for name, param in self.module.named_parameters()
        }

    def forward(self, *args, **kwargs):
        """Forward pass through wrapped module."""
        return self.module(*args, **kwargs)

    @torch.enable_grad()
    def update(
        self,
        loss: torch.Tensor,
        learning_rate: Optional[float] = None,
    ) -> Dict[str, float]:
        """Perform a test-time update step.

        Args:
            loss: Loss to minimize.
            learning_rate: Optional override learning rate.

        Returns:
            Dictionary of update statistics.
        """
        lr = learning_rate if learning_rate is not None else self.learning_rate

        # Compute gradients
        grads = torch.autograd.grad(
            loss,
            self.module.parameters(),
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        # Compute gradient norm for clipping and stats
        total_norm = 0.0
        for grad in grads:
            if grad is not None:
                total_norm += grad.norm().item() ** 2
        total_norm = total_norm ** 0.5

        # Clip gradients
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        clip_coef = min(clip_coef, 1.0)

        # Apply updates
        with torch.no_grad():
            for param, grad in zip(self.module.parameters(), grads):
                if grad is not None:
                    param.sub_(lr * clip_coef * grad)

        return {
            "grad_norm": total_norm,
            "clip_coef": clip_coef,
            "effective_lr": lr * clip_coef,
        }

    def reset(self) -> None:
        """Reset module to initial state."""
        if self._initial_state is not None:
            with torch.no_grad():
                for name, param in self.module.named_parameters():
                    if name in self._initial_state:
                        param.copy_(self._initial_state[name])


class SurpriseGate(nn.Module):
    """Gates operations based on surprise signal.

    High surprise triggers memory writes; low surprise enables
    fast-path processing.
    """

    def __init__(
        self,
        hidden_dim: int,
        threshold: float = 0.1,
        learnable_threshold: bool = True,
        soft_gating: bool = True,
    ):
        """Initialize surprise gate.

        Args:
            hidden_dim: Hidden dimension for gate computation.
            threshold: Base surprise threshold.
            learnable_threshold: Whether threshold is learnable.
            soft_gating: Use soft (sigmoid) vs hard (threshold) gating.
        """
        super().__init__()
        self.soft_gating = soft_gating

        if learnable_threshold:
            self.threshold = nn.Parameter(torch.tensor(threshold))
        else:
            self.register_buffer("threshold", torch.tensor(threshold))

        # Linear layer to compute gate from surprise
        self.gate_proj = nn.Linear(1, hidden_dim)

    def forward(
        self,
        surprise: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply surprise-based gating.

        Args:
            surprise: Surprise signal of shape (batch, seq_len).
            values: Values to gate of shape (batch, seq_len, hidden_dim).

        Returns:
            Tuple of (gated_values, gate_values).
        """
        # Compute gate
        surprise_expanded = surprise.unsqueeze(-1)  # (batch, seq_len, 1)

        if self.soft_gating:
            # Soft sigmoid gating based on distance from threshold
            gate_logits = self.gate_proj(surprise_expanded - self.threshold)
            gate = torch.sigmoid(gate_logits)
        else:
            # Hard thresholding
            gate = (surprise_expanded > self.threshold).float()
            gate = gate.expand_as(values)

        gated_values = values * gate

        return gated_values, gate


class MemoryAttention(nn.Module):
    """Cross-attention mechanism for querying memory slots.

    Allows the model to attend to a fixed set of memory slots
    that can be updated at test time.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_memory_slots: int = 64,
        dropout: float = 0.0,
    ):
        """Initialize memory attention.

        Args:
            hidden_dim: Model hidden dimension.
            num_heads: Number of attention heads.
            num_memory_slots: Number of memory slots.
            dropout: Dropout probability.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_memory_slots = num_memory_slots

        # Query projection (from input)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)

        # Memory slots (key-value pairs)
        self.memory_keys = nn.Parameter(torch.randn(num_memory_slots, hidden_dim) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(num_memory_slots, hidden_dim) * 0.02)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Query memory with input.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            Retrieved memory of shape (batch, seq_len, hidden_dim).
        """
        batch_size, seq_len, _ = x.shape

        # Compute queries from input
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # (batch, heads, seq, head_dim)

        # Expand memory keys and values for batch
        k = self.memory_keys.view(1, self.num_memory_slots, self.num_heads, self.head_dim)
        k = k.transpose(1, 2).expand(batch_size, -1, -1, -1)  # (batch, heads, slots, head_dim)

        v = self.memory_values.view(1, self.num_memory_slots, self.num_heads, self.head_dim)
        v = v.transpose(1, 2).expand(batch_size, -1, -1, -1)

        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        return self.out_proj(attn_output)

    def update_memory(
        self,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> None:
        """Update memory slots.

        Args:
            new_keys: New key values.
            new_values: New value values.
            indices: Optional specific indices to update.
        """
        with torch.no_grad():
            if indices is not None:
                self.memory_keys[indices] = new_keys
                self.memory_values[indices] = new_values
            else:
                self.memory_keys.copy_(new_keys)
                self.memory_values.copy_(new_values)


class AdaptiveLearningRate(nn.Module):
    """Learns per-parameter adaptive learning rates for test-time updates.

    Instead of using a fixed learning rate, this module learns
    optimal learning rates for each parameter during training.
    """

    def __init__(
        self,
        param_shapes: List[Tuple[int, ...]],
        base_lr: float = 0.01,
        min_lr: float = 1e-6,
        max_lr: float = 1.0,
    ):
        """Initialize adaptive learning rate module.

        Args:
            param_shapes: Shapes of parameters to learn rates for.
            base_lr: Base learning rate.
            min_lr: Minimum learning rate.
            max_lr: Maximum learning rate.
        """
        super().__init__()
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_lr = max_lr

        # Learnable log learning rates for each parameter
        self.log_lr = nn.ParameterList([
            nn.Parameter(torch.full(shape, torch.log(torch.tensor(base_lr))))
            for shape in param_shapes
        ])

    def get_learning_rates(self) -> List[torch.Tensor]:
        """Get current learning rates (clamped to valid range).

        Returns:
            List of learning rate tensors.
        """
        return [
            torch.clamp(lr.exp(), self.min_lr, self.max_lr)
            for lr in self.log_lr
        ]

    def apply_updates(
        self,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
    ) -> None:
        """Apply gradient updates with adaptive learning rates.

        Args:
            params: List of parameters to update.
            grads: List of gradients.
        """
        lrs = self.get_learning_rates()

        with torch.no_grad():
            for param, grad, lr in zip(params, grads, lrs):
                if grad is not None:
                    param.sub_(lr * grad)


class MemorySnapshot(nn.Module):
    """Manages saving and restoring memory states.

    Useful for:
    - Checkpointing during generation
    - A/B testing different memory states
    - Implementing memory rollback
    """

    def __init__(self, memory_modules: List[nn.Module]):
        """Initialize memory snapshot manager.

        Args:
            memory_modules: List of memory modules to manage.
        """
        super().__init__()
        self.memory_modules = memory_modules
        self._snapshots: Dict[str, List[Dict]] = {}

    def save_snapshot(self, name: str = "default") -> None:
        """Save current memory state.

        Args:
            name: Name for this snapshot.
        """
        snapshot = []
        for module in self.memory_modules:
            module_state = {
                name: param.clone().detach()
                for name, param in module.named_parameters()
            }
            snapshot.append(module_state)

        self._snapshots[name] = snapshot

    def load_snapshot(self, name: str = "default") -> bool:
        """Load a saved memory state.

        Args:
            name: Name of snapshot to load.

        Returns:
            True if snapshot was found and loaded.
        """
        if name not in self._snapshots:
            return False

        snapshot = self._snapshots[name]
        with torch.no_grad():
            for module, module_state in zip(self.memory_modules, snapshot):
                for param_name, param in module.named_parameters():
                    if param_name in module_state:
                        param.copy_(module_state[param_name])

        return True

    def delete_snapshot(self, name: str) -> bool:
        """Delete a saved snapshot.

        Args:
            name: Name of snapshot to delete.

        Returns:
            True if snapshot was found and deleted.
        """
        if name in self._snapshots:
            del self._snapshots[name]
            return True
        return False

    def list_snapshots(self) -> List[str]:
        """Get list of saved snapshot names."""
        return list(self._snapshots.keys())


class ReconstructionLoss(nn.Module):
    """Computes reconstruction loss for memory training.

    The memory module is trained to reconstruct its inputs,
    with the reconstruction error serving as the surprise signal.
    """

    def __init__(
        self,
        hidden_dim: int,
        reconstruction_weight: float = 1.0,
        contrastive_weight: float = 0.1,
    ):
        """Initialize reconstruction loss.

        Args:
            hidden_dim: Hidden dimension.
            reconstruction_weight: Weight for reconstruction term.
            contrastive_weight: Weight for contrastive term.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.reconstruction_weight = reconstruction_weight
        self.contrastive_weight = contrastive_weight

    def forward(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute reconstruction loss.

        Args:
            original: Original input tensor.
            reconstructed: Reconstructed tensor from memory.
            negatives: Optional negative samples for contrastive learning.

        Returns:
            Tuple of (total_loss, component_losses).
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, original)

        # Contrastive loss (if negatives provided)
        contrastive_loss = torch.tensor(0.0, device=original.device)
        if negatives is not None and self.contrastive_weight > 0:
            # Push reconstructed away from negatives
            neg_dist = F.mse_loss(reconstructed, negatives)
            # We want neg_dist to be high, so minimize its negative
            contrastive_loss = torch.relu(1.0 - neg_dist)

        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.contrastive_weight * contrastive_loss
        )

        components = {
            "reconstruction": recon_loss,
            "contrastive": contrastive_loss,
        }

        return total_loss, components


class PositionalMemory(nn.Module):
    """Memory module with positional encoding.

    Allows the memory to store position-dependent information,
    useful for tasks requiring temporal or sequential awareness.
    """

    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int,
        max_positions: int = 8192,
        num_layers: int = 2,
    ):
        """Initialize positional memory.

        Args:
            hidden_dim: Input/output dimension.
            memory_dim: Internal memory dimension.
            max_positions: Maximum sequence positions.
            num_layers: Number of MLP layers.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_positions = max_positions

        # Positional encoding
        self.pos_embedding = nn.Embedding(max_positions, hidden_dim)

        # Memory network (takes input + position)
        layers = []
        current_dim = hidden_dim * 2  # Input + position

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, memory_dim),
                nn.GELU(),
            ])
            current_dim = memory_dim

        layers.append(nn.Linear(current_dim, hidden_dim))
        self.memory_net = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Query memory with positional context.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            positions: Position indices. If None, uses 0..seq_len-1.

        Returns:
            Memory output of shape (batch, seq_len, hidden_dim).
        """
        batch_size, seq_len, _ = x.shape

        # Get position indices
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)
            positions = positions.unsqueeze(0).expand(batch_size, -1)

        # Clamp positions to valid range
        positions = positions.clamp(0, self.max_positions - 1)

        # Get positional embeddings
        pos_embed = self.pos_embedding(positions)  # (batch, seq_len, hidden_dim)

        # Concatenate input with position
        combined = torch.cat([x, pos_embed], dim=-1)

        # Pass through memory network
        output = self.memory_net(combined)

        return output
