"""
TRM - Tiny Recursive Model

Main model class implementing the recursive reasoning architecture from Samsung SAIT Montreal.

Reference:
    "Less is More: Recursive Reasoning with Tiny Networks"
    https://arxiv.org/abs/2510.04871
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    TRMBlock,
    DeepRecursion,
    QHead,
    OutputHead,
    GridEmbedding,
    MLPSequence,
    RMSNorm,
)


@dataclass
class TRMConfig:
    """Configuration for Tiny Recursive Model.

    Attributes:
        grid_size: Size of input grid (e.g., 9 for Sudoku, 30 for ARC).
        vocab_size: Number of output classes (e.g., 10 for digits 0-9).
        embed_dim: Embedding dimension.
        n_layers: Number of layers in each network (recommended: 2).
        n_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension multiplier.
        T_cycles: Number of high-level recursion cycles.
        n_cycles: Number of low-level cycles per T cycle.
        max_supervision_steps: Maximum training supervision steps.
        dropout: Dropout probability.
        use_attention: Whether to use attention (False for MLP-only).
        use_rotary: Whether to use rotary position embeddings.
        use_swiglu: Whether to use SwiGLU activation.
        use_stable_max: Whether to use stable max in output head.
        q_threshold: Halting threshold for inference.
    """

    # Task configuration
    grid_size: int = 9
    vocab_size: int = 10
    max_seq_len: int = 81  # grid_size^2

    # Model architecture
    embed_dim: int = 512
    n_layers: int = 2
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # Recursion parameters
    T_cycles: int = 3
    n_cycles: int = 6
    max_supervision_steps: int = 16

    # Architecture options
    use_attention: bool = True
    use_rotary: bool = True
    use_swiglu: bool = True
    use_stable_max: bool = True

    # Inference
    q_threshold: float = 0.0  # Halt when q_hat > threshold

    def __post_init__(self):
        """Compute derived attributes."""
        if self.max_seq_len is None:
            self.max_seq_len = self.grid_size ** 2

    @property
    def effective_depth(self) -> int:
        """Calculate effective network depth through recursion."""
        return self.T_cycles * (self.n_cycles + 1) * self.n_layers

    @classmethod
    def for_sudoku(cls) -> "TRMConfig":
        """Preset configuration for Sudoku."""
        return cls(
            grid_size=9,
            vocab_size=10,
            max_seq_len=81,
            embed_dim=512,
            n_layers=2,
            T_cycles=3,
            n_cycles=6,
            use_attention=False,  # MLP better for small grids
        )

    @classmethod
    def for_maze(cls, grid_size: int = 30) -> "TRMConfig":
        """Preset configuration for Maze solving."""
        return cls(
            grid_size=grid_size,
            vocab_size=4,  # empty, wall, start/end, path
            max_seq_len=grid_size ** 2,
            embed_dim=512,
            n_layers=2,
            T_cycles=3,
            n_cycles=4,
            use_attention=True,
        )

    @classmethod
    def for_arc_agi(cls, grid_size: int = 30) -> "TRMConfig":
        """Preset configuration for ARC-AGI tasks."""
        return cls(
            grid_size=grid_size,
            vocab_size=11,  # 0-9 colors + background
            max_seq_len=grid_size ** 2,
            embed_dim=512,
            n_layers=2,
            T_cycles=3,
            n_cycles=6,
            use_attention=True,
        )


class TRM(nn.Module):
    """Tiny Recursive Model.

    A small recursive network that achieves strong generalization on reasoning
    tasks through iterative refinement rather than depth.

    Key innovations:
        1. Single 2-layer network applied recursively
        2. Deep supervision through all recursive steps
        3. Dual semantic states (y: solution, z: reasoning)
        4. Halting mechanism via Q-head

    Architecture:
        Input (x) -> Initialize y, z
                          |
                          v
                  +----------------+
                  | Deep Recursion |<----+
                  | z <- net(x,y,z)|     |
                  | (n times)      |     | T cycles
                  | y <- net(y,z)  |     |
                  +----------------+-----+
                          |
                          v
                  Supervision Step
                          |
                          v
                  Early Stop if q_hat > threshold

    Effective Depth: T × (n+1) × n_layers = 3 × 7 × 2 = 42 layers

    Args:
        config: TRMConfig instance.
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        # Input embedding
        self.embedding = GridEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            grid_size=config.grid_size,
            dropout=config.dropout,
        )

        # Initial state projections
        self.y_init = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.z_init = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

        # Deep recursion module
        self.recursion = DeepRecursion(
            embed_dim=config.embed_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            mlp_ratio=config.mlp_ratio,
            T_cycles=config.T_cycles,
            n_cycles=config.n_cycles,
            dropout=config.dropout,
            use_attention=config.use_attention,
            use_rotary=config.use_rotary,
            use_swiglu=config.use_swiglu,
            max_seq_len=config.max_seq_len,
        )

        # Output heads
        self.output_head = OutputHead(
            embed_dim=config.embed_dim,
            vocab_size=config.vocab_size,
            use_stable_max=config.use_stable_max,
        )
        self.q_head = QHead(config.embed_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def _initialize_states(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize y and z states from input embedding."""
        y = self.y_init(x)
        z = self.z_init(x)
        return y, z

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional loss computation.

        Args:
            input_ids: Input grid (batch, grid_size, grid_size) or (batch, seq_len)
            labels: Target grid (same shape as input_ids)

        Returns:
            Dictionary containing:
                - logits: Predicted logits (batch, seq_len, vocab_size)
                - q_hat: Halting probability (batch,)
                - loss: Total loss (if labels provided)
                - ce_loss: Cross-entropy loss (if labels provided)
                - q_loss: Q-head loss (if labels provided)
        """
        # Embed input
        x = self.embedding(input_ids)

        # Initialize states
        y, z = self._initialize_states(x)

        # Deep recursion
        y, z = self.recursion(x, y, z, detach_between_T=self.training)

        # Get predictions
        logits = self.output_head(y)
        q_hat = self.q_head(z)

        output = {
            "logits": logits,
            "q_hat": q_hat,
        }

        # Compute loss if labels provided
        if labels is not None:
            # Flatten for cross-entropy
            if labels.dim() == 3:
                labels = labels.view(labels.shape[0], -1)

            ce_loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

            # Q-head loss: BCE with whether prediction is correct
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                # Check if prediction matches target (ignoring masked positions)
                mask = labels != -100
                correct = ((preds == labels) | ~mask).all(dim=-1).float()

            q_loss = F.binary_cross_entropy_with_logits(q_hat, correct)

            total_loss = ce_loss + q_loss

            output.update({
                "loss": total_loss,
                "ce_loss": ce_loss,
                "q_loss": q_loss,
            })

        return output

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        max_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training step with deep supervision.

        Performs multiple supervision steps, halting early if q_hat > 0.

        Args:
            input_ids: Input grid
            labels: Target grid
            max_steps: Maximum supervision steps (default: config.max_supervision_steps)

        Returns:
            Dictionary with accumulated losses and metrics
        """
        max_steps = max_steps or self.config.max_supervision_steps

        # Embed input once
        x = self.embedding(input_ids)

        # Initialize states
        y, z = self._initialize_states(x)

        total_loss = 0.0
        total_ce_loss = 0.0
        total_q_loss = 0.0
        steps_taken = 0

        # Flatten labels once
        if labels.dim() == 3:
            flat_labels = labels.view(labels.shape[0], -1)
        else:
            flat_labels = labels

        for step in range(max_steps):
            # Deep recursion
            y, z = self.recursion(x, y, z, detach_between_T=True)

            # Get predictions
            logits = self.output_head(y)
            q_hat = self.q_head(z)

            # Compute losses
            ce_loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                flat_labels.view(-1),
                ignore_index=-100,
            )

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = flat_labels != -100
                correct = ((preds == flat_labels) | ~mask).all(dim=-1).float()

            q_loss = F.binary_cross_entropy_with_logits(q_hat, correct)

            step_loss = ce_loss + q_loss
            total_loss = total_loss + step_loss
            total_ce_loss = total_ce_loss + ce_loss
            total_q_loss = total_q_loss + q_loss
            steps_taken += 1

            # Early stopping based on q_hat
            with torch.no_grad():
                if (q_hat > self.config.q_threshold).all():
                    break

            # Detach states for next supervision step
            y = y.detach()
            z = z.detach()

        return {
            "loss": total_loss / steps_taken,
            "ce_loss": total_ce_loss / steps_taken,
            "q_loss": total_q_loss / steps_taken,
            "steps": steps_taken,
            "accuracy": correct.mean(),
        }

    @torch.no_grad()
    def solve(
        self,
        input_ids: torch.Tensor,
        max_steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> Dict[str, Any]:
        """
        Inference with recursive refinement.

        Args:
            input_ids: Input grid
            max_steps: Maximum inference steps
            return_trajectory: Whether to return intermediate predictions

        Returns:
            Dictionary containing:
                - solution: Final predicted grid (batch, seq_len)
                - confidence: Final q_hat value (batch,)
                - steps: Number of steps taken
                - trajectory: List of intermediate predictions (if requested)
        """
        max_steps = max_steps or self.config.max_supervision_steps

        # Embed input
        x = self.embedding(input_ids)

        # Initialize states
        y, z = self._initialize_states(x)

        trajectory = [] if return_trajectory else None
        steps_taken = 0

        for step in range(max_steps):
            # Deep recursion
            y, z = self.recursion(x, y, z, detach_between_T=False)

            # Get predictions
            logits = self.output_head(y)
            q_hat = self.q_head(z)
            preds = logits.argmax(dim=-1)

            steps_taken += 1

            if return_trajectory:
                trajectory.append({
                    "step": step,
                    "prediction": preds.clone(),
                    "q_hat": torch.sigmoid(q_hat).clone(),
                })

            # Early stopping
            if (q_hat > self.config.q_threshold).all():
                break

        result = {
            "solution": preds,
            "confidence": torch.sigmoid(q_hat),
            "steps": steps_taken,
            "logits": logits,
        }

        if return_trajectory:
            result["trajectory"] = trajectory

        return result

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, config: TRMConfig) -> "TRM":
        """Create TRM from configuration."""
        return cls(config)

    @classmethod
    def from_pretrained(cls, path: str) -> "TRM":
        """Load pretrained TRM model."""
        checkpoint = torch.load(path, map_location="cpu")
        config = TRMConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def save_pretrained(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "config": {
                "grid_size": self.config.grid_size,
                "vocab_size": self.config.vocab_size,
                "max_seq_len": self.config.max_seq_len,
                "embed_dim": self.config.embed_dim,
                "n_layers": self.config.n_layers,
                "n_heads": self.config.n_heads,
                "mlp_ratio": self.config.mlp_ratio,
                "T_cycles": self.config.T_cycles,
                "n_cycles": self.config.n_cycles,
                "max_supervision_steps": self.config.max_supervision_steps,
                "dropout": self.config.dropout,
                "use_attention": self.config.use_attention,
                "use_rotary": self.config.use_rotary,
                "use_swiglu": self.config.use_swiglu,
                "use_stable_max": self.config.use_stable_max,
                "q_threshold": self.config.q_threshold,
            },
            "model_state_dict": self.state_dict(),
        }, path)
