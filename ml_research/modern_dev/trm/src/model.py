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
    # Code repair extensions
    GridPositionalEncoding,
    GridAttention,
    RecursiveBlock,
    FeedForward,
    IterationController,
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


# =============================================================================
# Code Repair Architecture (64x48 Grid)
# =============================================================================


@dataclass
class CodeRepairConfig:
    """Configuration for Code Repair TRM model.

    Designed for code repair tasks with 64x48 grid input.

    Attributes:
        grid_height: Height of input grid (lines of code).
        grid_width: Width of input grid (tokens per line).
        vocab_size: BPE vocabulary size.
        embed_dim: Embedding dimension.
        n_heads: Number of attention heads.
        ffn_dim: Feed-forward hidden dimension.
        n_blocks: Number of recursive blocks (weight-shared).
        max_iterations: Maximum recursion iterations.
        min_iterations: Minimum iterations before early stopping.
        dropout: Dropout probability.
        q_threshold: Confidence threshold for early stopping.
        use_gradient_checkpointing: Enable gradient checkpointing for memory efficiency.
    """

    # Grid configuration (for code)
    grid_height: int = 64
    grid_width: int = 48
    vocab_size: int = 32768

    # Model architecture
    embed_dim: int = 256
    n_heads: int = 8
    ffn_dim: int = 1024
    n_blocks: int = 6  # Weight-shared blocks

    # Recursion parameters
    max_iterations: int = 8
    min_iterations: int = 2

    # Training
    dropout: float = 0.0
    q_threshold: float = 0.95

    # Memory optimization
    use_gradient_checkpointing: bool = False

    @property
    def max_seq_len(self) -> int:
        """Maximum sequence length."""
        return self.grid_height * self.grid_width

    @property
    def effective_depth(self) -> int:
        """Effective depth through recursion."""
        return self.max_iterations * self.n_blocks

    @classmethod
    def for_code_repair_tiny(cls) -> "CodeRepairConfig":
        """Tiny config for unit testing (~1M params with small vocab)."""
        return cls(
            vocab_size=1024,
            embed_dim=64,
            n_heads=4,
            ffn_dim=256,
            n_blocks=2,
            max_iterations=4,
        )

    @classmethod
    def for_code_repair_small(cls) -> "CodeRepairConfig":
        """Small config (~9M params with 32K vocab).

        Note: With 32K BPE vocab, embeddings alone are ~8M params.
        This is the minimum practical size for code repair.
        """
        return cls(
            embed_dim=128,
            n_heads=4,
            ffn_dim=512,
            n_blocks=4,
        )

    @classmethod
    def for_code_repair_base(cls) -> "CodeRepairConfig":
        """Base config (~12M params with 32K vocab).

        Suitable for single-GPU training.
        """
        return cls(
            embed_dim=160,
            n_heads=8,
            ffn_dim=640,
            n_blocks=6,
        )

    @classmethod
    def for_code_repair_large(cls) -> "CodeRepairConfig":
        """Large config (~23M params with 32K vocab).

        Higher capacity for complex code repair tasks.
        """
        return cls(
            embed_dim=256,
            n_heads=8,
            ffn_dim=1024,
            n_blocks=6,
        )

    @classmethod
    def for_arc_agi_7m(cls) -> "CodeRepairConfig":
        """7M param config optimized for ARC-AGI (smaller vocab).

        With vocab_size=256 (for discrete ARC colors), we can achieve
        the target 7M parameter count while having meaningful model capacity.
        """
        return cls(
            vocab_size=256,  # ARC uses 10 colors + padding
            embed_dim=256,
            n_heads=8,
            ffn_dim=1024,
            n_blocks=6,
            grid_height=30,  # ARC max grid size
            grid_width=30,
        )


class CodeRepairDeepRecursion(nn.Module):
    """Deep recursion module for code repair with weight sharing.

    Key features:
    - Weight sharing across iterations (8 iterations, 6 blocks = 42 effective layers)
    - Early stopping based on confidence threshold
    - Support for gradient checkpointing
    - Attention hooks for visualization
    """

    def __init__(self, config: CodeRepairConfig):
        super().__init__()
        self.config = config
        self.max_iterations = config.max_iterations
        self.n_blocks = config.n_blocks
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

        # Weight-shared recursive blocks
        self.blocks = nn.ModuleList([
            RecursiveBlock(
                embed_dim=config.embed_dim,
                n_heads=config.n_heads,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                max_height=config.grid_height,
                max_width=config.grid_width,
                use_swiglu=True,
            )
            for _ in range(config.n_blocks)
        ])

        # Iteration controller for early stopping
        self.controller = IterationController(
            q_threshold=config.q_threshold,
            min_iterations=config.min_iterations,
            max_iterations=config.max_iterations,
        )

        # Optional Q-head for confidence prediction
        self.q_head = nn.Sequential(
            RMSNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Linear(config.embed_dim // 2, 1),
        )
        nn.init.zeros_(self.q_head[-1].weight)
        nn.init.zeros_(self.q_head[-1].bias)

        # Hooks for attention visualization
        self._attention_hooks = {}

    def _apply_block(
        self,
        block: RecursiveBlock,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply a single block with optional gradient checkpointing."""
        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                block, x, mask, use_reentrant=False
            )
        return block(x, mask)

    def _compute_q_hat(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute confidence score from hidden state."""
        # Global average pooling
        pooled = hidden.mean(dim=(1, 2))  # (batch, embed_dim)
        q_hat = self.q_head(pooled).squeeze(-1)  # (batch,)
        return q_hat

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_all_iterations: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Perform deep recursion with weight sharing.

        Args:
            x: Input tensor (batch, height, width, embed_dim)
            mask: Optional attention mask (batch, height, width)
            return_all_iterations: Whether to return all intermediate states

        Returns:
            Tuple of (output, info_dict):
                - output: Final hidden state (batch, height, width, embed_dim)
                - info_dict: Dictionary with iteration info
        """
        all_states = [] if return_all_iterations else None
        iterations_performed = 0

        for iteration in range(self.max_iterations):
            # Apply all blocks (weight sharing across iterations)
            for block_idx, block in enumerate(self.blocks):
                x = self._apply_block(block, x, mask)

            iterations_performed = iteration + 1

            # Store intermediate state if requested
            if return_all_iterations:
                all_states.append(x.detach().clone())

            # Check early stopping (only during inference)
            if not self.training:
                q_hat = self._compute_q_hat(x)
                should_stop, confidence = self.controller.should_stop(
                    x, iteration, q_hat
                )
                if should_stop:
                    break

        # Final confidence computation
        q_hat = self._compute_q_hat(x)
        confidence = torch.sigmoid(q_hat)

        info = {
            "iterations": iterations_performed,
            "confidence": confidence,
            "q_hat": q_hat,
        }

        if return_all_iterations:
            info["all_states"] = all_states

        return x, info


class CodeRepairTRM(nn.Module):
    """Tiny Recursive Model for Code Repair.

    A 7M parameter recursive reasoning architecture for code repair tasks.
    Input: 64 rows x 48 tokens grid (representing code)
    Output: Repaired code grid

    Key innovations:
    - 8 recursive iterations with weight sharing
    - 42 effective layers of depth
    - Early stopping based on confidence threshold
    - Grid-aware 2D attention with relative position encoding

    Architecture:
        Input (batch, 64, 48) token IDs
                |
                v
        Token Embedding + Position Encoding
                |
                v
        +-------------------+
        | Deep Recursion    |<---+
        | (6 weight-shared  |    | 8 iterations
        |  blocks)          |    |
        +-------------------+----+
                |
                v
        Early Stop if confident
                |
                v
        Output Head -> (batch, 64, 48, vocab) logits
    """

    def __init__(self, config: CodeRepairConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        # 2D positional encoding
        self.pos_encoding = GridPositionalEncoding(
            embed_dim=config.embed_dim,
            max_height=config.grid_height,
            max_width=config.grid_width,
            dropout=config.dropout,
        )

        # Deep recursion module
        self.recursion = CodeRepairDeepRecursion(config)

        # Output head
        self.output_norm = RMSNorm(config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

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

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass.

        Args:
            x: Input token IDs (batch, height, width) - [batch, 64, 48]
            mask: Optional attention mask (batch, height, width)
            labels: Optional target token IDs for loss computation

        Returns:
            Tuple of (logits, info):
                - logits: Output logits (batch, height, width, vocab_size)
                - info: Dictionary with iteration info and optional loss
        """
        batch_size, height, width = x.shape

        # Token embedding: (batch, height, width, embed_dim)
        emb = self.embedding(x)

        # Add positional encoding
        emb = emb + self.pos_encoding(x)

        # Deep recursion
        hidden, info = self.recursion(emb, mask)

        # Output projection
        hidden = self.output_norm(hidden)
        logits = self.output_head(hidden)

        # Compute loss if labels provided
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=0,  # Assuming 0 is pad token
            )
            info["loss"] = loss

        return logits, info

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        max_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate repaired code.

        Args:
            x: Input token IDs (batch, height, width)
            mask: Optional attention mask
            max_iterations: Override max iterations

        Returns:
            Dictionary with:
                - output: Predicted token IDs (batch, height, width)
                - logits: Output logits
                - iterations: Number of iterations performed
                - confidence: Model confidence
        """
        self.eval()

        # Temporarily override max iterations if specified
        original_max = self.recursion.max_iterations
        if max_iterations is not None:
            self.recursion.max_iterations = max_iterations

        logits, info = self.forward(x, mask)

        # Restore original
        if max_iterations is not None:
            self.recursion.max_iterations = original_max

        output = logits.argmax(dim=-1)

        return {
            "output": output,
            "logits": logits,
            "iterations": info["iterations"],
            "confidence": info["confidence"],
        }

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, config: CodeRepairConfig) -> "CodeRepairTRM":
        """Create model from configuration."""
        return cls(config)

    @classmethod
    def from_pretrained(cls, path: str) -> "CodeRepairTRM":
        """Load pretrained model."""
        checkpoint = torch.load(path, map_location="cpu")
        config = CodeRepairConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def save_pretrained(self, path: str):
        """Save model checkpoint."""
        import dataclasses
        torch.save({
            "config": dataclasses.asdict(self.config),
            "model_state_dict": self.state_dict(),
        }, path)
