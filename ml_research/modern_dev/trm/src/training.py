"""
TRM Training Loop

Comprehensive training infrastructure for the Transformer Repair Model.
Includes gradient accumulation, mixed precision, curriculum learning, and logging.
"""

import os
import math
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Optional imports for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from .model import TRM, CodeRepairTRM, TRMConfig


class MixedPrecision(str, Enum):
    """Mixed precision training modes."""
    NONE = "none"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass
class TrainingConfig:
    """
    Configuration for TRM training.

    Attributes:
        learning_rate: Base learning rate
        batch_size: Training batch size
        num_epochs: Number of training epochs
        gradient_accumulation_steps: Steps to accumulate gradients before update
        mixed_precision: Mixed precision mode (none/fp16/bf16)
        max_grad_norm: Maximum gradient norm for clipping
        warmup_steps: Number of warmup steps for learning rate
        weight_decay: Weight decay for AdamW optimizer
        checkpoint_dir: Directory for saving checkpoints
        log_dir: Directory for tensorboard logs
        eval_steps: Steps between evaluations
        save_steps: Steps between checkpoint saves
        logging_steps: Steps between logging
        max_steps: Maximum training steps (overrides num_epochs if set)
        seed: Random seed for reproducibility
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
        wandb_run_name: W&B run name
        use_tensorboard: Whether to use TensorBoard logging
        early_stopping_patience: Epochs to wait before early stopping (0 to disable)
        save_total_limit: Maximum checkpoints to keep (0 for unlimited)
    """
    # Core training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    mixed_precision: MixedPrecision = MixedPrecision.NONE
    max_grad_norm: float = 1.0

    # Learning rate schedule
    warmup_steps: int = 1000
    weight_decay: float = 0.01

    # Directories
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Evaluation and saving
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    max_steps: int = 0  # 0 means use num_epochs

    # Reproducibility
    seed: int = 42

    # Logging
    use_wandb: bool = False
    wandb_project: str = "trm-training"
    wandb_run_name: Optional[str] = None
    use_tensorboard: bool = True

    # Training control
    early_stopping_patience: int = 0
    save_total_limit: int = 3

    def __post_init__(self):
        """Validate configuration."""
        if isinstance(self.mixed_precision, str):
            self.mixed_precision = MixedPrecision(self.mixed_precision)

        # Create directories
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class CurriculumConfig:
    """
    Configuration for curriculum learning.

    Attributes:
        initial_seq_len: Starting sequence length
        max_seq_len: Maximum sequence length
        seq_len_increment: Sequence length increase per stage
        stages: Number of curriculum stages
        difficulty_warmup_epochs: Epochs before increasing difficulty
        initial_difficulty: Starting difficulty (0.0-1.0)
    """
    initial_seq_len: int = 128
    max_seq_len: int = 2048
    seq_len_increment: int = 128
    stages: int = 8
    difficulty_warmup_epochs: int = 1
    initial_difficulty: float = 0.5


class CurriculumScheduler:
    """
    Curriculum scheduler for gradual training progression.

    Manages:
    1. Sequence length progression (shorter -> longer sequences)
    2. Task difficulty progression (simpler -> more complex repairs)
    """

    def __init__(self, config: CurriculumConfig):
        """
        Args:
            config: Curriculum configuration
        """
        self.config = config
        self.current_stage = 0
        self.current_step = 0
        self.current_epoch = 0

        # Sequence length schedule
        self.seq_len_schedule = self._build_seq_len_schedule()

        # Difficulty schedule
        self.difficulty_schedule = self._build_difficulty_schedule()

    def _build_seq_len_schedule(self) -> List[int]:
        """Build sequence length progression schedule."""
        schedule = []
        current_len = self.config.initial_seq_len

        for _ in range(self.config.stages):
            schedule.append(min(current_len, self.config.max_seq_len))
            current_len += self.config.seq_len_increment

        return schedule

    def _build_difficulty_schedule(self) -> List[float]:
        """Build difficulty progression schedule."""
        schedule = []
        difficulty_increment = (1.0 - self.config.initial_difficulty) / max(self.config.stages - 1, 1)

        for i in range(self.config.stages):
            difficulty = min(
                self.config.initial_difficulty + i * difficulty_increment,
                1.0
            )
            schedule.append(difficulty)

        return schedule

    def step(self, epoch: int, step: int):
        """
        Update curriculum state.

        Args:
            epoch: Current epoch
            step: Current step within epoch
        """
        self.current_epoch = epoch
        self.current_step = step

        # Update stage based on epoch
        if self.config.difficulty_warmup_epochs > 0:
            self.current_stage = min(
                epoch // self.config.difficulty_warmup_epochs,
                self.config.stages - 1
            )

    def get_current_config(self) -> Dict[str, Any]:
        """
        Get current curriculum configuration.

        Returns:
            Dictionary with current sequence length and difficulty
        """
        return {
            'seq_len': self.seq_len_schedule[self.current_stage],
            'difficulty': self.difficulty_schedule[self.current_stage],
            'stage': self.current_stage,
            'total_stages': self.config.stages,
        }

    def get_seq_len(self) -> int:
        """Get current sequence length."""
        return self.seq_len_schedule[self.current_stage]

    def get_difficulty(self) -> float:
        """Get current task difficulty."""
        return self.difficulty_schedule[self.current_stage]


class TRMTrainer:
    """
    Trainer for TRM models.

    Features:
    - Gradient accumulation for large effective batch sizes
    - Mixed precision training with torch.cuda.amp
    - Gradient clipping
    - Learning rate scheduling (linear warmup + cosine decay)
    - Periodic validation
    - Best model tracking
    - Wandb/TensorBoard logging
    - Curriculum learning support
    """

    def __init__(
        self,
        model: Union[TRM, CodeRepairTRM],
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        curriculum: Optional[CurriculumScheduler] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: TRM model to train
            config: Training configuration
            train_loader: Training data loader
            val_loader: Optional validation data loader
            curriculum: Optional curriculum scheduler
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.curriculum = curriculum

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # Setup mixed precision
        self.scaler = self._setup_scaler()

        # Setup logging
        self.logger = self._setup_logger()
        self.writer = self._setup_tensorboard()
        self._setup_wandb()

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history: List[Dict[str, float]] = []

        # Checkpoint tracking
        self.saved_checkpoints: List[str] = []

    def _setup_optimizer(self) -> AdamW:
        """
        Setup AdamW optimizer with weight decay.

        Applies weight decay to all parameters except:
        - Bias terms
        - LayerNorm/RMSNorm weights

        Returns:
            Configured AdamW optimizer
        """
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # No weight decay for biases and norm layers
            if 'bias' in name or 'norm' in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

        optimizer = AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        return optimizer

    def _setup_scheduler(self) -> LambdaLR:
        """
        Setup learning rate scheduler.

        Uses linear warmup followed by cosine decay.

        Returns:
            Configured learning rate scheduler
        """
        # Calculate total steps
        if self.config.max_steps > 0:
            total_steps = self.config.max_steps
        else:
            steps_per_epoch = len(self.train_loader) // self.config.gradient_accumulation_steps
            total_steps = steps_per_epoch * self.config.num_epochs

        warmup_steps = self.config.warmup_steps

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(self.optimizer, lr_lambda)
        return scheduler

    def _setup_scaler(self) -> Optional[torch.amp.GradScaler]:
        """
        Setup gradient scaler for mixed precision.

        Returns:
            GradScaler if using mixed precision, None otherwise
        """
        if self.config.mixed_precision == MixedPrecision.NONE:
            return None

        return torch.amp.GradScaler('cuda')

    def _setup_logger(self) -> logging.Logger:
        """Setup Python logger."""
        logger = logging.getLogger("TRMTrainer")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)

        return logger

    def _setup_tensorboard(self) -> Optional[Any]:
        """Setup TensorBoard writer."""
        if not self.config.use_tensorboard or not TENSORBOARD_AVAILABLE:
            return None

        return SummaryWriter(log_dir=self.config.log_dir)

    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        if not self.config.use_wandb or not WANDB_AVAILABLE:
            return

        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config=asdict(self.config),
        )
        wandb.watch(self.model)

    def _get_autocast_context(self):
        """Get autocast context for mixed precision."""
        if self.config.mixed_precision == MixedPrecision.FP16:
            return torch.autocast('cuda', dtype=torch.float16)
        elif self.config.mixed_precision == MixedPrecision.BF16:
            return torch.autocast('cuda', dtype=torch.bfloat16)
        else:
            return torch.autocast('cuda', enabled=False)

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        accumulated_loss = 0.0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._move_to_device(batch)

            # Update curriculum if present
            if self.curriculum is not None:
                self.curriculum.step(self.current_epoch, batch_idx)

            # Forward pass with mixed precision
            with self._get_autocast_context():
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    tree_mask=batch.get('tree_mask'),
                    depth=batch.get('depth'),
                    relationship_matrix=batch.get('relationship_matrix'),
                )

                # Compute loss
                if isinstance(self.model, CodeRepairTRM):
                    losses = self.model.compute_loss(outputs, batch)
                    loss = losses['total_loss']
                else:
                    loss = self._compute_default_loss(outputs, batch)

                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += loss.item()

            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                # Logging
                self.global_step += 1
                total_loss += accumulated_loss
                num_batches += 1

                if self.global_step % self.config.logging_steps == 0:
                    self._log_metrics({
                        'train/loss': accumulated_loss,
                        'train/lr': self.scheduler.get_last_lr()[0],
                        'train/epoch': self.current_epoch,
                        'train/step': self.global_step,
                    })

                accumulated_loss = 0.0

                # Validation
                if self.val_loader is not None and self.global_step % self.config.eval_steps == 0:
                    val_metrics = self.validate()
                    self._log_metrics(val_metrics)

                    # Check for best model
                    if val_metrics['val/loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val/loss']
                        self.save_checkpoint(is_best=True)
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1

                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

                # Check max steps
                if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                    break

        avg_loss = total_loss / max(num_batches, 1)
        return {'train/epoch_loss': avg_loss}

    def train(self) -> Dict[str, float]:
        """
        Full training loop.

        Returns:
            Dictionary of final training metrics
        """
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {self.model.get_num_params():,}")

        best_metrics = {}

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            # Log curriculum state
            if self.curriculum is not None:
                curr_config = self.curriculum.get_current_config()
                self.logger.info(
                    f"Curriculum: stage {curr_config['stage'] + 1}/{curr_config['total_stages']}, "
                    f"seq_len={curr_config['seq_len']}, difficulty={curr_config['difficulty']:.2f}"
                )

            # Train epoch
            epoch_metrics = self.train_epoch()
            self.training_history.append(epoch_metrics)

            # Validation
            if self.val_loader is not None:
                val_metrics = self.validate()
                epoch_metrics.update(val_metrics)

                if val_metrics['val/loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val/loss']
                    best_metrics = epoch_metrics.copy()
                    self.save_checkpoint(is_best=True)
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

            self.logger.info(f"Epoch {epoch + 1} metrics: {epoch_metrics}")

            # Early stopping
            if (self.config.early_stopping_patience > 0 and
                    self.patience_counter >= self.config.early_stopping_patience):
                self.logger.info(f"Early stopping after {epoch + 1} epochs")
                break

            # Check max steps
            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                self.logger.info(f"Reached max steps ({self.config.max_steps})")
                break

        # Save final checkpoint
        self.save_checkpoint(is_final=True)

        # Close logging
        if self.writer is not None:
            self.writer.close()
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()

        return best_metrics if best_metrics else epoch_metrics

    def validate(self) -> Dict[str, float]:
        """
        Run validation pass.

        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_to_device(batch)

                with self._get_autocast_context():
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask'),
                        tree_mask=batch.get('tree_mask'),
                        depth=batch.get('depth'),
                        relationship_matrix=batch.get('relationship_matrix'),
                    )

                    if isinstance(self.model, CodeRepairTRM):
                        losses = self.model.compute_loss(outputs, batch)
                        loss = losses['total_loss']
                    else:
                        loss = self._compute_default_loss(outputs, batch)

                total_loss += loss.item()
                num_batches += 1

        self.model.train()

        avg_loss = total_loss / max(num_batches, 1)
        return {'val/loss': avg_loss}

    def save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """
        Save model checkpoint.

        Args:
            is_best: Whether this is the best model so far
            is_final: Whether this is the final checkpoint
        """
        # Determine checkpoint name
        if is_best:
            checkpoint_name = "checkpoint_best.pt"
        elif is_final:
            checkpoint_name = "checkpoint_final.pt"
        else:
            checkpoint_name = f"checkpoint_step_{self.global_step}.pt"

        checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint_name

        # Save checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
            'training_history': self.training_history,
        }

        if hasattr(self.model, 'config'):
            checkpoint['model_config'] = asdict(self.model.config)

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Track saved checkpoints (for cleanup)
        if not is_best and not is_final:
            self.saved_checkpoints.append(str(checkpoint_path))

            # Remove old checkpoints if over limit
            if (self.config.save_total_limit > 0 and
                    len(self.saved_checkpoints) > self.config.save_total_limit):
                old_checkpoint = self.saved_checkpoints.pop(0)
                if Path(old_checkpoint).exists():
                    Path(old_checkpoint).unlink()
                    self.logger.info(f"Removed old checkpoint: {old_checkpoint}")

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        """
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.scaler is not None and checkpoint['scaler_state_dict'] is not None:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint.get('training_history', [])

        self.logger.info(f"Loaded checkpoint from step {self.global_step}")

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _compute_default_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute default loss for base TRM model.

        Args:
            outputs: Model outputs
            batch: Input batch

        Returns:
            Combined loss tensor
        """
        total_loss = torch.tensor(0.0, device=self.device)

        # Token prediction loss
        if 'target_tokens' in batch:
            token_loss = F.cross_entropy(
                outputs['token_logits'].view(-1, outputs['token_logits'].size(-1)),
                batch['target_tokens'].view(-1),
                ignore_index=-100,
            )
            total_loss = total_loss + token_loss

        # Error detection loss
        if 'target_errors' in batch:
            error_loss = F.binary_cross_entropy(
                outputs['error_probs'].squeeze(-1),
                batch['target_errors'].float(),
            )
            total_loss = total_loss + error_loss

        return total_loss

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to all configured destinations."""
        # Console logging
        self.logger.info(f"Step {self.global_step}: {metrics}")

        # TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, self.global_step)

        # Weights & Biases
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log(metrics, step=self.global_step)


def create_synthetic_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Create a synthetic batch for testing.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        vocab_size: Vocabulary size
        device: Device to create tensors on

    Returns:
        Dictionary of synthetic batch tensors
    """
    return {
        'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.bool, device=device),
        'target_tokens': torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        'target_spans': torch.zeros(batch_size, seq_len, 2, device=device),
        'target_ops': torch.randint(0, 3, (batch_size, seq_len), device=device),
        'target_errors': torch.randint(0, 2, (batch_size, seq_len), device=device).float(),
        'target_error_types': torch.randint(0, 10, (batch_size, seq_len), device=device),
    }


class SyntheticDataset(torch.utils.data.Dataset):
    """Synthetic dataset for testing."""

    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        vocab_size: int,
    ):
        """
        Args:
            num_samples: Number of synthetic samples
            seq_len: Sequence length
            vocab_size: Vocabulary size
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': torch.randint(0, self.vocab_size, (self.seq_len,)),
            'attention_mask': torch.ones(self.seq_len, dtype=torch.bool),
            'target_tokens': torch.randint(0, self.vocab_size, (self.seq_len,)),
            'target_spans': torch.zeros(self.seq_len, 2),
            'target_ops': torch.randint(0, 3, (self.seq_len,)),
            'target_errors': torch.randint(0, 2, (self.seq_len,)).float(),
            'target_error_types': torch.randint(0, 10, (self.seq_len,)),
        }
