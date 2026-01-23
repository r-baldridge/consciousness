"""
TRM Training CLI

Train Tiny Recursive Model on reasoning tasks.

Usage:
    python -m trm.cli.train --config configs/sudoku.yaml
    python -m trm.cli.train --task sudoku --epochs 100
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from ..src.model import TRM, TRMConfig


class ExponentialMovingAverage:
    """Exponential Moving Average for model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class DummyDataset(Dataset):
    """Placeholder dataset for demonstration."""

    def __init__(self, grid_size: int = 9, vocab_size: int = 10, size: int = 1000):
        self.grid_size = grid_size
        self.vocab_size = vocab_size
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate random puzzle/solution pair (placeholder)
        puzzle = torch.randint(0, self.vocab_size, (self.grid_size, self.grid_size))
        solution = torch.randint(0, self.vocab_size, (self.grid_size, self.grid_size))
        return puzzle, solution


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """Create AdamW optimizer with weight decay."""
    # Separate parameters that should/shouldn't have weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create learning rate scheduler with warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: TRM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: Optional[ExponentialMovingAverage],
    device: torch.device,
    gradient_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_q_loss = 0.0
    total_accuracy = 0.0
    total_steps = 0
    num_batches = 0

    iterator = tqdm(dataloader, desc="Training") if HAS_TQDM else dataloader

    for batch in iterator:
        puzzles, solutions = batch
        puzzles = puzzles.to(device)
        solutions = solutions.to(device)

        optimizer.zero_grad()

        # Training step with deep supervision
        output = model.train_step(puzzles, solutions)

        loss = output["loss"]
        loss.backward()

        # Gradient clipping
        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()
        scheduler.step()

        if ema is not None:
            ema.update()

        total_loss += output["loss"].item()
        total_ce_loss += output["ce_loss"].item()
        total_q_loss += output["q_loss"].item()
        total_accuracy += output["accuracy"].item()
        total_steps += output["steps"]
        num_batches += 1

        if HAS_TQDM:
            iterator.set_postfix({
                "loss": f"{output['loss'].item():.4f}",
                "acc": f"{output['accuracy'].item():.2%}",
                "steps": output["steps"],
            })

    return {
        "loss": total_loss / num_batches,
        "ce_loss": total_ce_loss / num_batches,
        "q_loss": total_q_loss / num_batches,
        "accuracy": total_accuracy / num_batches,
        "avg_steps": total_steps / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: TRM,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on dataset."""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_steps = 0

    iterator = tqdm(dataloader, desc="Evaluating") if HAS_TQDM else dataloader

    for batch in iterator:
        puzzles, solutions = batch
        puzzles = puzzles.to(device)
        solutions = solutions.to(device)

        # Solve puzzles
        output = model.solve(puzzles)

        # Check accuracy
        preds = output["solution"]
        if solutions.dim() == 3:
            solutions = solutions.view(solutions.shape[0], -1)

        correct = (preds == solutions).all(dim=-1).sum().item()
        total_correct += correct
        total_samples += puzzles.shape[0]
        total_steps += output["steps"]

    return {
        "accuracy": total_correct / total_samples,
        "avg_steps": total_steps / len(dataloader),
    }


def main():
    parser = argparse.ArgumentParser(description="Train TRM model")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--task", type=str, default="sudoku", choices=["sudoku", "maze", "arc_agi"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Load config
    if args.config:
        config_dict = load_config(args.config)
    else:
        config_dict = {}

    # Create model config
    if args.task == "sudoku":
        config = TRMConfig.for_sudoku()
    elif args.task == "maze":
        config = TRMConfig.for_maze()
    elif args.task == "arc_agi":
        config = TRMConfig.for_arc_agi()
    else:
        config = TRMConfig()

    # Create model
    device = torch.device(args.device)
    model = TRM(config).to(device)

    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Effective depth: {config.effective_depth} layers")

    # Create dummy dataset (replace with real data)
    train_dataset = DummyDataset(
        grid_size=config.grid_size,
        vocab_size=config.vocab_size,
        size=10000,
    )
    val_dataset = DummyDataset(
        grid_size=config.grid_size,
        vocab_size=config.vocab_size,
        size=1000,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Create optimizer and scheduler
    total_steps = len(train_loader) * args.epochs
    optimizer = create_optimizer(model, lr=args.lr)
    scheduler = create_scheduler(optimizer, args.warmup_steps, total_steps)

    # EMA
    ema = ExponentialMovingAverage(model, decay=args.ema_decay)

    # Training loop
    best_accuracy = 0.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, ema, device
        )
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.2%}, "
              f"Steps: {train_metrics['avg_steps']:.1f}")

        # Evaluate with EMA weights
        ema.apply_shadow()
        val_metrics = evaluate(model, val_loader, device)
        ema.restore()

        print(f"Val - Acc: {val_metrics['accuracy']:.2%}, "
              f"Steps: {val_metrics['avg_steps']:.1f}")

        # Save best model
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            ema.apply_shadow()
            model.save_pretrained(save_dir / "best_model.pt")
            ema.restore()
            print(f"New best model saved! Accuracy: {best_accuracy:.2%}")

    print(f"\nTraining complete. Best accuracy: {best_accuracy:.2%}")


if __name__ == "__main__":
    main()
