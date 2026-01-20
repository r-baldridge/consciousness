#!/usr/bin/env python3
"""
CTM Training CLI

Command-line interface for training Continuous Thought Machine models.

Usage:
    python -m ctm.cli.train --config configs/default.yaml
    python -m ctm.cli.train --config configs/default.yaml --batch-size 64 --lr 1e-4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Merge CLI arguments into configuration.

    CLI arguments override config file values.

    Args:
        config: Base configuration dictionary.
        args: Parsed CLI arguments.

    Returns:
        Merged configuration dictionary.
    """
    # Override training params if specified
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed

    # Override model params
    if args.num_neurons is not None:
        config["model"]["num_neurons"] = args.num_neurons
    if args.max_internal_steps is not None:
        config["model"]["max_internal_steps"] = args.max_internal_steps

    # Override paths
    if args.data_dir is not None:
        config["data"]["data_dir"] = args.data_dir
    if args.checkpoint_dir is not None:
        config["logging"]["checkpoint_dir"] = args.checkpoint_dir

    # Experiment name
    if args.experiment_name is not None:
        config["experiment"]["name"] = args.experiment_name

    return config


def setup_environment(config: Dict[str, Any]) -> None:
    """Setup training environment.

    Args:
        config: Configuration dictionary.
    """
    import torch

    # Set random seed
    seed = config["experiment"]["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic mode
    if config["experiment"].get("deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Environment setup complete. Seed: {seed}")


def build_model(config: Dict[str, Any]):
    """Build CTM model from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Initialized CTM model.
    """
    from ..src.model import CTM, CTMConfig

    model_config = CTMConfig(
        hidden_dim=config["model"]["hidden_dim"],
        num_neurons=config["model"]["num_neurons"],
        history_length=config["model"]["history_length"],
        max_internal_steps=config["model"]["max_internal_steps"],
        sync_window=config["model"]["sync_window"],
        num_sync_heads=config["model"]["num_sync_heads"],
        neuron_activation=config["model"]["neuron_activation"],
        dropout=config["model"]["dropout"],
        use_adaptive_halt=config["model"]["use_adaptive_halt"],
        halt_threshold=config["model"]["halt_threshold"],
        input_dim=config["task"]["input_dim"],
        output_dim=config["task"]["output_dim"],
    )

    model = CTM(model_config)
    logger.info(f"Built CTM model with {sum(p.numel() for p in model.parameters()):,} parameters")

    return model


def build_optimizer(model, config: Dict[str, Any]):
    """Build optimizer from configuration.

    Args:
        model: CTM model.
        config: Configuration dictionary.

    Returns:
        Optimizer instance.
    """
    import torch.optim as optim

    training_config = config["training"]

    if training_config["optimizer"] == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
            betas=(training_config["adam_beta1"], training_config["adam_beta2"]),
            eps=training_config["adam_eps"],
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_config["learning_rate"],
        )

    return optimizer


def build_dataloader(config: Dict[str, Any], split: str = "train"):
    """Build dataloader from configuration.

    Args:
        config: Configuration dictionary.
        split: Data split ("train", "val", "test").

    Returns:
        DataLoader instance.

    Note:
        This is a placeholder. Actual implementation would load
        the specified dataset.
    """
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    # Placeholder: Create dummy data
    # In real implementation, load actual dataset
    logger.warning("Using placeholder dummy data. Implement actual data loading.")

    batch_size = config["training"]["batch_size"]
    input_dim = config["task"]["input_dim"]
    output_dim = config["task"]["output_dim"]

    # Create dummy tensors
    num_samples = 1000 if split == "train" else 100
    x = torch.randn(num_samples, input_dim)
    y = torch.randint(0, output_dim, (num_samples,))

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
    )

    return dataloader


def train_step(
    model,
    batch,
    optimizer,
    config: Dict[str, Any],
    device: str,
):
    """Execute single training step.

    Args:
        model: CTM model.
        batch: Training batch (inputs, targets).
        optimizer: Optimizer instance.
        config: Configuration dictionary.
        device: Device to run on.

    Returns:
        Loss value.
    """
    import torch
    import torch.nn.functional as F

    model.train()
    optimizer.zero_grad()

    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Forward pass
    outputs = model(inputs)
    logits = outputs["output"]

    # Compute loss
    loss = F.cross_entropy(logits, targets)

    # Add sync loss if configured
    sync_weight = config["ctm"].get("sync_loss_weight", 0.0)
    if sync_weight > 0:
        # Placeholder: Add synchronization regularization loss
        pass

    # Add ponder cost if using adaptive halting
    if config["model"]["use_adaptive_halt"]:
        ponder_weight = config["ctm"].get("ponder_cost_weight", 0.01)
        # Placeholder: Add ponder cost
        pass

    # Backward pass
    loss.backward()

    # Gradient clipping
    if config["training"]["gradient_clip_norm"] > 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config["training"]["gradient_clip_norm"],
        )

    optimizer.step()

    return loss.item()


def evaluate(
    model,
    dataloader,
    config: Dict[str, Any],
    device: str,
) -> Dict[str, float]:
    """Evaluate model on validation set.

    Args:
        model: CTM model.
        dataloader: Validation dataloader.
        config: Configuration dictionary.
        device: Device to run on.

    Returns:
        Dictionary of evaluation metrics.
    """
    import torch

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            logits = outputs["output"]

            # Compute accuracy
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

            # Compute loss
            loss = torch.nn.functional.cross_entropy(logits, targets)
            total_loss += loss.item() * targets.size(0)

    metrics = {
        "accuracy": total_correct / total_samples,
        "loss": total_loss / total_samples,
    }

    return metrics


def train(config: Dict[str, Any]) -> None:
    """Main training loop.

    Args:
        config: Configuration dictionary.
    """
    import torch

    # Setup
    device = config["hardware"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    setup_environment(config)

    # Build components
    model = build_model(config)
    model = model.to(device)

    optimizer = build_optimizer(model, config)
    train_loader = build_dataloader(config, "train")
    val_loader = build_dataloader(config, "val")

    # Optional: Compile model for PyTorch 2.0+
    if config["hardware"].get("compile_model", False):
        try:
            model = torch.compile(model)
            logger.info("Model compiled with torch.compile")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    # Training loop
    global_step = 0
    max_steps = config["training"]["max_steps"]
    log_interval = config["logging"]["log_interval"]
    eval_interval = config["evaluation"]["eval_interval"]
    save_interval = config["logging"]["save_interval"]

    logger.info(f"Starting training for {max_steps} steps")

    while global_step < max_steps:
        for batch in train_loader:
            if global_step >= max_steps:
                break

            loss = train_step(model, batch, optimizer, config, device)
            global_step += 1

            # Logging
            if global_step % log_interval == 0:
                logger.info(f"Step {global_step}/{max_steps} - Loss: {loss:.4f}")

            # Evaluation
            if global_step % eval_interval == 0:
                metrics = evaluate(model, val_loader, config, device)
                logger.info(
                    f"Eval @ step {global_step} - "
                    f"Accuracy: {metrics['accuracy']:.4f}, "
                    f"Loss: {metrics['loss']:.4f}"
                )

            # Checkpointing
            if global_step % save_interval == 0:
                checkpoint_dir = Path(config["logging"]["checkpoint_dir"])
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f"checkpoint_{global_step}.pt"
                model.save_pretrained(str(checkpoint_path))
                logger.info(f"Saved checkpoint to {checkpoint_path}")

    logger.info("Training complete!")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train Continuous Thought Machine (CTM) model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    # Training overrides
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )

    # Model overrides
    parser.add_argument(
        "--num-neurons",
        type=int,
        default=None,
        help="Number of neurons (overrides config)",
    )
    parser.add_argument(
        "--max-internal-steps",
        type=int,
        default=None,
        help="Maximum internal time steps (overrides config)",
    )

    # Path overrides
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory (overrides config)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory (overrides config)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (overrides config)",
    )

    # Utility arguments
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without training",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Merge with CLI arguments
    config = merge_config_with_args(config, args)

    # Dry run mode
    if args.dry_run:
        import json
        print("Configuration:")
        print(json.dumps(config, indent=2))
        return

    # Run training
    train(config)


if __name__ == "__main__":
    main()
