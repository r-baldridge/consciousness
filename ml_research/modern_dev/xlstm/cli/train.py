#!/usr/bin/env python3
"""
xLSTM Training CLI

Command-line interface for training xLSTM models.

Usage:
    python -m xlstm.cli.train --config configs/default.yaml
    python -m xlstm.cli.train --embedding_dim 1024 --num_layers 24
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train an xLSTM model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )

    # Model architecture
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument(
        "--embedding_dim",
        type=int,
        default=768,
        help="Model embedding dimension",
    )
    model_group.add_argument(
        "--num_layers",
        type=int,
        default=12,
        help="Number of layers",
    )
    model_group.add_argument(
        "--vocab_size",
        type=int,
        default=50257,
        help="Vocabulary size",
    )
    model_group.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    model_group.add_argument(
        "--head_dim",
        type=int,
        default=64,
        help="Dimension per head",
    )
    model_group.add_argument(
        "--up_proj_factor",
        type=float,
        default=2.0,
        help="Up projection expansion factor",
    )
    model_group.add_argument(
        "--layer_types",
        type=str,
        nargs="+",
        default=None,
        help="Layer types (s or m for each layer)",
    )
    model_group.add_argument(
        "--slstm_ratio",
        type=int,
        default=4,
        help="Place sLSTM every N layers (default: 4 means 7:1 ratio)",
    )

    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--learning_rate",
        type=float,
        default=6e-4,
        help="Peak learning rate",
    )
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Micro batch size",
    )
    train_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    train_group.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    train_group.add_argument(
        "--max_steps",
        type=int,
        default=100000,
        help="Maximum training steps",
    )
    train_group.add_argument(
        "--warmup_steps",
        type=int,
        default=2000,
        help="Warmup steps",
    )
    train_group.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="Weight decay",
    )
    train_group.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm",
    )

    # Data
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--train_data",
        type=str,
        default=None,
        help="Path to training data",
    )
    data_group.add_argument(
        "--val_data",
        type=str,
        default=None,
        help="Path to validation data",
    )
    data_group.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer to use",
    )

    # Checkpointing
    ckpt_group = parser.add_argument_group("Checkpointing")
    ckpt_group.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints",
    )
    ckpt_group.add_argument(
        "--save_interval",
        type=int,
        default=1000,
        help="Save checkpoint every N steps",
    )
    ckpt_group.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )

    # Hardware
    hw_group = parser.add_argument_group("Hardware")
    hw_group.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for training",
    )
    hw_group.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile",
    )
    hw_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train on",
    )

    # Logging
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log every N steps",
    )
    log_group.add_argument(
        "--eval_interval",
        type=int,
        default=500,
        help="Evaluate every N steps",
    )
    log_group.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project name",
    )
    log_group.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(file_config: dict, args: argparse.Namespace) -> dict:
    """Merge file configuration with command-line arguments.

    Command-line arguments override file configuration.

    Args:
        file_config: Configuration from file.
        args: Command-line arguments.

    Returns:
        Merged configuration dictionary.
    """
    config = file_config.copy()

    # Override with command-line arguments if provided
    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is not None and key != "config":
            config[key] = value

    return config


def setup_training(config: dict):
    """Set up training environment.

    Args:
        config: Training configuration.

    Returns:
        Tuple of (model, optimizer, dataloader, scheduler).
    """
    logger.info("Setting up training environment...")
    logger.info(f"Configuration: {config}")

    try:
        import torch
        from ..src.model import xLSTMConfig, xLSTM
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please install PyTorch: pip install torch")
        sys.exit(1)

    # Handle layer_types
    layer_types = config.get("layer_types")
    if layer_types is None:
        # Generate from slstm_ratio
        num_layers = config.get("num_layers", 12)
        slstm_ratio = config.get("slstm_ratio", 4)
        layer_types = [
            "s" if (i + 1) % slstm_ratio == 0 else "m"
            for i in range(num_layers)
        ]

    # Create model config
    model_config = xLSTMConfig(
        vocab_size=config.get("vocab_size", 50257),
        embedding_dim=config.get("embedding_dim", 768),
        num_layers=config.get("num_layers", 12),
        layer_types=layer_types,
        num_heads=config.get("num_heads", 8),
        head_dim=config.get("head_dim", 64),
        up_proj_factor=config.get("up_proj_factor", 2.0),
    )

    # Create model
    logger.info("Creating xLSTM model...")
    model = xLSTM(model_config)

    # Move to device
    device = config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    model = model.to(device)

    # Log model info
    num_params = model.num_parameters()
    layer_stats = model.get_layer_stats()
    logger.info(f"Model created with {num_params:,} parameters")
    logger.info(f"Layer stats: {layer_stats}")

    return model, None, None, None


def save_checkpoint(model, optimizer, step: int, config: dict, output_dir: str):
    """Save a training checkpoint.

    Args:
        model: The xLSTM model.
        optimizer: Optimizer.
        step: Current training step.
        config: Training configuration.
        output_dir: Directory to save checkpoint.
    """
    import torch

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"checkpoint_{step}.pt")

    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "config": config,
    }, checkpoint_path)

    logger.info(f"Saved checkpoint to {checkpoint_path}")


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    if args.config:
        file_config = load_config(args.config)
        config = merge_configs(file_config, args)
    else:
        config = vars(args)

    logger.info("=" * 60)
    logger.info("xLSTM Training")
    logger.info("=" * 60)

    # Setup training
    model, optimizer, dataloader, scheduler = setup_training(config)

    # Training loop placeholder
    logger.info("Starting training loop (placeholder)...")
    logger.info("Training not yet implemented. Model structure validated.")

    max_steps = min(config.get("max_steps", 10), 10)

    for step in range(max_steps):
        metrics = {"loss": 0.0, "step": step}

        if step % config.get("log_interval", 10) == 0:
            logger.info(f"Step {step}: loss={metrics['loss']:.4f}")

        if step > 0 and step % config.get("save_interval", 1000) == 0:
            save_checkpoint(
                model, optimizer, step, config,
                config.get("output_dir", "./outputs")
            )

    logger.info("Training complete (placeholder).")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
