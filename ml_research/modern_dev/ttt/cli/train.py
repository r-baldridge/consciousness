#!/usr/bin/env python3
"""
TTT Training CLI

Command-line interface for training Test-Time Training language models.

Usage:
    python -m ttt.cli.train --config configs/default.yaml
    python -m ttt.cli.train --config configs/default.yaml --batch-size 64 --ttt-type mlp
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

    Args:
        config: Base configuration dictionary.
        args: Parsed CLI arguments.

    Returns:
        Merged configuration dictionary.
    """
    # Override training params
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed

    # Override model params
    if args.hidden_dim is not None:
        config["model"]["hidden_dim"] = args.hidden_dim
    if args.num_layers is not None:
        config["model"]["num_layers"] = args.num_layers
    if args.ttt_type is not None:
        config["model"]["ttt_type"] = args.ttt_type
    if args.ttt_lr is not None:
        config["model"]["ttt_learning_rate"] = args.ttt_lr
    if args.mini_batch_size is not None:
        config["model"]["mini_batch_size"] = args.mini_batch_size

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
    """Build TTT model from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Initialized TTT model.
    """
    from ..src.model import TTTLanguageModel, TTTConfig

    model_config = TTTConfig(
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        vocab_size=config["model"]["vocab_size"],
        max_seq_len=config["model"]["max_seq_len"],
        ttt_type=config["model"]["ttt_type"],
        mlp_hidden_dim=config["model"]["mlp_hidden_dim"],
        ttt_learning_rate=config["model"]["ttt_learning_rate"],
        mini_batch_size=config["model"]["mini_batch_size"],
        use_rope=config["model"]["use_rope"],
        rope_base=config["model"]["rope_base"],
        layer_norm_eps=config["model"]["layer_norm_eps"],
        dropout=config["model"]["dropout"],
        initializer_range=config["model"]["initializer_range"],
        tie_word_embeddings=config["model"]["tie_word_embeddings"],
    )

    model = TTTLanguageModel(model_config)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Built TTT model with {num_params:,} parameters")
    logger.info(f"  TTT type: {config['model']['ttt_type']}")
    logger.info(f"  Hidden dim: {config['model']['hidden_dim']}")
    logger.info(f"  Num layers: {config['model']['num_layers']}")

    return model


def build_optimizer(model, config: Dict[str, Any]):
    """Build optimizer from configuration.

    Args:
        model: TTT model.
        config: Configuration dictionary.

    Returns:
        Optimizer instance.
    """
    import torch.optim as optim

    training_config = config["training"]

    # Separate parameters for weight decay
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "layer_norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": training_config["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if training_config["optimizer"] == "adamw":
        optimizer = optim.AdamW(
            param_groups,
            lr=training_config["learning_rate"],
            betas=(training_config["adam_beta1"], training_config["adam_beta2"]),
            eps=training_config["adam_eps"],
        )
    else:
        optimizer = optim.Adam(
            param_groups,
            lr=training_config["learning_rate"],
        )

    return optimizer


def build_scheduler(optimizer, config: Dict[str, Any]):
    """Build learning rate scheduler.

    Args:
        optimizer: Optimizer instance.
        config: Configuration dictionary.

    Returns:
        LR scheduler instance.
    """
    import torch.optim.lr_scheduler as lr_scheduler

    training_config = config["training"]
    total_steps = training_config["max_steps"]
    warmup_steps = training_config["warmup_steps"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max(training_config["lr_min_ratio"], 0.5 * (1.0 + math.cos(math.pi * progress)))

    import math
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return scheduler


def build_dataloader(config: Dict[str, Any], split: str = "train"):
    """Build dataloader from configuration.

    Args:
        config: Configuration dictionary.
        split: Data split ("train", "val").

    Returns:
        DataLoader instance.

    Note:
        This is a placeholder. Actual implementation would load
        tokenized text data.
    """
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    logger.warning("Using placeholder dummy data. Implement actual data loading.")

    batch_size = config["training"]["batch_size"]
    seq_len = config["model"]["max_seq_len"]
    vocab_size = config["model"]["vocab_size"]

    # Create dummy tokenized sequences
    num_samples = 10000 if split == "train" else 1000
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))

    dataset = TensorDataset(input_ids)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        drop_last=True,
    )

    return dataloader


def train_step(
    model,
    batch,
    optimizer,
    scheduler,
    config: Dict[str, Any],
    device: str,
    scaler=None,
):
    """Execute single training step.

    Args:
        model: TTT model.
        batch: Training batch.
        optimizer: Optimizer instance.
        scheduler: LR scheduler.
        config: Configuration dictionary.
        device: Device to run on.
        scaler: Optional gradient scaler for mixed precision.

    Returns:
        Dictionary of training metrics.
    """
    import torch

    model.train()
    optimizer.zero_grad()

    # Unpack batch
    (input_ids,) = batch
    input_ids = input_ids.to(device)

    # Labels are shifted input_ids
    labels = input_ids.clone()

    # Forward pass with mixed precision
    use_amp = config["training"].get("bf16", False) or config["training"].get("fp16", False)

    if use_amp and scaler is not None:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if config["training"].get("bf16") else torch.float16):
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]

        scaler.scale(loss).backward()

        # Gradient clipping
        if config["training"]["gradient_clip_norm"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["training"]["gradient_clip_norm"],
            )

        scaler.step(optimizer)
        scaler.update()
    else:
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()

        # Gradient clipping
        if config["training"]["gradient_clip_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["training"]["gradient_clip_norm"],
            )

        optimizer.step()

    scheduler.step()

    return {
        "loss": loss.item(),
        "perplexity": torch.exp(loss).item(),
        "lr": optimizer.param_groups[0]["lr"],
    }


def evaluate(
    model,
    dataloader,
    config: Dict[str, Any],
    device: str,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate model on validation set.

    Args:
        model: TTT model.
        dataloader: Validation dataloader.
        config: Configuration dictionary.
        device: Device to run on.
        max_steps: Maximum evaluation steps.

    Returns:
        Dictionary of evaluation metrics.
    """
    import torch

    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_steps and i >= max_steps:
                break

            (input_ids,) = batch
            input_ids = input_ids.to(device)
            labels = input_ids.clone()

            outputs = model(input_ids, labels=labels)
            total_loss += outputs["loss"].item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
    }


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
    scheduler = build_scheduler(optimizer, config)
    train_loader = build_dataloader(config, "train")
    val_loader = build_dataloader(config, "val")

    # Mixed precision scaler
    scaler = None
    if config["training"].get("bf16") or config["training"].get("fp16"):
        scaler = torch.cuda.amp.GradScaler()

    # Optional: Compile model
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

            metrics = train_step(
                model, batch, optimizer, scheduler, config, device, scaler
            )
            global_step += 1

            # Logging
            if global_step % log_interval == 0:
                logger.info(
                    f"Step {global_step}/{max_steps} - "
                    f"Loss: {metrics['loss']:.4f}, "
                    f"PPL: {metrics['perplexity']:.2f}, "
                    f"LR: {metrics['lr']:.2e}"
                )

            # Evaluation
            if global_step % eval_interval == 0:
                eval_metrics = evaluate(
                    model, val_loader, config, device,
                    max_steps=config["evaluation"]["num_eval_steps"],
                )
                logger.info(
                    f"Eval @ step {global_step} - "
                    f"Loss: {eval_metrics['loss']:.4f}, "
                    f"PPL: {eval_metrics['perplexity']:.2f}"
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
        description="Train Test-Time Training (TTT) language model",
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
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )

    # Model overrides
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of layers",
    )
    parser.add_argument(
        "--ttt-type",
        type=str,
        choices=["linear", "mlp"],
        default=None,
        help="TTT layer type",
    )
    parser.add_argument(
        "--ttt-lr",
        type=float,
        default=None,
        help="TTT inner learning rate",
    )
    parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=None,
        help="Mini-batch size for TTT",
    )

    # Path overrides
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name",
    )

    # Utility arguments
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
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
        print(json.dumps(config, indent=2, default=str))
        return

    # Run training
    train(config)


if __name__ == "__main__":
    main()
