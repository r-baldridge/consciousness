#!/usr/bin/env python3
"""
JEPA Training CLI

Command-line interface for training Joint Embedding Predictive Architecture models.

Usage:
    python -m jepa.cli.train --config configs/default.yaml
    python -m jepa.cli.train --config configs/default.yaml --batch-size 128 --epochs 100
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
    # Override training params if specified
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.epochs is not None:
        config["training"]["max_epochs"] = args.epochs
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed

    # Override model params
    if args.embed_dim is not None:
        config["model"]["embed_dim"] = args.embed_dim
    if args.encoder_depth is not None:
        config["model"]["encoder_depth"] = args.encoder_depth

    # Override paths
    if args.data_dir is not None:
        config["data"]["data_dir"] = args.data_dir
    if args.checkpoint_dir is not None:
        config["logging"]["checkpoint_dir"] = args.checkpoint_dir

    # EMA momentum
    if args.ema_momentum is not None:
        config["ema"]["momentum"] = args.ema_momentum

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
    """Build JEPA model from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Initialized JEPA model.
    """
    from ..src.model import JEPA, JEPAConfig

    model_config = JEPAConfig(
        image_size=config["model"]["image_size"],
        patch_size=config["model"]["patch_size"],
        in_channels=config["model"]["in_channels"],
        embed_dim=config["model"]["embed_dim"],
        encoder_depth=config["model"]["encoder_depth"],
        encoder_heads=config["model"]["encoder_heads"],
        predictor_embed_dim=config["model"]["predictor_embed_dim"],
        predictor_depth=config["model"]["predictor_depth"],
        predictor_heads=config["model"]["predictor_heads"],
        num_targets=config["masking"]["num_targets"],
        target_scale=(
            config["masking"]["target_scale_min"],
            config["masking"]["target_scale_max"],
        ),
        target_aspect_ratio=(
            config["masking"]["target_aspect_ratio_min"],
            config["masking"]["target_aspect_ratio_max"],
        ),
        ema_momentum=config["ema"]["momentum"],
        mlp_ratio=config["model"]["mlp_ratio"],
        dropout=config["model"]["dropout"],
        use_vicreg=config["loss"]["use_vicreg"],
        vicreg_weights=(
            config["loss"]["vicreg_var_weight"],
            config["loss"]["vicreg_inv_weight"],
            config["loss"]["vicreg_cov_weight"],
        ),
    )

    model = JEPA(model_config)

    # Count parameters
    context_params = sum(p.numel() for p in model.context_encoder.parameters())
    predictor_params = sum(p.numel() for p in model.predictor.parameters())
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"Built JEPA model:")
    logger.info(f"  Context encoder: {context_params:,} parameters")
    logger.info(f"  Predictor: {predictor_params:,} parameters")
    logger.info(f"  Total: {total_params:,} parameters")

    return model


def build_optimizer(model, config: Dict[str, Any]):
    """Build optimizer from configuration.

    Args:
        model: JEPA model.
        config: Configuration dictionary.

    Returns:
        Optimizer instance.
    """
    import torch.optim as optim

    training_config = config["training"]

    # Only optimize context encoder and predictor (not target encoder)
    params = list(model.context_encoder.parameters()) + list(model.predictor.parameters())

    if training_config["optimizer"] == "adamw":
        optimizer = optim.AdamW(
            params,
            lr=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
            betas=(training_config["adam_beta1"], training_config["adam_beta2"]),
            eps=training_config["adam_eps"],
        )
    else:
        optimizer = optim.Adam(
            params,
            lr=training_config["learning_rate"],
        )

    return optimizer


def build_scheduler(optimizer, config: Dict[str, Any], steps_per_epoch: int):
    """Build learning rate scheduler.

    Args:
        optimizer: Optimizer instance.
        config: Configuration dictionary.
        steps_per_epoch: Number of training steps per epoch.

    Returns:
        LR scheduler instance.
    """
    import torch.optim.lr_scheduler as lr_scheduler

    training_config = config["training"]
    total_steps = training_config["max_epochs"] * steps_per_epoch
    warmup_steps = training_config["warmup_epochs"] * steps_per_epoch

    if training_config["lr_scheduler"] == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=training_config["learning_rate"] * training_config["lr_min_ratio"],
        )
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

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
        ImageNet or other datasets.
    """
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    logger.warning("Using placeholder dummy data. Implement actual data loading.")

    batch_size = config["training"]["batch_size"]
    image_size = config["model"]["image_size"]
    in_channels = config["model"]["in_channels"]

    # Create dummy tensors
    num_samples = 10000 if split == "train" else 1000
    images = torch.randn(num_samples, in_channels, image_size, image_size)

    dataset = TensorDataset(images)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        drop_last=True,
    )

    return dataloader


def get_ema_momentum(config: Dict[str, Any], epoch: int) -> float:
    """Get EMA momentum with schedule.

    Args:
        config: Configuration dictionary.
        epoch: Current epoch.

    Returns:
        EMA momentum value.
    """
    if not config["ema"]["momentum_schedule"]:
        return config["ema"]["momentum"]

    # Linear warmup from momentum to momentum_end
    warmup_epochs = config["ema"]["warmup_epochs"]
    max_epochs = config["training"]["max_epochs"]
    start = config["ema"]["momentum"]
    end = config["ema"]["momentum_end"]

    if epoch < warmup_epochs:
        return start
    else:
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        return start + (end - start) * progress


def train_step(
    model,
    batch,
    optimizer,
    config: Dict[str, Any],
    device: str,
):
    """Execute single training step.

    Args:
        model: JEPA model.
        batch: Training batch (images only for self-supervised).
        optimizer: Optimizer instance.
        config: Configuration dictionary.
        device: Device to run on.

    Returns:
        Dictionary of loss values.
    """
    import torch

    model.train()
    optimizer.zero_grad()

    # Unpack batch
    (images,) = batch
    images = images.to(device)

    # Forward pass
    outputs = model(images)

    # Total loss
    loss = outputs["loss"]

    # Backward pass
    loss.backward()

    # Gradient clipping
    if config["training"]["gradient_clip_norm"] > 0:
        torch.nn.utils.clip_grad_norm_(
            list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
            config["training"]["gradient_clip_norm"],
        )

    optimizer.step()

    # Update target encoder with EMA
    model.update_target_encoder()

    return {
        "loss": loss.item(),
        "pred_loss": outputs["pred_loss"].item(),
        "vicreg_loss": outputs["vicreg_loss"].item() if config["loss"]["use_vicreg"] else 0.0,
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
    train_loader = build_dataloader(config, "train")
    scheduler = build_scheduler(optimizer, config, len(train_loader))

    # Optional: Compile model for PyTorch 2.0+
    if config["hardware"].get("compile_model", False):
        try:
            model = torch.compile(model)
            logger.info("Model compiled with torch.compile")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    # Training loop
    max_epochs = config["training"]["max_epochs"]
    log_interval = config["logging"]["log_interval"]
    save_interval = config["logging"]["save_interval"]

    logger.info(f"Starting training for {max_epochs} epochs")

    global_step = 0
    for epoch in range(max_epochs):
        # Update EMA momentum if using schedule
        if config["ema"]["momentum_schedule"]:
            model.ema_momentum = get_ema_momentum(config, epoch)

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            losses = train_step(model, batch, optimizer, config, device)
            scheduler.step()

            epoch_loss += losses["loss"]
            num_batches += 1
            global_step += 1

            # Logging
            if global_step % log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch+1}/{max_epochs} - Step {global_step} - "
                    f"Loss: {losses['loss']:.4f} (pred: {losses['pred_loss']:.4f}, "
                    f"vicreg: {losses['vicreg_loss']:.4f}) - LR: {lr:.2e}"
                )

        # Epoch summary
        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{max_epochs} complete - Avg Loss: {avg_loss:.4f}")

        # Checkpointing
        if (epoch + 1) % save_interval == 0:
            checkpoint_dir = Path(config["logging"]["checkpoint_dir"])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            model.save_pretrained(str(checkpoint_path))
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    logger.info("Training complete!")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train JEPA (Joint Embedding Predictive Architecture) model",
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
        "--epochs",
        type=int,
        default=None,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )

    # Model overrides
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=None,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--encoder-depth",
        type=int,
        default=None,
        help="Encoder depth (number of layers)",
    )
    parser.add_argument(
        "--ema-momentum",
        type=float,
        default=None,
        help="EMA momentum for target encoder",
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
