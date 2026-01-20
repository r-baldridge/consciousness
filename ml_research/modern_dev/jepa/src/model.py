"""
Joint Embedding Predictive Architecture (JEPA) Model Implementation

This module contains the main JEPA model architecture with:
- Configuration dataclass
- Context and Target encoders
- Predictor network
- VICReg loss computation

Reference: https://arxiv.org/abs/2301.08243 (I-JEPA)
Reference: https://arxiv.org/abs/2402.03192 (V-JEPA)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ContextEncoder, TargetEncoder, Predictor, VICRegLoss


@dataclass
class JEPAConfig:
    """Configuration for JEPA model.

    Attributes:
        image_size: Input image size (assumed square).
        patch_size: Size of image patches.
        in_channels: Number of input channels.
        embed_dim: Embedding dimension for encoders.
        encoder_depth: Number of transformer layers in encoders.
        encoder_heads: Number of attention heads in encoders.
        predictor_embed_dim: Embedding dimension for predictor.
        predictor_depth: Number of transformer layers in predictor.
        predictor_heads: Number of attention heads in predictor.
        num_targets: Number of target blocks to predict.
        target_scale: Scale range for target block size (min, max).
        target_aspect_ratio: Aspect ratio range for target blocks (min, max).
        ema_momentum: EMA momentum for target encoder update.
        mlp_ratio: MLP expansion ratio in transformers.
        dropout: Dropout probability.
        use_vicreg: Whether to use VICReg regularization.
        vicreg_weights: Weights for VICReg loss (variance, invariance, covariance).
    """
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 768
    encoder_depth: int = 12
    encoder_heads: int = 12
    predictor_embed_dim: int = 384
    predictor_depth: int = 12
    predictor_heads: int = 6
    num_targets: int = 4
    target_scale: Tuple[float, float] = (0.15, 0.2)
    target_aspect_ratio: Tuple[float, float] = (0.75, 1.5)
    ema_momentum: float = 0.996
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    use_vicreg: bool = True
    vicreg_weights: Tuple[float, float, float] = (1.0, 1.0, 0.04)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "in_channels": self.in_channels,
            "embed_dim": self.embed_dim,
            "encoder_depth": self.encoder_depth,
            "encoder_heads": self.encoder_heads,
            "predictor_embed_dim": self.predictor_embed_dim,
            "predictor_depth": self.predictor_depth,
            "predictor_heads": self.predictor_heads,
            "num_targets": self.num_targets,
            "target_scale": self.target_scale,
            "target_aspect_ratio": self.target_aspect_ratio,
            "ema_momentum": self.ema_momentum,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "use_vicreg": self.use_vicreg,
            "vicreg_weights": self.vicreg_weights,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "JEPAConfig":
        """Create config from dictionary."""
        # Handle tuple fields
        if "target_scale" in config_dict:
            config_dict["target_scale"] = tuple(config_dict["target_scale"])
        if "target_aspect_ratio" in config_dict:
            config_dict["target_aspect_ratio"] = tuple(config_dict["target_aspect_ratio"])
        if "vicreg_weights" in config_dict:
            config_dict["vicreg_weights"] = tuple(config_dict["vicreg_weights"])
        return cls(**config_dict)

    @property
    def num_patches(self) -> int:
        """Get total number of patches."""
        return (self.image_size // self.patch_size) ** 2

    @property
    def grid_size(self) -> int:
        """Get grid size (patches per side)."""
        return self.image_size // self.patch_size


class MultiBlockMasking(nn.Module):
    """Multi-block masking strategy for JEPA.

    Generates target blocks and context masks for training.
    Predicts K large target blocks from context (unmasked regions).
    """

    def __init__(self, config: JEPAConfig):
        """Initialize masking module.

        Args:
            config: JEPA configuration.
        """
        super().__init__()
        self.config = config
        self.grid_size = config.grid_size
        self.num_targets = config.num_targets
        self.target_scale = config.target_scale
        self.target_aspect_ratio = config.target_aspect_ratio

    def sample_block(
        self,
        device: torch.device,
    ) -> Tuple[int, int, int, int]:
        """Sample a single target block.

        Args:
            device: Device to create tensors on.

        Returns:
            Tuple of (top, left, height, width) for the block.
        """
        # Sample scale and aspect ratio
        scale = torch.empty(1, device=device).uniform_(*self.target_scale).item()
        aspect = torch.empty(1, device=device).uniform_(*self.target_aspect_ratio).item()

        # Compute block size
        area = self.grid_size ** 2 * scale
        height = int(round((area / aspect) ** 0.5))
        width = int(round(height * aspect))

        # Clamp to valid range
        height = min(height, self.grid_size)
        width = min(width, self.grid_size)
        height = max(height, 1)
        width = max(width, 1)

        # Sample position
        top = torch.randint(0, self.grid_size - height + 1, (1,), device=device).item()
        left = torch.randint(0, self.grid_size - width + 1, (1,), device=device).item()

        return top, left, height, width

    def forward(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Generate masks for a batch.

        Args:
            batch_size: Number of samples in batch.
            device: Device to create tensors on.

        Returns:
            Tuple of:
                - context_mask: Boolean mask for context patches (batch, num_patches).
                - target_masks: List of boolean masks for each target block.
                - target_indices: List of patch indices for each target.
        """
        num_patches = self.grid_size ** 2

        # Initialize all patches as context (True = visible)
        context_mask = torch.ones(batch_size, num_patches, dtype=torch.bool, device=device)

        target_masks = []
        target_indices_list = []

        for _ in range(self.num_targets):
            target_mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
            target_indices = []

            for b in range(batch_size):
                # Sample a block
                top, left, height, width = self.sample_block(device)

                # Convert to patch indices
                indices = []
                for i in range(height):
                    for j in range(width):
                        patch_idx = (top + i) * self.grid_size + (left + j)
                        indices.append(patch_idx)

                indices = torch.tensor(indices, device=device)
                target_mask[b, indices] = True
                context_mask[b, indices] = False
                target_indices.append(indices)

            target_masks.append(target_mask)
            target_indices_list.append(target_indices)

        return context_mask, target_masks, target_indices_list


class JEPA(nn.Module):
    """Joint Embedding Predictive Architecture.

    Self-supervised learning model that predicts representations in
    latent space rather than pixel/token space. Uses:
    - Context encoder: Processes visible (unmasked) patches
    - Target encoder: Provides ground truth for masked regions (EMA updated)
    - Predictor: Predicts target representations from context

    Reference: https://arxiv.org/abs/2301.08243
    """

    def __init__(self, config: JEPAConfig):
        """Initialize JEPA model.

        Args:
            config: JEPA configuration object.
        """
        super().__init__()
        self.config = config

        # Context encoder (trainable)
        self.context_encoder = ContextEncoder(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            depth=config.encoder_depth,
            num_heads=config.encoder_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
        )

        # Target encoder (EMA updated, not trainable directly)
        self.target_encoder = TargetEncoder(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            depth=config.encoder_depth,
            num_heads=config.encoder_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
        )

        # Copy initial weights from context to target encoder
        self._init_target_encoder()

        # Predictor
        self.predictor = Predictor(
            context_dim=config.embed_dim,
            pred_dim=config.predictor_embed_dim,
            output_dim=config.embed_dim,
            depth=config.predictor_depth,
            num_heads=config.predictor_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
        )

        # Masking
        self.masking = MultiBlockMasking(config)

        # Loss
        if config.use_vicreg:
            self.vicreg_loss = VICRegLoss(
                var_weight=config.vicreg_weights[0],
                inv_weight=config.vicreg_weights[1],
                cov_weight=config.vicreg_weights[2],
            )
        else:
            self.vicreg_loss = None

        # EMA momentum
        self.ema_momentum = config.ema_momentum

    def _init_target_encoder(self) -> None:
        """Initialize target encoder with context encoder weights."""
        for param_t, param_c in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
        ):
            param_t.data.copy_(param_c.data)
            param_t.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        """Update target encoder with EMA of context encoder."""
        for param_t, param_c in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
        ):
            param_t.data.mul_(self.ema_momentum).add_(
                param_c.data, alpha=1 - self.ema_momentum
            )

    def forward(
        self,
        images: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for JEPA training.

        Args:
            images: Input images (batch, channels, height, width).
            return_embeddings: Whether to return intermediate embeddings.

        Returns:
            Dictionary containing:
                - loss: Total training loss.
                - pred_loss: Prediction loss (MSE in latent space).
                - vicreg_loss: VICReg regularization loss (if enabled).
                - context_embeddings: Context encoder outputs (if requested).
                - target_embeddings: Target encoder outputs (if requested).
                - predicted_embeddings: Predictor outputs (if requested).
        """
        batch_size = images.size(0)
        device = images.device

        # Generate masks
        context_mask, target_masks, target_indices = self.masking(batch_size, device)

        # Encode context (visible patches)
        context_embeddings = self.context_encoder(images, context_mask)

        # Encode targets (stop gradient - no backprop through target encoder)
        with torch.no_grad():
            target_embeddings = self.target_encoder(images)

        # Predict target representations
        total_pred_loss = 0.0
        total_vicreg_loss = 0.0
        num_targets = len(target_masks)

        all_predicted = []
        all_targets = []

        for i, (target_mask, target_indices_batch) in enumerate(
            zip(target_masks, target_indices)
        ):
            # Get target positions for prediction
            predicted = self.predictor(
                context_embeddings,
                target_mask,
                context_mask,
            )
            all_predicted.append(predicted)

            # Extract target embeddings for this block
            # Gather the target patches from target encoder output
            target_for_block = self._gather_targets(
                target_embeddings,
                target_indices_batch,
            )
            all_targets.append(target_for_block)

            # Compute prediction loss (MSE in latent space)
            pred_loss = F.mse_loss(predicted, target_for_block)
            total_pred_loss = total_pred_loss + pred_loss

        # Average over targets
        total_pred_loss = total_pred_loss / num_targets

        # Compute VICReg loss if enabled
        if self.vicreg_loss is not None:
            # Use context embeddings for regularization
            total_vicreg_loss = self.vicreg_loss(context_embeddings)
        else:
            total_vicreg_loss = torch.tensor(0.0, device=device)

        # Total loss
        loss = total_pred_loss + total_vicreg_loss

        result = {
            "loss": loss,
            "pred_loss": total_pred_loss,
            "vicreg_loss": total_vicreg_loss,
        }

        if return_embeddings:
            result["context_embeddings"] = context_embeddings
            result["target_embeddings"] = target_embeddings
            result["predicted_embeddings"] = all_predicted

        return result

    def _gather_targets(
        self,
        target_embeddings: torch.Tensor,
        target_indices_batch: List[torch.Tensor],
    ) -> torch.Tensor:
        """Gather target embeddings for a batch of target indices.

        Args:
            target_embeddings: Full target embeddings (batch, num_patches, embed_dim).
            target_indices_batch: List of indices for each batch element.

        Returns:
            Gathered target embeddings, padded to same length.
        """
        batch_size = target_embeddings.size(0)
        embed_dim = target_embeddings.size(-1)

        # Find max target size
        max_targets = max(len(indices) for indices in target_indices_batch)

        # Gather and pad
        gathered = torch.zeros(
            batch_size, max_targets, embed_dim,
            device=target_embeddings.device,
        )

        for b, indices in enumerate(target_indices_batch):
            gathered[b, :len(indices)] = target_embeddings[b, indices]

        return gathered

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to representations (for inference).

        Args:
            images: Input images (batch, channels, height, width).

        Returns:
            Image representations (batch, num_patches, embed_dim).
        """
        # Use context encoder for inference
        self.context_encoder.eval()
        with torch.no_grad():
            # No masking - encode all patches
            embeddings = self.context_encoder(images, mask=None)
        return embeddings

    def get_cls_token(self, images: torch.Tensor) -> torch.Tensor:
        """Get CLS token representation for classification.

        Args:
            images: Input images (batch, channels, height, width).

        Returns:
            CLS token representation (batch, embed_dim).
        """
        embeddings = self.encode(images)
        # CLS token is typically the first token
        return embeddings[:, 0]

    @classmethod
    def from_pretrained(cls, path: str) -> "JEPA":
        """Load pretrained JEPA model.

        Args:
            path: Path to pretrained model checkpoint.

        Returns:
            Loaded JEPA model.
        """
        checkpoint = torch.load(path, map_location="cpu")
        config = JEPAConfig.from_dict(checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def save_pretrained(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        checkpoint = {
            "config": self.config.to_dict(),
            "model_state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)


class IJEPAForClassification(nn.Module):
    """I-JEPA with classification head for downstream tasks.

    Uses frozen JEPA encoder with trainable classification head.
    """

    def __init__(
        self,
        jepa: JEPA,
        num_classes: int,
        freeze_encoder: bool = True,
    ):
        """Initialize classification model.

        Args:
            jepa: Pretrained JEPA model.
            num_classes: Number of output classes.
            freeze_encoder: Whether to freeze encoder weights.
        """
        super().__init__()
        self.jepa = jepa
        self.num_classes = num_classes

        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.jepa.parameters():
                param.requires_grad = False

        # Classification head
        embed_dim = jepa.config.embed_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification.

        Args:
            images: Input images (batch, channels, height, width).

        Returns:
            Class logits (batch, num_classes).
        """
        # Get CLS token representation
        cls_embedding = self.jepa.get_cls_token(images)

        # Classify
        logits = self.classifier(cls_embedding)
        return logits
