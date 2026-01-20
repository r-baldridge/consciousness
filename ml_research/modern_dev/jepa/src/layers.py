"""
JEPA Architecture-Specific Layers

Core components for Joint Embedding Predictive Architecture:
- ContextEncoder: Encodes visible (context) patches
- TargetEncoder: Encodes target patches (EMA updated)
- Predictor: Predicts target representations from context
- VICRegLoss: Variance-Invariance-Covariance regularization

Reference: https://arxiv.org/abs/2301.08243
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbed(nn.Module):
    """Patch embedding layer.

    Converts images to sequences of patch embeddings.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        """Initialize patch embedding.

        Args:
            image_size: Input image size.
            patch_size: Size of each patch.
            in_channels: Number of input channels.
            embed_dim: Embedding dimension.
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.grid_size = image_size // patch_size

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert images to patch embeddings.

        Args:
            x: Input images (batch, channels, height, width).

        Returns:
            Patch embeddings (batch, num_patches, embed_dim).
        """
        # (batch, embed_dim, grid_h, grid_w)
        x = self.projection(x)
        # (batch, embed_dim, num_patches)
        x = x.flatten(2)
        # (batch, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for patches."""

    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        dropout: float = 0.0,
        include_cls: bool = True,
    ):
        """Initialize positional encoding.

        Args:
            num_patches: Number of patches.
            embed_dim: Embedding dimension.
            dropout: Dropout probability.
            include_cls: Whether to include CLS token position.
        """
        super().__init__()
        self.include_cls = include_cls
        num_positions = num_patches + 1 if include_cls else num_patches

        self.pos_embed = nn.Parameter(torch.randn(1, num_positions, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

        if include_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        else:
            self.cls_token = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to patch embeddings.

        Args:
            x: Patch embeddings (batch, num_patches, embed_dim).

        Returns:
            Embeddings with positional encoding (batch, num_patches+1, embed_dim) if CLS.
        """
        batch_size = x.size(0)

        if self.include_cls and self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        """Initialize attention layer.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-head self-attention.

        Args:
            x: Input tensor (batch, seq_len, embed_dim).
            mask: Optional attention mask.

        Returns:
            Output tensor (batch, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        x = self.proj(x)
        x = self.dropout(x)

        return x


class MLP(nn.Module):
    """MLP block for transformer."""

    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        """Initialize MLP block.

        Args:
            embed_dim: Embedding dimension.
            mlp_ratio: Hidden dimension expansion ratio.
            dropout: Dropout probability.
        """
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        """Initialize transformer block.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            mlp_ratio: MLP expansion ratio.
            dropout: Dropout probability.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, embed_dim).
            mask: Optional attention mask.

        Returns:
            Output tensor (batch, seq_len, embed_dim).
        """
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class ContextEncoder(nn.Module):
    """Context encoder for JEPA.

    Encodes visible (unmasked) patches into representations.
    Uses Vision Transformer architecture.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        """Initialize context encoder.

        Args:
            image_size: Input image size.
            patch_size: Size of each patch.
            in_channels: Number of input channels.
            embed_dim: Embedding dimension.
            depth: Number of transformer blocks.
            num_heads: Number of attention heads.
            mlp_ratio: MLP expansion ratio.
            dropout: Dropout probability.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            num_patches=num_patches,
            embed_dim=embed_dim,
            dropout=dropout,
            include_cls=True,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        images: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode images.

        Args:
            images: Input images (batch, channels, height, width).
            mask: Boolean mask for visible patches (batch, num_patches).
                  True = visible, False = masked.

        Returns:
            Patch embeddings (batch, num_patches+1, embed_dim).
        """
        # Get patch embeddings
        x = self.patch_embed(images)  # (batch, num_patches, embed_dim)

        # Apply mask if provided (keep only visible patches)
        if mask is not None:
            # For context encoder, we only keep visible patches
            # But we need to maintain batch structure
            pass  # Placeholder - actual implementation would handle sparse attention

        # Add positional encoding and CLS token
        x = self.pos_encoding(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final normalization
        x = self.norm(x)

        return x


class TargetEncoder(nn.Module):
    """Target encoder for JEPA.

    Identical architecture to context encoder but:
    - Updated via EMA (exponential moving average)
    - No gradient flow (stop gradient)

    Provides ground truth representations for prediction targets.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        """Initialize target encoder.

        Same parameters as ContextEncoder.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            num_patches=num_patches,
            embed_dim=embed_dim,
            dropout=dropout,
            include_cls=True,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images (no gradient computation).

        Args:
            images: Input images (batch, channels, height, width).

        Returns:
            Patch embeddings (batch, num_patches+1, embed_dim).
        """
        # Get patch embeddings
        x = self.patch_embed(images)

        # Add positional encoding and CLS token
        x = self.pos_encoding(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        return x


class Predictor(nn.Module):
    """Predictor network for JEPA.

    Takes context representations and predicts target representations.
    Uses a narrower transformer architecture.
    """

    def __init__(
        self,
        context_dim: int = 768,
        pred_dim: int = 384,
        output_dim: int = 768,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        """Initialize predictor.

        Args:
            context_dim: Dimension of context embeddings.
            pred_dim: Internal predictor dimension.
            output_dim: Output dimension (should match target encoder).
            depth: Number of transformer blocks.
            num_heads: Number of attention heads.
            mlp_ratio: MLP expansion ratio.
            dropout: Dropout probability.
        """
        super().__init__()
        self.context_dim = context_dim
        self.pred_dim = pred_dim
        self.output_dim = output_dim

        # Project context to predictor dimension
        self.context_proj = nn.Linear(context_dim, pred_dim)

        # Learnable mask tokens for targets
        self.mask_token = nn.Parameter(torch.randn(1, 1, pred_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=pred_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # Output projection
        self.norm = nn.LayerNorm(pred_dim)
        self.output_proj = nn.Linear(pred_dim, output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        context: torch.Tensor,
        target_mask: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict target representations.

        Args:
            context: Context encoder output (batch, num_patches+1, context_dim).
            target_mask: Boolean mask for target patches (batch, num_patches).
            context_mask: Boolean mask for context patches (batch, num_patches).

        Returns:
            Predicted target representations (batch, num_targets, output_dim).
        """
        batch_size = context.size(0)
        device = context.device

        # Project context to predictor dimension
        x = self.context_proj(context)

        # Count target patches per batch element
        num_targets = target_mask.sum(dim=1).max().item()

        # Create mask tokens for targets
        mask_tokens = self.mask_token.expand(batch_size, int(num_targets), -1)

        # Concatenate context and mask tokens
        # Note: This is a simplified version; actual implementation
        # would insert mask tokens at correct positions
        x = torch.cat([x, mask_tokens], dim=1)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Extract predictions for target positions
        # (taking last num_targets tokens as predictions)
        predictions = x[:, -int(num_targets):]

        # Project to output dimension
        predictions = self.norm(predictions)
        predictions = self.output_proj(predictions)

        return predictions


class VICRegLoss(nn.Module):
    """Variance-Invariance-Covariance Regularization Loss.

    Prevents representation collapse in self-supervised learning by:
    - Maintaining variance in each dimension
    - Encouraging invariance to transformations
    - Decorrelating different dimensions

    Reference: https://arxiv.org/abs/2105.04906
    """

    def __init__(
        self,
        var_weight: float = 1.0,
        inv_weight: float = 1.0,
        cov_weight: float = 0.04,
        eps: float = 1e-4,
    ):
        """Initialize VICReg loss.

        Args:
            var_weight: Weight for variance loss.
            inv_weight: Weight for invariance loss.
            cov_weight: Weight for covariance loss.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.var_weight = var_weight
        self.inv_weight = inv_weight
        self.cov_weight = cov_weight
        self.eps = eps

    def forward(
        self,
        z: torch.Tensor,
        z_prime: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute VICReg loss.

        Args:
            z: Representations (batch, seq_len, embed_dim) or (batch, embed_dim).
            z_prime: Optional second view representations.

        Returns:
            VICReg loss scalar.
        """
        # Handle 3D input (flatten sequence dimension)
        if z.dim() == 3:
            z = z.reshape(-1, z.size(-1))
        if z_prime is not None and z_prime.dim() == 3:
            z_prime = z_prime.reshape(-1, z_prime.size(-1))

        # Variance loss: std of each dimension should be > 1
        var_loss = self._variance_loss(z)
        if z_prime is not None:
            var_loss = var_loss + self._variance_loss(z_prime)

        # Covariance loss: off-diagonal elements should be small
        cov_loss = self._covariance_loss(z)
        if z_prime is not None:
            cov_loss = cov_loss + self._covariance_loss(z_prime)

        # Invariance loss: MSE between views
        if z_prime is not None:
            inv_loss = F.mse_loss(z, z_prime)
        else:
            inv_loss = torch.tensor(0.0, device=z.device)

        # Combine losses
        total_loss = (
            self.var_weight * var_loss +
            self.inv_weight * inv_loss +
            self.cov_weight * cov_loss
        )

        return total_loss

    def _variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Compute variance loss (hinge loss on std).

        Encourages std >= 1 for each dimension.

        Args:
            z: Representations (batch, embed_dim).

        Returns:
            Variance loss scalar.
        """
        std = z.std(dim=0) + self.eps
        var_loss = F.relu(1.0 - std).mean()
        return var_loss

    def _covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Compute covariance loss.

        Encourages decorrelation between dimensions.

        Args:
            z: Representations (batch, embed_dim).

        Returns:
            Covariance loss scalar.
        """
        batch_size, embed_dim = z.shape

        # Center the representations
        z_centered = z - z.mean(dim=0)

        # Compute covariance matrix
        cov = (z_centered.T @ z_centered) / (batch_size - 1)

        # Sum squared off-diagonal elements
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        cov_loss = off_diag / embed_dim

        return cov_loss
