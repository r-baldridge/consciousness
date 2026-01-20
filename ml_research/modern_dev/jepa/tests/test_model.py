"""
JEPA Model Tests

Basic tests for Joint Embedding Predictive Architecture model components.

Run with: pytest tests/test_model.py -v
"""

from __future__ import annotations

import pytest
import torch


class TestJEPAConfig:
    """Tests for JEPAConfig dataclass."""

    def test_default_config(self):
        """Test default configuration creation."""
        from ..src.model import JEPAConfig

        config = JEPAConfig()

        assert config.image_size == 224
        assert config.patch_size == 16
        assert config.embed_dim == 768
        assert config.encoder_depth == 12
        assert config.num_targets == 4

    def test_custom_config(self):
        """Test custom configuration creation."""
        from ..src.model import JEPAConfig

        config = JEPAConfig(
            image_size=384,
            patch_size=14,
            embed_dim=1024,
        )

        assert config.image_size == 384
        assert config.patch_size == 14
        assert config.embed_dim == 1024

    def test_config_num_patches(self):
        """Test num_patches property."""
        from ..src.model import JEPAConfig

        config = JEPAConfig(image_size=224, patch_size=16)
        assert config.num_patches == 196  # (224/16)^2

        config = JEPAConfig(image_size=384, patch_size=16)
        assert config.num_patches == 576  # (384/16)^2

    def test_config_to_dict(self):
        """Test configuration serialization."""
        from ..src.model import JEPAConfig

        config = JEPAConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["embed_dim"] == 768
        assert "target_scale" in config_dict

    def test_config_from_dict(self):
        """Test configuration deserialization."""
        from ..src.model import JEPAConfig

        config_dict = {
            "image_size": 224,
            "patch_size": 16,
            "in_channels": 3,
            "embed_dim": 512,
            "encoder_depth": 6,
            "encoder_heads": 8,
            "predictor_embed_dim": 256,
            "predictor_depth": 6,
            "predictor_heads": 4,
            "num_targets": 4,
            "target_scale": [0.15, 0.2],
            "target_aspect_ratio": [0.75, 1.5],
            "ema_momentum": 0.996,
            "mlp_ratio": 4.0,
            "dropout": 0.0,
            "use_vicreg": True,
            "vicreg_weights": [1.0, 1.0, 0.04],
        }

        config = JEPAConfig.from_dict(config_dict)

        assert config.embed_dim == 512
        assert config.encoder_depth == 6


class TestPatchEmbed:
    """Tests for patch embedding layer."""

    def test_forward(self):
        """Test patch embedding forward pass."""
        from ..src.layers import PatchEmbed

        embed = PatchEmbed(
            image_size=224,
            patch_size=16,
            in_channels=3,
            embed_dim=768,
        )

        images = torch.randn(4, 3, 224, 224)
        patches = embed(images)

        assert patches.shape == (4, 196, 768)

    def test_different_sizes(self):
        """Test patch embedding with different sizes."""
        from ..src.layers import PatchEmbed

        embed = PatchEmbed(
            image_size=384,
            patch_size=14,
            in_channels=3,
            embed_dim=1024,
        )

        # 384 / 14 = 27.4... rounded to 27 (should be exact division)
        # Actually 384/16 = 24, let's use 16
        embed = PatchEmbed(image_size=256, patch_size=16, embed_dim=512)
        images = torch.randn(2, 3, 256, 256)
        patches = embed(images)

        assert patches.shape == (2, 256, 512)


class TestPositionalEncoding:
    """Tests for positional encoding."""

    def test_forward_with_cls(self):
        """Test positional encoding with CLS token."""
        from ..src.layers import PositionalEncoding

        pos_enc = PositionalEncoding(
            num_patches=196,
            embed_dim=768,
            include_cls=True,
        )

        x = torch.randn(4, 196, 768)
        out = pos_enc(x)

        assert out.shape == (4, 197, 768)  # +1 for CLS

    def test_forward_without_cls(self):
        """Test positional encoding without CLS token."""
        from ..src.layers import PositionalEncoding

        pos_enc = PositionalEncoding(
            num_patches=196,
            embed_dim=768,
            include_cls=False,
        )

        x = torch.randn(4, 196, 768)
        out = pos_enc(x)

        assert out.shape == (4, 196, 768)


class TestContextEncoder:
    """Tests for context encoder."""

    def test_initialization(self):
        """Test encoder initialization."""
        from ..src.layers import ContextEncoder

        encoder = ContextEncoder(
            image_size=224,
            patch_size=16,
            embed_dim=384,
            depth=4,
            num_heads=6,
        )

        assert encoder.embed_dim == 384

    def test_forward(self):
        """Test encoder forward pass."""
        from ..src.layers import ContextEncoder

        encoder = ContextEncoder(
            image_size=224,
            patch_size=16,
            embed_dim=384,
            depth=4,
            num_heads=6,
        )

        images = torch.randn(2, 3, 224, 224)
        embeddings = encoder(images)

        # Should have CLS + patches
        assert embeddings.shape == (2, 197, 384)


class TestTargetEncoder:
    """Tests for target encoder."""

    def test_forward_no_grad(self):
        """Test target encoder runs without gradients."""
        from ..src.layers import TargetEncoder

        encoder = TargetEncoder(
            image_size=224,
            patch_size=16,
            embed_dim=384,
            depth=4,
            num_heads=6,
        )

        images = torch.randn(2, 3, 224, 224)

        # Should not accumulate gradients
        with torch.no_grad():
            embeddings = encoder(images)

        assert embeddings.shape == (2, 197, 384)
        assert not embeddings.requires_grad


class TestPredictor:
    """Tests for predictor network."""

    def test_initialization(self):
        """Test predictor initialization."""
        from ..src.layers import Predictor

        predictor = Predictor(
            context_dim=768,
            pred_dim=384,
            output_dim=768,
            depth=6,
            num_heads=6,
        )

        assert predictor.pred_dim == 384
        assert predictor.output_dim == 768

    def test_forward(self):
        """Test predictor forward pass."""
        from ..src.layers import Predictor

        predictor = Predictor(
            context_dim=768,
            pred_dim=384,
            output_dim=768,
            depth=4,
            num_heads=6,
        )

        batch_size = 2
        num_patches = 197
        context = torch.randn(batch_size, num_patches, 768)
        target_mask = torch.zeros(batch_size, 196, dtype=torch.bool)
        target_mask[:, :16] = True  # 16 target patches
        context_mask = ~target_mask

        predicted = predictor(context, target_mask, context_mask)

        # Should predict for target patches
        assert predicted.shape[0] == batch_size
        assert predicted.shape[-1] == 768


class TestVICRegLoss:
    """Tests for VICReg loss."""

    def test_initialization(self):
        """Test VICReg loss initialization."""
        from ..src.layers import VICRegLoss

        loss_fn = VICRegLoss(
            var_weight=1.0,
            inv_weight=1.0,
            cov_weight=0.04,
        )

        assert loss_fn.var_weight == 1.0

    def test_forward_2d(self):
        """Test VICReg loss with 2D input."""
        from ..src.layers import VICRegLoss

        loss_fn = VICRegLoss()
        z = torch.randn(32, 256)

        loss = loss_fn(z)

        assert loss.ndim == 0  # scalar
        assert loss >= 0

    def test_forward_3d(self):
        """Test VICReg loss with 3D input."""
        from ..src.layers import VICRegLoss

        loss_fn = VICRegLoss()
        z = torch.randn(4, 197, 768)

        loss = loss_fn(z)

        assert loss.ndim == 0


class TestMultiBlockMasking:
    """Tests for multi-block masking."""

    def test_forward(self):
        """Test masking generation."""
        from ..src.model import MultiBlockMasking, JEPAConfig

        config = JEPAConfig(image_size=224, patch_size=16, num_targets=4)
        masking = MultiBlockMasking(config)

        context_mask, target_masks, target_indices = masking(
            batch_size=4,
            device=torch.device("cpu"),
        )

        # Context mask shape
        assert context_mask.shape == (4, 196)
        assert context_mask.dtype == torch.bool

        # Should have correct number of target masks
        assert len(target_masks) == 4

        # Context and targets should be disjoint
        # (context_mask is True for visible, targets are masked regions)


class TestJEPA:
    """Tests for main JEPA model."""

    def test_initialization(self):
        """Test JEPA model initialization."""
        from ..src.model import JEPA, JEPAConfig

        config = JEPAConfig(
            image_size=224,
            patch_size=16,
            embed_dim=384,
            encoder_depth=4,
            encoder_heads=6,
            predictor_embed_dim=192,
            predictor_depth=4,
            predictor_heads=4,
        )
        model = JEPA(config)

        assert model.config == config

    def test_forward(self):
        """Test JEPA forward pass."""
        from ..src.model import JEPA, JEPAConfig

        config = JEPAConfig(
            image_size=224,
            patch_size=16,
            embed_dim=384,
            encoder_depth=2,
            encoder_heads=6,
            predictor_embed_dim=192,
            predictor_depth=2,
            predictor_heads=4,
            num_targets=2,
        )
        model = JEPA(config)

        images = torch.randn(2, 3, 224, 224)
        outputs = model(images)

        assert "loss" in outputs
        assert "pred_loss" in outputs
        assert "vicreg_loss" in outputs
        assert outputs["loss"].ndim == 0

    def test_encode(self):
        """Test JEPA encoding for inference."""
        from ..src.model import JEPA, JEPAConfig

        config = JEPAConfig(
            image_size=224,
            patch_size=16,
            embed_dim=384,
            encoder_depth=2,
            encoder_heads=6,
            predictor_embed_dim=192,
            predictor_depth=2,
            predictor_heads=4,
        )
        model = JEPA(config)
        model.eval()

        images = torch.randn(2, 3, 224, 224)
        embeddings = model.encode(images)

        assert embeddings.shape == (2, 197, 384)

    def test_get_cls_token(self):
        """Test CLS token extraction."""
        from ..src.model import JEPA, JEPAConfig

        config = JEPAConfig(
            image_size=224,
            patch_size=16,
            embed_dim=384,
            encoder_depth=2,
            encoder_heads=6,
            predictor_embed_dim=192,
            predictor_depth=2,
            predictor_heads=4,
        )
        model = JEPA(config)
        model.eval()

        images = torch.randn(2, 3, 224, 224)
        cls_token = model.get_cls_token(images)

        assert cls_token.shape == (2, 384)

    def test_ema_update(self):
        """Test target encoder EMA update."""
        from ..src.model import JEPA, JEPAConfig

        config = JEPAConfig(
            embed_dim=384,
            encoder_depth=2,
            encoder_heads=6,
            predictor_embed_dim=192,
            predictor_depth=2,
            predictor_heads=4,
            ema_momentum=0.99,
        )
        model = JEPA(config)

        # Get initial target encoder param
        target_param = next(model.target_encoder.parameters()).clone()

        # Modify context encoder
        with torch.no_grad():
            for p in model.context_encoder.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Update target encoder
        model.update_target_encoder()

        # Target should have changed (but not completely)
        new_target_param = next(model.target_encoder.parameters())
        assert not torch.allclose(target_param, new_target_param)

    def test_parameter_count(self):
        """Test model has trainable parameters."""
        from ..src.model import JEPA, JEPAConfig

        config = JEPAConfig(
            embed_dim=384,
            encoder_depth=4,
            predictor_embed_dim=192,
            predictor_depth=4,
        )
        model = JEPA(config)

        # Context encoder and predictor should have trainable params
        context_params = sum(
            p.numel() for p in model.context_encoder.parameters() if p.requires_grad
        )
        predictor_params = sum(
            p.numel() for p in model.predictor.parameters() if p.requires_grad
        )

        assert context_params > 0
        assert predictor_params > 0

        # Target encoder should NOT have trainable params
        target_trainable = sum(
            p.numel() for p in model.target_encoder.parameters() if p.requires_grad
        )
        assert target_trainable == 0


class TestIJEPAForClassification:
    """Tests for classification wrapper."""

    def test_forward(self):
        """Test classification forward pass."""
        from ..src.model import JEPA, JEPAConfig, IJEPAForClassification

        config = JEPAConfig(
            embed_dim=384,
            encoder_depth=2,
            encoder_heads=6,
            predictor_embed_dim=192,
            predictor_depth=2,
            predictor_heads=4,
        )
        jepa = JEPA(config)

        classifier = IJEPAForClassification(
            jepa=jepa,
            num_classes=100,
            freeze_encoder=True,
        )

        images = torch.randn(2, 3, 224, 224)
        logits = classifier(images)

        assert logits.shape == (2, 100)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
